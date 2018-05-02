# -*- coding: utf-8 -*-
import os
from abc import abstractmethod

import numpy as np
from joblib import delayed, Parallel

from .errors import TileExtractionException
from .image import Image, TileBuilder
from .information import WorkflowInformation
from .locator import BinaryLocator, SemanticLocator
from .logging import Loggable, SilentLogger
from .merger import SemanticMerger
from .timing import WorkflowTiming
from .util import batch_split

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version = "0.1"


def _segment_locate(tile, segmenter, locator, timing):
    """Load the tile numpy representation and applies it segmentation and location using the given objects

        Parameters
        ----------
        tile: Tile
            The tile to process for the segment locate
        segmenter: Segmenter
            For segmenting the image
        locator: Locator
            For converting a mask to polygons
        timing: WorkflowTiming
            The workflow timing object for measuring the execution times of the various steps

        Returns
        -------
        polygons: iterable (subtype: shapely.geometry.Polygon)
            Iterable containing the polygons found by the locate step
        """
    timing.start(SLDCWorkflow.TIMING_DETECT_LOAD)
    np_image = tile.np_image
    timing.end(SLDCWorkflow.TIMING_DETECT_LOAD)
    timing.start(SLDCWorkflow.TIMING_DETECT_SEGMENT)
    segmented = segmenter.segment(np_image)
    timing.end(SLDCWorkflow.TIMING_DETECT_SEGMENT)
    timing.start(SLDCWorkflow.TIMING_DETECT_LOCATE)
    located = locator.locate(segmented, offset=tile.offset)
    timing.end(SLDCWorkflow.TIMING_DETECT_LOCATE)
    return located


def _batch_segment_locate(tile_ids, tile_topology, segmenter, locator, logger=SilentLogger(), timing_root=None):
    """Helper function for parallel execution. Error occurring in this method is notified by returning None in place of
    the found polygons list.

    Parameters
    ----------
    tile_ids: iterable (subtype: int, size: N)
        The identifiers of the tiles to be processed
    tile_topology: TileTopology
        The tile topology from which were extracted the identifiers to process
    segmenter: Segmenter
        The segmenter object
    locator: Locator
        The locator object

    Returns
    -------
    timing: WorkflowTiming
        The timing of execution for processing of the tile.
    tiles_polygons: iterable (subtype: (int, shapely.geometry.Polygon), size: N)
        A list containing the tile identifiers and the polygons found inside them
    """
    timing = WorkflowTiming(root=timing_root)
    tiles_polygons = list()
    for tile_id in tile_ids:
        try:
            tile = tile_topology.tile(tile_id)
            tiles_polygons.append((tile_id, _segment_locate(tile, segmenter, locator, timing)))
        except TileExtractionException:
            logger.w("SLDCWorkflow : a tile (id:{}) couldn't be fetched computations.".format(tile_id))
            tiles_polygons.append((tile_id, []))
    return timing, tiles_polygons


def _dc_with_timing(dispatcher_classifier, image, polygons, timing_root=None):
    """

    dispatcher_classifier:
    image:
    polygons:

    Returns
    -------
    pred: iterable (subtype: int, size: N)
        The classes predicted for the passed polygons
    proba: iterable (subtype: float, size: N)
        The probability estimates for the predicted classes
    dispatch: iterable (subtype: hashable, size: N)
        The label of the dispatching rules which dispatched the polygons
    timing: WorkflowTiming
        The timing object containing the execution times of the dispatching and classification steps
    """
    return dispatcher_classifier.dispatch_classify_batch(image, polygons, timing_root=timing_root)


def _parallel_segment_locate(pool, segmenter, locator, logger, tile_topology, timing):
    """Execute the segment locate phase
    Parameters
    ----------
    pool: Parallel
        A pool of processes
    segmenter: SemanticSegmenter
        A segmenter
    locator: Locator
        A locator
    logger: Logger
        A logger
    tile_topology: TileTopology
        A tile topology
    timing: WorkflowTiming
        A workflow timing object for computing time

    Returns
    -------
    tiles: iterable (size: n, subtype: int) 
        Iterable containing the tiles ids
    tile_polygons: iterable (size: n, subtype: iterable of Polygon objects))
        The iterable at index i contains the polygons and pixel values found in the tile having index tiles[i]
    """
    # partition the tiles into batches for submitting them to processes
    batches = tile_topology.partition_identifiers(pool.n_jobs)

    # execute
    results = pool(delayed(_batch_segment_locate)(
        tile_ids,
        tile_topology,
        segmenter,
        locator,
        logger,
        ".".join([SLDCWorkflow.TIMING_ROOT, SLDCWorkflow.TIMING_DETECT])
    ) for tile_ids in batches)

    sub_timings, tiles_polygons = list(zip(*results))
    tiles = np.array([tid for result in tiles_polygons for tid, _ in result])
    tile_polygons = np.array([polygons for result in tiles_polygons for _, polygons in result])

    # merge sub timings
    for sub_timing in sub_timings:
        timing.merge(sub_timing)

    return tiles, tile_polygons


class Workflow(Loggable):
    """Abstract base class to be implemented by workflows"""
    def __init__(self, tile_builder, tile_max_width=1024, tile_max_height=1024, tile_overlap=7, n_jobs=1, logger=SilentLogger()):
        super(Workflow, self).__init__(logger=logger)
        self._tile_max_width = tile_max_width
        self._tile_max_height = tile_max_height
        self._tile_overlap = tile_overlap
        self._tile_builder = tile_builder
        self._n_jobs = n_jobs
        self._pool = None  # cache across execution

    @property
    def tile_max_width(self):
        return self._tile_max_width

    @property
    def tile_max_height(self):
        return self._tile_max_height

    @property
    def tile_overlap(self):
        return self._tile_overlap

    @property
    def tile_builder(self):
        return self._tile_builder

    @property
    def n_jobs(self):
        return self._n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        if value != self._n_jobs:
            self._pool = None
            self._n_jobs = value

    def _set_pool(self):
        """Create a pool with self._n_jobs jobs in the self._pool variable.
        If the pool already exists, this method does nothing.
        """
        if self._pool is None:
            self._pool = Parallel(n_jobs=self._n_jobs)

    @property
    def pool(self):
        self._set_pool()
        return self._pool

    def __getstate__(self):
        self._pool = None  # so that the workflow is serializable
        return self.__dict__

    def _tile_topology(self, image):
        """Create a tile topology using the tile parameters for the given image
        Parameters
        ----------
        image: Image
            The image to generate a topology for
        Returns
        -------
        topology: TileTopology
            The topology
        """
        return image.tile_topology(
            self.tile_builder,
            max_width=self.tile_max_width,
            max_height=self.tile_max_height,
            overlap=self.tile_overlap
        )

    @abstractmethod
    def process(self, image):
        """Process the given image using the workflow
        Parameters
        ----------
        image: Image
            The image to process

        Returns
        -------
        workflow_information: WorkflowInformation
            The workflow information object containing all the information about detected objects, execution times...

        Notes
        -----
        This method doesn't modify the image passed as parameter.
        This method doesn't modify the object's attributes (except for the pool).
        """
        pass


class SLDCWorkflow(Workflow):
    """A class that coordinates various components of the SLDC workflow in order to detect objects and return
    their information.
    """
    TIMING_ROOT = "workflow.sldc"
    TIMING_DETECT = "detect"
    TIMING_DETECT_LOAD = "load"
    TIMING_DETECT_SEGMENT = "segment"
    TIMING_DETECT_LOCATE = "locate"
    TIMING_MERGE = "merge"
    TIMING_DC = "dispatch_classify"

    def __init__(self, segmenter, dispatcher_classifier, tile_builder,
                 tile_max_width=1024, tile_max_height=1024, tile_overlap=7,
                 dist_tolerance=1, logger=SilentLogger(), n_jobs=None, parallel_dispatch_classify=False):
        """Constructor for SLDCWorkflow objects

        Parameters
        ----------
        segmenter: Segmenter
            The segmenter implementing segmentation procedure to apply on tiles.
        dispatcher_classifier: DispatcherClassifier
            The dispatcher classifier object for dispatching polygons and classify them.
        tile_builder: TileBuilder
            An object for building specific tiles
        tile_max_width: int (optional, default: 1024)
            The maximum width of the tiles when iterating over the image
        tile_max_height: int (optional, default: 1024)
            The maximum height of the tiles when iterating over the image
        tile_overlap: int (optional, default: 5)
            The number of pixels of overlap between tiles when iterating over the image
        dist_tolerance: int (optional, default, 7)
            Maximal distance between two polygons so that they are considered from the same object
        logger: Logger (optional, default: SilentLogger)
            A logger object
        n_jobs: int (optional, default: 1)
            The number of job available for executing the workflow.
        parallel_dispatch_classify: boolean (optional, default: False)
            True for executing dispatching and classification in parallel, False for sequential.
        """
        super(SLDCWorkflow, self).__init__(
            n_jobs=n_jobs,
            tile_max_height=tile_max_height,
            tile_max_width=tile_max_width,
            tile_builder=tile_builder,
            tile_overlap=tile_overlap,
            logger=logger
        )
        self._segmenter = segmenter
        self._locator = BinaryLocator()
        self._merger = SemanticMerger(dist_tolerance)
        self._dispatch_classifier = dispatcher_classifier
        self._parallel_dispatch_classify = parallel_dispatch_classify

    def process(self, image):
        """Process function"""
        timing = WorkflowTiming(root=SLDCWorkflow.TIMING_ROOT)
        tile_topology = self._tile_topology(image)

        # segment locate
        self.logger.info("SLDCWorkflow : start segment/locate.")
        timing.start(SLDCWorkflow.TIMING_DETECT)
        tiles, tile_polygons = self._segment_locate(tile_topology, timing)
        timing.end(SLDCWorkflow.TIMING_DETECT)
        self.logger.info(
            "SLDCWorkflow : end segment/locate." + os.linesep +
            "SLDCWorkflow : {} tile(s) processed in {} s.".format(len(tiles), timing.total(SLDCWorkflow.TIMING_DETECT)) + os.linesep +
            "SLDCWorkflow : {} polygon(s) found on those tiles.".format(sum([len(polygons) for polygons in tile_polygons]))
        )

        # merge
        self.logger.info("SLDCWorkflow : start merging")
        timing.start(SLDCWorkflow.TIMING_MERGE)
        polygons = self._merger.merge(tiles, tile_polygons, tile_topology)
        timing.end(SLDCWorkflow.TIMING_MERGE)
        self.logger.info(
            "SLDCWorkflow : end merging." + os.linesep +
            "SLDCWorkflow : {} polygon(s) found.".format(len(polygons)) + os.linesep +
            "SLDCWorkflow : executed in {} s.".format(timing.total(SLDCWorkflow.TIMING_MERGE))
        )

        # dispatch classify
        self.logger.info("SLDCWorkflow : start dispatch/classify.")
        timing.start(SLDCWorkflow.TIMING_DC)
        pred, proba, dispatch_indexes = self._dispatch_classify(image, polygons, timing)
        timing.end(SLDCWorkflow.TIMING_DC)
        self.logger.info(
            "SLDCWorkflow : end dispatch/classify.\n" +
            "SLDCWorkflow : executed in {} s.".format(timing.total(SLDCWorkflow.TIMING_DC))
        )

        return WorkflowInformation(polygons, pred, timing, dispatches=(dispatch_indexes, "dispatch"), probas=(proba, "proba"))

    def _segment_locate(self, tile_topology, timing):
        """Execute the segment locate phase
        Parameters
        ----------
        tile_topology: TileTopology
            A tile topology
        timing: WorkflowTiming
            A workflow timing object for computing time

        Returns
        -------
        tiles: iterable (size: n, subtype: int) 
            Iterable containing the tiles ids
        tile_polygons: iterable (size: n, subtype: iterable of Polygon objects))
            The iterable at index i contains the polygons and pixel values found in the tile having index tiles[i]
        """
        # partition the tiles into batches for submitting them to processes
        tiles, tile_polygons = _parallel_segment_locate(
            self.pool,
            segmenter=self._segmenter,
            locator=self._locator,
            logger=self.logger,
            tile_topology=tile_topology,
            timing=timing
        )
        return tiles, list(map(lambda l: [t[0] for t in l], tile_polygons))

    def _dispatch_classify(self, image, polygons, timing):
        """Execute dispatching and classification on several processes

        Parameters
        ----------
        image: Image
            The image to process
        polygons: iterable (subtype: shapely.geometry.Polygon, size: N)
            The polygons to process
        timing: WorkflowTiming
            The workflow timing object to which must be appended the execution times

        Returns
        -------
        predictions: iterable (subtype: int|None, size: N)
            A list of integer codes indicating the predicted classes.
            If none of the dispatching rules matched the polygon, the prediction associated with it is the one produced
            by the fail callback for the given polygon. Especially, if no fail callback was registered, None is
            returned.
        probabilities: iterable (subtype: float, range: [0,1], size: N)
            The probabilities associated with each predicted classes (0.0 for polygons that were not dispatched)
        dispatches: iterable (size: N)
            An iterable containing the identifiers of the rule that matched the polygons. If dispatching labels were
            provided at construction, those are used to identify the rules. Otherwise, the integer indexes of the rules
            in the list provided at construction are used. Polygons that weren't matched by any rule are returned -1 as
            dispatch index.
        """
        if len(polygons) == 0:
            return np.array([]), np.array([]), np.array([])

        timing_root = ".".join([SLDCWorkflow.TIMING_ROOT, SLDCWorkflow.TIMING_DC])

        # disable parallel processing if user
        pool = self.pool if self._parallel_dispatch_classify else Parallel(n_jobs=1)
        n_jobs = self.n_jobs if self._parallel_dispatch_classify else 1

        batches = batch_split(n_jobs, polygons)
        results = pool(delayed(_dc_with_timing)(self._dispatch_classifier, image, batch, timing_root) for batch in batches)
        predictions, probabilities, dispatch, timings = zip(*results)

        # flatten
        predictions = [pred for preds in predictions for pred in preds]
        probabilities = [proba for probs in probabilities for proba in probs]
        dispatch = [disp for disps in dispatch for disp in disps]

        # merge timings
        for curr_timing in timings:
            timing.merge(curr_timing)

        return predictions, probabilities, dispatch


class SSLWorkflow(Workflow):
    """SSL stands for Semantic-Segment-Locate. Detection is performed by a semantic segmentation algorithm.
    """
    TIMING_ROOT = "workflow.ssl"
    TIMING_DETECT = "detect"
    TIMING_DETECT_LOAD = "load"
    TIMING_DETECT_SEGMENT = "segment"
    TIMING_DETECT_LOCATE = "locate"
    TIMING_MERGE = "merge"

    def __init__(self, segmenter, tile_builder, background_class=-1, tile_max_width=1024, tile_max_height=1024,
                 tile_overlap=7, dist_tolerance=1, logger=SilentLogger(), n_jobs=None):
        """Constructor
        Parameters
        ----------
        segmenter: SemanticSegmenter
            The semantic segmenter
        tile_builder: TileBuilder
            An object for building specific tiles
        background_class: int (default: -1)
            The background class not to locate (-1 
        tile_max_width: int (optional, default: 1024)
            The maximum width of the tiles when iterating over the image
        tile_max_height: int (optional, default: 1024)
            The maximum height of the tiles when iterating over the image
        tile_overlap: int (optional, default: 5)
            The number of pixels of overlap between tiles when iterating over the image
        dist_tolerance: int (optional, default, 7)
            Maximal distance between two polygons so that they are considered from the same object
        logger: Logger (optional, default: SilentLogger)
            A logger object
        n_jobs: int (optional, default: 1)
            The number of job available for executing the workflow.
        """
        super(SSLWorkflow, self).__init__(
            n_jobs=n_jobs,
            tile_max_height=tile_max_height,
            tile_max_width=tile_max_width,
            tile_builder=tile_builder,
            tile_overlap=tile_overlap,
            logger=logger
        )
        self._segmenter = segmenter
        self._locator = SemanticLocator(background=background_class)
        self._merger = SemanticMerger(tolerance=dist_tolerance)

    def process(self, image):
        """Process function"""
        timing = WorkflowTiming(root=SSLWorkflow.TIMING_ROOT)
        tile_topology = self._tile_topology(image)

        # segment locate
        self.logger.info("SLDCWorkflow : start segment/locate.")
        timing.start(SSLWorkflow.TIMING_DETECT)
        tiles, tile_polygons, tile_labels = self._segment_locate(tile_topology, timing)
        timing.end(SSLWorkflow.TIMING_DETECT)
        self.logger.info(
            "SLDCWorkflow : end segment/locate." + os.linesep +
            "SLDCWorkflow : {} tile(s) processed in {} s.".format(len(tiles), timing.total(SSLWorkflow.TIMING_DETECT)) + os.linesep +
            "SLDCWorkflow : {} polygon(s) found on those tiles.".format(sum([len(polygons) for polygons in tile_polygons]))
        )

        # merge
        self.logger.info("SLDCWorkflow : start merging")
        timing.start(SSLWorkflow.TIMING_MERGE)
        polygons, labels = self._merger.merge(tiles, tile_polygons, tile_topology, labels=tile_labels)
        timing.end(SSLWorkflow.TIMING_MERGE)
        self.logger.info(
            "SLDCWorkflow : end merging." + os.linesep +
            "SLDCWorkflow : {} polygon(s) found.".format(len(polygons)) + os.linesep +
            "SLDCWorkflow : executed in {} s.".format(timing.total(SSLWorkflow.TIMING_MERGE))
        )

        n_polygons = polygons.shape[0]
        return WorkflowInformation(polygons, labels, timing)

    def _segment_locate(self, tile_topology, timing):
        """Execute the segment locate phase
        Parameters
        ----------
        tile_topology: TileTopology
            A tile topology
        timing: WorkflowTiming
            A workflow timing object for computing time

        Returns
        -------
        tiles: iterable (size: n, subtype: int) 
            Iterable containing the tiles ids
        tile_polygons: iterable (size: n, subtype: iterable of Polygon objects))
            The iterable at index i contains the polygons and pixel values found in the tile having index tiles[i]
        """
        tiles, tile_polygons_labels = _parallel_segment_locate(
            self.pool,
            segmenter=self._segmenter,
            locator=self._locator,
            logger=self.logger,
            tile_topology=tile_topology,
            timing=timing
        )
        tile_polygons = list(map(lambda l: [t[0] for t in l], tile_polygons_labels))
        tile_labels = list(map(lambda l: [t[1] for t in l], tile_polygons_labels))
        return tiles, tile_polygons, tile_labels