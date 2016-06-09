# -*- coding: utf-8 -*-
from joblib import delayed

from .errors import TileExtractionException
from .image import Image, TileBuilder
from .information import WorkflowInformation
from .locator import Locator
from .logging import Loggable, SilentLogger
from .merger import Merger
from .timing import WorkflowTiming

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
    timing.start_loading()
    np_image = tile.np_image
    timing.end_loading()
    timing.start_segment()
    segmented = segmenter.segment(np_image)
    timing.end_segment()
    timing.start_location()
    located = locator.locate(segmented, offset=tile.offset)
    timing.end_location()
    return located


def _sl_with_timing(tile_ids, tile_topology, segmenter, locator):
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
    timing = WorkflowTiming()
    tiles_polygons = list()
    for tile_id in tile_ids:
        try:
            tile = tile_topology.tile(tile_id)
            tiles_polygons.append((tile_id, _segment_locate(tile, segmenter, locator, timing)))
        except TileExtractionException:
            tiles_polygons.append((tile_id, None))
    return timing, tiles_polygons


class SLDCWorkflow(Loggable):
    """A class that coordinates various components of the SLDC workflow in order to detect objects and return
    their information.
    """

    def __init__(self, segmenter, dispatcher_classifier, tile_builder,
                 tile_max_width=1024, tile_max_height=1024, tile_overlap=7,
                 dist_tolerance=1, logger=SilentLogger(), worker_pool=None):
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
        worker_pool: Parallel (optional, default: null)
            The number of jobs for segmenting and locating the tiles
        """
        Loggable.__init__(self, logger)
        self._tile_max_width = tile_max_width
        self._tile_max_height = tile_max_height
        self._tile_overlap = tile_overlap
        self._tile_builder = tile_builder
        self._segmenter = segmenter
        self._locator = Locator()
        self._merger = Merger(dist_tolerance)
        self._dispatch_classifier = dispatcher_classifier
        self._pool = worker_pool
        self._n_jobs = worker_pool.n_jobs if self._pool is not None else 1

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
        This method doesn't modify the object's attributes.
        """
        timing = WorkflowTiming()
        tile_topology = image.tile_topology(self._tile_builder, max_width=self._tile_max_width,
                                            max_height=self._tile_max_height, overlap=self._tile_overlap)

        # segment locate
        self.logger.info("SLDCWorkflow : start segment/locate.")
        if self._n_jobs == 1:
            polygons_tiles = self._sl_sequential(tile_topology, timing)
        else:
            polygons_tiles = self._sl_parallel(tile_topology, timing)

        # log end of segment locate
        self.logger.info("SLDCWorkflow : end segment/locate.\n" +
                         "SLDCWorkflow : {} tile(s) processed in {} s.\n".format(len(polygons_tiles), timing.lsl_total_duration()) +
                         "SLDCWorkflow : {} polygon(s) found on those tiles.".format(sum([len(polygons) for _, polygons in polygons_tiles])))

        # merge
        self.logger.info("SLDCWorkflow : start merging")
        timing.start_merging()
        polygons = self._merger.merge(polygons_tiles, tile_topology)
        timing.end_merging()
        self.logger.info("SLDCWorkflow : end merging.\n" +
                         "SLDCWorkflow : {} polygon(s) found.\n".format(len(polygons)) +
                         "SLDCWorkflow : executed in {} s.".format(timing.duration_of(WorkflowTiming.MERGING)))

        # dispatch classify
        self.logger.info("SLDCWorkflow : start dispatch/classify.")
        pred, proba, dispatch_indexes = self._dispatch_classifier.dispatch_classify_batch(image, polygons, timing)
        self.logger.info("SLDCWorkflow : end dispatch/classify.\n" +
                         "SLDCWorkflow : executed in {} s.".format(timing.dc_total_duration()))

        return WorkflowInformation(polygons, dispatch_indexes, pred, proba, timing)

    def _segment_locate(self, tile, timing):
        """Fetch a tile and applies it segmentation and location

        Parameters
        ----------
        tile: Tile
            The tile to process for the segment locate
        timing: WorkflowTiming
            The workflow timing object for measuring the execution times of the various steps

        Returns
        -------
        polygons: iterable (subtype: shapely.geometry.Polygon)
            Iterable containing the polygons found by the locate step
        """
        try:
            return _segment_locate(tile, self._segmenter, self._locator, timing)
        except TileExtractionException as e:
            self.logger.warning("SLDCWorkflow : skip tile {} because it couldn't be extracted.\n".format(tile.identifier) +
                                "SLDCWorkflow : fetch error : {}".format(e.message))
            return []

    def _sl_sequential(self, tile_topology, timing):
        """Execute the segment locate phase in a sequential order
        Parameters
        ----------
        tile_topology: TileTopology
            A tile topology
        timing: WorkflowTiming
            A workflow timing object for computing time

        Returns
        -------
        tiles_polygons: iterable (subtype: (int, shapely.geometry.Polygon))
            An iterable containing tuples (tile_id, polygons) where the tile is a Tile object and polygons another
            iterable containing the polygons found on the tile
        """
        polygons_tiles = list()
        timing.start_lsl()
        for i, tile in enumerate(tile_topology):
            # log percentage of progress if there are enough tiles
            if tile_topology.tile_count > 25 and (i + 1) % (tile_topology.tile_count // 10) == 0:
                self.logger.info("SLDCWorkflow : {}/{} tiles processed (segment/locate).\n".format(i+1, tile_topology.tile_count) +
                                 "SLDCWorkflow : segment/locate duration is {} s until now.".format(timing.sl_total_duration()))
            polygons_tiles.append((tile.identifier, self._segment_locate(tile, timing)))
        timing.end_lsl()
        return polygons_tiles

    def _sl_parallel(self, tile_topology, timing):
        """Execute the segment locate phase in parallel
        Parameters
        ----------
        tile_topology: TileTopology
            The tile topology defining the tile to process
        timing: WorkflowTiming
            A workflow timing object for computing time

        Returns
        -------
        tiles_polygons: iterable
            An iterable containing tuples (tile_id, polygons) where the tile_id is the tile identifier and polygons
            another iterable containing the polygons found on the tile
        """
        # partition the tiles into batches for submitting them to processes
        batches = tile_topology.partition_identifiers(self._pool.n_jobs)

        # execute in parallel
        timing.start_lsl()
        results = self._pool(delayed(_sl_with_timing)(tile_ids, tile_topology, self._segmenter, self._locator)
                             for tile_ids in batches)
        timing.end_lsl()

        # merge sub timings, merge tile and located polygons and  log info of tiles that weren't be loaded successfully
        merged_tiles_polygons = list()
        for sub_timing, tiles_polygons in results:
            timing.merge(sub_timing)
            for i, (tile_id, polygons) in enumerate(tiles_polygons):
                if polygons is None:
                    self._logger.warning("SLDCWorkflow : Tile {} couldn't be fetched during parallel computations.".format(tile))
                    merged_tiles_polygons.append((tile_id, []))
                else:
                    merged_tiles_polygons.append((tile_id, polygons))

        # return tiles polygons
        return merged_tiles_polygons
