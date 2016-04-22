# -*- coding: utf-8 -*-

from image import Image, TileBuilder, TileTopologyIterator
from merger import Merger
from locator import Locator
from information import WorkflowInformation
from timing import WorkflowTiming

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"


class SLDCWorkflow(object):
    """
    A workflow for finding objects on large images and computing a class for these objects.
    """

    def __init__(self, segmenter, dispatcher_classifier, tile_builder,
                 tile_max_width=1024, tile_max_height=1024, tile_overlap=5,
                 boundary_thickness=7):
        """Constructor for SLDCWorkflow objects

        Parameters
        ----------
        segmenter: Segmenter
            The segmenter to use for the "Segment" step
        dispatcher_classifier: DispatcherClassifier
            The dispatcher classifier to use for the "Dispatch" and "Classify" steps
        tile_builder: TileBuilder
            An object for building tiles
        tile_max_width: int, optional (default: 1024)
            The maximum width of the tiles when iterating over the image
        tile_max_height: int, optional (default: 1024)
            The maximum height of the tiles when iterating over the image
        tile_overlap: int, optional (default: 5)
            The number of pixels of overlap between tiles when iterating over the image
        boundary_thickness:
            The thickness between of the boundaries between the tiles for merging
        """
        self._tile_max_width = tile_max_width
        self._tile_max_height = tile_max_height
        self._tile_overlap = tile_overlap
        self._tile_builder = tile_builder
        self._segmenter = segmenter
        self._locator = Locator()
        self._merger = Merger(boundary_thickness)
        self._dispatch_classifier = dispatcher_classifier

    def process(self, image, logger):
        """Process the image using the SLDCWorkflow
        Parameters
        ----------
        image: Image
            The image to process
        Returns
        -------
        polygons_classes: array of 2-tuples
            An array containing the found polygons as well as the predicted class. These data are
            structured in an array of tuple where a tuple contains as its first element the polygon
            object (shapely.geometry.Polygon) and as second element the predicted class (integer code).
        logger: Logger
            A logger
        Notes
        -----
        This method doesn't modify the image passed as parameter.
        This method doesn't modify the object's attributes.
        """
        timing = WorkflowTiming()
        tile_topology = image.tile_topology(max_width=self._tile_max_width, max_height=self._tile_max_height,
                                            overlap=self._tile_overlap)
        tile_iterator = TileTopologyIterator(self._tile_builder, tile_topology, silent_fail=True)

        # segment locate
        polygons_tiles = list()
        logger.info("SLDCWorkflow : start segment/locate.")
        for i, tile in enumerate(tile_iterator):
            # log percentage of progress if there are enough tiles
            if tile_topology.tile_count > 25 and (i + 1) % (tile_topology.tile_count // 10) == 0:
                pass
            logger.info("SLDCWorkflow : {}% of tile segmented in {} s.".format(100.0 * i / tile_topology.tile_count,
                                                                               self._sl_total_time(timing)))
        polygons_tiles.append((tile, self._segment_locate(tile, timing)))
        logger.info("SLDCWorkflow : end segment/locate. {} tile(s) processed in {} s.".format(len(polygons_tiles)))

        # merge
        logger.info("SLDCWorkflow : start merging")
        timing.start_merging()
        polygons = self._merger.merge(polygons_tiles, tile_topology)
        timing.end_merging()
        logger.info("SLDCWorkflow : end segment locate. {} polygon(s) found.".format(len(polygons)))

        # dispatch classify
        logger.info("SLDCWorkflow : start dispatch/classify")
        timing.start_dispatch_classify()
        predictions, dispatch_indexes = self._dispatch_classifier.dispatch_classify_batch(image, polygons)
        timing.end_dispatch_classify()
        logger.info("SLDCWorkflow : end dispatch/classify ")

        return WorkflowInformation(polygons, dispatch_indexes, predictions, timing, metadata=self.get_metadata())

    def _segment_locate(self, tile, timing):
        timing.start_fetching()
        np_image = tile.np_image
        timing.end_fetching()
        timing.start_segment()
        segmented = self._segmenter.segment(np_image)
        timing.end_segment()
        timing.start_location()
        located = self._locator.locate(segmented, offset=tile.offset)
        timing.end_location()
        return located

    def get_metadata(self):
        """Return the metadata associated with this workflow
        The metadata should be a way for the implementor to document the results of the workflow execution.
        Returns
        -------
        metadata: string
            The workflow metadata
        """
        return "Workflow class: {}\n".format(self.__class__.__name__)