# -*- coding: utf-8 -*-

from image import Image, TileBuilder, TileTopologyIterator
from merger import Merger
from locator import Locator

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

    def process(self, image):
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
        Notes
        -----
        This method doesn't modify the image passed as parameter.
        This method doesn't modify the object's attributes.
        """
        tile_topology = image.tile_topology(max_width=self._tile_max_width,
                                            max_height=self._tile_max_height,
                                            overlap=self._tile_overlap)
        tile_iterator = TileTopologyIterator(self._tile_builder, tile_topology, skip_on_error=True)
        polygons_tiles = [(tile, self._segment_locate(tile)) for tile in tile_iterator]
        polygons = self._merger.merge(polygons_tiles, tile_topology)
        return zip(polygons, self._dispatch_classifier.dispatch_classify_batch(image, polygons))

    def _segment_locate(self, tile):
        segmented = self._segmenter.segment(tile.get_numpy_repr())
        return self._locator.locate(segmented, offset=tile.offset)
