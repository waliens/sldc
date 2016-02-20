# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"

from sldc import Image
from sldc.merger import Merger
from sldc.tile import TileBuilder


class SLDCWorkflow(object):
    """
    A workflow for finding objects on large images and computing a class for these objects.
    """

    def __init__(self, segmenter, locator, dispatcher_classifier, tiles_shape=(1024, 1024), boundary_thickness=7):
        """Constructor for SLDCWorkflow objects

        Parameters
        ----------
        segmenter: Segmenter
            The segmenter to use for the "Segment" step
        locator: Locator
            The locator to use for the "Locate" step
        dispatcher_classifier: DispatcherClassifier
            The dispatcher classifier to use for the "Dispatch" and "Classify" steps
        tiles_shape:
            The shape of the tiles to extract when iterating over the image. Make sure the resulting
            tiles fit into the machine's memory.
        boundary_thickness:
            The thickness between of the boundaries between the tiles for merging
        """
        self._tiles_shape = tiles_shape
        self._segmenter = segmenter
        self._locator = locator
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
        tile_builder = TileBuilder(image)
        polygons_tiles = [(tile, self._segment_locate(tile))
                          for tile in image.tile_iterator(tile_builder,
                                                          max_width=self._tiles_shape[0],
                                                          max_height=self._tiles_shape[1])]
        polygons = self._merger.merge(polygons_tiles)
        return [(polygon, self._dispatch_classifier.dispatch_classify(polygon))
                for polygon in polygons]

    def _segment_locate(self, tile):
        segmented = self._segmenter.segment(tile.get_numpy_repr())
        return self._locator.locate(segmented)
