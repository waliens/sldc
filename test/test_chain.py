# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
from shapely.geometry import box

from sldc import WorkflowBuilder, Segmenter, PolygonClassifier, WorkflowChainBuilder
from test.util import draw_poly, NumpyImage, relative_error, draw_multisquare

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class BigSquareSegmenter(Segmenter):
    """A segmenter which matches pixels greater than 0"""
    def segment(self, image):
        return (image > 0).astype(np.uint8)


class SmallSquareSegmenter(Segmenter):
    """A segmenter which matches pixels in the range [100, 200]"""
    def segment(self, image):
        return np.logical_and(image < 200, image > 100).astype(np.uint8)


class DumbClassifier(PolygonClassifier):
    """A classifier which always return class 1 and probability 1.0"""
    def predict_batch(self, image, polygons):
        return [1] * len(polygons), [1.0] * len(polygons)


class TestChaining(TestCase):
    """A test case for testing the chaining"""
    def testSquareIncluded(self):
        # generate the image to be processed
        w, h = 4000, 4000
        image = np.zeros((h, w), dtype=np.uint8)

        # locations of the 9 multi-squares
        positions = [
            (w / 7, h / 7),
            (3 * w / 7, h / 7),
            (5 * w / 7, h / 7),
            (w / 7, 3 * h / 7),
            (3 * w / 7, 3 * h / 7),
            (5 * w / 7, 3 * h / 7),
            (w / 7, 5 * h / 7),
            (3 * w / 7, 5 * h / 7),
            (5 * w / 7, 5 * h / 7)
        ]

        for position in positions:
            image = draw_multisquare(image, position, w / 7, color_in=127)

        # Build workflow
        builder = WorkflowBuilder(n_jobs=1)

        # Build workflow 1
        builder.set_segmenter(BigSquareSegmenter())
        builder.add_catchall_classifier(DumbClassifier())
        builder.set_tile_size(512, 512)
        workflow1 = builder.get()

        # Build workflow 2
        builder.set_segmenter(SmallSquareSegmenter())
        builder.add_catchall_classifier(DumbClassifier())
        workflow2 = builder.get()

        # Build chaining
        chain_builder = WorkflowChainBuilder()
        chain_builder.set_first_workflow(workflow1, label="big_squares")
        chain_builder.add_executor(workflow2, label="small_squares")
        chain = chain_builder.get()

        # Launch
        chain_info = chain.process(NumpyImage(image))

        # check results
        big_area = (w / 7) ** 2
        small_area = (w / 35) ** 2

        info1 = chain_info["big_squares"]
        self.assertEqual(9, len(info1))
        for polygon, disp, cls, proba in info1:
            self.assertTrue(relative_error(polygon.area, big_area) < 0.005)
            self.assertEqual("catchall", disp)
            self.assertEqual(1, cls)
            self.assertAlmostEqual(1.0, proba)

        info2 = chain_info["small_squares"]
        self.assertEqual(36, len(info2))
        for polygon, disp, cls, proba in info2:
            self.assertTrue(relative_error(polygon.area, small_area) < 0.005)
            self.assertEqual("catchall", disp)
            self.assertEqual(1, cls)
            self.assertAlmostEqual(1.0, proba)
