# -*- coding: utf-8 -*-
from unittest import TestCase

import cv2
import numpy as np
from shapely.geometry import box

from sldc import WorkflowBuilder, Segmenter, PolygonClassifier, WorkflowChainBuilder, PolygonTranslatorWorkflowExecutor, PostProcessor
from test.util import draw_poly, NumpyImage

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def draw_multisquare(image, position, size, color_out=255, color_in=255):
    """Draw a square with color 'color_out' and given size at a given position (x, y)
        Then draw four square of size (size/5) with color 'color_in' at:
            1) coord: (y + (size / 5), x + (size / 5))
            2) coord: (y + (size / 5), x + (3 * size / 5))
            3) coord: (y + (3 * size / 5), x + (size / 5))
            4) coord: (y + (3 * size / 5), x + (3 * size / 5))
    """
    x, y = position
    small_size = size / 5
    image = draw_poly(image, box(x, y, x + size, y + size), color=color_out)
    square1 = box(x +     small_size, y +     small_size, x + 2 * small_size, y + 2 * small_size)
    square2 = box(x + 3 * small_size, y +     small_size, x + 4 * small_size, y + 2 * small_size)
    square3 = box(x +     small_size, y + 3 * small_size, x + 2 * small_size, y + 4 * small_size)
    square4 = box(x + 3 * small_size, y + 3 * small_size, x + 4 * small_size, y + 4 * small_size)
    squares = [square1, square2, square3, square4]
    for square in squares:
        image = draw_poly(image, square, color=color_in)
    return image


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


class SecondExecutor(PolygonTranslatorWorkflowExecutor):
    """A workflow executor which processes all polygons"""
    def get_windows(self, image, workflow_info_collection):
        return [image.window_from_polygon(polygon) for polygon, _, _, _ in workflow_info_collection.polygons()]


class TestPostProcessor(PostProcessor):
    """A post processor which performs unit tests on the results"""
    def __init__(self, testcase, big_area, small_area):
        PostProcessor.__init__(self)
        self._testcase = testcase
        self._barea = big_area
        self._sarea = small_area

    def post_process(self, image, workflow_info_collection):
        info1 = workflow_info_collection[0]
        self._testcase.assertEqual(9, len(info1))
        for polygon, disp, cls, proba in info1:
            self._testcase.assertTrue(self.relative_error(polygon.area, self._barea) < 0.005)
            self._testcase.assertEqual("catchall", disp)
            self._testcase.assertEqual(1, cls)
            self._testcase.assertAlmostEqual(1.0, proba)

        info2 = workflow_info_collection[1]
        self._testcase.assertEqual(36, len(info2))
        for polygon, disp, cls, proba in info2:
            self._testcase.assertTrue(self.relative_error(polygon.area, self._sarea) < 0.005)
            self._testcase.assertEqual("catchall", disp)
            self._testcase.assertEqual(1, cls)
            self._testcase.assertAlmostEqual(1.0, proba)

    @staticmethod
    def relative_error(val, ref):
        return np.abs(val - ref) / ref


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
        chain_builder.add_full_image_executor(workflow1)
        chain_builder.add_executor(SecondExecutor(workflow2))
        chain_builder.set_default_provider([NumpyImage(image)])
        chain_builder.set_post_processor(TestPostProcessor(self, (w / 7) ** 2, (w / 35) ** 2))
        chain = chain_builder.get()

        # Launch
        chain.execute()



