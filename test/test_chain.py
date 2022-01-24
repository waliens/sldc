# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np

from sldc import SLDCWorkflowBuilder, Segmenter, PolygonClassifier, WorkflowChainBuilder, DispatchingRule, PolygonFilter
from sldc.util import has_alpha_channel
from test.util import NumpyImage, relative_error, draw_multisquare, draw_multicircle, circularity

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class BigShapeSegmenter(Segmenter):
    """A segmenter which matches pixels greater than 0"""
    def segment(self, image):
        if has_alpha_channel(image):
            image = np.squeeze(image[:, :, 0:-1])
        return (image > 0).astype(np.uint8)


class SmallSquareSegmenter(Segmenter):
    """A segmenter which matches pixels in the range ]100, 200["""
    def segment(self, image):
        if has_alpha_channel(image):
            image = np.squeeze(image[:, :, 0:-1])
        return np.logical_and(image < 200, image > 100).astype(np.uint8)


class SmallCircleSegmenter(Segmenter):
    """A segementer which matches pixels in the range ]50, 100["""
    def segment(self, image):
        if has_alpha_channel(image):
            image = np.squeeze(image[:, :, 0:-1])
        return np.logical_and(image < 100, image > 50).astype(np.uint8)


class DumbClassifier(PolygonClassifier):
    """A classifier which always return class 1 and probability 1.0"""
    def predict(self, image, polygon):
        return 1, 1.0


class CircleDispatch(DispatchingRule):
    """A rule that dispatches circles"""
    def evaluate(self, image, polygon):
        return circularity(polygon.buffer(5).buffer(-5)) > 0.85


class SquareDispatch(DispatchingRule):
    """A rule that dispatches squares"""
    def evaluate(self, image, polygon):
        return circularity(polygon.buffer(5).buffer(-5)) < 0.85


class CircleShapeFilter(PolygonFilter):
    """A filter which excludes all shapes which were not detected by the first workflow and which are not circles"""
    def filter(self, chain_information):
        workflow_info = chain_information[0]
        return [workflow_info[i].polygon for i in range(len(workflow_info)) if workflow_info[i].dispatch == "circle"]


class SquareShapeFilter(PolygonFilter):
    """A filter which excludes all shapes which were not detected by the first workflow and which are not squares"""
    def filter(self, chain_information):
        workflow_info = chain_information[0]
        return [workflow_info[i].polygon for i in range(len(workflow_info)) if workflow_info[i].dispatch == "square"]


class TestChaining(TestCase):
    """A test case for testing the chaining"""
    def testSquareIncluded(self):
        # generate the image to be processed
        w, h = 2000, 2000
        image = np.zeros((h, w), dtype=np.uint8)

        # locations of the 9 multi-squares
        positions = [
            (w // 7, h // 7),
            (3 * w // 7, h // 7),
            (5 * w // 7, h // 7),
            (w // 7, 3 * h // 7),
            (3 * w // 7, 3 * h // 7),
            (5 * w // 7, 3 * h // 7),
            (w // 7, 5 * h // 7),
            (3 * w // 7, 5 * h // 7),
            (5 * w // 7, 5 * h // 7)
        ]

        for position in positions:
            image = draw_multisquare(image, position, w // 7, color_in=127)

        # Build workflow
        builder = SLDCWorkflowBuilder()

        # Build workflow 1
        builder.set_segmenter(BigShapeSegmenter())
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
        big_area = (1 + (w // 7)) ** 2
        small_area = (1 + (w / 35)) ** 2

        info1 = chain_info["big_squares"]
        self.assertEqual(9, len(info1))
        for object_info in info1:
            self.assertTrue(relative_error(object_info.polygon.area, big_area) < 0.005)
            self.assertEqual("catchall", object_info.dispatch)
            self.assertEqual(1, object_info.label)
            self.assertAlmostEqual(1.0, object_info.proba)

        info2 = chain_info["small_squares"]
        self.assertEqual(36, len(info2))
        for object_info in info2:
            self.assertTrue(relative_error(object_info.polygon.area, small_area) < 0.005)
            self.assertEqual("catchall", object_info.dispatch)
            self.assertEqual(1, object_info.label)
            self.assertAlmostEqual(1.0, object_info.proba)

    def testSquareAndCircleIncluded(self):
        w, h = 2000, 2000
        image = np.zeros((h, w), dtype=np.uint8)
        # locations of the 9 multi-squares
        shapes = [
            ("c", (w // 7, h // 7)),
            ("s", (3 * w // 7, h // 7)),
            ("s", (5 * w // 7, h // 7)),
            ("s", (w // 7, 3 * h // 7)),
            ("c", (3 * w // 7, 3 * h // 7)),
            ("s", (5 * w // 7, 3 * h // 7)),
            ("c", (w // 7, 5 * h // 7)),
            ("s", (3 * w // 7, 5 * h // 7)),
            ("c", (5 * w // 7, 5 * h // 7))
        ]

        for shape, position in shapes:
            if shape == "c":
                image = draw_multicircle(image, position, w // 7, color_in=87)
            elif shape == "s":
                image = draw_multisquare(image, position, w // 7, color_in=187)

        # Build workflows
        # 1st: find big shapes and dispatch them as circle or square
        # 2nd: find small circles in found circle shapes
        # 3rd: find small squares in found square shape
        builder = SLDCWorkflowBuilder()

        builder.set_segmenter(BigShapeSegmenter())
        builder.add_classifier(CircleDispatch(), DumbClassifier(), dispatching_label="circle")
        builder.add_classifier(SquareDispatch(), DumbClassifier(), dispatching_label="square")
        builder.set_tile_size(512, 512)
        workflow1 = builder.get()

        builder.set_segmenter(SmallCircleSegmenter())
        builder.add_catchall_classifier(DumbClassifier())
        workflow2 = builder.get()

        builder.set_segmenter(SmallSquareSegmenter())
        builder.add_catchall_classifier(DumbClassifier())
        workflow3 = builder.get()

        # Build chain
        chain_builder = WorkflowChainBuilder()
        chain_builder.set_first_workflow(workflow1)
        chain_builder.add_executor(workflow2, filter=CircleShapeFilter())
        chain_builder.add_executor(workflow3, filter=SquareShapeFilter(), n_jobs=2)
        chain = chain_builder.get()

        chain_info = chain.process(NumpyImage(image))

        info1 = chain_info[0]
        self.assertEqual(9, len(info1))
        self.assertEqual(4, len([d for d in info1.dispatches if d == "circle"]))
        self.assertEqual(5, len([d for d in info1.dispatches if d == "square"]))

        info2 = chain_info[1]
        self.assertEqual(16, len(info2))

        info3 = chain_info[2]
        self.assertEqual(20, len(info3))
