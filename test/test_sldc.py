# -*- coding: utf-8 -*-
import unittest
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from sldc import Dispatcher, report_timing, StandardOutputLogger, Logger
from sldc import DispatchingRule, PolygonClassifier, SLDCWorkflowBuilder, Segmenter
from test.util import circularity, draw_circle, draw_square, draw_poly, NumpyImage, relative_error

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class CircleSegmenter(Segmenter):
    def segment(self, image):
        """Segment a grey circle in black image"""
        segmented = (image[:, :, 0] > 50)
        return segmented.astype("uint8") * 255


class CircleClassifier(PolygonClassifier):
    def predict(self, image, polygon):
        """A polygon classifier which always predict 1 with a probablility 1.0"""
        return 1, 1.0


class CircleRule(DispatchingRule):
    """A rule which matches circle polygons"""
    def evaluate(self, image, polygon):
        return circularity(polygon.buffer(5).buffer(-5)) > 0.85


class SquareRule(DispatchingRule):
    """A rule that matches square polygons"""
    def evaluate(self, image, polygon):
        return circularity(polygon.buffer(5).buffer(-5)) <= 0.8


class MinAreaRule(DispatchingRule):
    """A rule matching polygons greater than a given area"""
    def __init__(self, min_area):
        self._min_area = min_area

    def evaluate(self, image, polygon):
        return polygon.area > self._min_area


class ColorClassifier(PolygonClassifier):
    """A classifier which returns the color class of the center point of the polygon"""
    GREY = 0
    WHITE = 1

    def predict(self, image, polygon):
        window = image.window_from_polygon(polygon)
        sub_image = window.np_image
        pxl = sub_image[int(polygon.centroid.y) - window.offset_y][int(polygon.centroid.x) - window.offset_x]
        if pxl == 255:
            return ColorClassifier.WHITE, 1.0
        elif 0 < pxl < 255:
            return ColorClassifier.GREY, 1.0
        else:
            return None, 1.0


class CustomSegmenter(Segmenter):
    """Every non black pixel are in object of interests"""
    def segment(self, image):
        return (image > 0).astype("uint8")


class CustomDispatcher(Dispatcher):
    """Dispatch polygons as big or small"""
    def __init__(self, thresh_area):
        super(CustomDispatcher, self).__init__()
        self._thresh_area = thresh_area

    def dispatch(self, image, polygon):
        return "BIG" if polygon.area > self._thresh_area else "SMALL"


class TestFullWorkflow(TestCase):
    def testNoObjects(self):
        """Test detection on empty image"""
        w, h = 200, 200
        image = np.zeros((h, w), dtype="uint8")

        builder = SLDCWorkflowBuilder()
        builder.set_segmenter(CustomSegmenter())
        builder.add_classifier(CircleRule(), ColorClassifier(), dispatching_label="circle")
        builder.add_classifier(SquareRule(), ColorClassifier())
        workflow = builder.get()

        # Execute
        results = workflow.process(NumpyImage(image))

        self.assertEqual(len(results), 0)

    def testFindCircleAndSquare(self):
        """Test the workflow on an image containing both squares and circles of different colors
        Two squares of side: 200
            - white centered in (1000, 1000)
            - grey centered in (3000, 3000)
        Two circles of side: 100
            - white centered in (1000, 3000)
            - grey centered in (3000, 1000)
        Another white square of side 300 centered in (2000,2000)
        """
        # square
        w, h = 2000, 2000
        image = np.zeros((w, h,), dtype="uint8")
        image = draw_circle(image, 100, (500, 1500), 255)
        image = draw_circle(image, 100, (1500, 600), 127)
        image = draw_square(image, 200, (500, 500), 255)
        image = draw_square(image, 200, (1500, 1500), 127)
        image = draw_square(image, 300, (1000, 1000), 255)

        # Build the workflow
        builder = SLDCWorkflowBuilder()
        builder.set_segmenter(CustomSegmenter())
        builder.add_classifier(CircleRule(), ColorClassifier(), dispatching_label="circle")
        builder.add_classifier(SquareRule(), ColorClassifier())
        workflow = builder.get()

        # Execute
        results = workflow.process(NumpyImage(image))

        # count check
        count = len(results)
        self.assertEqual(count, 5)

        # sort polygons
        sorted_idx = sorted(range(count), key=lambda i: (results.polygons[i].centroid.y, results.polygons[i].centroid.x))

        # first square
        square1 = results.polygons[sorted_idx[0]]
        self.assertEqual(relative_error(square1.area, 201 * 201) < 0.005, True)
        self.assertEqual(relative_error(square1.centroid.x, 500) < 0.005, True)
        self.assertEqual(relative_error(square1.centroid.y, 500) < 0.005, True)
        self.assertEqual(results.dispatches[sorted_idx[0]], "1")  # square
        self.assertEqual(results.labels[sorted_idx[0]], ColorClassifier.WHITE)  # white
        self.assertAlmostEqual(results.probas[sorted_idx[0]], 1.0)

        # first circle
        circle1 = results.polygons[sorted_idx[1]]
        self.assertEqual(relative_error(circle1.area, np.pi * 100 * 101) < 0.005, True)
        self.assertEqual(relative_error(circle1.centroid.x, 1500) < 0.005, True)
        self.assertEqual(relative_error(circle1.centroid.y, 600) < 0.005, True)
        self.assertEqual(results.dispatches[sorted_idx[1]], "circle")  # circle
        self.assertEqual(results.labels[sorted_idx[1]], ColorClassifier.GREY)  # grey
        self.assertAlmostEqual(results.probas[sorted_idx[1]], 1.0)

        # second square (centered)
        square2 = results.polygons[sorted_idx[2]]
        self.assertEqual(relative_error(square2.area, 301 * 301) < 0.005, True)
        self.assertEqual(relative_error(square2.centroid.x, 1000) < 0.005, True)
        self.assertEqual(relative_error(square2.centroid.y, 1000) < 0.005, True)
        self.assertEqual(results.dispatches[sorted_idx[2]], "1")  # square
        self.assertEqual(results.labels[sorted_idx[2]], ColorClassifier.WHITE)  # white
        self.assertAlmostEqual(results.probas[sorted_idx[2]], 1.0)

        # second circle
        circle2 = results.polygons[sorted_idx[3]]
        self.assertEqual(relative_error(circle2.area, np.pi * 100 * 101) < 0.005, True)
        self.assertEqual(relative_error(circle2.centroid.x, 500) < 0.005, True)
        self.assertEqual(relative_error(circle2.centroid.y, 1500) < 0.005, True)
        self.assertEqual(results.dispatches[sorted_idx[3]], "circle")  # circle
        self.assertEqual(results.labels[sorted_idx[3]], ColorClassifier.WHITE)  # grey
        self.assertAlmostEqual(results.probas[sorted_idx[3]], 1.0)

        # third square
        square3 = results.polygons[sorted_idx[4]]
        self.assertEqual(relative_error(square3.area, 201 * 201) < 0.005, True)
        self.assertEqual(relative_error(square3.centroid.x, 1500) < 0.005, True)
        self.assertEqual(relative_error(square3.centroid.y, 1500) < 0.005, True)
        self.assertEqual(results.dispatches[sorted_idx[4]], "1")  # square
        self.assertEqual(results.labels[sorted_idx[4]], ColorClassifier.GREY)  # white
        self.assertAlmostEqual(results.probas[sorted_idx[4]], 1.0)

        # check other information
        timing = results.timing
        self.assertEqual(timing.get_phases_hierarchy(), {"workflow.sldc": {
            "detect": {"load": None, "segment": None, "locate": None},
            "merge": None,
            "dispatch_classify": {"dispatch": None, "classify": None}
        }})

    def testDetectCircle(self):
        """A test which executes a full workflow on image containing a white circle in the center of an black image
        """
        # generate circle image
        w, h = 2000, 2000
        image = np.zeros((w, h, 3), dtype="uint8")
        image = draw_circle(image, 750, (1000, 1000), color=[129, 129, 129])

        # build workflow
        builder = SLDCWorkflowBuilder()
        builder.set_segmenter(CircleSegmenter())
        builder.add_catchall_classifier(CircleClassifier())
        workflow = builder.get()

        # process image
        workflow_info = workflow.process(NumpyImage(image))

        # Check results
        self.assertEqual(len(workflow_info.polygons), 1)

        # Check circle
        polygon = workflow_info.polygons[0]
        self.assertEqual(relative_error(polygon.area, np.pi * 750 * 750) <= 0.005, True)
        self.assertEqual(relative_error(polygon.centroid.x, 1000) <= 0.005, True)
        self.assertEqual(relative_error(polygon.centroid.y, 1000) <= 0.005, True)
        assert_array_equal(workflow_info.labels, [1])
        assert_array_almost_equal(workflow_info.probas, [1.0])
        assert_array_equal(workflow_info.dispatches, ["catchall"])

        # check other information
        timing = workflow_info.timing
        self.assertEqual(timing.get_phases_hierarchy(), {"workflow.sldc": {
            "detect": {"load": None, "segment": None, "locate": None},
            "merge": None,
            "dispatch_classify": {"dispatch": None, "classify": None}
        }})

    def testDetectCircleParallel(self):
        """A test which executes a full workflow on image containing a white circle in the center of an black image in
        parallel
        """
        # generate circle image
        w, h = 2000, 2000
        image = np.zeros((w, h, 3), dtype="uint8")
        image = draw_circle(image, 750, (1000, 1000), [129, 129, 129])

        # build workflow
        builder = SLDCWorkflowBuilder()
        builder.set_n_jobs(2)
        builder.set_segmenter(CircleSegmenter())
        builder.add_catchall_classifier(CircleClassifier())
        builder.set_parallel_dc(True)
        workflow = builder.get()

        # process image
        workflow_info = workflow.process(NumpyImage(image))

        # Check results
        self.assertEqual(len(workflow_info.polygons), 1)

        # Check circle
        polygon = workflow_info.polygons[0]
        self.assertEqual(relative_error(polygon.area, np.pi * 750 * 750) <= 0.005, True)
        self.assertEqual(relative_error(polygon.centroid.x, 1000) <= 0.005, True)
        self.assertEqual(relative_error(polygon.centroid.y, 1000) <= 0.005, True)
        assert_array_equal(workflow_info.labels, [1])
        assert_array_almost_equal(workflow_info.probas, [1.0])
        assert_array_equal(workflow_info.dispatches, ["catchall"])

        # check other information
        timing = workflow_info.timing
        self.assertEqual(timing.get_phases_hierarchy(), {"workflow.sldc": {
            "detect": {"load": None, "segment": None, "locate": None},
            "merge": None,
            "dispatch_classify": {"dispatch": None, "classify": None}
        }})

    def testWorkflowWithCustomDispatcher(self):
        # generate circle image
        w, h = 1000, 1000
        image = np.zeros((w, h,), dtype="uint8")
        image = draw_circle(image, 10, (125, 125), 255)  # pi * 10 * 10 -> ~ 314
        image = draw_circle(image, 25, (250, 750), 255)  # pi * 25 * 25 -> ~ 1963
        image = draw_square(image, 26, (250, 250), 255)  # 26 * 26 -> 676
        image = draw_square(image, 50, (750, 750), 127)  # 50 * 50 -> 2500

        # build the workflow
        builder = SLDCWorkflowBuilder()
        builder.set_segmenter(CustomSegmenter())
        builder.set_one_shot_dispatcher(CustomDispatcher(1000), {
            "BIG": ColorClassifier(),
            "SMALL": ColorClassifier()
        })
        workflow = builder.get()

        # execute
        results = workflow.process(NumpyImage(image))

        # validate number of results
        count = len(results)
        self.assertEqual(count, 4)

        # sort polygons
        sorted_idx = sorted(range(count), key=lambda i: (results.polygons[i].centroid.y, results.polygons[i].centroid.x))

        # first circle
        circle1 = results.polygons[sorted_idx[0]]
        self.assertTrue(relative_error(circle1.area, np.pi * 10 * 11) < 0.025)
        self.assertTrue(relative_error(circle1.centroid.x, 125) < 0.025)
        self.assertTrue(relative_error(circle1.centroid.y, 125) < 0.025)
        self.assertEqual(results.dispatches[sorted_idx[0]], "SMALL")
        self.assertEqual(results.labels[sorted_idx[0]], ColorClassifier.WHITE)
        self.assertAlmostEqual(results.probas[sorted_idx[0]], 1.0)

        # first square
        square1 = results.polygons[sorted_idx[1]]
        self.assertTrue(relative_error(square1.area, 27 * 27) < 0.025)
        self.assertTrue(relative_error(square1.centroid.x, 250) < 0.025)
        self.assertTrue(relative_error(square1.centroid.y, 250) < 0.025)
        self.assertEqual(results.dispatches[sorted_idx[1]], "SMALL")
        self.assertEqual(results.labels[sorted_idx[1]], ColorClassifier.WHITE)
        self.assertAlmostEqual(results.probas[sorted_idx[1]], 1.0)

        # second circle
        circle2 = results.polygons[sorted_idx[2]]
        self.assertTrue(relative_error(circle2.area, np.pi * 25 * 26) < 0.025)
        self.assertTrue(relative_error(circle2.centroid.x, 250) < 0.025)
        self.assertTrue(relative_error(circle2.centroid.y, 750) < 0.025)
        self.assertEqual(results.dispatches[sorted_idx[2]], "BIG")
        self.assertEqual(results.labels[sorted_idx[2]], ColorClassifier.WHITE)
        self.assertAlmostEqual(results.probas[sorted_idx[2]], 1.0)

        # second square
        square2 = results.polygons[sorted_idx[3]]
        self.assertTrue(relative_error(square2.area, 51 * 51) < 0.025)
        self.assertTrue(relative_error(square2.centroid.x, 750) < 0.025)
        self.assertTrue(relative_error(square2.centroid.y, 750) < 0.025)
        self.assertEqual(results.dispatches[sorted_idx[3]], "BIG")
        self.assertEqual(results.labels[sorted_idx[3]], ColorClassifier.GREY)
        self.assertAlmostEqual(results.probas[sorted_idx[3]], 1.0)

        # check other information
        timing = results.timing
        self.assertEqual(timing.get_phases_hierarchy(), {"workflow.sldc": {
            "detect": {"load": None, "segment": None, "locate": None},
            "merge": None,
            "dispatch_classify": {"dispatch": None, "classify": None}
        }})

    def testWorkflowWithExcludedObjects(self):
        # generate circle image
        w, h = 300, 100
        image = np.zeros((h, w,), dtype="uint8")
        image = draw_circle(image, 25, (100, 40), 255)  # pi * 25 * 25 -> ~ 1963
        image = draw_circle(image, 35, (200, 60), 255)  # pi * 35 * 35 -> ~ 3858

        # build the workflow
        builder = SLDCWorkflowBuilder()
        builder.set_segmenter(CustomSegmenter())
        builder.add_classifier(MinAreaRule(2000), ColorClassifier(), "big")
        workflow = builder.get()

        # execute
        results = workflow.process(NumpyImage(image))

        # validate number of results
        count = len(results)
        self.assertEqual(count, 2)

        # sort polygons
        sorted_idx = sorted(range(count), key=lambda i: (results.polygons[i].centroid.y, results.polygons[i].centroid.x))

        # first shape (excluded)
        shape1 = results.polygons[sorted_idx[0]]
        self.assertLess(relative_error(shape1.area, np.pi * 25 * 25), 0.025)
        self.assertLess(relative_error(shape1.centroid.x, 100), 0.025)
        self.assertLess(relative_error(shape1.centroid.y, 40), 0.025)
        self.assertEqual(results.dispatches[sorted_idx[0]], None)
        self.assertEqual(results.labels[sorted_idx[0]], None)
        self.assertAlmostEqual(results.probas[sorted_idx[0]], 0.0)

        # second shape (include)
        shape2 = results.polygons[sorted_idx[1]]
        self.assertLess(relative_error(shape2.area, np.pi * 35 * 35), 0.025)
        self.assertLess(relative_error(shape2.centroid.x, 200), 0.025)
        self.assertLess(relative_error(shape2.centroid.y, 60), 0.025)
        self.assertEqual(results.dispatches[sorted_idx[1]], "big")
        self.assertEqual(results.labels[sorted_idx[1]], ColorClassifier.WHITE)
        self.assertAlmostEqual(results.probas[sorted_idx[1]], 1.0)

        # check other information
        timing = results.timing
        self.assertEqual(timing.get_phases_hierarchy(), {"workflow.sldc": {
            "detect": {"load": None, "segment": None, "locate": None},
            "merge": None,
            "dispatch_classify": {"dispatch": None, "classify": None}
        }})