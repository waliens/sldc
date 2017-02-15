# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
from sldc import DispatchingRule, PolygonClassifier, WorkflowBuilder, Segmenter
from .util import circularity, draw_circle, draw_square, draw_poly, NumpyImage, relative_error

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class CircleSegmenter(Segmenter):
    def segment(self, image):
        """Segment a grey circle in black image"""
        segmented = (image[:, :, 0] > 50)
        return segmented.astype("uint8") * 255


class CircleClassifier(PolygonClassifier):
    def predict_batch(self, image, polygons):
        """A polygon classifier which always predict 1 with a probablility 1.0"""
        return [1] * len(polygons), [1.0] * len(polygons)


class CircleRule(DispatchingRule):
    """A rule which matches circle polygons"""
    def evaluate(self, image, polygon):
        return circularity(polygon) > 0.85


class SquareRule(DispatchingRule):
    """A rule that matches square polygons"""
    def evaluate(self, image, polygon):
        return circularity(polygon) <= 0.8


class ColorClassifier(PolygonClassifier):
    """A classifier which returns the color class of the center point of the polygon"""
    GREY = 0
    WHITE = 1

    def predict_batch(self, image, polygons):
        classes = []
        for polygon in polygons:
            window = image.window_from_polygon(polygon)
            sub_image = window.np_image
            pxl = sub_image[int(polygon.centroid.y) - window.offset_y][int(polygon.centroid.x) - window.offset_x]
            if pxl == 255:
                classes.append(ColorClassifier.WHITE)
            elif 0 < pxl < 255:
                classes.append(ColorClassifier.GREY)
            else:
                classes.append(None)
        return classes, [1.0] * len(polygons)


class CustomSegementer(Segmenter):
    """Every non black pixel are in object of interests"""
    def segment(self, image):
        return (image > 0).astype("uint8")


class TestFullWorkflow(TestCase):
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
        w, h = 4000, 4000
        image = np.zeros((w, h,), dtype="uint8")
        image = draw_circle(image, 100, (1000, 3000), 255)
        image = draw_circle(image, 100, (3000, 1000), 127)
        image = draw_square(image, 200, (1000, 1000), 255)
        image = draw_square(image, 200, (3000, 3000), 127)
        image = draw_square(image, 300, (2000, 2000), 255)

        # Build the workflow
        builder = WorkflowBuilder()
        builder.set_segmenter(CustomSegementer())
        builder.add_classifier(CircleRule(), ColorClassifier(), dispatching_label="circle")
        builder.add_classifier(SquareRule(), ColorClassifier())
        workflow = builder.get()

        # Execute
        results = workflow.process(NumpyImage(image))

        image = np.zeros((h, w), dtype="uint8")
        color = 50
        for p in results.polygons:
            image = draw_poly(image, p, color)
            color += 35
        # cv2.imwrite("image.png", image)
        self.assertEqual(len(results.polygons), 5)

        # first square
        square1 = results.polygons[0]
        self.assertEqual(relative_error(square1.area, 200 * 200) < 0.005, True)
        self.assertEqual(relative_error(square1.centroid.x, 1000) < 0.005, True)
        self.assertEqual(relative_error(square1.centroid.y, 1000) < 0.005, True)
        self.assertEqual(results.dispatch[0], 1)  # square
        self.assertEqual(results.classes[0], ColorClassifier.WHITE)  # white
        self.assertAlmostEquals(results.probas[0], 1.0)

        # first circle
        circle1 = results.polygons[1]
        self.assertEqual(relative_error(circle1.area, np.pi * 100 * 100) < 0.005, True)
        self.assertEqual(relative_error(circle1.centroid.x, 3000) < 0.005, True)
        self.assertEqual(relative_error(circle1.centroid.y, 1000) < 0.005, True)
        self.assertEqual(results.dispatch[1], "circle")  # circle
        self.assertEqual(results.classes[1], ColorClassifier.GREY)  # grey
        self.assertAlmostEquals(results.probas[1], 1.0)

        # second square (centered)
        square2 = results.polygons[2]
        self.assertEqual(relative_error(square2.area, 300 * 300) < 0.005, True)
        self.assertEqual(relative_error(square2.centroid.x, 2000) < 0.005, True)
        self.assertEqual(relative_error(square2.centroid.y, 2000) < 0.005, True)
        self.assertEqual(results.dispatch[2], 1)  # square
        self.assertEqual(results.classes[2], ColorClassifier.WHITE)  # white
        self.assertAlmostEquals(results.probas[2], 1.0)

        # second circle
        circle2 = results.polygons[3]
        self.assertEqual(relative_error(circle2.area, np.pi * 100 * 100) < 0.005, True)
        self.assertEqual(relative_error(circle2.centroid.x, 1000) < 0.005, True)
        self.assertEqual(relative_error(circle2.centroid.y, 3000) < 0.005, True)
        self.assertEqual(results.dispatch[3], "circle")  # circle
        self.assertEqual(results.classes[3], ColorClassifier.WHITE)  # grey
        self.assertAlmostEquals(results.probas[3], 1.0)

        # third square
        square3 = results.polygons[4]
        self.assertEqual(relative_error(square3.area, 200 * 200) < 0.005, True)
        self.assertEqual(relative_error(square3.centroid.x, 3000) < 0.005, True)
        self.assertEqual(relative_error(square3.centroid.y, 3000) < 0.005, True)
        self.assertEqual(results.dispatch[4], 1)  # square
        self.assertEqual(results.classes[4], ColorClassifier.GREY)  # white
        self.assertAlmostEquals(results.probas[4], 1.0)

    def testDetectCircle(self):
        """A test which executes a full workflow on image containing a white circle in the center of an black image
        """
        # generate circle image
        w, h = 2000, 2000
        image = np.zeros((w, h, 3), dtype="uint8")
        image = draw_circle(image, 750, (1000, 1000), color=[129, 129, 129])

        # build workflow
        builder = WorkflowBuilder()
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
        self.assertEqual(workflow_info.classes, [1])
        self.assertEqual(workflow_info.probas, [1.0])
        self.assertEqual(workflow_info.dispatch, ["catchall"])

    def testDetectCircleParallel(self):
        """A test which executes a full workflow on image containing a white circle in the center of an black image in
        parallel
        """
        # generate circle image
        w, h = 2000, 2000
        image = np.zeros((w, h, 3), dtype="uint8")
        image = draw_circle(image, 750, (1000, 1000), [129, 129, 129])

        # build workflow
        builder = WorkflowBuilder()
        builder.set_n_jobs(2)
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
        self.assertEqual(workflow_info.classes, [1])
        self.assertEqual(workflow_info.probas, [1.0])
        self.assertEqual(workflow_info.dispatch, ["catchall"])

