# -*- coding: utf-8 -*-
from unittest import TestCase

import cv2
import numpy as np
from PIL.Image import fromarray
from PIL.ImageDraw import ImageDraw
from shapely.geometry import Point, Polygon

from sldc import DispatchingRule, PolygonClassifier, Image, WorkflowBuilder, Segmenter

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def circularity(polygon):
    return 4 * np.pi * polygon.area / (polygon.length * polygon.length)


def draw_square(image, side, center, color):
    """Draw a square centered in 'center' and of which the side has 'side'"""
    top_left = (center[1] - side / 2, center[0] - side / 2)
    top_right = (center[1] + side / 2, center[0] - side / 2)
    bottom_left = (center[1] - side / 2, center[0] + side / 2)
    bottom_right = (center[1] + side / 2, center[0] + side / 2)
    p = Polygon([top_left, top_right, bottom_right, bottom_left, top_left])
    return draw(image, p, color)


def draw_circle(image, radius, center, color):
    """Draw a circle of radius 'radius' and centered in 'centered'"""
    circle_center = Point(*center)
    circle_polygon = circle_center.buffer(radius)
    return draw(image, circle_polygon, color)


def draw(image, polygon, color):
    """Draw a polygon in the given color at the given location"""
    pil_image = fromarray(image)
    validated_color = color
    draw = ImageDraw(pil_image)
    if len(image.shape) > 2 and image.shape[2] > 1:
        validated_color = tuple(color)
    draw.polygon(polygon.boundary.coords, fill=validated_color, outline=validated_color)
    return np.asarray(pil_image)


class NumpyImage(Image):
    def __init__(self, np_image):
        """An image represented as a numpy ndarray"""
        self._np_image = np_image

    @property
    def np_image(self):
        return self._np_image

    @property
    def channels(self):
        shape = self._np_image.shape
        return shape[2] if len(shape) == 3 else 1

    @property
    def width(self):
        return self._np_image.shape[1]

    @property
    def height(self):
        return self._np_image.shape[0]


class CircleRule(DispatchingRule):
    """A rule which matches circle polygons"""
    def evaluate_batch(self, image, polygons):
        return [circularity(p) > 0.85 for p in polygons]


class SquareRule(DispatchingRule):
    """A rule that matches square polygons"""
    def evaluate_batch(self, image, polygons):
        return [circularity(p) <= 0.85 for p in polygons]


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
        builder = WorkflowBuilder(n_jobs=1)
        builder.set_segmenter(CustomSegementer())
        builder.add_classifier(CircleRule(), ColorClassifier())
        builder.add_classifier(SquareRule(), ColorClassifier())
        workflow = builder.get()

        # Execute
        results = workflow.process(NumpyImage(image))

        image = np.zeros((h, w), dtype="uint8")
        color = 50
        for p in results.polygons:
            image = draw(image, p, color)
            color += 35
        cv2.imwrite("image.png", image)
        self.assertEquals(len(results.polygons), 5)

        # first square
        square1 = results.polygons[0]
        self.assertEquals(self.relative_error(square1.area, 200 * 200) < 0.005, True)
        self.assertEquals(self.relative_error(square1.centroid.x, 1000) < 0.005, True)
        self.assertEquals(self.relative_error(square1.centroid.y, 1000) < 0.005, True)
        self.assertEquals(results.dispatch[0], 1)  # square
        self.assertEquals(results.classes[0], ColorClassifier.WHITE)  # white
        self.assertAlmostEquals(results.probas[0], 1.0)

        # first circle
        circle1 = results.polygons[1]
        self.assertEquals(self.relative_error(circle1.area, np.pi * 100 * 100) < 0.005, True)
        self.assertEquals(self.relative_error(circle1.centroid.x, 3000) < 0.005, True)
        self.assertEquals(self.relative_error(circle1.centroid.y, 1000) < 0.005, True)
        self.assertEquals(results.dispatch[1], 0)  # circle
        self.assertEquals(results.classes[1], ColorClassifier.GREY)  # grey
        self.assertAlmostEquals(results.probas[1], 1.0)

        # second square (centered)
        square2 = results.polygons[2]
        self.assertEquals(self.relative_error(square2.area, 300 * 300) < 0.005, True)
        self.assertEquals(self.relative_error(square2.centroid.x, 2000) < 0.005, True)
        self.assertEquals(self.relative_error(square2.centroid.y, 2000) < 0.005, True)
        self.assertEquals(results.dispatch[2], 1)  # square
        self.assertEquals(results.classes[2], ColorClassifier.WHITE)  # white
        self.assertAlmostEquals(results.probas[2], 1.0)

        # second circle
        circle2 = results.polygons[3]
        self.assertEquals(self.relative_error(circle2.area, np.pi * 100 * 100) < 0.005, True)
        self.assertEquals(self.relative_error(circle2.centroid.x, 1000) < 0.005, True)
        self.assertEquals(self.relative_error(circle2.centroid.y, 3000) < 0.005, True)
        self.assertEquals(results.dispatch[3], 0)  # circle
        self.assertEquals(results.classes[3], ColorClassifier.WHITE)  # grey
        self.assertAlmostEquals(results.probas[3], 1.0)

        # third square
        square3 = results.polygons[4]
        self.assertEquals(self.relative_error(square3.area, 200 * 200) < 0.005, True)
        self.assertEquals(self.relative_error(square3.centroid.x, 3000) < 0.005, True)
        self.assertEquals(self.relative_error(square3.centroid.y, 3000) < 0.005, True)
        self.assertEquals(results.dispatch[4], 1)  # square
        self.assertEquals(results.classes[4], ColorClassifier.GREY)  # white
        self.assertAlmostEquals(results.probas[4], 1.0)

    @staticmethod
    def relative_error(val, ref):
        return np.abs(val - ref) / ref
