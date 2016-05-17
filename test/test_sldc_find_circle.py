# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
from PIL.Image import fromarray
from PIL.ImageDraw import ImageDraw
from shapely.geometry import Point

from sldc import Segmenter, WorkflowBuilder, DefaultTileBuilder, PolygonClassifier, Image

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


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


class CircleSegmenter(Segmenter):
    def segment(self, image):
        """Segment a grey circle in black image"""
        segmented = (image[:, :, 0] > 50)
        return segmented.astype("uint8") * 255


class CircleClassifier(PolygonClassifier):
    def predict_batch(self, image, polygons):
        """A polygon classifier which always predict 1 with a probablility 1.0"""
        return [1] * len(polygons), [1.0] * len(polygons)


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


class TestFullWorkflow(TestCase):
    def testDetectCircle(self):
        """A test which executes a full workflow on image containing a white circle in the center of an black image
        """
        # generate circle image
        w, h = 2000, 2000
        image = np.zeros((w, h, 3), dtype="uint8")
        image = draw_circle(image, 750, (1000, 1000), [129, 129, 129])

        # build workflow
        builder = WorkflowBuilder()
        builder.set_segmenter(CircleSegmenter())
        builder.add_catchall_classifier(CircleClassifier())
        workflow = builder.get()

        # process image
        workflow_info = workflow.process(NumpyImage(image))

        # Check results
        self.assertEquals(len(workflow_info.polygons), 1)

        # Check circle
        polygon = workflow_info.polygons[0]
        self.assertEquals(self.relative_error(polygon.area, np.pi * 750 * 750) <= 0.005, True)
        self.assertEquals(self.relative_error(polygon.centroid.x, 1000) <= 0.005, True)
        self.assertEquals(self.relative_error(polygon.centroid.y, 1000) <= 0.005, True)
        self.assertEquals(workflow_info.classes, [1])
        self.assertEquals(workflow_info.probas, [1.0])
        self.assertEquals(workflow_info.dispatch, [0])

    @staticmethod
    def relative_error(val, ref):
        return np.abs(val - ref) / ref
