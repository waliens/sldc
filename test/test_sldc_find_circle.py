# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
from sldc import Segmenter, WorkflowBuilder, PolygonClassifier
from test.util import draw_circle, NumpyImage

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


class TestFullWorkflow(TestCase):
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
        self.assertEquals(len(workflow_info.polygons), 1)

        # Check circle
        polygon = workflow_info.polygons[0]
        self.assertEquals(self.relative_error(polygon.area, np.pi * 750 * 750) <= 0.005, True)
        self.assertEquals(self.relative_error(polygon.centroid.x, 1000) <= 0.005, True)
        self.assertEquals(self.relative_error(polygon.centroid.y, 1000) <= 0.005, True)
        self.assertEquals(workflow_info.classes, [1])
        self.assertEquals(workflow_info.probas, [1.0])
        self.assertEquals(workflow_info.dispatch, ["catchall"])

    def testDetectCircleParallel(self):
        """A test which executes a full workflow on image containing a white circle in the center of an black image in
        parallel
        """
        # generate circle image
        w, h = 2000, 2000
        image = np.zeros((w, h, 3), dtype="uint8")
        image = draw_circle(image, 750, (1000, 1000), [129, 129, 129])

        # build workflow
        builder = WorkflowBuilder(n_jobs=2)
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
        self.assertEquals(workflow_info.dispatch, ["catchall"])

    @staticmethod
    def relative_error(val, ref):
        return np.abs(val - ref) / ref
