# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
from shapely.geometry import box, Polygon

from sldc import DispatcherClassifier, PolygonClassifier, DispatchingRule, WorkflowTiming, CatchAllRule

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class AreaClassifier(PolygonClassifier):
    """Predict 1 if the polygon has an area greater than a value, 0 otherwise
    """
    def __init__(self, value):
        self._value = value

    def predict_batch(self, image, polygons):
        return [1 if p.area > self._value else 0 for p in polygons], np.full((len(polygons),), 1.0)


class QuadrilaterRule(DispatchingRule):
    """A rule that matches polygons that are quadrilaters
    """
    def evaluate_batch(self, image, polygons):
        return [len(polygon.boundary.coords) == 5 for polygon in polygons]


class NotQuadrilaterRule(QuadrilaterRule):
    """A rule that matches polygons which are not quadrilaters
    """
    def evaluate_batch(self, image, polygons):
        booleans = super(NotQuadrilaterRule, self).evaluate_batch(image, polygons)
        return [not b for b in booleans]


class TestDispatcherClassifier(TestCase):
    def testDispatcherClassifierOneRule(self):
        # create polygons to test
        box1 = box(0, 0, 100, 100)
        box2 = box(0, 0, 10, 10)

        dispatcher_classifier = DispatcherClassifier([CatchAllRule()], [AreaClassifier(500)])
        # simple dispatch test
        cls, probability, dispatch = dispatcher_classifier.dispatch_classify(None, box1, WorkflowTiming())
        self.assertEquals(1, cls)
        self.assertEquals(1.0, probability)
        self.assertEquals(0, dispatch)
        classes, probas, dispatches = dispatcher_classifier.dispatch_classify_batch(None, [box1, box2], WorkflowTiming())
        self.assertEquals(1, classes[0])
        self.assertEquals(0, classes[1])
        self.assertEquals(1.0, probas[0])
        self.assertEquals(1.0, probas[1])
        self.assertEquals(0, dispatches[0])
        self.assertEquals(0, dispatches[1])

    def testDispatcherClassifierThreeRule(self):
        # create polygons to test
        box1 = box(0, 0, 100, 100)
        box2 = box(0, 0, 10, 10)
        poly = Polygon([(0, 0), (0, 1000), (50, 1250), (1000, 1000), (1000, 0), (0, 0)])

        dispatcher_classifier = DispatcherClassifier([QuadrilaterRule(), NotQuadrilaterRule()],
                                                     [AreaClassifier(500), AreaClassifier(500)])

        # simple dispatch test
        cls, probability, dispatch = dispatcher_classifier.dispatch_classify(None, box1, WorkflowTiming())
        self.assertEquals(1, cls)
        self.assertEquals(1.0, probability)
        self.assertEquals(0, dispatch)

        # batch dispatch test
        classes, probas, dispatches = dispatcher_classifier.dispatch_classify_batch(None, [box1, box2, poly],
                                                                                    WorkflowTiming())
        self.assertEquals(1, classes[0])
        self.assertEquals(0, classes[1])
        self.assertEquals(1, classes[2])
        self.assertEquals(1.0, probas[0])
        self.assertEquals(1.0, probas[1])
        self.assertEquals(1.0, probas[2])
        self.assertEquals(0, dispatches[0])
        self.assertEquals(0, dispatches[1])
        self.assertEquals(1, dispatches[2])