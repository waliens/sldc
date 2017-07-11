# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from shapely.geometry import box, Polygon

from sldc import DispatcherClassifier, PolygonClassifier, DispatchingRule, WorkflowTiming, CatchAllRule
from sldc import RuleBasedDispatcher, Dispatcher

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class AreaClassifier(PolygonClassifier):
    """Predict 1 if the polygon has an area greater than a value, 0 otherwise
    """
    def __init__(self, value):
        self._value = value

    def predict(self, image, polygon):
        return 1 if polygon.area > self._value else 0, 1.0


class QuadrilaterRule(DispatchingRule):
    """A rule that matches polygons that are quadrilaters
    """
    def evaluate(self, image, polygon):
        return len(polygon.boundary.coords) == 5


class NotQuadrilaterRule(QuadrilaterRule):
    """A rule that matches polygons which are not quadrilaters
    """
    def evaluate(self, image, polygon):
        return not super(NotQuadrilaterRule, self).evaluate(image, polygon)


class CustomDispatcher(Dispatcher):
    """Dispatch 'BIG' if area is larger 1000, otherwise 'SMALL'"""

    def __init__(self):
        super(CustomDispatcher, self).__init__(["SMALL", "BIG"])

    def dispatch(self, image, polygon):
        return "BIG" if polygon.area > 1000 else "SMALL"


class TestDispatcher(TestCase):
    def testRuleBasedDispatcherNoLabels(self):
        # prepare data for test
        box1 = box(0, 0, 100, 100)
        box2 = box(0, 0, 10, 10)

        dispatcher = RuleBasedDispatcher([CatchAllRule()])
        self.assertEqual(dispatcher.dispatch(None, box1), 0)
        dispatch_batch = dispatcher.dispatch_batch(None, [box1, box2])
        assert_array_equal(dispatch_batch, [0, 0])
        labels, dispatch_map = dispatcher.dispatch_map(None, [box1, box2])
        assert_array_equal(labels, dispatch_batch)
        assert_array_equal(dispatch_batch, dispatch_map)

    def testRuleBasedDispatcher(self):
        # prepare data for test
        box1 = box(0, 0, 100, 100)
        box2 = box(0, 0, 10, 10)

        dispatcher = RuleBasedDispatcher([CatchAllRule()], ["catchall"])
        self.assertEqual(dispatcher.dispatch(None, box1), "catchall")
        dispatch_batch = dispatcher.dispatch_batch(None, [box1, box2])
        assert_array_equal(dispatch_batch, ["catchall", "catchall"])
        labels, dispatch_map = dispatcher.dispatch_map(None, [box1, box2])
        assert_array_equal(labels, dispatch_batch)
        assert_array_equal(dispatch_map, [0, 0])

    def testCustomDispatcher(self):
        # prepare data for test
        box1 = box(0, 0, 500, 500)
        box2 = box(0, 0, 10, 10)
        box3 = box(0, 0, 1000, 1000)

        dispatcher = CustomDispatcher()
        self.assertEqual(dispatcher.dispatch(None, box1), "BIG")
        dispatch_batch = dispatcher.dispatch_batch(None, [box1, box2, box3])
        assert_array_equal(dispatch_batch, ["BIG", "SMALL", "BIG"])
        labels, dispatch_map = dispatcher.dispatch_map(None, [box1, box2, box3])
        assert_array_equal(labels, dispatch_batch)
        assert_array_equal(dispatch_map, [1, 0, 1])


class TestDispatcherClassifier(TestCase):
    def testDispatcherClassifierOneRule(self):
        # create polygons to test
        box1 = box(0, 0, 100, 100)
        box2 = box(0, 0, 10, 10)

        dispatcher = RuleBasedDispatcher([CatchAllRule()])
        dispatcher_classifier = DispatcherClassifier(dispatcher, [AreaClassifier(500)])
        # simple dispatch test
        cls, probability, dispatch, _ = dispatcher_classifier.dispatch_classify(None, box1)
        self.assertEqual(1, cls)
        self.assertEqual(1.0, probability)
        self.assertEqual(0, dispatch)
        classes, probas, dispatches, _ = dispatcher_classifier.dispatch_classify_batch(None, [box1, box2])
        self.assertEqual(1, classes[0])
        self.assertEqual(0, classes[1])
        self.assertEqual(1.0, probas[0])
        self.assertEqual(1.0, probas[1])
        self.assertEqual(0, dispatches[0])
        self.assertEqual(0, dispatches[1])

    def testDispatcherClassifierThreeRule(self):
        # create polygons to test
        box1 = box(0, 0, 100, 100)
        box2 = box(0, 0, 10, 10)
        poly = Polygon([(0, 0), (0, 1000), (50, 1250), (1000, 1000), (1000, 0), (0, 0)])

        dispatcher = RuleBasedDispatcher([QuadrilaterRule(), NotQuadrilaterRule()])
        dispatcher_classifier = DispatcherClassifier(dispatcher, [AreaClassifier(500), AreaClassifier(500)])

        # simple dispatch test
        cls, probability, dispatch, _ = dispatcher_classifier.dispatch_classify(None, box1)
        self.assertEqual(1, cls)
        self.assertEqual(1.0, probability)
        self.assertEqual(0, dispatch)

        # batch dispatch test
        classes, probas, dispatches, _ = dispatcher_classifier.dispatch_classify_batch(None, [box1, box2, poly])
        self.assertEqual(1, classes[0])
        self.assertEqual(0, classes[1])
        self.assertEqual(1, classes[2])
        self.assertEqual(1.0, probas[0])
        self.assertEqual(1.0, probas[1])
        self.assertEqual(1.0, probas[2])
        self.assertEqual(0, dispatches[0])
        self.assertEqual(0, dispatches[1])
        self.assertEqual(1, dispatches[2])