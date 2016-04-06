# -*- coding: utf-8 -*-
from unittest import TestCase

from sldc import DispatcherClassifier, PolygonClassifier, DispatchingRule

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class FakeClassifier(PolygonClassifier):
    def predict(self, image, polygon):
        return polygon


class FakeRule(DispatchingRule):
    def evaluate(self, image, polygon):
        return True


class FakeBetweenRule(DispatchingRule):
    def __init__(self, low, high):
        self._low = low
        self._high = high

    def evaluate(self, image, polygon):
        return self._low <= polygon < self._high


class FakeLTRule(DispatchingRule):
    def __init__(self, threshold):
        self._threshold = threshold

    def evaluate(self, image, polygon):
        return polygon < self._threshold


class FakeGERule(DispatchingRule):
    def __init__(self, threshold):
        self._threshold = threshold

    def evaluate(self, image, polygon):
        return polygon >= self._threshold


class TestDispatcherClassifier(TestCase):
    def testDispatcherClassifierOneRule(self):
        dispatcher_classifier = DispatcherClassifier([FakeRule()], [FakeClassifier()])
        range_list = list(range(0, 15))
        returned_list = dispatcher_classifier.dispatch_classify(None, range_list)
        self.assertEqual(range_list, returned_list)
        ranges_list = [list(range(0, 15)), list(range(0, 16))]
        returned_list_batch = dispatcher_classifier.dispatch_classify_batch(None, ranges_list)
        self.assertEqual(ranges_list, returned_list_batch)

    def testDispatcherClassifierThreeRule(self):
        rules = [FakeLTRule(5), FakeBetweenRule(5, 10), FakeGERule(10)]
        classifiers = [FakeClassifier(), FakeClassifier(), FakeClassifier()]
        dispatcher_classifier = DispatcherClassifier(rules, classifiers)
        range_list = list(range(0, 15))
        returned_list = dispatcher_classifier.dispatch_classify(None, range_list)
        self.assertEqual(range_list, returned_list)
        ranges_list = [list(range(0, 15)), list(range(0, 16))]
        returned_list_batch = dispatcher_classifier.dispatch_classify_batch(None, ranges_list)
        self.assertEqual(ranges_list, returned_list_batch)
