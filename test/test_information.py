
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from shapely.geometry import Polygon, Point

from sldc import WorkflowInformation, WorkflowTiming, merge_information


class TestInformation(TestCase):
    def testNoAdditionalFields(self):
        polygons = [Point((0, 0))]
        labels = [1]
        timing = WorkflowTiming()
        info = WorkflowInformation(polygons, labels, timing=timing)

        self.assertEqual(len(info), 1)
        self.assertEqual(timing, info.timing)
        assert_array_equal(labels, info.labels)
        assert_array_equal(np.array(polygons, dtype=np.object), info.polygons)
        self.assertSetEqual(set(info.fields), {info.DATA_FIELD_POLYGONS, info.DATA_FIELD_LABELS})

        object_info = info[0]
        self.assertEqual(len(object_info), 2)
        self.assertEqual(object_info.polygon, polygons[0])
        self.assertEqual(object_info.label, labels[0])

        object_info_iter = next(iter(info))
        self.assertEqual(len(object_info_iter), 2)
        self.assertEqual(object_info_iter.polygon, polygons[0])
        self.assertEqual(object_info_iter.label, labels[0])

    def testAdditionalFields(self):
        polygons = [Point((0, 0)), Point((0, 1))]
        labels = [1, 2]
        dispatch = [5, 3]
        timing = WorkflowTiming()
        info = WorkflowInformation(polygons, labels, timing=timing, dispatches=(dispatch, "dispatch"))

        self.assertEqual(len(info), 2)
        self.assertEqual(timing, info.timing)
        assert_array_equal(labels, info.labels)
        assert_array_equal(np.array(polygons, dtype=np.object), info.polygons)
        assert_array_equal(dispatch, info.dispatches)
        self.assertSetEqual(set(info.fields), {info.DATA_FIELD_POLYGONS, info.DATA_FIELD_LABELS, "dispatches"})

        for i, object_info in enumerate(info):
            self.assertEqual(info[i], object_info)
            self.assertEqual(len(object_info), 3)
            self.assertEqual(object_info.polygon, polygons[i])
            self.assertEqual(object_info.label, labels[i])
            self.assertEqual(object_info.dispatch, dispatch[i])

    def testMerge(self):
        polygons = [Point((0, 0)), Point((0, 1))]
        labels = [1, 2]
        dispatch = [5, 3]
        timing = WorkflowTiming()
        info1 = WorkflowInformation(polygons[:1], labels[:1], timing=timing, dispatches=(dispatch[:1], "dispatch"))
        info0 = WorkflowInformation(polygons[1:], labels[1:], timing=timing, dispatches=(dispatch[1:], "dispatch"))

        info = merge_information(info1, info0)

        self.assertEqual(len(info), 2)
        self.assertEqual(timing, info.timing)
        assert_array_equal(labels, info.labels)
        assert_array_equal(np.array(polygons, dtype=np.object), info.polygons)
        assert_array_equal(dispatch, info.dispatches)
        self.assertSetEqual(set(info.fields), {info.DATA_FIELD_POLYGONS, info.DATA_FIELD_LABELS, "dispatches"})

        for i, object_info in enumerate(info):
            self.assertEqual(info[i], object_info)
            self.assertEqual(len(object_info), 3)
            self.assertEqual(object_info.polygon, polygons[i])
            self.assertEqual(object_info.label, labels[i])
            self.assertEqual(object_info.dispatch, dispatch[i])

    def testErrors(self):
        p1, p2 = Point(0, 0), Point(1, 0)
        timing = WorkflowTiming()
        with self.assertRaises(ValueError):
            WorkflowInformation([p1, p2], [1], timing)
        with self.assertRaises(ValueError):
            WorkflowInformation([p1, p2], [1, 2], timing, others=([2], "other"))
        with self.assertRaises(ValueError):
            WorkflowInformation([p1, p2], [1, 2], timing, __init__=([1, 2], "__init__s"))

        first = WorkflowInformation([p1, p2], [1, 2], timing, others=([2, 1], "other"))
        second = WorkflowInformation([p1, p2], [1, 2], timing)
        third = WorkflowInformation([p1, p2], [1, 2], timing, others=([2, 1], "oo"))
        with self.assertRaises(TypeError):
            first._is_compatible(dict())
        with self.assertRaises(ValueError):
            first._is_compatible(second)
        with self.assertRaises(ValueError):
            first._is_compatible(third)


class TestChainInformation(object):
    def testBase(self):
        pass