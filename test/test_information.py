
from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal
from shapely.geometry import Polygon, Point

from sldc import WorkflowInformation, WorkflowTiming, merge_information, ChainInformation
from sldc.util import shape_array


class TestInformation(TestCase):
    def testNoAdditionalFields(self):
        polygons = [Point((0, 0))]
        labels = [1]
        timing = WorkflowTiming()
        info = WorkflowInformation(polygons, labels, timing=timing)

        self.assertEqual(len(info), 1)
        self.assertEqual(timing, info.timing)
        assert_array_equal(labels, info.labels)
        assert_array_equal(shape_array(polygons), info.polygons)
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
        assert_array_equal(shape_array(polygons), info.polygons)
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
        assert_array_equal(shape_array(polygons), info.polygons)
        assert_array_equal(dispatch, info.dispatches)
        self.assertSetEqual(set(info.fields), {info.DATA_FIELD_POLYGONS, info.DATA_FIELD_LABELS, "dispatches"})

        for i, object_info in enumerate(info):
            self.assertEqual(info[i], object_info)
            self.assertEqual(len(object_info), 3)
            self.assertEqual(object_info.polygon, polygons[i])
            self.assertEqual(object_info.label, labels[i])
            self.assertEqual(object_info.dispatch, dispatch[i])

        with self.assertRaises(TypeError):
            merge_information(labels, info0)

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


class TestChainInformation(TestCase):
    def testChainInformation(self):
        p1, p2 = Point(0, 0), Point(1, 0)
        polygons = [p1, p2]
        labels = [1, 2]
        timing = WorkflowTiming()
        w1 = WorkflowInformation(polygons=polygons, labels=labels, timing=timing)
        w2 = WorkflowInformation(polygons=polygons, labels=labels, timing=timing, others=(labels, "other"))
        workflows = [("w1", w1), ("w2", w2)]

        chain_info = ChainInformation()
        chain_info.append("w1", w1)
        chain_info.append("w2", w2)

        assert_array_equal(["w1", "w2"], chain_info.info_labels)

        self.assertEqual(len(chain_info), 2)
        self.assertEqual(chain_info["w1"], w1)
        self.assertEqual(chain_info["w2"], w2)
        self.assertEqual(chain_info.information("w1"), w1)
        self.assertEqual(chain_info.information("w2"), w2)

        for (e_label, e_workflow), (a_label, a_workflow) in zip(chain_info, workflows):
            self.assertEqual(e_label, a_label)
            self.assertEqual(e_workflow, a_workflow)

        assert_array_equal(chain_info.labels, np.array(labels + labels))
        assert_array_equal(chain_info.polygons, shape_array(polygons + polygons))
