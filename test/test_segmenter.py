from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from sldc import SemanticSegmenter, ProbabilisticSegmenter
from test import draw_square


class BasicSegmenter(SemanticSegmenter):
    def segment(self, image):
        return image


class BasicProbalisticSegmenter(ProbabilisticSegmenter):
    def segment_proba(self, image):
        height, width = image.shape
        values = np.sort(np.unique(image))
        n_values = values.shape[0]
        probas = np.zeros((height, width, n_values), dtype=np.float64)
        for v in values:
            probas[:, :, v][image == v] = 1.0
        return probas


class TestSegmenter(TestCase):
    def testBasicSegmenter(self):
        height, width = 10, 10
        image = np.zeros((height, width), dtype=np.uint8)
        image = draw_square(image, 3, (2, 2), 1)
        image = draw_square(image, 3, (6, 6), 2)

        true_seg = np.full((height, width), fill_value=125, dtype=np.uint8)
        true_seg = draw_square(true_seg, 3, (2, 2), 128)
        true_seg = draw_square(true_seg, 3, (6, 6), 129)

        segmenter = BasicSegmenter(classes=[125, 128, 129])

        prod_seg = segmenter.segment(image)
        prod_true_seg = segmenter.true_segment(image)

        assert_array_equal(image, prod_seg)
        assert_array_equal(true_seg, prod_true_seg)

        no_class_segmenter = BasicSegmenter()
        with self.assertRaises(ValueError):
            no_class_segmenter.true_segment(image)

    def testClasses(self):
        segmenter1 = BasicSegmenter()
        self.assertIsNone(segmenter1.classes)
        self.assertEqual(segmenter1.n_classes, -1)
        segmenter2 = BasicSegmenter(classes=[1, 2])
        self.assertEqual(set(segmenter2.classes), {1, 2})
        self.assertEqual(segmenter2.n_classes, 2)

    def testProbabilisticSegmenter(self):
        height, width = 10, 10
        image = np.zeros((height, width), dtype=np.uint8)
        image = draw_square(image, 3, (2, 2), 1)
        image = draw_square(image, 3, (6, 6), 2)

        true_seg = np.full((height, width), fill_value=125, dtype=np.uint8)
        true_seg = draw_square(true_seg, 3, (2, 2), 128)
        true_seg = draw_square(true_seg, 3, (6, 6), 129)

        segmenter = BasicProbalisticSegmenter(classes=[125, 128, 129])

        prod_seg = segmenter.segment(image)
        prod_true_seg = segmenter.true_segment(image)

        assert_array_equal(image, prod_seg)
        assert_array_equal(true_seg, prod_true_seg)
