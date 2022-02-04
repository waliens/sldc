from unittest import TestCase

import numpy as np
from shapely.geometry import Polygon

from sldc.util import emplace, batch_split, take, has_alpha_channel, alpha_rasterize


class TestUtil(TestCase):
    def test_emplace(self):
        src = list(range(1, 6))
        mapping = [5, 2, 3, 7, 0]
        dest = [0] * 10
        emplace(src, dest, mapping)
        self.assertListEqual([5, 0, 2, 3, 0, 1, 0, 4, 0, 0], dest)

    def test_batch_split(self):
        items = list(range(0, 10))
        splitted = batch_split(3, items)
        self.assertEqual(len(splitted), 3)
        self.assertListEqual([0, 1, 2, 3], splitted[0])
        self.assertListEqual([4, 5, 6], splitted[1])
        self.assertListEqual([7, 8, 9], splitted[2])

        items2 = list(range(0, 3))
        splitted2 = batch_split(3, items2)
        self.assertEqual(len(splitted2), 3)
        self.assertListEqual([0], splitted2[0])
        self.assertListEqual([1], splitted2[1])
        self.assertListEqual([2], splitted2[2])

    def test_take(self):
        src = list(range(0, 10))
        idx = [0, 4, 4, 3, 2, 7, 9]
        self.assertListEqual(idx, take(src, idx))

    def test_has_alpha_channel(self):
        fake_image = np.zeros((36, 36))
        fake_image1 = np.zeros((36, 36, 1))
        fake_image2 = np.zeros((36, 36, 2))
        fake_image3 = np.zeros((36, 36, 3))
        fake_image4 = np.zeros((36, 36, 4))

        self.assertFalse(has_alpha_channel(fake_image))
        self.assertFalse(has_alpha_channel(fake_image1))
        self.assertTrue(has_alpha_channel(fake_image2))
        self.assertFalse(has_alpha_channel(fake_image3))
        self.assertTrue(has_alpha_channel(fake_image4))

    def test_alpha_rasterize(self):
        fake_image = np.zeros((25, 25), dtype=np.int32)
        polygon = Polygon([(5, 5), (5, 20), (20, 20), (20, 5), (5, 5)])

        masked_image = alpha_rasterize(fake_image, polygon)
        self.assertTupleEqual((25, 25, 2), masked_image.shape)
        self.assertEqual(255, masked_image[12, 12, 1])
        self.assertEqual(0, masked_image[0, 0, 1])

        fake_image1 = np.zeros((25, 25, 1), dtype=np.int32)
        masked_image1 = alpha_rasterize(fake_image1, polygon)
        self.assertTupleEqual((25, 25, 2), masked_image1.shape)
        self.assertEqual(255, masked_image1[12, 12, 1])
        self.assertEqual(0, masked_image1[0, 0, 1])

        fake_image2 = np.zeros((25, 25, 2), dtype=np.int32)
        masked_image2 = alpha_rasterize(fake_image2, polygon)
        self.assertTupleEqual((25, 25, 2), masked_image2.shape)
        self.assertEqual(255, masked_image2[12, 12, 1])
        self.assertEqual(0, masked_image2[0, 0, 1])

        fake_image3 = np.zeros((25, 25, 3), dtype=np.int32)
        masked_image3 = alpha_rasterize(fake_image3, polygon)
        self.assertTupleEqual((25, 25, 4), masked_image3.shape)
        self.assertEqual(255, masked_image3[12, 12, 3])
        self.assertEqual(0, masked_image3[0, 0, 1])

        fake_image4 = np.zeros((25, 25, 4), dtype=np.int32)
        masked_image4 = alpha_rasterize(fake_image4, polygon)
        self.assertTupleEqual((25, 25, 4), masked_image4.shape)
        self.assertEqual(255, masked_image4[12, 12, 3])
        self.assertEqual(0, masked_image4[0, 0, 1])
