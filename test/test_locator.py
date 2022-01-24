from unittest import TestCase

import numpy as np
from shapely.affinity import translate, affine_transform
from shapely.geometry import Polygon

from sldc import BinaryLocator, SemanticLocator
from test.util import mk_img, draw_circle, draw_poly, relative_error


class TestLocatorNothingToLocate(TestCase):
    def testLocator(self):
        image = mk_img(400, 600)
        locator = BinaryLocator()
        located = locator.locate(image)
        self.assertEqual(0, len(located), "No polygon found on black image")


class TestLocatorRectangle(TestCase):
    def testLocator(self):
        image = mk_img(400, 600)

        # draw a rectangle
        A = (5, 5)
        B = (5, 300)
        C = (250, 5)
        D = (250, 300)
        ABCD = Polygon([A, B, D, C, A])
        image = draw_poly(image, ABCD)

        # locate it
        locator = BinaryLocator()
        located = locator.locate(image)
        polygons, labels = zip(*located)

        self.assertEqual(1, len(located), "One polygon found")
        expected_polygon = Polygon([A, (5, 301), (251, 301), (251, 5), A])
        self.assertTrue(expected_polygon.equals(polygons[0]), "Found polygon has the same shape")

        # test locate with an offset
        locator2 = BinaryLocator()
        located2 = locator2.locate(image, offset=(50, 40))
        polygons2, labels2 = zip(*located2)
        self.assertEqual(1, len(located2), "One polygon found")
        self.assertTrue(translate(expected_polygon, 50, 40).equals(polygons2[0]), "Found translated polygon")


class TestLocatorCircleAndRectangle(TestCase):
    def testLocator(self):
        image = mk_img(400, 600)

        # draw a rectangle
        A = (5, 80)
        B = (5, 300)
        C = (250, 80)
        D = (250, 300)
        ABCD = Polygon([A, B, D, C, A])
        image = draw_poly(image, ABCD)
        image, circle = draw_circle(image, 85, (500, 300), return_circle=True)

        # test locator
        locator = BinaryLocator()
        located = locator.locate(image)
        polygons, labels = zip(*located)

        self.assertEqual(2, len(polygons), "Two polygons found")
        expected_polygon = Polygon([A, (5, 301), (251, 301), (251, 80), A])
        self.assertTrue(expected_polygon.equals(polygons[0]), "Rectangle polygon is found")
        self.assertLessEqual(relative_error(polygons[1].area, np.pi * 85 * 86), 0.025)


class TestSemanticLocatorCircleAndRectangle(TestCase):
    def testLocate(self):
        image = mk_img(200, 300)

        # draw a rectangle
        A = (3, 40)
        B = (3, 150)
        C = (125, 40)
        D = (125, 150)
        ABCD = Polygon([A, B, D, C, A])
        image = draw_poly(image, ABCD, color=1)
        image = draw_circle(image, 40, (250, 150), color=2)

        # test locator
        locator = SemanticLocator(background=0)
        located = locator.locate(image)
        located = sorted(located, key=lambda o: (o[0].centroid.x, o[0].centroid.y))
        polygons, labels = zip(*located)

        self.assertEqual(2, len(polygons), "Two polygons found")
        expected_polygon = Polygon([A, (3, 151), (126, 151), (126, 40), A])
        self.assertTrue(expected_polygon.equals(polygons[0]), "Rectangle polygon is found")
        self.assertLessEqual(relative_error(polygons[1].area, np.pi * 40 * 41), 0.025)

