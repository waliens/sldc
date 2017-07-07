from unittest import TestCase

import numpy as np
from shapely.affinity import translate
from shapely.geometry import Polygon

from sldc import BinaryLocator, SemanticLocator
from .util import mk_img, draw_circle, draw_poly, relative_error


class TestLocatorNothingToLocate(TestCase):
    def testLocator(self):
        image = mk_img(2000, 3000)
        locator = BinaryLocator()
        polygons = locator.locate(image)
        self.assertEqual(0, len(polygons), "No polygon found on black image")


class TestLocatorRectangle(TestCase):
    def testLocator(self):
        image = mk_img(2000, 3000)

        # draw a rectangle
        A = (25, 25)
        B = (25, 1500)
        C = (1250, 25)
        D = (1250, 1500)
        ABCD = Polygon([A, B, D, C, A])
        image = draw_poly(image, ABCD)

        # locate it
        locator = BinaryLocator()
        polygons = locator.locate(image)

        self.assertEqual(1, len(polygons), "One polygon found")
        self.assertTrue(ABCD.equals(polygons[0]), "Found polygon has the same shape")

        # test locate with an offset
        locator2 = BinaryLocator()
        polygons2 = locator2.locate(image, offset=(250, 200))
        self.assertEqual(1, len(polygons2), "One polygon found")
        self.assertTrue(translate(ABCD, 250, 200).equals(polygons2[0]), "Found translated polygon")


class TestLocatorCircleAndRectangle(TestCase):
    def testLocator(self):
        image = mk_img(2000, 3000)

        # draw a rectangle
        A = (25, 400)
        B = (25, 1500)
        C = (1250, 400)
        D = (1250, 1500)
        ABCD = Polygon([A, B, D, C, A])
        image = draw_poly(image, ABCD)
        image, circle = draw_circle(image, 400, (2500, 1500), return_circle=True)

        # test locator
        locator = BinaryLocator()
        polygons = locator.locate(image)

        self.assertEqual(2, len(polygons), "Two polygons found")
        self.assertTrue(ABCD.equals(polygons[1]), "Rectangle polygon is found")

        # use recall and false discovery rate to evaluate the error on the surface
        tpr = circle.difference(polygons[0]).area / circle.area
        fdr = polygons[0].difference(circle).area / polygons[0].area
        self.assertLessEqual(tpr, 0.005, "Recall is low for circle area")
        self.assertLessEqual(fdr, 0.005, "False discovery rate is low for circle area")


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
        polygons = locator.locate(image)

        self.assertEqual(2, len(polygons), "Two polygons found")
        self.assertTrue(ABCD.equals(polygons[0]), "Rectangle polygon is found")
        self.assertLessEqual(relative_error(polygons[1].area, np.pi * 40 * 40), 0.005)

    def testClassLocate(self):
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
        polygons = locator.class_locate(image)

        self.assertEqual(2, len(polygons))
        rectangle, rectangle_class = polygons[0]
        self.assertTrue(ABCD.equals(rectangle))
        self.assertTrue(rectangle_class, 1)
        circle, circle_class = polygons[1]
        self.assertLessEqual(relative_error(circle.area, np.pi * 40 * 40), 0.005)
        self.assertTrue(circle_class, 2)
