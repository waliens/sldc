import numpy as np
from unittest import TestCase

from PIL.ImageDraw import ImageDraw
from PIL.Image import fromarray
from shapely.geometry import Polygon
from shapely.affinity import translate

from sldc import Locator


def mk_gray_img(w,h,level=0):
    return np.ones((w, h)).astype("uint8") * level

def draw_poly(image, poly, fill=255, edge=255):
    pil = fromarray(image)
    drawer = ImageDraw(pil)
    drawer.polygon(poly.boundary.coords, fill=fill, outline=edge)
    return np.asarray(pil)

def draw_circle(image, center, radius, fill=255, edge=255):
    from shapely.geometry import Point
    polygon = Point(center[0], center[1])
    polygon = polygon.buffer(radius)
    return draw_poly(image, polygon, fill=fill, edge=edge), polygon


class TestLocatorNothingToLocate(TestCase):
    def testLocator(self):
        image = mk_gray_img(2000, 3000)
        locator = Locator()
        polygons = locator.locate(image)
        self.assertEqual(0, len(polygons), "No polygon found on black image")


class TestLocatorRectangle(TestCase):
    def testLocator(self):
        image = mk_gray_img(2000, 3000)

        # draw a rectangle
        A = (25, 25)
        B = (25, 1500)
        C = (1250, 25)
        D = (1250, 1500)
        ABCD = Polygon([A, B, D, C, A])
        image = draw_poly(image, ABCD)

        # locate it
        locator = Locator()
        polygons = locator.locate(image)

        self.assertEqual(1, len(polygons), "One polygon found")
        self.assertTrue(ABCD.equals(polygons[0]), "Found polygon has the same shape")

        # test locate with an offset
        locator2 = Locator()
        polygons2 = locator.locate(image, offset=(250, 200))
        self.assertEqual(1, len(polygons2), "One polygon found")
        self.assertTrue(translate(ABCD, 250, 200).equals(polygons2[0]), "Found translated polygon")


class TestLocatorCircleAndRectangle(TestCase):
    def testLocator(self):
        image = mk_gray_img(2000, 3000)

        # draw a rectangle
        A = (25, 400)
        B = (25, 1500)
        C = (1250, 400)
        D = (1250, 1500)
        ABCD = Polygon([A, B, D, C, A])
        image = draw_poly(image, ABCD)
        image, circle = draw_circle(image, (2500, 1500), 400)

        # test locator
        locator = Locator()
        polygons = locator.locate(image)

        self.assertEqual(2, len(polygons), "Two polygons found")
        self.assertTrue(ABCD.equals(polygons[1]), "Rectangle polygon is found")

        # compute percentage of the area of the circle that is not covered by the area of the found circle
        error = (circle.union(polygons[0]) - circle.intersection(polygons[0])).area / circle.area
        self.assertLessEqual(error, 0.005, "Found circle covers well the drawn circle")
