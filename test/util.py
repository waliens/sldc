import numpy as np
from PIL.Image import fromarray
from PIL.ImageDraw import ImageDraw
from shapely.geometry import Point, Polygon
from sldc import Image


def mk_gray_img(w, h, level=0):
    return np.ones((w, h)).astype("uint8") * level


def circularity(polygon):
    return 4 * np.pi * polygon.area / (polygon.length * polygon.length)


def draw_square(image, side, center, color):
    """Draw a square centered in 'center' and of which the side has 'side'"""
    top_left = (center[1] - side / 2, center[0] - side / 2)
    top_right = (center[1] + side / 2, center[0] - side / 2)
    bottom_left = (center[1] - side / 2, center[0] + side / 2)
    bottom_right = (center[1] + side / 2, center[0] + side / 2)
    p = Polygon([top_left, top_right, bottom_right, bottom_left, top_left])
    return draw_poly(image, p, color)


def draw_circle(image, radius, center, color=255, return_circle=False):
    """Draw a circle of radius 'radius' and centered in 'centered'"""
    circle_center = Point(*center)
    circle_polygon = circle_center.buffer(radius)
    image_out = draw_poly(image, circle_polygon, color)
    if return_circle:
        return image_out, circle_polygon
    else:
        return image_out


def draw_poly(image, polygon, color=255):
    """Draw a polygon in the given color at the given location"""
    pil_image = fromarray(image)
    validated_color = color
    draw = ImageDraw(pil_image)
    if len(image.shape) > 2 and image.shape[2] > 1:
        validated_color = tuple(color)
    draw.polygon(polygon.boundary.coords, fill=validated_color, outline=validated_color)
    return np.asarray(pil_image)


class NumpyImage(Image):
    def __init__(self, np_image):
        """An image represented as a numpy ndarray"""
        self._np_image = np_image

    @property
    def np_image(self):
        return self._np_image

    @property
    def channels(self):
        shape = self._np_image.shape
        return shape[2] if len(shape) == 3 else 1

    @property
    def width(self):
        return self._np_image.shape[1]

    @property
    def height(self):
        return self._np_image.shape[0]
