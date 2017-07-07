import numpy as np
from PIL.Image import fromarray
from PIL.ImageDraw import ImageDraw
from shapely.geometry import Point, Polygon, box

from sldc import Image


def mk_img(w, h, level=0):
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


def relative_error(val, ref):
    return np.abs(val - ref) / ref


def draw_multisquare(image, position, size, color_out=255, color_in=255):
    """Draw a square with color 'color_out' and given size at a given position (x, y)
        Then draw four square of size (size/5) with color 'color_in' at:
            1) coord: (y + (size / 5), x + (size / 5))
            2) coord: (y + (size / 5), x + (3 * size / 5))
            3) coord: (y + (3 * size / 5), x + (size / 5))
            4) coord: (y + (3 * size / 5), x + (3 * size / 5))
    """
    x, y = position
    small_size = size / 5
    image = draw_poly(image, box(x, y, x + size, y + size), color=color_out)
    square1 = box(x + small_size, y + small_size, x + 2 * small_size, y + 2 * small_size)
    square2 = box(x + 3 * small_size, y + small_size, x + 4 * small_size, y + 2 * small_size)
    square3 = box(x + small_size, y + 3 * small_size, x + 2 * small_size, y + 4 * small_size)
    square4 = box(x + 3 * small_size, y + 3 * small_size, x + 4 * small_size, y + 4 * small_size)
    squares = [square1, square2, square3, square4]
    for square in squares:
        image = draw_poly(image, square, color=color_in)
    return image


def draw_multicircle(image, position, diameter, color_out=255, color_in=255):
    """Draw a circle with color 'color_out' and given diameter at the given position (center of the circle will be at
    coordinates (c_x, c_y) = (position[0] + diameter / 2, position[1] + diameter / 2).
    Then, draw four circles with color 'color_in' of diameter diameter / 5. Those four circles are located at:
        1) coord: (c_x - a, c_y - a)
        2) coord: (c_x - a, c_y + a)
        3) coord: (c_x + a, c_y - a)
        4) coord: (c_x + a, c_y + a)
    where a = diameter * cos (45 deg) / 5
    """
    x, y = position
    radius = diameter / 2
    c_x, c_y = x + radius, y + radius
    center = (c_x, c_y)
    image = draw_circle(image, diameter / 2, center, color=color_out)
    center_offset = diameter * np.sqrt(2) / 10
    image = draw_circle(image, diameter / 10, (c_x - center_offset, c_y - center_offset), color=color_in)
    image = draw_circle(image, diameter / 10, (c_x - center_offset, c_y + center_offset), color=color_in)
    image = draw_circle(image, diameter / 10, (c_x + center_offset, c_y - center_offset), color=color_in)
    return draw_circle(image, diameter / 10, (c_x + center_offset, c_y + center_offset), color=color_in)
