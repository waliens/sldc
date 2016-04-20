# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pickle
from shapely.geometry import box, Polygon
from PIL import ImageDraw, Image

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


def clamp_polygon(polygon, minx, miny, maxx, maxy):
    b = box(minx,miny, maxx, maxy)
    return b.intersection(polygon)


def alpha_rasterize(image, polygon):
    """
    Rasterize the given polygon as an alpha mask of the given image. The
    polygon is assumed to be expressed in a traditional (i.e. lower left)
    coordinate system and will be translated so that the left most point
    will be on the first column and the top most on the first row.

    Parameters
    ----------
    polygon : :class:`shapely.Polygon`
        The polygon to rasterize

    Return
    ------
    np_image : numpy.ndarray
        The image (in numpy format) of the rasterization of the polygon.
        The image should have the same dimension as the bounding box of
        the polygon.
    """
    # Creating holder
    np_img = np.asarray(image)
    height, width, depth = np_img.shape
    # if there is already an alpha mask, replace it
    if depth == 4 or depth == 2:
        np_img = np_img[:, :, 0:depth-1]
    else:
        depth += 1
    np_results = np.zeros((height, width, depth), dtype=np.uint)
    np_results[:, :, 0:depth-1] = np_img
    # Rasterization
    polygon = clamp_polygon(polygon, 0, 0, width, height)
    alpha = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(alpha)
    seq_pts = polygon.boundary.coords
    draw.polygon(seq_pts, outline=0, fill=255)
    np_results[:, :, depth-1] = alpha
    return np_results


if __name__ == "__main__":
    random_image = np.random.randint(0, 255, size=(500, 400, 3))
    poly = Polygon([(50, 50), (150, 250), (600, 250), (300, 150)])
    alpha = alpha_rasterize(random_image, poly).astype("uint8")
    cv2.imwrite("image.png", alpha)