# -*- coding: utf-8 -*-
from functools import partial
from warnings import warn

import cv2
import numpy as np
from shapely.affinity import affine_transform as aff_transfo
from shapely.geometry import Polygon, MultiPolygon
from shapely.validation import explain_validity
from skimage.measure import points_in_poly

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"
__contributors__ = ["Begon Jean-Michel <jm.begon@gmail.com>"]
__version__ = "0.1"


def affine_transform(xx_coef=1, xy_coef=0, yx_coef=0, yy_coef=1,
                     delta_x=0, delta_y=0):
    """
    Represents a 2D affine transformation:
    x' = xx_coef * x + xy_coef * y + delta_x
    y' = yx_coef * x + yy_coef * y + delta_y
    Constructor parameters
    ----------------------
    xx_coef: float (default: 1)
        The x from x coefficient
    xy_coef: float (default: 0)
        The x from y coefficient
    yx_coef: float (default: 0)
        The y from x coefficient
    yy_coef: float (default: 1)
        The y from y coefficient
    delta_x: float (default: 0)
        The translation over x-axis
    delta_y: float (default: 0)
        The translation over y-axis
    Return
    ------
    affine_transformer : callable: shapely.Geometry => shapely.Geometry
        The function representing the 2D affine transformation
    """
    return partial(aff_transfo, matrix=[xx_coef, xy_coef, yx_coef, yy_coef,
                                        delta_x, delta_y])


def identity(x):
    """Identity function
    Parameters
    ----------
    x: T
        The object to return
    Returns
    -------
    x: T
        The passed object
    """
    return x


def geom_as_list(geometry):
    """Return the list of sub-polygon a polygon is made up of"""
    if geometry.geom_type == "Polygon":
        return [geometry]
    elif geometry.geom_type == "MultiPolygon":
        return geometry.geoms


def linear_ring_is_valid(ring):
    points = set([(x, y) for x, y in ring.coords])
    return len(points) >= 3


def fix_geometry(geometry):
    """Attempts to fix an invalid geometry (from https://goo.gl/nfivMh)"""
    try:
        return geometry.buffer(0)
    except ValueError:
        pass

    polygons = geom_as_list(geometry)

    fixed_polygons = list()
    for i, polygon in enumerate(polygons):
        if not linear_ring_is_valid(polygon.exterior):
            continue

        interiors = []
        for ring in polygon.interiors:
            if linear_ring_is_valid(ring):
                interiors.append(ring)

        fixed_polygon = Polygon(polygon.exterior, interiors)

        try:
            fixed_polygon = fixed_polygon.buffer(0)
        except ValueError:
            continue

        fixed_polygons.extend(geom_as_list(fixed_polygon))

    if len(fixed_polygons) > 0:
        return MultiPolygon(fixed_polygons)
    else:
        return None


def _locate(segmented, offset=None):
    """Inspired from: https://goo.gl/HYPrR1"""
    # CV_RETR_EXTERNAL to only get external contours.
    _, contours, hierarchy = cv2.findContours(segmented.copy(),
                                              cv2.RETR_CCOMP,
                                              cv2.CHAIN_APPROX_SIMPLE)

    # Note: points are represented as (col, row)-tuples apparently
    transform = identity
    if offset is not None:
        col_off, row_off = offset
        transform = affine_transform(delta_x=col_off, delta_y=row_off)
    components = []
    if len(contours) > 0:
        top_index = 0
        tops_remaining = True
        while tops_remaining:
            exterior = contours[top_index][:, 0, :].tolist()

            interiors = []
            # check if there are childs and process if necessary
            if hierarchy[0][top_index][2] != -1:
                sub_index = hierarchy[0][top_index][2]
                subs_remaining = True
                while subs_remaining:
                    interiors.append(contours[sub_index][:, 0, :].tolist())

                    # check if there is another sub contour
                    if hierarchy[0][sub_index][0] != -1:
                        sub_index = hierarchy[0][sub_index][0]
                    else:
                        subs_remaining = False

            # add component tuple to components only if exterior is a polygon
            if len(exterior) > 3:
                polygon = Polygon(exterior, interiors)
                polygon = transform(polygon)
                if polygon.is_valid:  # some polygons might be invalid
                    components.append(polygon)
                else:
                    fixed = fix_geometry(polygon)
                    if fixed.is_valid:
                        components.append(fixed)
                    else:
                        warn("Attempted to fix invalidity '{}' in polygon but failed... "
                             "Output polygon still invalid '{}'".format(explain_validity(polygon),
                                                                        explain_validity(fixed)))

            # check if there is another top contour
            if hierarchy[0][top_index][0] != -1:
                top_index = hierarchy[0][top_index][0]
            else:
                tops_remaining = False

    del contours
    del hierarchy
    return components


def get_polygon_inner_point(polygon):
    """
    Algorithm:
        1) Take a point on the exterior boundary
        2) Find an adjacent point (with digitized coordinates) that lies in the polygon
        3) Return the coordinates of this point
    Parameters
    ----------
    polygon: Polygon
        The polygon
    Returns
    -------
    point: tuple
        (x, y) coordinates for the found points. x and y are integers.
    """
    exterior = polygon.exterior.coords
    for x, y in exterior:  # usually this function will return in one iteration
        neighbours = np.array(neighbour_pixels(int(x), int(y)))
        in_poly = np.array(points_in_poly(list(neighbours), exterior))
        if np.count_nonzero(in_poly) > 0:  # make sure at least one point is in the polygon
            return neighbours[in_poly][0]
    raise ValueError("No points could be found inside the polygon ({}) !".format(polygon.wkt))


def neighbour_pixels(x, y):
    """Get the neigbours pixel of x and y"""
    return [
        (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
        (x - 1, y    ), (x, y    ), (x + 1, y    ),
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)
    ]


def mask_to_objects_2d(mask, background=0, offset=None):
    """Convert 2D (binary or label) mask to polygons.
    Parameters
    ----------
    mask: ndarray
        2D mask array. Expected shape: (height, width).
    background: int
        Value used for encoding background pixels.
    offset: tuple (optional, default: None)
        (x, y) coordinate offset to apply to all the extracted polygons.
    Returns
    -------
    extracted: list of ObjectSlice
        Each object slice represent an object from the image. Fields time and depth of ObjectSlice are set to None.
    Notes
    -----
    Adjacent but distinct polygons must be separated by at least one line of background (e.g. value 0) pixels.
    The mask array is not modified by the function.
    """
    if mask.ndim != 2:
        raise ValueError("Cannot handle image with ndim different from 2 ({} dim. given).".format(mask.ndim))
    # opencv only supports contour extraction for binary masks
    mask_cpy = np.zeros(mask.shape, dtype=np.uint8)
    mask_cpy[mask != background] = 255
    # extract polygons and labels
    polygons = _locate(mask_cpy, offset=offset)
    objects = list()
    for polygon in polygons:
        x, y = get_polygon_inner_point(polygon)
        if offset is not None:
            x, y = x - offset[0], y - offset[1]
        objects.append((polygon, mask[y, x]))
    return objects


class SemanticLocator(object):
    """Interface to be implemented by Locator objects"""
    def __init__(self, background=-1):
        """
        Parameters
        ----------
        background: int (default: -1)
            The value corresponding to the background pixel label. Polygons won't be extracted for this class.
            If the value is not a class label, all the classes contained in the mask will be extracted.
        """
        self._background = background

    def locate(self, mask, offset=None):
        """Extract polygons representing the foreground elements of the segmented image.
        
        Parameters
        ----------
        mask: ndarray (shape: (width, height))
            A NumPy representation of a segmented image. Each pixel is associated with its (integer) class label. 
            
        offset: (int, int) (optional, default: (0,0))
            An offset indicating the coordinates of the top-leftmost pixel of the segmented image in the
            original image.
            
        Returns
        -------
        polygons : iterable (subtype: (shapely.geometry.Polygon, int))
            An iterable containing the polygons extracted from the segmented image as well as the pixel value
            which resulted in generating this polygon. The reference point (0,0) for the polygons coordinates is 
            the upper-left corner of the initial image.
        """
        return mask_to_objects_2d(mask, background=self._background, offset=offset)


class BinaryLocator(SemanticLocator):
    """Locator that assigns the 0-label to background and 255 to foreground
    Mostly there for backward compatibility.
    """
    def __init__(self):
        super(BinaryLocator, self).__init__(background=0)
