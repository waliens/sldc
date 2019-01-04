# -*- coding: utf-8 -*-
from warnings import warn

import cv2
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from shapely.validation import explain_validity
from skimage.measure import points_in_poly
from skimage.morphology import dilation, square, erosion

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"
__contributors__ = ["Begon Jean-Michel <jm.begon@gmail.com>"]
__version__ = "0.1"



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


def clean_mask(mask, background=0):
    """Remove ill-structured objects from a mask which prevent conversion to valid polygons.
    Parameters
    ----------
    mask: ndarray (2d)
        The mask to remove
    background: int
        Value of the background
    Returns
    -------
    mask: ndarray
        Cleaned mask
    Notes
    -----
    Example of ill-structured mask (caused by pixel 2)
    0 0 0 0 0
    0 1 1 0 0
    0 0 1 0 0
    0 0 0 2 0
    0 0 0 0 0
    """
    kernels = [
        np.array([[ 1, -1, -1], [-1,  1, -1], [-1, -1, -1]]),  # top left standalone pixel
        np.array([[-1, -1,  1], [-1,  1, -1], [-1, -1, -1]]),  # top right standalone pixel
        np.array([[-1, -1, -1], [-1,  1, -1], [ 1, -1, -1]]),  # bottom left standalone pixel
        np.array([[-1, -1, -1], [-1,  1, -1], [-1, -1,  1]])   # bottom right standalone pixel
    ]

    proc_masks = [cv2.morphologyEx(mask, cv2.MORPH_HITMISS, kernel).astype(np.bool) for kernel in kernels]

    for proc_mask in proc_masks:
        mask[proc_mask] = background
    return mask


def flatten_geoms(geoms):
    """Flatten (possibly nested) multipart geometry."""
    geometries = []
    for g in geoms:
        if hasattr(g, "geoms"):
            geometries.extend(flatten_geoms(g))
        else:
            geometries.append(g)
    return geometries


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
        transform = lambda p: affine_transform(p, [1, 0, 0, 1, col_off, row_off])
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
            if len(exterior) == 1:
                components.append(Point(exterior[0]))
            elif len(exterior) == 2:
                components.append(LineString(exterior))
            elif len(exterior) > 2:
                polygon = Polygon(exterior, interiors)
                polygon = transform(polygon)
                if polygon.is_valid:  # some polygons might be invalid
                    components.append(polygon)
                else:
                    fixed = fix_geometry(polygon)
                    if fixed.is_valid and not fixed.is_empty:
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
    if isinstance(polygon, Point):
        return int(polygon.x), int(polygon.y)
    if isinstance(polygon, LineString):
        return [int(c) for c in polygon.coords[0]]
    # this function works whether or not the boundary is inside or outside (one pixel around) the
    # object boundary in the mask
    exterior = polygon.exterior.coords
    for x, y in exterior:  # usually this function will return in one iteration
        neighbours = np.array(neighbour_pixels(int(x), int(y)))
        in_poly = np.array(points_in_poly(list(neighbours), exterior))
        if np.count_nonzero(in_poly) > 0:  # make sure at least one point is in the polygon
            return neighbours[in_poly][0]
    if len(exterior) == 4:  # fallback for three pixel polygons
        return [int(v) for v in exterior[0]]
    raise ValueError("No points could be found inside the polygon ({}) !".format(polygon.wkt))


def neighbour_pixels(x, y):
    """Get the neigbours pixel of x and y"""
    return [
        (x - 1, y - 1), (x, y - 1), (x + 1, y - 1),
        (x - 1, y    ), (x, y    ), (x + 1, y    ),
        (x - 1, y + 1), (x, y + 1), (x + 1, y + 1)
    ]


def mask_to_objects_2d(mask, background=0, offset=None):
    """Convert 2D (binary or label) mask to polygons. Generates borders fitting in the objects.
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
    extracted: list of AnnotationSlice
        Each object slice represent an object from the image. Fields time and depth of AnnotationSlice are set to None.
    """
    if mask.ndim != 2:
        raise ValueError("Cannot handle image with ndim different from 2 ({} dim. given).".format(mask.ndim))
    if offset is None:
        offset = (0, 0)
    # opencv only supports contour extraction for binary masks: clean mask and binarize
    mask_cpy = np.zeros(mask.shape, dtype=np.uint8)
    mask_cpy[mask != background] = 255
    # create artificial separation between adjacent touching each other + clean
    contours = dilation(mask, square(3)) - mask
    mask_cpy[np.logical_and(contours > 0, mask > 0)] = background
    mask_cpy = clean_mask(mask_cpy, background=background)
    # extract polygons and labels
    polygons = _locate(mask_cpy, offset=offset)
    objects = list()
    for polygon in polygons:
        # loop for handling multipart geometries
        for curr in flatten_geoms(polygon.geoms) if hasattr(polygon, "geoms") else [polygon]:
            x, y = get_polygon_inner_point(curr)
            objects.append((polygon, mask[y - offset[1], x - offset[0]]))
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
