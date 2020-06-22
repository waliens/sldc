# -*- coding: utf-8 -*-

import numpy as np
from affine import Affine
from rasterio.features import shapes
from shapely.geometry import shape, Polygon, MultiPolygon

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"
__contributors__ = ["Begon Jean-Michel <jm.begon@gmail.com>"]
__version__ = "0.1"


def clamp(x, l, h):
    return max(l, min(h, x))


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


def flatten_geoms(geoms):
    """Flatten (possibly nested) multipart geometry."""
    geometries = []
    for g in geoms:
        if hasattr(g, "geoms"):
            geometries.extend(flatten_geoms(g))
        else:
            geometries.append(g)
    return geometries


def representative_point(polygon, mask, label, offset=None):
    """ Extract a representative point with integer coordinates from the given polygon and the label image.
    Parameters
    ----------
    polygon: Polygon
        A polygon
    mask: ndarray
        The label mask from which the polygon was generated
    label: int
        The label associated with the polygon
    offset: tuple
        An (x, y) offset that was applied to polygon

    Returns
    -------
    point: tuple
        The representative point (x, y)
    """
    if offset is None:
        offset = (0, 0)
    rpoint = polygon.representative_point()
    h, w = mask.shape[:2]
    x = clamp(int(rpoint.x) - offset[0], 0, w - 1)
    y = clamp(int(rpoint.y) - offset[1], 0, h - 1)

    # check if start point is withing polygon
    if mask[y, y] == label:
        return x, y

    # circle around central pixel with at most 9 pixels radius
    direction = 1
    for i in range(1, 10):
        # -> x
        for j in range(0, i):
            x += direction
            if 0 <= x < w and mask[y, x] == label:
                return x, y

        # -> y
        for j in range(0, i):
            y += direction
            if 0 <= y < h and mask[y, x] == label:
                return x, y

        direction *= -1

    raise ValueError("could not find a representative point for pol")


def mask_to_objects_2d(mask, background=0, offset=None, flatten_collection=True):
    """Convert 2D (binary or label) mask to polygons. Generates borders fitting in the objects.

    Parameters
    ----------
    mask: ndarray
        2D mask array. Expected shape: (height, width).
    background: int
        Value used for encoding background pixels.
    offset: tuple (optional, default: None)
        (x, y) coordinate offset to apply to all the extracted polygons.
    flatten_collection: bool
        True for flattening geometry collections into individual geometries.

    Returns
    -------
    extracted: list of AnnotationSlice
        Each object slice represent an object from the image. Fields time and depth of AnnotationSlice are set to None.
    """
    if mask.ndim != 2:
        raise ValueError("Cannot handle image with ndim different from 2 ({} dim. given).".format(mask.ndim))
    if offset is None:
        offset = (0, 0)
    exclusion = np.logical_not(mask == background)
    affine = Affine(1, 0, offset[0], 0, 1, offset[1])
    slices = list()
    for gjson, label in shapes(mask.astype(np.int32), mask=exclusion, transform=affine):
        polygon = shape(gjson)

        # fixing polygon
        if not polygon.is_valid:  # attempt to fix
            polygon = fix_geometry(polygon)
        if not polygon.is_valid:  # could not be fixed
            continue

        if not hasattr(polygon, "geoms") or not flatten_collection:
            slices.append((polygon, int(label)))
        else:
            for curr in flatten_geoms(polygon.geoms):
                slices.append((curr, int(label)))
    return slices


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
