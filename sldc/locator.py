# -*- coding: utf-8 -*-

import cv2
from abc import ABCMeta, abstractmethod
from shapely.geometry import Polygon
from shapely.affinity import affine_transform

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"


class Locator(object):
    """
    A locator is an object that can extract polygons from a segmented image
    """

    def locate(self, segmented, offset=(0,0)):
        """Extract polygons for the foreground elements of the segmented image.
        Parameters
        ----------
        segmented: array-like, shape = [width, height]
            An array-like representation of a segmented image. Background pixels are represented by
            the value 0 ('black') while foreground ones are represented by the value 255 ('white').
            The type of the array values is 'uint8'.
        offset: tuple, optional (default: (0,0))
            An offset indicating the coordinates of the top-leftmost pixel of the segmented image in the
            original image.
        Returns
        -------
        polygons : array of shapely.geometry.Polygon objects
            An array containing the polygons extracted from the segmented image. The reference
            point (0,0) for the polygons coordinates is the upper-left corner of the initial image.
        """
        #borrowed from cytomine_utilities.objectfinder (v 1.0)
        #CV_RETR_EXTERNAL to only get external contours.
        contours, hierarchy = cv2.findContours(segmented.copy(),
                                               cv2.RETR_CCOMP,
                                               cv2.CHAIN_APPROX_SIMPLE)

        # Note: points are represented as (col, row)-tuples apparently
        transform = lambda x:x
        if offset is not None:
            row_off, col_off = offset
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
                    components.append(polygon)

                # check if there is another top contour
                if hierarchy[0][top_index][0] != -1:
                    top_index = hierarchy[0][top_index][0]
                else:
                    tops_remaining = False

        del contours
        del hierarchy
        return components
