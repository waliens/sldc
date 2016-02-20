# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"

from abc import ABCMeta, abstractmethod


class Locator(object):
    """
    An interface to be implemented by any locator object
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def locate(self, segmented):
        """Extract polygons for the foreground elements of the segmented image.
        Parameters
        ----------
        segmented: array-like, shape = [width, height]
            An array-like representation of a segmented image. Background pixels are represented by
            the value 0 ('black') while foreground ones are represented by the value 255 ('white').
            The type of the array values is 'uint8'.
        Returns
        -------
        polygons : array of shapely.geometry.Polygon objects
            An array containing the polygons extracted from the segmented image. The reference
            point (0,0) for the polygons coordinates is the upper-left corner if the image.
        """
        pass