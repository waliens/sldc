# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"

from abc import ABCMeta, abstractmethod


class PolygonClassifier(object):
    """
    A classifier for polygons of an image
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, polygon):
        """Predict the class associated with the given polygon

        Parameters
        ----------
        polygon: shapely.geometry.Polygon
            The polygon of which the class must be predicted

        Returns
        -------
        prediction: int
            An integer code indicating the predicted class
        """
        pass
