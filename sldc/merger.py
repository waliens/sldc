# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"


class Merger(object):

    def __init__(self, boundary_thickness):
        self._boundary_thickness = boundary_thickness

    def merge(self, polygons_tiles):
        return []