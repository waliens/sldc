# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"

from abc import ABCMeta, abstractproperty

from sldc.tile import TilesIterator


class Image(object):
    """
    Abstract representation of an image
    """
    __metaclass__ = ABCMeta

    @abstractproperty
    def height(self):
        """Return the height of the image
        Returns
        -------
        height: int
            Height of the image
        """
        pass

    @abstractproperty
    def width(self):
        """Return the width of the image
        Returns
        -------
        width: int
            Width of the image
        """
        pass

    @abstractproperty
    def channels(self):
        """Return the number of channels in the image
        Returns
        -------
        width: int
            Width of the image
        """
        pass

    def tile_iterator(self, builder, max_width=1024, max_height=1024):
        """Build and return a tile iterator that iterates over the image

        Parameters
        ----------
        builder: TileBuilder
            The builder to user for actually constructing the tiles while iterating over the image
        max_width: int, optional (default: 1024)
            The maximum width of the tiles to build
        max_height: int, optional (default: 1024)
            The maximum height of the tiles to build
        """
        builder.image = self
        return TilesIterator(builder, max_width=max_width, max_height=max_height)
