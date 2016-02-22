# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractproperty
from sldc.tile import TileTopologyIterator, TileTopology

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"


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

    def tile(self, tile_builder, offset, max_width, max_height):
        """Extract a tile from the image

        Parameters
        ----------
        tile_builder: TileBuilder
            A tile builder for constructing the Tile object
        offset: (int, int)
            The (x, y) coordinates of the pixel at the origin point of the tile in the parent image
        max_width:
            The maximum width of the tile
        max_height:
            The maximum height of the tile

        Returns
        -------
        tile: Tile
            The extracted tile

        Raises
        ------
        IndexError: if the offset is not inside the image
        """
        if not self._check_tile_offset(offset):
            raise IndexError("Offset {} is out of the image.".format(offset))
        width = min(max_width, self.width - offset[0])
        height = min(max_height, self.height - offset[1])
        return tile_builder.build(self, offset, width, height)

    def tile_iterator(self, builder, max_width=1024, max_height=1024, overlap=0):
        """Build and return a tile iterator that iterates over the image

        Parameters
        ----------
        builder: TileBuilder
            The builder to user for actually constructing the tiles while iterating over the image
        max_width: int, optional (default: 1024)
            The maximum width of the tiles to build
        max_height: int, optional (default: 1024)
            The maximum height of the tiles to build
        overlap: int, optional (default: 0)
            The overlapping between tiles

        Returns
        -------
        iterator: TileTopologyIterator
            An iterator that iterates over a tile topology of the image
        """
        topology = TileTopology(self, max_width=max_width, max_height=max_height, overlap=overlap)
        return TileTopologyIterator(builder, topology)

    def tile_topology(self, max_width=1024, max_height=1024, overlap=0):
        """Builds a tile topology over the image

        Parameters
        ----------
        max_width: int, optional (default: 1024)
            The maximum width of the tiles to build
        max_height: int, optional (default: 1024)
            The maximum height of the tiles to build
        overlap: int, optional (default: 0)
            The overlapping between tiles

        Returns
        -------
        topology: TileTopology
            The image tile topology
        """
        return TileTopology(self, max_width=max_width, max_height=max_height, overlap=overlap)

    def _check_tile_offset(self, offset):
        """Check whether the given tile offset belongs to the image

        Parameters
        ----------
        offset: (int, int)
            The (x, y) coordinates of the pixel at the origin point of the tile in the parent image

        Returns
        -------
        valid: bool
            True if the offset is valid, False otherwise
        """
        return 0 <= offset[0] < self.width and 0 <= offset[1] < self.height