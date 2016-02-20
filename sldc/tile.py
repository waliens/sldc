# -*- coding: utf-8 -*-

__author__ = "Romain Mormont <r.mormont@student.ulg.ac.be>"

from abc import ABCMeta, abstractmethod
from image import Image


class Tile(Image):
    """
    Abstract representation of an image's tile
    A tile is an image extracted from a bigger image
    """

    __metaclass__ = ABCMeta

    def __init__(self, parent, offset, width, height):
        """Constructor for Tile objects

        Parameters
        ----------
        parent: Image
            The image from which is extracted the tile
        offset: (int, int)
            The x and y coordinates of the pixel at the origin point of the slide in the parent image.
            Coordinates order is the following : (x, y).
        width: int
            The width of the tile
        height: int
            The height of the tile

        Notes
        -----
        The coordinates origin is the leftmost pixel at the top of the slide
        """
        self._parent = parent
        self._offset = offset
        self._width = width
        self._height = height

    @property
    def offset_x(self):
        """Return the x offset of the tile
        Returns
        -------
        offset_x: int
            X offset of the tile
        """
        return self._offset[0]

    @property
    def offset_y(self):
        """Return the y offset of the tile
        Returns
        -------
        offset_y: int
            Y offset of the tile
        """
        return self._offset[1]

    @property
    def offset(self):
        """Return the offset of the tile
        Returns
        -------
        offset: (int, int)
            The (x, y) offset of the tile
        """
        return self._offset

    @property
    def channels(self):
        return self._parent.channels

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    @abstractmethod
    def get_numpy_repr(self):
        """Return a numpy representation of the tile
        Returns
        -------
        numpy_repr: array-like, shape = [width, heigth{, channels}]
            The array-like representation of the tile image.
        """
        pass


class TileBuilder(object):
    """
    A class for building tiles for a given image
    """
    __metaclass__ = ABCMeta

    def __init__(self, image=None):
        """Constructor for TileBuilder
        Parameters
        ----------
        image: Image, optional (default: None)
            The image for which tiles should be built
        Notes
        -----
        For the builder to work properly, the image field MUST be set but
        the operation can be deferred and performed using the image setter.
        Calling builder.build() without setting the image will cause an error.
        """
        self._image = image

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        self._image = image

    def build(self, offset, width, height):
        """Build and return a tile object

        Parameters
        ----------
        offset: (int, int)
            The (x, y) coordinates of the pixel at the origin point of the slide in the parent image
        width: int
            The width of the tile
        height: int
            The height of the tile

        Returns
        -------
        tile: Tile
            The built tile object

        Errors
        ------
        TypeError: when the reference image is not set

        Notes
        -----
        The coordinates origin is the leftmost pixel at the top of the slide
        """
        if self._image is None:
            raise TypeError("Reference 'image' is set to None.")
        return self._get_instance(offset, width, height)

    @abstractmethod
    def _get_instance(self, offset, width, height):
        """Build and return a tile object

        Parameters
        ----------
        offset: (int, int)
            The (x, y) coordinates of the pixel at the origin point of the slide in the parent image
        width: int
            The width of the tile
        height: int
            The height of the tile

        Returns
        -------
        tile: Tile
            The built tile object
        """
        pass


# TODO implement iterating with overlap of tiles
class TilesIterator(object):
    """
    An object to iterate over an image tile per tile
    """

    def __init__(self, builder, max_width=1024, max_height=1024):
        """Constructor for TilesIterator objects

        Parameters
        ----------
        builder: TileBuilder
            The builder to user for actually constructing the tiles while iterating over the image
        max_width: int, optional (default: 1024)
            The maximum width of the tile
        max_height: int, optional (default: 1024)
            The maximum height of the tile

        Notes
        -----
        Some tiles might actually be smaller than (max_width, max_height) on the edges of the image
        """
        self._builder = builder
        self._max_width = max_width
        self._max_height = max_height
        self._curr_offset = (0, 0)

    def __iter__(self):
        while self._is_valid_offset(self._curr_offset):
            offset = self._curr_offset
            image = self._builder.image
            width = min(image.width - offset[0], self._max_width)
            height = min(image.height - offset[1], self._max_height)
            yield self._builder.build(offset, width, height)
            self._curr_offset = self._next_offset(offset)

    def _is_valid_offset(self, offset):
        """Check whether the given offset fits withing the image
        Returns
        -------
        last: bool
            True if the offset fits, False otherwise
        """
        image = self._builder.image
        return offset[0] < image.width and offset[1] < image.height

    def _is_last_offset_col(self, offset):
        """Check whether the given offset yields the last tile on this image tile row
        Returns
        -------
        last: bool
            True if the tile is the last on its image row of tiles, False otherwise
        """
        return offset[0] + self._max_width > self._builder.image.width

    def _is_offset_last_row(self, offset):
        """Check whether the given offset yields the last tile on this image tile column
        Returns
        -------
        last: bool
            True if the tile is the last on its image column of tiles, False otherwise
        """
        return offset[1] + self._max_height > self._builder.image.height

    def _next_offset(self, offset):
        """Compute the next offset for iterating row by row through the image
        Parameters
        ----------
        offset: (int, int)
            The current offset
        Returns
        -------
        next_offset: (int, int)
            The next offset
        Notes
        -----
        The function make sure that an offset of which the column is out of bound is never returned.
        However, it returns an offset of which the row is out of bound when the given offset is the
        last in the image.
        """
        if self._is_last_offset_col(offset):
            return 0, (offset[1] + self._max_height)
        return offset[0] + self._max_width, offset[1]

    @staticmethod
    def _start_offset():
        """Return first offset
        Returns
        -------
        offset: (int, int)
            The offset on which the iterator must start. The tuple is ordered as follows : (x, y).
        """
        return 0, 0
