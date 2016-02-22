import numpy as np
from sldc import Image, Tile, TileBuilder

class FakeImage(Image):
    """
    Fake image for testing
    """
    def __init__(self, w, h, c):
        Image.__init__(self)
        self._w = w
        self._h = h
        self._c = c

    @property
    def width(self):
        return self._w

    @property
    def height(self):
        return self._h

    @property
    def channels(self):
        return self._c


class FakeTile(Tile):
    """
    Fake tile for testing
    """
    def __init__(self, parent, offset, width, height):
        Tile.__init__(self, parent, offset, width, height)

    def get_numpy_repr(self):
        return np.zeros((self.width, self.height, self.channels))


class FakeTileBuilder(TileBuilder):
    """
    Fake tile builder for testing
    """
    def build(self, image, offset, width, height):
        return FakeTile(image, offset, width, height)