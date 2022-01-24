from unittest import TestCase

import numpy as np
from numpy.testing import assert_array_equal

from sldc.image import SkipBordersTileTopology, FixedSizeTileTopology
from test.util import NumpyImage
from test.fake_image import FakeImage, FakeTileBuilder


class TestTileFromImage(TestCase):
    def testTile(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(2500, 1750, 3)
        # Simple tile extraction
        tile = fake_image.tile(fake_builder, (1250, 1300), 250, 300)
        self.assertEqual(1250, tile.offset_x, "Tile from image : x offset")
        self.assertEqual(1300, tile.offset_y, "Tile from image : y offset")
        self.assertEqual(250, tile.width, "Tile from image : width")
        self.assertEqual(300, tile.height, "Tile from image : height")

        # Overflowing tile extraction
        tile = fake_image.tile(fake_builder, (1250, 1300), 1000, 1000)
        self.assertEqual(1250, tile.offset_x, "Overflowing tile from image : x offset")
        self.assertEqual(1300, tile.offset_y, "Overflowing tile from image : y offset")
        self.assertEqual(1000, tile.width, "Overflowing tile from image : width")
        self.assertEqual(450, tile.height, "Overflowing tile from image : height")

        # Both dimension overflowing
        tile = fake_image.tile(fake_builder, (2400, 1650), 300, 300)
        self.assertEqual(2400, tile.offset_x, "Both overflowing tile from image : x offset")
        self.assertEqual(1650, tile.offset_y, "Both overflowing tile from image : y offset")
        self.assertEqual(100, tile.width, "Both overflowing tile from image : width")
        self.assertEqual(100, tile.height, "Both overflowing tile from image : height")


class TestSingleTileTopology(TestCase):
    def testSingleTileTopology(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(700, 700, 3)
        topology = fake_image.tile_topology(fake_builder, 700, 700, 100)

        # topology metrics
        self.assertEqual(1, topology.tile_count, "Topology : tile count")
        self.assertEqual(1, topology.tile_horizontal_count, "Topology : tile horizontal count")
        self.assertEqual(1, topology.tile_vertical_count, "Topology : tile vertical count")

        tile = topology.tile(1)
        self.assertEqual(0, tile.offset_x, "Tile from image : x offset")
        self.assertEqual(0, tile.offset_y, "Tile from image : y offset")
        self.assertEqual(700, tile.width, "Tile from image : width")
        self.assertEqual(700, tile.height, "Tile from image : height")


class TestFittingTileTopology(TestCase):
    def testFittingTileTopology(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(700, 700, 3)
        topology = fake_image.tile_topology(fake_builder, 300, 300, 100)

        # topology metrics
        self.assertEqual(9, topology.tile_count, "Topology : tile count")
        self.assertEqual(3, topology.tile_horizontal_count, "Topology : tile horizontal count")
        self.assertEqual(3, topology.tile_vertical_count, "Topology : tile vertical count")

        # Topology that fits exactely the image
        tile = topology.tile(1)
        self.assertEqual(1, tile.identifier, "Topology, tile 1 : x offset")
        self.assertEqual(0, tile.offset_x, "Topology, tile 1 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 1 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 1 : width")
        self.assertEqual(300, tile.height, "Topology, tile 1 : height")

        tile = topology.tile(2)
        self.assertEqual(2, tile.identifier, "Topology, tile 2 : x offset")
        self.assertEqual(200, tile.offset_x, "Topology, tile 2 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 2 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 2 : width")
        self.assertEqual(300, tile.height, "Topology, tile 2 : height")

        tile = topology.tile(3)
        self.assertEqual(3, tile.identifier, "Topology, tile 3 : x offset")
        self.assertEqual(400, tile.offset_x, "Topology, tile 3 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 3 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 3 : width")
        self.assertEqual(300, tile.height, "Topology, tile 3 : height")

        tile = topology.tile(4)
        self.assertEqual(4, tile.identifier, "Topology, tile 4 : x offset")
        self.assertEqual(0, tile.offset_x, "Topology, tile 4 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 4 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 4 : width")
        self.assertEqual(300, tile.height, "Topology, tile 4 : height")

        tile = topology.tile(5)
        self.assertEqual(5, tile.identifier, "Topology, tile 5 : x offset")
        self.assertEqual(200, tile.offset_x, "Topology, tile 5 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 5 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 5 : width")
        self.assertEqual(300, tile.height, "Topology, tile 5 : height")

        tile = topology.tile(6)
        self.assertEqual(6, tile.identifier, "Topology, tile 6 : x offset")
        self.assertEqual(400, tile.offset_x, "Topology, tile 6 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 6 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 6 : width")
        self.assertEqual(300, tile.height, "Topology, tile 6 : height")

        tile = topology.tile(7)
        self.assertEqual(7, tile.identifier, "Topology, tile 7 : x offset")
        self.assertEqual(0, tile.offset_x, "Topology, tile 7 : x offset")
        self.assertEqual(400, tile.offset_y, "Topology, tile 7 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 7 : width")
        self.assertEqual(300, tile.height, "Topology, tile 7 : height")

        tile = topology.tile(8)
        self.assertEqual(8, tile.identifier, "Topology, tile 8 : x offset")
        self.assertEqual(200, tile.offset_x, "Topology, tile 8 : x offset")
        self.assertEqual(400, tile.offset_y, "Topology, tile 8 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 8 : width")
        self.assertEqual(300, tile.height, "Topology, tile 8 : height")

        tile = topology.tile(9)
        self.assertEqual(9, tile.identifier, "Topology, tile 9 : x offset")
        self.assertEqual(400, tile.offset_x, "Topology, tile 9 : x offset")
        self.assertEqual(400, tile.offset_y, "Topology, tile 9 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 9 : width")
        self.assertEqual(300, tile.height, "Topology, tile 9 : height")

        # neighbours
        self.assertEqual(topology.tile_neighbours(1), (None, 4, None, 2))
        self.assertEqual(topology.tile_neighbours(2), (None, 5, 1, 3))
        self.assertEqual(topology.tile_neighbours(3), (None, 6, 2, None))
        self.assertEqual(topology.tile_neighbours(4), (1, 7, None, 5))
        self.assertEqual(topology.tile_neighbours(5), (2, 8, 4, 6))
        self.assertEqual(topology.tile_neighbours(6), (3, 9, 5, None))
        self.assertEqual(topology.tile_neighbours(7), (4, None, None, 8))
        self.assertEqual(topology.tile_neighbours(8), (5, None, 7, 9))
        self.assertEqual(topology.tile_neighbours(9), (6, None, 8, None))


class TestOverflowingTopology(TestCase):

    def testOverFlowingTopology(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(600, 450, 3)
        topology = fake_image.tile_topology(fake_builder, 300, 300, 100)

        # topology metrics
        self.assertEqual(6, topology.tile_count, "Topology : tile count")
        self.assertEqual(3, topology.tile_horizontal_count, "Topology : tile horizontal count")
        self.assertEqual(2, topology.tile_vertical_count, "Topology : tile vertical count")

        # Topology that fits exactely the image
        tile = topology.tile(1)
        self.assertEqual(1, tile.identifier, "Topology, tile 1 : x offset")
        self.assertEqual(0, tile.offset_x, "Topology, tile 1 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 1 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 1 : width")
        self.assertEqual(300, tile.height, "Topology, tile 1 : height")

        tile = topology.tile(2)
        self.assertEqual(2, tile.identifier, "Topology, tile 2 : x offset")
        self.assertEqual(200, tile.offset_x, "Topology, tile 2 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 2 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 2 : width")
        self.assertEqual(300, tile.height, "Topology, tile 2 : height")

        tile = topology.tile(3)
        self.assertEqual(3, tile.identifier, "Topology, tile 3 : x offset")
        self.assertEqual(400, tile.offset_x, "Topology, tile 3 : x offset")
        self.assertEqual(0, tile.offset_y, "Topology, tile 3 : y offset")
        self.assertEqual(200, tile.width, "Topology, tile 3 : width")
        self.assertEqual(300, tile.height, "Topology, tile 3 : height")

        tile = topology.tile(4)
        self.assertEqual(4, tile.identifier, "Topology, tile 4 : x offset")
        self.assertEqual(0, tile.offset_x, "Topology, tile 4 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 4 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 4 : width")
        self.assertEqual(250, tile.height, "Topology, tile 4 : height")

        tile = topology.tile(5)
        self.assertEqual(5, tile.identifier, "Topology, tile 5 : x offset")
        self.assertEqual(200, tile.offset_x, "Topology, tile 5 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 5 : y offset")
        self.assertEqual(300, tile.width, "Topology, tile 5 : width")
        self.assertEqual(250, tile.height, "Topology, tile 5 : height")

        tile = topology.tile(6)
        self.assertEqual(6, tile.identifier, "Topology, tile 6 : x offset")
        self.assertEqual(400, tile.offset_x, "Topology, tile 6 : x offset")
        self.assertEqual(200, tile.offset_y, "Topology, tile 6 : y offset")
        self.assertEqual(200, tile.width, "Topology, tile 6 : width")
        self.assertEqual(250, tile.height, "Topology, tile 6 : height")

        # neighbours
        self.assertEqual(topology.tile_neighbours(1), (None, 4, None, 2))
        self.assertEqual(topology.tile_neighbours(2), (None, 5, 1, 3))
        self.assertEqual(topology.tile_neighbours(3), (None, 6, 2, None))
        self.assertEqual(topology.tile_neighbours(4), (1, None, None, 5))
        self.assertEqual(topology.tile_neighbours(5), (2, None, 4, 6))
        self.assertEqual(topology.tile_neighbours(6), (3, None, 5, None))


class TestTileTopologyPartition(TestCase):
    def testPartition(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(600, 700, 3)
        topology = fake_image.tile_topology(fake_builder, 300, 300, 100)

        # Test 5 batches
        batches1 = topology.partition_tiles(5)
        self.assertEqual(len(batches1), 5)
        self.checkBatches(batches1)

        # Test 1 batches
        batches2 = topology.partition_tiles(1)
        self.assertEqual(len(batches2), 1)
        self.checkBatches(batches2)

        # Test 10 batches
        batches3 = topology.partition_tiles(10)
        self.assertEqual(len(batches3), 9)
        self.checkBatches(batches3)

        # Test 10 batches
        batches4 = topology.partition_tiles(9)
        self.assertEqual(len(batches4), 9)
        self.checkBatches(batches4)

    def checkBatches(self, batches):
        identifier = 1
        for i, batch in enumerate(batches):
            for tile in batch:
                self.assertEqual(tile.identifier, identifier, msg="Expect tile {} in batch {}, got tile {}".format(identifier, i, tile.identifier))
                identifier += 1


class TestSkipBordersTileTopology(TestCase):
    def testVerticallyOverflowing(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(525, 450, 3)
        base_topology = fake_image.tile_topology(fake_builder, 175, 175, 0)
        topology = SkipBordersTileTopology(base_topology)

        self.assertEqual(topology.tile_horizontal_count, 3)
        self.assertEqual(topology.tile_vertical_count, 2)
        self.assertEqual(topology.tile_count, 6)
        self.assertEqual(6, len([t for t in topology]))

        self.assertEqual(topology.tile_neighbours(1), (None, 4, None, 2))
        self.assertEqual(topology.tile_neighbours(2), (None, 5, 1, 3))
        self.assertEqual(topology.tile_neighbours(3), (None, 6, 2, None))
        self.assertEqual(topology.tile_neighbours(4), (1, None, None, 5))
        self.assertEqual(topology.tile_neighbours(5), (2, None, 4, 6))
        self.assertEqual(topology.tile_neighbours(6), (3, None, 5, None))

    def testHorizontallyOverflowing(self):
        fake_builder = FakeTileBuilder()
        fake_image = FakeImage(450, 525, 3)
        base_topology = fake_image.tile_topology(fake_builder, 175, 175, 0)
        topology = SkipBordersTileTopology(base_topology)

        self.assertEqual(topology.tile_horizontal_count, 2)
        self.assertEqual(topology.tile_vertical_count, 3)
        self.assertEqual(topology.tile_count, 6)
        self.assertEqual(6, len([t for t in topology]))

        self.assertEqual(topology.tile_neighbours(1), (None, 3, None, 2))
        self.assertEqual(topology.tile_neighbours(2), (None, 4, 1, None))
        self.assertEqual(topology.tile_neighbours(3), (1, 5, None, 4))
        self.assertEqual(topology.tile_neighbours(4), (2, 6, 3, None))
        self.assertEqual(topology.tile_neighbours(5), (3, None, None, 6))
        self.assertEqual(topology.tile_neighbours(6), (4, None, 5, None))

    def testTileImageContent(self):
        fake_builder = FakeTileBuilder()
        image = np.zeros((100, 50), dtype=np.uint8)
        image[:45, :45] = 1
        image[45:90, :45] = 2
        fake_image = NumpyImage(image)

        base_topology = fake_image.tile_topology(fake_builder, 45, 45, 0)
        topology = SkipBordersTileTopology(base_topology)

        self.assertEqual(topology.tile_horizontal_count, 1)
        self.assertEqual(topology.tile_vertical_count, 2)
        self.assertEqual(topology.tile_count, 2)
        self.assertEqual(2, len([t for t in topology]))

        tile1 = topology.tile(1).np_image
        self.assertEqual(np.unique(tile1).tolist(), [1])
        tile2 = topology.tile(2).np_image
        self.assertEqual(np.unique(tile2).tolist(), [2])


class TestFixedSizeTileTopology(TestCase):
    def testTopology(self):
        fake_builder = FakeTileBuilder()
        image = np.zeros((100, 50), dtype=np.uint8)
        image[:45, :45] = 1
        image[45:90, :45] = 2
        fake_image = NumpyImage(image)
        base_topology = fake_image.tile_topology(fake_builder, 45, 45, 0)
        topology = FixedSizeTileTopology(base_topology)

        self.assertEqual(topology.tile_horizontal_count, 2)
        self.assertEqual(topology.tile_vertical_count, 3)
        self.assertEqual(topology.tile_count, 6)
        self.assertEqual(6, len([t for t in topology]))

        assert_array_equal(topology.tile(1).np_image, image[:45, :45])
        assert_array_equal(topology.tile(2).np_image, image[:45, 5:])
        assert_array_equal(topology.tile(3).np_image, image[45:90, :45])
        assert_array_equal(topology.tile(4).np_image, image[45:90, 5:])
        assert_array_equal(topology.tile(5).np_image, image[55:, :45])
        assert_array_equal(topology.tile(6).np_image, image[55:, 5:])
