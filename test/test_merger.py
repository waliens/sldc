from operator import add
from unittest import TestCase

from shapely.affinity import translate
from shapely.geometry import Polygon, box, Point

from test.fake_image import FakeTileBuilder, FakeImage
from sldc import SemanticMerger


class TestMergerNoPolygon(TestCase):
    def testMerge(self):
        fake_image = FakeImage(11, 8, 3)
        fake_builder = FakeTileBuilder()
        topology = fake_image.tile_topology(fake_builder, 5, 5, 1)

        tile1 = topology.tile(1)
        tile2 = topology.tile(2)
        tile3 = topology.tile(3)
        tile4 = topology.tile(4)

        #    0    5    10 (col)
        #  0 +---------+
        #    |    |    |
        #    |    |    |
        #    |    |    |
        #  4 |----+----+
        #    |    |    |
        #    |    |    |
        #  7 +----G----H
        # (row)
        tiles = [tile1.identifier, tile2.identifier, tile3.identifier, tile4.identifier]
        tile_polygons = [[]] * 4
        merger = SemanticMerger(1)
        polygons = merger.merge(tiles, tile_polygons, topology)
        self.assertEqual(len(polygons), 0, "Number of found polygon")


class TestMergerSingleTile(TestCase):
    def testMerge(self):
        fake_image = FakeImage(11, 8, 3)
        fake_builder = FakeTileBuilder()
        topology = fake_image.tile_topology(fake_builder, 11, 8, 1)

        tile1 = topology.tile(1)

        #    0    5    10 (col)
        #  0 +---------+
        #    | A--B    |
        #    | |  |    |
        #    | C--D    |
        #  4 |         |
        #    |    E----F
        #    |    |    |
        #  7 +----G----H
        # (row)

        A = (1, 2)
        B = (1, 5)
        C = (3, 1)
        D = (2, 5)

        E = (5, 5)
        F = (5, 10)
        G = (7, 5)
        H = (7, 10)

        ABCD = Polygon([A, B, D, C, A])
        EFGH = Polygon([E, F, H, G, E])

        tiles = [tile1.identifier]
        tile_polygons = [[ABCD, EFGH]]

        merger = SemanticMerger(1)
        polygons = merger.merge(tiles, tile_polygons, topology)
        self.assertEqual(len(polygons), 2, "Number of found polygon")
        self.assertTrue(polygons[0].equals(ABCD), "ABCD polygon")
        self.assertTrue(polygons[1].equals(EFGH), "EFHG polygon")


class TestMergerRectangle(TestCase):
    @staticmethod 
    def get_test_data(*topology_params, efgh_near_edge=False, add_non_unique_labels=False):
        fake_image = FakeImage(30, 12 if efgh_near_edge else 11, 3)
        fake_builder = FakeTileBuilder()
        topology = fake_image.tile_topology(fake_builder, *topology_params)

        # efgh_near_edge = False
        #                   15           27
        #    0    5    10    16  20 23     30  (col)
        #  0 +---------+---------+---------+
        #    |         |         |         |
        #    |         |  E--F   |         |
        #    |         |  |  |   |         |
        #    |         |  G--H   |         |
        #  5 |         |         |         |
        #    |    A----z----B    |  I---J  |
        #    |    |    |    |    |  |   |  |
        #  8 +----u----t----s----+--p---q--+
        #    |    |    |    |    |  |   |  |
        # 10 |    C----w----D    |  K---L  |
        #    |         |         |         |
        # 12 +---------+---------+---------+
        # (row)

        
        # efgh_near_edge = True
        #    0    5    10   15   20        30  (col)
        #  0 +---------+---------+---------+
        #    |         | E--F    |         |
        #    |         | |  |    |         |
        #    |         | G--H    |         |
        #  4 |         |         |         |
        #    |    A----z----B    |  I---J  |
        #    |    |    |    |    |  |   |  |
        #  7 +----u----t----s----+--p---q--+
        #    |    |    |    |    |  |   |  |
        #  9 |    C----w----D    |  K---L  |
        #    |         |         |         |
        # 11 +---------+---------+---------+
        # (row)

        # efgh_near_edge = True
        # add_non_unique_labels = True
        #    0    5    10   15   20        30  (col)
        #  0 +---------+---------+---------+
        #    |         | E--F    |         |
        #    |         | |1 |    |         |
        #    |         | G--H    |         |
        #  4 |         |         |         |
        #    |    A----z----B    |  I---J  |
        #    |    | 1  | 2  |    |  | 2 |  |
        #  7 +----u----t----s----+--p---q--+
        #    |    | 1  | 1  |    |  | 2 |  |
        #  9 |    C----w----D    |  K---L  |
        #    |         |         |         |
        # 11 +---------+---------+---------+
        # (row)

        tile1 = topology.tile(1)
        tile2 = topology.tile(2)
        tile3 = topology.tile(3)
        tile4 = topology.tile(4)
        tile5 = topology.tile(5)
        tile6 = topology.tile(6)

        A = (5,  5)
        B = (15, 5)
        C = (5,  9)
        D = (15, 9)

        E = (12, 1) if efgh_near_edge else (13, 2)
        F = (15, 1) if efgh_near_edge else (16, 2)
        G = (12, 3) if efgh_near_edge else (13, 4)
        H = (15, 3) if efgh_near_edge else (16, 4)

        I = (23, 5)
        J = (27, 5)
        K = (23, 9)
        L = (27, 9)

        p = (23, 7)
        q = (27, 7)
        s = (15, 7)
        t = (10, 7)
        u = (5,  7)
        w = (10, 9)
        z = (10, 5)

        EFHG = Polygon([E, F, H, G, E])
        Aztu = Polygon([A, z, t, u, A])
        zBst = Polygon([z, B, s, t, z])
        tsDw = Polygon([t, s, D, w, t])
        utwC = Polygon([u, t, w, C, u])
        IJqp = Polygon([I, J, q, p, I])
        pqLK = Polygon([p, q, L, K, p])
        ABCD = Polygon([A, B, D, C, A])
        IJLK = Polygon([I, J, L, K, I])
        AztsDwCu = Polygon([A, z, t, s, D, w, C, u, A])

        poly_dict = {
            "EFHG": EFHG, "Aztu": Aztu, "zBst": zBst, "tsDw": tsDw, 
            "utwC": utwC, "IJqp": IJqp, "pqLK": pqLK, "ABCD": ABCD, 
            "IJLK": IJLK, "AztsDwCu": AztsDwCu
        }

        tiles = [tile1.identifier, tile2.identifier, tile3.identifier, tile4.identifier, tile5.identifier, tile6.identifier]
        tile_polygons = [[Aztu], [EFHG, zBst], [IJqp], [utwC], [tsDw], [pqLK]]

        if not add_non_unique_labels:
            return topology, tiles, tile_polygons, poly_dict
        else:
            tile_labels = [[1], [1, 2], [2], [1], [1], [2]]
            return topology, tiles, tile_polygons, poly_dict, tile_labels

    def testMergeNoOverlap(self):
        topology, tiles, tile_polygons, poly_dict = TestMergerRectangle.get_test_data(11, 8, 0, efgh_near_edge=False)
        polygons = SemanticMerger(1).merge(tiles, tile_polygons, topology)
        self.assertEqual(len(polygons), 3, "Number of found polygon")
        self.assertTrue(polygons[0].equals(poly_dict["ABCD"]), "ABCD polygon")
        self.assertTrue(polygons[1].equals(poly_dict["EFHG"]), "EFHG polygon")
        self.assertTrue(polygons[2].equals(poly_dict["IJLK"]), "IJLK polygon")

    def testMergeNoOverlapNoTol(self):
        topology, tiles, tile_polygons, poly_dict = TestMergerRectangle.get_test_data(11, 8, 0, efgh_near_edge=False)
        polygons = SemanticMerger(0).merge(tiles, tile_polygons, topology)
        self.assertEqual(len(polygons), 7, "Number of found polygon")

    def testMergeWithOverlapAndTolerance(self):
        topology, tiles, tile_polygons, poly_dict = TestMergerRectangle.get_test_data(12, 9, 2, efgh_near_edge=True)
        polygons = SemanticMerger(1).merge(tiles, tile_polygons, topology)
        self.assertEqual(len(polygons), 3, "Number of found polygon")
        self.assertTrue(polygons[0].equals(poly_dict["ABCD"]), "ABCD polygon")
        self.assertTrue(polygons[1].equals(poly_dict["EFHG"]), "EFHG polygon")
        self.assertTrue(polygons[2].equals(poly_dict["IJLK"]), "IJLK polygon")

    def testSemanticMerge(self):
        topology, tiles, tile_polygons, poly_dict, tile_labels = TestMergerRectangle.get_test_data(12, 9, 2, efgh_near_edge=True, add_non_unique_labels=True)
        polygons, labels = SemanticMerger(1).merge(tiles, tile_polygons, topology, labels=tile_labels)
        self.assertEqual(len(polygons), 4, "Number of found polygon")
        self.assertTrue(polygons[0].equals(poly_dict["AztsDwCu"]), "AztsDwCu polygon")
        self.assertTrue(labels[0], 1)
        self.assertTrue(polygons[1].equals(poly_dict["EFHG"]), "EFHG polygon")
        self.assertTrue(labels[1], 1)
        self.assertTrue(polygons[2].equals(poly_dict["zBst"]), "zBst polygon")
        self.assertTrue(labels[2], 2)
        self.assertTrue(polygons[3].equals(poly_dict["IJLK"]), "IJLK polygon")
        self.assertTrue(labels[3], 2)


class TestMergerBigCircle(TestCase):
    def testMerger(self):
        # build chunks for the polygons
        tile_box = box(0, 0, 512, 256)  # a box having the same dimension as the tile
        circle = Point(600, 360)
        circle = circle.buffer(250)

        circle_part1 = tile_box.intersection(circle)
        circle_part2 = translate(tile_box, xoff=512).intersection(circle)
        circle_part3 = translate(tile_box, yoff=256).intersection(circle)
        circle_part4 = translate(tile_box, xoff=512, yoff=256).intersection(circle)
        circle_part5 = translate(tile_box, yoff=512).intersection(circle)
        circle_part6 = translate(tile_box, xoff=512, yoff=512).intersection(circle)

        # create topology
        fake_image = FakeImage(1024, 768, 3)
        fake_builder = FakeTileBuilder()
        topology = fake_image.tile_topology(fake_builder, 512, 256)

        tile1 = topology.tile(1)
        tile2 = topology.tile(2)
        tile3 = topology.tile(3)
        tile4 = topology.tile(4)
        tile5 = topology.tile(5)
        tile6 = topology.tile(6)

        tiles = [tile1.identifier, tile2.identifier, tile3.identifier, tile4.identifier, tile5.identifier, tile6.identifier]
        tile_polygons = [[circle_part1], [circle_part2], [circle_part3], [circle_part4], [circle_part5], [circle_part6]]

        polygons = SemanticMerger(5).merge(tiles, tile_polygons, topology)
        self.assertEqual(len(polygons), 1, "Number of found polygon")

        # use recall and false discovery rate to evaluate the error on the surface
        tpr = circle.difference(polygons[0]).area / circle.area
        fdr = polygons[0].difference(circle).area / polygons[0].area
        self.assertLessEqual(tpr, 0.002, "Recall is low for circle area")
        self.assertLessEqual(fdr, 0.002, "False discovery rate is low for circle area")
