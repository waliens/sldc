from unittest import TestCase
from fake import FakeTileBuilder, FakeImage
from shapely.geometry import Polygon
from sldc.merger import Merger


class TestMergerRectangle(TestCase):
    def test_merge(self):
        fake_image = FakeImage(30, 11, 3)
        fake_builder = FakeTileBuilder()
        topology = fake_image.tile_topology(12, 9, 2)

        tile1 = topology.tile(1, fake_builder)
        tile2 = topology.tile(2, fake_builder)
        tile3 = topology.tile(3, fake_builder)
        tile4 = topology.tile(4, fake_builder)
        tile5 = topology.tile(5, fake_builder)
        tile6 = topology.tile(6, fake_builder)

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

        A = (5, 5)
        B = (5, 15)
        C = (9, 5)
        D = (9, 15)

        E = (1, 12)
        F = (1, 15)
        G = (3, 12)
        H = (3, 15)

        I = (5, 23)
        J = (5, 27)
        K = (9, 23)
        L = (9, 27)

        p = (7, 23)
        q = (7, 27)
        s = (7, 15)
        t = (7, 10)
        u = (7, 5)
        w = (9, 10)
        z = (5, 10)

        EFHG = Polygon([E, F, H, G, E])
        Aztu = Polygon([A, z, t, u, A])
        zBst = Polygon([z, B, s, t, z])
        tsDw = Polygon([t, s, D, w, t])
        utwC = Polygon([u, t, w, C, u])
        IJqp = Polygon([I, J, q, p, I])
        pqLK = Polygon([p, q, L, K, p])
        ABCD = Polygon([A, B, D, C, A])
        IJLK = Polygon([I, J, L, K, I])

        polygons_tiles = [(tile1, [Aztu]),
                          (tile2, [EFHG, zBst]),
                          (tile3, [IJqp]),
                          (tile4, [utwC]),
                          (tile5, [tsDw]),
                          (tile6, [pqLK])]

        polygons = Merger(1).merge(polygons_tiles, topology)
        self.assertEqual(len(polygons), 3, "Number of found polygon")
        self.assertTrue(polygons[0].equals(ABCD), "ABCD polygon")
        self.assertTrue(polygons[1].equals(EFHG), "EFHG polygon")
        self.assertTrue(polygons[2].equals(IJLK), "IJLK polygon")
