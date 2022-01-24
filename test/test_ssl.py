from unittest import TestCase

import numpy as np

from sldc import SemanticSegmenter, SSLWorkflowBuilder
from test import draw_square_by_corner, NumpyImage


class BasicSemanticSegmenter(SemanticSegmenter):
    def segment(self, image):
        return image


class TestFullWorkflow(TestCase):

    def testEmptyImage(self):
        image = np.zeros((200, 250), dtype=np.int64)
        builder = SSLWorkflowBuilder()
        builder.set_segmenter(BasicSemanticSegmenter())
        builder.set_default_tile_builder()
        builder.set_tile_size(100, 90)
        builder.set_background_class(0)
        workflow = builder.get()
        results = workflow.process(NumpyImage(image))

        self.assertEqual(len(results), 0, msg="no result")

    def testDetectSquares(self):
        # sorted by area
        all_poly = [
            (10, (10, 10), 85),
            (15, (50, 150), 255),
            (20, (130, 50), 190),
            (30, (10, 90), 85),
            (50, (70, 150), 190)
        ]
        image = np.zeros((200, 250), dtype=np.uint8)
        for side, top_left, color in all_poly:
            image = draw_square_by_corner(image, side, top_left, color)

        builder = SSLWorkflowBuilder()
        builder.set_segmenter(BasicSemanticSegmenter())
        builder.set_default_tile_builder()
        builder.set_tile_size(100, 90)
        builder.set_background_class(0)
        workflow = builder.get()

        results = workflow.process(NumpyImage(image))
        self.assertEqual(len(results), 5)

        idx = np.argsort([p.area for p in results.polygons])

        self.assertEqual((all_poly[0][0] + 1) ** 2, int(results.polygons[idx[0]].area))
        self.assertEqual((all_poly[1][0] + 1) ** 2, int(results.polygons[idx[1]].area))
        self.assertEqual((all_poly[2][0] + 1) ** 2, int(results.polygons[idx[2]].area))
        self.assertEqual((all_poly[3][0] + 1) ** 2, int(results.polygons[idx[3]].area))
        self.assertEqual((all_poly[4][0] + 1) ** 2, int(results.polygons[idx[4]].area))

        self.assertEqual(85, results.labels[idx[0]])
        self.assertEqual(255, results.labels[idx[1]])
        self.assertEqual(190, results.labels[idx[2]])
        self.assertEqual(85, results.labels[idx[3]])
        self.assertEqual(190, results.labels[idx[4]])

    def testDetectSquaresParallel(self):
        # sorted by area
        all_poly = [
            (10, (10, 10), 85),
            (15, (50, 150), 255),
            (20, (130, 50), 190),
            (30, (10, 90), 85),
            (50, (70, 150), 190)
        ]
        image = np.zeros((200, 250), dtype=np.uint8)
        for side, top_left, color in all_poly:
            image = draw_square_by_corner(image, side, top_left, color)

        builder = SSLWorkflowBuilder()
        builder.set_segmenter(BasicSemanticSegmenter())
        builder.set_default_tile_builder()
        builder.set_tile_size(100, 90)
        builder.set_background_class(0)
        builder.set_n_jobs(2)
        workflow = builder.get()

        results = workflow.process(NumpyImage(image))
        self.assertEqual(len(results), 5)

        idx = np.argsort([p.area for p in results.polygons])

        self.assertEqual((all_poly[0][0] + 1) ** 2, int(results.polygons[idx[0]].area))
        self.assertEqual((all_poly[1][0] + 1) ** 2, int(results.polygons[idx[1]].area))
        self.assertEqual((all_poly[2][0] + 1) ** 2, int(results.polygons[idx[2]].area))
        self.assertEqual((all_poly[3][0] + 1) ** 2, int(results.polygons[idx[3]].area))
        self.assertEqual((all_poly[4][0] + 1) ** 2, int(results.polygons[idx[4]].area))

        self.assertEqual(85, results.labels[idx[0]])
        self.assertEqual(255, results.labels[idx[1]])
        self.assertEqual(190, results.labels[idx[2]])
        self.assertEqual(85, results.labels[idx[3]])
        self.assertEqual(190, results.labels[idx[4]])