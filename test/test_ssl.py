from unittest import TestCase

import cv2
import numpy as np
from shapely.geometry import Polygon

from sldc import SemanticSegmenter, SSLWorkflowBuilder
from test import draw_square, draw_square_by_corner, NumpyImage


class BasicSemanticSegmenter(SemanticSegmenter):
    def segment(self, image):
        return image


class TestFullWorkflow(TestCase):
    def testDetectSquares(self):
        image = np.zeros((200, 250), dtype=np.uint8)
        image = draw_square_by_corner(image, 10, (10, 10), 85)
        image = draw_square_by_corner(image, 30, (10, 90), 85)
        image = draw_square_by_corner(image, 50, (70, 150), 190)
        image = draw_square_by_corner(image, 20, (130, 50), 190)
        image = draw_square_by_corner(image, 15, (50, 150), 255)

        builder = SSLWorkflowBuilder()
        builder.set_segmenter(BasicSemanticSegmenter())
        builder.set_default_tile_builder()
        builder.set_tile_size(100, 90)
        builder.set_background_class(0)
        workflow = builder.get()

        results = workflow.process(NumpyImage(image))
        self.assertEqual(len(results), 5)

        idx = np.argsort([p.area for p in results.polygons])

        self.assertEqual(100, int(results.polygons[idx[0]].area))
        self.assertEqual(225, int(results.polygons[idx[1]].area))
        self.assertEqual(400, int(results.polygons[idx[2]].area))
        self.assertEqual(900, int(results.polygons[idx[3]].area))
        self.assertEqual(2500, int(results.polygons[idx[4]].area))

        self.assertEqual(85, results.labels[idx[0]])
        self.assertEqual(255, results.labels[idx[1]])
        self.assertEqual(190, results.labels[idx[2]])
        self.assertEqual(85, results.labels[idx[3]])
        self.assertEqual(190, results.labels[idx[4]])

    def testDetectSquaresParallel(self):
        image = np.zeros((200, 250), dtype=np.uint8)
        image = draw_square_by_corner(image, 10, (10, 10), 85)
        image = draw_square_by_corner(image, 30, (10, 90), 85)
        image = draw_square_by_corner(image, 50, (70, 150), 190)
        image = draw_square_by_corner(image, 20, (130, 50), 190)
        image = draw_square_by_corner(image, 15, (50, 150), 255)

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

        self.assertEqual(100, int(results.polygons[idx[0]].area))
        self.assertEqual(225, int(results.polygons[idx[1]].area))
        self.assertEqual(400, int(results.polygons[idx[2]].area))
        self.assertEqual(900, int(results.polygons[idx[3]].area))
        self.assertEqual(2500, int(results.polygons[idx[4]].area))

        self.assertEqual(85, results.labels[idx[0]])
        self.assertEqual(255, results.labels[idx[1]])
        self.assertEqual(190, results.labels[idx[2]])
        self.assertEqual(85, results.labels[idx[3]])
        self.assertEqual(190, results.labels[idx[4]])