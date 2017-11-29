import os
import numpy as np
from sldc import Segmenter, DispatchingRule, PolygonClassifier, report_timing, StandardOutputLogger, Logger
from sldc.builder import SLDCWorkflowBuilder
from test import NumpyImage, draw_circle, draw_square


""" (1) Define a segmenting procedure for locating the objects """
class BasicSegmenter(Segmenter):
    def __init__(self):
        super(BasicSegmenter, self).__init__()

    def segment(self, image):
        """Assume grayscale image"""
        mask = (image > 0).astype(np.uint8)
        mask[mask == 1] = 255
        return mask


""" (2) Define a dispatching rule for identifying circle and squares"""
class ShapeRule(DispatchingRule):
    CIRCLE = "circle"
    SQUARE = "square"

    def __init__(self, which=CIRCLE):
        self._which = which

    def evaluate(self, image, polygon):
        circularity = 4 * np.pi * polygon.area / (polygon.length * polygon.length)
        if self._which == self.CIRCLE:
            return circularity > 0.85
        else:
            return circularity <= 0.85


""" (3) Define a classifier that classify the object according to its color"""
class ColorClassifier(PolygonClassifier):
    def predict(self, image, polygon):
        windows = image.np_image
        x, y = int(polygon.centroid.x), int(polygon.centroid.y)
        return windows[y, x], 1.0


def main():
    """ (4) build the workflow """
    logger = StandardOutputLogger(Logger.INFO)
    builder = SLDCWorkflowBuilder()
    builder.set_n_jobs(4)
    builder.set_logger(logger)
    builder.set_default_tile_builder()
    builder.set_tile_size(256, 256)
    builder.set_segmenter(BasicSegmenter())
    builder.add_classifier(ShapeRule(ShapeRule.CIRCLE), ColorClassifier(), "circle")
    builder.add_classifier(ShapeRule(ShapeRule.SQUARE), ColorClassifier(), "square")

    workflow = builder.get()

    """ (5) build a (fake) image """
    np_image = np.zeros([2000, 2000], dtype=np.uint8)
    np_image = draw_circle(np_image, 100, (500, 1500), 255)
    np_image = draw_circle(np_image, 100, (1500, 600), 127)
    np_image = draw_square(np_image, 200, (500, 500), 255)
    np_image = draw_square(np_image, 200, (1500, 1500), 127)
    np_image = draw_square(np_image, 300, (1000, 1000), 255)
    image = NumpyImage(np_image)

    """ (6) process the image """
    results = workflow.process(image)

    """ (7) report execution times """
    report_timing(results.timing, logger)

    """ (8) explore the results """
    for i, object_info in enumerate(results):
        print(
            "Object {}:".format(i + 1) + os.linesep +
            "> area    : {}".format(object_info.polygon.area) + os.linesep +
            "> label   : '{}'".format(object_info.label) + os.linesep +
            "> proba   : {}".format(object_info.proba) + os.linesep +
            "> dispatch: '{}'".format(object_info.dispatch)
        )


if __name__ == "__main__":
    main()