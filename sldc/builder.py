# -*- coding: utf-8 -*-
from dispatcher import DispatcherClassifier, CatchAllRule
from workflow import SLDCWorkflow
from logging import SilentLogger
from errors import MissingComponentException

__author__ = "Mormont Romain <romainmormont@hotmail.com>"
__version__ = "0.1"


class WorkflowBuilder(object):
    """A class for building SLDC Workflow objects.
    """
    def __init__(self):
        """Constructor for WorkflowBuilderObjects"""
        self._segmenter = None
        self._rules = None
        self._classifiers = None
        self._tile_max_width = None
        self._tile_max_height = None
        self._boundary_thickness = None
        self._logger = None
        self._tile_builder = None
        self._reset()

    def _reset(self):
        """Reset the sldc workflow fields to their default values"""
        self._segmenter = None
        self._tile_builder = None
        self._rules = []
        self._classifiers = []
        self._tile_max_width = 1024
        self._tile_max_height = 1024
        self._boundary_thickness = 7
        self._logger = SilentLogger()

    def set_segmenter(self, segmenter):
        """Set the segmenter (mandatory)
        Parameters
        ----------
        segmenter: Segmenter
            The segmenter

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._segmenter = segmenter
        return self

    def set_logger(self, logger):
        """Set the logger. If not called, a SilentLogger is provided by default.
        Parameters
        ----------
        logger: Logger
            The logger

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._logger = logger
        return self

    def set_tile_builder(self, tile_builder):
        """Set the tile builder (mandatory)
        Parameters
        ----------
        tile_builder: TileBuilder
            The tile builder

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._tile_builder = tile_builder
        return self

    def set_tile_size(self, width, height):
        """Set the tile sizes. If not called, sizes (1024, 1024) are provided by default.
        Parameters
        ----------
        width: int
            The maximum width of the tiles
        height: int
            The maximum height of the tiles

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._tile_max_width = width
        self._tile_max_height = height
        return self

    def set_boundary_thickness(self, thickness):
        """Set the boundary thickness. If not called, a thickness of 7 is provided by default.
        Parameters
        ----------
        thickness: int
            The boundary thickness

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._boundary_thickness = thickness
        return self

    def add_classifier(self, rule, classifier):
        """Add a classifier to which polygons can be dispatched (mandatory, at least on time).

        Parameters
        ----------
        rule: DispatchingRule
            The dispatching rule that matches the polygons to be dispatched to the classifier
        classifier: PolygonClassifier
            The polygon that classifies polygons

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._rules.append(rule)
        self._classifiers.append(classifier)
        return self

    def add_catchall_classifier(self, classifier):
        """Add a classifier which is dispatched all the polygons that were note dispatched by the previously added
        classifiers.
        """
        return self.add_classifier(CatchAllRule(), classifier)

    def get(self):
        """Build the workflow with the set parameters
        Returns
        -------
        workflow: SLDCWorkflow
            The SLDC Workflow
        """
        if self._segmenter is None:
            raise MissingComponentException("Missing segmenter.")
        if self._tile_builder is None:
            raise MissingComponentException("Missing tile builder.")
        if len(self._rules) == 0 or len(self._classifiers) == 0:
            raise MissingComponentException("Missing classifiers.")

        dispatcher_classifier = DispatcherClassifier(self._rules, self._classifiers, logger=self._logger)
        return SLDCWorkflow(self._segmenter, dispatcher_classifier, self._tile_builder,
                            boundary_thickness=self._boundary_thickness,
                            tile_max_height=self._tile_max_height,
                            tile_max_width=self._tile_max_width, logger=self._logger)
