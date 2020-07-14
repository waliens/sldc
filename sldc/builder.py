# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

from .chaining import WorkflowChain, WorkflowExecutor, DefaultFilter
from .dispatcher import DispatcherClassifier, CatchAllRule, RuleBasedDispatcher
from .errors import MissingComponentException, InvalidBuildingException
from .image import DefaultTileBuilder
from .logging import SilentLogger
from .workflow import SLDCWorkflow, SSLWorkflow, Workflow

__author__ = "Mormont Romain <romainmormont@hotmail.com>"
__version__ = "0.1"


class WorkflowBuilder(object):
    """Base class to be extended by any workflow builder object"""
    __metaclass__ = ABCMeta

    def __init__(self):
        self._distance_tolerance = None
        self._logger = None
        self._tile_max_width = None
        self._tile_max_height = None
        self._overlap = None
        self._tile_builder = None
        self._n_jobs = None
        self._seg_batch_size = None
        self._border_tiles = None

    @abstractmethod
    def _reset(self):
        """Reset workflow fields"""
        self._tile_builder = DefaultTileBuilder()
        self._distance_tolerance = 1
        self._tile_max_width = 1024
        self._tile_max_height = 1024
        self._overlap = 7
        self._n_jobs = 1
        self._logger = SilentLogger()
        self._seg_batch_size = 1
        self._border_tiles = Workflow.BORDER_TILES_KEEP

    @abstractmethod
    def get(self):
        """To call to get the built workflow
        Returns
        -------
        workflow: Workflow
            The built workflow
            
        Raises
        ------
        MissingComponentException:
            If some mandatory elements were not provided to the builder
        """
        pass

    def set_seg_batch_size(self, batch_size):
        """Set the batch size for segmentation
        Parameters
        ----------
        batch_size: int
            The batch size for segmentation

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._seg_batch_size = batch_size
        return self

    def set_border_tiles(self, border_tiles):
        """Set the border tiles policy
        Parameters
        ----------
        border_tiles: str
            The border tiles policy

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._border_tiles = border_tiles
        return self

    def set_tile_builder(self, tile_builder):
        """Set the tile builder
        Parameters
        ----------
        tile_builder: TileBuilder
            The tile builder

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._tile_builder = tile_builder
        return self

    def set_default_tile_builder(self):
        """Set the default tile builder as tile builder

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._tile_builder = DefaultTileBuilder()
        return self

    def set_tile_size(self, height, width):
        """Set the tile sizes. If not called, sizes (1024, 1024) are provided by default.
        Parameters
        ----------
        height: int
            The maximum height of the tiles
        width: int
            The maximum width of the tiles

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._tile_max_width = width
        self._tile_max_height = height
        return self

    def set_overlap(self, overlap):
        """Set the tile overlap. If not called, an overlap of 5 is provided by default.
        Parameters
        ----------
        overlap: int
            The tile overlap

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._overlap = overlap
        return self

    def set_distance_tolerance(self, tolerance):
        """Set the distance tolerance. If not called, a thickness of 7 is provided by default.
        Parameters
        ----------
        tolerance: int
            The maximal distance between two polygons so that they are considered from the same object

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._distance_tolerance = tolerance
        return self

    def set_n_jobs(self, n_jobs):
        """Set the number of available jobs (optional)
        Parameters
        ----------
        n_jobs: int
            The number of jobs available to execute the workflow

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._n_jobs = n_jobs
        return self

    def set_logger(self, logger):
        """Set the logger. If not called, a SilentLogger is provided by default.
        Parameters
        ----------
        logger: Logger
            The logger

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._logger = logger
        return self

    def get_kwargs(self):
        """Returns a dictionary mapping Workflow constructor parameters and their values

        Returns
        -------
        kwargs: dict
        """
        return {
            "tile_builder": self._tile_builder,
            "dist_tolerance": self._distance_tolerance,
            "tile_max_width": self._tile_max_width,
            "tile_max_height": self._tile_max_height,
            "tile_overlap": self._overlap,
            "n_jobs": self._n_jobs,
            "logger": self._logger,
            "seg_batch_size": self._seg_batch_size,
            "border_tiles": self._border_tiles
        }


class SLDCWorkflowBuilder(WorkflowBuilder):
    """A class for building SLDC Workflow objects. When several instances of SLDCWorkflow should be built, they should
    be with the same Builder object, especially if the workflows should work in parallel.
    """
    def __init__(self):
        """Constructor for WorkflowBuilderObjects
        Parameters
        ----------
        n_jobs: int
            Number of jobs to use for executing the workflow
        """
        # Fields below are reset after each get()
        super(SLDCWorkflowBuilder, self).__init__()
        self._segmenter = None
        self._rules = None
        self._dispatching_labels = None
        self._one_shot_dispatcher = None
        self._classifiers = None
        self._parallel_dc = None
        self._reset()

    def _reset(self):
        """Reset the sldc workflow fields to their default values"""
        super(SLDCWorkflowBuilder, self)._reset()
        self._segmenter = None
        self._rules = []
        self._dispatching_labels = []
        self._one_shot_dispatcher = None
        self._classifiers = []
        self._parallel_dc = False

    def get_kwargs(self):
        """extends parent method with SLDCWorkflow specifics"""
        # define the dispatcher and classifier
        if self._one_shot_dispatcher is None:
            dispatcher = RuleBasedDispatcher(
                self._rules,
                labels=self._dispatching_labels,
                logger=self._logger
            )
        else:
            dispatcher = self._one_shot_dispatcher

        dispatcher_classifier = DispatcherClassifier(
            dispatcher,
            self._classifiers,
            logger=self._logger
        )

        return {
            "segmenter": self._segmenter,
            "dispatcher_classifier": dispatcher_classifier,
            "parallel_dispatch_classify": self._parallel_dc,
            **super().get_kwargs()
        }

    def set_parallel_dc(self, parallel_dc):
        """Specify whether the dispatching and classification will be parallelized at the workflow level (optional)
        Parameters
        ----------
        parallel_dc: boolean
            True for enabling parallelization of dispatching and classification at the workflow level

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._parallel_dc = parallel_dc
        return self

    def set_segmenter(self, segmenter):
        """Set the segmenter (mandatory)
        Parameters
        ----------
        segmenter: Segmenter
            The segmenter

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        self._segmenter = segmenter
        return self

    def add_classifier(self, rule, classifier, dispatching_label=None):
        """Add a classifier to which polygons can be dispatched (mandatory, at least on time).

        Parameters
        ----------
        rule: DispatchingRule
            The dispatching rule that matches the polygons to be dispatched to the classifier
        classifier: PolygonClassifier
            The polygon that classifies polygons
        dispatching_label: key (optional, default: None)
            The dispatching label for this classifier. By default, (n) is used where n is the number of rules and
            classifiers added before (n is formatted as a string)

        Returns
        -------
        builder: SLDCWorkflowBuilder
            The builder
        """
        if self._one_shot_dispatcher is not None:
            raise InvalidBuildingException("Cannot use a rule based dispatcher alongside a one shot dispatcher.")
        self._dispatching_labels.append(dispatching_label if dispatching_label is not None else str(len(self._rules)))
        self._rules.append(rule)
        self._classifiers.append(classifier)
        return self

    def add_catchall_classifier(self, classifier, dispatching_label="catchall"):
        """Add a classifier which is dispatched all the polygons that were note dispatched by the previously added
        classifiers.

        Parameters
        ----------
        classifier: PolygonClassifier
            The classifier
        dispatching_label: key (optional, default: "catchall")
            The dispatching label
        """
        if self._one_shot_dispatcher is not None:
            raise InvalidBuildingException("Cannot use a rule based dispatcher alongside a one shot dispatcher.")
        return self.add_classifier(CatchAllRule(), classifier, dispatching_label=dispatching_label)

    def set_one_shot_dispatcher(self, dispatcher, classifier_mapping):
        """Use the one shot user dispatching strategy and sets the dispatcher and classifiers.

        Parameters
        ----------
        dispatcher: Dispatcher
            A dispatcher
        classifier_mapping: dict (key: hashable, subtype: PolygonClassifier)
            Maps labels returned by the dispatcher with their corresponding classifiers.
        """
        if len(self._rules) > 0:
            raise InvalidBuildingException("Cannot use a one shot dispatcher alongside "
                                           "a rule based one (already defined {} rules).".format(len(self._rules)))

        # extract mapping and classifiers
        self._one_shot_dispatcher = dispatcher
        dispatcher.mapping, self._classifiers = zip(*[
            (label, classifier) for label, classifier in classifier_mapping.items()
        ])

    def get(self):
        if self._segmenter is None:
            raise MissingComponentException("Missing segmenter.")
        if self._tile_builder is None:
            raise MissingComponentException("Missing tile builder.")
        if self._one_shot_dispatcher is None and len(self._rules) == 0:
            raise MissingComponentException("Missing dispatching strategy. Either one shot or rule based "
                                            "dispatching must be used.")
        if len(self._classifiers) == 0:
            raise MissingComponentException("Missing classifiers.")
        workflow = SLDCWorkflow(**self.get_kwargs())
        self._reset()
        return workflow


class SSLWorkflowBuilder(WorkflowBuilder):
    """For building ssl workflows"""
    def __init__(self):
        """Constructor for WorkflowBuilderObjects
        Parameters
        ----------
        n_jobs: int
            Number of jobs to use for executing the workflow
        """
        # Fields below are reset after each get()
        super(SSLWorkflowBuilder, self).__init__()
        self._segmenter = None
        self._background_class = None
        self._reset()

    def _reset(self):
        """Reset the sldc workflow fields to their default values"""
        super(SSLWorkflowBuilder, self)._reset()
        self._segmenter = None
        self._background_class = -1

    def get_kwargs(self):
        """extends parent method with SLDCWorkflow specifics"""
        return {
            "segmenter": self._segmenter,
            "background_class": self._background_class,
            **super().get_kwargs()
        }

    def set_segmenter(self, segmenter):
        """Set the segmenter
        Parameters
        ----------
        segmenter: SemanticSegmenter
            The segmenter
        """
        self._segmenter = segmenter
        return self

    def set_background_class(self, background_class):
        """Set the background class
        Parameters
        ----------
        background_class: int
            The background class
        """
        self._background_class = background_class
        return self

    def get(self):
        if self._segmenter is None:
            raise MissingComponentException("Missing segmenter.")
        if self._tile_builder is None:
            raise MissingComponentException("Missing tile builder.")
        workflow = SSLWorkflow(**self.get_kwargs())
        self._reset()
        return workflow


class WorkflowChainBuilder(object):
    """A class for building workflow chains objects
    """
    def __init__(self):
        self._first_workflow = None
        self._executors = None
        self._filters = None
        self._labels = None
        self._logger = None
        self._reset()

    def _reset(self):
        """Resets the builder so that it can build a new workflow chain
        """
        self._first_workflow = None
        self._executors = []
        self._filters = []
        self._labels = []
        self._logger = SilentLogger()

    def set_first_workflow(self, workflow, label=None):
        """Set the workflow that will process the full image
        Parameters
        ----------
        workflow: Workflow
            The workflow
        label: hashable (optional)
            The label identifying the workflow. If not set, this label is set to 0.

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        actual_label = 0 if label is None else label
        if self._first_workflow is None:
            self._labels.insert(0, actual_label)
        else:
            self._labels[0] = actual_label
        self._first_workflow = workflow
        return self

    def add_executor(self, workflow, filter=DefaultFilter(), label=None, logger=SilentLogger(), n_jobs=1):
        """Adds a workflow executor to be executed by the workflow chain.

        Parameters
        ----------
        workflow: Workflow
            The workflow object
        filter: PolygonFilter (optional, default: DefaultFilter)
            The polygon filter implementing the filtering of polygons of which the windows will be processed to
            the workflow.
        label: hashable (optional)
            The label identifying the executor. If not set, the number of the executor is used instead (starting at 1)
        logger: Logger (optional, default: SilentLogger)
            The logger to be used by the executor object
        n_jobs: int (optional, default: 1)
            The number of jobs for executing the workflow on the images

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._executors.append(WorkflowExecutor(workflow, logger=logger, n_jobs=n_jobs))
        self._filters.append(filter)
        actual_label = len(self._executors) if label is None else label
        self._labels.append(actual_label)
        return self

    def set_logger(self, logger):
        """Set the logger of the workflow chain

        Parameters
        ----------
        logger: Logger
            The logger

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._logger = logger
        return self

    def get(self):
        """Build the workflow chain with the set parameters
        Returns
        -------
        workflow: WorkflowChain
            The workflow chain

        Raises
        ------
        MissingComponentException:
            If some mandatory elements were not provided to the builder
        """
        if self._first_workflow is None:
            raise MissingComponentException("Missing first workflow.")
        if len(self._labels) != len(self._executors) + 1:
            raise MissingComponentException("The number of labels ({}) should be the".format(len(self._labels)) +
                                            " same as the number of workflows ({}).".format(len(self._executors) + 1))
        if len(self._filters) != len(self._executors):
            raise MissingComponentException("The number of filters ({}) should be the".format(len(self._filters)) +
                                            " same as the number of executors ({}).".format(len(self._executors)))

        chain = WorkflowChain(self._first_workflow, self._executors, self._filters, self._labels, logger=self._logger)
        self._reset()
        return chain
