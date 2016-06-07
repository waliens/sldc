# -*- coding: utf-8 -*-
from joblib import Parallel

from .chaining import WorkflowChain, FullImageWorkflowExecutor, DefaultImageProvider
from .dispatcher import DispatcherClassifier, CatchAllRule
from .errors import MissingComponentException
from .image import DefaultTileBuilder
from .logging import SilentLogger
from .workflow import SLDCWorkflow

__author__ = "Mormont Romain <romainmormont@hotmail.com>"
__version__ = "0.1"


class WorkflowBuilder(object):
    """A class for building SLDC Workflow objects. When several instances of SLDCWorkflow should be built, they should
    be with the same Builder object, especially if the workflows should work in parallel.
    """
    def __init__(self, n_jobs=1):
        """Constructor for WorkflowBuilderObjects
        Parameters
        ----------
        n_jobs: int
            Number of jobs to use for executing the workflow
        """
        # Pool is preserved for building several instances of the workflow
        self._pool = Parallel(n_jobs=n_jobs)
        # Fields below are reset after each get()
        self._segmenter = None
        self._rules = None
        self._classifiers = None
        self._dispatching_labels = None
        self._tile_max_width = None
        self._tile_max_height = None
        self._overlap = None
        self._distance_tolerance = None
        self._logger = None
        self._tile_builder = None
        self._parallel = None
        self._reset()

    def _reset(self):
        """Reset the sldc workflow fields to their default values"""
        self._segmenter = None
        self._tile_builder = DefaultTileBuilder()
        self._rules = []
        self._classifiers = []
        self._dispatching_labels = []
        self._tile_max_width = 1024
        self._tile_max_height = 1024
        self._overlap = 7
        self._distance_tolerance = 1
        self._parallel = self._pool.n_jobs > 1
        self._logger = SilentLogger()

    @property
    def pool(self):
        """Return the builder's parallel pool"""
        return self._pool

    def set_parallel(self, in_parallel=True):
        """Enable/Disable parallelism parallel processing for the workflow
        By default, parallelism is enabled if the number of jobs passed in the constructor is more than 0.

        Parameters
        ----------
        in_parallel: bool (optional, default: True)
            True for executing the workflow in parallel, False to execute it sequentially

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._parallel = in_parallel
        return self

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
        """Set the tile builder
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

    def set_default_tile_builder(self):
        """Set the default tile builder as tile builder

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._tile_builder = DefaultTileBuilder()
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

    def set_overlap(self, overlap):
        """Set the tile overlap. If not called, an overlap of 5 is provided by default.
        Parameters
        ----------
        overlap: int
            The tile overlap

        Returns
        -------
        builder: WorkflowBuilder
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
        builder: WorkflowBuilder
            The builder
        """
        self._distance_tolerance = tolerance
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
            classifiers added before.

        Returns
        -------
        builder: WorkflowBuilder
            The builder
        """
        self._dispatching_labels.append(dispatching_label if dispatching_label is not None else len(self._rules))
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
        return self.add_classifier(CatchAllRule(), classifier, dispatching_label=dispatching_label)

    def get(self):
        """Build the workflow with the set parameters
        Returns
        -------
        workflow: SLDCWorkflow
            The SLDC Workflow

        Raises
        ------
        MissingComponentException:
            If some mandatory elements were not provided to the builder
        """
        if self._segmenter is None:
            raise MissingComponentException("Missing segmenter.")
        if self._tile_builder is None:
            raise MissingComponentException("Missing tile builder.")
        if len(self._rules) == 0 or len(self._classifiers) == 0:
            raise MissingComponentException("Missing classifiers.")

        dispatcher_classifier = DispatcherClassifier(self._rules, self._classifiers,
                                                     dispatching_labels=self._dispatching_labels, logger=self._logger)
        workflow = SLDCWorkflow(self._segmenter, dispatcher_classifier, self._tile_builder,
                                dist_tolerance=self._distance_tolerance,
                                tile_max_height=self._tile_max_height, tile_max_width=self._tile_max_width,
                                tile_overlap=self._overlap, logger=self._logger,
                                worker_pool=self._pool if self._parallel else None)
        self._reset()
        return workflow


class WorkflowChainBuilder(object):
    """A class for building workflow chains objects
    """
    def __init__(self):
        self._executors = None
        self._provider = None
        self._post_processor = None
        self._logger = None
        self._reset()

    def _reset(self):
        """Resets the builder so that it can build a new workflow chain
        """
        self._executors = []
        self._provider = None
        self._post_processor = None
        self._logger = SilentLogger()

    def add_executor(self, workflow_executor):
        """Adds a workflow executor to be executed by the workflow chain.
        Executors submitted through this method and 'add_full_image_executor' are submitted to the built WorkflowChain
        in the same order.

        Parameters
        ----------
        workflow_executor: WorkflowExecutor
            The workflow executor

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._executors.append(workflow_executor)
        return self

    def add_full_image_executor(self, workflow):
        """Adds a full image workflow executor.
        Executors submitted through this method and 'add_executor' are submitted to the built WorkflowChain
        in the same order.

        Parameters
        ----------
        workflow: SLDCWorkflow
            The workflow to encapsulate into the full image workflow executor

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._executors.append(FullImageWorkflowExecutor(workflow))
        return self

    def set_post_processor(self, post_processor):
        """Set the post processor of the workflow chain

        Parameters
        ----------
        post_processor: PostProcessor
            The post processor

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._post_processor = post_processor
        return self

    def set_image_provider(self, image_provider):
        """Set the image provider of the workflow chain

        Parameters
        ----------
        image_provider: ImageProvider

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._provider = image_provider
        return self

    def set_default_provider(self, images):
        """Set the image provider of the workflow chain

        Parameters
        ----------
        images: iterable (subtype: Image)
            The images to be processed

        Returns
        -------
        builder: WorkflowChainBuilder
            The builder
        """
        self._provider = DefaultImageProvider(images)
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
        if self._provider is None:
            raise MissingComponentException("Missing image provider.")
        if self._post_processor is None:
            raise MissingComponentException("Missing post processor")
        if len(self._executors) <= 0:
            raise MissingComponentException("At least one workflow executor should be provided.")

        chain = WorkflowChain(self._provider, self._executors, self._post_processor, logger=self._logger)
        self._reset()
        return chain
