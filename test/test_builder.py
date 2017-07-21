from unittest import TestCase
from sldc.dispatcher import RuleBasedDispatcher, CatchAllRule

from sldc import SLDCWorkflowBuilder, MissingComponentException, Segmenter, DefaultTileBuilder, Dispatcher, \
    PolygonClassifier, SLDCWorkflow, DispatchingRule, InvalidBuildingException, StandardOutputLogger, Logger, \
    SSLWorkflow, SSLWorkflowBuilder, SilentLogger, WorkflowChainBuilder, DefaultFilter, WorkflowChain


class DumbSegmenter(Segmenter):
    def segment(self, image):
        return image


class DumbDispatcher(Dispatcher):
    def dispatch(self, image, polygon):
        return "default"


class DumbClassifier(PolygonClassifier):
    def predict(self, image, polygon):
        return 1


class DumbRule(DispatchingRule):
    def evaluate(self, image, polygon):
        return 1


class TestBuilder(TestCase):
    def testSLDCWorkflowWithOneShotDispatcher(self):
        # pre build components
        segmenter = DumbSegmenter()
        dispatcher = DumbDispatcher()
        classifier = DumbClassifier()
        rule = DumbRule()
        logger = StandardOutputLogger(Logger.DEBUG)

        builder = SLDCWorkflowBuilder()
        builder.set_tile_size(512, 768)
        builder.set_overlap(3)
        builder.set_distance_tolerance(5)
        builder.set_n_jobs(5)
        builder.set_logger(logger)
        builder.set_parallel_dc(True)
        builder.set_tile_builder(None)

        with self.assertRaises(MissingComponentException):
            builder.get()

        builder.set_segmenter(segmenter)

        with self.assertRaises(MissingComponentException):
            builder.get()

        builder.set_default_tile_builder()

        with self.assertRaises(MissingComponentException):
            builder.get()

        builder.set_one_shot_dispatcher(dispatcher, {"default": classifier})

        with self.assertRaises(InvalidBuildingException):
            builder.add_classifier(rule, classifier, dispatching_label="default")

        with self.assertRaises(InvalidBuildingException):
            builder.add_catchall_classifier(classifier, dispatching_label="default")

        workflow = builder.get()
        self.assertIsInstance(workflow, SLDCWorkflow)
        self.assertEqual(workflow._segmenter, segmenter)
        self.assertEqual(workflow._n_jobs, 5)
        self.assertEqual(workflow._tile_overlap, 3)
        self.assertEqual(workflow._tile_max_height, 512)
        self.assertEqual(workflow._tile_max_width, 768)
        self.assertEqual(workflow.logger, logger)
        self.assertIsInstance(workflow._tile_builder, DefaultTileBuilder)
        self.assertEqual(workflow._dispatch_classifier._dispatcher, dispatcher)
        self.assertEqual(len(workflow._dispatch_classifier._classifiers), 1)
        self.assertEqual(workflow._dispatch_classifier._classifiers[0], classifier)

    def testSLDCWorkflowWithRuleDispatcher(self):
        # pre build components
        segmenter = DumbSegmenter()
        tile_builder = DefaultTileBuilder()
        dispatcher = DumbDispatcher()
        classifier1 = DumbClassifier()
        classifier2 = DumbClassifier()
        rule1 = DumbRule()
        rule2 = DumbRule()

        builder = SLDCWorkflowBuilder()
        builder.set_segmenter(segmenter)
        builder.set_tile_builder(tile_builder)
        builder.add_classifier(rule1, classifier1)
        builder.add_classifier(rule2, classifier2)

        with self.assertRaises(InvalidBuildingException):
            builder.set_one_shot_dispatcher(dispatcher, {"default": classifier1})

        workflow = builder.get()
        self.assertIsInstance(workflow, SLDCWorkflow)
        self.assertEqual(len(workflow._dispatch_classifier._classifiers), 2)
        self.assertEqual(workflow._dispatch_classifier._classifiers[0], classifier1)
        self.assertEqual(workflow._dispatch_classifier._classifiers[1], classifier2)
        self.assertIsInstance(workflow._dispatch_classifier._dispatcher, RuleBasedDispatcher)
        self.assertEqual(len(workflow._dispatch_classifier._dispatcher._rules), 2)
        self.assertEqual(workflow._dispatch_classifier._dispatcher._rules[0], rule1)
        self.assertEqual(workflow._dispatch_classifier._dispatcher._rules[1], rule2)

    def testSLDCWorkflowWithCatchAllClassifier(self):
        # pre build components
        segmenter = DumbSegmenter()
        tile_builder = DefaultTileBuilder()
        dispatcher = DumbDispatcher()
        classifier = DumbClassifier()

        builder = SLDCWorkflowBuilder()
        builder.set_segmenter(segmenter)
        builder.set_tile_builder(tile_builder)

        builder.add_catchall_classifier(classifier)

        with self.assertRaises(InvalidBuildingException):
            builder.set_one_shot_dispatcher(dispatcher, {"default": classifier})

        workflow = builder.get()
        self.assertIsInstance(workflow, SLDCWorkflow)
        self.assertIsInstance(workflow._dispatch_classifier._dispatcher, RuleBasedDispatcher)
        self.assertEqual(len(workflow._dispatch_classifier._classifiers), 1)
        self.assertEqual(workflow._dispatch_classifier._classifiers[0], classifier)
        self.assertIsInstance(workflow._dispatch_classifier._dispatcher, RuleBasedDispatcher)
        self.assertEqual(len(workflow._dispatch_classifier._dispatcher._rules), 1)
        self.assertIsInstance(workflow._dispatch_classifier._dispatcher._rules[0], CatchAllRule)

    def testSSLWorkflowBuilder(self):
        segmenter = DumbSegmenter()
        builder = SSLWorkflowBuilder()
        builder.set_n_jobs(5)
        builder.set_tile_size(512, 768)
        builder.set_overlap(3)
        builder.set_distance_tolerance(7)
        builder.set_background_class(5)
        builder.set_logger(StandardOutputLogger(Logger.DEBUG))

        with self.assertRaises(MissingComponentException):
            builder.get()

        builder.set_segmenter(segmenter)
        builder.set_default_tile_builder()

        workflow = builder.get()
        self.assertIsInstance(workflow, SSLWorkflow)
        self.assertEqual(workflow._n_jobs, 5)
        self.assertEqual(workflow._tile_overlap, 3)
        self.assertEqual(workflow._tile_max_height, 512)
        self.assertEqual(workflow._tile_max_width, 768)
        self.assertIsInstance(workflow._tile_builder, DefaultTileBuilder)
        self.assertIsInstance(workflow.logger, StandardOutputLogger)
        self.assertEqual(workflow._locator._background, 5)

    def testWorkflowChainBuilder(self):
        segmenter = DumbSegmenter()
        dispatcher = DumbDispatcher()
        classifier = DumbClassifier()

        builder = SLDCWorkflowBuilder()
        builder.set_segmenter(segmenter)
        builder.set_default_tile_builder()
        builder.set_one_shot_dispatcher(dispatcher, {"default": classifier})
        workflow1 = builder.get()

        builder2 = SSLWorkflowBuilder()
        builder2.set_segmenter(segmenter)
        builder2.set_default_tile_builder()
        workflow2 = builder2.get()

        _filter = DefaultFilter()
        logger = StandardOutputLogger(Logger.DEBUG)
        chain_builder = WorkflowChainBuilder()
        chain_builder.set_logger(logger)

        with self.assertRaises(MissingComponentException):
            chain_builder.get()

        chain_builder.set_first_workflow(workflow1, label="first")
        chain_builder.add_executor(workflow2, label="second", filter=_filter, n_jobs=2, logger=logger)
        chain = chain_builder.get()

        self.assertIsInstance(chain, WorkflowChain)
        self.assertEqual(chain.logger, logger)
        self.assertEqual(chain._first_workflow, workflow1)
        self.assertEqual(len(chain._executors), 1)
        self.assertEqual(chain._executors[0]._workflow, workflow2)
        self.assertEqual(len(chain._labels), 2)
        self.assertEqual(tuple(chain._labels), ("first", "second"))
        self.assertEqual(len(chain._filters), 1)
        self.assertEqual(chain._filters[0], _filter)
