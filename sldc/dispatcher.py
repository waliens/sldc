# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod

import numpy as np

from .timing import WorkflowTiming
from .logging import Loggable, SilentLogger
from .util import emplace, shape_array, take

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version__ = "0.1"


class Dispatcher(Loggable):
    """A dispatcher is an object that for a given set of polygons and an image returns a dispatch index
    computed using some user defined logic.
    """
    __metaclass__ = ABCMeta

    def __init__(self, mapping=None, timing=None, logger=SilentLogger()):
        """
        Parameters
        ----------
        mapping: iterable (subtype: hashable)
            List containing the dispatch labels/indexes returned by the user defined procedure.
            This list is used by the dispatch_map to produce the dispatch indexes returned by
            the dispatcher (see dispatch_map documentation). By default assumes no mapping.
        timing: WorkflowTiming
            An optional workflow timing for computing dispatching time
        """
        super(Dispatcher, self).__init__(logger=logger)
        self._mapping, self._inverse_mapping = None, None  # to avoid duplicate initialization
        self.mapping = mapping
        self._timing = WorkflowTiming() if timing is None else timing

    @property
    def label_count(self):
        """Return the number of possible dispatch indexes/labels"""
        return len(self._mapping)

    def __len__(self):
        """Return the number of possible dispatch indexes/labels"""
        return self.label_count

    @property
    def timing(self):
        return self._timing

    @timing.setter
    def timing(self, timing):
        self._timing = timing

    @property
    def mapping(self):
        return self._mapping.keys()

    @mapping.setter
    def mapping(self, mapping):
        self._mapping = self._transform_mapping(mapping)
        self._inverse_mapping = mapping

    @staticmethod
    def _transform_mapping(mapping):
        return None if mapping is None else {v: i for i, v in enumerate(mapping)}

    @abstractmethod
    def dispatch(self, image, polygon):
        """Return the dispatch index/label for the given polygon and image

        Parameters
        ----------
        image: Image
            The image from which is extracted the polygon
        polygon: shapely.geometry.Polygon
            The polygon to dispatch

        Returns
        -------
        index: hashable
            The dispatch index/label. None if the polygon cannot be dispatched
        """
        pass

    def dispatch_batch(self, image, polygons):
        """Return the dispatch index/label for the all the given polygons from the image

        Parameters
        ----------
        image: Image
            The image from which is extracted the polygon
        polygons: iterable (subtype: shapely.geometry.Polygon)
            The polygons to dispatch

        Returns
        -------
        index: ndarray (subtype: hashable)
            The dispatch index/label
        """
        return np.array([self.dispatch(image, polygon) for polygon in polygons])

    def dispatch_map(self, image, polygons):
        """Transforms the dispatch indexes/labels returned by dispatch_batch to actual integer indexes defined by the
        mapping. Especially, let there be a mapping array M of size N containing mapped values M[i] (i in [0, N[), a
        polygon P dispatched to index k, then the corresponding index c returned by dispatch_map will be:
            - if k in M, then c will be i where M[i] = k
            - if k not in M, then c will be -1

        If no mapping is defined, then dispatch_map simply forwards the result of dispatch_batch.

        Parameters
        ----------
        image: Image
            The image from which is extracted the polygon
        polygons: iterable (subtype: shapely.geometry.Polygon)
            The polygons to dispatch

        Returns
        -------
        labels: ndarray (subtype: hashable)
            The dispatch indexes/labels
        index: ndarray (subtype: int)
            The mapped dispatch indexes/labels

        Notes
        -----
        If a timing object was provided at construction, it is used to compute the dispatching time
        """
        self.logger.i("Dispatcher: start dispatching.")
        dispatch_labels = self.dispatch_batch(image, polygons)
        self.logger.i("Dispatcher: end dispatching.")

        if self._mapping is None:  # no mapping defined
            return dispatch_labels, dispatch_labels

        # compute mapping indexes
        indexes = np.full(dispatch_labels.shape, -1, dtype=np.int32)
        for label, index in self._mapping.items():
            indexes[dispatch_labels == label] = index

        # report dispatching (TODO is it relevant not to report when the implementer hasn't defined any labels ?)
        not_dispatched = np.equal(indexes, -1)
        dispatched = np.logical_not(not_dispatched)
        dispatched_indexes, counts = np.unique(indexes[dispatched], return_counts=True)
        all_count = len(polygons)

        for index, count in zip(dispatched_indexes, counts):
            label = self._inverse_mapping[index]
            self.logger.i("Dispatcher: {}/{} polygons dispatched to '{}'.".format(count, all_count, label))
        self.logger.w("Dispatcher: {}/{} polygons not dispatched.".format(np.count_nonzero(not_dispatched), all_count))

        return dispatch_labels, indexes


class DispatchingRule(object):
    """An interface to be implemented by any class that defined a dispatching rule for polygons
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def evaluate(self, image, polygon):
        """Evaluate a polygon

        Parameters
        ----------
        image: Image
            The image from which is extracted the polygon
        polygon: shapely.geometry.Polygon
            The polygon to evaluate

        Returns
        -------
        result: bool
            True if the dispatching rule matched the polygon, False otherwise.
        """
        pass

    def evaluate_batch(self, image, polygons):
        """Evaluate the polygons

        Parameters
        ----------
        image: Image
            The image from which is extracted the polygons
        polygons: iterable (subtype: shapely.geometry.Polygon)
            The list of polygons to dispatch

        Returns
        -------
        result: iterable (subtype: bool)
            Iterable of which the ith element is True if the ith polygon is evaluated to True, False otherwise
        """
        return [self.evaluate(image, polygon) for polygon in polygons]


class CatchAllRule(DispatchingRule):
    """A rule which evaluates all the polygons to True"""
    def evaluate(self, image, polygon):
        return True


class RuleBasedDispatcher(Dispatcher):
    """A dispatcher which dispatches polygon evaluating them with dispatching rules"""
    def __init__(self, rules, labels=None, timing=None, logger=SilentLogger()):
        """
        Parameters
        ----------
        rules: iterable (subtype: DispatchingRule)
        labels:
        timing:
        logger:
        """
        labels = labels if labels is not None else list(range(len(rules)))
        super(RuleBasedDispatcher, self).__init__(labels, timing=timing, logger=logger)
        self._rules = rules
        self._labels = labels

    def dispatch(self, image, polygon):
        return self.dispatch_batch(image, [polygon])[0]

    def dispatch_batch(self, image, polygons):
        poly_count = len(polygons)
        dispatch_labels = np.full((poly_count,), None, dtype=object)  # dispatch indexes
        remaining = np.arange(poly_count)  # remaining indexes to process
        # check which rule matched the polygons
        for i, (rule, label) in enumerate(zip(self._rules, self._labels)):
            if remaining.shape[0] == 0:  # if there are no more elements to evaluate
                break
            match, no_match = self._split_by_rule(image, rule, polygons, remaining)
            if len(match) > 0:  # skip if there are no match
                dispatch_labels[match] = label
                remaining = np.setdiff1d(remaining, match, assume_unique=True)
        return dispatch_labels

    @staticmethod
    def _split_by_rule(image, rule, polygons, poly_indexes):
        """Given a rule, splits all the poly_indexes list into two lists. The first list contains
        the indexes corresponding to the polygons that were evaluated True by the rule, the indexes that
        were evaluated False by the rule.

        Parameters
        ----------
        image: Image
            The image from which were extracted the polygons
        rule: DispatchingRule
            The rule with which the polygons must be evaluated
        polygons: iterable
            The list of polygons
        poly_indexes: iterable
            The indexes of the polygons from the list polygons to process
        timing: WorkflowTiming
            The timing object for computing the dispatching time

        Returns
        -------
        true_list: iterable (subtype: int)
            The indexes that were evaluated true
        false_list: iterable (subtype: int)
            The indexes that were evaluated false
        """
        polygons_to_evaluate = take(polygons, poly_indexes)
        eval_results = rule.evaluate_batch(image, polygons_to_evaluate)
        np_indexes = np.array(poly_indexes)
        return np_indexes[np.where(eval_results)], np_indexes[np.where(np.logical_not(eval_results))]


def default_fail_callback(polygon):
    """The default fail callback which associates None to assessed polygon"""
    return None


class DispatcherClassifier(Loggable):
    """A dispatcher classifier is an object that evaluates a set of polygons extracted from an
    image. It first dispatches the polygons using a Dispatcher and then classifies the polygons using some classifiers.
    Especially, the polygons are classified by the classifier they were dispatched to.
    """

    TIMING_DISPATCH = "dispatch"
    TIMING_CLASSIFY = "classify"

    def __init__(self, dispatcher, classifiers, logger=SilentLogger()):
        """Constructor for ClassifierDispatcher object

        Parameters
        ----------
        dispatcher: Dispatcher
            An dispatcher object to dispatch polygons to their most appropriate classifiers. Should produce as many
            dispatch indexes/labels as there are classifiers (i.e. N).
        classifiers: iterable (subtype: PolygonClassifiers, size: N)
            An iterable of polygon classifiers associated with the rules.
        """
        Loggable.__init__(self, logger)
        self._dispatcher = dispatcher
        self._classifiers = classifiers

    def dispatch_classify(self, image, polygon, timing_root=None):
        """Dispatch a single polygon to its corresponding classifier according to the dispatching rules,
        then compute and return the associated prediction.

        Parameters
        ----------
        image: Image
            The image to which belongs the polygon
        polygon: shapely.geometry.Polygon
            The polygon of which the class must be predicted
        timing_root: str
            A root phase for the inner timing object

        Returns
        -------
        prediction: int|None
            An integer code indicating the predicted class.
            If none of the dispatching rules matched a polygon, the value returned is the one produced
            by the fail callback for the given polygon. Especially, if no fail callback was registered,
            None is returned.
        probability: float
            The probability associated with the prediction (0.0 if the polygon wasn't dispatched)
        dispatch: int
            The identifier of the rule that matched the polygon (see dispatch_classify_batch)
        timing: WorkflowTiming
            The timing object containing times of the different dispatch/classify phases
        """
        classes, probabilities, dispatches, timing = self.dispatch_classify_batch(image, [polygon], timing_root=timing_root)
        return classes[0], probabilities[0], dispatches[0], timing

    def dispatch_classify_batch(self, image, polygons, timing_root=None):
        """Apply the dispatching and classification steps to an ensemble of polygons.

        Parameters
        ----------
        image: Image
            The image to which belongs the polygon
        polygons: iterable (subtype: shapely.geometry.Polygon, size: N)
            The polygons of which the classes must be predicted
        timing_root: str
            A root phase for the inner timing object
            
        Returns
        -------
        predictions: iterable (subtype: int|None, size: N)
            A list of integer codes indicating the predicted classes.
            If none of the dispatching rules matched the polygon, the prediction associated with it is the one produced
            by the fail callback for the given polygon. Especially, if no fail callback was registered, None is
            returned.
        probabilities: iterable (subtype: float, range: [0,1], size: N)
            The probabilities associated with each predicted classes (0.0 for polygons that were not dispatched)
        dispatches: iterable (size: N)
            An iterable containing the identifiers of the rule that matched the polygons. If dispatching labels were
            provided at construction, those are used to identify the rules. Otherwise, the integer indexes of the rules
            in the list provided at construction are used. Polygons that weren't matched by any rule are returned -1 as
            dispatch index.
        timing: WorkflowTiming
            The timing object containing times of the different dispatch/classify phases
        """
        timing = WorkflowTiming(root=timing_root)
        # dispatch
        timing.start(DispatcherClassifier.TIMING_DISPATCH)
        disp_labels, disp_indexes = self._dispatcher.dispatch_map(image, polygons)
        timing.end(DispatcherClassifier.TIMING_DISPATCH)

        # filter not dispatched
        unique_disp_indexes, first_occurs = np.unique(disp_indexes, return_index=True)

        # classify
        poly_count = len(polygons)
        predictions = np.full((poly_count,), None, dtype=object)
        probabilities = np.full((poly_count,), 0.0, dtype=np.float32)
        np_polygons = shape_array(polygons)

        self.logger.info("DispatcherClassifier: start classification.")
        for index, first in zip(unique_disp_indexes, first_occurs):
            if index == -1:  # not dispatched
                continue
            curr_disp_idx = (disp_indexes == index)  # indexes of the currently processed polygons
            # predicts classes
            timing.start(DispatcherClassifier.TIMING_CLASSIFY)
            pred, proba = self._classifiers[index].predict_batch(image, np_polygons[curr_disp_idx])
            timing.end(DispatcherClassifier.TIMING_CLASSIFY)
            # save results
            predictions[curr_disp_idx] = pred
            probabilities[curr_disp_idx] = proba
        self.logger.info("DispatcherClassifier: end classification.")
        return list(predictions), list(probabilities), list(disp_labels), timing
