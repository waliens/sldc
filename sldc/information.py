# -*- coding: utf-8 -*-
import os

import numpy as np

from .timing import WorkflowTiming

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version__ = "0.1"


class ChainInformation(object):
    """ A class for storing information gathered at various stages of execution of a workflow chain.
    """
    def __init__(self):
        self._order = []
        self._infos = dict()

    def __len__(self):
        return len(self._order)

    def __setitem__(self, key, value):
        """Set a workflow information object with the given label. If the label already exists, the previously
        recorded workflow information is overwritten.

        Parameters
        ----------
        key: hashable
            The label of the workflow that generated the
        value: WorkflowInformation
            The workflow information
        """
        if key not in self._infos:
            self._order.append(key)
        self._infos[key] = value

    def __getitem__(self, item):
        """Get the workflow information associated with the given label

        Parameters
        ----------
        item: hashable
        :return:
        """
        return self._infos[item]

    def all_information(self):
        """Yields all the registered workflow information objects

        Yields
        ------
        label: hashable
            The label associated with the workflow which generated the information object
        information: WorkflowInformation
            A workflow information
        """
        for key in self._order:
            yield key, self._infos[key]

    def information(self, label):
        """Get the information generated by the workflow associated with the given label

        Parameters
        ----------
        label: hashable
            The label

        Returns
        -------
        information: WorkflowInformation
            TheA workflow information
        """
        return self[label]

    def append(self, label, information):
        """Append a new workflow information object to the chain information (if the label already exist, the
        the corresponding

        Parameters
        ----------
        label: hashable
            The label of the workflow which generated the workflow information object
        information: WorkflowInformation
            The workflow information object
        """
        self[label] = information

    def polygons(self):
        """Yields all polygons stored in the chain information object

        Yields
        ------
        label: hashable
            The label identifying the workflow which generated the polygon
        polygon: shapely.geometry.Polygon
            A polygon
        dispatch: hashable
            The label of the dispatching rule which dispatched the polygon
        class: int
            The class predicted for this polygon
        probability: float
            The probability estimates for predicted class
        """
        for key, information in self.all_information():
            for polygon, dispatch, cls, proba in information:
                yield key, polygon, dispatch, cls, proba

    def __iter__(self):
        for k, p, d, c, r in self.polygons():
            yield k, p, d, c, r

    def report(self, logger):
        # count objects
        object_count = 0
        discarded_count = 0
        for _, _, dispatch, _, _ in self:
            if dispatch != -1:
                object_count += 1
            else:
                discarded_count += 1

        logger.i("{}Workflow chain report: ".format(os.linesep))
        logger.i("  Objects found      : {}".format(discarded_count + object_count))
        logger.i("  Objects dispatched : {}".format(object_count))
        logger.i("  Objects discarded  : {}".format(discarded_count))

        for name in self._order:
            logger.i("{}Workflow '{}': ".format(os.linesep, name))
            self._infos[name].report(logger)


class WorkflowInformation(object):
    """Workflow information : execution about a workflow run. A run is a complete execution
    (segment, locate, dispatch and classify) of a single workflow and comprises the following information:
        - polygons : the polygons generated by a given run
        - dispatch : list of which the ith element matches the identifier of the dispatching rule that matched
            the ith polygon (either an integer or a user defined identifier), -1 if none did
        - class : list of which the ith integer is the class predicted by the classifier to
            which was dispatched the ith polygon, -1 if none did
        - probas : list of which the ith float (in [0,1]) is the probability of the ith predicted class if the
            corresponding polygon was dispatched, 0 if it wasn't dispatched
        - timing : the information about the execution time of the workflow
    """
    def __init__(self, polygons, dispatch, classes, probas, timing):
        """Construct a run object
        Parameters
        ----------
        polygons: iterable (type: shapely.geometry.Polygon, size: N)
            The polygons generated by the run
        dispatch: iterable (size: N)
            The identifiers of the dispatching rules that matched the polygons
        classes: iterable (type: int, size: N)
            Their predicted classes
        probas: iterable (type: float, size: N)
            The probabilities of the predicted classes
        timing: SLDCTiming
            Execution time information
        """
        self._polygons = polygons
        self._dispatch = dispatch
        self._classes = classes
        self._timing = timing
        self._probas = probas

    @property
    def polygons(self):
        return self._polygons

    @property
    def dispatch(self):
        return self._dispatch

    @property
    def classes(self):
        return self._classes

    @property
    def probas(self):
        return self._probas

    @property
    def metadata(self):
        return self._metadata

    @property
    def timing(self):
        return self._timing

    def __len__(self):
        return len(self.polygons)

    def __iter__(self):
        for polygon, dispatch, cls, proba in self.results():
            yield polygon, dispatch, cls, proba

    def results(self, filter_dispatch=None, filter_classes=None):
        """Yields an iterator that goes through the list of polygons of the workflow information
        The result is a tuple containing in this order the polygon, the dispatch index and the class

        Parameters
        ----------
        filter_dispatch: iterable (optional, default: [-1])
            The dispatching rule identifiers to exclude from the iterated list
        filter_classes: iterable (subtype: int, optional, default: [])
            The classes to exclude from the iterated list
        """
        if filter_dispatch is None:
            filter_dispatch = [-1]
        for polygon, dispatch, cls, proba in zip(self.polygons, self.dispatch, self.classes, self.probas):
            if (filter_dispatch is None or dispatch not in filter_dispatch) and \
                    (filter_classes is None or cls not in filter_classes):
                yield polygon, dispatch, cls, proba

    def merge(self, other):
        """Merge the other workflow information object into the current one. The id and metadata of the first are kept
        if they are set. Otherwise, the metadata and id of the other object are also merged

        Parameters
        ----------
        other: WorkflowInformation
            The workflow information object to merge
        """
        if other is None:
            return

        self._polygons += other.polygons
        self._dispatch += other.dispatch
        self._classes += other.classes
        self._probas += other.probas
        self._timing = WorkflowTiming.merge_timings(self._timing, other.timing)

    def report(self, logger, indent_count=2):
        """Compute and print a bunch of stats about the execution of this workflow"""
        # pre compute some metrics
        indent = " " * indent_count
        space = "  "
        total = len(self._polygons)
        dispatched = len([d for d in self._dispatch if d != -1])

        # summary
        report = "{}Total objects found      : {}".format(indent, total)
        report += "{}{}Total objects dispatched : {}".format(os.linesep, indent, dispatched)
        report += "{}{}Total objects discarded  : {}".format(os.linesep, indent, total - dispatched)
        report += os.linesep

        # break down objects stats per dispatching label
        classes = np.array(self._classes)
        dispatch = np.array(self._dispatch)
        labels, counts = np.unique(dispatch, return_counts=True)
        report += "{}{}Per dispatching rule :".format(os.linesep, indent)
        for l, c in zip(labels, counts):
            if l == -1:
                report += "{}{}{}discarded: {}".format(os.linesep, indent, space, c)
            else:
                ind = np.where(dispatch == l)
                report += "{}{}{}'{}': {} ".format(os.linesep, indent, space, l, c)

                # report classes distrib
                unique_classes, classes_count = np.unique(classes[ind], return_counts=True)
                cls_rep = ", ".join(["class '{}': {}".format(cls, count) for cls, count in zip(unique_classes, classes_count)])
                report += "({})".format(cls_rep)

        logger.i(report)

        # display times
        logger.i("{}{}Execution times:".format(os.linesep, indent))
        self._timing.report(logger, indent_count=indent_count)
