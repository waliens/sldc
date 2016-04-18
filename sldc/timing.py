# -*- coding: utf-8 -*-

import datetime
import numpy as np

__author__ = "Mormont Romain <romain.mormont@gmail.com>"
__version__ = "0.1"


class WorkflowTiming(object):

    FETCHING = "fetching"
    SEGMENTATION = "segmentation"
    MERGING = "merging"
    LOCATION = "location"
    DISPATCH_CLASSIFY = "dispatch_classify"

    def __init__(self):
        self._durations = {
            WorkflowTiming.FETCHING: [],
            WorkflowTiming.SEGMENTATION: [],
            WorkflowTiming.MERGING: [],
            WorkflowTiming.LOCATION: [],
            WorkflowTiming.DISPATCH_CLASSIFY: []
        }
        self._start_dict = dict()

    def start_fetching(self):
        self._record_start(WorkflowTiming.FETCHING)

    def end_fetching(self):
        self._record_end(WorkflowTiming.FETCHING)

    def start_segment(self):
        self._record_start(WorkflowTiming.SEGMENTATION)

    def end_segment(self):
        self._record_end(WorkflowTiming.SEGMENTATION)

    def start_location(self):
        self._record_start(WorkflowTiming.LOCATION)

    def end_location(self):
        self._record_end(WorkflowTiming.LOCATION)

    def start_dispatch_classify(self):
        self._record_start(WorkflowTiming.DISPATCH_CLASSIFY)

    def end_dispatch_classify(self):
        self._record_end(WorkflowTiming.DISPATCH_CLASSIFY)

    def start_merging(self):
        self._record_start(WorkflowTiming.MERGING)

    def end_merging(self):
        self._record_end(WorkflowTiming.MERGING)

    def stats(self):
        stats = dict()
        for key in self._durations.keys():
            stats[key] = self._stat_tuple(key)
        return stats

    def _record_start(self, code):
        self._start_dict[code] = datetime.datetime.now()

    def _record_end(self, code):
        start = self._start_dict.get(code)
        if start is not None:
            self._durations[code].append((datetime.datetime.now() - start).total_seconds())
            del self._start_dict[code]

    def _stat_tuple(self, code):
        """
        :param code:
        :return: (min, max, mean, count)
        """
        durations = np.array(self._durations[code])
        count = durations.shape[0]
        if count == 0:
            return 0, 0, 0, 0, 0
        return round(np.sum(durations), 5), round(np.min(durations), 5), round(np.mean(durations), 5), round(np.max(durations), 5), count

    def report(self, image, polygons_classes):
        print "========================================"
        print "Image {}".format(str(image))
        print "Polygon count : {}".format(len(polygons_classes))
        print "Timing : "
        stats = self.stats()
        for key in stats.keys():
            print "- {} : {}".format(key, stats[key])
