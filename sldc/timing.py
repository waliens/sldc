# -*- coding: utf-8 -*-

import os
import timeit

import numpy as np

from .logging import Logger

__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version = "0.1"


class WorkflowTiming(dict):
    """A class that computes and stores execution times for various phases of the workflow.
    
    The timing object compute times on a per-phase basis. The phases are chosen by the caller and must
    be strings. To indicate sub-phases, the caller can use dot-notation. For instance, in "segment.cut", "cut" is 
    a sub-phase of "segment". 
    """

    def __init__(self, root=None):
        """
        Parameters
        ----------
        root: str (default: None)
            A root phase to prepend to any phase passed to the timing object. Set to None for not preprending anything.
        
        Raises
        ------
        ValueError: if the root phase is an invalid phase
        """
        super(WorkflowTiming, self).__init__()
        if root is not None:
            self._validate_phase(root)
        self._starts = dict()
        self._root = root

    def _full_phase(self, phase):
        """Return the full phase (user phase with prepended root phase)"""
        if self._root is not None:
            return "{}.{}".format(self._root, phase)
        else:
            return phase

    @classmethod
    def _validate_phase(cls, phase: str):
        """Validate the phase identifier 
        Parameters
        ----------
        phase: str
            The phase identifier to validate.
        
        Raises
        ------
        ValueError: if the phase is invalid (starts or ends with a dot or contains consecutive dots)
        """
        if phase is None or len(phase) == 0 or phase.startswith(".") or phase.endswith(".") or ".." in phase:
            raise ValueError("Invalid phase identifier ''.".format(phase))

    def start(self, phase):
        """Register the start time of the given phase
        Parameters
        ----------
        phase: str
            The phase for which to record the start time
        
        Raises
        ------
        ValueError: if the phase is invalid
        """
        WorkflowTiming._validate_phase(phase)
        full_phase = self._full_phase(phase)
        self._starts[full_phase] = timeit.default_timer()

    def end(self, phase):
        """Register the end time of the given phase
        Parameters
        ----------
        phase: str
            The phase for which to record the end time
        
        Raises
        ------
        ValueError: if the start time for this phase wasn't recorded
        """
        self._validate_phase(phase)
        full_phase = self._full_phase(phase)
        if full_phase not in self._starts:
            raise KeyError("No start was recorded for the phase '{}' (root:{})".format(full_phase, self._root))
        start = self._starts.pop(full_phase)
        duration = timeit.default_timer() - start
        if full_phase not in self:
            self[full_phase] = np.array([duration])
        else:
            self[full_phase] = np.concatenate((self[full_phase], [duration]))

    def total(self, phase):
        """Total duration of the given phase
        Parameters
        ----------
        phase: str  
            The phase to compute the duration
        
        Returns
        -------
        duration: float 
            Duration of the phase
        """
        self._validate_phase(phase)
        full_phase = self._full_phase(phase)
        if full_phase not in self:
            raise ValueError("Unknown phase '{}'.".format(full_phase))
        return np.sum(self[full_phase])

    def merge(self, other):
        """Merge two workflow timing objects. The recorded times for same phases are registered together.
        
        Parameters
        ----------
        other: WorkflowTiming
            The workflow timing to merge in the current one
        """
        if not isinstance(other, WorkflowTiming):
            raise TypeError("The `other` parameter should be a workflow timing object, object of type `{}` found.".format(type(other)))
        for phase, times in other.items():
            curr_times = self.get(phase, None)
            if curr_times is None:
                self[phase] = times
            else:
                self[phase] = np.concatenate((self.get(phase), times))

    def get_phases_hierarchy(self):
        """Return the hierarchy of phases"""
        keys = sorted(self.keys())
        splitted_keys = [k.split(".") for k in keys]
        return self._get_phases_hierarchy(splitted_keys)

    @classmethod
    def _get_phases_hierarchy(cls, phases):
        # create top level hierarchy
        hierarchy = {phase[0]: None for phase in phases if len(phase) >= 1}
        # extract non-final hierarchies and group according to top level phase
        splitted = [(phase[0], phase[1:]) for phase in phases if len(phase) > 1]
        non_final = dict()
        for top_phase, subphases in splitted:
            non_final[top_phase] = non_final.get(top_phase, []) + [subphases]
        # build final hierarchy
        for top_phase, subphases in non_final.items():
            hierarchy[top_phase] = cls._get_phases_hierarchy(subphases)
        return hierarchy

    def get_phase_statistics(self, phase):
        """Return time statistics in seconds for the given phase
        Parameters
        ----------
        phase: str
            The phase identifier
        
        Returns
        -------
        statistics: dict    
            A dictionary mapping statistic name (among 'max', 'min', 'mean', 'std' and 'count) with their respective 
            values
        """
        self._validate_phase(phase)
        full_phase = self._full_phase(phase)
        if full_phase not in self:
            raise ValueError("Unknown phase '{}'.".format(full_phase))
        times = self[full_phase]
        return {
            "count": times.shape[0],
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times)
        }


def _report_timing(hierarchy: dict, parent_phase: str, count: int, timing: WorkflowTiming, logger: Logger, indent="  "):
    if hierarchy is None:
        return
    for phase, sub_hierarchy in hierarchy.items():
        full_phase = "{}.{}".format(parent_phase, phase) if len(parent_phase) > 0 else phase
        stats = timing.get_phase_statistics(full_phase)
        logger.i("{}* {} -> mean:{} std:{} min:{} max:{} count:{}".format(
            indent * count,
            phase,
            np.round(stats["mean"], 5),
            np.round(stats["std"], 5),
            np.round(stats["min"], 5),
            np.round(stats["max"], 5),
            stats["count"]
        ))
        _report_timing(sub_hierarchy, full_phase, count + 1, timing,  logger, indent=indent)


def merge_timings(timing1: WorkflowTiming, timing2: WorkflowTiming):
    """Merge timing into a new timing object. The passed objects are not modified.

    Parameters
    ----------
    timing1: WorkflowTiming;
        First timing to merge
    timing2: WorkflowTiming;
        Second timing to merge

    Return
    ------
    merged: WorkflowTiming
        The merged workflow timing
    """
    new_timing = WorkflowTiming()
    if timing1 is not None:
        new_timing.merge(timing1)
    if timing2 is not None:
        new_timing.merge(timing2)
    return new_timing


def report_timing(timing: WorkflowTiming, logger: Logger, indent="  "):
    """Report the recorded times in the given workflow timing object
    Parameters
    ----------
    timing: WorkflowTiming
        The timing containing to times to report
    logger: Logger
        The logger through which the times must be reported
    indent: str
        Indent to use to separate the different phase levels
    """
    logger.i("Timing report:")
    _report_timing(timing.get_phases_hierarchy(), "", 0, timing, logger, indent=indent)


# class WorkflowTimings(object):
#     """A class that computes and stores execution times for various phases of the workflow.
#     WorkflowTiming objects can be combined (their stored execution times are added)
#
#     Class constants:
#         - LOADING : Time required for loading image into memory (call of tile.np_image)
#         - SEGMENTATION : Time for segmenting the tiles (call of segmenter.segment)
#         - MERGING : Time for merging the polygons found in the tiles (call of merger.merge)
#         - LOCATION : Time for locating the polygons in the segmented tiles (call of locator.locate)
#         - DISPATCH : Time for dispatching the polygons (call of rule.evaluate_batch or rule.evaluate)
#         - CLASSIFY : Time for classifying the polygons (call of polygon_classifier.predict_batch or
#                      polygon_classifier.predict)
#         - FSL : Total time for executing the fetch/segment/locate (same as FETCHING + SEGMENTATION + MERGING in case of
#                 sequential execution. Less then these in case of parallel execution)
#     """
#
#     LOADING = "loading"
#     SEGMENTATION = "segmentation"
#     MERGING = "merging"
#     LOCATION = "location"
#     DISPATCH = "dispatch"
#     CLASSIFY = "classify"
#     LSL = "load_segment_locate"
#     DC = "dispatch_classify"
#
#     def __init__(self):
#         """Construct a WorkflowTiming object
#         """
#         self._durations = {
#             WorkflowTiming.LOADING: [],
#             WorkflowTiming.SEGMENTATION: [],
#             WorkflowTiming.MERGING: [],
#             WorkflowTiming.LOCATION: [],
#             WorkflowTiming.DISPATCH: [],
#             WorkflowTiming.CLASSIFY: [],
#             WorkflowTiming.LSL: [],
#             WorkflowTiming.DC: []
#         }
#         self._start_dict = dict()
#
#     def start_loading(self):
#         """Record the start for the 'loading' phase
#         """
#         self._record_start(WorkflowTiming.LOADING)
#
#     def end_loading(self):
#         """Record the end for the 'loading' phase
#         """
#         self._record_end(WorkflowTiming.LOADING)
#
#     def start_segment(self):
#         """Record the start for the 'segment' phase
#         """
#         self._record_start(WorkflowTiming.SEGMENTATION)
#
#     def end_segment(self):
#         """Record the end for the 'segment' phase
#         """
#         self._record_end(WorkflowTiming.SEGMENTATION)
#
#     def start_location(self):
#         """Record the start for the 'location' phase
#         """
#         self._record_start(WorkflowTiming.LOCATION)
#
#     def end_location(self):
#         """Record the end for the 'location' phase
#         """
#         self._record_end(WorkflowTiming.LOCATION)
#
#     def start_dispatch(self):
#         """Record the start for the 'dispatch' phase
#         """
#         self._record_start(WorkflowTiming.DISPATCH)
#
#     def end_dispatch(self):
#         """Record the end for the 'dispatch' phase
#         """
#         self._record_end(WorkflowTiming.DISPATCH)
#
#     def start_classify(self):
#         """Record the start for the 'classify' phase
#         """
#         self._record_start(WorkflowTiming.CLASSIFY)
#
#     def end_classify(self):
#         """Record the end for the 'classify' phase
#         """
#         self._record_end(WorkflowTiming.CLASSIFY)
#
#     def start_merging(self):
#         """Record the start for the 'merging' phase
#         """
#         self._record_start(WorkflowTiming.MERGING)
#
#     def end_merging(self):
#         """Record the end for the 'merging' phase
#         """
#         self._record_end(WorkflowTiming.MERGING)
#
#     def start_lsl(self):
#         """Record the start for the 'load_segment_locate' phase
#         """
#         self._record_start(WorkflowTiming.LSL)
#
#     def end_lsl(self):
#         """Record the end for the 'load_segment_locate' phase
#         """
#         self._record_end(WorkflowTiming.LSL)
#
#     def start_dc(self):
#         """Record the start for the 'dispatch_classify' phase
#         """
#         self._record_start(WorkflowTiming.DC)
#
#     def end_dc(self):
#         """Record the end for the 'dispatch_classify' phase
#         """
#         self._record_end(WorkflowTiming.DC)
#
#     def statistics(self):
#         """Compute time statistics tuples for each phase of the algorithm
#         Returns
#         -------
#         stats: dict
#             A dictionary mapping phase string with a stat tuple containing time statistics for the given phase
#         """
#         stats = dict()
#         for key in self._durations.keys():
#             stats[key] = self._stat_tuple(key)
#         return stats
#
#     def total(self):
#         """Compute the total execution times of the algorithm recorded so far
#         """
#         total_time = 0
#         for key in self._durations.keys():
#             total_time += sum(self._durations[key])
#         return total_time
#
#     def sl_total_duration(self):
#         """Return the total execution time for segmenting tiles and locating polygons recoreded so far
#         Returns
#         -------
#         time: float
#             The execution time in second
#         """
#         return self.total_duration_of([WorkflowTiming.SEGMENTATION, WorkflowTiming.LOCATION])
#
#     def lsl_total_duration(self):
#         """Return the total execution time for the loading, segment and locate phase
#         Returns
#         -------
#         time: float
#             The execution time in second
#         """
#         return self.total_duration_of([WorkflowTiming.LSL])
#
#     def dc_total_duration(self):
#         """Return the total execution time for dispatching and classifying polygons recoreded so far
#
#         Returns
#         -------
#         time: float
#             The execution time in second
#         """
#         return self.total_duration_of([WorkflowTiming.DC])
#
#     def duration_of(self, phase):
#         """Return the total duration of the given phase
#         Parameters
#         ----------
#         phase: string
#             The phase string
#
#         Returns
#         -------
#         time: float
#             Total time in seconds
#         """
#         if phase not in self._durations:
#             return 0
#         return sum(self._durations[phase])
#
#     def total_duration_of(self, phases):
#         """Return the total dûration of the given phases
#         Parameters
#         ----------
#         phases: iterable (subtype: string)
#             Iterable containing the strings of the phases to be included in the computed times
#         Returns
#         -------
#         time: float
#             Total time in seconds
#         """
#         if len(phases) == 0:
#             return 0
#         return sum([self.duration_of(phase) for phase in phases])
#
#     def _record_start(self, phase):
#         """Record a start for a given phase
#         Parameters
#         ----------
#         phase: string
#             The string of the phase that starts
#         """
#         self._start_dict[phase] = timeit.default_timer()
#
#     def _record_end(self, phase):
#         """Record an end for a given phase
#         Parameters
#         ----------
#         phase: string
#             The string of the phase that ends
#         """
#         start = self._start_dict.get(phase)
#         if start is not None:
#             self._durations[phase].append(timeit.default_timer() - start)
#             del self._start_dict[phase]
#
#     def _stat_tuple(self, phase):
#         """Make a statistics tuple from the given phase string
#         Parameters
#         ----------
#         phase: string
#             The phase string of which statistics tuple is wanted
#         Returns
#         -------
#         stats: tuple of (float, float, float, float, float, float)
#             Tuple containing the following stats (sum, min, mean, max, std, count)
#         """
#         durations = np.array(self._durations[phase])
#         count = durations.shape[0]
#         if count == 0:
#             return 0, 0, 0, 0, 0
#         return round(np.sum(durations), 5), \
#             round(np.min(durations), 5), \
#             round(np.mean(durations), 5), \
#             round(np.max(durations), 5), \
#             round(np.std(durations), 5), \
#             count
#
#     @classmethod
#     def merge_timings(cls, timing1, timing2):
#         """Merge the two timings into a new timing object
#         Parameters
#         ----------
#         timing1: WorkflowTiming
#             The first timing object to merge
#         timing2: WorkflowTiming
#             The second timing object to merge
#         Returns
#         -------
#         timing: WorkflowTiming
#             A new timing object containing the merging of the passed timings
#         """
#         if timing1 is None and timing2 is None:
#             return WorkflowTiming()
#         elif timing1 is None or timing2 is None:
#             return timing1 if timing1 is not None else timing2
#
#         timing = WorkflowTiming()
#         for key in timing._durations.keys():
#             timing._durations[key] = timing1._durations.get(key, []) + timing2._durations.get(key, [])
#         return timing
#
#     def merge(self, other):
#         """Merge the other WorkflowTiming into the current one
#
#         Parameters
#         ----------
#         other: WorkflowTiming
#             The WorkflowTiming object to merge
#         """
#         if other is None or not isinstance(other, WorkflowTiming):
#             return
#         for key in self._durations.keys():
#             self._durations[key] += other._durations.get(key, [])
#
#     def report(self, logger, indent_count=2):
#         """Report the execution times of the workflow phases using the given logger
#         Parameters
#         ----------
#         logger: Logger
#             The logger to which the times must be notified
#         """
#         indent = " " * indent_count
#         space = "  "
#         to_report = list()
#         stats = self.statistics()
#         for key in stats.keys():
#             curr_stat = stats[key]
#             to_report.append("{}{}{} : {} s (avg: {} s, std: {} s)".format(indent, space, key, curr_stat[0], curr_stat[2], curr_stat[4]))
#         logger.info(os.linesep.join(to_report))
