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
            return ".".join([self._root, phase])
        else:
            return phase

    @classmethod
    def _validate_phase(cls, phase):
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

    def cm(self, phase):
        """Returns a context manager for computing time for a given phase"""
        return TimingContextManager(self, phase)

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
            raise KeyError("Unknown phase '{}'.".format(full_phase))
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
        if other is self:
            return
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
        return self._simplify_hierarchy(self._get_phases_hierarchy(splitted_keys), root=None)

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

    def _simplify_hierarchy(self, hierarchy, root=None):
        if hierarchy is None:
            return None
        simplified = dict()
        for key, value in hierarchy.items():
            full_phase = ".".join([root, key]) if root is not None else key
            if full_phase not in self and value is not None and len(value) == 1:
                subkey = list(value.keys())[0]
                simplified[".".join([key, subkey])] = self._simplify_hierarchy(value[subkey], root=full_phase)
            else:
                simplified[key] = self._simplify_hierarchy(value, root=full_phase)
        return simplified

    def get_phase_statistics(self, phase):
        """Return time statistics in seconds for the given phase
        Parameters
        ----------
        phase: str
            The phase identifier
        
        Returns
        -------
        statistics: dict    
            A dictionary mapping statistic name (among 'max', 'min', 'mean', 'std', 'sum' and 'count') with their 
            respective values
        """
        if phase not in self:
            raise KeyError("Unknown phase '{}'.".format(phase))
        times = self[phase]
        return {
            "count": times.shape[0],
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "sum": np.sum(times)
        }


class TimingContextManager(object):
    def __init__(self, timing, phase):
        """A context manager for computing a duration for a given phase"""
        self._timing = timing
        self._phase = phase

    def __enter__(self):
        self._timing.start(self._phase)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timing.end(self._phase)


def merge_timings(timing1, timing2):
    """Merge timing into a new timing object. The passed objects are not modified.

    Parameters
    ----------
    timing1: WorkflowTiming
        First timing to merge
    timing2: WorkflowTiming
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


def _report_timing(hierarchy, parent_phase, count, timing, logger, indent="  "):
    if hierarchy is None:
        return
    for phase, sub_hierarchy in hierarchy.items():
        full_phase = "{}.{}".format(parent_phase, phase) if len(parent_phase) > 0 else phase
        new_count = count
        if full_phase in timing:
            stats = timing.get_phase_statistics(full_phase)
            logger.i("{}* {}: {}s (mean:{}s std:{}s min:{}s max:{}s, count:{})".format(
                indent * count,
                phase,
                np.round(stats["sum"], 5),
                np.round(stats["mean"], 5),
                np.round(stats["std"], 5),
                np.round(stats["min"], 5),
                np.round(stats["max"], 5),
                stats["count"]
            ))
            new_count += 1
        _report_timing(sub_hierarchy, full_phase, new_count, timing, logger, indent=indent)


def report_timing(timing, logger, indent="  "):
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

