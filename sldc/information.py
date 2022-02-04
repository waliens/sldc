# -*- coding: utf-8 -*-
from copy import copy
from collections import namedtuple

import numpy as np

from .util import shape_array
from .timing import WorkflowTiming, merge_timings


__author__ = "Romain Mormont <romainmormont@hotmail.com>"
__version__ = "0.1"


class WorkflowInformation(object):
    """Workflow information classes (immutable)"""

    DATA_FIELD_POLYGONS = "polygons"
    DATA_FIELD_LABELS = "labels"
    TUPLE_FIELD_POLYGONS = "polygon"
    TUPLE_FIELD_LABELS = "label"

    def __init__(self, polygons, labels, timing, **kwargs):
        """Constructor
        Parameters
        ----------
        polygons: ndarray (subtype: Polygon, size: n)
            The detected objects
        labels: ndarray (subtype: int, size: n)
            Their associated labels
        timing: WorkflowTiming 
            The workflow execution timing information
        **kwargs: 
            Any other information gathered during execution. Each key must be the (pluralized) name of the data. Each 
            value is a tuple of which the first element is a ndarray of size n containing the items for this 
            information and the second element is the the singularized name (key singularized).
        
        Raises
        ------
        ValueError: if data size is invalid or if the passed names clash with class fields
        """
        self._data = {key: np.asarray(value[0]) for key, value in kwargs.items()}
        self._data[self.DATA_FIELD_POLYGONS] = shape_array(polygons)
        self._data[self.DATA_FIELD_LABELS] = np.array(labels)
        self._timing = timing

        # check arrays sizes
        curr_size = None
        for data_name, data_array in self._data.items():
            if curr_size is not None and curr_size != len(data_array):
                raise ValueError(
                    "All information vectors should have the same size. Field '"
                    "{}' has size {} while other(s) have {}.".format(data_name, len(data_array), curr_size)
                )
            curr_size = len(data_array)

        # add data fields as properties
        data_fields = set(kwargs.keys()).intersection(dir(self.__class__))
        if len(data_fields) > 0:
            raise ValueError("Some data fields clash with some class fields '{}'.".format(data_fields))

        self._tuple_type_fields = [self.TUPLE_FIELD_POLYGONS, self.TUPLE_FIELD_LABELS] + [v[1] for v in kwargs.values()]
        self._tuple_type = self._get_namedtuple(self._tuple_type_fields)
        self._fields = [self.DATA_FIELD_POLYGONS, self.DATA_FIELD_LABELS] + list(kwargs.keys())
        for field in kwargs.keys():
            self.__dict__[field] = self._data[field]

    @property
    def polygons(self):
        return self._data[self.DATA_FIELD_POLYGONS]

    @property
    def labels(self):
        return self._data[self.DATA_FIELD_LABELS]

    @property
    def timing(self):
        return self._timing

    @property
    def fields(self):
        return self._fields

    def __getitem__(self, item):
        return self._tuple_type(**{
            tuple_field: self._data[data_field][item]
            for data_field, tuple_field
            in zip(self._fields, self._tuple_type_fields)
        })

    def __len__(self):
        return len(self._data[self.DATA_FIELD_POLYGONS])

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def merge(self, other):
        """Merge the current information object with other into a new object"""
        self._is_compatible(other)
        other_fields = dict()
        for field, tuple_field in zip(self.fields, self._tuple_type_fields):
            if field == self.DATA_FIELD_POLYGONS or field == self.DATA_FIELD_LABELS :
                continue
            other_fields[field] = (
                np.concatenate((self._data[field], other._data[field])),
                tuple_field
            )
        return WorkflowInformation(
            polygons=np.concatenate((self.polygons, other.polygons)),
            labels=np.concatenate((self.labels, other.labels)),
            timing=merge_timings(self.timing, other.timing),
            **other_fields
        )

    def _is_compatible(self, other):
        """Check whether the other object is compatible for merging with the current one"""
        if not isinstance(other, WorkflowInformation):
            raise TypeError("'other' should be a WorkflowInformation object (actual type is '{}').".format(type(other)))
        other_fields = set(other.fields)
        if len(other_fields.intersection(self.fields)) != len(self.fields):
            raise ValueError("Fields in other information object are different.")
        other_tuple_fields = set(other._tuple_type_fields)
        if len(other_tuple_fields.intersection(self._tuple_type_fields)) != len(self._tuple_type_fields):
            raise ValueError("Tuple namefFields in other information object are different.")

    @classmethod
    def _get_namedtuple(cls, fields):
        return namedtuple('object_info', fields)

    def __getstate__(self):  # to make it serializable
        d = copy(self.__dict__)
        d["_tuple_type"] = None
        return d

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._tuple_type = self._get_namedtuple(self._tuple_type_fields)


def merge_information(info1, info2):
    if not isinstance(info1, WorkflowInformation):
        raise TypeError("The first object is not a WorkflowInformation object (actual type: {})".format(type(info1)))
    return info1.merge(info2)


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

    @property
    def info_labels(self):
        """Return the registered workflow information labels"""
        return self._order

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

    def __iter__(self):
        for label in self._order:
            yield label, self._infos[label]

    @property
    def polygons(self):
        """Return all the found polygons"""
        return shape_array([p for info_label in self._order for p in self._infos[info_label].polygons])

    @property
    def labels(self):
        """Return all the found labels"""
        return np.array([l for info_label in self._order for l in self._infos[info_label].labels])
