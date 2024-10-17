# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib
import logging
import os
from abc import ABC
from abc import abstractmethod

import earthkit.data as ekd
import entrypoints

from anemoi.transform.grouping import GroupByMarsParam

LOG = logging.getLogger(__name__)


class Filter(ABC):

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __call__(self, *args, **kwargs):
        # This is a convenience method to call the forward method
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, x: ekd.FieldList) -> ekd.FieldList:
        pass

    @abstractmethod
    def backward(self, x: ekd.FieldList) -> ekd.FieldList:
        pass

    def reverse(self) -> "Filter":
        return ReversedFilter(self)

    @classmethod
    def reversed(cls, *args, **kwargs):
        return ReversedFilter(cls(*args, **kwargs))


class ReversedFilter(Filter):
    """Swap the forward and backward methods of a filter."""

    def __init__(self, filter) -> None:
        self.filter = filter

    def __repr__(self) -> str:
        return f"Reversed({self.filter})"

    def forward(self, x: ekd.FieldList) -> ekd.FieldList:
        return self.filter.backward(x)

    def backward(self, x: ekd.FieldList) -> ekd.FieldList:
        return self.filter.forward(x)


class TransformFilter(Filter):
    """A filter to convert only some fields.
    The fields are matched by their metadata.
    """

    def _transform(self, data, transform, *group_by):

        result = []

        groupping = GroupByMarsParam(group_by)

        for matching in groupping.iterate(data, other=lambda x: result.append(x)):
            for f in transform(*matching):
                result.append(f)

        return self.new_fieldlits_from_list(result)

    def new_field_from_numpy(self, array, *, template, param):
        """Create a new field from a numpy array."""
        md = template.metadata().override(shortName=param)
        # return ekd.ArrayField(array, md)
        return ekd.FieldList.from_array(array, md)[0]

    def new_fieldlits_from_list(self, fields):
        from earthkit.data.indexing.fieldlist import FieldArray

        return FieldArray(fields)

    @abstractmethod
    def forward_transform(self, *fields):
        """To be implemented by subclasses."""
        pass

    @abstractmethod
    def backward_transform(self, *fields):
        """To be implemented by subclasses."""
        pass


FILTERS = {}


def register_filter(name, klass):
    FILTERS[name] = klass


def _load(file):
    name, _ = os.path.splitext(file)
    try:
        # The module is expected to register the filter
        # with the register_filter function
        importlib.import_module(f".{name}", package=__name__)
    except Exception:
        LOG.warning(f"Error loading filter '{name}'", exc_info=True)


def filter_registry(name) -> Filter:
    if name in FILTERS:
        return FILTERS[name]

    for entry_point in entrypoints.get_group_all("anemoi.filters"):
        if entry_point.name == name:
            FILTERS[name] = entry_point.load()
            return FILTERS[name]

    here = os.path.dirname(__file__)
    for file in os.listdir(here):

        if file[0] == ".":
            continue

        if file == "__init__.py":
            continue

        full = os.path.join(here, file)
        if os.path.isdir(full):
            if os.path.exists(os.path.join(full, "__init__.py")):
                _load(file)
            continue

        if file.endswith(".py"):
            _load(file)

    if name not in FILTERS:
        raise ValueError(f"Unknown filter '{name}'")

    return FILTERS[name]


def filter_factory(name, *args, **kwargs) -> Filter:
    return filter_registry(name)(*args, **kwargs)
