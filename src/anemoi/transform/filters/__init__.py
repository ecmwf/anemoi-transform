# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from typing import Any

import earthkit.data as ekd
import numpy as np
from earthkit.data import FieldList
from earthkit.meteo.wind.array import polar_to_xy
from earthkit.meteo.wind.array import xy_to_polar


class Filter(ABC):

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    @abstractmethod
    def forward(self, x: ekd.FieldList) -> ekd.FieldList:
        pass

    @abstractmethod
    def backward(self, x: ekd.FieldList) -> ekd.FieldList:
        pass

    def reverse(self) -> "Filter":
        return ReversedFilter(self)


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

        groups = defaultdict(dict)

        for f in data:
            key = f.metadata(namespace="mars")
            param = key.pop("param")

            if param not in group_by:
                result.append(f)
                continue

            key = tuple(key.items())

            if param in groups[key]:
                raise ValueError(f"Duplicate component {param} for {key}")

            groups[key][param] = f

        for _, group in groups.items():
            if len(group) != len(group_by):
                raise ValueError("Missing component")

            for f in transform(*[group[p] for p in group_by]):
                result.append(f)

        return self.new_fieldlits_from_list(result)

    def new_field_from_numpy(self, array, *, template, param):
        """Create a new field from a numpy array."""
        md = template.metadata().override(shortName=param)
        return FieldList.from_array(array, md)[0]

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
