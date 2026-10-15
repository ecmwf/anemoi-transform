# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from collections import defaultdict

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters.fields import filter_registry
from anemoi.transform.grouping import GroupByParam

LOG = logging.getLogger(__name__)


class _GroupByParamIgnoreLevel(GroupByParam):
    """GroupByParam that ignores the vertical level in the grouping key."""

    def _get_groups(self, data, *, other=None):
        from anemoi.transform.grouping import _lost

        other = other or _lost
        assert callable(other), type(other)
        self.groups = defaultdict(dict)
        self.groups_params = set()
        for f in data:
            key, extras = self._get_grouping_key(
                f,
                extract_from_grouping_key=["param"],
                remove_from_grouping_key=["variable", "levelist", "levtype"],
            )
            param = extras["param"]
            if param not in self.params:
                other(f)
                continue
            key = frozenset(key.items())
            if param in self.groups[key]:
                raise ValueError(f"Duplicate component {param} for {key}")
            self.groups[key][param] = f
            self.groups_params.add(param)


@filter_registry.register("sum")
class Sum(Filter):
    """Computes the sum over a set of variables.

    The ``sum`` filter computes the sum over multiple variables. This can be
    useful for computing total precipitation from its components (snow,
    rain) or summing the components of total column-integrated water. This
    filter must follow a source that provides the list of variables to be
    summed up. These variables are removed by the filter and replaced by a
    single summed variable.

    Examples
    --------

    .. code-block:: yaml

        input:
        pipe:
        - source:
            # mars, grib, netcdf, etc.
            # source attributes here
            # ...
            # Must load the variables to be summed

        - sum:
            params:
            # List of input variables
            - variable1
            - variable2
            - variable3
            output: variable_total # Name of output variable

    """

    def __init__(self, *, params: list[str], output: str, ignore_level: bool = False):
        """Initialize the Sum filter.

        Parameters:
        ----------
        params : List[str]
            The list of parameters to sum over.
        output : str
            The name for the output field.
        """
        self.params = params
        self.output = output
        self.ignore_level = ignore_level

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Computes the sum over a set of variables.

        Args:
            fields (List[Any]): The list of input fields.

        Returns:
            ekd.FieldList: The resulting FieldArray with summed fields.
        """
        result = []

        grouping = _GroupByParamIgnoreLevel(self.params) if self.ignore_level else GroupByParam(self.params)
        for group in grouping.iterate(fields, other=result.append):
            s = np.sum([f.to_numpy(flatten=True) for f in group], axis=0)
            result.append(new_field_from_numpy(s, template=group[0], param=self.output))

        return new_fieldlist_from_list(result)

    def backward(self, data: ekd.FieldList) -> ekd.FieldList:
        raise NotImplementedError("Sum filter is not reversible")
