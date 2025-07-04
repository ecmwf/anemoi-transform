# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict
from typing import List

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry


@filter_registry.register("sum")
class Sum(Filter):
    """Computes the sum over a set of variables."""

    def __init__(self, *, params: List[str], output: str):
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

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """sum over variables

        Parameters:
        ----------
        data : ekd.FieldList)
            The input FieldArray with the variables to be summed according to self.params

        Returns:
        ----------
        ekd.FieldList:
            The resulting FieldArray with summed fields.
        """

        result = []
        s = defaultdict(list)
        templates = defaultdict(list)

        for f in data:
            param = f.metadata("param")
            if param in self.params:
                s[param].append(f.to_numpy())
                templates[param].append(f)
            else:
                result.append(f)

        assert set(s.keys()) == set(self.params)

        sum_values = []
        for param in s.keys():
            sum_values.append(np.stack(s[param]))
        stack_summed = np.stack(sum_values)

        for level in range(stack_summed.shape[1]):
            template_level = templates[self.params[0]][level]
            result.append(
                new_field_from_numpy(np.sum(stack_summed[:, level], axis=0), template=template_level, param=self.output)
            )

        return new_fieldlist_from_list(result)

    def backward(self, data: ekd.FieldList) -> ekd.FieldList:
        raise NotImplementedError("Sum filter is not reversible")
