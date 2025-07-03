# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Iterator
from typing import List

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry

from .matching import MatchingFieldsFilter
from .matching import matching


@filter_registry.register("sum")
class Sum(MatchingFieldsFilter):

    @matching(
        select="param",
        forward=("input_variables",),
        backward=("summed_variable",),
    )
    def __init__(self, *, input_variables: List[str], summed_variable: str = "sum"):
        self.input_variables = input_variables
        self.summed_variable = summed_variable

    def forward_transform(self, input_variables: ekd.Field) -> Iterator[ekd.Field]:
        """Compute the element-wise sum of the input fields"""
        summed = np.sum(input_variables.to_numpy(), axis=0)
        # Use the first field as template for metadata
        yield self.new_field_from_numpy(summed, template=input_variables, param=self.summed_variable)

    def backward_transform(self, data: ekd.FieldList) -> Iterator[ekd.Field]:
        raise NotImplementedError("Sum filter is not reversible")
