# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict
from typing import Dict
from typing import Hashable
from typing import List
from typing import Tuple

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry


@filter_registry.register("sum")
class Sum(Filter):

    def __init__(self, *, params: List[str], aggregated: bool = True, output: str):
        """Computes the sum over a set of variables.

        Parameters:
            aggregated (Bool): If True aggregates the sum for all variables listed in params.
            params (List[str]): The list of parameters to sum over.
            output (str): The name for the output field.

        Returns:
            ekd.FieldList: The resulting FieldArray with summed fields.
        """
        self.params = params
        self.output = output
        self.aggregated = aggregated

        if not self.aggregated:
            assert len(self.output) == len(self.params)

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:

        needed_fields: Dict[Tuple[Hashable, ...], Dict[str, ekd.Field]] = defaultdict(dict)
        result = []
        for f in data:
            param = f.metadata("param")
            if param in self.params:
                needed_fields[param] = f

        if set(needed_fields.values()).issubset(self.params):
            raise ValueError("Missing fields")

        s = []
        for key, values in needed_fields.items():
            index = list(needed_fields.keys()).index(key)
            c = values.to_numpy(flatten=True)
            s.append(np.sum(c))
            result.append(new_field_from_numpy(np.sum(c), template=needed_fields[key], param=self.output[index]))
        if self.aggregated:
            s = np.sum(s)
            result = [new_field_from_numpy(np.sum(s), template=needed_fields[self.params[0]], param=self.output)]

        return new_fieldlist_from_list(result)

    def backward(self, data: ekd.FieldList) -> ekd.FieldList:
        raise NotImplementedError("Sum filter is not reversible")
