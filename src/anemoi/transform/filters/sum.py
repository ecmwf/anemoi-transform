# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict
from collections.abc import Hashable

import earthkit.data as ekd

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry


@filter_registry.register("sum")
class Sum(Filter):
    """Computes the sum over a set of variables.

    The ``sum`` filter computes the sum over multiple variables. This can be
    useful for computing total precipitation from its components (snow,
    rain) or summing the components of total column-integrated water. This
    filter must follow a source that provides the list of variables to be
    summed up. These variables are removed by the filter and replaced by a
    single summed variable.

    Example
    -------

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

    def __init__(self, *, params: list[str], output: str):
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

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Computes the sum over a set of variables.

        Args:
            fields (List[Any]): The list of input fields.

        Returns:
            ekd.FieldList: The resulting FieldArray with summed fields.
        """
        result = []

        needed_fields: dict[tuple[Hashable, ...], dict[str, ekd.Field]] = defaultdict(dict)

        for f in fields:
            key = f.metadata(namespace="mars")
            param = key.pop("param", None)
            if param is None:
                param = f.metadata("param")
            if param in self.params:
                key = tuple(key.items())

                if param in needed_fields[key]:
                    raise ValueError(f"Duplicate field {param} for {key}")

                needed_fields[key][param] = f
            else:
                result.append(f)

        for keys, values in needed_fields.items():

            if len(values) != len(self.params):
                raise ValueError("Missing fields")

            s = None
            for k, v in values.items():
                c = v.to_numpy(flatten=True)
                if s is None:
                    s = c
                else:
                    s += c
            result.append(new_field_from_numpy(s, template=values[list(values.keys())[0]], param=self.output))

        return new_fieldlist_from_list(result)

    def backward(self, data: ekd.FieldList) -> ekd.FieldList:
        raise NotImplementedError("Sum filter is not reversible")
