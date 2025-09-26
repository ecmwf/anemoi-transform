# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any

import earthkit.data as ekd

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


def make_list_int(value: Any) -> list[int]:
    """Convert a value to a list of integers.

    Parameters
    ----------
    value : Any
        The value to convert.

    Returns
    -------
    list
        The converted list of integers.
    """
    if isinstance(value, str):
        if "/" not in value:
            return [int(value)]
        bits = value.split("/")
        if len(bits) == 3 and bits[1].lower() == "to":
            value = list(range(int(bits[0]), int(bits[2]) + 1, 1))

        elif len(bits) == 5 and bits[1].lower() == "to" and bits[3].lower() == "by":
            value = list(range(int(bits[0]), int(bits[2]) + int(bits[4]), int(bits[4])))

    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, int):
        return [value]

    raise ValueError(f"Cannot make list from {value}")


@filter_registry.register("repeat_members")
class RepeatMembers(Filter):
    """The filter can be used to replicate non-ensembles fields into ensemble fields.

    Only one of the following parameters should be provided.

    - ``numbers``: A list of 1-based numbers.
    - ``members``: A list of 0-based indices.
    - ``count``: The number of times to replicate the fields.

    The resulting fields will have their field number set to the corresponding member index + 1.

    Examples
    --------

    The code below replicates each input field 5 times, setting the ``number`` attribute of the resulting fields from 1 to 5.

    .. code-block:: yaml

        input:
          pipe:
            - source: # Can be `mars`, `netcdf`, etc.
                param: ...
            - repeat_members:
                count: 5 # Replicate each input field 5 times

    Or using specific member indices (0-based), to set the ``number`` attribute of the resulting fields to 1, 3 and 5:

    .. code-block:: yaml

        input:
          pipe:
            - source: # Can be `mars`, `netcdf`, etc.
                param: ...
            - repeat_members:
                members: [0, 2, 4]

    Or using specific field numbers (1-based), to set the ``number`` attribute of the resulting fields to 1, 3 and 5:

    .. code-block:: yaml

        input:
          pipe:
            - source: # Can be `mars`, `netcdf`, etc.
                param: ...
            - repeat_members:
                numbers: [1, 3, 5]
    """

    def __init__(
        self,
        *,
        numbers: list[int] | None = None,
        members: list[int] | None = None,
        count: int | None = None,
    ) -> None:
        """Parameters
        -------------
        numbers : list of int, optional
            A list of numbers (1-based) of the fields to replicate.
        members : list of int, optional
            A list of 0-based indices of the fields to replicate.
        count : int, optional
            The number of times to replicate the fields.
        """

        if sum(x is not None for x in (members, count, numbers)) != 1:
            raise ValueError("Exactly one of members, count or numbers must be given")

        if numbers is not None:
            numbers = make_list_int(numbers)
            members = [n - 1 for n in numbers]

        if count is not None:
            members = list(range(count))

        members = make_list_int(members)
        self.members = members
        assert isinstance(members, (tuple, list)), f"members must be a list or tuple, got {type(members)}"

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Apply the forward transformation to replicate fields.

        Parameters
        ----------
        data : ekd.FieldList
            The input data to be transformed.

        Returns
        -------
        ekd.FieldList
            The transformed data.
        """
        result = []
        for f in data:
            array = f.to_numpy()
            for member in self.members:
                number = member + 1
                new_field = new_field_from_numpy(array, template=f, number=number)
                result.append(new_field)

        return new_fieldlist_from_list(result)
