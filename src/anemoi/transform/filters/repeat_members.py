# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

from ..fields import new_field_from_numpy
from ..fields import new_fieldlist_from_list
from . import filter_registry
from .base import Filter

LOG = logging.getLogger(__name__)


@filter_registry.register("repeat_members")
class RepeatMembers(Filter):
    """A filter to replicate non-ensemble data into ensemble data"""

    def __init__(
        self,
        numbers=None,  # 1-based
        members=None,  # 0-based
        count=None,
    ):
        if sum(x is not None for x in (members, count, numbers)) != 1:
            raise ValueError("Exactly one of members, count or numbers must be given")

        if numbers is not None:
            assert isinstance(numbers, (list, tuple)), f"numbers must be a list or tuple, got {type(numbers)}"
            members = [n - 1 for n in numbers]

        if count is not None:
            members = list(range(count))

        self.members = members
        assert isinstance(members, (tuple, list)), f"members must be a list or tuple, got {type(members)}"

    def forward(self, data):
        result = []
        for f in data:
            array = f.to_numpy()
            for member in self.members:
                number = member + 1
                new_field = new_field_from_numpy(array, template=f, number=number)
                result.append(new_field)

        return new_fieldlist_from_list(result)

    def backward(self, data):
        # this could be implemented
        raise NotImplementedError("RepeatMembers is not reversible")
