# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod

from ..fields import new_field_from_numpy
from ..fields import new_fieldlist_from_list
from ..filter import Filter
from ..grouping import GroupByMarsParam

LOG = logging.getLogger(__name__)


class SimpleFilter(Filter):
    """A filter to convert only some fields.
    The fields are matched by their metadata.
    """

    def _transform(self, data, transform, *group_by):

        result = []

        grouping = GroupByMarsParam(group_by)

        for matching in grouping.iterate(data, other=result.append):
            for f in transform(*matching):
                result.append(f)

        return self.new_fieldlist_from_list(result)

    def new_field_from_numpy(self, array, *, template, param):
        """Create a new field from a numpy array."""
        return new_field_from_numpy(array, template=template, param=param)

    def new_fieldlist_from_list(self, fields):
        return new_fieldlist_from_list(fields)

    @abstractmethod
    def forward_transform(self, *fields):
        """To be implemented by subclasses."""
        pass

    @abstractmethod
    def backward_transform(self, *fields):
        """To be implemented by subclasses."""
        pass
