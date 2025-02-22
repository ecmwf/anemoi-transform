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
from typing import Callable
from typing import Iterator
from typing import List

import earthkit.data as ekd
import numpy as np

from ..fields import new_field_from_numpy
from ..fields import new_fieldlist_from_list
from ..filter import Filter
from ..grouping import GroupByMarsParam

LOG = logging.getLogger(__name__)


class SimpleFilter(Filter):
    """A filter to convert only some fields.
    The fields are matched by their metadata.
    """

    def _transform(
        self, data: ekd.FieldList, transform: Callable[..., Iterator[ekd.Field]], *group_by: str
    ) -> ekd.FieldList:
        result = []

        grouping = GroupByMarsParam(group_by)

        for matching in grouping.iterate(data, other=result.append):
            for f in transform(*matching):
                result.append(f)

        return self.new_fieldlist_from_list(result)

    def new_field_from_numpy(self, array: np.ndarray, *, template: ekd.Field, param: str) -> ekd.Field:
        """Create a new field from a numpy array."""
        return new_field_from_numpy(array, template=template, param=param)

    def new_fieldlist_from_list(self, fields: List[ekd.Field]) -> ekd.FieldList:
        return new_fieldlist_from_list(fields)

    @abstractmethod
    def forward_transform(self, *fields: ekd.Field) -> Iterator[ekd.Field]:
        """To be implemented by subclasses."""
        pass

    @abstractmethod
    def backward_transform(self, *fields: ekd.Field) -> Iterator[ekd.Field]:
        """To be implemented by subclasses."""
        pass
