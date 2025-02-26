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
        """Transform the data using the specified transformation function.

        Parameters
        ----------
        data : ekd.FieldList
            Input data to be transformed.
        transform : callable
            Transformation function to apply to the data.
        group_by : str
            Fields to group by.

        Returns
        -------
        ekd.FieldList
            Transformed data.
        """
        result = []

        grouping = GroupByMarsParam(group_by)

        for matching in grouping.iterate(data, other=result.append):
            for f in transform(*matching):
                result.append(f)

        return self.new_fieldlist_from_list(result)

    def new_field_from_numpy(self, array: np.ndarray, *, template: ekd.Field, param: str) -> ekd.Field:
        """Create a new field from a numpy array.

        Parameters
        ----------
        array : np.ndarray
            Numpy array containing the field data.
        template : ekd.Field
            Template field to use for metadata.
        param : str
            Parameter name for the new field.

        Returns
        -------
        ekd.Field
            New field created from the numpy array.
        """
        return new_field_from_numpy(array, template=template, param=param)

    def new_fieldlist_from_list(self, fields: List[ekd.Field]) -> ekd.FieldList:
        """Create a new field list from a list of fields.

        Parameters
        ----------
        fields : list of ekd.Field
            List of fields to create the field list from.

        Returns
        -------
        ekd.FieldList
            New field list created from the list of fields.
        """
        return new_fieldlist_from_list(fields)

    @abstractmethod
    def forward_transform(self, *fields: ekd.Field) -> Iterator[ekd.Field]:
        """To be implemented by subclasses."""
        pass

    def backward_transform(self, *fields: ekd.Field) -> Iterator[ekd.Field]:
        """To be implemented by subclasses."""
        raise NotImplementedError(f"{self} backward transformation is not implemented.")


class SimpleFilter2(SimpleFilter):
    """Temporarily empty class to avoid breaking the code"""

    def __init__(self, forward_params, backward_params, **kwargs):
        self.forward_params = forward_params
        self.backward_params = backward_params

        for long, short in forward_params.items():
            setattr(self, long, kwargs.get(long, short))

        for long, short in backward_params.items():
            setattr(self, long, kwargs.get(long, short))

    def forward(self, data):

        args = []

        for long in self.forward_params.keys():
            args.append(getattr(self, long))

        def forward_transform(*fields):
            kwargs = {short: field for field, short in zip(fields, self.forward_params.values())}
            return self.forward_transform(**kwargs)

        return self._transform(data, forward_transform, *args)

    def backward(self, data):

        args = []

        for long in self.backward_params.keys():
            args.append(getattr(self, long))

        def backward_transform(*fields):
            kwargs = {short: field for field, short in zip(fields, self.backward_params.values())}
            return self.backward_transform(**kwargs)

        return self._transform(data, backward_transform, *args)
