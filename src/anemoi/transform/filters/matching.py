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
from functools import wraps
from inspect import signature
from typing import Callable
from typing import Iterator
from typing import List

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.grouping import GroupByParam

LOG = logging.getLogger(__name__)


def get_params_and_defaults(method: Callable) -> dict:
    """Get the list of parameters and their default values from a method.

    Parameters
    ----------
    method : callable
        The method to inspect.

    Returns
    -------
    dict
        A dictionary with parameter names as keys and their default values as values.
    """
    sig = signature(method)
    return {k: v.default for k, v in sig.parameters.items() if v.default is not v.empty}


class matching:
    def __init__(self, match, forward=[], backward=[]):
        self.match = match

        if not isinstance(forward, (list, tuple)):
            forward = [forward]

        if not isinstance(backward, (list, tuple)):
            backward = [backward]

        self.forward = forward
        self.backward = backward

    def __call__(self, method):

        self.params_and_defaults = get_params_and_defaults(method)

        forward = {}
        for name in self.params_and_defaults.keys():
            if name in self.forward:
                forward[name] = name

        backward = {}
        for name in self.params_and_defaults.keys():
            if name in self.backward:
                backward[name] = name

        @wraps(method)
        def wrapped(obj, *args, **kwargs):
            obj._forward_arguments = forward
            obj._backward_arguments = backward
            obj._initialised = True
            return method(obj, *args, **kwargs)

        return wrapped


class MatchingFieldsFilter(Filter):
    """A filter to convert only some fields.
    The fields are matched by their metadata.
    """

    _initialised = False

    @property
    def forward_arguments(self):
        if not self._initialised:
            raise ValueError("Filter not initialised.")

        return self._forward_arguments

    @property
    def backward_arguments(self):
        if not self._initialised:
            raise ValueError("Filter not initialised.")

        return self._backward_arguments

    def forward(self, data):

        args = []

        for name in self.forward_arguments:
            args.append(getattr(self, name))

        def forward_transform(*fields):
            assert len(fields) == len(self.forward_arguments)
            kwargs = {name: field for field, name in zip(fields, self.forward_arguments)}
            return self.forward_transform(**kwargs)

        return self._transform(data, forward_transform, *args)

    def backward(self, data):

        args = []

        for name in self.backward_arguments:
            args.append(getattr(self, name))

        def backward_transform(*fields):
            assert len(fields) == len(self.backward_arguments)
            kwargs = {name: field for field, name in zip(fields, self.backward_arguments)}
            return self.backward_transform(**kwargs)

        return self._transform(data, backward_transform, *args)

    def _transform(
        self,
        data: ekd.FieldList,
        transform: Callable[..., Iterator[ekd.Field]],
        *group_by: str,
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

        grouping = GroupByParam(group_by)

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
