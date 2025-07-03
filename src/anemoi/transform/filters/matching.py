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
from typing import Any
from typing import Callable
from typing import Iterator
from typing import List
from typing import Tuple

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.grouping import GroupByParam

LOG = logging.getLogger(__name__)


def _get_params_and_defaults(method: Callable) -> dict:
    """Get the list of parameters and their default values from a method.

    Parameters
    ----------
    method : Callable
        The method to inspect.

    Returns
    -------
    dict
        A dictionary with parameter names as keys and their default values as values.
    """
    sig = signature(method)
    return {k: v.default for k, v in sig.parameters.items()}  # if v.default is not v.empty}


def _check_arguments(method: Callable) -> Tuple[bool, bool, bool]:
    """Check the types of arguments in the method signature.

    Parameters
    ----------
    method : Callable
        The method to inspect.

    Returns
    -------
    Tuple[bool, bool, bool]
        A tuple indicating the presence of positional or keyword arguments,
        variable positional arguments, and variable keyword arguments.
    """
    sig = signature(method)
    has_params = any(param.kind == param.POSITIONAL_OR_KEYWORD for param in sig.parameters.values())
    has_args = any(param.kind == param.VAR_POSITIONAL for param in sig.parameters.values())
    has_kwargs = any(param.kind == param.VAR_KEYWORD for param in sig.parameters.values())

    result = (has_params, has_args, has_kwargs)

    if all(a is False for a in result):
        raise ValueError(f"{method}: no arguments found in method signature.")

    if sum(a is True for a in result) > 1:
        raise ValueError(f"{method}: cannot mix named parameters and *args and/or **kargs.")

    if has_kwargs:  # For now
        raise NotImplementedError(f"{method}: cannot have **kwargs.")

    return has_params, has_args, has_kwargs


class matching:
    """A decorator to decorate the __init__ method of a subclass of MatchingFieldsFilter"""

    def __init__(self, *, select: str, forward: list = [], backward: list = []) -> None:
        """Initialize the matching decorator.

        Parameters
        ----------
        select : str
            The attribute to select.
        forward : list, optional
            List of forward arguments, by default [].
        backward : list, optional
            List of backward arguments, by default [].
        """
        self.select = select

        if select != "param":
            raise NotImplementedError("Only 'select=param' is supported for now.")

        if not isinstance(forward, (list, tuple)):
            forward = [forward]

        if not isinstance(backward, (list, tuple)):
            backward = [backward]

        self.forward = forward
        self.backward = backward

    def __call__(self, method: Callable) -> Callable:
        """Wrap the method with forward and backward argument initialization.

        Parameters
        ----------
        method : Callable
            The method to wrap.

        Returns
        -------
        Callable
            The wrapped method.
        """
        self.params_and_defaults = _get_params_and_defaults(method)

        seen = set()
        forward = {}
        for name in self.params_and_defaults.keys():
            if name in self.forward:
                forward[name] = name
                seen.add(name)

        for name in self.forward:
            if name not in seen:
                LOG.warning(f"{method}: forward argument `{name}` not found in method signature.")

        seen = set()
        backward = {}
        for name in self.params_and_defaults.keys():
            if name in self.backward:
                backward[name] = name
                seen.add(name)

        for name in self.backward:
            if name not in seen:
                LOG.warning(f"{method}: backward argument `{name}` not found in method signature.")

        @wraps(method)
        def wrapped(obj: Any, *args: Any, **kwargs: Any) -> Any:

            obj._forward_arguments_types = _check_arguments(getattr(obj, "forward_transform"))
            obj._backward_arguments_types = _check_arguments(getattr(obj, "backward_transform"))

            obj._select = self.select
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
    def forward_arguments(self) -> dict:
        """Get the forward arguments.

        Raises
        ------
        ValueError
            If the filter is not initialised.

        """
        if not self._initialised:
            raise ValueError("Filter not initialised.")

        return self._forward_arguments

    @property
    def backward_arguments(self) -> dict:
        """Get the backward arguments.

        Raises
        ------
        ValueError
            If the filter is not initialised.

        """
        if not self._initialised:
            raise ValueError("Filter not initialised.")

        return self._backward_arguments

    def _check_metadata_match(self, data: ekd.FieldList, args: List[str]) -> None:
        """Checks the parameters names of the data and the groups match

        Parameters
        ----------
        data : str
            List with grouped input param names
        args : str
            List with fields to group by.
        """

        error_msg = (
            f"Please ensure your filter is configured to match the input variables metadata "
            f"current mismatch between inputs {data} and filter metadata {args}"
        )
        if not set(args).issubset(data):
            raise ValueError(error_msg)

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Transform the data using the forward transformation function.

        Parameters
        ----------
        data : ekd.FieldList
            Input data to be transformed.

        Returns
        -------
        ekd.FieldList
            Transformed data.
        """
        args = []

        for name in self.forward_arguments:
            args.append(getattr(self, name))

        named_args = self._forward_arguments_types[0]

        def forward_transform_named(*fields: ekd.Field) -> Iterator[ekd.Field]:
            assert len(fields) == len(self.forward_arguments)
            kwargs = {name: field for field, name in zip(fields, self.forward_arguments)}
            return self.forward_transform(**kwargs)

        return self._transform(
            data,
            forward_transform_named if named_args else self.forward_transform,
            *args,
        )

    def backward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Transform the data using the backward transformation function.

        Parameters
        ----------
        data : ekd.FieldList
            Input data to be transformed.

        Returns
        -------
        ekd.FieldList
            Transformed data.
        """
        args = []

        for name in self.backward_arguments:
            args.append(getattr(self, name))

        named_args = self._backward_arguments_types[0]

        def backward_transform(*fields: ekd.Field) -> Iterator[ekd.Field]:
            assert len(fields) == len(self.backward_arguments)
            kwargs = {name: field for field, name in zip(fields, self.backward_arguments)}
            return self.backward_transform(**kwargs)

        return self._transform(
            data,
            backward_transform if named_args else self.backward_transform,
            *args,
        )

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
        transform : Callable
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
        input_params = set(data.metadata("param"))
        self._check_metadata_match(input_params, group_by)
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
        fields : List[ekd.Field]
            List of fields to create the field list from.

        Returns
        -------
        ekd.FieldList
            New field list created from the list of fields.
        """
        return new_fieldlist_from_list(fields)

    @abstractmethod
    def forward_transform(self, *fields: ekd.Field) -> Iterator[ekd.Field]:
        """Forward transformation to be implemented by subclasses.

        Parameters
        ----------
        fields : ekd.Field
            Fields to be transformed.

        Returns
        -------
        Iterator[ekd.Field]
            Transformed fields.
        """
        pass

    def backward_transform(self, *fields: ekd.Field) -> Iterator[ekd.Field]:
        """Backward transformation to be implemented by subclasses.

        Parameters
        ----------
        fields : ekd.Field
            Fields to be transformed.

        Returns
        -------
        Iterator[ekd.Field]
            Transformed fields.
        """
        raise NotImplementedError("Backward transformation not implemented.")
