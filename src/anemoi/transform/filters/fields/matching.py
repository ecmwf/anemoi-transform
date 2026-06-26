# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import abstractmethod
from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import replace
from inspect import signature
from itertools import chain
from typing import Iterable
from typing import Literal
from typing import cast

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.grouping import GroupByParam
from anemoi.transform.grouping import GroupByParamVertical

LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MatchingSpec:
    select: Literal["param"] = "param"
    forward: tuple[str, ...] = ()
    backward: tuple[str, ...] = ()
    return_inputs: Literal["all", "none"] | tuple[str, ...] = "none"
    vertical: bool = False

    @staticmethod
    def _to_tuple_of_str(x: str | Iterable[str]) -> tuple[str, ...]:
        if isinstance(x, str):
            return (x,)
        try:
            return tuple(x)
        except TypeError as e:
            raise TypeError(f"Expected str or iterable, got {type(x)}") from e

    def __post_init__(self) -> None:
        if self.select != "param":
            raise NotImplementedError("Only 'select=param' is supported for now.")

        object.__setattr__(self, "forward", self._to_tuple_of_str(self.forward))
        object.__setattr__(self, "backward", self._to_tuple_of_str(self.backward))

        if self.return_inputs not in ("all", "none"):
            object.__setattr__(self, "return_inputs", self._to_tuple_of_str(self.return_inputs))
            all_params = set(self.forward) | set(self.backward)
            if not set(self.return_inputs).issubset(all_params):
                raise ValueError(f"Returned input names must subset {all_params}")

    def update_return_inputs(self, return_inputs: Literal["all", "none"] | Iterable[str]) -> "MatchingSpec":
        if return_inputs in ("all", "none"):
            return_inputs = cast('Literal["all", "none"]', return_inputs)
        else:
            return_inputs = self._to_tuple_of_str(return_inputs)

        if return_inputs == self.return_inputs:
            return self

        return replace(self, return_inputs=return_inputs)

    def inputs(self, direction: Literal["forward", "backward"]) -> tuple[str, ...]:
        if self.return_inputs == "all":
            return tuple(getattr(self, direction))
        if self.return_inputs == "none":
            return ()
        return self.return_inputs


def inputs_generator(input_list: Iterable[str], **kwargs) -> Iterator[ekd.Field]:
    for name in input_list:
        if name in kwargs:
            yield kwargs[name]


class MatchingFieldsFilter(Filter):
    """A filter to convert only some fields.
    The fields are matched by their metadata.
    """

    MATCHING: MatchingSpec

    @staticmethod
    def _check_expected_method_parameters(method, expected_parameters):
        method_params = signature(method).parameters
        missing = set(expected_parameters) - set(method_params)
        if missing:
            raise ValueError(f"{method}: missing parameters {missing}")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "MATCHING") or not isinstance(cls.MATCHING, MatchingSpec):
            raise TypeError(f"Class {cls.__name__} must define a 'MATCHING' attribute of type MatchingSpec.")

        forward_required_params = set(cls.MATCHING.forward)
        backward_required_params = set(cls.MATCHING.backward)
        constructor_required_params = forward_required_params | backward_required_params

        MatchingFieldsFilter._check_expected_method_parameters(cls.__init__, constructor_required_params)
        MatchingFieldsFilter._check_expected_method_parameters(cls.forward_transform, forward_required_params)
        MatchingFieldsFilter._check_expected_method_parameters(cls.backward_transform, backward_required_params)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._prepare_matching()

    def _prepare_matching(self) -> None:
        if hasattr(self, "return_inputs"):
            self.MATCHING = self.MATCHING.update_return_inputs(self.return_inputs)

        for direction in ("forward", "backward"):
            params = getattr(self.MATCHING, direction)
            inputs = self.MATCHING.inputs(direction=direction)

            if inputs and params and not set(inputs).issubset(params):
                diff = set(inputs) - set(params)
                LOG.warning(
                    f"Some {direction} inputs will not be returned because they are not in the filter parameters: {diff}"
                )

    def _check_metadata_match(self, data: ekd.FieldList, args: list[str] | tuple[str, ...]) -> None:
        """Checks the parameters names of the data and the groups match.

        Parameters
        ----------
        data : str
            List with grouped input param names.
        args : str
            List with fields to group by.
        """

        msg = (
            f"Please ensure your filter is configured to match the input variables metadata "
            f"current mismatch between inputs {data} and filter metadata {args}"
        )
        if not set(args).issubset(data):
            LOG.warning(msg)

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

        # passed into this...
        def _forward_transform(*fields: ekd.Field) -> Iterator[ekd.Field]:
            kwargs = dict(zip(self.MATCHING.forward, fields, strict=True))
            return chain(
                inputs_generator(self.MATCHING.inputs(direction="forward"), **kwargs), self.forward_transform(**kwargs)
            )

        group_by = (getattr(self, name) for name in self.MATCHING.forward)
        return self._transform(
            data,
            _forward_transform,
            *group_by,
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

        # passed into this...
        def _backward_transform(*fields: ekd.Field) -> Iterator[ekd.Field]:
            kwargs = dict(zip(self.MATCHING.backward, fields, strict=True))
            return chain(
                inputs_generator(self.MATCHING.inputs(direction="backward"), **kwargs),
                self.backward_transform(**kwargs),
            )

        group_by = (getattr(self, name) for name in self.MATCHING.backward)
        return self._transform(
            data,
            _backward_transform,
            *group_by,
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
        if self.MATCHING.vertical:
            grouping = GroupByParamVertical(group_by)
        else:
            grouping = GroupByParam(group_by)

        input_params = set(data.metadata(self.MATCHING.select))
        self._check_metadata_match(input_params, group_by)

        result: list[ekd.Field] = []
        for matching in grouping.iterate(data, other=result.append):
            for f in transform(*matching):
                result.append(f)
        return self.new_fieldlist_from_list(result)

    def new_field_from_numpy(self, array: np.ndarray, *, template: ekd.Field, **kwargs) -> ekd.Field:
        """Create a new field from a numpy array.

        Parameters
        ----------
        array : np.ndarray
            Numpy array containing the field data.
        template : ekd.Field
            Template field to use for metadata.
        **kwargs : Any
            Additional keyword arguments for the new field.

        Returns
        -------
        ekd.Field
            New field created from the numpy array.
        """
        return new_field_from_numpy(array, template=template, **kwargs)

    def new_fieldlist_from_list(self, fields: list[ekd.Field]) -> ekd.FieldList:
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
