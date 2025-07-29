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
from typing import Any
from typing import Callable
from typing import Union

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import FieldSelection
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.transform import Transform

LOG = logging.getLogger(__name__)


class Filter(Transform):
    """A filter transform that processes data."""

    pass


class SingleFieldFilter(Filter):
    """A filter that transforms fields individually (one at a time)."""

    required_inputs = None
    optional_inputs = {}

    def __init__(self, **kwargs) -> None:
        """Initialize the SingleFieldFilter.

        kwargs are accessible as attributes for use in transform and selection methods.
        """
        self._config = self.optional_inputs | kwargs
        self._validate_inputs()

        self.prepare_filter()

        self._forward_selection = FieldSelection(**self.forward_select())
        self._backward_selection = FieldSelection(**self.backward_select())

    def prepare_filter(self) -> None:
        """Provide an opportunity for subclasses to do additional work prior to use.
        E.g. validating inputs or loading ancillary data.

        Example use:
            if self.positive_number < 0:
                raise ValueError("positive_number must be positive")
        """
        pass

    def forward_select(self) -> dict[str, Union[str, list[str], tuple[str]]]:
        """Provide an opportunity for subclasses to select specific fields for processing.
        Only matching fields will be transformed (those not matching will be passed through unchanged).

        Return an empty dict to process all fields.

        Example:
            If "temperature" is in self.required_inputs, to transform fields where the field name is provided
            through the constructor as temperature, the following can be used:
            return {"field": self.temperature}
        """
        return {}

    def backward_select(self) -> dict[str, Union[str, list[str], tuple[str]]]:
        """Provide an opportunity for subclasses to select specific fields for processing on the backward transform.
        Defaults to the same fields as the forward select. If metadata is changed on the forward transform (e.g. param renamed),
        then the backward select may need to be updated accordingly.

        (See forward_select for more details.)
        """
        return self.forward_select()

    @abstractmethod
    def forward_transform(self, field: ekd.Field) -> ekd.Field:
        """Apply the transformation to a field. Must be implemented by subclasses."""
        pass

    def backward_transform(self, field: ekd.Field) -> ekd.Field:
        """Apply the backward transformation to a field."""
        raise NotImplementedError("Field backward transform not implemented.")

    def new_field_from_numpy(self, array: np.ndarray, *, template: ekd.Field, **metadata: dict) -> ekd.Field:
        return new_field_from_numpy(array, template=template, **metadata)

    def _validate_inputs(self) -> None:
        if not self.required_inputs:
            return

        if not isinstance(self.required_inputs, (list, tuple)):
            raise TypeError("Required inputs must be a list or tuple.")

        if not all(input in self._config for input in self.required_inputs):
            raise TypeError(f"Missing required input(s): '{set(self.required_inputs) - set(self._config)}'.")

        valid_keys = set(self.required_inputs) | set(self.optional_inputs)
        leftover_keys = set(self._config) - valid_keys
        if leftover_keys:
            raise ValueError(f"Unknown input(s): '{leftover_keys}'.")

    def __getattr__(self, name: str) -> Any:
        # Allow access to kwargs passed into constructor as attributes
        return self._config[name]

    @staticmethod
    def _map_transform(transform_function: Callable, fields: ekd.FieldList) -> ekd.FieldList:
        return new_fieldlist_from_list([transform_function(field) for field in fields])

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        def transform(field: ekd.Field) -> ekd.Field:
            return self.forward_transform(field) if self._forward_selection.match(field) else field

        return self._map_transform(transform, data)

    def backward(self, data: ekd.FieldList) -> ekd.FieldList:
        def transform(field: ekd.Field) -> ekd.Field:
            return self.backward_transform(field) if self._backward_selection.match(field) else field

        return self._map_transform(transform, data)
