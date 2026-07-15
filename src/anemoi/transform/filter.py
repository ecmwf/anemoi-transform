# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import functools
import logging
from abc import abstractmethod
from collections.abc import Callable
from functools import singledispatchmethod
from typing import Any

import pandas as pd

from anemoi.transform import Field
from anemoi.transform import FieldList
from anemoi.transform import Frame
from anemoi.transform.fields import FieldSelection
from anemoi.transform.transform import Transform

LOG = logging.getLogger(__name__)


class Filter(Transform):
    """A filter transform that processes field data."""

    pass


def _preserve_frame_type(method: Callable) -> Callable:
    """Wrap a raw-``DataFrame`` filter method so it also accepts/returns :class:`Frame`.

    When called with a :class:`~anemoi.transform.frames.Frame`, the underlying
    pandas ``DataFrame`` is unwrapped, the wrapped method runs on it, and a
    ``DataFrame`` result is re-wrapped as a ``Frame``. When called with a raw
    ``pd.DataFrame`` (or anything else), the method runs unchanged and the result
    type is preserved. This lets a ``Frame`` flow through a tabular pipeline
    end-to-end without changing filter bodies or the raw-``DataFrame`` contract
    that existing callers and tests rely on.
    """

    def _propagate_attrs(data: Any, result: Any) -> Any:
        # pandas propagates DataFrame.attrs inconsistently; carry them over
        # explicitly (e.g. the origin attached by the dataset-create actions)
        # without overwriting anything the filter set itself.
        if isinstance(data, pd.DataFrame) and isinstance(result, pd.DataFrame):
            for k, v in data.attrs.items():
                result.attrs.setdefault(k, v)
        return result

    @functools.wraps(method)
    def wrapper(self: Any, data: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(data, Frame):
            frame = data.to_pandas()
            result = _propagate_attrs(frame, method(self, frame, *args, **kwargs))
            return Frame.from_pandas(result) if isinstance(result, pd.DataFrame) else result
        return _propagate_attrs(data, method(self, data, *args, **kwargs))

    wrapper._frame_type_preserving = True
    return wrapper


class TabularFilter(Filter):
    """A filter that transforms tabular data (a pandas ``DataFrame``).

    Subclasses implement ``forward`` (and optionally ``backward``) operating on a
    raw :class:`pandas.DataFrame`, exactly as before. This base transparently also
    accepts and returns an :class:`~anemoi.transform.frames.Frame` (the tabular
    counterpart of :class:`~anemoi.transform.fields.FieldList`), so tabular data
    can flow through a pipeline as a ``Frame`` — while a raw ``DataFrame`` in still
    yields a raw ``DataFrame`` out, preserving the existing contract.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for name in ("forward", "backward"):
            method = cls.__dict__.get(name)
            if method is not None and not getattr(method, "_frame_type_preserving", False):
                setattr(cls, name, _preserve_frame_type(method))


class DispatchingFilter(Transform):
    """A filter transform that processes either tabular or field data."""

    @classmethod
    def _ensure_specialist_forward_provided(cls):
        if cls is DispatchingFilter:
            return

        def overridden(name):
            return getattr(cls, name) is not getattr(DispatchingFilter, name)

        if not (overridden("forward_fields") or overridden("forward_tabular")):
            raise TypeError(f"{cls.__name__} must override at least one of `forward_fields` or `forward_tabular`")

        for data_type in ("fields", "tabular"):
            forward_name = f"forward_{data_type}"
            backward_name = f"backward_{data_type}"
            if overridden(backward_name) and not overridden(forward_name):
                raise TypeError(f"{cls.__name__} overrides `{backward_name}` but not `{forward_name}`")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._ensure_specialist_forward_provided()

    @singledispatchmethod
    def forward(self, data: Any) -> Any:
        return self.forward_fallback(data)

    @forward.register
    def _(self, data: FieldList) -> FieldList:
        return self.forward_fields(data)

    @forward.register
    def _(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.forward_tabular(data)

    @forward.register
    def _(self, data: Frame) -> Frame:
        return Frame.from_pandas(self.forward_tabular(data.to_pandas()))

    def forward_fallback(self, data: Any) -> Any:
        raise TypeError(f"No forward method for {type(data)}")

    def forward_fields(self, data: FieldList) -> FieldList:
        return self.forward_fallback(data)

    def forward_tabular(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.forward_fallback(data)

    @singledispatchmethod
    def backward(self, data: Any) -> Any:
        return self.backward_fallback(data)

    @backward.register
    def _(self, data: FieldList) -> FieldList:
        return self.backward_fields(data)

    @backward.register
    def _(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.backward_tabular(data)

    @backward.register
    def _(self, data: Frame) -> Frame:
        return Frame.from_pandas(self.backward_tabular(data.to_pandas()))

    def backward_fallback(self, data: Any) -> Any:
        raise NotImplementedError(f"No backward method for {type(data)}")

    def backward_tabular(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.backward_fallback(data)

    def backward_fields(self, data: FieldList) -> FieldList:
        return self.backward_fallback(data)


class SingleFieldFilter(Filter):
    """A filter that transforms fields individually (one at a time)."""

    required_inputs: tuple[str, ...] | list[str] | None = None
    optional_inputs: dict[str, Any] = {}

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

    def forward_select(self) -> dict[str, str | list[str] | tuple[str]]:
        """Provide an opportunity for subclasses to select specific fields for processing.
        Only matching fields will be transformed (those not matching will be passed through unchanged).

        Return an empty dict to process all fields.

        Example:
            If "temperature" is in self.required_inputs, to transform fields where the field name is provided
            through the constructor as temperature, the following can be used:
            return {"field": self.temperature}
        """
        return {}

    def backward_select(self) -> dict[str, str | list[str] | tuple[str]]:
        """Provide an opportunity for subclasses to select specific fields for processing on the backward transform.
        Defaults to the same fields as the forward select. If metadata is changed on the forward transform (e.g. param renamed),
        then the backward select may need to be updated accordingly.

        (See forward_select for more details.)
        """
        return self.forward_select()

    @abstractmethod
    def forward_transform(self, field: Field) -> Field:
        """Apply the transformation to a field. Must be implemented by subclasses."""
        pass

    def backward_transform(self, field: Field) -> Field:
        """Apply the backward transformation to a field."""
        raise NotImplementedError("Field backward transform not implemented.")

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

    @property
    def config(self):
        return self._config

    def __getattr__(self, name: str) -> Any:
        # Allow access to kwargs passed into constructor as attributes
        return self._config[name]

    @staticmethod
    def _map_transform(transform_function: Callable, fields: FieldList) -> FieldList:
        return FieldList.from_fields([transform_function(field) for field in fields])

    def forward(self, data: FieldList) -> FieldList:
        def transform(field: Field) -> Field:
            return self.forward_transform(field) if self._forward_selection.match(field) else field

        return self._map_transform(transform, data)

    def backward(self, data: FieldList) -> FieldList:
        def transform(field: Field) -> Field:
            return self.backward_transform(field) if self._backward_selection.match(field) else field

        return self._map_transform(transform, data)
