# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import typing as tp
import importlib

from anemoi.transform.filters.base import SimpleFilter
from anemoi.transform.filters import filter_registry
from earthkit.data.core.fieldlist import Field, FieldList


@filter_registry.register("earthkitfieldlambda")
class EarthkitFieldLambdaFilter(SimpleFilter):
    """A filter to apply an arbitrary function to individual fields."""

    def __init__(
        self,
        fn: str | tp.Callable[[Field, tp.Any], Field],
        param: str | list[str],
        args: list = [],
        kwargs: dict[str, tp.Any] = {},
        backward_fn: str | tp.Callable[[Field, tp.Any], Field] | None = None,
    ):  
        """Initialise the EarthkitFieldLambdaFilter.
        
        Parameters
        ----------
        fn: callable or str
            The lambda function as a callable with the general signature 
            `fn(*earthkit.data.Field, *args, **kwargs) -> earthkit.data.Field` or
            a string path to the function, such as "package.module.function".
        param: list or str
            The parameter name or list of parameter names to apply the function to.
        args: list
            The list of arguments to pass to the lambda function.
        kwargs: dict
            The dictionary of keyword arguments to pass to the lambda function.
        backward_fn (optional): callable, str or None
            The backward lambda function as a callable with the general signature 
            `backward_fn(*earthkit.data.Field, *args, **kwargs) -> earthkit.data.Field` or
            a string path to the function, such as "package.module.function".

        Examples
        --------
        >>> from anemoi.transform.filters.lambda_filters import EarthkitFieldLambdaFilter
        >>> import earthkit.data as ekd
        >>> fields = ekd.from_source(
        ...        "mars",{"param": ["2t"],
        ...                "levtype": "sfc",
        ...                "dates": ["2023-11-17 00:00:00"]})
        >>> kelvin_to_celsius = EarthkitFieldLambdaFilter(
        ...     fn=lambda x, s: x.clone(values=x.values - s),
        ...     param="2t",
        ...     args=[273.15],
        ... )
        >>> fields = kelvin_to_celsius.forward(fields)
        """
        if not isinstance(args, list):
            raise ValueError("Expected 'args' to be a list. "
                             f"Got {args} instead.")
        if not isinstance(kwargs, dict):
            raise ValueError("Expected 'kwargs' to be a dictionary. "
                             f"Got {kwargs} instead.")

        self.fn = self._import_fn(fn) if isinstance(fn, str) else fn

        if isinstance(backward_fn, str):
            self.backward_fn = self._import_fn(backward_fn)
        else:
            self.backward_fn = backward_fn
                    
        self.param = param if isinstance(param, list) else [param]
        self.args = args
        self.kwargs = kwargs

    def forward(self, data: FieldList) -> FieldList:
        return self._transform(data, self.forward_transform, *self.param)

    def backward(self, data: FieldList) -> FieldList:
        if self.backward_fn:
            return self._transform(data, self.backward_transform, *self.param)
        raise NotImplementedError(f"{self} is not reversible.")

    def forward_transform(self, *fields: Field) -> tp.Iterator[Field]:
        """Apply the lambda function to the field."""
        yield self.fn(*fields, *self.args, **self.kwargs)

    def backward_transform(self, *fields: Field) -> tp.Iterator[Field]:
        """Apply the backward lambda function to the field."""
        yield self.backward_fn(*fields, *self.args, **self.kwargs)

    def _import_fn(self, fn: str) -> tp.Callable[..., Field]:
        try:
            module_name, fn_name = fn.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, fn_name)
        except Exception as e:
            raise ValueError(f"Could not import function {fn}") from e

    def __repr__(self):
        out = f"{self.__class__.__name__}(fn={self.fn},"
        if self.backward_fn:
            out += f"backward_fn={self.backward_fn},"
        out += f"param={self.param},"
        out += f"args={self.args},"
        out += f"kwargs={self.kwargs},"
        out += ")"
        return out