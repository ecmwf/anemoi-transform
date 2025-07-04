# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Union

from earthkit.data.core.fieldlist import Field

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


@filter_registry.register("earthkitfieldlambda")
class EarthkitFieldLambdaFilter(MatchingFieldsFilter):
    """A filter to apply an arbitrary function to individual fields."""

    @matching(
        select="param",
        forward="param",
        backward="param",
    )
    def __init__(
        self,
        fn: Union[str, Callable[[Field, Any], Field]],
        param: Union[str, list[str]],
        fn_args: list = [],
        fn_kwargs: Dict[str, Any] = {},
        backward_fn: Optional[Union[str, Callable[[Field, Any], Field]]] = None,
    ) -> None:
        """Initialize the EarthkitFieldLambdaFilter.

        Parameters
        ----------
        fn : Union[str, Callable[[Field, Any], Field]]
            The lambda function as a callable with the general signature
            `fn(*earthkit.data.Field, *args, **kwargs) -> earthkit.data.Field` or
            a string path to the function, such as "package.module.function".
        param : Union[str, list[str]]
            The parameter name or list of parameter names to apply the function to.
        fn_args : list
            The list of arguments to pass to the lambda function.
        fn_kwargs : Dict[str, Any]
            The dictionary of keyword arguments to pass to the lambda function.
        backward_fn : Optional[Union[str, Callable[[Field, Any], Field]]], optional
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
        ...     fn_args=[273.15],
        ... )
        >>> fields = kelvin_to_celsius.forward(fields)
        """

        if not isinstance(fn_args, list):
            raise ValueError("Expected 'fn_args' to be a list. " f"Got {fn_args} instead.")
        if not isinstance(fn_kwargs, dict):
            raise ValueError("Expected 'fn_kwargs' to be a dictionary. " f"Got {fn_kwargs} instead.")

        self.fn = self._import_fn(fn) if isinstance(fn, str) else fn

        if isinstance(backward_fn, str):
            self.backward_fn = self._import_fn(backward_fn)
        else:
            self.backward_fn = backward_fn

        self.param = param
        self.fn_args = fn_args
        self.fn_kwargs = fn_kwargs

    def forward_transform(self, *fields: Field) -> Iterator[Field]:
        """Apply the forward lambda function to the fields.

        Parameters
        ----------
        fields : Field
            The fields to apply the forward lambda function to.

        Returns
        -------
        Iterator[Field]
            Transformed fields.
        """
        yield self.fn(*fields, *self.fn_args, **self.fn_kwargs)

    def backward_transform(self, *fields: Field) -> Iterator[Field]:
        """Apply the backward lambda function to the fields.

        Parameters
        ----------
        fields : Field
            The fields to apply the backward lambda function to.

        Returns
        -------
        Iterator[Field]
            Transformed fields.
        """
        yield self.backward_fn(*fields, *self.fn_args, **self.fn_kwargs)

    def _import_fn(self, fn: str) -> Callable[..., Field]:
        """Import a function from a string path.

        Parameters
        ----------
        fn : str
            The string path to the function, such as "package.module.function".

        Returns
        -------
        Callable[..., Field]
            The imported function.

        Raises
        ------
        ValueError
            If the function cannot be imported.
        """
        try:
            module_name, fn_name = fn.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, fn_name)
        except Exception as e:
            raise ValueError(f"Could not import function {fn}") from e

    def __repr__(self) -> str:
        """Return a string representation of the EarthkitFieldLambdaFilter.

        Returns
        -------
        str
            The string representation of the filter.
        """
        out = f"{self.__class__.__name__}(fn={self.fn},"
        if self.backward_fn:
            out += f"backward_fn={self.backward_fn},"
        out += f"param={self.param},"
        out += f"fn_args={self.fn_args},"
        out += f"fn_kwargs={self.fn_kwargs},"
        out += ")"
        return out
