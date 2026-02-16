# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import importlib
from collections.abc import Callable

from earthkit.data.core.fieldlist import Field

from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters import filter_registry


@filter_registry.register("earthkitfieldlambda")
class EarthkitFieldLambdaFilter(SingleFieldFilter):
    """A filter to apply an arbitrary function to individual fields.

    This filter allows you to apply an arbitrary
    Python function (either provided inline as a lambda or imported from a
    module) to fields selected by parameter name. This enables advanced and
    flexible transformations that aren't covered by built-in filters. This
    filter must follow a source or filter that provides the necessary
    parameter(s) as input. No assumptions are made about physical
    quantities, it is entirely user-defined.

    Notes
    -----

    This general purpose filter allows users to quickly prototype some data transformation.
    For an operational usage it is recommended to develop dedicated filters,
    that can be contributed to the project or developed
    as :ref:`plugins <anemoi-plugins:index-page>`.

    Examples
    --------

    For example, you can use it to convert temperatures from Kelvin to Celsius by subtracting a constant.

    .. code-block:: yaml

      input:
        pipe:
            - source:
            # mars, grib, netcdf, etc.
            # source attributes here
            # ...
            # Must load the input variable

            - earthkitfieldlambda:
                param: "2t" # Name of variable (input) to be transformed
                fn: "lambda x, s: x.clone(values=x.values - s)"
                fn_args: [273.15]

    """

    required_inputs = ("fn", "param")
    optional_inputs = {"fn_args": None, "fn_kwargs": None, "backward_fn": None}

    def prepare_filter(self):
        if self.fn_args is None:
            self.fn_args = []
        if self.fn_kwargs is None:
            self.fn_kwargs = {}

        if not isinstance(self.fn_args, list):
            raise ValueError("Expected 'fn_args' to be a list. " f"Got {self.fn_args} instead.")
        if not isinstance(self.fn_kwargs, dict):
            raise ValueError("Expected 'fn_kwargs' to be a dictionary. " f"Got {self.fn_kwargs} instead.")

        if isinstance(self.fn, str):
            self.fn = self._import_fn(self.fn)

        if isinstance(self.backward_fn, str):
            self.backward_fn = self._import_fn(self.backward_fn)

    def forward_select(self):
        return {"param": self.param}

    def forward_transform(self, field: Field) -> Field:
        """Apply the forward lambda function to a field."""
        return self.fn(field, *self.fn_args, **self.fn_kwargs)

    def backward_transform(self, field: Field) -> Field:
        """Apply the backward lambda function to a field."""
        if self.backward_fn is None:
            raise ValueError("Backward function is undefined.")
        return self.backward_fn(field, *self.fn_args, **self.fn_kwargs)

    @staticmethod
    def _import_fn(fn: str) -> Callable[..., Field]:
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
        """Return a string representation of the EarthkitFieldLambdaFilter."""

        out = f"{self.__class__.__name__}(fn={self.fn},"
        if self.backward_fn is not None:
            out += f"backward_fn={self.backward_fn},"
        out += f"param={self.param},"
        out += f"fn_args={self.fn_args},"
        out += f"fn_kwargs={self.fn_kwargs},"
        out += ")"
        return out
