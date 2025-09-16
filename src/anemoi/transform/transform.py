# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import inspect
import textwrap
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import TypeVar

import earthkit.data as ekd
import yaml

T = TypeVar("T", bound="Transform")


class _TransformMetaClass(ABCMeta):

    # This metaclass adds a `reversed` property to all Transform subclasses.
    # And forwards the docstring of the original class to the reversed property.
    # This allows to document the reversed transform in the same way as the original one.

    @property
    def reversed(cls: type[T]) -> Callable[..., "ReversedTransform"]:

        def wrap_reversed(*args: Any, **kwargs: Any) -> "ReversedTransform":
            return ReversedTransform(cls(*args, **kwargs))

        def wrap_reversed_str(*args, **kwargs) -> str:
            return cls.documentation(*args, **kwargs)

        wrap_reversed.documentation = wrap_reversed_str

        return wrap_reversed


class Transform(ABC, metaclass=_TransformMetaClass):
    """Abstract base class for all transformations."""

    def __new__(cls, *args: Any, **kwargs: Any) -> "Transform":
        print(f"Creating instance of {cls.__name__} with args={args}, kwargs={kwargs}")
        instance = super().__new__(cls)
        return instance

    def __repr__(self) -> str:
        """Returns a string representation of the transform.

        Returns
        -------
        str
            A string representation of the transform.
        """
        return f"{self.__class__.__name__}()"

    def __call__(self, data: ekd.Field = None) -> ekd.Field:
        """Applies the forward transformation to the data.

        Parameters
        ----------
        data : ekd.Field, optional
            The input data to be transformed.

        Returns
        -------
        Any
            The transformed data.
        """
        return self.forward(data)

    @abstractmethod
    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Applies the forward transformation to the data.

        Parameters
        ----------
        data : ekd.FieldList
            The input data to be transformed.

        Returns
        -------
        ekd.FieldList
            The transformed data.
        """
        pass

    def backward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Applies the backward transformation to the data.

        Parameters
        ----------
        data : ekd.FieldList
            The input data to be transformed.

        Returns
        -------
        ekd.FieldList
            The transformed data.
        """
        raise NotImplementedError(f"{self} is not reversible.")

    def reverse(self) -> "Transform":
        """Returns a transform that applies the backward transformation.

        Returns
        -------
        Transform
            A transform that applies the backward transformation.
        """
        return ReversedTransform(self)

    def __or__(self, other: "Transform") -> "Transform":
        """Combines two transforms into a pipeline.

        Parameters
        ----------
        other : Transform
            The other transform to combine with.

        Returns
        -------
        Transform
            A pipeline transform.
        """
        from anemoi.transform.workflows import workflow_registry

        return workflow_registry.create("pipeline", filters=[self, other])

    def patch_data_request(self, data_request: Any) -> Any:
        """Patch the data request with additional information.

        Parameters
        ----------
        data_request : Any
            The data request to patch.

        Returns
        -------
        Any
            The patched data request.
        """
        return data_request

    @classmethod
    def documentation(cls, filter_name) -> str:
        result = (cls.__doc__ or "").splitlines()
        # Extract __init__ parameters and add them to the documentation

        examples = []
        examples.append("")
        examples.append("Examples")
        examples.append("--------")
        examples.append("")
        examples.append(
            """
To use this filter in a dataset recipe, include it as show below, adjusting parameters as needed.
The `source` could be `mars`, `grib`, `netcdf`, etc. See the `anemoi-datasets documentation <https://anemoi.readthedocs.io/l>`_ for more details.
"""
        )

        sig = inspect.signature(cls.__init__)
        params = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.default is inspect.Parameter.empty:
                params[name] = f"...{name}..."
            else:
                params[name] = param.default

        dataset_example = {
            "input": {
                "pipe": [{"source": {"param1": "value1", "param2": "value2", "param3": "..."}}, {filter_name: params}]
            }
        }

        dataset_example = yaml.dump(dataset_example, width=70, sort_keys=False)
        dataset_example = textwrap.indent(dataset_example, "  ")

        examples.append(".. code-block:: yaml")
        examples.append("")
        examples.append(dataset_example)

        examples.append("")

        result.extend(examples)

        return "\n".join(result)


class ReversedTransform(Transform):
    """Swap the forward and backward methods of a filter."""

    def __init__(self, filter: Transform) -> None:
        """Initializes the reversed transform.

        Parameters
        ----------
        filter : Transform
            The transform to be reversed.
        """
        self.filter = filter

    def __repr__(self) -> str:
        """Returns a string representation of the reversed transform.

        Returns
        -------
        str
            A string representation of the reversed transform.
        """
        return f"Reversed({self.filter})"

    def forward(self, x: Any) -> Any:
        """Applies the backward transformation to the data.

        Parameters
        ----------
        x : Any
            The input data to be transformed.

        Returns
        -------
        Any
            The transformed data.
        """
        return self.filter.backward(x)

    def backward(self, x: Any) -> Any:
        """Applies the forward transformation to the data.

        Parameters
        ----------
        x : Any
            The input data to be transformed.

        Returns
        -------
        Any
            The transformed data.
        """
        return self.filter.forward(x)

    def patch_data_request(self, data_request: Any) -> Any:
        """Patch the data request with additional information.

        Parameters
        ----------
        data_request : Any
            The data request to patch.

        Returns
        -------
        Any
            The patched data request.
        """
        return self.filter.patch_data_request(data_request)

    @classmethod
    def repr_rst(cls, reversed: bool = False) -> str:
        return cls.filter.repr_rst()
