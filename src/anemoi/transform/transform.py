# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from abc import ABC
from abc import abstractmethod
from typing import Any
from typing import Type
from typing import TypeVar

import earthkit.data as ekd

T = TypeVar("T", bound="Transform")


class Transform(ABC):
    """Abstract base class for all transformations."""

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

    @classmethod
    def reversed(cls: Type[T], *args: Any, **kwargs: Any) -> "ReversedTransform":
        """Creates a reversed transform.

        Parameters
        ----------
        args : Any
            Positional arguments for the transform.
        kwargs : Any
            Keyword arguments for the transform.

        Returns
        -------
        ReversedTransform
            A reversed transform.
        """
        return ReversedTransform(cls(*args, **kwargs))

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
