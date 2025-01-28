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


class Transform(ABC):

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"

    def __call__(self, data=None):
        return self.forward(data)

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def backward(self, data):
        pass

    def reverse(self) -> "Transform":
        return ReversedTransform(self)

    @classmethod
    def reversed(cls, *args, **kwargs):
        return ReversedTransform(cls(*args, **kwargs))

    def __or__(self, other):
        from .workflows import workflow_registry

        return workflow_registry.create("pipeline", filters=[self, other])

    def forward_processor(self, state):
        raise NotImplementedError("Not implemented")

    def backward_processor(self, state):
        raise NotImplementedError("Not implemented")


class ReversedTransform(Transform):
    """Swap the forward and backward methods of a filter."""

    def __init__(self, filter) -> None:
        self.filter = filter

    def __repr__(self) -> str:
        return f"Reversed({self.filter})"

    def forward(self, x):
        return self.filter.backward(x)

    def backward(self, x):
        return self.filter.forward(x)

    def forward_processor(self, state):
        return self.filter.backward_processor(state)

    def backward_processor(self, state):
        return self.filter.forward_processor(state)
