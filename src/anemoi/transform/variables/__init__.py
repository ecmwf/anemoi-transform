# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from abc import ABC
from abc import abstractmethod


class Variable(ABC):
    """Variable is a class that represents a variable during
    training and inference.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def from_dict(cls, name, data: dict):
        from .variables import VariableFromDict

        return VariableFromDict(name, data)

    @classmethod
    def from_earthkit(cls, field):
        from .variables import VariableFromEarthkit

        return VariableFromEarthkit(field)

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.name == other.name

    @property
    @abstractmethod
    def is_pressure_level(self):
        pass

    @property
    @abstractmethod
    def level(self):
        pass

    @property
    @abstractmethod
    def is_constant_in_time(self):
        pass
