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
from typing import Dict


class Variable(ABC):
    """Variable is a class that represents a variable during
    training and inference.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> Any:
        from .variables import VariableFromDict

        return VariableFromDict(name, data)

    @classmethod
    def from_earthkit(cls, field: Any) -> Any:
        from .variables import VariableFromEarthkit

        return VariableFromEarthkit(field)

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Variable):
            return False
        return self.name == other.name

    @property
    @abstractmethod
    def is_pressure_level(self) -> bool:
        pass

    @property
    @abstractmethod
    def level(self) -> Any:
        pass

    @property
    @abstractmethod
    def is_constant_in_time(self) -> bool:
        pass

    @property
    @abstractmethod
    def is_instantanous(self) -> bool:
        pass

    @property
    def is_valid_over_a_period(self) -> bool:
        return not self.is_instantanous

    @property
    @abstractmethod
    def is_accumulation(self) -> bool:
        pass

    # This may need to move to a different class
    @property
    def grib_keys(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Method `grib_keys` not implemented for {self.__class__.__name__}")

    @property
    def is_computed_forcing(self) -> bool:
        raise NotImplementedError(f"Method `is_computed_forcing` not implemented for {self.__class__.__name__}")

    @property
    def is_from_input(self) -> bool:
        pass

    def similarity(self, other: Any) -> int:
        """Compute the similarity between two variables. This is used when
        encoding a variables in GRIB and we do not have a template for it.
        We can then try to find the most similar variable for which we have a template.
        """
        return 0
