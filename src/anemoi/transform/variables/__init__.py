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
        """Parameters
        -------------
        name : str
            The name of the variable.
        """
        self.name: str = name

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> Any:
        """Create a Variable instance from a dictionary.

        Parameters
        ----------
        name : str
            The name of the variable.
        data : Dict[str, Any]
            The data dictionary.

        Returns
        -------
        Any
            The created Variable instance.
        """
        from anemoi.transform.variables.variables import VariableFromDict

        return VariableFromDict(name, data)

    @classmethod
    def from_earthkit(cls, field: Any) -> Any:
        """Create a Variable instance from an Earthkit field.

        Parameters
        ----------
        field : Any
            The Earthkit field.

        Returns
        -------
        Any
            The created Variable instance.
        """
        from anemoi.transform.variables.variables import VariableFromEarthkit

        return VariableFromEarthkit(field)

    def __repr__(self) -> str:
        """Return a string representation of the Variable.

        Returns
        -------
        str
            The name of the variable.
        """
        return self.name

    def __hash__(self) -> int:
        """Return the hash of the Variable.

        Returns
        -------
        int
            The hash of the variable name.
        """
        return hash(self.name)

    def __eq__(self, other: Any) -> bool:
        """Check if two Variable instances are equal.

        Parameters
        ----------
        other : Any
            The other variable to compare with.

        Returns
        -------
        bool
            True if the variables are equal, False otherwise.
        """
        if not isinstance(other, Variable):
            return False
        return self.name == other.name

    @property
    @abstractmethod
    def is_pressure_level(self) -> bool:
        """Check if the variable is a pressure level."""
        pass

    @property
    @abstractmethod
    def is_model_level(self) -> bool:
        """Check if the variable is a pressure level."""
        pass

    @property
    @abstractmethod
    def is_surface_level(self) -> bool:
        """Check if the variable is on the surface."""
        pass

    @property
    @abstractmethod
    def level(self) -> Any:
        """Get the level of the variable."""
        pass

    @property
    @abstractmethod
    def is_constant_in_time(self) -> bool:
        """Check if the variable is constant in time."""
        pass

    @property
    @abstractmethod
    def is_instantanous(self) -> bool:
        """Check if the variable is instantaneous."""
        pass

    @property
    def is_valid_over_a_period(self) -> bool:
        """Check if the variable is valid over a period."""

        return not self.is_instantanous

    @property
    @abstractmethod
    def is_accumulation(self) -> bool:
        """Check if the variable is an accumulation."""
        pass

    @property
    def param(self) -> str:
        """Get the parameter name of the variable."""
        return self.name

    # This may need to move to a different class
    @property
    def grib_keys(self) -> Dict[str, Any]:
        """Get the GRIB keys for the variable."""
        raise NotImplementedError(f"Method `grib_keys` not implemented for {self.__class__.__name__}")

    @property
    def is_computed_forcing(self) -> bool:
        """Check if the variable is a computed forcing."""
        raise NotImplementedError(f"Method `is_computed_forcing` not implemented for {self.__class__.__name__}")

    @property
    def is_from_input(self) -> bool:
        """Check if the variable is from input."""
        pass

    def similarity(self, other: Any) -> int:
        """Compute the similarity between two variables. This is used when
        encoding a variable in GRIB and we do not have a template for it.
        We can then try to find the most similar variable for which we have a template.

        Parameters
        ----------
        other : Any
            The other variable to compare with.

        Returns
        -------
        int
            The similarity score.
        """
        return 0
