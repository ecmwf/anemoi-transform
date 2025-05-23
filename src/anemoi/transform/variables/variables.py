# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Dict
from typing import Union

from anemoi.transform.variables import Variable


class VariableFromMarsVocabulary(Variable):
    """A variable that is defined by the Mars vocabulary."""

    def __init__(self, name: str, data: Dict[str, Any]) -> None:
        """Initialize the variable with a name and data.

        Parameters
        ----------
        name : str
            The name of the variable.
        data : dict
            The data defining the variable.
        """
        super().__init__(name)
        self.data = data
        self.mars = self.data.get("mars", {})

    @property
    def is_surface_level(self) -> bool:
        """Check if the variable is at a surface level."""
        return self.mars.get("levtype", None) == "sfc"

    @property
    def is_pressure_level(self) -> bool:
        """Check if the variable is at a pressure level."""
        return self.mars.get("levtype", None) == "pl"

    @property
    def is_model_level(self) -> bool:
        """Check if the variable is at a model level."""
        return self.mars.get("levtype", None) == "ml"

    @property
    def level(self) -> Union[str, None]:
        """Get the level of the variable."""
        return self.mars.get("levelist", None)

    @property
    def is_constant_in_time(self) -> bool:
        """Check if the variable is constant in time."""
        return self.data.get("constant_in_time", False)

    @property
    def is_from_input(self) -> bool:
        """Check if the variable is from input data."""
        return "mars" in self.data

    @property
    def is_computed_forcing(self) -> bool:
        """Check if the variable is a computed forcing."""
        return self.data.get("computed_forcing", False)

    @property
    def is_accumulation(self) -> bool:
        """Check if the variable is an accumulation."""
        return self.data.get("process") == "accumulation"

    @property
    def is_instantanous(self) -> bool:
        """Check if the variable is instantaneous."""
        return "process" not in self.data

    @property
    def time_processing(self):
        """Get the time processing type of the variable."""
        return self.data.get("process")

    @property
    def grib_keys(self) -> Dict[str, Any]:
        """Get the GRIB keys of the variable."""
        return self.data.get("mars", {}).copy()

    @property
    def param(self) -> str:
        """Get the parameter of the variable."""
        return self.mars.get("param", super().param)

    def similarity(self, other: Any) -> int:
        """Calculate the similarity between this variable and another.

        Parameters
        ----------
        other : Any
            The other variable to compare with.

        Returns
        -------
        int
            The similarity score between the two variables.
        """
        if not isinstance(other, VariableFromMarsVocabulary):
            return 0

        def __similarity(a: Any, b: Any) -> int:
            if isinstance(a, dict) and isinstance(b, dict):
                return sum(__similarity(a[k], b[k]) for k in set(a.keys()) & set(b.keys()))

            if isinstance(a, list) and isinstance(b, list):
                return sum(__similarity(a[i], b[i]) for i in range(min(len(a), len(b))))

            return 1 if a == b else 0

        return __similarity(self.data, other.data)


class VariableFromDict(VariableFromMarsVocabulary):
    """A variable that is defined by a user provided dictionary."""

    def __init__(self, name: str, data: Dict[str, Any]) -> None:
        """Initialize the variable with a name and data.

        Parameters
        ----------
        name : str
            The name of the variable.
        data : dict
            The data defining the variable.
        """
        super().__init__(name, data)


class VariableFromEarthkit(VariableFromMarsVocabulary):
    """A variable that is defined by an EarthKit field."""

    def __init__(self, name: str, field: Any, namespace: str = "mars") -> None:
        """Initialize the variable with a name, field, and namespace.

        Parameters
        ----------
        name : str
            The name of the variable.
        field : Any
            The EarthKit field defining the variable.
        namespace : str, optional
            The namespace for the field metadata, by default "mars".
        """
        super().__init__(name, field.metadata(namespace=namespace))
        self.field = field

    @property
    def is_pressure_level(self) -> bool:
        """Check if the variable is at a pressure level."""
        return self.field.is_pressure_level()

    @property
    def level(self) -> Any:
        """Get the level of the variable."""
        return self.field.level()


class PostProcessedVariable(VariableFromMarsVocabulary):
    """A variable that is defined by a post-processed dictionary."""

    def __init__(self, name: str, data: Dict[str, Any]) -> None:
        """Initialize the variable with a name and data.

        Parameters
        ----------
        name : str
            The name of the variable.
        data : dict
            The data defining the variable.
        """
        super().__init__(name, data)
