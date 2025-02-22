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

from . import Variable


class VariableFromMarsVocabulary(Variable):
    """A variable that is defined by the Mars vocabulary."""

    def __init__(self, name: str, data: Dict[str, Any]) -> None:
        super().__init__(name)
        self.data = data
        self.mars = self.data.get("mars", {})

    @property
    def is_pressure_level(self) -> bool:
        return self.mars.get("levtype", None) == "pl"

    @property
    def level(self) -> Any:
        return self.mars.get("levelist", None)

    @property
    def is_constant_in_time(self) -> bool:
        return self.data.get("constant_in_time", False)

    @property
    def is_from_input(self) -> bool:
        return "mars" in self.data

    @property
    def is_computed_forcing(self) -> bool:
        return self.data.get("computed_forcing", False)

    @property
    def is_accumulation(self) -> bool:
        return self.data.get("process") == "accumulation"

    @property
    def is_instantanous(self) -> bool:
        return "process" not in self.data.get

    @property
    def grib_keys(self) -> Dict[str, Any]:
        return self.data.get("mars", {}).copy()

    def similarity(self, other: Any) -> int:
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
        super().__init__(name, data)


class VariableFromEarthkit(VariableFromMarsVocabulary):
    """A variable that is defined by an EarthKit field."""

    def __init__(self, name: str, field: Any, namespace: str = "mars") -> None:
        super().__init__(name, field.metadata(namespace=namespace))
        self.field = field

    def is_pressure_level(self) -> bool:
        return self.field.is_pressure_level()

    def level(self) -> Any:
        return self.field.level()
