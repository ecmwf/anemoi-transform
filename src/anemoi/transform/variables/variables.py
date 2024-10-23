# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from . import Variable


class VariableFromMarsVocabulary(Variable):
    """A variable that is defined by the Mars vocabulary."""

    def __init__(self, name, data: dict) -> None:
        super().__init__(name)
        self.data = data
        self.mars = self.data.get("mars", {})

    @property
    def is_pressure_level(self):
        return self.mars.get("levtype", None) == "pl"

    @property
    def level(self):
        return self.mars.get("levelist", None)

    @property
    def is_constant_in_time(self):
        return self.data.get("constant_in_time", False)

    @property
    def is_from_input(self):
        return "mars" in self.data

    @property
    def is_computed_forcing(self):
        return self.data.get("computed_forcing", False)


class VariableFromDict(VariableFromMarsVocabulary):
    """A variable that is defined by a user provided dictionary."""

    def __init__(self, name, data: dict) -> None:
        super().__init__(name, data)


class VariableFromEarthkit(VariableFromMarsVocabulary):
    """A variable that is defined by an EarthKit field."""

    def __init__(self, name, field, namespace="mars") -> None:
        super().__init__(name, field.metadata(namespace=namespace))
        self.field = field

    def is_pressure_level(self):
        return self.field.is_pressure_level()

    def level(self):
        return self.field.level()
