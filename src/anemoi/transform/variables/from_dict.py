# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Variables deserialised from metadata dictionaries (legacy MARS-vocabulary schema)."""

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Union

from anemoi.utils.dates import as_timedelta

from anemoi.transform.units import Units
from anemoi.transform.variables import Variable
from anemoi.transform.variables.schemas import LegacyVariableSchema
from anemoi.transform.variables.schemas import VariableSchema

if TYPE_CHECKING:
    from datetime import timedelta

LOG = logging.getLogger(__name__)


class VariableFromDictionary(Variable):
    """Base for variables deserialised from a metadata dictionary.

    The dictionary is validated against the class's pydantic model
    (``schema_model``); the schema-independent parts of the layout —
    the time-processing keys (``process``, ``period``) and the
    create-time flags (``computed_forcing``, ``constant_in_time``) —
    are implemented here, on the shared base model.
    """

    schema_model: ClassVar[type[VariableSchema]] = VariableSchema

    def __init__(self, name: str, data: dict[str, Any] | VariableSchema) -> None:
        """Initialize the variable with a name and data.

        Parameters
        ----------
        name : str
            The name of the variable.
        data : dict or VariableSchema
            The serialised variable; dictionaries are validated against
            the class's ``schema_model``.
        """
        super().__init__(name)
        if not isinstance(data, VariableSchema):
            data = self.schema_model.model_validate(data)
        self.data = data

    def as_dict(self) -> dict[str, Any]:
        """Serialise the variable back to a dictionary (round-trip identity).

        Returns
        -------
        dict
            The validated dictionary, with only explicitly-set keys.
        """
        return self.data.as_dict()

    @property
    def is_constant_in_time(self) -> bool:
        """Check if the variable is constant in time."""
        return self.data.constant_in_time

    @property
    def is_computed_forcing(self) -> bool:
        """Check if the variable is a computed forcing."""
        return self.data.computed_forcing

    @property
    def time_processing(self):
        """Get the time processing type of the variable."""
        return self.data.process

    @property
    def period(self) -> Union["timedelta", None]:
        """Get the variable's period as a timedelta.
        For instantaneous variables, returns a timedelta of 0. For non-instantaneous variables, returns `None` if this information is missing.
        """
        if self.is_instantaneous:
            return as_timedelta(0)

        if (period := self.data.period) is None:
            return None

        return period[1] - period[0]


class VariableFromMarsVocabulary(VariableFromDictionary):
    """A variable that is defined by the Mars vocabulary."""

    schema_model: ClassVar[type[VariableSchema]] = LegacyVariableSchema

    def __init__(self, name: str, data: dict[str, Any] | LegacyVariableSchema) -> None:
        """Initialize the variable with a name and data.

        Parameters
        ----------
        name : str
            The name of the variable.
        data : dict or LegacyVariableSchema
            The data defining the variable.
        """
        super().__init__(name, data)
        self.mars = self.data.mars

    @property
    def _level_type(self) -> str | None:
        """The MARS ``levtype`` of the variable, or None when unknown."""
        return self.mars.get("levtype", None)

    @property
    def level(self) -> str | None:
        """Get the level of the variable."""
        return self.mars.get("levelist", None)

    @property
    def units(self):
        """Get the units of the variable (a :class:`Units`, or None if missing)."""
        units = self.data.units
        return Units(units) if units else None

    @property
    def grib_keys(self) -> dict[str, Any]:
        """Get the GRIB keys of the variable."""
        return dict(self.mars)

    @property
    def param(self) -> str:
        """Get the parameter of the variable."""
        return self.mars.get("param", super().param)


class VariableFromDict(VariableFromMarsVocabulary):
    """A variable that is defined by a user provided dictionary."""

    def __init__(self, name: str, data: dict[str, Any] | LegacyVariableSchema) -> None:
        """Initialize the variable with a name and data.

        Parameters
        ----------
        name : str
            The name of the variable.
        data : dict or LegacyVariableSchema
            The data defining the variable.
        """
        super().__init__(name, data)
