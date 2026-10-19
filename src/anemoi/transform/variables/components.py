# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Variables described by the earthkit-data 1.0 component vocabulary.

This module defines the ``variable/1`` serialisation schema: the
variable's attributes are stored under their earthkit-data component
names (``parameter.variable``, ``parameter.units``,
``vertical.level_type``, ``vertical.level``), so variables collected
from any earthkit-data source (GRIB, NetCDF, ...) serialise the same
way. MARS request metadata, when the source provides it, is kept under
the ``"mars"`` key (a sanctioned legacy surface — it feeds MARS request
generation and GRIB encoding).
"""

import logging
from abc import abstractmethod
from typing import Any
from typing import ClassVar
from typing import get_args

from anemoi.transform.units import Units
from anemoi.transform.variables import Variable
from anemoi.transform.variables.from_dict import VariableFromDictionary
from anemoi.transform.variables.schemas import VariableSchema
from anemoi.transform.variables.schemas import VariableSchemaV1

LOG = logging.getLogger(__name__)

# The serialisation schema implemented by VariableFromComponents,
# taken from the pydantic model's "schema" discriminator field.
SCHEMA = get_args(VariableSchemaV1.model_fields["schema_"].annotation)[0]

# Mapping from earthkit-data 1.0 level type names to MARS-style
# abbreviations. Level types without an entry map to None (unknown):
# `Variable.compatible` tolerates unknown level types, while a wrong
# `False` would fail hard against legacy metadata.
LEVEL_TYPE_MAPPING = {
    "surface": "sfc",
    "pressure": "pl",
    "model": "ml",
    "hybrid": "ml",
    "depth_below_ground_level": "sfc",
    "depth_below_land_layer": "sfc",
    "depth_below_land_level": "sfc",
    "height_above_ground": "sfc",
    "height_above_ground_level": "sfc",
    "mean_sea": "sfc",
    "entire_atmosphere": "sfc",
    "potential_vorticity": "pv",
    "potential_temperature": "pt",
}


class ComponentVariable(Variable):
    """Base for variables whose attributes are earthkit-data components.

    Subclasses provide :meth:`_component`, resolving a dotted component
    path (e.g. ``"vertical.level"``) to a value or None — from a live
    field or from a serialised dictionary.
    """

    @abstractmethod
    def _component(self, path: str) -> Any:
        """Resolve a dotted earthkit-data component path.

        Parameters
        ----------
        path : str
            The component path (e.g. ``"parameter.units"``).

        Returns
        -------
        Any
            The value, or None when the component is missing.
        """
        pass

    @property
    def _level_type(self) -> str | None:
        """The level type as a MARS-style abbreviation, or None when unknown or unmapped."""
        level_type = self._component("vertical.level_type")
        if level_type is None:
            return None
        return LEVEL_TYPE_MAPPING.get(level_type)

    @property
    def level(self) -> Any:
        """Get the level of the variable."""
        return self._component("vertical.level")

    @property
    def units(self):
        """Get the units of the variable (a :class:`Units`, or None if missing)."""
        units = self._component("parameter.units")
        return Units(str(units)) if units is not None else None

    @property
    def param(self) -> str:
        """Get the parameter name of the variable (falls back to its name)."""
        param = self._component("parameter.variable")
        return param if param is not None else super().param


class VariableFromComponents(ComponentVariable, VariableFromDictionary):
    """A variable deserialised from the ``variable/1`` schema.

    The dictionary is validated against
    :class:`~anemoi.transform.variables.schemas.VariableSchemaV1`;
    component values are read from the model's nested sections
    (``data.vertical.level`` for ``"vertical.level"``). The
    time-processing keys and create-time flags come from
    :class:`VariableFromDictionary`.
    """

    schema_model: ClassVar[type[VariableSchema]] = VariableSchemaV1

    def __init__(self, name: str, data: dict[str, Any] | VariableSchemaV1) -> None:
        """Initialize the variable with a name and data.

        Parameters
        ----------
        name : str
            The name of the variable.
        data : dict or VariableSchemaV1
            The serialised variable (``variable/1`` schema).
        """
        super().__init__(name, data)

    def _component(self, path: str) -> Any:
        """Resolve a dotted component path on the validated model.

        Extra (unmodelled) keys of a newer same-schema writer are
        reachable too — the models allow extras, and pydantic exposes
        them as attributes.

        Parameters
        ----------
        path : str
            The component path (e.g. ``"parameter.units"``).

        Returns
        -------
        Any
            The value, or None when the component is missing.
        """
        value: Any = self.data
        for part in path.split("."):
            value = getattr(value, part, None)
            if value is None:
                return None
        return value

    @property
    def grib_keys(self) -> dict[str, Any]:
        """Get MARS-style GRIB keys for the variable.

        Returns the stored ``"mars"`` request when the source provided
        one, otherwise MARS-style keys derived from the components (for
        data that was never GRIB, e.g. NetCDF).
        """
        if self.data.mars:
            return dict(self.data.mars)

        keys: dict[str, Any] = {}
        if (param := self._component("parameter.variable")) is not None:
            keys["param"] = param
        if (level_type := self._level_type) is not None:
            keys["levtype"] = level_type
        if (level := self.level) is not None:
            keys["levelist"] = level
        return keys
