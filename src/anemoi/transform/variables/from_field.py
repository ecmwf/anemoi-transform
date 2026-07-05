# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Union

from anemoi.transform.fields import METADATA_KEY_MAPPING
from anemoi.transform.units import Units
from anemoi.transform.variables import Variable

if TYPE_CHECKING:
    from datetime import timedelta

LOG = logging.getLogger(__name__)

# Mapping from earthkit-data 1.0 level type names to MARS-style abbreviations.
_LEVEL_TYPE_MAPPING = {
    "surface": "sfc",
    "pressure": "pl",
    "model": "ml",
    "hybrid": "ml",
    "depth_below_ground_level": "sfc",
    "height_above_ground": "sfc",
    "potential_vorticity": "pv",
    "potential_temperature": "pt",
}

# The MARS-style GRIB keys reported by ``grib_keys`` (component paths come
# from the canonical ``METADATA_KEY_MAPPING``).
_GRIB_KEYS = ("param", "levtype", "levelist", "step", "number")


class VariableFromField(Variable):
    """A variable described by a :class:`~anemoi.transform.fields.Field`.

    The variable's attributes are read from the field's components on
    demand — there is no intermediate metadata dictionary or delegate
    object.
    """

    def __init__(self, name: str, field: Any) -> None:
        """Initialize the variable with a name and a field.

        Parameters
        ----------
        name : str
            The name of the variable.
        field : Field
            The field describing the variable.
        """
        super().__init__(name)
        self.field = field

    @property
    def _level_type(self) -> str | None:
        """The field's level type, as a MARS-style abbreviation (or None)."""
        level_type = self.field.get("vertical.level_type", default=None)
        return _LEVEL_TYPE_MAPPING.get(level_type)

    @property
    def level(self) -> Any:
        """Get the level of the variable."""
        return self.field.get("vertical.level", default=None)

    @property
    def is_pressure_level(self) -> bool:
        """Check if the variable is on a pressure level."""
        level_type = self._level_type
        if level_type is None:
            return None
        return level_type == "pl"

    @property
    def is_model_level(self) -> bool:
        """Check if the variable is on a model level."""
        level_type = self._level_type
        if level_type is None:
            return None
        return level_type == "ml"

    @property
    def is_surface_level(self) -> bool:
        """Check if the variable is on the surface."""
        level_type = self._level_type
        if level_type is None:
            return None
        return level_type == "sfc"

    @property
    def is_constant_in_time(self) -> bool:
        """Check if the variable is constant in time (unknown for a single field)."""
        return None

    @property
    def is_instantanous(self) -> bool:
        """Check if the variable is instantaneous (unknown when the field has no statistical process)."""
        process = self.time_processing
        if process is None:
            return None
        return False

    @property
    def is_valid_over_a_period(self) -> bool:
        """Check if the variable is valid over a period (e.g. accumulated or averaged)."""
        process = self.time_processing
        if process is None:
            return None
        return True

    @property
    def time_processing(self):
        """Get the time processing type of the variable (e.g. ``"accumulation"``)."""
        return self.field.get("time.statistical_process", default=None)

    @property
    def period(self) -> Union["timedelta", None]:
        """Get the variable's period as a timedelta.

        For instantaneous variables, returns a timedelta of 0. For
        non-instantaneous variables, returns ``None`` if this information
        is missing.
        """
        return None

    @property
    def is_accumulation(self) -> bool:
        """Check if the variable is an accumulation."""
        process = self.time_processing
        if process is None:
            return None
        return process == "accumulation"

    @property
    def param(self) -> str:
        """Get the parameter name of the variable."""
        return self.field.param

    # This may need to move to a different class
    @property
    def grib_keys(self) -> dict[str, Any]:
        """Get MARS-style GRIB keys for the variable, from the field's components."""
        keys = {}
        for grib_key in _GRIB_KEYS:
            value = self.field.get(METADATA_KEY_MAPPING[grib_key], default=None)
            if value is not None:
                keys[grib_key] = value
        if "levtype" in keys:
            keys["levtype"] = _LEVEL_TYPE_MAPPING.get(keys["levtype"], keys["levtype"])
        return keys

    @property
    def is_computed_forcing(self) -> bool:
        """Check if the variable is a computed forcing."""
        raise NotImplementedError()

    @property
    def units(self):
        """Get the units of the variable."""
        units = self.field.get("parameter.units", default=None)
        return Units(str(units)) if units is not None else None
