# (C) Copyright 2024 Anemoi contributors.
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

from anemoi.transform.units import Units
from anemoi.transform.variables import Variable

if TYPE_CHECKING:
    from datetime import timedelta

LOG = logging.getLogger(__name__)


class VariableFromEarthkit(Variable):
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
        super().__init__(name)
        metadata = field.metadata(namespace=namespace)
        self.delegate = Variable.from_dict(name, dict(mars=metadata))
        self.field = field

    @property
    def level(self) -> Any:
        """Get the level of the variable."""
        return self.field.level()

    @property
    def is_pressure_level(self) -> bool:
        """Check if the variable is a pressure level."""
        return self.delegate.is_pressure_level

    @property
    def is_model_level(self) -> bool:
        """Check if the variable is a pressure level."""
        return self.delegate.is_model_level

    @property
    def is_surface_level(self) -> bool:
        """Check if the variable is on the surface."""
        return self.delegate.is_surface_level

    @property
    def is_constant_in_time(self) -> bool:
        """Check if the variable is constant in time."""
        return None

    @property
    def is_instantanous(self) -> bool:
        """Check if the variable is instantaneous."""
        return None

    @property
    def is_valid_over_a_period(self) -> bool:
        """Check if the variable is valid over a period."""

        return None

    @property
    def time_processing(self):
        """Get the time processing type of the variable."""
        return None

    @property
    def period(self) -> Union["timedelta", None]:
        """Get the variable's period as a timedelta.
        For instantaneous variables, returns a timedelta of 0. For non-instantaneous variables, returns `None` if this information is missing.
        """
        return None

    @property
    def is_accumulation(self) -> bool:
        """Check if the variable is an accumulation."""
        return None

    @property
    def param(self) -> str:
        """Get the parameter name of the variable."""
        return self.delegate.param

    # This may need to move to a different class
    @property
    def grib_keys(self) -> dict[str, Any]:
        """Get the GRIB keys for the variable."""
        return self.delegate.grib_keys

    @property
    def is_computed_forcing(self) -> bool:
        """Check if the variable is a computed forcing."""
        raise NotImplementedError()

    @property
    def units(self):
        """Get the units of the variable."""
        units = self.field.metadata("units", default=None)
        return Units(units) if units is not None else None
