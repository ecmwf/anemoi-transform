# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from collections.abc import Iterator
from typing import Literal

import earthkit.data as ekd
import numpy as np
from earthkit.meteo import thermo
from earthkit.meteo import vertical
from numpy.typing import NDArray

from ..constants import model_level_AB as predefined_AB
from . import filter_registry
from .matching import MatchingFieldsFilter
from .matching import matching

# Protection against zero relative or specific humidity when calculating dewpoint temperature
EPS_SPECIFIC = 1.0e-8


def _set_AB(model_level_AB: str | dict[str, NDArray]) -> tuple:
    if isinstance(model_level_AB, str):
        model_level_AB = model_level_AB.upper()
        try:
            model_level_AB = predefined_AB[model_level_AB]
        except KeyError:
            raise KeyError(
                "%s is not in the list of predefined AB-coefficients. Possible options are %s."
                % (model_level_AB, ", ".join(predefined_AB.keys()))
            )
    if not isinstance(model_level_AB, dict):
        raise TypeError("model_level_AB must be a string or a dictionary.")
    return (np.array(model_level_AB["A"]), np.array(model_level_AB["B"]))


def _check_consistency(A: NDArray, B: NDArray, model_level_fields: dict[str, ekd.FieldList]):
    # Assert that A and B coefficient have the same shape.
    assert A.shape == B.shape, "A and B coefficients must have same shape"
    for name, field in model_level_fields.items():
        # Assert that model levels are passed
        assert all(item == "ml" for item in field.metadata("levtype")), "Field {} does not contain model levels".format(
            name,
        )
        # Assert that A and B coefficients have one more vertical level than the model level field
        assert (
            A.shape[-1] == field.to_numpy().shape[0] + 1
        ), f"model level AB-coefficients should have one more vertical level than {name}"


class SpecificToRelativeAtHeightLevel(MatchingFieldsFilter):
    """A filter to convert specific humidity (kg/kg) to relative humidity (%)
    at a specified height level (in meters) with standard thermodynamical formulas
    """

    @matching(
        select="param",
        forward=(
            "specific_humidity_at_height_level",
            "temperature_at_height_level",
            "surface_pressure",
            "specific_humidity_at_model_levels",
            "temperature_at_model_levels",
        ),
        backward=(
            "relative_humidity_at_height_level",
            "temperature_at_height_level",
            "surface_pressure",
            "specific_humidity_at_model_levels",
            "temperature_at_model_levels",
        ),
        vertical=True,
    )
    def __init__(
        self,
        *,
        height: float = 2.0,
        specific_humidity_at_height_level: str = "2q",
        relative_humidity_at_height_level: str = "2r",
        temperature_at_height_level: str = "2t",
        surface_pressure: str = "sp",
        specific_humidity_at_model_levels: str = "q",
        temperature_at_model_levels: str = "t",
        model_level_AB: str | dict[str, NDArray],
        return_inputs: Literal["all", "none"] | list[str] = [
            "specific_humidity_at_height_level",
            "relative_humidity_at_height_level",
            "temperature_at_height_level",
            "surface_pressure",
        ],
    ):
        """Initializes the filter for converting specific humidity (kg/kg) to relative humidity (%) at a specified height.

        Parameters:
        -----------
        height : float, optional
            Height level in meters where the conversion is performed, by default is 2.0.
        specific_humidity_at_height_level : str, optional
            Name of the variable for specific humidity at the given height, by default "2q".
        relative_humidity_at_height_level : str, optional
            Name of the variable for relative humidity at the given height, by default "2r".
        temperature_at_height_level : str, optional
            Name of the variable for temperature at the given height, by default "2t".
        surface_pressure : str, optional
            Name of the variable for surface pressure, by default "sp".
        specific_humidity_at_model_levels : str, optional
            Name of the variable for specific humidity at model levels, by default "q".
        temperature_at_model_levels : str, optional
            Name of the variable for temperature at model levels, by default "t".
        AB : str | dict[str, NDArray]
            A string key for predefined A and B coefficients or a dictionary with "A" and "B" arrays for vertical interpolation.
            Possible predefined keys are: "IFS_137".
        return_inputs : Literal["all", "none"] | list[str], optional
            List of which filter inputs should be returned, by default ["specific_humidity_at_height_level", "relative_humidity_at_height_level", "temperature_at_height_level", "surface_pressure"]
        """

        self.return_inputs = return_inputs
        self.height = float(height)
        self.specific_humidity_at_height_level = specific_humidity_at_height_level
        self.relative_humidity_at_height_level = relative_humidity_at_height_level
        self.temperature_at_height_level = temperature_at_height_level
        self.surface_pressure = surface_pressure
        self.specific_humidity_at_model_levels = specific_humidity_at_model_levels
        self.temperature_at_model_levels = temperature_at_model_levels

        self.A, self.B = _set_AB(model_level_AB)

    def _get_pressure_at_height_level(
        self,
        temperature_at_model_levels: NDArray,
        specific_humidity_at_model_levels: NDArray,
        surface_pressure: NDArray,
    ) -> NDArray:

        return vertical.pressure_at_height_levels(
            height=self.height,
            t=temperature_at_model_levels,
            q=specific_humidity_at_model_levels,
            sp=surface_pressure,
            A=self.A,
            B=self.B,
        )

    def forward_transform(
        self,
        specific_humidity_at_height_level: ekd.Field,
        temperature_at_height_level: ekd.Field,
        surface_pressure: ekd.Field,
        specific_humidity_at_model_levels: ekd.FieldList,
        temperature_at_model_levels: ekd.FieldList,
    ) -> Iterator[ekd.Field]:
        """This will return the relative humidity along with temperature from specific humidity and temperature"""

        # Check vertical consistency
        _check_consistency(
            self.A,
            self.B,
            {
                self.specific_humidity_at_model_levels: specific_humidity_at_model_levels,
                self.temperature_at_model_levels: temperature_at_model_levels,
            },
        )

        # Make sure model levels are ordered ascending (highest level first):
        specific_humidity_at_model_levels = specific_humidity_at_model_levels.order_by(level="ascending")
        temperature_at_model_levels = temperature_at_model_levels.order_by(level="ascending")

        pressure_at_height_level = self._get_pressure_at_height_level(
            temperature_at_model_levels.to_numpy(),
            specific_humidity_at_model_levels.to_numpy(),
            surface_pressure.to_numpy(),
        )

        # If we want to take into account the mixed / ice phase when T ~ 0C / T < 0C
        # Then it is best to go through td: q --> td --> rh. (see https://github.com/ecmwf/earthkit-meteo/issues/15)
        # However, going straight to relative humidity seems to be a closer match to the RH values calculated by IFS

        relative_humidity_at_height_level = thermo.relative_humidity_from_specific_humidity(
            t=temperature_at_height_level.to_numpy(),
            q=specific_humidity_at_height_level.to_numpy(),
            p=pressure_at_height_level,
        )

        # Return the fields
        yield self.new_field_from_numpy(
            relative_humidity_at_height_level,
            template=specific_humidity_at_height_level,
            param=self.relative_humidity_at_height_level,
        )

    def backward_transform(
        self,
        relative_humidity_at_height_level: ekd.Field,
        temperature_at_height_level: ekd.Field,
        surface_pressure: ekd.Field,
        specific_humidity_at_model_levels: ekd.FieldList,
        temperature_at_model_levels: ekd.FieldList,
    ) -> Iterator[ekd.Field]:
        """This will return the specific humidity along with temperature from relative humidity and temperature"""

        # Check vertical consistency
        _check_consistency(
            self.A,
            self.B,
            {
                self.specific_humidity_at_model_levels: specific_humidity_at_model_levels,
                self.temperature_at_model_levels: temperature_at_model_levels,
            },
        )

        # Make sure model levels are ordered ascending (highest level first):
        specific_humidity_at_model_levels = specific_humidity_at_model_levels.order_by(level="ascending")
        temperature_at_model_levels = temperature_at_model_levels.order_by(level="ascending")

        pressure_at_height_level = self._get_pressure_at_height_level(
            temperature_at_model_levels.to_numpy(),
            specific_humidity_at_model_levels.to_numpy(),
            surface_pressure.to_numpy(),
        )

        specific_humidity_at_height_level = thermo.specific_humidity_from_relative_humidity(
            t=temperature_at_height_level.to_numpy(),
            r=relative_humidity_at_height_level.to_numpy(),
            p=pressure_at_height_level,
        )

        yield self.new_field_from_numpy(
            specific_humidity_at_height_level,
            template=relative_humidity_at_height_level,
            param=self.specific_humidity_at_height_level,
        )


filter_registry.register("q_to_r_height", SpecificToRelativeAtHeightLevel)
filter_registry.register("r_to_q_height", SpecificToRelativeAtHeightLevel.reversed)


class SpecificToDewpointAtHeightLevel(MatchingFieldsFilter):
    """A filter to convert specific humidity (kg/kg) to dewpoint temperature (K)
    at a specified height level (in meters) with standard thermodynamical formulas

    Notes
    -----
    For more information, see the :func:`dewpoint_from_specific_humidity <earthkit.meteo.thermo.array.dewpoint_from_specific_humidity>`
    and the :func:`specific_humidity_from_dewpoint <earthkit.meteo.thermo.array.specific_humidity_from_dewpoint>`
    functions in the earthkit-meteo documentation.

    """

    @matching(
        select="param",
        forward=(
            "specific_humidity_at_height_level",
            "surface_pressure",
            "specific_humidity_at_model_levels",
            "temperature_at_model_levels",
        ),
        backward=(
            "dewpoint_temperature_at_height_level",
            "surface_pressure",
            "specific_humidity_at_model_levels",
            "temperature_at_model_levels",
        ),
        vertical=True,
    )
    def __init__(
        self,
        *,
        height: float = 2.0,
        specific_humidity_at_height_level: str = "2q",
        dewpoint_temperature_at_height_level: str = "2d",
        surface_pressure: str = "sp",
        specific_humidity_at_model_levels: str = "q",
        temperature_at_model_levels: str = "t",
        model_level_AB: str | dict,
        return_inputs: Literal["all", "none"] | list[str] = [
            "specific_humidity_at_height_level",
            "dewpoint_temperature_at_height_level",
            "surface_pressure",
        ],
    ):
        """Initializes the filter for transforming specific humidity at a given height to dewpoint temperature.

        Parameters
        ----------
        height : float, optional
            Height level in meters where the conversion is performed, by default is 2.0.
        specific_humidity_at_height_level : str, optional
            Name of the variable for specific humidity at the given height, by default "2q".
        dewpoint_temperature_at_height_level : str, optional
            Name of the variable representing dewpoint temperature at the given height, by default "2d".
        temperature_at_height_level : str, optional
            Name of the variable for temperature at the given height, by default "2t".
        surface_pressure : str, optional
            Name of the variable for surface pressure, by default "sp".
        specific_humidity_at_model_levels : str, optional
            Name of the variable for specific humidity at model levels, by default "q".
        temperature_at_model_levels : str, optional
            Name of the variable for temperature at model levels, by default "t".
        AB : str | dict[str, NDArray]
            A string key for predefined A and B coefficients or a dictionary with "A" and "B" arrays for vertical interpolation.
            Possible predefined keys are: "IFS_137".
        return_inputs : Literal["all", "none"] | list[str], optional
            List of which filter inputs should be returned, by default ["specific_humidity_at_height_level", "relative_humidity_at_height_level", "temperature_at_height_level", "surface_pressure"]
        """

        self.return_inputs = return_inputs
        self.height = float(height)
        self.specific_humidity_at_height_level = specific_humidity_at_height_level
        self.dewpoint_temperature_at_height_level = dewpoint_temperature_at_height_level
        self.surface_pressure = surface_pressure
        self.specific_humidity_at_model_levels = specific_humidity_at_model_levels
        self.temperature_at_model_levels = temperature_at_model_levels

        self.A, self.B = _set_AB(model_level_AB)

    def _get_pressure_at_height_level(
        self,
        temperature_at_model_levels: NDArray,
        specific_humidity_at_model_levels: NDArray,
        surface_pressure: NDArray,
    ) -> NDArray:

        return vertical.pressure_at_height_levels(
            height=self.height,
            t=temperature_at_model_levels,
            q=specific_humidity_at_model_levels,
            sp=surface_pressure,
            A=self.A,
            B=self.B,
        )

    def forward_transform(
        self,
        specific_humidity_at_height_level: ekd.Field,
        surface_pressure: ekd.Field,
        specific_humidity_at_model_levels: ekd.FieldList,
        temperature_at_model_levels: ekd.FieldList,
    ) -> Iterator[ekd.Field]:
        """This will return the relative humidity along with temperature from specific humidity and temperature"""
        # Check vertical consistency

        _check_consistency(
            self.A,
            self.B,
            {
                self.specific_humidity_at_model_levels: specific_humidity_at_model_levels,
                self.temperature_at_model_levels: temperature_at_model_levels,
            },
        )

        # Make sure model levels are ordered ascending (highest level first):
        specific_humidity_at_model_levels = specific_humidity_at_model_levels.order_by(level="ascending")
        temperature_at_model_levels = temperature_at_model_levels.order_by(level="ascending")

        pressure_at_height_level = self._get_pressure_at_height_level(
            temperature_at_model_levels.to_numpy(),
            specific_humidity_at_model_levels.to_numpy(),
            surface_pressure.to_numpy(),
        )

        specific_humidity_at_height_level_values = specific_humidity_at_height_level.to_numpy()
        specific_humidity_at_height_level_values[specific_humidity_at_height_level_values == 0] = EPS_SPECIFIC

        dewpoint_temperature_at_height_level = thermo.dewpoint_from_specific_humidity(
            q=specific_humidity_at_height_level_values, p=pressure_at_height_level
        )

        # Return the fields
        yield self.new_field_from_numpy(
            dewpoint_temperature_at_height_level,
            template=specific_humidity_at_height_level,
            param=self.dewpoint_temperature_at_height_level,
        )

    def backward_transform(
        self,
        dewpoint_temperature_at_height_level: ekd.Field,
        surface_pressure: ekd.Field,
        specific_humidity_at_model_levels: ekd.FieldList,
        temperature_at_model_levels: ekd.FieldList,
    ) -> Iterator[ekd.Field]:
        """This will return the specific humidity along with temperature from relative humidity and temperature"""

        # Check vertical consistency
        _check_consistency(
            self.A,
            self.B,
            {
                self.specific_humidity_at_model_levels: specific_humidity_at_model_levels,
                self.temperature_at_model_levels: temperature_at_model_levels,
            },
        )

        # Make sure model levels are ordered ascending (highest level first):
        specific_humidity_at_model_levels = specific_humidity_at_model_levels.order_by(level="ascending")
        temperature_at_model_levels = temperature_at_model_levels.order_by(level="ascending")

        pressure_at_height_level = self._get_pressure_at_height_level(
            temperature_at_model_levels.to_numpy(),
            specific_humidity_at_model_levels.to_numpy(),
            surface_pressure.to_numpy(),
        )

        specific_humidity_at_height_level = thermo.specific_humidity_from_dewpoint(
            td=dewpoint_temperature_at_height_level.to_numpy(), p=pressure_at_height_level
        )

        yield self.new_field_from_numpy(
            specific_humidity_at_height_level,
            template=dewpoint_temperature_at_height_level,
            param=self.specific_humidity_at_height_level,
        )


filter_registry.register("q_to_d_height", SpecificToDewpointAtHeightLevel)
filter_registry.register("d_to_q_height", SpecificToDewpointAtHeightLevel.reversed)
