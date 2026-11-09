# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Iterator

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters.fields import filter_registry
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter
from anemoi.transform.filters.fields.matching import MatchingSpec


def compute_snow_depth_m(snow_depth: np.ndarray, snow_density: np.ndarray) -> np.ndarray:
    """Convert snow depth (water equivalent) to snow depth in metres.

    The snow depth in metres (``sde``) is computed from ``snow depth`` (``sd``,
    in metres of water equivalent) and ``snow density`` (``rsn``, in kg/m³) as:

    .. math::

        sde = 1000 \\cdot \\frac{sd}{rsn}

    Parameters
    ----------
    snow_depth : np.ndarray
        Snow depth in metres of water equivalent (sd).
    snow_density : np.ndarray
        Snow density in kg/m³ (rsn).

    Returns
    -------
    np.ndarray
        Snow depth in metres (sde).
    """
    return 1000.0 * snow_depth / snow_density


@filter_registry.register("snow_depth_m")
class SnowDepthM(MatchingFieldsFilter):
    """A filter to compute snow depth in metres from snow depth (water equivalent) and snow density.

    Notes
    -----
    The ``snow depth in metres`` (``sde``) is computed from ``snow depth`` (``sd``,
    in metres of water equivalent) and ``snow density`` (``rsn``, in kg/m³) as:

    .. math::

        sde = 1000 \\cdot \\frac{sd}{rsn}

    This conversion is based on the relationship between snow water equivalent,
    snow density, and actual snow depth:

    .. math::

        sd = sde \\cdot \\frac{rsn}{1000}

    where 1000 kg/m³ is the density of water.

    """

    MATCHING = MatchingSpec(
        select="param",
        forward=("snow_depth", "snow_density"),
    )

    def __init__(
        self,
        *,
        snow_depth: str = "sd",
        snow_density: str = "rsn",
        snow_depth_m: str = "sde",
    ) -> None:
        """Initialize the SnowDepthM filter.

        Parameters
        ----------
        snow_depth : str, optional
            The parameter name for snow depth (water equivalent), by default "sd".
        snow_density : str, optional
            The parameter name for snow density, by default "rsn".
        snow_depth_m : str, optional
            The parameter name for snow depth in metres, by default "sde".
        """
        self.snow_depth = snow_depth
        self.snow_density = snow_density
        self.snow_depth_m = snow_depth_m
        super().__init__()

    def forward_transform(self, snow_depth: ekd.Field, snow_density: ekd.Field) -> Iterator[ekd.Field]:
        """Convert snow depth (water equivalent) and snow density to snow depth in metres.

        Parameters
        ----------
        snow_depth : ekd.Field
            The snow depth data (water equivalent).
        snow_density : ekd.Field
            The snow density data.

        Returns
        -------
        Iterator[ekd.Field]
            Transformed fields containing snow depth in metres.
        """
        sde = compute_snow_depth_m(snow_depth.to_numpy(), snow_density.to_numpy())

        yield self.new_field_from_numpy(sde, template=snow_depth, param=self.snow_depth_m, units="m")
