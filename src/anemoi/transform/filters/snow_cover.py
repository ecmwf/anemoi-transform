# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Iterator

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


def compute_snow_cover(snow_depth: np.ndarray, snow_density: np.ndarray) -> np.ndarray:
    """Convert snow depth to snow cover.

    Parameters
    ----------
    snow_depth : np.ndarray
        The depth of the snow.
    snow_density : np.ndarray
        The density of the snow.

    Returns
    -------
    np.ndarray
        The computed snow cover.
    """
    tmp1 = (1000 * snow_depth) / snow_density
    tmp2 = np.clip(snow_density, 100, 400)
    snow_cover = np.clip(np.tanh((4000 * tmp1) / tmp2), 0, 1)
    snow_cover[snow_cover > 0.99] = 1.0
    return snow_cover


@filter_registry.register("snow_cover")
class SnowCover(MatchingFieldsFilter):
    """A filter to compute snow cover from snow density and snow depth."""

    @matching(
        select="param",
        forward=("snow_depth", "snow_density"),
    )
    def __init__(
        self,
        *,
        snow_depth: str = "sd",
        snow_density: str = "rsn",
        snow_cover: str = "snowc",
    ) -> None:
        """Initialize the SnowCover filter.

        Parameters
        ----------
        snow_depth : str, optional
            The parameter name for snow depth, by default "sd".
        snow_density : str, optional
            The parameter name for snow density, by default "rsn".
        snow_cover : str, optional
            The parameter name for snow cover, by default "snowc".
        """

        self.snow_depth = snow_depth
        self.snow_density = snow_density
        self.snow_cover = snow_cover

    def forward_transform(self, snow_depth: ekd.Field, snow_density: ekd.Field) -> Iterator[ekd.Field]:
        """Convert snow depth and snow density to snow cover.

        Parameters
        ----------
        snow_depth : ekd.Field
            The snow depth data.
        snow_density : ekd.Field
            The snow density data.

        Returns
        -------
        Iterator[ekd.Field]
            Transformed fields.
        """
        snow_cover = compute_snow_cover(snow_depth.to_numpy(), snow_density.to_numpy())

        yield self.new_field_from_numpy(snow_cover, template=snow_depth, param=self.snow_cover)
