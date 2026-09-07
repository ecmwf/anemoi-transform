# (C) Copyright 2024- Anemoi contributors.
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
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter, MatchingSpec


def compute_fraction_of_snow_cover(snow_depth: np.ndarray, snow_density: np.ndarray) -> np.ndarray:
    """Convert snow depth to fraction of snow cover.

    Parameters
    ----------
    snow_depth : np.ndarray
        The depth of the snow.
    snow_density : np.ndarray
        The density of the snow.

    Returns
    -------
    np.ndarray
        The computed fraction of snow cover.
    """
    tmp1 = (1000 * snow_depth) / snow_density
    tmp2 = np.clip(snow_density, 100, 400)
    fscov = np.clip(np.tanh((4000 * tmp1) / tmp2), 0, 1)
    fscov[fscov > 0.99] = 1.0
    return fscov


@filter_registry.register("fraction_of_snow_cover")
class FractionOfSnowCover(MatchingFieldsFilter):
    """A filter to compute fraction of snow cover from snow density and snow depth.

    Notes
    -----
    The ``fraction of snow cover`` (``fscov``) is computed from ``snow depth`` (``sd``) and ``snow density`` (``rsn``) as:

    .. math::

        \\text{fscov}(sd,rsn) =
        \\operatorname{clip}\\left(
        \\tanh\\left(
        \\frac{4000 \\cdot \\tfrac{1000 \\cdot sd}{rsn}}
        {\\operatorname{clip}(rsn,100,400)}
        \\right), 0, 1
        \\right)

    Post-processing rule:

    .. math::

        \\text{fscov} =
        \\begin{cases}
        1.0 & \\text{if } \\text{fscov} > 0.99 \\\\[0.8em]
        \\text{fscov} & \\text{otherwise}
        \\end{cases}

    with clipping defined as:

    .. math::

        \\operatorname{clip}(x,a,b) = \\min(\\max(x,a),b).

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
        fraction_of_snow_cover: str = "fscov",
    ) -> None:
        """Initialize the FractionOfSnowCover filter.

        Parameters
        ----------
        snow_depth : str, optional
            The parameter name for snow depth, by default "sd".
        snow_density : str, optional
            The parameter name for snow density, by default "rsn".
        fraction_of_snow_cover : str, optional
            The parameter name for fraction of snow cover, by default "fscov".
        """

        self.snow_depth = snow_depth
        self.snow_density = snow_density
        self.fraction_of_snow_cover = fraction_of_snow_cover
        super().__init__()

    def forward_transform(self, snow_depth: ekd.Field, snow_density: ekd.Field) -> Iterator[ekd.Field]:
        """Convert snow depth and snow density to fraction of snow cover.

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
        fscov = compute_fraction_of_snow_cover(snow_depth.to_numpy(), snow_density.to_numpy())

        yield self.new_field_from_numpy(fscov, template=snow_depth, param=self.fraction_of_snow_cover, units="Fraction")
