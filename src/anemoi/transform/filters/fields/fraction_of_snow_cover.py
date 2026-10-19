# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Iterator
from typing import Any

import numpy as np

from anemoi.transform import Field
from anemoi.transform.filters.fields import filter_registry
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter
from anemoi.transform.filters.fields.matching import MatchingSpec


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

    def forward_transform(self, snow_depth: Field, snow_density: Field) -> Iterator[Field]:
        """Convert snow depth and snow density to fraction of snow cover.

        Parameters
        ----------
        snow_depth : Field
            The snow depth data.
        snow_density : Field
            The snow density data.

        Returns
        -------
        Iterator[Field]
            Transformed fields.
        """
        snow_depth.check_units("m of water equivalent")
        snow_density.check_units("kg m**-3")
        fscov = compute_fraction_of_snow_cover(snow_depth.to_numpy(), snow_density.to_numpy())

        yield Field.from_numpy(
            fscov,
            template=snow_depth,
            name=self.fraction_of_snow_cover,
            parameter={
                "variable": self.fraction_of_snow_cover,
                "units": "Fraction",
            },
        )

    def patch_data_request(self, data_request: dict[str, Any]) -> dict[str, Any]:
        """Make sure snow depth and snow density are both requested.

        Fraction of snow cover is not archived; it is computed by this filter. Because it
        is created from a snow depth template, its stored MARS metadata requests
        snow depth (``param: sd``). The request is therefore patched when it
        contains either fraction of snow cover or snow depth: fraction of snow cover is removed and snow
        depth and snow density are added if not already present.

        Parameters
        ----------
        data_request : dict[str, Any]
            The original data request.

        Returns
        -------
        dict[str, Any]
            The modified data request.
        """
        param = data_request.get("param")
        if param is None:
            return data_request

        params = [param] if isinstance(param, str) else list(param)
        if self.fraction_of_snow_cover not in params and self.snow_depth not in params:
            return data_request

        params = [p for p in params if p != self.fraction_of_snow_cover]
        for p in (self.snow_depth, self.snow_density):
            if p not in params:
                params.append(p)

        data_request["param"] = params
        return data_request
