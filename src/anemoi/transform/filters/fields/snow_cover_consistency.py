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

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.fields.matching import MatchingFieldsFilter
from anemoi.transform.filters.fields.matching import matching


@filter_registry.register("snow_cover_consistency")
class SnowCoverConsistency(MatchingFieldsFilter):
    """A filter to ensure snow cover consistency from snow depth.

    Notes
    -----
    The `snow cover` (``snowc``) is kept consistent with `snow depth` (``sd``) using the following rules:

    .. math::

        \\text{If } sd \\leq \\frac{1}{1000}, \\quad sc = 0

        sc = \\begin{cases} \\dfrac{sd}{15000} & \\text{if } sc = 0 \\\\ sc & \\text{otherwise} \\end{cases}

        sc = \\text{clip}(sc, 0, 1)

    Where:
        - ``sc`` is the snow cover value (fraction, [0, 1])
        - ``sd`` is the snow depth in metres of water equivalent

    The first rule zeros snow cover when snow depth is negligible (less than 1 mm).
    The second rule assigns a minimum snow cover derived from the ERA-Interim relationship
    (snow depth in kg m⁻² divided by 15, where 1/1000 converts metres to kg m⁻²), but
    only when snow cover is exactly zero — existing non-zero values are left unchanged.
    The final step clips snow cover to the valid range [0, 1].
    """

    @matching(
        select="param",
        forward=("snow_depth", "snow_cover"),
    )
    def __init__(
        self,
        *,
        snow_depth: str = "sd",
        snow_cover: str = "snowc",
    ) -> None:
        """Initialize the SnowCover filter.

        Parameters
        ----------
        snow_depth : str, optional
            The parameter name for snow depth, by default "sd".
        snow_cover : str, optional
            The parameter name for snow cover, by default "snowc".
        """

        self.snow_depth = snow_depth
        self.snow_cover = snow_cover

    def forward_transform(self, snow_depth: ekd.Field, snow_cover: ekd.Field) -> Iterator[ekd.Field]:
        """Ensure snow cover consistency from snow depth.

        Parameters
        ----------
        snow_depth : ekd.Field
            The snow depth data.
        snow_cover : ekd.Field
            The snow cover data.

        Returns
        -------
        Iterator[ekd.Field]
            Transformed fields.
        """

        snow_cover_np = snow_cover.to_numpy()
        snow_depth_np = snow_depth.to_numpy()

        # If sd is less than 1mm, set snow cover to 0
        snow_cover_where_sd_zero = np.where(snow_depth_np <= 1 / 1000, 0, snow_cover_np)

        # If snowc is equal to 0, then set to snow depth divided by 15 (converted to kg m⁻²)
        # Otherwise, keep the original snow cover value
        snow_cover_consistent = np.where(
            snow_cover_where_sd_zero == 0, (snow_depth_np / 1000) / 15, snow_cover_where_sd_zero
        )

        # Clip snow cover to the range [0, 1]
        snow_cover_consistent = np.clip(snow_cover_consistent, 0, 1)

        yield self.new_field_from_numpy(snow_cover_consistent, template=snow_depth, param=self.snow_cover)
        yield snow_depth
