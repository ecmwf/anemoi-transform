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

        \\text{If } sc > 0 \text{ and } sd = 0, \\quad sc = 0

        \\text{If } sd > 5 \\text{ and } sc = 0, \\quad sc = \\min(sd \\times 0.15,\\, 1)

        sc = \text{clip}(sc, 0, 1)

    Where:
        - ``sc`` is the snow cover value
        - ``sd`` is the snow depth value

    These rules ensure that snow cover is only present when there is snow depth, that a minimum snow cover is assigned when snow depth exceeds a threshold but cover is absent, and that snow cover is always bounded to the valid range ``[0, 1]``.
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

        snow_cover_where_sd_zero = np.where(snow_depth_np == 0, 0, snow_cover_np)
        snow_cover_consistent = np.where(
            (snow_depth_np > 5) & (snow_cover_where_sd_zero == 0), snow_depth_np * 0.15, snow_cover_where_sd_zero
        )
        snow_cover_consistent = np.clip(snow_cover_consistent, 0, 1)

        yield self.new_field_from_numpy(snow_cover_consistent, template=snow_depth, param=self.snow_cover)
        yield snow_depth
