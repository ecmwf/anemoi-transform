# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any
from typing import Dict
from typing import Iterator

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


@filter_registry.register("cos_sin_mean_wave_direction")
class CosSinWaveDirection(MatchingFieldsFilter):
    """A filter to convert mean wave direction to cos() and sin() and back."""

    @matching(
        select="param",
        forward=("mean_wave_direction",),
        backward=("cos_mean_wave_direction", "sin_mean_wave_direction"),
    )
    def __init__(
        self,
        mean_wave_direction="mwd",
        cos_mean_wave_direction="cos_mwd",
        sin_mean_wave_direction="sin_mwd",
    ) -> None:
        """Initialize the CosSinWaveDirection filter."""

        self.mean_wave_direction = mean_wave_direction
        self.cos_mean_wave_direction = cos_mean_wave_direction
        self.sin_mean_wave_direction = sin_mean_wave_direction

    def forward_transform(
        self,
        mean_wave_direction: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Convert mean wave direction to its cosine and sine components.

        Parameters
        ----------
        mean_wave_direction : ekd.Field
            The mean wave direction field.

        Returns
        -------
        Iterator[ekd.Field]
            Fields of cosine and sine of the mean wave direction.
        """
        data = mean_wave_direction.to_numpy()
        data = np.deg2rad(data)

        yield self.new_field_from_numpy(np.cos(data), template=mean_wave_direction, param=self.cos_mean_wave_direction)
        yield self.new_field_from_numpy(np.sin(data), template=mean_wave_direction, param=self.sin_mean_wave_direction)

    def backward_transform(
        self,
        cos_mean_wave_direction: ekd.Field,
        sin_mean_wave_direction: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Convert cosine and sine components back to mean wave direction.

        Parameters
        ----------
        cos_mean_wave_direction : ekd.Field
            The cosine of the mean wave direction field.
        sin_mean_wave_direction : ekd.Field
            The sine of the mean wave direction field.

        Returns
        -------
        Iterator[ekd.Field]
            Field of the mean wave direction.
        """
        mwd = np.rad2deg(np.arctan2(sin_mean_wave_direction.to_numpy(), cos_mean_wave_direction.to_numpy()))
        mwd = np.where(mwd >= 360, mwd - 360, mwd)
        mwd = np.where(mwd < 0, mwd + 360, mwd)

        yield self.new_field_from_numpy(mwd, template=cos_mean_wave_direction, param=self.mean_wave_direction)

    def patch_data_request(self, data_request: Dict[str, Any]) -> Dict[str, Any]:
        """Modify the data request to include mean wave direction.

        Parameters
        ----------
        data_request : Dict[str, Any]
            The original data request.

        Returns
        -------
        Dict[str, Any]
            The modified data request.
        """
        """We have a chance to modify the data request here."""

        param = data_request.get("param")
        if param is None:
            return data_request

        if self.cos_mean_wave_direction in param or self.sin_mean_wave_direction in param:
            data_request["param"] = [
                p for p in param if p not in (self.cos_mean_wave_direction, self.sin_mean_wave_direction)
            ]
            data_request["param"].append(self.mean_wave_direction)

        return data_request
