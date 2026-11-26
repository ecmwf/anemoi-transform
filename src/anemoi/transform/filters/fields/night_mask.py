# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime

import earthkit.data as ekd
import numpy as np
from anemoi.utils.dates import frequency_to_timedelta
from earthkit.data.utils.dates import to_datetime

from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters.fields import filter_registry

_CACHE_SIZE = 8


@filter_registry.register("night_mask")
class NightMask(SingleFieldFilter):
    """Set accumulated fields to zero where the sun was down for the whole accumulation window.

    A field valued at time ``t`` that is accumulated over ``(t - accumulation_period, t]`` (for example
    ``ssr``, ``ssrd`` or ``fdir`` accumulated over one hour) can only be non-zero if the sun was up at some point
    in that window. The filter therefore sets the field to zero where the insolation (the cosine of the solar
    zenith angle, zero at night) is ``<= 0`` both at ``t`` and at ``t - accumulation_period``. Testing ``t`` alone
    would wrongly zero the hour that follows sunset.

    The insolation is computed from the field's grid and valid date, with the same code that provides the
    ``insolation`` forcing of anemoi datasets and models.

    The window is only tested at its two ends, so a sun above the horizon only in the middle of the window is
    missed. This can only happen at the edge of polar day and night.

    This filter is not invertible: ``backward`` returns the fields unchanged.

    Parameters
    ----------
    variables : str or list of str
        ``param`` of the accumulated fields to mask. Other fields are passed through unchanged.
    accumulation_period : str or int or datetime.timedelta
        Period over which the fields are accumulated, e.g. ``1h``. A number is read as hours.

    Examples
    --------
    Zero the one-hour accumulated surface radiation at night:

    .. code-block:: yaml

        night_mask:
            variables: [ssr, ssrd, fdir]
            accumulation_period: 1h

    """

    required_inputs = ("variables", "accumulation_period")

    def prepare_filter(self) -> None:
        variables = self.variables
        self._variables = [variables] if isinstance(variables, str) else list(variables)
        if not self._variables:
            raise ValueError("night_mask: `variables` must list at least one param.")

        self._period = frequency_to_timedelta(self.accumulation_period)
        if self._period <= datetime.timedelta(0):
            raise ValueError(f"night_mask: accumulation_period must be positive, got {self._period}.")

        self._cache: dict[tuple, np.ndarray] = {}

    def forward_select(self) -> dict:
        return {"param": self._variables}

    def _is_dark(self, lat: np.ndarray, lon: np.ndarray, date: datetime.datetime) -> np.ndarray:
        """Return where the insolation is not positive at ``date``, caching the latest dates."""
        key = (date, lat.size, float(lat[0]), float(lon[0]), float(lat[-1]), float(lon[-1]))
        if key not in self._cache:
            if len(self._cache) >= _CACHE_SIZE:  # dates mostly advance in time: drop the oldest
                self._cache.pop(next(iter(self._cache)))
            request = dict(latitudes=lat, longitudes=lon, date=[date], param=["insolation"])
            insolation = ekd.from_source("forcings", **request)[0]
            self._cache[key] = insolation.to_numpy(dtype=np.float32, flatten=True) <= 0
        return self._cache[key]

    def forward_transform(self, field: ekd.Field) -> ekd.Field:
        values = field.to_numpy()
        date = to_datetime(field.metadata("valid_datetime"))
        latlon = field.to_latlon(flatten=True)
        lat, lon = np.asarray(latlon["lat"]), np.asarray(latlon["lon"])

        night = self._is_dark(lat, lon, date) & self._is_dark(lat, lon, date - self._period)
        masked = np.where(night.reshape(values.shape), np.zeros((), dtype=values.dtype), values)
        return self.new_field_from_numpy(masked, template=field)

    def backward_transform(self, field: ekd.Field) -> ekd.Field:
        return field

    def __repr__(self) -> str:
        return f"NightMask(variables={self._variables}, accumulation_period={self._period})"
