# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections import defaultdict
from typing import Any

import earthkit.data as ekd
from earthkit.geo.rotate import rotate_vector

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry


@filter_registry.register("rotate_winds")
class RotateWinds(Filter):
    """Rotate wind components from one projection to another."""

    def __init__(
        self,
        *,
        x_wind: str,
        y_wind: str,
        source_projection: str | None = None,
        target_projection: str = "+proj=longlat",
    ):
        """Initialize the RotateWinds filter.

        Parameters
        ----------
        x_wind : str
            X wind component parameter.
        y_wind : str
            Y wind component parameter.
        source_projection : str | None, optional
            Source projection, by default None.
        target_projection : str, optional
            Target projection, by default "+proj=longlat".
        """
        self.x_wind = x_wind
        self.y_wind = y_wind
        self.source_projection = source_projection
        self.target_projection = target_projection

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Rotate wind components from one projection to another.

        Parameters
        ----------
        fields : ekd.FieldList
            List of input fields.

        Returns
        -------
        ekd.FieldList
            Array of fields with rotated wind components.
        """
        from pyproj import CRS

        result = []

        wind_params: tuple[str, str] = (self.x_wind, self.y_wind)
        wind_pairs: dict[tuple, dict[str, Any]] = defaultdict(dict)

        for f in fields:
            key = f.metadata(namespace="mars")
            param = key.pop("param")

            if param not in wind_params:
                result.append(f)
                continue

            key = tuple(key.items())

            if param in wind_pairs[key]:
                raise ValueError(f"Duplicate wind component {param} for {key}")

            wind_pairs[key][param] = f

        for pairs in wind_pairs.values():
            if len(pairs) != 2:
                raise ValueError("Missing wind component")

            x = pairs[self.x_wind]
            y = pairs[self.y_wind]

            assert x.grid_mapping == y.grid_mapping

            lats, lons = x.grid_points()
            x_new, y_new = rotate_vector(
                lats,
                lons,
                x.to_numpy(flatten=True),
                y.to_numpy(flatten=True),
                (self.source_projection if self.source_projection is not None else CRS.from_cf(x.grid_mapping)),
                self.target_projection,
            )

            result.append(new_field_from_numpy(x, x_new))
            result.append(new_field_from_numpy(y, y_new))

        return new_fieldlist_from_list(result)
