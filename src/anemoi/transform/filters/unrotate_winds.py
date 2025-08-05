# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict

import earthkit.data as ekd
from earthkit.geo.rotate import unrotate_vector

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry


@filter_registry.register("unrotate_winds")
class UnrotateWinds(Filter):
    """Unrotate the wind components of a GRIB file."""

    def __init__(self, *, u: str, v: str):
        """Initialize the UnrotateWinds filter.

        Parameters
        ----------
        u : str
            The parameter name for the u-component of the wind.
        v : str
            The parameter name for the v-component of the wind.
        """
        self.u = u
        self.v = v

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        """Unrotate the wind components of a GRIB file.

        Parameters
        ----------
        fields : ekd.FieldList
            The list of input fields.

        Returns
        -------
        ekd.FieldList
            The resulting field array with unrotated wind components.
        """
        result = []

        wind_params = (self.u, self.v)
        wind_pairs = defaultdict(dict)

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

            x = pairs[self.u]
            y = pairs[self.v]

            lats, lons = x.grid_points()
            raw_lats, raw_longs = x.grid_points_unrotated()

            assert x.rotation == y.rotation

            u_new, v_new = unrotate_vector(
                lats,
                lons,
                x.to_numpy(flatten=True),
                y.to_numpy(flatten=True),
                *x.rotation[:2],
                south_pole_rotation_angle=x.rotation[2],
                lat_unrotated=raw_lats,
                lon_unrotated=raw_longs,
            )

            result.append(new_field_from_numpy(x, u_new))
            result.append(new_field_from_numpy(y, v_new))

        return new_fieldlist_from_list(result)
