# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Iterator
import earthkit.data as ekd
from pyproj import CRS
from earthkit.geo.rotate import rotate_vector
from . import filter_registry
from .matching import MatchingFieldsFilter, matching

class RotateWinds(MatchingFieldsFilter):
    """A filter to rotate wind components from one projection to another."""

    @matching(
        select="param",
        forward=("x_wind", "y_wind")
        #,backward=("x_wind", "y_wind"),
    )
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

    def forward_transform(self, x_wind: ekd.Field, y_wind: ekd.Field) -> Iterator[ekd.Field]:
        """Rotate wind components from source projection to target projection.

        Parameters
        ----------
        x_wind : ekd.Field
            The x wind component field.
        y_wind : ekd.Field
            The y wind component field.

        Yields
        ------
        Iterator[ekd.Field]
            The rotated wind component fields.
        """
        lats, lons = x_wind.grid_points()
        proj_string = str(x_wind.projection())

        x_new, y_new = rotate_vector(
            lats,
            lons,
            x_wind.to_numpy(flatten=True),
            y_wind.to_numpy(flatten=True),
            (self.source_projection if self.source_projection is not None else CRS.from_string(proj_string)),
            self.target_projection,
        )

        yield self.new_field_from_numpy(x_new, template=x_wind, param=x_wind.metadata('param'))
        yield self.new_field_from_numpy(y_new, template=y_wind, param=y_wind.metadata('param'))

    # def backward_transform(self, x_wind: ekd.Field, y_wind: ekd.Field) -> Iterator[ekd.Field]:

filter_registry.register("rotate_winds", RotateWinds)

