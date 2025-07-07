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

from . import filter_registry
from .matching import MatchingFieldsFilter
from .matching import matching


class Orography(MatchingFieldsFilter):
    """A filter to convert orography in m to surface geopotential in m²/s², and back."""

    @matching(
        select="param",
        forward=("orog"),
        backward=("z"),
    )
    def __init__(
        self,
        *,
        orog="orog",
        z="z",
        g=9.80665,
    ):

        self.orog = orog
        self.z = z
        self.g = g

    def forward_transform(
        self,
        orog: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Convert orography in m to surface geopotential in m²/s².
        Parameters
        ----------
        orog : ekd.Field
            The orography field in m.
        Returns
        -------
        Iterator[ekd.Field]
            The surface geopotential in m²/s².
        """

        z = orog.to_numpy() * self.g

        yield self.new_field_from_numpy(z, template=orog, param=self.z)
        yield orog

    def backward_transform(self, z: ekd.Field) -> Iterator[ekd.Field]:
        """Convert geopotential from m²/s² to orography in m.
        Parameters
        ----------
        z : ekd.Field
            The surface geopotential field in m²/s².
        Returns
        -------
        Iterator[ekd.Field]
            The orography in m.
        """

        yield self.new_field_from_numpy(z.to_numpy() / self.g, template=z, param=self.orog)
        yield z


filter_registry.register("orog_to_z", Orography)
filter_registry.register("z_to_orog", Orography.reversed)
