# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd

from anemoi.transform.constants import g_gravitational_acceleration
from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters import filter_registry


class Orography(SingleFieldFilter):
    """A filter to convert orography in m to surface geopotential in m²/s², and back."""

    optional_inputs = {"orog": "orog", "z": "z"}

    def forward_select(self):
        # select only fields where the param is self.orog
        return {"param": self.orog}

    def backward_select(self):
        # select only fields where the param is self.z
        return {"param": self.z}

    def forward_transform(self, orog: ekd.Field) -> ekd.Field:
        """Convert orography in m to surface geopotential in m²/s².

        Parameters
        ----------
        orog : ekd.Field
            The orography field in m.

        Returns
        -------
        ekd.Field
            The surface geopotential in m²/s².
        """
        new_metadata = {"param": self.z}
        return self.new_field_from_numpy(orog.to_numpy() * g_gravitational_acceleration, template=orog, **new_metadata)

    def backward_transform(self, z: ekd.Field) -> ekd.Field:
        """Convert surface geopotential in m²/s² to orography in m.

        Parameters
        ----------
        z : ekd.Field
            The surface geopotential in m²/s².

        Returns
        -------
        ekd.Field
            The orography in m.
        """
        orig_metadata = {"param": self.orog}
        return self.new_field_from_numpy(z.to_numpy() / g_gravitational_acceleration, template=z, **orig_metadata)


filter_registry.register("orog_to_z", Orography)
filter_registry.register("z_to_orog", Orography.reversed)
