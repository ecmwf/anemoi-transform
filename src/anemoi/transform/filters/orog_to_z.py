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
    r"""A filter to convert orography in m to surface geopotential in m²/s², and back.

    This filter converts orography (in metres) to surface
    geopotential height (m\ :sup:`2`/s\ :sup:`2`) using the equation

    .. math:: geopotential = g \cdot orography

    where `g` refers to the :data:`gravitational acceleration constant <earthkit.meteo.constants.g>`.

    This filter must follow a source that provides orography, which is
    replaced by surface geopotential height.

    """

    optional_inputs = {"orography": "orog", "geopotential": "z"}

    def forward_select(self):
        # select only fields where the param is self.orography
        return {"param": self.orography}

    def backward_select(self):
        # select only fields where the param is self.geopotential
        return {"param": self.geopotential}

    def forward_transform(self, orography: ekd.Field) -> ekd.Field:
        """Convert orography in m to surface geopotential in m²/s².

        Parameters
        ----------
        orography : ekd.Field
            The orography field in m.

        Returns
        -------
        ekd.Field
            The surface geopotential in m²/s².
        """
        new_metadata = {"param": self.geopotential}
        return self.new_field_from_numpy(
            orography.to_numpy() * g_gravitational_acceleration, template=orography, **new_metadata
        )

    def backward_transform(self, geopotential: ekd.Field) -> ekd.Field:
        """Convert surface geopotential in m²/s² to orography in m.

        Parameters
        ----------
        geopotential : ekd.Field
            The surface geopotential in m²/s².

        Returns
        -------
        ekd.Field
            The orography in m.
        """
        orig_metadata = {"param": self.orography}
        return self.new_field_from_numpy(
            geopotential.to_numpy() / g_gravitational_acceleration, template=geopotential, **orig_metadata
        )


filter_registry.register("orog_to_z", Orography)
filter_registry.register("z_to_orog", Orography.reversed)
