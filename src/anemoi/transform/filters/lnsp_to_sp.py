# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
import numpy as np

from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters import filter_registry


class LnspToSp(SingleFieldFilter):
    """A filter to convert natural log of surface pressure (lnsp) to surface pressure (sp), and back."""

    optional_inputs = {"log_of_surface_pressure": "lnsp", "surface_pressure": "sp"}

    def forward_select(self):
        # select only fields where the param is self.log_of_surface_pressure
        return {"param": self.log_of_surface_pressure}

    def backward_select(self):
        # select only fields where the param is self.surface_pressure
        return {"param": self.surface_pressure}

    def forward_transform(self, log_of_surface_pressure: ekd.Field) -> ekd.Field:
        """Convert ln(sp) to sp.

        Parameters
        ----------
        log_of_surface_pressure : ekd.Field
            The natural log of surface pressure

        Returns
        -------
        ekd.Field
            The surface pressure
        """
        new_metadata = {"param": self.surface_pressure, "levelist": None, "level": None}
        return self.new_field_from_numpy(
            np.exp(log_of_surface_pressure.to_numpy()), template=log_of_surface_pressure, **new_metadata
        )

    def backward_transform(self, surface_pressure: ekd.Field) -> ekd.Field:
        """Convert surface surface pressure to ln(surface pressure)

        Parameters
        ----------
        surface_pressure : ekd.Field
            The surface pressure.

        Returns
        -------
        ekd.Field
            The natural log of surface pressure.
        """
        orig_metadata = {"param": self.log_of_surface_pressure}
        return self.new_field_from_numpy(
            np.log(surface_pressure.to_numpy()), template=surface_pressure, **orig_metadata
        )


filter_registry.register("lnsp_to_sp", LnspToSp)
filter_registry.register("sp_to_lnsp", LnspToSp.reversed)
