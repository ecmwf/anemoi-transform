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

    optional_inputs = {"lnsp": "lnsp", "sp": "sp"}

    def forward_select(self):
        # select only fields where the param is self.lnsp
        return {"param": self.lnsp}

    def backward_select(self):
        # select only fields where the param is self.sp
        return {"param": self.sp}

    def forward_transform(self, lnsp: ekd.Field) -> ekd.Field:
        """Convert ln(sp) to sp.

        Parameters
        ----------
        lnsp : ekd.Field
            The natural log of surface pressure

        Returns
        -------
        ekd.Field
            The surface pressure
        """
        new_metadata = {"param": self.sp, "levelist": None, "level": None}
        return self.new_field_from_numpy(np.exp(lnsp.to_numpy()), template=lnsp, **new_metadata)

    def backward_transform(self, sp: ekd.Field) -> ekd.Field:
        """Convert surface surface pressure to ln(surface pressure)

        Parameters
        ----------
        sp : ekd.Field
            The surface pressure.

        Returns
        -------
        ekd.Field
            The natural log of surface pressure.
        """
        orig_metadata = {"param": self.lnsp}
        return self.new_field_from_numpy(np.log(sp.to_numpy()), template=sp, **orig_metadata)


filter_registry.register("lnsp_to_sp", LnspToSp)
filter_registry.register("sp_to_lnsp", LnspToSp.reversed)
