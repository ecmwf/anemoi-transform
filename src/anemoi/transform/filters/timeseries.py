# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import numpy as np

from . import filter_registry
from .base import SimpleFilter

LOG = logging.getLogger(__name__)

# class MyFilter(SuperSimpleFilter):
#    def __init__(self, *, param):
#        self.param = param
#
#    def transform(date, tp, lsm):
#        new = tp + lsm + self.data(self.param, date)
#        return dict(q_500 = new)


@filter_registry.register("timeseries")
class Timeseries(SimpleFilter):
    """A source to add a timeseries depending on time but not on location"""

    def __init__(self, *, netcdf=None, template_param="2t"):
        if netcdf:
            import xarray as xr

            self.ds = xr.open_dataset(netcdf["path"])  # .to_dataframe()
        LOG.warning("Using the timeseries filter will be deprecated in the future. Please do not rely on it.")

        self.template_param = template_param

    def forward(self, data):
        return self._transform(
            data,
            self.forward_transform,
            self.template_param,
        )

    def forward_transform(self, template):
        """Convert snow depth and snow density to snow cover"""
        dt = template.metadata("valid_datetime")
        template_array = template.to_numpy()

        sel = self.ds.sel(time=dt)

        for name in self.ds.data_vars:
            value = sel[name].values
            data = np.full_like(template_array, value)
            yield self.new_field_from_numpy(data, template=template, param=name)

    def backward(self, data):
        raise NotImplementedError("SnowCover is not reversible")

    def backward_transform(self, sd, rsn):
        raise NotImplementedError("SnowCover is not reversible")
