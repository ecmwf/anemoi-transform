# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np

from . import filter_registry
from .base import SimpleFilter

SOIL_TYPE_DIC = {
    0: {"theta_pwp": 0, "theta_cap": 0},
    1: {"theta_pwp": 0.059, "theta_cap": 0.244},
    2: {"theta_pwp": 0.151, "theta_cap": 0.347},
    3: {"theta_pwp": 0.133, "theta_cap": 0.383},
    4: {"theta_pwp": 0.279, "theta_cap": 0.448},
    5: {"theta_pwp": 0.335, "theta_cap": 0.541},
    6: {"theta_pwp": 0.267, "theta_cap": 0.663},
    7: {"theta_pwp": 0.151, "theta_cap": 0.347},
}

VEG_TYPE_DIC = {
    0: {"veg_rsmin": 250.0, "veg_cov": 0.0, "veg_z0m": 0.013},
    1: {"veg_rsmin": 125.0, "veg_cov": 0.9, "veg_z0m": 0.25},
    2: {"veg_rsmin": 80.0, "veg_cov": 0.85, "veg_z0m": 0.1},
    3: {"veg_rsmin": 395.0, "veg_cov": 0.9, "veg_z0m": 2.0},
    4: {"veg_rsmin": 320.0, "veg_cov": 0.9, "veg_z0m": 2.0},
    5: {"veg_rsmin": 215.0, "veg_cov": 0.9, "veg_z0m": 2.0},
    6: {"veg_rsmin": 320.0, "veg_cov": 0.99, "veg_z0m": 2.0},
    7: {"veg_rsmin": 100.0, "veg_cov": 0.7, "veg_z0m": 0.5},
    8: {"veg_rsmin": 250.0, "veg_cov": 0.0, "veg_z0m": 0.013},
    9: {"veg_rsmin": 45.0, "veg_cov": 0.5, "veg_z0m": 0.03},
    10: {"veg_rsmin": 110.0, "veg_cov": 0.9, "veg_z0m": 0.5},
    11: {"veg_rsmin": 45.0, "veg_cov": 0.1, "veg_z0m": 0.03},
    12: {"veg_rsmin": 0.0, "veg_cov": 0.0, "veg_z0m": 0.0013},
    13: {"veg_rsmin": 130.0, "veg_cov": 0.6, "veg_z0m": 0.25},
    14: {"veg_rsmin": 0.0, "veg_cov": 0.0, "veg_z0m": 0.0001},
    15: {"veg_rsmin": 0.0, "veg_cov": 0.0, "veg_z0m": 0.0001},
    16: {"veg_rsmin": 230.0, "veg_cov": 0.5, "veg_z0m": 0.5},
    17: {"veg_rsmin": 110.0, "veg_cov": 0.4, "veg_z0m": 0.1},
    18: {"veg_rsmin": 180.0, "veg_cov": 0.9, "veg_z0m": 1.50},
    19: {"veg_rsmin": 175.0, "veg_cov": 0.9, "veg_z0m": 1.1},
    20: {"veg_rsmin": 150.0, "veg_cov": 0.6, "veg_z0m": 0.02},
}


def read_crosswalking_table(param, param_dic):
    arrays = [np.array([param_dic[x][key] for x in param]) for key in param_dic[0].keys()]
    return arrays


@filter_registry.register("land_parameters")
class LandParameters(SimpleFilter):
    """A filter to add static parameters from table based on soil/vegetation type."""

    def __init__(
        self,
        *,
        # Input parameters
        high_veg_type="tvh",
        low_veg_type="tvl",
        soil_type="slt",
        # Output parameters
        hveg_rsmin="hveg_rsmin",
        hveg_cov="hveg_cov",
        hveg_z0m="hveg_z0m",
        lveg_rsmin="lveg_rsmin",
        lveg_cov="lveg_cov",
        lveg_z0m="lveg_z0m",
        theta_pwp="theta_pwp",
        theta_cap="theta_cap",
    ):
        self.high_veg_type = high_veg_type
        self.low_veg_type = low_veg_type
        self.soil_type = soil_type
        self.hveg_rsmin = hveg_rsmin
        self.hveg_cov = hveg_cov
        self.hveg_z0m = hveg_z0m
        self.lveg_rsmin = lveg_rsmin
        self.lveg_cov = lveg_cov
        self.lveg_z0m = lveg_z0m
        self.theta_pwp = theta_pwp
        self.theta_cap = theta_cap

    def forward(self, data):
        return self._transform(
            data,
            self.forward_transform,
            self.high_veg_type,
            self.low_veg_type,
            self.soil_type,
        )

    def backward(self, data):
        raise NotImplementedError("LandParameters is not reversible")

    def forward_transform(self, tvh, tvl, sotype):
        """Get static parameters from table based on soil/vegetation type"""

        hveg_rsmin, hveg_cov, hveg_z0m = read_crosswalking_table(tvh.to_numpy(), VEG_TYPE_DIC)
        lveg_rsmin, lveg_cov, lveg_z0m = read_crosswalking_table(tvl.to_numpy(), VEG_TYPE_DIC)
        theta_pwp, theta_cap = read_crosswalking_table(sotype.to_numpy(), SOIL_TYPE_DIC)

        yield self.new_field_from_numpy(hveg_rsmin, template=tvh, param=self.hveg_rsmin)
        yield self.new_field_from_numpy(hveg_cov, template=tvh, param=self.hveg_cov)
        yield self.new_field_from_numpy(hveg_z0m, template=tvh, param=self.hveg_z0m)
        yield self.new_field_from_numpy(lveg_rsmin, template=tvl, param=self.lveg_rsmin)
        yield self.new_field_from_numpy(lveg_cov, template=tvl, param=self.lveg_cov)
        yield self.new_field_from_numpy(lveg_z0m, template=tvl, param=self.lveg_z0m)
        yield self.new_field_from_numpy(theta_pwp, template=sotype, param=self.theta_pwp)
        yield self.new_field_from_numpy(theta_cap, template=sotype, param=self.theta_cap)

    def backward_transform(self, tvh, tvl, sotype):
        raise NotImplementedError("LandParameters is not reversible")
