# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any
from typing import Dict
from typing import Iterator
from typing import List

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching

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


def read_crosswalking_table(param: Any, param_dic: Dict[int, Dict[str, float]]) -> List[np.ndarray]:
    """Read crosswalking table and return arrays for each key.

    Parameters
    ----------
    param : Any
        The parameter to read.
    param_dic : Dict[int, Dict[str, float]]
        The dictionary containing the crosswalking table.

    Returns
    -------
    List[np.ndarray]
        The arrays for each key in the crosswalking table.
    """
    arrays = [np.array([param_dic[x][key] for x in param]) for key in param_dic[0].keys()]
    return arrays


@filter_registry.register("land_parameters")
class LandParameters(MatchingFieldsFilter):
    """A filter to add static parameters from table based on soil/vegetation type."""

    @matching(
        select="param",
        forward=("high_veg_type", "low_veg_type", "soil_type"),
    )
    def __init__(
        self,
        *,
        high_veg_type: str = "tvh",
        low_veg_type: str = "tvl",
        soil_type: str = "slt",
        hveg_rsmin: str = "hveg_rsmin",
        hveg_cov: str = "hveg_cov",
        hveg_z0m: str = "hveg_z0m",
        lveg_rsmin: str = "lveg_rsmin",
        lveg_cov: str = "lveg_cov",
        lveg_z0m: str = "lveg_z0m",
        theta_pwp: str = "theta_pwp",
        theta_cap: str = "theta_cap",
    ) -> None:

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

    def forward_transform(
        self,
        high_veg_type: ekd.Field,
        low_veg_type: ekd.Field,
        soil_type: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Get static parameters from table based on soil/vegetation type.

        Parameters
        ----------
        high_veg_type : ekd.Field
            High vegetation type.
        low_veg_type : ekd.Field
            Low vegetation type.
        soil_type : ekd.Field
            Soil type.

        Returns
        -------
        Iterator[ekd.Field]
            An iterator over the new fields with static parameters.
        """
        hveg_rsmin, hveg_cov, hveg_z0m = read_crosswalking_table(high_veg_type.to_numpy(), VEG_TYPE_DIC)
        lveg_rsmin, lveg_cov, lveg_z0m = read_crosswalking_table(low_veg_type.to_numpy(), VEG_TYPE_DIC)
        theta_pwp, theta_cap = read_crosswalking_table(soil_type.to_numpy(), SOIL_TYPE_DIC)

        yield self.new_field_from_numpy(hveg_rsmin, template=high_veg_type, param=self.hveg_rsmin)
        yield self.new_field_from_numpy(hveg_cov, template=high_veg_type, param=self.hveg_cov)
        yield self.new_field_from_numpy(hveg_z0m, template=high_veg_type, param=self.hveg_z0m)
        yield self.new_field_from_numpy(lveg_rsmin, template=low_veg_type, param=self.lveg_rsmin)
        yield self.new_field_from_numpy(lveg_cov, template=low_veg_type, param=self.lveg_cov)
        yield self.new_field_from_numpy(lveg_z0m, template=low_veg_type, param=self.lveg_z0m)
        yield self.new_field_from_numpy(theta_pwp, template=soil_type, param=self.theta_pwp)
        yield self.new_field_from_numpy(theta_cap, template=soil_type, param=self.theta_cap)
