# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import pandas as pd

from anemoi.transform.filter import DispatchingFilter
from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.fields.orog_to_z import Orography as OrographyFields
from anemoi.transform.filters.tabular.geopotential_to_height import GeopotentialToHeight as GeopotentialToHeightTabular


class GeopotentialToHeight(DispatchingFilter):
    """Convert from geopotential to height for field or tabular datasets."""

    def __init__(self, **config):
        config["geopotential"] = config.get("geopotential", "z")
        if ("height" in config) and ("orography" in config):
            raise ValueError("Must specify either 'height' or 'orography' parameter, but not both.")

        # use "height" as the canonical key - override with "orography" if "height" not present
        if "height" not in config:
            config["height"] = config.pop("orography", "orog")

        # field orography filter uses geopotential and orography keys
        self.field_filter = OrographyFields(geopotential=config["geopotential"], orography=config["height"])
        # tabular filter uses geopotential and height keys
        self.tabular_filter = GeopotentialToHeightTabular(geopotential=config["geopotential"], height=config["height"])

    def forward_fields(self, data: ekd.FieldList) -> ekd.FieldList:
        return self.field_filter.forward(data)

    def backward_fields(self, data: ekd.FieldList) -> ekd.FieldList:
        return self.field_filter.backward(data)

    def forward_tabular(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.tabular_filter.forward(data)


filter_registry.register("geopotential_to_height", GeopotentialToHeight, aliases=["orog_to_z"])
filter_registry.register("height_to_geopotential", GeopotentialToHeight.reversed, aliases=["z_to_orog"])
