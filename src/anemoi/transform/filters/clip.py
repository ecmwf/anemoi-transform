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
from anemoi.transform.filters import dispatching_filter_registry as filter_registry
from anemoi.transform.filters.fields.clipper import Clipper as ClipperFields
from anemoi.transform.filters.tabular.clip import Clip as ClipTabular


class Clip(DispatchingFilter):
    """Clip field or tabular datasets."""

    def __init__(self, **config):
        if "param" in config and isinstance(config["param"], str):
            self.filter = ClipperFields(**config)
        else:
            self.filter = ClipTabular(**config)

    def forward_fields(self, data: ekd.FieldList) -> ekd.FieldList:
        return self.filter.forward(data)

    def forward_tabular(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.filter.forward(data)


filter_registry.register("clip", Clip, aliases=["clipper"])
