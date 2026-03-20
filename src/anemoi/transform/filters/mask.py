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
from anemoi.transform.filters.fields.apply_mask import MaskVariable as MaskVariableFields
from anemoi.transform.filters.tabular.mask import MaskValues as MaskValuesTabular


class Mask(DispatchingFilter):
    """Mask field or tabular datasets."""

    def __init__(self, **config):
        if "path" in config:
            self.filter = MaskVariableFields(**config)
        else:
            self.filter = MaskValuesTabular(**config)

    def forward_fields(self, data: ekd.FieldList) -> ekd.FieldList:
        return self.filter.forward(data)

    def forward_tabular(self, data: pd.DataFrame) -> pd.DataFrame:
        return self.filter.forward(data)


filter_registry.register("mask", Mask, aliases=["apply_mask"])
