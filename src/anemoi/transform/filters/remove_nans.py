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
from anemoi.transform.filters.fields.remove_nans import RemoveNaNs as RemoveNaNsFields
from anemoi.transform.filters.tabular.drop_nans import DropNaNs as DropNaNsTabular


class RemoveNaNs(DispatchingFilter):
    """Remove NaNs in field or tabular datasets."""

    def __init__(self, **config):
        if len(config) == 0:
            # empty config is valid for both types of filter
            self.tabular_filter = DropNaNsTabular()
            self.field_filter = RemoveNaNsFields()
        elif ("columns" in config) or ("column_prefix" in config) or ("how" in config):
            # only these columns supported in tabular filter
            self.tabular_filter = DropNaNsTabular(**config)
            self.field_filter = None
        else:
            # assume field filter
            self.tabular_filter = None
            self.field_filter = RemoveNaNsFields(**config)

    def forward_fields(self, data: ekd.FieldList) -> ekd.FieldList:
        if self.field_filter is None:
            raise ValueError("Ambigious config for RemoveNaNs filter.")
        return self.field_filter.forward(data)

    def forward_tabular(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.tabular_filter is None:
            raise ValueError("Ambigious config for RemoveNaNs filter.")
        return self.tabular_filter.forward(data)


filter_registry.register("remove_nans", RemoveNaNs, aliases=["drop_nans"])
