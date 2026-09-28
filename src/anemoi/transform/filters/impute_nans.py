# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pandas as pd

from anemoi.transform import FieldList
from anemoi.transform.filter import DispatchingFilter
from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.fields.impute_nans import ImputeNaNs as ImputeNaNsFields
from anemoi.transform.filters.tabular.impute_nans import ImputeNaNs as ImputeNaNsTabular


class ImputeNaNs(DispatchingFilter):
    """Impute NaN values in field or tabular datasets."""

    def __init__(self, **config):
        if len(config) == 0:
            # empty config is valid for both types of filter
            self.tabular_filter = ImputeNaNsTabular()
            self.field_filter = ImputeNaNsFields()
        elif ("columns" in config) or ("column_prefix" in config):

            # only these columns supported in tabular filter
            self.tabular_filter = ImputeNaNsTabular(**config)
            self.field_filter = None
        else:
            # assume field filter
            self.tabular_filter = None
            self.field_filter = ImputeNaNsFields(**config)

    def forward_fields(self, data: FieldList) -> FieldList:
        if self.field_filter is None:
            raise ValueError("Ambiguous config for ImputeNaNs field filter.")
        return self.field_filter.forward(data)

    def forward_tabular(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.tabular_filter is None:
            raise ValueError("Ambiguous config for ImputeNans tabular filter.")
        return self.tabular_filter.forward(data)


filter_registry.register("impute_nans", ImputeNaNs, aliases=["replace_nans"])
