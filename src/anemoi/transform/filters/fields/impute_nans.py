# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters.fields import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("impute_nans_fields")
class ImputeNaNs(SingleFieldFilter):
    """A filter to impute NaN values in specified fields with a fixed value.

    Examples
    --------

    .. code-block:: yaml

        input:
          pipe:
            - source: # Can be `mars`, `netcdf`, etc.
                param: ...
            - impute_nans:
                param: [t, q]  # Parameters whose NaNs should be imputed
                value: 0.0     # Replacement value

    Notes
    -----

    Only the fields listed in ``param`` are modified. All other fields pass
    through unchanged.

    """

    required_inputs = ("param", "value")

    def forward_select(self):
        return {"param": self.param}

    def forward_transform(self, field: ekd.Field) -> ekd.Field:
        values = field.to_numpy(flatten=True).copy()
        values[np.isnan(values)] = self.value
        return self.new_field_from_numpy(values, template=field)
