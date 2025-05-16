# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import datetime
import logging

import earthkit.data as ekd
from earthkit.data.utils.dates import to_datetime

from anemoi.transform.fields import new_field_with_valid_datetime
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


@filter_registry.register("clear_step")
class ClearStepFilter(Filter):
    """Set the step of the field to 0."""

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Adjusts the valid_datetime of each field by subtracting the step in hours.

        Parameters
        ----------
        data : ekd.FieldList
            List of fields to be processed.

        Returns
        -------
        ekd.FieldList
            List of fields with updated valid_datetime.
        """
        result = []
        for field in data:
            valid_datetime = to_datetime(field.metadata("valid_datetime"))
            step = field.metadata("step")
            result.append(new_field_with_valid_datetime(field, valid_datetime - datetime.timedelta(hours=step)))

        return new_fieldlist_from_list(result)
