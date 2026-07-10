# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Callable

import earthkit.data as ekd

from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import SingleFieldFilter


class Drop(SingleFieldFilter):
    """A filter to drop based on metadata selection"""

    def __init__(self, **kwargs):
        """Initialise the Drop filter.

        Parameters
        ----------
        **kwargs : dict
            Metadata selection criteria for dropping fields.
        """
        self.selection_criteria = kwargs
        super().__init__()

    def forward_select(self):
        return self.selection_criteria

    def backward_select(self):
        return self.selection_criteria

    @staticmethod
    def _map_transform(transform_function: Callable, fields: ekd.FieldList) -> ekd.FieldList:
        fieldlist = [transform_function(field) for field in fields]
        return new_fieldlist_from_list([f for f in fieldlist if f is not None])

    def forward_transform(self, drop_field: ekd.Field) -> None:
        """Drop fields based on metadata selection."""
        return None

    def backward_transform(self, drop_field: ekd.Field) -> None:
        """Drop fields based on metadata selection."""
        return None
