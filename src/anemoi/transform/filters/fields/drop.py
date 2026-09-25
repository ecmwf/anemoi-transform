# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd

from anemoi.transform.filter import SingleFieldFilter


class Drop(SingleFieldFilter):
    """A filter to drop based on metadata selection"""

    def __init__(self, param: str | list[str] | None = None, levelist: int | list[int] | None = None):
        """Initialise the Drop filter.

        Parameters
        ----------
        **kwargs : dict
            Metadata selection criteria for dropping fields.
        """
        self.selection_criteria = {}
        if param is not None:
            self.selection_criteria["param"] = param
        if levelist is not None:
            self.selection_criteria["levelist"] = levelist
        super().__init__()

    def forward_select(self):
        return self.selection_criteria

    def backward_select(self):
        return self.selection_criteria

    def forward_transform(self, drop_field: ekd.Field) -> None:
        """Drop fields based on metadata selection."""
        return

    def backward_transform(self, drop_field: ekd.Field) -> None:
        """Drop fields based on metadata selection."""
        return
