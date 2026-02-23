# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
from earthkit.data.core.metadata import WrappedMetadata

from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters import filter_registry


@filter_registry.register("remove_metadata")
class RemoveMetadata(SingleFieldFilter):
    """A filter to remove metadata from fields."""

    required_inputs = ("keys",)
    optional_inputs = {"param": None}

    def prepare_filter(self):
        if isinstance(self.keys, str):
            self.keys = (self.keys,)
        elif not isinstance(self.keys, (list, tuple)):
            raise TypeError("Keys must be a string or list of strings.")

    def forward_select(self):
        if self.param is None:
            return {}
        return {"param": self.param}

    def forward_transform(self, field: ekd.Field) -> ekd.Field:
        """Create a new field with wrapped metadata which hides keys of the original field's metadata."""
        new_metadata = WrappedMetadata(field.metadata(), hidden=self.keys)
        return field.clone(metadata=new_metadata)
