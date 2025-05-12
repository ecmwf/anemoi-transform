# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd

from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry


@filter_registry.register("rename")
class Rename(Filter):

    def __init__(
        self,
        *,
        rename: dict = None,
    ):

        self._rename = rename

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:

        result = []

        for field in data:

            param = field.metadata("param")
            if param in self._rename:
                result.append(new_field_with_metadata(template=field, param=self._rename[param]))
            else:
                result.append(field)

        return new_fieldlist_from_list(result)
