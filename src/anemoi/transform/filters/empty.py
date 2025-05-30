# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd

from anemoi.transform.fields import new_empty_fieldlist
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry


@filter_registry.register("empty")
class Empty(Filter):

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        return new_empty_fieldlist()
