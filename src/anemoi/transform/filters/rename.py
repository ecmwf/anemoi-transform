# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re

import earthkit.data as ekd

from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry


class FormatRename:
    def __init__(self, what, format):
        self.what = what
        self.format = format
        self.bits = re.findall(r"{(\w+)}", format)

    def rename(self, field):
        md = field.metadata(self.what, default=None)
        if md is None:
            return field

        values = field.metadata(*self.bits)
        kwargs = {k: v for k, v in zip(self.bits, values)}
        kwargs = {self.what: self.format.format(**kwargs)}
        return new_field_with_metadata(template=field, **kwargs)


class DictRename:
    def __init__(self, what, renaming):
        self.what = what
        self.renaming = renaming

    def rename(self, field):
        md = field.metadata(self.what, default=None)
        if md is None:
            return field

        if md not in self.renaming:
            return field

        kwargs = {self.what: self.renaming[md]}

        return new_field_with_metadata(template=field, **kwargs)


@filter_registry.register("rename")
class Rename(Filter):

    def __init__(self, *, rename: dict):

        self._rename = {}
        for key, value in rename.items():
            if isinstance(value, str):
                self._rename[key] = FormatRename(key, value)
            elif isinstance(value, dict):
                self._rename[key] = DictRename(key, value)
            else:
                raise ValueError(f"Invalid value for rename: {value}")

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:

        result = []
        for field in data:
            for _, renamer in self._rename.items():
                field = renamer.rename(field)
            result.append(field)

        return new_fieldlist_from_list(result)
