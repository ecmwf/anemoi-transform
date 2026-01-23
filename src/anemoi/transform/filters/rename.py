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
        self.bits = re.findall(r"{([\w:]+)}", format)

        # Escape ":" type delimiter used by eccodes as ":" is a reserved symbol in str.format.
        self._delimiter = "|"
        self.format = re.sub(r"{([^}]+)}", lambda m: "{" + m.group(1).replace(":", self._delimiter) + "}", self.format)
        self.format_keys = [b.replace(":", self._delimiter) for b in self.bits]

    def rename(self, field):
        md = field.metadata(self.what, default=None)
        if md is None:
            return field

        values = field.metadata(*self.bits)
        values = (
            [
                values,
            ]
            if isinstance(values, str)
            else values
        )

        kwargs = dict(zip(self.format_keys, values))
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
    """A filter to rename fields based on their metadata.

    When combining several sources, it is common to have different values
    for a given attribute to represent the same concept. For example,
    ``temperature_850hPa`` and ``t_850`` are two different ways to represent
    the temperature at 850 hPa. The ``rename`` filter allows renaming a key
    to another key.

    Notes
    -----

    The ``rename`` filter was primarily designed to rename the ``param``
    attribute, but any key can be renamed. The ``rename`` filter can take
    several renaming keys.

    Examples
    --------

    You can rename using a dictionary:

    .. code-block:: yaml

        input:
          pipe:
            - source:
                ...
            - rename:
                param:
                    z: geopotential
                    t: temperature
                levelist:
                    1000: 1000hPa
                    850: 850hPa

    or using a format string:

    .. code-block:: yaml

        input:
            pipe:
                - source:
                    ...
                - rename:
                    param: "{param}_{levelist}_{levtype}_{level:d}"

    In the latter case, the keys between curly braces are replaced by their
    corresponding metadata values in the field. The type of metadata values
    requested via eccodes can be chosen by appending ":i", ":d", ":s" for
    int, double and str, respectively. (See https://confluence.ecmwf.int/display/ECC/grib_get)

    """

    def __init__(self, **kwargs):

        self._rename = {}
        for key, value in kwargs.items():
            if isinstance(value, str):
                self._rename[key] = FormatRename(key, value)
            elif isinstance(value, dict):
                self._rename[key] = DictRename(key, value)
            else:
                raise ValueError(f"Invalid value for rename: {key}: {value}")

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:

        result = []
        for field in data:
            for _, renamer in self._rename.items():
                field = renamer.rename(field)
            result.append(field)

        return new_fieldlist_from_list(result)
