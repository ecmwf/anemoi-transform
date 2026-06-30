# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import re

from anemoi.transform import Field
from anemoi.transform.fields import new_field_with_metadata
from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters.fields import filter_registry

# Mapping from old metadata keys to component-based accessor paths
_KEY_MAPPING = {
    "param": "parameter.variable",
    "levelist": "vertical.level",
    "levtype": "vertical.level_type",
    "step": "time.step",
    "valid_datetime": "time.valid_datetime",
    "number": "ensemble.member",
}


def _get_metadata(field, key):
    """Get metadata value by key, trying original metadata keys first, then component API."""
    try:
        return field.metadata(key)
    except (KeyError, TypeError):
        pass

    # Try the mapped component key
    mapped = _KEY_MAPPING.get(key)
    if mapped is not None:
        try:
            return field.get(mapped)
        except (KeyError, TypeError) as e:
            raise KeyError(f"Cannot get metadata for key '{key}'") from e


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
        try:
            md = _get_metadata(field, self.what)
        except KeyError:
            return field
        if md is None:
            return field

        values = [_get_metadata(field, b) for b in self.bits]

        kwargs = dict(zip(self.format_keys, values))
        kwargs = {self.what: self.format.format(**kwargs)}
        return new_field_with_metadata(template=field, **kwargs)


class DictRename:
    def __init__(self, what, renaming):
        self.what = what
        self.renaming = renaming

    def rename(self, field):
        try:
            md = _get_metadata(field, self.what)
        except KeyError:
            return field
        if md is None:
            return field

        if md not in self.renaming:
            return field

        kwargs = {self.what: self.renaming[md]}

        return new_field_with_metadata(template=field, **kwargs)


@filter_registry.register("rename_fields")
class Rename(SingleFieldFilter):
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

    def prepare_filter(self):
        renamers = {}
        for key, value in self.config.items():
            if isinstance(value, str):
                renamers[key] = FormatRename(key, value)
            elif isinstance(value, dict):
                renamers[key] = DictRename(key, value)
            else:
                raise ValueError(f"Invalid value for rename: {key}: {value}")
        self.renamers = tuple(renamers.values())

    def forward_transform(self, field: Field) -> Field:
        for renamer in self.renamers:
            field = renamer.rename(field)
        return field
