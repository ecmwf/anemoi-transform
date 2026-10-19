# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re

from anemoi.transform import Field
from anemoi.transform.fields import metadata_key
from anemoi.transform.naming import Naming
from anemoi.transform.naming import naming_registry

LOG = logging.getLogger(__name__)


# Pattern that matches bare {key} or {key:type} template variables but leaves
# already-prefixed {component.key} forms (e.g. {parameter.variable}) untouched.
# Group 1: key name (no dots → bare key).  Group 2: optional eccodes type
# qualifier (e.g. ":d" for double, ":l" for long).
_BARE_TEMPLATE_VAR = re.compile(r"\{(\w+)(:\w+)?\}")

# Splits a template into alternating literal / key parts, e.g.
# "{parameter.variable}_{vertical.level}" -> ["", "parameter.variable", "_",
# "vertical.level", ""].  Odd indices are keys.
_TEMPLATE_SPLIT = re.compile(r"\{([^}]*)\}")

# Computed forcing variables are identified by name alone and never carry a
# meaningful level distinction: any level suffix inherited from the template
# field they were built from is stripped from their name.
COMPUTED_FORCINGS = frozenset(
    [
        "cos_julian_day",
        "cos_latitude",
        "cos_local_time",
        "cos_longitude",
        "cos_solar_zenith_angle",
        "insolation",
        "latitude",
        "longitude",
        "sin_julian_day",
        "sin_latitude",
        "sin_local_time",
        "sin_longitude",
    ]
)


def _to_earthkit10_template(template: str) -> str:
    """Translate bare ``{key}`` → earthkit 1.0 path ``{component.key}``.

    Known legacy keys (``param``, ``level``, ``levelist``) are mapped to their
    earthkit 1.0 equivalents (``parameter.variable``, ``vertical.level``).
    Already-prefixed forms such as ``{parameter.variable}`` are left unchanged
    (idempotent).  Unknown bare keys fall back to ``{metadata.key}`` so that
    custom template entries still work.

    Eccodes type qualifiers (e.g. ``{level:d}`` for double) are preserved:
    the base key is mapped and the qualifier is appended.  When a type
    qualifier is present, the mapping always targets ``metadata.key:type``
    (not a component path) because component paths like ``vertical.level:d``
    are not supported.
    """

    def _replace(m: re.Match) -> str:
        key = m.group(1)
        type_qual = m.group(2) or ""  # e.g. ":d" or ""

        if type_qual:
            # Type-qualified keys must use metadata.key:type — component
            # paths (e.g. vertical.level:d) do not support eccodes types.
            return "{metadata." + key + type_qual + "}"
        return "{" + metadata_key(key, default=f"metadata.{key}") + "}"

    return _BARE_TEMPLATE_VAR.sub(_replace, template)


def _strip_zero_level_suffix(name: str) -> str:
    """Strip level suffixes from surface fields and computed forcing variables.

    In earthkit-data 1.0 ``vertical.level`` returns 0 for surface-type fields
    (level_type='surface', 'meanSea', etc.) rather than None.  A
    ``{param}_{levelist}`` template therefore produces names like ``"2t_0"``
    or ``"cos_latitude_0"`` for surface variables; the ``_0`` suffix is
    removed after template evaluation, restoring the legacy variable naming
    (``"2t"``, ``"cos_latitude"``).

    For computed forcing variables (e.g. ``cos_julian_day``), any trailing
    ``_<digits>`` level suffix is stripped regardless of the level value.

    Level-bearing names such as ``"t_700"`` are returned unchanged.
    """
    if name is None:
        return name

    # Surface level: always strip _0
    if name.endswith("_0"):
        return name[:-2]

    # Computed forcing variables: strip any trailing _<digits> level suffix
    for var_name in COMPUTED_FORCINGS:
        if name.startswith(var_name + "_"):
            suffix = name[len(var_name) + 1 :]
            if suffix.isdigit():
                return var_name

    return name


@naming_registry.register("template")
class TemplateNaming(Naming):
    """Name fields by evaluating a ``{key}`` template against their metadata.

    Bare legacy keys (``param``, ``level``, ``levelist``) are translated to
    their earthkit 1.0 component paths; unknown bare keys are looked up under
    ``metadata.``.  A trailing ``_0`` level suffix (surface fields) and level
    suffixes on computed forcing variables are stripped from the result.
    """

    def __init__(self, template: str) -> None:
        self.template = template
        self._parts = _TEMPLATE_SPLIT.split(_to_earthkit10_template(template))

    def name(self, field: Field) -> str:
        # Missing values drop the preceding literal separator (so a surface
        # field named with "{param}_{levelist}" is "2t", not "2t_") — the
        # same behaviour as earthkit-data's Remapping.
        bits: list[str] = []
        for i, part in enumerate(self._parts):
            if i % 2:
                value = field.get(part, default=None)
                if value is None:
                    bits = bits[:-1]
                else:
                    bits.append(str(value))
            else:
                bits.append(part)
        return _strip_zero_level_suffix("".join(bits))

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.template!r})"
