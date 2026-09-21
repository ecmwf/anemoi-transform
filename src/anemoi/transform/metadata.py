# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Mapping between MARS-style metadata keys and earthkit-data component keys.

earthkit-data 1.0 replaced flat MARS-style metadata keys (``param``, ``levelist``,
...) with namespaced component keys (``parameter.variable``, ``vertical.level``,
...). This module is the single source of truth for that mapping.

All the keys below can be both read (``field.get(key)``) and written
(``field.set(**{key: value})``).
"""

from typing import Any

import earthkit.data as ekd

# MARS-style metadata key -> earthkit-data component key.
_MARS_TO_COMPONENT: dict[str, str] = {
    "param": "parameter.variable",
    "units": "parameter.units",
    "levtype": "vertical.level_type",
    "levelist": "vertical.level",
    "number": "ensemble.member",
    "step": "time.step",
    "valid_datetime": "time.valid_datetime",
    "base_datetime": "time.base_datetime",
}

# earthkit-data component key -> MARS-style metadata key.
_COMPONENT_TO_MARS: dict[str, str] = {v: k for k, v in _MARS_TO_COMPONENT.items()}

# Sentinel so that None can be passed as a default.
_RAISE = object()


def mars_to_component(key: str, default: Any = _RAISE) -> Any:
    """Map a MARS-style metadata key to its earthkit-data component key.

    Parameters
    ----------
    key : str
        The MARS-style metadata key, e.g. ``"param"``.
    default : Any, optional
        Value to return if the key is unknown. If not given, a KeyError is raised.

    Returns
    -------
    Any
        The component key, e.g. ``"parameter.variable"``, or ``default``.

    Raises
    ------
    KeyError
        If the key is unknown and no default was given.
    """
    if key in _MARS_TO_COMPONENT:
        return _MARS_TO_COMPONENT[key]
    if default is _RAISE:
        raise KeyError(f"Unknown MARS metadata key '{key}'. Known keys are: {sorted(mars_keys())}")
    return default


def component_to_mars(key: str, default: Any = _RAISE) -> Any:
    """Map an earthkit-data component key to its MARS-style metadata key.

    Parameters
    ----------
    key : str
        The component key, e.g. ``"parameter.variable"``.
    default : Any, optional
        Value to return if the key is unknown. If not given, a KeyError is raised.

    Returns
    -------
    Any
        The MARS-style key, e.g. ``"param"``, or ``default``.

    Raises
    ------
    KeyError
        If the key is unknown and no default was given.
    """
    if key in _COMPONENT_TO_MARS:
        return _COMPONENT_TO_MARS[key]
    if default is _RAISE:
        raise KeyError(f"Unknown component metadata key '{key}'. Known keys are: {sorted(component_keys())}")
    return default


def mars_keys() -> frozenset[str]:
    """The set of known MARS-style metadata keys.

    Returns
    -------
    frozenset[str]
        The known MARS-style keys.
    """
    return frozenset(_MARS_TO_COMPONENT)


def component_keys() -> frozenset[str]:
    """The set of known earthkit-data component keys.

    Returns
    -------
    frozenset[str]
        The known component keys.
    """
    return frozenset(_COMPONENT_TO_MARS)


def get_metadata(field: ekd.Field, key: str) -> Any:
    """Get a metadata value from a field by key.

    Raw metadata keys (e.g. GRIB keys such as ``shortName``) are tried first, then
    the key is mapped to its earthkit-data component key, if there is one.

    Parameters
    ----------
    field : ekd.Field
        The field to read the metadata from.
    key : str
        The metadata key. Can be a raw metadata key, a MARS-style key, or an
        earthkit-data component key.

    Returns
    -------
    Any
        The metadata value.

    Raises
    ------
    KeyError
        If the key cannot be resolved for this field.
    """
    try:
        return field.metadata(key)
    except (KeyError, TypeError):
        pass

    try:
        return field.get(mars_to_component(key, default=key))
    except (KeyError, TypeError) as e:
        raise KeyError(f"Cannot get metadata for key '{key}'") from e
