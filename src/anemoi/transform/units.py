# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Physical units of variables, compared through their canonical (pint) form."""

import threading
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pint

# Spellings that pint cannot parse, rewritten before parsing. Applied
# repeatedly (alternating with pint) until the string reaches a fixed point.
UNITS_MAPPING = {
    "Numeric": "dimensionless",  # This is WMO, but Numeric will choke pint or cfunits
    # GRIB/eccodes writes water equivalent dot-separated ("m.of.water.equivalent")
    # while the parameter database uses spaces ("m of water equivalent"). pint can
    # parse neither, so without this they would canonicalise to two different
    # strings and compare unequal. Deliberately NOT mapped to "m": a depth of
    # water equivalent is a mass per area (rho_snow/rho_water times the real snow
    # depth), so treating it as a length would let snow depth in metres (sde,
    # paramId 3066) pass a check meant for sd (paramId 141).
    "m.of.water.equivalent": "m of water equivalent",
}

_UNIT_REGISTRY = None
_UNIT_REGISTRY_LOCK = threading.Lock()


def _format_units_wmo(unit, registry, **options) -> str:
    """Render a pint unit WMO/GRIB-style: short symbols, ``**`` negative exponents.

    Registered with :func:`pint.register_unit_format` under the name
    ``"anemoi"`` (see
    https://pint.readthedocs.io/en/stable/user/formatting.html), so
    ``f"{unit:anemoi}"`` produces e.g. ``"kg m**-2"`` or ``"m s**-1"``.
    Positive exponents come first, then negative, alphabetical within each
    group — a deterministic order for canonical comparison.

    Parameters
    ----------
    unit : pint.util.UnitsContainer
        The unit to format, as a ``{name: exponent}`` mapping.
    registry : pint.UnitRegistry
        The registry the unit belongs to (used to resolve symbols).
    **options : Any
        Ignored (part of the pint formatter interface).

    Returns
    -------
    str
        The formatted units (``"dimensionless"`` for a dimensionless unit).
    """
    if not unit:
        return "dimensionless"

    def _symbol(name: str) -> str:
        try:
            return registry.get_symbol(name)
        except Exception:  # e.g. "dimensionless" has no symbol
            return name

    parts = sorted(((_symbol(name), power) for name, power in unit.items()), key=lambda i: (i[1] < 0, i[0]))
    return " ".join(s if p == 1 else f"{s}**{float(p):g}" for s, p in parts)


def _unit_registry() -> "pint.UnitRegistry":
    """Return the shared pint unit registry, created lazily (it is slow to build).

    First use also registers the ``"anemoi"`` unit format (see
    :func:`_format_units_wmo`).

    Returns
    -------
    pint.UnitRegistry
        The shared registry.
    """
    global _UNIT_REGISTRY
    if _UNIT_REGISTRY is None:
        with _UNIT_REGISTRY_LOCK:
            if _UNIT_REGISTRY is None:
                import pint

                pint.register_unit_format("anemoi")(_format_units_wmo)
                _UNIT_REGISTRY = pint.UnitRegistry()
    return _UNIT_REGISTRY


class Units:
    """The units of a variable.

    The original spelling is kept (``str()``/``repr()`` return it unchanged)
    but equality and hashing use the canonical form, so ``Units("m/s") ==
    Units("m s**-1")``.
    """

    def __init__(self, units: str) -> None:
        """Parameters
        -------------
        units : str
            The units, in any spelling (e.g. ``"m s**-1"``).
        """
        self.units = units

    def __str__(self) -> str:
        """Return the original units spelling."""
        return self.units

    def __repr__(self) -> str:
        """Return the original units spelling."""
        return self.units

    def __format__(self, format_spec: str) -> str:
        """Format the units; the ``"c"`` spec selects the canonical form.

        ``f"{units}"`` prints the original spelling, ``f"{units:c}"`` the
        canonical form; any other spec is applied to the original spelling
        (e.g. ``f"{units:>10}"``).

        Parameters
        ----------
        format_spec : str
            The format specification.

        Returns
        -------
        str
            The formatted units.
        """
        match format_spec:
            case "c":
                return self.canonical
            case "":
                return self.units
            case _:
                return format(self.units, format_spec)

    def __eq__(self, value) -> bool:
        """Compare canonical forms, so different spellings of the same units are equal.

        Parameters
        ----------
        value : Units or str
            The units to compare with.

        Returns
        -------
        bool
            True when both canonical forms match.
        """
        if isinstance(value, Units):
            return self.canonical == value.canonical
        elif isinstance(value, str):
            return self.canonical == self.to_canonical(value)
        else:
            return NotImplemented

    def __hash__(self) -> int:
        """Hash the canonical form (consistent with ``__eq__``)."""
        return hash(self.canonical)

    @cached_property
    def canonical(self) -> str:
        """The canonical form of these units (see :meth:`to_canonical`)."""
        return self.to_canonical(self.units)

    @classmethod
    def to_canonical(cls, units: str) -> str:
        """Convert a units string to its canonical form.

        Alternates ``UNITS_MAPPING`` rewrites with pint parsing until the
        string no longer changes; the canonical spelling is the pint unit
        rendered with the custom ``"anemoi"`` format (WMO/GRIB-style short
        symbols with negative exponents, e.g. ``"m s**-1"`` — registered
        via :func:`pint.register_unit_format`, see
        https://pint.readthedocs.io/en/stable/user/formatting.html).
        Strings pint cannot parse are returned as they are after the
        mapping (e.g. WMO oddities such as ``"(0 - 1)"``).

        Parameters
        ----------
        units : str
            The units to convert, in any spelling.

        Returns
        -------
        str
            The canonical form (e.g. ``"m s**-1"`` for ``"m/s"``,
            ``"m/sec"`` or ``"meter / second"``).
        """
        registry = _unit_registry()
        result = units
        while True:
            previous = result
            result = UNITS_MAPPING.get(result, result)
            try:
                result = f"{registry.parse_units(result):anemoi}"
            except Exception:
                pass
            if result == previous:
                return result


# --------------------------------------------------------------------------- #
# earthkit-data units work-around
#
# earthkit's unit parser aliases the WMO/GRIB spelling "(0 - 1)" (a 0-1
# *fraction*, e.g. cloud cover / albedo) to "percent" -- wrong by a factor of
# 100 relative to the data (see ``earthkit-units-issue.md``). A 0-1 fraction is
# dimensionless, so map "(0 - 1)"/"0 - 1" to "dimensionless" instead. ("%" is
# left to earthkit, which already normalises it to "percent".)
#
# The patch is installed ONLY when earthkit is observed to mishandle "(0 - 1)",
# so it disables itself automatically once earthkit is fixed upstream. earthkit
# itself is left untouched.
# --------------------------------------------------------------------------- #

# WMO/GRIB unit spellings normalised at dataset-build time. Native (un-rescaled)
# fields are recorded straight from their GRIB ``metadata.units`` and never pass
# through earthkit's parser, so the fraction/percent spellings must be remapped
# here too. Only these two are touched -- every other GRIB spelling (``m``,
# ``K``, ``kg m**-2`` ...) is left as-is.
BUILD_UNIT_ALIASES = {"(0 - 1)": "dimensionless", "%": "percent"}

_FRACTION_SPELLINGS = ("(0 - 1)", "0 - 1")


def _earthkit_mishandles_fraction() -> bool:
    """Return True iff earthkit maps the 0-1 fraction spelling "(0 - 1)" to percent."""
    try:
        from earthkit.utils.units import Units as _EkUnits

        return _EkUnits.from_any("(0 - 1)") == "%"
    except Exception:
        return False


def _install_earthkit_fraction_fix() -> bool:
    """Map the 0-1 fraction spellings "(0 - 1)"/"0 - 1" to "dimensionless"
    instead of earthkit's (wrong) "percent".

    No-op unless earthkit actually exhibits the bug, so the work-around vanishes
    once earthkit is fixed. Returns whether the patch is now in place.
    """
    if not _earthkit_mishandles_fraction():
        return False

    from earthkit.utils.units import units as _ek

    if getattr(_ek, "_anemoi_fraction_fix", False):
        return True

    _ek.UNIT_STR_ALIASES.pop("(0 - 1)", None)
    _original_from_any = _ek.Units.from_any

    def _from_any(units):
        if isinstance(units, str) and units.strip() in _FRACTION_SPELLINGS:
            return _original_from_any("dimensionless")
        return _original_from_any(units)

    _ek.Units.from_any = staticmethod(_from_any)
    _ek._anemoi_fraction_fix = True
    return True


_install_earthkit_fraction_fix()
