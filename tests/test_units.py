# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest

from anemoi.transform.units import Units


@pytest.mark.parametrize(
    "units,expected",
    [
        # Different spellings of the same units converge
        ("m/s", "m s**-1"),
        ("m s**-1", "m s**-1"),
        ("m/sec", "m s**-1"),
        ("meter / second", "m s**-1"),
        # Base and derived units
        ("K", "K"),
        ("Pa", "Pa"),
        ("hPa", "hPa"),
        ("s", "s"),
        ("J kg**-1", "J kg**-1"),
        ("W m**-2", "W m**-2"),
        ("kg m**-2", "kg m**-2"),
        ("m**2 s**-2", "m**2 s**-2"),
        ("s/m", "s m**-1"),  # positive exponents first
        ("degC", "°C"),
        ("degrees", "deg"),
        ("%", "%"),
        # Dimensionless spellings all converge
        ("1", "dimensionless"),
        ("dimensionless", "dimensionless"),
        ("kg kg**-1", "dimensionless"),
        ("Numeric", "dimensionless"),  # WMO, via UNITS_MAPPING then pint
        ("", "dimensionless"),
        # Strings pint cannot parse are returned unchanged
        ("(0 - 1)", "(0 - 1)"),
        ("m of water equivalent", "m of water equivalent"),
    ],
)
def test_to_canonical(units: str, expected: str) -> None:
    """Test that to_canonical converts many different units to their canonical form."""
    assert Units.to_canonical(units) == expected


def test_units_equality() -> None:
    """Test that Units compare (and hash) through their canonical form."""
    assert Units("m/s") == Units("m s**-1")
    assert Units("m/s") == "m s**-1"
    assert Units("Numeric") == Units("kg kg**-1")
    assert Units("K") != Units("degC")
    assert Units("(0 - 1)") == Units("(0 - 1)")
    assert Units("(0 - 1)") != Units("dimensionless")

    assert hash(Units("m/s")) == hash(Units("m s**-1"))
    assert str(Units("m s**-1")) == "m s**-1"


def test_units_format() -> None:
    """Test that formatting preserves the spelling and ':c' selects the canonical form."""
    units = Units("m/s")
    assert f"{units}" == "m/s"
    assert f"{units:c}" == "m s**-1"
    assert f"{units:>10}" == "       m/s"


if __name__ == "__main__":
    pytest.main([__file__])
