# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import pytest
from anemoi.utils.dates import as_timedelta

from anemoi.transform.variables import Variable


def test_variables() -> None:
    """Test the Variable class for pressure level and surface level variables.

    Tests:
    - Creating a pressure level variable and checking its properties.
    - Creating a surface level variable and checking its properties.
    """
    z500: Variable = Variable.from_dict("z500", {"mars": {"param": "z", "levtype": "pl", "levelist": 500}})

    assert z500.is_pressure_level
    assert z500.level == 500

    msl: Variable = Variable.from_dict("msl", {"mars": {"param": "msl", "levtype": "sfc"}})

    assert not msl.is_pressure_level
    assert msl.level is None
    assert msl.period == as_timedelta(0)

    avg_tos: Variable = Variable.from_dict(
        "avg_tos", {"mars": {"param": "avg_tos", "levtype": "o2d"}, "period": [5, "6h"], "process": "average"}
    )

    assert avg_tos.is_valid_over_a_period
    assert avg_tos.period == as_timedelta("1h")
    assert avg_tos.time_processing == "average"


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()


def _earthkit_field(level_type: str, level: int | None = None):
    """Build a minimal earthkit field with the given vertical level type."""
    import earthkit.data as ekd
    import numpy as np

    return ekd.from_source(
        "list-of-dicts",
        [
            {
                "parameter": {"variable": "t"},
                "data": {"values": np.array([1.0])},
                "geography": {"latitudes": np.array([0.0]), "longitudes": np.array([0.0])},
                "vertical": {"level_type": level_type, "level": level},
            }
        ],
    ).to_fieldlist()[0]


@pytest.mark.parametrize(
    "level_type,expected",
    [
        ("surface", "sfc"),
        ("pressure", "pl"),
        ("hybrid", "ml"),
        ("potential_vorticity", "pv"),
        ("potential_temperature", "pt"),
        # MARS represents these as surface
        ("height_above_ground_level", "sfc"),
        ("depth_below_land_level", "sfc"),
        # no MARS equivalent - treated as unset
        ("snow", None),
        ("mean_sea", None),
        ("entire_atmosphere", None),
    ],
)
def test_variable_from_earthkit_level_type(level_type, expected) -> None:
    """earthkit-data level types are converted to MARS level types."""
    variable = Variable.from_earthkit("t", _earthkit_field(level_type, 500))
    assert variable.mars.get("levtype") == expected


def test_variable_from_earthkit_level_type_properties() -> None:
    """Unmapped level types leave the level properties undetermined (None)."""
    assert Variable.from_earthkit("t", _earthkit_field("surface")).is_surface_level is True
    assert Variable.from_earthkit("t", _earthkit_field("pressure", 500)).is_pressure_level is True
    assert Variable.from_earthkit("t", _earthkit_field("hybrid", 1)).is_model_level is True

    unmapped = Variable.from_earthkit("t", _earthkit_field("snow"))
    assert unmapped.is_surface_level is None
    assert unmapped.is_pressure_level is None
    assert unmapped.is_model_level is None


def test_variable_level_type_mapping_keys_are_known() -> None:
    """The level type mapping must be keyed on real earthkit-data level type names.

    earthkit-data's get_level_type() silently registers unknown names rather than
    raising, so a typo in a key would simply never match and fail silently.
    """
    from earthkit.data.field.component.level_type import LevelTypes

    from anemoi.transform.variables.from_dict import VariableFromEarthkit

    known_names = {level_type.value.name for level_type in LevelTypes}
    unknown = set(VariableFromEarthkit._LEVEL_TYPE_MAPPING) - known_names
    assert not unknown, f"Unknown earthkit level type names: {sorted(unknown)}"
