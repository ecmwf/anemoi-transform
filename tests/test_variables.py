# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import json

import pytest
from anemoi.utils.dates import as_timedelta

from anemoi.transform.fields import FieldList
from anemoi.transform.variables import Variable
from anemoi.transform.variables.components import VariableFromComponents
from anemoi.transform.variables.from_dict import VariableFromDict


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


def test_from_dict_dispatches_on_schema() -> None:
    """Test that from_dict selects the class from the per-variable schema key.

    Tests:
    - Legacy dictionaries (no schema key) deserialise to VariableFromDict.
    - 'variable/1' dictionaries deserialise to VariableFromComponents.
    - Unknown schemas raise a ValueError.
    """
    legacy = Variable.from_dict("2t", {"mars": {"param": "2t", "levtype": "sfc"}})
    assert isinstance(legacy, VariableFromDict)

    components = Variable.from_dict(
        "t_850",
        {
            "schema": "variable/1",
            "parameter": {"variable": "t", "units": "K"},
            "vertical": {"level_type": "pressure", "level": 850},
        },
    )
    assert isinstance(components, VariableFromComponents)

    with pytest.raises(ValueError, match="unknown serialisation schema"):
        Variable.from_dict("x", {"schema": "variable/99"})


def test_variable_from_components() -> None:
    """Test the properties of a variable deserialised from the 'variable/1' schema.

    Tests:
    - Level type and level come from the vertical component (ekd names mapped to MARS-style).
    - Units come from parameter.units, param from parameter.variable.
    - Time processing, period and the create-time flags are read from the top level.
    - grib_keys falls back to component-derived MARS-style keys when there is no mars section.
    """
    tp = Variable.from_dict(
        "tp",
        {
            "schema": "variable/1",
            "parameter": {"variable": "tp", "units": "m"},
            "vertical": {"level_type": "surface", "level": 0},
            "process": "accumulation",
            "period": ["0h", "6h"],
        },
    )
    assert tp.is_surface_level
    assert not tp.is_pressure_level
    assert tp.param == "tp"
    assert tp.is_accumulation
    assert tp.is_valid_over_a_period
    assert tp.time_processing == "accumulation"
    assert tp.period == as_timedelta("6h")
    assert not tp.is_computed_forcing
    assert not tp.is_constant_in_time
    assert tp.grib_keys == {"param": "tp", "levtype": "sfc", "levelist": 0}

    t850 = Variable.from_dict(
        "t_850",
        {
            "schema": "variable/1",
            "parameter": {"variable": "t", "units": "K"},
            "vertical": {"level_type": "pressure", "level": 850},
            "mars": {"param": "t", "levtype": "pl", "levelist": 850},
        },
    )
    assert t850.is_pressure_level
    assert t850.level == 850
    assert t850.period == as_timedelta(0)
    assert t850.grib_keys == {"param": "t", "levtype": "pl", "levelist": 850}

    forcing = Variable.from_dict(
        "cos_latitude",
        {
            "schema": "variable/1",
            "parameter": {"variable": "cos_latitude"},
            "computed_forcing": True,
            "constant_in_time": True,
        },
    )
    assert forcing.is_computed_forcing
    assert forcing.is_constant_in_time
    assert forcing.is_surface_level is None


def test_retrieval_request() -> None:
    """Test that retrieval_request rebuilds a repository request from a variable.

    Tests:
    - The legacy layout exposes its mars block as a 'mars' retrieval request.
    - The 'variable/1' layout exposes its stored mars block the same way.
    - A variable without a mars block returns None.
    - An unknown repository raises a ValueError.
    """
    legacy = Variable.from_dict("2t", {"mars": {"param": "2t", "levtype": "sfc"}})
    assert legacy.retrieval_request("mars") == {"param": "2t", "levtype": "sfc"}
    assert legacy.retrieval_request() == {"param": "2t", "levtype": "sfc"}

    components = Variable.from_dict(
        "t_850",
        {
            "schema": "variable/1",
            "parameter": {"variable": "t", "units": "K"},
            "vertical": {"level_type": "pressure", "level": 850},
            "mars": {"param": "t", "levtype": "pl", "levelist": 850},
        },
    )
    assert components.retrieval_request("mars") == {"param": "t", "levtype": "pl", "levelist": 850}

    no_mars = Variable.from_dict(
        "cos_latitude",
        {"schema": "variable/1", "parameter": {"variable": "cos_latitude"}},
    )
    assert no_mars.retrieval_request("mars") is None

    with pytest.raises(ValueError, match="Unknown retrieval system"):
        legacy.retrieval_request("does-not-exist")


def test_from_field_round_trip() -> None:
    """Test that Variable.from_field serialises to JSON and deserialises equivalently.

    Tests:
    - as_dict() is JSON-compatible and carries the 'variable/1' schema.
    - The deserialised variable agrees with the live-field one on every property.
    - Both variables are compatible() with each other.
    """
    fields = FieldList.from_source("sample", "test.grib")

    for field in fields:
        name = field.get("parameter.variable")
        live = Variable.from_field(name, field)
        data = json.loads(json.dumps(live.as_dict()))

        assert data["schema"] == "variable/1"
        assert data["parameter"]["variable"] == name

        stored = Variable.from_dict(name, data)
        assert isinstance(stored, VariableFromComponents)

        for prop in (
            "param",
            "level",
            "is_pressure_level",
            "is_model_level",
            "is_surface_level",
            "time_processing",
            "period",
            "is_accumulation",
            "units",
        ):
            assert getattr(live, prop) == getattr(stored, prop), prop

        assert live.compatible(stored)
        assert stored.as_dict() == data


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
