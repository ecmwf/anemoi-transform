# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


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


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
