# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from anemoi.transform.variables import Variable


def test_variables():
    z500 = Variable.from_dict("z500", {"mars": {"param": "z", "levtype": "pl", "levelist": 500}})

    assert z500.is_pressure_level
    assert z500.level == 500

    msl = Variable.from_dict("msl", {"mars": {"param": "msl", "levtype": "sfc"}})

    assert not msl.is_pressure_level
    assert msl.level is None


if __name__ == "__main__":
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
