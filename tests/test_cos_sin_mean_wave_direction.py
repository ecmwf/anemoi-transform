# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import earthkit.data as ekd

from anemoi.transform.filters import filter_registry

prototype = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "values": [1, 2, 3, 4, 5, 6],
    "valid_datetime": "2018-08-01T09:00:00Z",
}


def test_cos_sin_mean_wave_direction():
    """Test the cos_sin_mean_wave_direction filter."""
    data = [{"param": "mwd", **prototype}]

    filter = filter_registry.create(
        "cos_sin_mean_wave_direction",
        mean_wave_direction="mwd",
        cos_mean_wave_direction="cos_mwd",
        sin_mean_wave_direction="sin_mwd",
    )

    source = ekd.from_source("list-of-dicts", data)
    target = filter.forward(source)
    assert len(target) == 2, f"Expected 2 fields, got {len(target)}"


if __name__ == "__main__":
    """
    Run all test functions that start with 'test_'.
    """
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
