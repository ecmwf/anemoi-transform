# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import filter_registry
from anemoi.transform.sources import source_registry
from tests.utils import collect_fields_by_param

# Mock data for testing
MOCK_FIELD_METADATA = {
    "latitudes": [39.639006],  # , 39.64331737],
    "longitudes": [334.553253],  #  , 334.57510742],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

U_VALUES = np.array([[1.6866188]])  # , 1.6768532], [2.6513138, 2.6454544]]).flatten()
V_VALUES = np.array([[-3.9099693]])  # , -3.9724693], [-4.6877003, -4.69063]]).flatten()

# Expected values after rotation and unrotation
U_ROTATED_VALUES = np.array([[2.604725]])  # , 2.6097064], [3.732331, 3.7261608]]).flatten()
V_ROTATED_VALUES = np.array([[-3.3686721]])  # , -3.4324598], [-3.8824868, -3.8879511]]).flatten()


@pytest.fixture
def wind_source(test_source):
    WIND_SPEC = [
        {"param": "u", "values": U_VALUES, **MOCK_FIELD_METADATA},
        {"param": "v", "values": V_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(WIND_SPEC)


@pytest.fixture
def rotated_wind_source(test_source):
    ROTATED_WIND_SPEC = [
        {"param": "u", "values": U_ROTATED_VALUES, **MOCK_FIELD_METADATA},
        {"param": "v", "values": V_ROTATED_VALUES, **MOCK_FIELD_METADATA},
    ]
    return test_source(ROTATED_WIND_SPEC)


@skip_if_offline
def test_rotate_winds_from_file(test_source):
    source = source_registry.create(
        "testing", dataset=ekd.from_source("file", "mylocalfile/fc2020100106+000POSTP_CONTROL_grib2")
    )
    rotate_winds = filter_registry.create("rotate_winds", x_wind="10u", y_wind="10u")

    pipeline = source | rotate_winds

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    print("Source contents:", source)
    print("source.ds.fields :", source.ds.fields)
    print("Set Input fields:", set(input_fields))

    # Check for expected params
    assert set(input_fields) == {"10u"}
    assert set(output_fields) == {"10u"}

    # Test pipeline output matches known good output
    expected_u_rotated = (
        source_registry.create("testing", dataset=ekd.from_source("file", "uwcwest_20201001_10u_10v_rotated.zarr"))
        .ds.to_numpy()
        .reshape(shape)
    )  # test_source("uwcwest_20201001_10u_10v_rotated.zarr").ds.to_numpy().reshape(shape)
    expected_v_rotated = (
        source_registry.create("testing", dataset=ekd.from_source("file", "uwcwest_20201001_10u_10v_rotated.zarr"))
        .ds.to_numpy()
        .reshape(shape)
    )  # test_source("uwcwest_20201001_10u_10v_rotated.zarr").ds.to_numpy().reshape(shape)

    assert len(output_fields["u"]) == 1
    assert len(output_fields["v"]) == 1

    result_u = output_fields["u"][0]
    result_v = output_fields["v"][0]

    assert np.allclose(result_u.to_numpy(), expected_u_rotated)
    assert np.allclose(result_v.to_numpy(), expected_v_rotated)


# def test_rotate_winds(wind_source):
#     rotate_winds = filter_registry.create("rotate_winds", x_wind="u", y_wind="v")
#     pipeline = wind_source | rotate_winds
#     input_fields = collect_fields_by_param(wind_source)
#     output_fields = collect_fields_by_param(pipeline)

#  # Check for expected params
#     print("Input fields:", input_fields)
#     print("Output fields:", output_fields)
#     assert set(input_fields) == {"u", "v"}
#     assert set(output_fields) == {"u", "v"}

#     # Test new output matches expected values
#     assert len(output_fields["u"]) == 1
#     assert len(output_fields["v"]) == 1

#     result_u = output_fields["u"][0]
#     result_v = output_fields["v"][0]

#     print("Result U:", result_u.to_numpy())
#     print("Expected U:", U_ROTATED_VALUES)
#     print("Result V:", result_v.to_numpy())
#     print("Expected V:", V_ROTATED_VALUES)

#     assert np.allclose(result_u.to_numpy(), U_ROTATED_VALUES)
#     assert np.allclose(result_v.to_numpy(), V_ROTATED_VALUES)

#     # Capture and print output
#     captured = capsys.readouterr()
#     print("Captured output:", captured.out)


# def test_unrotate_winds(rotated_wind_source):
#     unrotate_winds = filter_registry.create("unrotate_winds", x_wind="u", y_wind="v")
#     pipeline = rotated_wind_source | unrotate_winds
#     input_fields = collect_fields_by_param(rotated_wind_source)
#     output_fields = collect_fields_by_param(pipeline)

#     # Check for expected params
#     assert set(input_fields) == {"u", "v"}
#     assert set(output_fields) == {"u", "v"}

#     # Test new output matches expected values
#     assert len(output_fields["u"]) == 1
#     assert len(output_fields["v"]) == 1

#     result_u = output_fields["u"][0]
#     result_v = output_fields["v"][0]

#     assert np.allclose(result_u.to_numpy(), U_VALUES)
#     assert np.allclose(result_v.to_numpy(), V_VALUES)

# def test_rotate_unrotate_winds_round_trip(wind_source):
#     rotate_winds = filter_registry.create("rotate_winds", x_wind="u", y_wind="v")
#     unrotate_winds = filter_registry.create("unrotate_winds", x_wind="u", y_wind="v")

#     rotated_wind_source = wind_source | rotate_winds
#     pipeline = rotated_wind_source | unrotate_winds

#     input_fields = collect_fields_by_param(wind_source)
#     output_fields = collect_fields_by_param(pipeline)

#     # Check for expected params
#     assert set(input_fields) == {"u", "v"}
#     assert set(output_fields) == {"u", "v"}

#     # Test new output matches expected values
#     assert len(output_fields["u"]) == 1
#     assert len(output_fields["v"]) == 1

#     result_u = output_fields["u"][0]
#     result_v = output_fields["v"][0]

#     assert np.allclose(result_u.to_numpy(), U_VALUES)
#     assert np.allclose(result_v.to_numpy(), V_VALUES)

# @skip_if_offline
# def test_rotate_winds_from_file(test_source):
#     source = test_source("path/to/your/wind_data.grib")
#     rotate_winds = filter_registry.create("rotate_winds", x_wind="u", y_wind="v")
#     pipeline = source | rotate_winds

#     input_fields = collect_fields_by_param(source)
#     output_fields = collect_fields_by_param(pipeline)

#     # Check for expected params
#     assert set(input_fields) == {"u", "v"}
#     assert set(output_fields) == {"u", "v"}

#     # Test pipeline output matches known good output
#     expected_u_rotated = test_source("path/to/expected_u_rotated.npy").ds.to_numpy().reshape(shape)
#     expected_v_rotated = test_source("path/to/expected_v_rotated.npy").ds.to_numpy().reshape(shape)

#     assert len(output_fields["u"]) == 1
#     assert len(output_fields["v"]) == 1

#     result_u = output_fields["u"][0]
#     result_v = output_fields["v"][0]

#     assert np.allclose(result_u.to_numpy(), expected_u_rotated)
#     assert np.allclose(result_v.to_numpy(), expected_v_rotated)

if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
