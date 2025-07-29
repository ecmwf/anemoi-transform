import numpy as np
import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import filter_registry
from anemoi.transform.fields import new_fieldlist_from_list

import earthkit.data as ekd

from .utils import SelectFieldSource
from .utils import assert_fields_equal
from .utils import collect_fields_by_param

prototype = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

pressure_level_relative_humidity_source = [
    {"param": "r", "levelist": 500, "values": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], **prototype},
    {"param": "t", "levelist": 500, "values": [10, 10, 10, 10, 10, 10], **prototype},
    {"param": "r", "levelist": 800, "values": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], **prototype},
    {"param": "t", "levelist": 800, "values": [10, 10, 10, 10, 10, 10], **prototype},
]

pressure_level_specific_humidity_source = [
    {"param": "q", "levelist": 500, "values": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], **prototype},
    {"param": "t", "levelist": 500, "values": [10, 10, 10, 10, 10, 10], **prototype},
    {"param": "q", "levelist": 800, "values": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], **prototype},
    {"param": "t", "levelist": 800, "values": [10, 10, 10, 10, 10, 10], **prototype},
]

# IFS A and B coeffients for level 137 - 129
AB_coefficients = {
    "A": [424.414063, 302.476563, 202.484375, 122.101563, 62.781250, 22.835938, 3.757813, 0.0, 0.0],
    "B": [0.969513, 0.975078, 0.980072, 0.984542, 0.988500, 0.991984, 0.995003, 0.997630, 1.000000],
}

@skip_if_offline
def test_height_level_specific_humidity_to_relative_humidity_from_file(test_source):
    source = test_source("anemoi-transform/filters/input_single_level_specific_humidity_to_relative_humidity.grib"    )
    q_to_r_height = filter_registry.create(
        "q_to_r_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        relative_humidity_at_height_level="2r",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        AB=AB_coefficients,
    )

    print(source.ds.metadata("param"))
    pipeline = source | q_to_r_height

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2sh", "2d", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2d", "2t", "sp", "2r"}
    
    # test unchanged fields agree
    for param in ("2sh", "2d", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)
    
    # test pipeline output matches known good output
    result = output_fields["2r"][0].to_numpy()
    expected_relative_humidity = test_source("anemoi-transform/filters/single_level_relative_humidity.npy").ds.to_numpy()
    np.testing.assert_allclose(result, expected_relative_humidity)

def test_specific_humidity_to_relative_humidity():
    pass


def test_relative_humidity_to_specific_humidity_from_file(test_source):
    source = test_source("anemoi-transform/filters/input_single_level_specific_humidity_to_relative_humidity.grib")
    input_relative_humidity = test_source("anemoi-transform/filters/single_level_relative_humidity.npy").ds.to_numpy()

    md = source.ds.sel(param="2d")[0].metadata().override(edition=2, shortName="2r")
    
    source.ds += ekd.FieldList.from_array(
            input_relative_humidity,
            md
        )
    
    print(source.ds.metadata("param"))

    r_to_q_height = filter_registry.create(
        "r_to_q_height",
        height=2,
        specific_humidity_at_height_level="2q",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        AB=AB_coefficients,
    )

    pipeline = source | r_to_q_height

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check for expected params
    assert set(input_fields) == {"2sh", "2r", "2d", "2t", "sp", "t", "q"}
    assert set(output_fields) == {"2sh", "2r", "2d", "2t", "sp", "2q"}
    
    # test unchanged fields agree
    for param in ("2sh", "2r", "2d", "2t", "sp"):
        for input_field, output_field in zip(input_fields[param], output_fields[param]):
            assert_fields_equal(input_field, output_field)
    
    # test pipeline output matches known good output
    result = output_fields["2q"][0].to_numpy()
    expected_specific_humidity = output_fields["2sh"][0].to_numpy()
    np.testing.assert_allclose(result, expected_specific_humidity)

def test_relative_humidity_to_specific_humidity():
    pass


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
