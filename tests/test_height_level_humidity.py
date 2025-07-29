import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.sources import source_registry
from anemoi.transform.testing import convert_to_ekd_fieldlist
import earthkit.meteo as ekm
import earthkit.data as ekd

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


def test_specific_humidity_to_relative_humidity_from_file():
    source = source_registry.create(
        "testing", dataset="anemoi-transform/filters/input_single_level_specific_humidity_to_relative_humidity.grib"
    )
    print(source.ds.metadata("param"))

    # IFS A and B coeffients for level 137 - 129
    AB_coefficients = {
        "A": [424.414063, 302.476563, 202.484375, 122.101563, 62.781250, 22.835938, 3.757813, 0.0, 0.0],
        "B": [0.969513, 0.975078, 0.980072, 0.984542, 0.988500, 0.991984, 0.995003, 0.997630, 1.000000],
    }

    q_to_r_height = filter_registry.create(
        "q_to_r_height",
        height=2,
        specific_humidity_at_height_level="2sh",
        relative_humidity_at_height_level="2r",
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        AB=AB_coefficients,
    )

    output = source | q_to_r_height

    # Input has 6 params (2d, 2sh, 2t, sp, t and q) 
    # output should have 5 params (2d, 2sh, 2t, sp and new 2rh), t and q at height levels are dropped
    assert len(list(output)) == 5 

    # Relative humidity calculated by the filter
    relative_humidity_transform_output = convert_to_ekd_fieldlist(output).sel(
        param="2r"
    )[0].to_numpy()   


    # The reference relative humidity
    output = source_registry.create(
        "testing", dataset="anemoi-transform/filters/input_single_level_specific_humidity_to_relative_humidity.grib"
    )
    relative_humidity_output = source_registry.create(
        "testing", dataset="anemoi-transform/filters/single_level_relative_humidity.npy"
    ).ds.to_numpy()

    np.testing.assert_allclose(relative_humidity_transform_output, relative_humidity_output)

def test_specific_humidity_to_relative_humidity():
    pass


def test_relative_humidity_to_specific_humidity_from_file():

    source = source_registry.create(
        "testing", dataset="anemoi-transform/filters/input_single_level_specific_humidity_to_relative_humidity.grib"
    )
    relative_humidity_at_height_level = source_registry.create(
        "testing", dataset="anemoi-transform/filters/single_level_relative_humidity.npy"
    ).ds.to_numpy()

    md = source.ds.sel(param="2d")[0].metadata().override(edition=2, shortName="2r")
    
    source.ds += ekd.FieldList.from_array(
            relative_humidity_at_height_level,
            md
        )
    
    print(source.ds.metadata("param"))
    
    
    # IFS A and B coeffients for level 137 - 129
    AB_coefficients = {
        "A": [424.414063, 302.476563, 202.484375, 122.101563, 62.781250, 22.835938, 3.757813, 0.0, 0.0],
        "B": [0.969513, 0.975078, 0.980072, 0.984542, 0.988500, 0.991984, 0.995003, 0.997630, 1.000000],
    }

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

    output = source | r_to_q_height

    # Input has 7 params (2sh, 2d, 2r, 2t, sp, t and q) 
    # output should have 6 params (2sh, 2d, 2r, 2t, sp and new 2q), t and q at height levels are dropped
    assert len(list(output)) == 6 

    specific_humidity_transform_output = convert_to_ekd_fieldlist(output).sel(
        param="2q"
    ).to_numpy()   

    specific_humidity_input = convert_to_ekd_fieldlist(source).sel(
        param="2sh"
    ).to_numpy()

    np.testing.assert_allclose(specific_humidity_transform_output, specific_humidity_input)


def test_relative_humidity_to_specific_humidity():
    pass


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
