import numpy as np
from utils import ListSource
from utils import convert_to_ekd_fieldlist

from anemoi.transform.filters import filter_registry
from anemoi.transform.sources import source_registry

prototype = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

relative_humidity_source = [
    {"param": "r", "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **prototype},
    {"param": "t", "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **prototype},
]

dewpoint_source = [
    {"param": "d", "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **prototype},
    {"param": "t", "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **prototype},
]


def test_relative_humidity_to_dewpoint():

    source_relative = source_registry.create("testing", fields=relative_humidity_source)
    source_dewpoint = source_registry.create("testing", fields=dewpoint_source)

    r_2_d = filter_registry.create("r_2_d")
    d_2_r = filter_registry.create("d_2_r")

    dewpoint_transform_output = source_relative | r_2_d
    assert len(list(dewpoint_transform_output)) == 3

    relative_humidity_transform_output = source_dewpoint | d_2_r
    assert len(list(relative_humidity_transform_output)) == 3

    dewpoint_transform_output = ListSource(convert_to_ekd_fieldlist(dewpoint_transform_output).sel(param=["d", "t"]))
    relative_transform_output = ListSource(
        convert_to_ekd_fieldlist(list(dewpoint_transform_output | d_2_r)).sel(param=["r", "t"])
    )

    for original, converted in zip(source_relative, relative_transform_output):
        assert np.allclose(original.to_numpy(), converted.to_numpy()), (
            (original.metadata("param")),
            (converted.metadata("param")),
            original.to_numpy(),
            converted.to_numpy(),
            original.to_numpy() == converted.to_numpy(),
            original.to_numpy() - converted.to_numpy(),
        )


def test_relative_humidity_to_dewpoint_from_file():
    # this grib file is CERRA data that contains 2t and 2r
    source = source_registry.create("testing", dataset="anemoi-transform/filters/cerra_20240601_single_level.grib")

    # !TODO THE TEST DOESN'T WORK IF THE VARIABLES OF INPUT DOESN'T MATCH THE FORWARD DEFAULTS
    # ! the code doesn't crash but do not produce the expected output - we should catch that
    r_2_d = filter_registry.create("r_2_d", relative_humidity="2r", temperature="2t", dewpoint="2d")

    output = source | r_2_d
    assert len(list(output)) == 3
    output_dict = {v.metadata("param"): v.to_numpy() for v in list(output)}

    output_cerra_2d = (
        source_registry.create("testing", dataset="anemoi-transform/filters/cerra_2d.npy")
        .ds.to_numpy()
        .reshape(1069, 1069)
    )
    np.testing.assert_allclose(output_dict["2d"], output_cerra_2d)

    assert np.sum(np.isnan(output_dict["2d"])) == 0
    assert np.any(
        source.ds.sel(param="2r").to_numpy(flatten=True) != output_dict["2d"].flatten()
    ), "Arrays are  different"



def test_dewpoint_to_relative_humidity():

    source_dewpoint = source_registry.create("testing", fields=dewpoint_source)

    d_2_r = filter_registry.create("d_2_r")
    r_2_d = filter_registry.create("r_2_d")

    relative_transform_output = source_dewpoint | d_2_r
    assert len(list(relative_transform_output)) == 3

    relative_transform_output = ListSource(convert_to_ekd_fieldlist(relative_transform_output).sel(param=["r", "t"]))
    dewpoint_transform_output = ListSource(
        convert_to_ekd_fieldlist(list(relative_transform_output | r_2_d)).sel(param=["d", "t"])
    )

    for original, converted in zip(source_dewpoint, dewpoint_transform_output):
        assert np.allclose(original.to_numpy(), converted.to_numpy()), (
            (original.metadata("param")),
            (converted.metadata("param")),
            original.to_numpy(),
            converted.to_numpy(),
            original.to_numpy() == converted.to_numpy(),
            original.to_numpy() - converted.to_numpy(),
        )


def test_dewpoint_to_relative_humidity_from_file():

    source = source_registry.create(
        "testing", dataset="anemoi-transform/filters/era_20240601_single_level_dewpoint.grib"
    )
    d_2_r = filter_registry.create("d_2_r", relative_humidity="2r", temperature="2t", dewpoint="2d")

    output = source | d_2_r
    assert len(list(output)) == 3
    output_dict = {v.metadata("param"): v.to_numpy() for v in list(output)}

    era5_2r = (
        source_registry.create("testing", dataset="anemoi-transform/filters/era5_2r.npy").ds.to_numpy().reshape(9, 18)
    )
    np.testing.assert_allclose(output_dict["2r"], era5_2r)

    assert np.sum(np.isnan(output_dict["2r"])) == 0
    assert np.any(
        source.ds.sel(param="2d").to_numpy(flatten=True) != output_dict["2r"].flatten()
    ), "Arrays are  different"


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
