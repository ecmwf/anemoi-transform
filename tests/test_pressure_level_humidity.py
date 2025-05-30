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


def test_pressure_level_specific_humidity_to_relative_humidity_from_file():
    source = source_registry.create(
        "testing", dataset="anemoi-transform/filters/era_20240601_pressure_level_specific_humidity.grib"
    )

    q_to_r = filter_registry.create("q_to_r")

    output = source | q_to_r
    assert len(list(output)) == 6  # since we have 2 levels
    output = np.stack([v.to_numpy() for v in list(output) if "r" in v.metadata("param")]).flatten()

    output_era_r = (
        source_registry.create("testing", dataset="anemoi-transform/filters/era_r.npy").ds.to_numpy().flatten()
    )
    np.testing.assert_allclose(output, output_era_r)


def test_pressure_level_specific_humidity_to_relative_humidity():
    source_specific_humidity = source_registry.create("testing", fields=pressure_level_specific_humidity_source)

    q_to_r = filter_registry.create("q_to_r")
    r_to_q = filter_registry.create("r_to_q")

    relative_humidity_transform_output = source_specific_humidity | q_to_r
    assert len(list(relative_humidity_transform_output)) == 6  # since we have 2 levels

    relative_humidity_transform_output = ListSource(
        convert_to_ekd_fieldlist(relative_humidity_transform_output).sel(param=["r", "t"])
    )
    specific_humidity_transform_output = ListSource(
        convert_to_ekd_fieldlist(list(relative_humidity_transform_output | r_to_q)).sel(param=["q", "t"])
    )

    assert len(specific_humidity_transform_output.fields) == 4

    for original, converted in zip(source_specific_humidity.ds.sel(param='t'),specific_humidity_transform_output.fields.sel(param='t')):
        assert np.allclose(original.to_numpy(), converted.to_numpy()), (
            (original.metadata("param")),
            (converted.metadata("param")),
            original.to_numpy(),
            converted.to_numpy(),
            original.to_numpy() == converted.to_numpy(),
            original.to_numpy() - converted.to_numpy(),
        )

    for original, converted in zip(source_specific_humidity.ds.sel(param='q'),specific_humidity_transform_output.fields.sel(param='q')):
        assert np.allclose(original.to_numpy(), converted.to_numpy()), (
            (original.metadata("param")),
            (converted.metadata("param")),
            original.to_numpy(),
            converted.to_numpy(),
            original.to_numpy() == converted.to_numpy(),
            original.to_numpy() - converted.to_numpy(),
        )


def test_pressure_level_relative_humidity_to_specific_humidity_from_file():
    source = source_registry.create("testing", dataset="anemoi-transform/filters/cerra_20240601_pressure_levels.grib")

    r_to_q = filter_registry.create("r_to_q")

    output = source | r_to_q
    assert len(list(output)) == 6  # since we have 2 levels
    output = np.stack([v.to_numpy() for v in list(output) if "q" in v.metadata("param")]).flatten()

    output_cerra_q = (
        source_registry.create("testing", dataset="anemoi-transform/filters/cerra_q.npy").ds.to_numpy().flatten()
    )
    np.testing.assert_allclose(output, output_cerra_q)


def test_pressure_level_relative_humidity_to_specific_humidity_from_file_AROME():
    source = source_registry.create("testing", dataset="anemoi-transform/filters/r_t_PAAROME_1S40_ECH0_ISOBARE.grib")

    r_to_q = filter_registry.create("r_to_q")
    q_to_r = filter_registry.create("q_to_r")
    output = source | r_to_q
    assert len(list(output)) == 6  # since we have 2 levels
    output = np.stack([v.to_numpy() for v in list(output) if "q" in v.metadata("param")]).flatten()
    output_cerra_q = (
        source_registry.create("testing", dataset="anemoi-transform/filters/arome_specific_humidity.npy")
        .ds.to_numpy()
        .flatten()
    )
    np.testing.assert_allclose(output, output_cerra_q)


def test_pressure_level_relative_humidity_to_relative_humidity():
    source_relative_humidity = source_registry.create("testing", fields=pressure_level_relative_humidity_source)
    q_to_r = filter_registry.create("q_to_r")
    r_to_q = filter_registry.create("r_to_q")

    specific_humidity_transform_output = source_relative_humidity | r_to_q
    assert len(list(specific_humidity_transform_output)) == 6  # since we have 2 levels

    specific_humidity_transform_output = ListSource(
        convert_to_ekd_fieldlist(specific_humidity_transform_output).sel(param=["q", "t"])
    )
    relative_humidity_transform_output = ListSource(
        convert_to_ekd_fieldlist(list(specific_humidity_transform_output | q_to_r)).sel(param=["r", "t"])
    )

    assert len(relative_humidity_transform_output.fields) == 4

    for original, converted in zip(source_relative_humidity.ds.sel(param='r'), relative_humidity_transform_output.fields.sel(param='r')):
        assert np.allclose(original.to_numpy(), converted.to_numpy()), (
            (original.metadata("param")),
            (converted.metadata("param")),
            original.to_numpy(),
            converted.to_numpy(),
            original.to_numpy() == converted.to_numpy(),
            original.to_numpy() - converted.to_numpy(),
        )

    for original, converted in zip(source_relative_humidity.ds.sel(param='t'), relative_humidity_transform_output.fields.sel(param='t')):
        assert np.allclose(original.to_numpy(), converted.to_numpy()), (
            (original.metadata("param")),
            (converted.metadata("param")),
            original.to_numpy(),
            converted.to_numpy(),
            original.to_numpy() == converted.to_numpy(),
            original.to_numpy() - converted.to_numpy(),
        )


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
