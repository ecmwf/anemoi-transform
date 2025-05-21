import numpy as np

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


def test_presurre_level_specific_humidity_to_relative_humidity_from_file():
    source = source_registry.create("testing", dataset='anemoi-transform/filters/era_20240601_pressure_level_specific_humidity.grib')

    q_2_r = filter_registry.create("q_2_r")

    output= source | q_2_r
    assert len(list(output))==6 # since we have 2 levels
    output = np.stack([v.to_numpy() for v in list(output) if 'r' in v.metadata('param')]).flatten()

    output_era_r = source_registry.create("testing", dataset="anemoi-transform/filters/era_r.npy").ds.to_numpy().flatten()
    np.testing.assert_allclose(output, output_era_r)
    

def test_presurre_level_relative_humidity_to_specific_humidity_from_file():
    source = source_registry.create("testing", dataset='anemoi-transform/filters/cerra_20240601_pressure_levels.grib')

    r_2_q = filter_registry.create("r_2_q")

    output= source | r_2_q
    assert len(list(output))==6 # since we have 2 levels
    output = np.stack([v.to_numpy() for v in list(output) if 'q' in v.metadata('param')]).flatten()

    output_cerra_q = source_registry.create("testing", dataset="anemoi-transform/filters/cerra_q.npy").ds.to_numpy().flatten()
    np.testing.assert_allclose(output, output_cerra_q)


def test_presurre_level_relative_humidity_to_specific_humidity_from_file_AROME():
    source = source_registry.create("testing", dataset='anemoi-transform/filters/r_t_PAAROME_1S40_ECH0_ISOBARE.grib')

    r_2_q = filter_registry.create("r_2_q")
    q_2_r = filter_registry.create("q_2_r")
    output= source | r_2_q
    assert len(list(output))==6 # since we have 2 levels
    output = np.stack([v.to_numpy() for v in list(output) if 'q' in v.metadata('param')]).flatten()
    output_cerra_q = source_registry.create("testing", dataset="anemoi-transform/filters/arome_specific_humidity.npy").ds.to_numpy().flatten()
    np.testing.assert_allclose(output, output_cerra_q)




if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
