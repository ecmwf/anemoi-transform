
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
        temperature_at_height_level="2t",
        surface_pressure="sp",
        specific_humidity_at_model_levels="q",
        temperature_at_model_levels="t",
        AB=AB_coefficients,
    )

    output = source | q_to_r_height
<<<<<<< HEAD
    assert len(list(output)) == 3  # since we have 2 levels
    output = np.stack([v.to_numpy() for v in list(output)]).flatten()
=======

    # q_to_r_height = filter_registry.create("q_to_r_height",height=2, AB=AB_coefficients)
    # output= list(q_to_r_height.forward_transform(
    #     specific_humidity_at_height_level = source.ds.sel(param='2sh'),
    #     temperature_at_height_level =source.ds.sel(param='2t'),
    #     surface_pressure= source.ds.sel(param='sp'),
    #     specific_humidity_at_model_levels = source.ds.sel(param='q'),
    #     temperature_at_model_levels = source.ds.sel(param='t'),
    # ))
    print((list(output)))
    stop
    # assert len(list(output)) == 6  # since we have 2 levels
    # output = np.stack([v.to_numpy() for v in list(output) if "q" in v.metadata("param")]).flatten()
>>>>>>> b350cd781d005e2bbc735adf0ba5d5fe80d9a9d8

    output_r_height = (
        source_registry.create(
            "testing", dataset="anemoi-transform/filters/single_level_specific_humidity_to_relative_humidity.grib"
        )
        .ds.to_numpy()
        .flatten()
    )
<<<<<<< HEAD
    np.testing.assert_allclose(output, output_r_height)
=======
    print(output_r_height)
    print(output_r_height.metadata("param"))
    # np.testing.assert_allclose(output, output_r_height)

>>>>>>> b350cd781d005e2bbc735adf0ba5d5fe80d9a9d8

def test_specific_humidity_to_relative_humidity():
    pass
def test_relative_humidity_to_specific_humidity_from_file():
    pass


def test_relative_humidity_to_specific_humidity():
    pass


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
