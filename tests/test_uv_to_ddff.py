import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.sources import source_registry

prototype = {
    "latitudes": [10.0, 0.0, -10.0],
    "longitudes": [20, 40.0],
    "valid_datetime": "2018-08-01T09:00:00Z",
}

winds = [
    {"param": "u", "levelist": 500, "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **prototype},
    {"param": "v", "levelist": 500, "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **prototype},
    {"param": "u", "levelist": 850, "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **prototype},
    {"param": "v", "levelist": 850, "values": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], **prototype},
]


def test_uv_to_ddff():

    source = source_registry.create("testing", fields=winds)
    uv_to_ddff = filter_registry.create("uv_to_ddff")

    for n in source | uv_to_ddff:
        print(n)


def test_uv_to_ddff_and_back():

    source = source_registry.create("testing", fields=winds)

    uv_to_ddff = filter_registry.create("uv_to_ddff")
    ddff_to_uv = filter_registry.create("ddff_to_uv")

    noop = filter_registry.create("noop")

    for original, converted in zip(source | noop, source | uv_to_ddff | ddff_to_uv):
        assert np.allclose(original.to_numpy(), converted.to_numpy()), (
            (original.metadata("param"), original.metadata("levelist")),
            (converted.metadata("param"), converted.metadata("levelist")),
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
