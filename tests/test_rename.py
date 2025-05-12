from anemoi.utils.testing import skip_if_offline


@skip_if_offline
def test_rename_grib():
    from anemoi.transform.filters import filter_registry
    from anemoi.transform.sources import source_registry

    source = source_registry.create("testing", "anemoi-datasets/create/grib-20100101.grib")
    rename = filter_registry.create("rename", rename={"z": "geopotential", "t": "temperature"})

    for n in source | rename:
        print(n.metadata("param"), n)
        assert n.metadata("param") in ("geopotential", "temperature")


@skip_if_offline
def test_rename_netcdf():
    from anemoi.transform.filters import filter_registry
    from anemoi.transform.sources import source_registry

    source = source_registry.create("testing", "anemoi-datasets/create/netcdf.nc")
    rename = filter_registry.create("rename", rename={"t2m": "2m temprature", "msl": "mean sea level pressure"})

    for n in source | rename:
        print(n.metadata("param"), n)
        assert n.metadata("param") in ("2m temprature", "mean sea level pressure")


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
