from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import filter_registry
from anemoi.transform.sources import source_registry


@skip_if_offline
def test_rename_grib():

    source = source_registry.create("testing", dataset="anemoi-datasets/create/grib-20100101.grib")
    rename = filter_registry.create(
        "rename",
        rename={
            "param": {"z": "geopotential", "t": "temperature"},
        },
    )

    for n in source | rename:
        print(n.metadata("param"), n)
        assert n.metadata("param") in ("geopotential", "temperature")

    rename = filter_registry.create(
        "rename",
        rename={
            "param": "{param}_{levelist}_{levtype}",
        },
    )
    for n in source | rename:
        print(n.metadata("param"), n)

        param, level, levtype = n.metadata("param").split("_")
        assert param in ("z", "t") and level in ("1000", "850", "700", "500", "400", "300") and levtype in ("pl",), (
            param,
            level,
            levtype,
        )

    rename = filter_registry.create(
        "rename",
        rename={
            "param": {"z": "geopotential", "t": "temperature"},
            "levelist": {1000: "1000hPa", 850: "850hPa", 700: "700hPa", 500: "500hPa", 400: "400hPa", 300: "300hPa"},
        },
    )
    for n in source | rename:
        print(n.metadata("param", "levelist"), n)

        param, level = n.metadata("param", "levelist")

        assert param in ("geopotential", "temperature") and level in (
            "1000hPa",
            "850hPa",
            "700hPa",
            "500hPa",
            "400hPa",
            "300hPa",
        ), (param, level)


@skip_if_offline
def test_rename_netcdf():
    from anemoi.transform.filters import filter_registry
    from anemoi.transform.sources import source_registry

    source = source_registry.create("testing", dataset="anemoi-datasets/create/netcdf.nc")
    rename = filter_registry.create(
        "rename",
        rename={
            "param": {"t2m": "2m temprature", "msl": "mean sea level pressure"},
        },
    )

    for n in source | rename:
        print(n.metadata("param"), n)
        assert n.metadata("param") in ("2m temprature", "mean sea level pressure")


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
