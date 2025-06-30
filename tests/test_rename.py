# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import filter_registry


@pytest.fixture
def grib_source(test_source):
    return test_source("anemoi-datasets/create/grib-20100101.grib")


@pytest.fixture
def netcdf_source(test_source):
    return test_source("anemoi-datasets/create/netcdf.nc")


@skip_if_offline
def test_rename_grib_dict_rename(grib_source):
    rename = filter_registry.create(
        "rename",
        rename={
            "param": {"z": "geopotential", "t": "temperature"},
        },
    )
    pipeline = grib_source | rename

    for original, result in zip(grib_source, pipeline):
        if original.metadata("param") == "z":
            assert result.metadata("param") == "geopotential"
        elif original.metadata("param") == "t":
            assert result.metadata("param") == "temperature"
        else:
            raise RuntimeError(f"Unexpected param: {original.metadata('param')}")


@skip_if_offline
def test_rename_grib_format_rename(grib_source):
    rename = filter_registry.create(
        "rename",
        rename={
            "param": "{param}_{levelist}_{levtype}",
        },
    )
    pipeline = grib_source | rename

    for original, result in zip(grib_source, pipeline):
        orig_param, orig_level, orig_levtype = original.metadata("param", "levelist", "levtype")
        assert result.metadata("param") == f"{orig_param}_{orig_level}_{orig_levtype}"


def test_rename_grib_dict_multiple(grib_source):
    rename = filter_registry.create(
        "rename",
        rename={
            "param": {"z": "geopotential", "t": "temperature"},
            "levelist": {1000: "1000hPa", 850: "850hPa", 700: "700hPa", 500: "500hPa", 400: "400hPa", 300: "300hPa"},
        },
    )
    pipeline = grib_source | rename

    for original, result in zip(grib_source, pipeline):
        assert result.metadata("levelist") == f"{original.metadata('levelist')}hPa"
        if original.metadata("param") == "z":
            assert result.metadata("param") == "geopotential"
        elif original.metadata("param") == "t":
            assert result.metadata("param") == "temperature"
        else:
            raise RuntimeError(f"Unexpected param: {original.metadata('param')}")


@skip_if_offline
def test_rename_netcdf(netcdf_source):
    rename = filter_registry.create(
        "rename",
        rename={
            "param": {"t2m": "2m temperature", "msl": "mean sea level pressure"},
        },
    )
    pipeline = netcdf_source | rename

    for original, result in zip(netcdf_source, pipeline):
        if original.metadata("param") == "t2m":
            assert result.metadata("param") == "2m temperature"
        elif original.metadata("param") == "msl":
            assert result.metadata("param") == "mean sea level pressure"
        else:
            raise RuntimeError(f"Unexpected param: {original.metadata('param')}")


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
