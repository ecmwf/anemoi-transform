import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.sources import source_registry


def test_pressure_level_relative_humidity_to_specific_humidity():

    source = source_registry.create("testing", dataset="anemoi-transform/filters/cerra_20240601_pressure_levels.grib")
    rename = filter_registry.create(
        "rename",
        rename={
            "param": {"z": "geopotential", "t": "temperature"},
        },
    )


def test_pressure_level_specific_humidity_to_relative_humidity():

    source = source_registry.create("testing", dataset="anemoi-transform/filters/era_20240601_pressure_level_specific_humidity.grib")
    rename = filter_registry.create(
        "rename",
        rename={
            "param": {"z": "geopotential", "t": "temperature"},
        },
    )