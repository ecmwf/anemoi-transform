# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Union

import earthkit.data as ekd
import pytest

from anemoi.transform.source import Source
from anemoi.transform.sources import source_registry

pytest_plugins = ["anemoi.utils.testing"]


@source_registry.register("testing")
class TestingSource(Source):
    def __init__(self, *, dataset: Union[str, list[dict]]) -> None:
        assert dataset is not None, "Dataset cannot be None"
        self.ds = dataset

    def forward(self, *args, **kwargs):
        return self.ds


@pytest.fixture
def fieldlist(get_test_data: callable) -> ekd.FieldList:
    """Fixture to create a fieldlist for testing."""
    return ekd.from_source("file", get_test_data("anemoi-filters/2t-sp.grib"))


@pytest.fixture
def test_source(get_test_data: callable) -> callable:
    def _source(dataset: Union[str, list[dict]]) -> Source:
        """Create a source from a known file or a list of dicts for testing."""
        if isinstance(dataset, str):
            ds = ekd.from_source("file", get_test_data(dataset))
        elif isinstance(dataset, list):
            ds = ekd.from_source("list-of-dicts", dataset)
        else:
            raise ValueError("dataset must be a string or a list of dicts")
        return source_registry.create("testing", dataset=ds)

    return _source
