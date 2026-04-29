# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from collections.abc import Callable

import earthkit.data as ekd
import numpy as np
import pytest
from anemoi.utils.testing import GetTestData
from earthkit.data.indexing.fieldlist import SimpleFieldList
from earthkit.data.sources.array_list import ArrayField
from earthkit.data.utils.metadata.dict import UserMetadata

from anemoi.transform.source import Source
from anemoi.transform.sources import source_registry

pytest_plugins = ["anemoi.utils.testing"]

# Create a ekd Metadata Class that mocks the mars metadata namespace
MARS_KEYS = {"param", "levelist", "type", "step", "date", "time", "number", "expver", "class", "stream", "domain"}


class MarsUserMetadata(UserMetadata):
    def namespaces(self):
        return ["mars"]

    def as_namespace(self, namespace=None):
        if namespace == "mars":
            return {k: v for k, v in self._data.items() if k in MARS_KEYS}
        return {}


@source_registry.register("testing")
class TestingSource(Source):
    def __init__(self, *, dataset: str | list[dict]) -> None:
        assert dataset is not None, "Dataset cannot be None"
        self.ds = dataset

    def forward(self, *args, **kwargs):
        return self.ds


@pytest.fixture
def fieldlist(get_test_data: GetTestData) -> ekd.FieldList:
    """Fixture to create a fieldlist for testing."""
    return ekd.from_source("file", get_test_data("anemoi-filters/2t-sp.grib"))


@pytest.fixture
def test_source(get_test_data: GetTestData) -> Callable[[str | list[dict]], Source]:
    def _source(dataset: str | list[dict]) -> Source:
        """Create a source from a known file or a list of dicts for testing."""
        if isinstance(dataset, str):
            ds = ekd.from_source("file", get_test_data(dataset))
        elif isinstance(dataset, list):
            ds = ekd.from_source("list-of-dicts", dataset)
        else:
            raise ValueError("dataset must be a string or a list of dicts")
        return source_registry.create("testing", dataset=ds)

    return _source


@pytest.fixture
def mars_test_source() -> Callable[[list[dict]], Source]:
    def _source(dataset: list[dict]) -> Source:
        fields = []
        for d in dataset:
            v = np.array(d["values"])
            fields.append(ArrayField(v, MarsUserMetadata(d, shape=v.shape)))
        return source_registry.create("testing", dataset=SimpleFieldList(fields=fields))

    return _source
