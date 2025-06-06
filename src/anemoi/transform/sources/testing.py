# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
from anemoi.utils.testing import get_test_data

from anemoi.transform.source import Source
from anemoi.transform.sources import source_registry


@source_registry.register("testing")
class Testing(Source):
    """A demo source for use in testing.

    Note: Either dataset or fields must be provided.

    Parameters
    ----------
    dataset : str, optional
        Dataset name to load from the test data bucket.
    fields : list[dict], optional
        List of dictionaries containing the field specifications (e.g. keys of param, level, values).
    """

    def __init__(self, *, dataset: str | None = None, fields: list[dict] | None = None) -> None:
        if dataset is None and fields is None:
            raise ValueError("Either dataset or fields must be provided")

        if dataset is not None:
            self.ds = ekd.from_source("file", get_test_data(dataset))

        if fields is not None:
            assert isinstance(fields, list), "Fields should be a list"
            self.ds = ekd.from_source("list-of-dicts", fields)

        assert self.ds is not None, "Dataset not found"

    def forward(self, *args, **kwargs) -> ekd.Source:
        return self.ds
