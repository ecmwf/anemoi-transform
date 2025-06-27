# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Any

import earthkit.data as ekd
import pytest

pytest_plugins = ["anemoi.utils.testing"]


@pytest.fixture
def fieldlist_fixture(get_test_data: callable, name: str = "2t-sp.grib") -> Any:
    def _fieldlist_fixture() -> Any:
        """Fixture to create a fieldlist for testing.

        Parameters
        ----------
        name : str, optional
            The name of the fieldlist to create, by default "2t-sp.grib".

        Returns
        -------
        Any
            The created fieldlist.
        """
        return ekd.from_source("file", get_test_data(f"anemoi-filters/{name}"))

    return _fieldlist_fixture
