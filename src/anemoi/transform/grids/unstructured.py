# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from typing import Dict
from typing import Tuple
from urllib.parse import urlparse

import numpy as np
from earthkit.data import from_source
from earthkit.data.indexing.fieldlist import FieldArray

LOG = logging.getLogger(__name__)


class Geography:
    """This class retrieve the latitudes and longitudes of unstructured grids,
    and checks if the fields are compatible with the grid.
    """

    def __init__(self, latitudes: np.ndarray, longitudes: np.ndarray, uuidOfHGrid: str = None) -> None:

        assert isinstance(latitudes, np.ndarray), type(latitudes)
        assert isinstance(longitudes, np.ndarray), type(longitudes)

        assert len(latitudes) == len(longitudes)

        self.uuidOfHGrid = uuidOfHGrid
        self.latitudes = latitudes
        self.longitudes = longitudes

    def shape(self) -> Tuple[int, ...]:
        return self.latitudes.shape


def _load(url_or_path: str, param: str) -> Tuple[np.ndarray, str]:
    parsed = urlparse(url_or_path)
    if parsed.scheme:
        source = "url"
    else:
        source = "file"

    ds = from_source(source, url_or_path)
    ds = ds.sel(param=param)

    assert len(ds) == 1, f"{url_or_path} {param}, expected one field, got {len(ds)}"
    ds = ds[0]

    return ds.to_numpy(flatten=True), ds.metadata("uuidOfHGrid")


class UnstructuredGridField:
    """An unstructured field."""

    def __init__(self, geography: Geography) -> None:
        self.geography = geography

    def metadata(self, *args: Any, default: Any = None, **kwargs: Any) -> Any:

        if len(args) == 0 and len(kwargs) == 0:
            return self

        return default

    def grid_points(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.geography.latitudes, self.geography.longitudes

    @property
    def resolution(self) -> str:
        return "unknown"

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.geography.shape()

    def to_latlon(self, flatten: bool = False) -> Dict[str, np.ndarray]:
        return dict(lat=self.geography.latitudes, lon=self.geography.longitudes)


class UnstructuredGridFieldList(FieldArray):
    @classmethod
    def from_grib(
        cls,
        latitudes_url_or_path: str,
        longitudes_url_or_path: str,
        latitudes_param: str = "tlat",
        longitudes_params: str = "tlon",
    ) -> "UnstructuredGridFieldList":
        latitudes, latitudes_uuid = _load(latitudes_url_or_path, latitudes_param)
        longitudes, longitudes_uuid = _load(longitudes_url_or_path, longitudes_params)

        if latitudes_uuid != longitudes_uuid:
            raise ValueError(f"uuidOfHGrid mismatch: lat={latitudes_uuid} != lon={longitudes_uuid}")

        return cls([UnstructuredGridField(Geography(latitudes, longitudes))])

    @classmethod
    def from_values(cls, *, latitudes: Any, longitudes: Any) -> "UnstructuredGridFieldList":
        if isinstance(latitudes, (list, tuple)):
            latitudes = np.array(latitudes)

        if isinstance(longitudes, (list, tuple)):
            longitudes = np.array(longitudes)

        return cls([UnstructuredGridField(Geography(latitudes, longitudes))])
