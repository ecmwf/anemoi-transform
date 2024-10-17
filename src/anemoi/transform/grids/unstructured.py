# (C) Copyright 2024 ECMWF.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
#


import logging
from urllib.parse import urlparse

import numpy as np
from earthkit.data import from_source
from earthkit.data.indexing.fieldlist import FieldArray

LOG = logging.getLogger(__name__)


class Geography:
    """This class retrieve the latitudes and longitudes of unstructured grids,
    and checks if the fields are compatible with the grid.
    """

    def __init__(self, latitudes, longitudes, uuidOfHGrid=None):

        assert isinstance(latitudes, np.ndarray)
        assert isinstance(longitudes, np.ndarray)

        LOG.info(f"Latitudes: {len(latitudes)}, Longitudes: {len(longitudes)}")
        assert len(latitudes) == len(longitudes)

        self.uuidOfHGrid = uuidOfHGrid
        self.latitudes = latitudes
        self.longitudes = longitudes


def _load(url_or_path, param):
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

    def __init__(self, geography):
        self._latitudes = geography.latitudes
        self._longitudes = geography.longitudes

    def metadata(self, name, default=None):
        if name == "uuidOfHGrid":
            return self.geography.uuidOfHGrid
        return default

    def grid_points(self):
        return self._latitudes, self._longitudes

    @property
    def resolution(self):
        return "unknown"

    @property
    def shape(self):
        return (len(self._latitudes),)


class UnstructuredGridFieldList(FieldArray):
    @classmethod
    def from_grib(cls, latitudes_url_or_path, longitudes_url_or_path, latitudes_param="tlat", longitudes_params="tlon"):
        latitudes, latitudes_uuid = _load(latitudes_url_or_path, latitudes_param)
        longitudes, longitudes_uuid = _load(longitudes_url_or_path, longitudes_params)

        if latitudes_uuid != longitudes_uuid:
            raise ValueError(f"uuidOfHGrid mismatch: lat={latitudes_uuid} != lon={longitudes_uuid}")

        return cls([UnstructuredGridField(Geography(latitudes, longitudes))])

    @classmethod
    def from_values(cls, latitudes, longitudes):
        if isinstance(latitudes, (list, tuple)):
            latitudes = np.array(latitudes)

        if isinstance(longitudes, (list, tuple)):
            longitudes = np.array(longitudes)

        return cls([UnstructuredGridField(Geography(latitudes, longitudes))])
