# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Any
from urllib.parse import urlparse

import numpy as np
from earthkit.data import SimpleFieldList
from earthkit.data import from_source

LOG = logging.getLogger(__name__)


class Geography:
    """This class retrieves the latitudes and longitudes of unstructured grids,
    and checks if the fields are compatible with the grid.

    Parameters
    ----------
    latitudes : np.ndarray
        Array of latitude values.
    longitudes : np.ndarray
        Array of longitude values.
    uuidOfHGrid : str, optional
        UUID of the horizontal grid.
    """

    def __init__(self, latitudes: np.ndarray, longitudes: np.ndarray, uuidOfHGrid: str | None = None) -> None:
        assert isinstance(latitudes, np.ndarray), type(latitudes)
        assert isinstance(longitudes, np.ndarray), type(longitudes)
        assert len(latitudes) == len(longitudes)

        self.uuidOfHGrid = uuidOfHGrid
        self.latitudes = latitudes
        self.longitudes = longitudes

    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the latitude array.

        Returns
        -------
        Tuple[int, ...]
            Shape of the latitude array.
        """
        return self.latitudes.shape


def _load(url_or_path: str, param: str) -> tuple[np.ndarray, str]:
    """Loads data from a given URL or file path.

    Parameters
    ----------
    url_or_path : str
        URL or file path to load data from.
    param : str
        Parameter to select from the data source.

    Returns
    -------
    Tuple[np.ndarray, str]
        Tuple containing the data as a flattened numpy array and the UUID of the horizontal grid.
    """
    parsed = urlparse(url_or_path)
    if parsed.scheme:
        source = "url"
    else:
        source = "file"

    ds = from_source(source, url_or_path).to_fieldlist()
    ds = ds.sel(**{"parameter.variable": param})

    assert len(ds) == 1, f"{url_or_path} {param}, expected one field, got {len(ds)}"
    ds = ds[0]

    return ds.to_numpy(flatten=True), ds.metadata("uuidOfHGrid")


def _new_grid_field(geography: "Geography") -> Any:
    """Create an earthkit-data field carrying only an unstructured grid.

    Parameters
    ----------
    geography : Geography
        Geography object containing latitude and longitude information.

    Returns
    -------
    Any
        A field defined on the given grid, with zero values.
    """
    # The grid is unstructured: latitudes and longitudes are parallel arrays holding
    # one coordinate pair per grid point, so the number of values is the number of
    # points, not the product of the two.
    return from_source(
        "list-of-dicts",
        [
            {
                "geography": {
                    "latitudes": geography.latitudes,
                    "longitudes": geography.longitudes,
                },
                "data": {"values": np.zeros(geography.latitudes.size)},
            }
        ],
    ).to_fieldlist()[0]


class UnstructuredGridFieldList(SimpleFieldList):
    """List of unstructured grid fields."""

    @classmethod
    def from_grib(
        cls,
        latitudes_url_or_path: str,
        longitudes_url_or_path: str,
        latitudes_param: str = "tlat",
        longitudes_params: str = "tlon",
    ) -> "UnstructuredGridFieldList":
        """Create an UnstructuredGridFieldList from GRIB files.

        Parameters
        ----------
        latitudes_url_or_path : str
            URL or file path for the latitudes data.
        longitudes_url_or_path : str
            URL or file path for the longitudes data.
        latitudes_param : str, optional
            Parameter name for latitudes, by default "tlat".
        longitudes_params : str, optional
            Parameter name for longitudes, by default "tlon".

        Returns
        -------
        UnstructuredGridFieldList
            The created UnstructuredGridFieldList.
        """
        latitudes, latitudes_uuid = _load(latitudes_url_or_path, latitudes_param)
        longitudes, longitudes_uuid = _load(longitudes_url_or_path, longitudes_params)

        if latitudes_uuid != longitudes_uuid:
            raise ValueError(f"uuidOfHGrid mismatch: lat={latitudes_uuid} != lon={longitudes_uuid}")

        return cls([_new_grid_field(Geography(latitudes, longitudes))])

    @classmethod
    def from_values(cls, *, latitudes: Any, longitudes: Any) -> "UnstructuredGridFieldList":
        """Create an UnstructuredGridFieldList from latitude and longitude values.

        Parameters
        ----------
        latitudes : Any
            Latitude values.
        longitudes : Any
            Longitude values.

        Returns
        -------
        UnstructuredGridFieldList
            The created UnstructuredGridFieldList.
        """
        if isinstance(latitudes, (list, tuple)):
            latitudes = np.array(latitudes)

        if isinstance(longitudes, (list, tuple)):
            longitudes = np.array(longitudes)

        return cls([_new_grid_field(Geography(latitudes, longitudes))])
