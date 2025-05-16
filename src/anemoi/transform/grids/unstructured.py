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
from typing import Optional
from typing import Tuple
from urllib.parse import urlparse

import numpy as np
from earthkit.data import from_source
from earthkit.data.indexing.fieldlist import FieldArray

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

    def __init__(self, latitudes: np.ndarray, longitudes: np.ndarray, uuidOfHGrid: Optional[str] = None) -> None:
        assert isinstance(latitudes, np.ndarray), type(latitudes)
        assert isinstance(longitudes, np.ndarray), type(longitudes)
        assert len(latitudes) == len(longitudes)

        self.uuidOfHGrid = uuidOfHGrid
        self.latitudes = latitudes
        self.longitudes = longitudes

    def shape(self) -> Tuple[int, ...]:
        """Returns the shape of the latitude array.

        Returns
        -------
        Tuple[int, ...]
            Shape of the latitude array.
        """
        return self.latitudes.shape


def _load(url_or_path: str, param: str) -> Tuple[np.ndarray, str]:
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

    ds = from_source(source, url_or_path)
    ds = ds.sel(param=param)

    assert len(ds) == 1, f"{url_or_path} {param}, expected one field, got {len(ds)}"
    ds = ds[0]

    return ds.to_numpy(flatten=True), ds.metadata("uuidOfHGrid")


class UnstructuredGridField:
    """An unstructured field.

    Parameters
    ----------
    geography : Geography
        Geography object containing latitude and longitude information.
    """

    def __init__(self, geography: Geography) -> None:
        self.geography = geography

    def metadata(self, *args: Any, default: Any = None, **kwargs: Any) -> Any:
        """Retrieves metadata for the field.

        Parameters
        ----------
        *args : Any
            Positional arguments for metadata retrieval.
        default : Any, optional
            Default value if no metadata is found.
        **kwargs : Any
            Keyword arguments for metadata retrieval.

        Returns
        -------
        Any
            Metadata value or default if not found.
        """
        if len(args) == 0 and len(kwargs) == 0:
            return self

        return default

    def grid_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the grid points (latitudes and longitudes).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing arrays of latitudes and longitudes.
        """
        return self.geography.latitudes, self.geography.longitudes

    @property
    def resolution(self) -> str:
        """Resolution of the grid."""
        return "unknown"

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of the grid."""
        return self.geography.shape()

    def to_latlon(self, flatten: bool = False) -> Dict[str, np.ndarray]:
        """Converts the grid to latitude and longitude.

        Parameters
        ----------
        flatten : bool, optional
            Whether to flatten the arrays, by default False.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing latitude and longitude arrays.
        """
        return dict(lat=self.geography.latitudes, lon=self.geography.longitudes)


class UnstructuredGridFieldList(FieldArray):
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

        return cls([UnstructuredGridField(Geography(latitudes, longitudes))])

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

        return cls([UnstructuredGridField(Geography(latitudes, longitudes))])
