# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from functools import lru_cache

import numpy as np

from . import Grid
from . import grid_registry

LOG = logging.getLogger(__name__)


@lru_cache(1)
def icon_grid(path: str, refinement_level_c: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Read the ICON grid from a file.

    Parameters
    ----------
    path : str
        The path to the ICON grid file.
    refinement_level_c : int, optional
        The refinement level to use.

    Returns
    -------
    tuple
        A tuple containing the latitudes and longitudes.
    """
    import xarray as xr

    LOG.info(f"Reading ICON grid from {path}, refinement level {refinement_level_c}")
    ds = xr.open_dataset(path)
    if refinement_level_c is not None:
        idx = ds.refinement_level_c <= refinement_level_c
    else:
        idx = slice(None)
    latitudes = np.rad2deg(ds.clat[idx].values)
    longitudes = np.rad2deg(ds.clon[idx].values)

    LOG.info(f"Latitudes {np.min(latitudes)} {np.max(latitudes)} ({len(latitudes)})")
    LOG.info(f"Longitudes {np.min(longitudes)} {np.max(longitudes)} ({len(longitudes)})")

    LOG.info("Done")
    return latitudes, longitudes


@grid_registry.register("icon")
class IconGrid(Grid):

    def __init__(self, path: str, refinement_level_c: int | None = None):
        self.path = path
        self.refinement_level_c = refinement_level_c

    def latlon(self) -> tuple[np.ndarray, np.ndarray]:
        return icon_grid(self.path, self.refinement_level_c)
