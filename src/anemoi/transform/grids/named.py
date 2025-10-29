# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Utilities for working with grids."""

import logging
import os
from io import BytesIO

import numpy as np
import requests
from anemoi.utils.caching import cached

LOG = logging.getLogger(__name__)


GRIDS_URL_PATTERN = "https://get.ecmwf.int/repository/anemoi/grids/grid-{name}.npz"


@cached(collection="grids", encoding="npz")
def _grids(name: str | list[float] | tuple[float, ...]) -> bytes:
    """Get grid data by name.

    Parameters
    ----------
    name : str
        The name of the grid

    Returns
    -------
    bytes
        The grid data
    """
    from anemoi.utils.config import load_config

    if isinstance(name, (tuple, list)):
        assert len(name) == 2, "Grid name must be a list or a tuple of length 2"
        assert all(isinstance(i, (int, float)) for i in name), "Grid name must be a list or a tuple of numbers"
        if name[0] == name[1]:
            name = str(float(name[0]))
        else:
            name = str(float(name[0])) + "x" + str(float(name[1]))
        name = name.replace(".", "p")

    user_path = load_config().get("utils", {}).get("grids_path")
    if user_path:
        path = os.path.expanduser(os.path.join(user_path, f"grid-{name}.npz"))
        if os.path.exists(path):
            LOG.warning("Loading grids from custom user path %s", path)
            with open(path, "rb") as f:
                return f.read()
        else:
            LOG.warning("Custom user path %s does not exist", path)

    # To add a grid
    # anemoi-transform get-grid --source mars grid=o400,levtype=sfc,param=2t grid-o400.npz
    # nexus-cli -u xxxx -p yyyy -s GET_INSTANCE --repository anemoi upload --remote-path grids --local-path grid-o400.npz

    url = GRIDS_URL_PATTERN.format(name=name.lower())
    LOG.warning("Downloading grids from %s", url)
    response = requests.get(url)
    response.raise_for_status()
    return response.content


def lookup(name: str | list[float] | tuple[float, ...]) -> dict:
    """Load grid data by name.

    Parameters
    ----------
    name : str
        The name of the grid

    Returns
    -------
    dict
        The grid data
    """
    if isinstance(name, str) and name.endswith(".npz"):
        return dict(np.load(name))

    data = _grids(name)
    npz = np.load(BytesIO(data))
    return dict(npz)
