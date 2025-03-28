# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


"""Grids"""

from abc import abstractmethod
from typing import Any

import numpy as np
from anemoi.utils.registry import Registry

from anemoi.transform.grids.unstructured import UnstructuredGridFieldList

grid_registry = Registry(__name__)


class Grid:
    """Base class for all grids."""

    @abstractmethod
    def latlon(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the latitudes and longitudes of the grid.

        Returns
        -------
        tuple
            A tuple containing the latitudes and longitudes.
        """
        pass


def create_grid(context: Any, config: Any) -> Grid:
    """Create a grid definition from the given context and configuration.

    Parameters
    ----------
    context : Any
        The context in which the grid is created.
    config : Any
        The configuration for the grid.

    Returns
    -------
    Grid
        The created grid.
    """
    grid = grid_registry.from_config(config)
    grid.context = context
    return grid


__all__ = ["UnstructuredGridFieldList"]
