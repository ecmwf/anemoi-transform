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

import earthkit.data as ekd
import tqdm
from earthkit.data.core.fieldlist import Field

from anemoi.transform.fields import new_field_from_latitudes_longitudes
from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry

LOG = logging.getLogger(__name__)


def as_gridspec(grid: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert grid specification to a dictionary format.

    Parameters
    ----------
    grid : Optional[Union[str, Dict[str, Any]]]
        The grid specification.

    Returns
    -------
    Optional[Dict[str, Any]]
        The grid specification as a dictionary.
    """
    if grid is None:
        return None

    if isinstance(grid, (str, list, tuple)):
        return {"grid": grid}

    return grid


def as_griddata(grid: str | Field | dict[str, Any] | None) -> dict[str, Any] | None:
    """Convert grid data to a dictionary format.

    Parameters
    ----------
    grid : Optional[Union[str, Field, Dict[str, Any]]]
        The grid data.

    Returns
    -------
    Optional[Dict[str, Any]]
        The grid data as a dictionary.
    """
    if grid is None:
        return None

    if isinstance(grid, Field):
        lat, lon = grid.grid_points()
        return dict(latitudes=lat, longitudes=lon)

    if isinstance(grid, dict) and "latitudes" in grid and "longitudes" in grid:
        return grid

    if isinstance(grid, str):
        from anemoi.utils.grids import grids

        return grids(grid)

    if isinstance(grid, (list, tuple)):
        from anemoi.utils.grids import grids

        return grids(grid)

    raise ValueError(f"Invalid grid: {grid}")


@filter_registry.register("regrid")
class RegridFilter(Filter):
    """A filter to regrid fields using earthkit-regrid.

    When building a dataset for a specific model, it is possible that the
    source grid or resolution does not fit the needs. In that case, it is
    possible to add a filter to interpolate the data to a target grid. It
    will call the ``interpolate`` function from `earthkit-regrid
    <https://earthkit-regrid.readthedocs.io/en/latest/interpolate.html>`_ if
    the keys ``method``, ``in_grid`` and ``out_grid`` are provided and if a
    `pre-generated matrix
    <https://earthkit-regrid.readthedocs.io/en/latest/inventory/index.html>`_
    exists for this transformation. Otherwise, it is possible to provide a
    ``regrid matrix`` previously generated with :ref:`make-regrid-matrix`.
    The generated matrix is an NPZ file containing the
    input/output coordinates, the indices, and the weights of the
    interpolation.

    ``regrid`` filter must follow a source or another filter in a
    `building-pipe
    <https://anemoi.readthedocs.io/projects/datasets/en/latest/datasets/building/operations.html#pipe>`_
    operation.

    Examples
    --------

    .. code-block:: yaml

      input:
        pipe:
        - source:
            # mars, grib, netcdf, etc.
            # source attributes here
            # ...

        - regrid:
            method: nearest
            in_grid: o32
            out_grid: o48

    .. code-block:: yaml

      input:
        pipe:
        - source:
            # mars, grib, netcdf, etc.
            # source attributes here
            # ...

        - regrid:
            matrix: /path/to/regrid/matrix.npz

    """

    def __init__(
        self,
        *,
        in_grid: Any | None = None,
        out_grid: Any | None = None,
        method: str | None = None,
        matrix: str | None = None,
        check: bool = False,
        interpolator: Any | None = None,
    ) -> None:
        """Parameters
        -------------
        in_grid : Optional[Any]
            The input grid specification.
        out_grid : Optional[Any]
            The output grid specification.
        method : Optional[str]
            The interpolation method.
        matrix : Optional[str]
            The regrid matrix file path.
        check : bool
            Whether to perform checks.
        interpolator : Optional[Any]
            The interpolator to use.
        """

        self.in_grid = in_grid
        self.out_grid = out_grid
        self.method = method
        self.interpolator = make_interpolator(in_grid, out_grid, method, matrix, check, interpolator)

    def forward(self, data: ekd.FieldList) -> ekd.FieldList:
        """Apply the forward regridding transformation.

        Parameters
        ----------
        data : ekd.FieldList
            The input data to be transformed.

        Returns
        -------
        ekd.FieldList
            The transformed data.
        """
        return self._interpolate(data)

    def _interpolate(self, data: Any) -> Any:
        """Interpolate the data from one grid to another.

        Parameters
        ----------
        data : Any
            The input data to be interpolated.

        Returns
        -------
        Any
            The interpolated data.
        """
        result = []
        for field in tqdm.tqdm(data, desc="Regridding"):
            result.append(self.interpolator(field))

        return new_fieldlist_from_list(result)


class EarthkitRegrid:
    """Default interpolator using earthkit."""

    def __init__(self, *, in_grid: Any, out_grid: Any, method: str, matrix: str, check: bool) -> None:
        """Parameters
        -------------
        in_grid : Any
            The input grid specification.
        out_grid : Any
            The output grid specification.
        method : str
            The interpolation method.
        matrix : str
            The regrid matrix file path.
        check : bool
            Whether to perform checks.
        """
        self.in_grid = as_gridspec(in_grid)
        self.out_grid = as_gridspec(out_grid)
        self.out_griddata = as_griddata(out_grid)
        self.method = method
        if check:
            LOG.warning("Check is not supported by EarthkitRegrid")

    def __call__(self, field: Any) -> Any:
        """Interpolate the field data.

        Parameters
        ----------
        field : Any
            The field to be interpolated.

        Returns
        -------
        Any
            The interpolated field.
        """
        from earthkit.regrid import interpolate

        return new_field_from_latitudes_longitudes(
            new_field_from_numpy(
                interpolate(
                    field.to_numpy(flatten=True),
                    in_grid=self.in_grid,
                    out_grid=self.out_grid,
                    method=self.method,
                ),
                template=field,
            ),
            **self.out_griddata,
        )


class MIRMatrix:
    """Assume matrix was created by `anemoi-transform make-regrid-matrix`."""

    def __init__(self, *, in_grid: Any, out_grid: Any, method: str, matrix: str, check: bool) -> None:
        """Parameters
        -------------
        in_grid : Any
            The input grid specification.
        out_grid : Any
            The output grid specification.
        method : str
            The interpolation method.
        matrix : str
            The regrid matrix file path.
        check : bool
            Whether to perform checks.
        """
        import numpy as np
        from scipy.sparse import csr_array

        self.check = check

        if in_grid is not None:
            raise ValueError("in_grid is not supported by MIRMatrix")

        if out_grid is not None:
            raise ValueError("out_grid is not supported by MIRMatrix")

        if method is not None:
            raise ValueError("method is not supported by MIRMatrix")

        # Assume matrix was created by `anemoi-transform make-regrid-matrix`

        loaded = dict(np.load(matrix))

        self.matrix: csr_array = csr_array(
            (loaded["matrix_data"], loaded["matrix_indices"], loaded["matrix_indptr"]), shape=loaded["matrix_shape"]
        )

        self.in_grid: dict[str, np.ndarray] = dict(latitudes=loaded["in_latitudes"], longitudes=loaded["in_longitudes"])
        self.out_grid: dict[str, np.ndarray] = dict(
            latitudes=loaded["out_latitudes"], longitudes=loaded["out_longitudes"]
        )

    def __call__(self, field: Any) -> Any:
        """Interpolate the field data using the regrid matrix.

        Parameters
        ----------
        field : Any
            The field to be interpolated.

        Returns
        -------
        Any
            The interpolated field.
        """
        if self.check:
            # TODO: Check that the field is on the same grid as the in_grid
            pass

        data = field.to_numpy(flatten=True)
        data = self.matrix @ data

        return new_field_from_latitudes_longitudes(new_field_from_numpy(data, template=field), **self.out_grid)


class ScipyKDTreeNearestNeighbours:
    """Interpolator tools for the grids that are not supported yet by earthkit."""

    nearest_grid_points = None

    def __init__(
        self, *, in_grid: Any, out_grid: Any, method: str, matrix: str | None = None, check: bool = False
    ) -> None:
        """Parameters
        -------------
        in_grid : Any
            The input grid specification.
        out_grid : Any
            The output grid specification.
        method : str
            The interpolation method.
        matrix : str, optional
            The regrid matrix file path.
        check : bool
            Whether to perform checks.
        """
        if method != "nearest":
            raise NotImplementedError(f"ScipyKDTreeNearestNeighbours does not support {method}, only 'nearest'")

        self.in_grid = as_griddata(in_grid)
        self.out_grid = as_griddata(out_grid)

        if self.out_grid is None:
            raise ValueError("out_grid is required, but not provided")

        if check:
            LOG.warning("Check is not supported by ScipyKDTreeNearestNeighbours")

    def __call__(self, field: Any) -> Any:
        """Interpolate the field data using nearest neighbours.

        Parameters
        ----------
        field : Any
            The field to be interpolated.

        Returns
        -------
        Any
            The interpolated field.
        """
        if self.in_grid is None:
            self.in_grid = as_griddata(field)
            assert self.in_grid is not None, field

        if self.nearest_grid_points is None:
            from anemoi.utils.grids import nearest_grid_points

            if self.out_grid is None:
                raise ValueError("out_grid is required, but not provided")

            self.nearest_grid_points = nearest_grid_points(
                self.in_grid["latitudes"],
                self.in_grid["longitudes"],
                self.out_grid["latitudes"],
                self.out_grid["longitudes"],
            )

        data = field.to_numpy(flatten=True)
        assert data.shape == self.in_grid["latitudes"].shape, (data.shape, self.in_grid["latitudes"].shape)
        assert data.shape == self.in_grid["longitudes"].shape, (data.shape, self.in_grid["longitudes"].shape)

        data = data[..., self.nearest_grid_points]
        return new_field_from_latitudes_longitudes(new_field_from_numpy(data, template=field), **self.out_grid)


def _interpolator(
    in_grid: Any,
    out_grid: Any,
    method: str | None = None,
    matrix: str | None = None,
    check: bool = False,
    interpolator: Any | None = None,
) -> str:
    """Determine the interpolator to use.

    Parameters
    ----------
    in_grid : Any
        The input grid specification.
    out_grid : Any
        The output grid specification.
    method : str, optional
        The interpolation method.
    matrix : str, optional
        The regrid matrix file path.
    check : bool, optional
        Whether to perform checks.
    interpolator : Any, optional
        The interpolator to use.

    Returns
    -------
    str
        The interpolator to use.
    """
    if interpolator is not None:
        return interpolator

    if matrix is not None:
        return "MIRMatrix"

    if method == "nearest":
        return "ScipyKDTreeNearestNeighbours"

    return "EarthkitRegrid"


def make_interpolator(
    in_grid: Any,
    out_grid: Any,
    method: str | None = None,
    matrix: str | None = None,
    check: bool = False,
    interpolator: Any | None = None,
) -> Any:
    """Create an interpolator.

    Parameters
    ----------
    in_grid : Any
        The input grid specification.
    out_grid : Any
        The output grid specification.
    method : str, optional
        The interpolation method.
    matrix : str, optional
        The regrid matrix file path.
    check : bool, optional
        Whether to perform checks.
    interpolator : Any, optional
        The interpolator to use.

    Returns
    -------
    Any
        The created interpolator.
    """
    interpolator = _interpolator(in_grid, out_grid, method, matrix, check, interpolator)

    return globals()[interpolator](in_grid=in_grid, out_grid=out_grid, method=method, matrix=matrix, check=check)
