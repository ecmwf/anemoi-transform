# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Grid identity based on lat/lon coordinates.

This module provides :class:`LatLonGrid`, a value object wrapping
materialized coordinate arrays for grid compatibility checks. It is
deliberately lightweight: numpy-only at module level, with scipy
imported lazily, and no earthkit dependency. Therefore it can be used
both on earthkit fields (via :meth:`LatLonGrid.from_field`) and on
plain arrays (via :meth:`LatLonGrid.from_latlon`).
"""

import enum
import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

from anemoi.transform.grids import Grid

LOG = logging.getLogger(__name__)

# Default tolerance for point-set matching, in unit-sphere xyz distance.
# ~1e-6 corresponds to ~6 m on the Earth's surface; well below any real
# grid spacing but above float32/float64 round-trip noise.
DEFAULT_MAX_DISTANCE = 1e-6

# Canonical ordering: latitude descending (N->S), longitude ascending
# in [0, 360), row-major — the ordering a regular_ll GRIB scan produces.
CANONICAL_DECIMALS = 6  # rounding precision for sort keys


def _canonical_key(latitudes, longitudes):
    """Sort keys: round to fixed precision, normalize lon to [0, 360)."""
    lat_key = np.round(np.asarray(latitudes, dtype=np.float64), CANONICAL_DECIMALS)
    lon_key = np.round(np.mod(np.asarray(longitudes, dtype=np.float64), 360.0), CANONICAL_DECIMALS)
    return lat_key, lon_key


class GridMismatchError(ValueError):
    """Raised when two grids mismatch cannot be reconciled with auto-
    reodering.

    Parameters
    ----------
    message : str
        Human-readable description.
    num_unmatched : int, optional
        Number of target points with no distinct source point
        (unmatched or duplicated in the nearest-neighbour mapping).
    max_distance : float, optional
        Largest nearest-neighbour distance observed (unit-sphere xyz).
    """

    def __init__(self, message: str, *, num_unmatched: int = 0, max_distance: float | None = None) -> None:
        super().__init__(message)
        self.num_unmatched = num_unmatched
        self.max_distance = max_distance


class GridComparison(enum.Enum):
    """Result of comparing two grids."""

    IDENTICAL_ORDERING = "identical_ordering"
    REORDERABLE = "reorderable"
    DIFFERENT = "different"


# Should not be registered with @grid_registry.registe like other Grid subclasses
# because LatLonGrid isn't constructible from config since it inits from latlon arrays.
class LatLonGrid(Grid):
    """A grid identity defined by its lat/lon coordinates.

    Wraps materialized coordinate arrays and provides ordering-aware
    and ordering-agnostic comparison. Comparison is done in Cartesian
    (xyz) space, so it is convention-agnostic: longitudes in [0, 360)
    and [-180, 180) denote the same points.

    Parameters
    ----------
    latitudes : np.ndarray
        Latitude values, one per grid point.
    longitudes : np.ndarray
        Longitude values, one per grid point, in any convention.
    metadata : dict, optional
        Optional identity hints (e.g. ``uuidOfHGrid``, ``gridType``).
        Used only to short-circuit comparison and enrich error
        messages, never as the equality criterion.
    """

    def __init__(
        self,
        latitudes: NDArray[Any],
        longitudes: NDArray[Any],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        latitudes = np.asarray(latitudes, dtype=np.float64).ravel()
        longitudes = np.asarray(longitudes, dtype=np.float64).ravel()
        if latitudes.shape != longitudes.shape:
            raise ValueError(f"latitudes/longitudes shape mismatch: {latitudes.shape} != {longitudes.shape}")
        self.latitudes = latitudes
        self.longitudes = longitudes
        self.metadata = dict(metadata or {})
        self._xyz: NDArray[Any] | None = None

    @classmethod
    def from_latlon(
        cls,
        latitudes: NDArray[Any],
        longitudes: NDArray[Any],
        **metadata: Any,
    ) -> "LatLonGrid":
        """Build a grid from coordinate arrays.

        Parameters
        ----------
        latitudes : np.ndarray
            Latitude values.
        longitudes : np.ndarray
            Longitude values.
        **metadata
            Optional identity hints stored in ``.metadata``.

        Returns
        -------
        LatLonGrid
        """
        return cls(latitudes, longitudes, metadata=metadata)

    @classmethod
    def from_field(cls, field: Any) -> "LatLonGrid":
        """Build a grid from a field-like object.

        Accepted inputs, tried in order:

        - Objects exposing ``grid_points()``: earthkit ``Field``'s and
          anemoi-transform wrappers (``WrappedField`` delegates via
          ``__getattr__``).
        - Dicts (or dict-like objects) carrying ``latitudes`` /
          ``longitudes`` keys, e.g. what ``ekd.from_source(
          "list-of-dicts", ...)`` yields on iteration.
        - ``WrappedField`` chains: ``_field`` is followed until one of
          the above matches.

        Parameters
        ----------
        field : Any
            The field-like object to extract coordinates from.

        Returns
        -------
        LatLonGrid

        Raises
        ------
        TypeError
            If no coordinates can be extracted — the message names the
            offending type rather than leaking an ``AttributeError``
            or ``KeyError`` from internals.
        """
        seen = field
        latitudes = longitudes = None
        for _ in range(32):  # bound against _field cycles
            if hasattr(seen, "grid_points"):
                latitudes, longitudes = seen.grid_points()
                break
            if isinstance(seen, dict):
                if "latitudes" in seen and "longitudes" in seen:
                    latitudes = np.asarray(seen["latitudes"])
                    longitudes = np.asarray(seen["longitudes"])
                    break
                raise TypeError(
                    f"LatLonGrid.from_field: dict field is missing 'latitudes'/'longitudes' keys (has: {sorted(seen)})"
                )
            inner = getattr(seen, "_field", None)
            if inner is None or inner is seen:
                break
            seen = inner

        if latitudes is None:
            raise TypeError(
                f"LatLonGrid.from_field: cannot extract coordinates from "
                f"{type(field).__name__}: no grid_points(), no "
                f"latitudes/longitudes dict, no _field to unwrap"
            )

        metadata: dict[str, Any] = {}
        meta = getattr(field, "metadata", None)
        if callable(meta):
            for key in ("uuidOfHGrid", "gridType"):
                try:
                    value = meta(key, default=None)
                except Exception:
                    value = None
                if value is not None:
                    metadata[key] = value
        return cls(latitudes, longitudes, metadata=metadata)

    @property
    def size(self) -> int:
        """Number of grid points."""
        return self.latitudes.size

    def latlon(self) -> tuple[NDArray[Any], NDArray[Any]]:
        """Return ``(latitudes, longitudes)``. Implements the ``Grid`` protocol."""
        return self.latitudes, self.longitudes

    def xyz(self) -> NDArray[Any]:
        """Return the (N, 3) unit-sphere Cartesian coordinates (cached)."""
        if self._xyz is None:
            from anemoi.transform.spatial import latlon_to_xyz

            self._xyz = np.stack(latlon_to_xyz(self.latitudes, self.longitudes), axis=-1)
        return self._xyz

    def _uuid_matches(self, other: "LatLonGrid") -> bool:
        """True if both grids carry an identical non-None uuidOfHGrid."""
        mine = self.metadata.get("uuidOfHGrid")
        theirs = other.metadata.get("uuidOfHGrid")
        return mine is not None and mine == theirs

    def same_ordering_as(
        self,
        other: "LatLonGrid",
        rtol: float = 1e-7,
        atol: float = 1e-8,
    ) -> bool:
        """Check whether ``other`` has the same coordinates in the same order.

        "Same ordering" means the same *xyz sequence*, not the same
        stored values: two grids with identical point ordering but
        different longitude conventions (0 to 360 vs -180 to 180) compare
        equal. This is a deliberate weakening relative to a naive
        ``np.allclose`` on raw lat/lon arrays.

        Parameters
        ----------
        other : LatLonGrid
            The grid to compare against.
        rtol : float, default 1e-7
            Relative tolerance (``np.allclose`` semantics on xyz).
        atol : float, default 1e-8
            Absolute tolerance (``np.allclose`` semantics on xyz).

        Returns
        -------
        bool
        """
        # Note: metadata is deliberately not used to short-circuit this check.
        # A matching uuidOfHGrid metadata means the same grid definition,
        # but this method exists precisely to catch stored orderings that diverge
        # from that definition.
        if self.size != other.size:
            return False
        return bool(np.allclose(self.xyz(), other.xyz(), rtol=rtol, atol=atol))

    def _match(self, other: "LatLonGrid", max_distance: float) -> tuple[NDArray[Any], NDArray[Any]]:
        """Nearest-neighbour indices and distances from self's points to other's."""
        from anemoi.transform.spatial import nearest_grid_points

        return nearest_grid_points(
            other.latitudes,
            other.longitudes,
            self.latitudes,
            self.longitudes,
            max_distance=max_distance,
            return_distances=True,
        )

    def permutation_to(self, other: "LatLonGrid", max_distance: float = DEFAULT_MAX_DISTANCE) -> NDArray[Any]:
        """Return the permutation reordering ``other``'s points onto ``self``'s.

        Returns ``p`` such that ``other.latlon()[i][p]`` coincides with
        ``self.latlon()[i]`` for ``i in (0, 1)``. Equivalently: applying
        ``p`` to data stored on ``other``'s ordering yields data on
        ``self``'s ordering (``data_self = data_other[p]``).

        Parameters
        ----------
        other : LatLonGrid
            The grid whose ordering is to be mapped onto ``self``.
        max_distance : float, default 1e-6
            Maximum nearest-neighbour distance (unit-sphere xyz) for a
            pair of points to be considered coincident.

        Returns
        -------
        np.ndarray
            Permutation indices of shape ``(self.size,)``.

        Raises
        ------
        GridMismatchError
            If the grids have different sizes, if any point's nearest
            neighbour exceeds ``max_distance``, or if the mapping is
            not a bijection (unmatched/duplicated targets).
        """
        if self.size != other.size:
            raise GridMismatchError(
                f"Grid size mismatch: {self.size} != {other.size}",
                num_unmatched=abs(self.size - other.size),
            )
        # uuidOfHGrid metadata is used here to short-circuit index-reordering.
        if self._uuid_matches(other) or self.same_ordering_as(other):
            return np.arange(self.size)

        indices, distances = self._match(other, max_distance)

        # cKDTree returns index N (out of range) for points beyond max_distance
        beyond = int(np.sum(indices >= other.size))
        unique = np.unique(indices[indices < other.size])
        num_unmatched = self.size - unique.size  # duplicated or missing targets

        if beyond or num_unmatched:
            offending = float(np.max(distances[indices < other.size])) if np.any(indices < other.size) else None
            raise GridMismatchError(
                f"Grids are not a reordering of each other: "
                f"{beyond} point(s) beyond max_distance={max_distance}, "
                f"{num_unmatched} unmatched/duplicated target point(s)",
                num_unmatched=beyond + num_unmatched,
                max_distance=offending,
            )
        return indices

    def same_points_as(self, other: "LatLonGrid", max_distance: float = DEFAULT_MAX_DISTANCE) -> bool:
        """Check whether ``other`` has the same grid point set regardless of order.

        Parameters
        ----------
        other : LatLonGrid
            The grid to compare against.
        max_distance : float, default 1e-6
            Maximum nearest-neighbour distance (unit-sphere xyz).

        Returns
        -------
        bool
        """
        try:
            self.permutation_to(other, max_distance=max_distance)
        except GridMismatchError:
            return False
        return True

    def compare(
        self,
        other: "LatLonGrid",
        rtol: float = 1e-7,
        atol: float = 1e-8,
        max_distance: float = DEFAULT_MAX_DISTANCE,
    ) -> "GridComparisonResult":
        """Classify the relationship between this grid and ``other``.

        Parameters
        ----------
        other : LatLonGrid
            The grid to compare against.
        rtol, atol : float
            Tolerances for the ordering check (see ``same_ordering_as``).
        max_distance : float
            Tolerance for the point-set check (see ``permutation_to``).

        Returns
        -------
        GridComparisonResult
            ``.kind`` is one of ``GridComparison.IDENTICAL_ORDERING``,
            ``REORDERABLE`` (with ``.permutation`` populated), or
            ``DIFFERENT`` (with ``.error`` populated).
        """
        if self.same_ordering_as(other, rtol=rtol, atol=atol):
            return GridComparisonResult(GridComparison.IDENTICAL_ORDERING)
        try:
            permutation = self.permutation_to(other, max_distance=max_distance)
        except GridMismatchError as e:
            return GridComparisonResult(GridComparison.DIFFERENT, error=e)
        return GridComparisonResult(GridComparison.REORDERABLE, permutation=permutation)

    def canonical_permutation(self) -> NDArray[Any]:
        """Indices ``p`` such that ``self.points[p]`` is in canonical order.

        Canonical order is latitude descending (N->S), then longitude
        ascending in [0, 360), row-major. For a ``regular_ll`` grid this
        is exactly the ``(nlat, nlon)`` row-major scan order, so
        ``data[p].reshape(nlat, nlon)`` is correct.

        Applying ``p`` to data stored on this grid's ordering yields
        data in canonical order: ``data_canonical = data[p]``.

        Returns
        -------
        np.ndarray
            Permutation indices of shape ``(self.size,)``.
        """
        lat_key, lon_key = _canonical_key(self.latitudes, self.longitudes)
        # lexsort uses the LAST key as primary: primary = -lat (desc), secondary = lon (asc)
        return np.lexsort((lon_key, -lat_key))

    def canonical(self) -> "LatLonGrid":
        """Return a new ``LatLonGrid`` with the same point set in canonical order.

        Output longitudes are normalized to [0, 360) — not merely
        reordered input values. Metadata is preserved.
        """
        p = self.canonical_permutation()
        return LatLonGrid(
            self.latitudes[p],
            np.mod(self.longitudes[p], 360.0),
            metadata=self.metadata,
        )

    def is_canonical(self) -> bool:
        """True if this grid is already in canonical order."""
        return bool(np.array_equal(self.canonical_permutation(), np.arange(self.size)))


class GridComparisonResult:
    """Outcome of :meth:`LatLonGrid.compare`."""

    def __init__(
        self,
        kind: GridComparison,
        permutation: NDArray[Any] | None = None,
        error: GridMismatchError | None = None,
    ) -> None:
        self.kind = kind
        self.permutation = permutation
        self.error = error

    def __repr__(self) -> str:
        return f"GridComparisonResult({self.kind.value})"

    @property
    def is_identical(self) -> bool:
        return self.kind is GridComparison.IDENTICAL_ORDERING

    @property
    def is_reorderable(self) -> bool:
        return self.kind is GridComparison.REORDERABLE

    @property
    def is_different(self) -> bool:
        return self.kind is GridComparison.DIFFERENT
