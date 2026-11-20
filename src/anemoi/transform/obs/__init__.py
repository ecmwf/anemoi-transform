# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Observations to gridded fields: QC, one obs (or one overpass mean) per grid cell per
time window, derived geometry and forcings. Pure (no I/O), so building a dataset and
building real-time inference input share it.

Input rows are a DataFrame with ``date``, ``latitude``, ``longitude`` and the columns the
:class:`~anemoi.transform.obs.config.Gridding` names; output is one float32 array of
(window, grid point) per variable (:func:`~anemoi.transform.obs.gridding.grid_windows`).
"""

from anemoi.transform.obs.config import Gridding
from anemoi.transform.obs.gridding import GriddedWindows
from anemoi.transform.obs.gridding import grid_windows

__all__ = ["Gridding", "GriddedWindows", "grid_windows"]
