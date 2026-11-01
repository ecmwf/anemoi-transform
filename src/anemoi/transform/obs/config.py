# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""What a gridded observation variable holds: QC, selection and variables (a recipe's
``grid_observations`` block, less the window and dates the dataset supplies).
"""

from typing import Literal

import pandas as pd
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

# Computed from the selected observation (anemoi.transform.obs.gridding.row_values)
DERIVED = (
    "time_delta",  # obs time - T [h]
    "time_delta_norm",  # time_delta / the dataset frequency: model steps
    "cos_vza",
    "scan_pos",  # signed cross-track position in [-1, 1], negative = left of the flight track
    "signed_sin_vza",
    "offset_east_km",  # obs position - cell centre on the cell's tangent plane
    "offset_north_km",
    "offset_east_norm",  # offsets / the grid's mean spacing (grid.mean_spacing_km)
    "offset_north_norm",
    "n_obs",  # candidates the cell's value was chosen from (or averaged)
    # cos of the obs' own solar zenith angle, unclipped (night < 0): computed from the obs'
    # time and position, or from variables.solar_zenith
    "cos_sza_obs",
)
# Known functions of (T, grid point), filled in every cell (anemoi.transform.obs.derived)
FORCINGS = (
    "cos_latitude",
    "sin_latitude",
    "cos_longitude",
    "sin_longitude",
    "sin_julian_day",
    "cos_julian_day",
    "sin_local_time",
    "cos_local_time",
    "cos_sza",
)


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Select(Strict):
    """What fills a cell at T (anemoi.transform.obs.select.candidate_groups).

    candidates: latest_pass = the cell's obs within pass_gap of its latest one (one overpass),
    window = every obs in the window. reduce: nearest = the candidate nearest the cell centre,
    latest = the latest candidate (both one whole row: every channel from one pixel); mean =
    the average of the candidates of the latest obs' group, e.g. satellite (a superob: every
    variable averaged per row, so cos_vza is the mean cosine and the offsets the centroid).
    """

    candidates: Literal["latest_pass", "window"]
    pass_gap: str = "15min"
    reduce: Literal["nearest", "latest", "mean"]
    min_obs: int = 1  # cells with fewer candidates stay empty
    # channel -> K: cells whose candidates' spread (population std) of the channel exceeds it
    # stay empty (stripes, mis-navigated scans); a cell without the channel is kept
    max_spread: dict[str, float] = Field(default_factory=dict)
    # mean only: average channels as radiance (variables.wavenumbers), stored as the BT of
    # the mean radiance; bt averages the BTs
    mean_of: Literal["bt", "radiance"] = "bt"

    @model_validator(mode="after")
    def _mean(self):
        if self.reduce == "mean" and self.candidates != "latest_pass":
            raise ValueError("reduce: mean averages one overpass: set candidates: latest_pass")
        if self.mean_of == "radiance" and self.reduce != "mean":
            raise ValueError("mean_of applies to reduce: mean")
        if self.min_obs < 1:
            raise ValueError("min_obs >= 1")
        return self


class Wavenumbers(Strict):
    """Channel <prefix>N has wavenumber first + step (N - 1) [cm-1] (IASI: 645, 0.25)."""

    first: float
    step: float

    def of(self, column: str) -> float:
        return self.first + self.step * (int(column.rsplit("_", 1)[1]) - 1)


class KnownBad(Strict):
    """Values set missing: `columns` of the groups listed (all if empty) from start
    (inclusive) to end (exclusive; open if unset).
    """

    columns: list[str]
    groups: list[int] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None
    reason: str


class Qc(Strict):
    """Applied in this order: flags and bits (rows out), value_flags, range and known_bad
    (values out), require_complete (rows out).
    """

    flags: dict[str, list[int]] = Field(default_factory=dict)  # keep rows whose column is one of the values
    # rows out where column & mask != 0 (bit fields, e.g. AAPP qualind bit 31: 2147483648)
    bits: dict[str, int] = Field(default_factory=dict)
    # channel -> flag column: the channel's value is missing where its flag is non-zero
    value_flags: dict[str, str] = Field(default_factory=dict)
    range: dict[str, tuple[float, float]] = Field(default_factory=dict)  # values outside [lo, hi] become missing
    known_bad: list[KnownBad] = Field(default_factory=list)
    # true: every channel present; "any": at least one; a list: those channels; false: no rule
    require_complete: bool | Literal["any"] | list[str] = True

    def required(self, channels: list[str]) -> list[str] | None:
        """The channels a row must all have, or None for "any"."""
        if self.require_complete == "any":
            return None
        if self.require_complete is True:
            return channels
        return list(self.require_complete or [])


class Scan(Strict):
    """Scan position from a field-of-view column: position k = fov // group_size in
    [0, n_positions); first_side = the side of the flight track position 0 views.
    """

    column: str = "fov"
    n_positions: int
    group_size: int = 1
    first_side: Literal["left", "right"]


class Variables(Strict):
    channels: list[str]  # observed values (require_complete checks these)
    # other input columns, gridded as they are
    columns: list[str] = Field(default_factory=list)
    derived: list[Literal[DERIVED]] = Field(default_factory=list)
    forcings: list[Literal[FORCINGS]] = Field(default_factory=list)
    # needed by scan_pos and signed_sin_vza; "input": the rows carry scan_pos, computed per
    # input (anemoi.transform.obs.inputs.Input.scan) when the inputs scan differently
    scan: Scan | Literal["input"] | None = None
    # channels whose spread over the cell's candidates is stored as <output name>_std
    spread: list[str] = Field(default_factory=list)
    # cos_sza_obs from this column [deg] (as reported); unset: computed from the obs' time,
    # latitude and longitude (derived.cos_sza), so it needs no column and is never stale
    solar_zenith: str | None = None
    wavenumbers: Wavenumbers | None = None  # needed by select.mean_of: radiance
    # output names: a key ending in "_" renames a prefix (bt_ -> obsvalue_rawbt_)
    rename: dict[str, str] = Field(default_factory=dict)
    # put before every output name, after rename (derived and forcings too): e.g. "iasi_"
    prefix: str = ""

    def out_name(self, column: str) -> str:
        if column in self.rename:
            return self.rename[column]
        for old, new in self.rename.items():
            if old.endswith("_") and column.startswith(old):
                return new + column[len(old) :]
        return column

    def observed(self) -> list[str]:
        """Output names of every variable that is missing where no obs was selected."""
        names = [self.out_name(c) for c in (*self.columns, *self.channels)] + list(self.derived)
        return names + [f"{self.out_name(c)}_std" for c in self.spread]

    def outputs(self) -> list[str]:
        return self.observed() + list(self.forcings)

    def final(self, name: str) -> str:
        """The stored name of an output (observed() / outputs() name it before the prefix)."""
        return self.prefix + name


class Gridding(Strict):
    grid: str  # oNNN (built exactly, no download) or any name anemoi.transform.grids knows
    # QC counts, stats and mean superobs are per group (an integer column, e.g. satid)
    group_by: str = "satid"
    select: Select
    qc: Qc = Qc()
    variables: Variables
    # output variables whose per-cell count, sum and sum of squares are kept for maps
    maps: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self):
        v = self.variables
        if {"scan_pos", "signed_sin_vza"} & set(v.derived) and v.scan is None:
            raise ValueError("scan_pos / signed_sin_vza need variables.scan")
        if self.select.mean_of == "radiance" and v.wavenumbers is None:
            raise ValueError("select.mean_of: radiance needs variables.wavenumbers")
        if unknown := set(v.spread) - set(v.channels):
            raise ValueError(f"spread names no channel: {sorted(unknown)}")
        if unknown := set(self.select.max_spread) - set(v.channels):
            raise ValueError(f"select.max_spread names no channel: {sorted(unknown)}")
        if unknown := set(self.qc.value_flags) - set(v.channels):
            raise ValueError(f"qc.value_flags names no channel: {sorted(unknown)}")
        if isinstance(self.qc.require_complete, list) and (unknown := set(self.qc.require_complete) - set(v.channels)):
            raise ValueError(f"qc.require_complete names no channel: {sorted(unknown)}")
        outputs = v.outputs()
        if len(set(outputs)) != len(outputs):
            raise ValueError("two variables share an output name")
        if unknown := set(self.maps) - {v.final(n) for n in v.observed()}:
            raise ValueError(f"maps names no observed output (with the prefix): {sorted(unknown)}")
        pd.Timedelta(self.select.pass_gap)
        return self

    def input_columns(self) -> list[str]:
        """The input columns gridding reads."""
        v = self.variables
        need = ["date", "latitude", "longitude", self.group_by, *self.qc.flags, *self.qc.bits]
        need += [*self.qc.value_flags.values(), *self.qc.range]
        need += [*v.columns, *v.channels]
        need += [c for bad in self.qc.known_bad for c in bad.columns]
        if {"cos_vza", "signed_sin_vza"} & set(v.derived):
            need.append("zenith")
        if "cos_sza_obs" in v.derived and v.solar_zenith:
            need.append(v.solar_zenith)
        if v.scan == "input":
            need.append("scan_pos")
        elif v.scan is not None:
            need.append(v.scan.column)
        return list(dict.fromkeys(need))
