# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Ported from obsbase (ai-obs-data/base-dataset tests/test_derive.py), whose gridded IASI
# stores these functions reproduce.

import numpy as np
import pandas as pd
import pytest
from anemoi.utils.window import Window

from anemoi.transform.obs import derived
from anemoi.transform.obs import qc
from anemoi.transform.obs import select
from anemoi.transform.obs.config import Gridding
from anemoi.transform.obs.config import Qc
from anemoi.transform.obs.config import Scan
from anemoi.transform.obs.grid import cell_index
from anemoi.transform.obs.grid import cell_offsets_km
from anemoi.transform.obs.grid import grid_points
from anemoi.transform.obs.grid import mean_spacing_km
from anemoi.transform.obs.gridding import grid_windows
from anemoi.transform.obs.gridding import read_interval

WINDOW = "(-350m,+10m]"
FREQ = pd.Timedelta("6h")
DAY = pd.date_range("2020-01-01", periods=4, freq="6h")


def small_config(**block) -> Gridding:
    base = {
        "grid": "o96",
        "select": {"candidates": "latest_pass", "reduce": "nearest"},
        "qc": {"flags": {"flag": [0]}, "range": {"bt_1": [100, 300]}},
        "variables": {
            "channels": ["bt_1"],
            "columns": ["satid"],
            "derived": ["time_delta", "time_delta_norm", "n_obs", "scan_pos", "offset_north_km"],
            "forcings": ["cos_sza"],
            "scan": {"n_positions": 30, "group_size": 4, "first_side": "left"},
            "rename": {"bt_": "obsvalue_rawbt_"},
        },
        "maps": ["obsvalue_rawbt_1"],
    }
    return Gridding(**(base | block))


def mean_config(**rule) -> Gridding:
    variables = small_config().variables.model_dump(exclude_unset=True)
    variables["derived"] = [*variables["derived"], "cos_sza_obs"]
    variables |= {"spread": ["bt_1"], "wavenumbers": {"first": 645.0, "step": 0.25}}
    return small_config(select={"candidates": "latest_pass", "reduce": "mean"} | rule, variables=variables)


def rows(cfg, cell, times, fov=0, bt=250.0):
    lat, lon = grid_points(cfg.grid)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(times),
            "latitude": lat[cell],
            "longitude": lon[cell],
            "satid": pd.array([3] * len(times), dtype="Int32"),
            "flag": pd.array([0] * len(times), dtype="Int32"),
            "fov": pd.array([fov] * len(times), dtype="Int32"),
            "bt_1": bt,
        }
    )


def pass_rows(cfg):
    """Cell 500, T = 06Z: MetOp-B (3) at 05:04 and 05:05 (the latest), MetOp-C (5) at 05:00
    (same pass, other satellite), MetOp-B at 01:00 (an older pass)."""
    times = ["2020-01-01 05:04", "2020-01-01 05:05", "2020-01-01 05:00", "2020-01-01 01:00"]
    df = rows(cfg, 500, times, bt=[250.0, 260.0, 300.0, 200.0])
    df["satid"] = pd.array([3, 3, 5, 3], dtype="Int32")
    df["solar_zenith"] = 120.0
    return df


def test_latest_pass_then_nearest():
    # cell 7: an old pass (t 0) and the latest pass (t 900..1000); cell 9: one obs
    cell = np.array([7, 7, 7, 7, 9])
    t = np.array([0, 900, 1000, 950, 500])
    dist = np.array([1.0, 5.0, 3.0, 2.0, 4.0])
    cells, picked, n = select.select_cells(cell, t, dist, "latest_pass", "nearest", 200)
    assert cells.tolist() == [7, 9] and picked.tolist() == [3, 4] and n.tolist() == [3, 1]
    _, picked, n = select.select_cells(cell, t, dist, "window", "nearest", 200)
    assert picked.tolist() == [0, 4] and n.tolist() == [4, 1]
    _, picked, _ = select.select_cells(cell, t, dist, "latest_pass", "latest", 200)
    assert picked.tolist() == [2, 4]
    # equal distance: the later wins
    _, picked, _ = select.select_cells(np.array([1, 1]), np.array([5, 6]), np.ones(2), "window", "nearest", 0)
    assert picked.tolist() == [1]
    assert select.select_cells(cell[:0], t[:0], dist[:0], "window", "nearest", 0)[0].size == 0


def test_offsets_on_the_tangent_plane():
    lat, lon = grid_points("o96")
    c = np.array([1000, 1000, 40000])  # a mid-latitude cell; a cell near the south pole
    obs_lat = lat[c] + np.array([0.1, 0.0, 0.0])
    obs_lon = lon[c] + np.array([0.0, 0.1, 0.0])
    dist, east, north = cell_offsets_km("o96", obs_lat, obs_lon, c)
    assert north[0] == pytest.approx(11.12, abs=0.01) and abs(east[0]) < 1e-6
    assert east[1] == pytest.approx(11.12 * np.cos(np.radians(lat[1000])), abs=0.01)
    assert dist[2] == pytest.approx(0, abs=1e-3) and abs(east[2]) < 1e-3
    assert mean_spacing_km("o96") == pytest.approx(112.5, abs=0.1)


def test_cells_across_the_dateline():
    lat, lon = grid_points("o96")
    c = int(np.flatnonzero(lon == 0)[50])  # a cell on the prime meridian: obs just west of it
    assert cell_index("o96", np.array([lat[c]]), np.array([-0.01]))[0] == c
    assert cell_index("o96", np.array([lat[c]]), np.array([359.99]))[0] == c


def test_octahedral_grid_matches_the_named_grid():
    from anemoi.transform.grids.named import lookup

    named = lookup("o96")  # downloads once
    lat, lon = grid_points("o96")
    assert np.allclose(named["latitudes"], lat, atol=1e-9)
    assert np.allclose(named["longitudes"] % 360, lon, atol=1e-9)


def test_scan_position_and_forcings():
    scan = Scan(n_positions=30, group_size=4, first_side="left")
    assert derived.scan_position(np.array([0, 3, 4, 116, 119]), scan).tolist() == pytest.approx(
        [-1, -1, -1 + 2 / 29, 1, 1]
    )
    right = scan.model_copy(update={"first_side": "right"})
    assert derived.scan_position(np.array([0]), right)[0] == 1
    with pytest.raises(ValueError, match="outside"):
        derived.scan_position(np.array([120]), scan)
    noon = pd.Timestamp("2020-06-21 12:00")
    f = derived.forcings(["cos_sza", "cos_latitude"], noon, np.array([23.4, 0.0]), np.array([0.0, 180.0]))
    assert f["cos_sza"][0] > 0.999 and f["cos_sza"][1] == 0  # sun overhead; midnight
    assert f["cos_latitude"][1] == 1


def test_qc_counts_and_masks():
    df = pd.DataFrame(
        {
            "satid": [3, 3, 3, 5],
            "flag": pd.array([0, 1, 0, 0], dtype="Int32"),
            "bt_1": [250.0, 250.0, 400.0, np.nan],
            "date": pd.to_datetime(["2020-01-01"] * 4),
        }
    )
    rule = Qc(flags={"flag": [0]}, range={"bt_1": (100, 300)})
    out, keep, counts = qc.apply(rule, df, ["bt_1"])
    assert keep.tolist() == [True, False, False, False]
    assert np.isnan(out["bt_1"][2])
    n = counts.set_index(["group", "step"])["n"]
    assert n[3, "in"] == 3 and n[3, "flag:flag"] == 1 and n[3, "range:bt_1"] == 1
    assert n[3, "incomplete"] == 1 and n[5, "incomplete"] == 1 and n[3, "kept"] == 1


def test_window_edges_and_values():
    cfg = small_config()
    # T = 06Z holds (00:10, 06:10]: 00:10 belongs to T = 00Z, 06:10 to 06Z
    df = rows(cfg, 500, ["2020-01-01 00:10", "2020-01-01 06:10"])
    out = grid_windows(cfg, df, DAY, WINDOW, FREQ)
    arrays = out.arrays
    td = arrays["time_delta"][:, 500]
    assert td[0] == pytest.approx(1 / 6) and td[1] == pytest.approx(1 / 6)
    assert np.isnan(td[2:]).all()
    assert arrays["time_delta_norm"][0, 500] == pytest.approx(1 / 36)
    assert arrays["scan_pos"][0, 500] == -1 and arrays["n_obs"][0, 500] == 1
    assert arrays["obsvalue_rawbt_1"][0, 500] == 250 and arrays["satid"][0, 500] == 3
    assert np.isfinite(arrays["cos_sza"]).all()  # forcings fill every cell
    assert np.isnan(arrays["obsvalue_rawbt_1"][:, :500]).all()
    assert out.sums["cover"][500] == 2 and out.sums["maps"][0, :, 500].tolist() == [2, 500, 2 * 250**2]
    assert out.picked["step"].tolist() == list(DAY[:2])
    lo, hi = read_interval(Window(WINDOW), DAY)
    assert (lo, hi) == (pd.Timestamp("2019-12-31 18:10"), pd.Timestamp("2020-01-01 18:10"))


def test_empty_input_gives_forcings_only():
    cfg = small_config()
    out = grid_windows(cfg, rows(cfg, 500, [])[:0], DAY, WINDOW, FREQ)
    assert np.isnan(out.arrays["obsvalue_rawbt_1"]).all() and np.isfinite(out.arrays["cos_sza"]).all()


def test_mean_superob_averages_the_latest_satellite_of_the_latest_pass():
    cfg = mean_config()
    arrays = grid_windows(cfg, pass_rows(cfg), DAY, WINDOW, FREQ).arrays
    step = 1  # T = 06Z
    assert arrays["obsvalue_rawbt_1"][step, 500] == pytest.approx(255)
    assert arrays["n_obs"][step, 500] == 2 and arrays["satid"][step, 500] == 3
    assert arrays["obsvalue_rawbt_1_std"][step, 500] == pytest.approx(5)
    assert arrays["time_delta"][step, 500] == pytest.approx(-(55.5 / 60))
    assert arrays["cos_sza_obs"][step, 500] == pytest.approx(-0.5)  # night, unclipped
    radiance = mean_config(mean_of="radiance")
    arrays = grid_windows(radiance, pass_rows(radiance), DAY, WINDOW, FREQ).arrays
    assert 255 < arrays["obsvalue_rawbt_1"][step, 500] < 255.5  # Planck is convex
    sparse = mean_config(min_obs=3)
    arrays = grid_windows(sparse, pass_rows(sparse), DAY, WINDOW, FREQ).arrays
    assert np.isnan(arrays["obsvalue_rawbt_1"][step]).all()


def test_nearest_picks_across_satellites_and_spread_covers_the_pass():
    cfg = mean_config(reduce="nearest")
    arrays = grid_windows(cfg, pass_rows(cfg), DAY, WINDOW, FREQ).arrays
    assert arrays["obsvalue_rawbt_1"][1, 500] == 260  # all at the centre: the latest wins
    assert arrays["n_obs"][1, 500] == 3
    assert arrays["obsvalue_rawbt_1_std"][1, 500] == pytest.approx(np.std([250, 260, 300]))


def test_mean_rules_and_planck_round_trip():
    with pytest.raises(ValueError, match="one overpass"):
        mean_config(candidates="window")
    with pytest.raises(ValueError, match="wavenumbers"):
        small_config(select={"candidates": "latest_pass", "reduce": "mean", "mean_of": "radiance"})
    bt = np.array([200.0, 250.0, 300.0])
    assert derived.radiance_to_bt(derived.bt_to_radiance(bt, 900.0), 900.0) == pytest.approx(bt)
    x, starts = np.array([1.0, np.nan, 3.0, 5.0]), np.array([0, 3])
    assert select.group_mean(x, starts).tolist() == [2.0, 5.0]
    assert select.group_std(x, starts).tolist() == [1.0, 0.0]


def test_unknown_keys_are_errors():
    with pytest.raises(ValueError, match="Extra inputs"):
        small_config(windw="6h")


def test_stats_per_day_and_group():
    cfg = mean_config(reduce="nearest")
    out = grid_windows(cfg, pass_rows(cfg), DAY, WINDOW, FREQ)
    stats = out.stats(["obsvalue_rawbt_1", "n_obs"], "satid").set_index(["group", "column"])
    assert stats.loc[(3, "obsvalue_rawbt_1"), "count"] == 1 and stats.loc[(3, "n_obs"), "sum"] == 3
    assert (stats["day"] == pd.Timestamp("2020-01-01")).all()
