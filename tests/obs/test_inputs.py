# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

# Combined inputs (anemoi.transform.obs.inputs) and the bit-field and per-channel QC rules.

import numpy as np
import pandas as pd
import pytest

from anemoi.transform.obs import qc
from anemoi.transform.obs.config import Gridding
from anemoi.transform.obs.config import Qc
from anemoi.transform.obs.grid import grid_points
from anemoi.transform.obs.gridding import grid_windows
from anemoi.transform.obs.inputs import Input
from anemoi.transform.obs.inputs import combine

WINDOW = "(-350m,+10m]"
FREQ = pd.Timedelta("6h")
DAY = pd.date_range("2020-01-01", periods=4, freq="6h")


def config(**block) -> Gridding:
    """Two instruments in one dataset: bt_1, bt_2 shared, bt_3 only the newer one's."""
    base = {
        "grid": "o96",
        "select": {"candidates": "latest_pass", "reduce": "mean"},
        "qc": {"require_complete": "any"},
        "variables": {
            "channels": ["bt_1", "bt_2", "bt_3"],
            "columns": ["satid"],
            "derived": ["scan_pos", "n_obs"],
            "scan": "input",
        },
    }
    return Gridding(**(base | block))


def frame(cell, times, satid, fov, **channels):
    lat, lon = grid_points("o96")
    n = len(times)
    return pd.DataFrame(
        {"date": pd.to_datetime(times), "latitude": lat[cell], "longitude": lon[cell], "satid": satid, "fov": fov}
        | {k: np.full(n, v, np.float32) for k, v in channels.items()}
    )


OLD = Input(
    dates=("2019-01-01", "2020-01-01 06:00"),
    channels={"bt_a": "bt_1", "bt_b": "bt_2"},
    group_offset=10000,
    scan={"n_positions": 11, "first_side": "left"},
)
NEW = Input(dates=("2020-01-01", None), groups=[3, 5], scan={"n_positions": 30, "first_side": "right"})


def test_inputs_fill_their_slots_with_their_own_scan_and_offset():
    cfg = config()
    old = frame(500, ["2020-01-01 05:00", "2020-01-01 07:00"], 204, 0, bt_a=210.0, bt_b=220.0)
    new = frame(500, ["2020-01-01 11:00"] * 2, [3, 4], 0, bt_1=250.0, bt_2=260.0, bt_3=270.0)
    assert OLD.columns(cfg) == ["date", "latitude", "longitude", "satid", "bt_a", "bt_b", "fov"]
    df = combine(cfg, [(OLD, old), (NEW, new)])
    assert list(df.columns) == cfg.input_columns()
    # OLD keeps rows before 06Z, NEW only satid 3 and 5
    assert df["satid"].tolist() == [10204, 3] and np.isnan(df["bt_3"][0]) and df["bt_1"].tolist() == [210, 250]
    assert df["scan_pos"].tolist() == [-1, 1]  # fov 0: left of track for OLD, right for NEW
    arrays = grid_windows(cfg, df, DAY, WINDOW, FREQ).arrays
    assert arrays["bt_1"][1, 500] == 210 and arrays["satid"][1, 500] == 10204 and np.isnan(arrays["bt_3"][1, 500])
    assert arrays["bt_3"][2, 500] == 270 and arrays["scan_pos"][2, 500] == 1


def test_input_rules():
    cfg = config()
    assert OLD.overlaps(pd.Timestamp("2019-12-31"), pd.Timestamp("2020-01-02"))
    assert not OLD.overlaps(pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03"))
    assert Input(channels=["bt_1"]).source_of("bt_1") == "bt_1" and Input(channels=["bt_1"]).source_of("bt_2") is None
    with pytest.raises(ValueError, match="no dataset channel"):
        Input(channels={"x": "bt_9"}, scan=OLD.scan).columns(cfg)
    with pytest.raises(ValueError, match="needs its scan"):
        Input().columns(cfg)
    with pytest.raises(ValueError, match="same dataset channel"):
        Input(channels={"a": "bt_1", "b": "bt_1"})
    with pytest.raises(ValueError, match="before end"):
        Input(dates=("2020-01-02", "2020-01-01"))
    assert combine(cfg, []).columns.tolist() == cfg.input_columns()


def test_bits_value_flags_and_required_channels():
    df = pd.DataFrame(
        {
            "satid": [3, 3, 3, 3],
            "qualind": [0, 1 << 31, 0, 0],
            "chanqual_1": [0, 0, 8, 0],
            "bt_1": [250.0, 250.0, 250.0, np.nan],
            "bt_2": [260.0, 260.0, 260.0, 260.0],
            "date": pd.to_datetime(["2020-01-01"] * 4),
        }
    )
    rule = Qc(bits={"qualind": 1 << 31}, value_flags={"bt_1": "chanqual_1"}, require_complete=["bt_1"])
    out, keep, counts = qc.apply(rule, df, ["bt_1", "bt_2"])
    assert keep.tolist() == [True, False, False, False] and np.isnan(out["bt_1"][2])
    n = counts.set_index(["group", "step"])["n"]
    assert n[3, "bits:qualind"] == 1 and n[3, "value_flag:bt_1"] == 1 and n[3, "incomplete"] == 2
    _, keep, _ = qc.apply(rule.model_copy(update={"require_complete": "any"}), df, ["bt_1", "bt_2"])
    assert keep.tolist() == [True, False, True, True]
    _, keep, _ = qc.apply(rule.model_copy(update={"require_complete": False}), df, ["bt_1", "bt_2"])
    assert keep.tolist() == [True, False, True, True]
    with pytest.raises(ValueError, match="value_flags"):
        config(qc={"value_flags": {"bt_9": "chanqual_1"}})
    with pytest.raises(ValueError, match="require_complete"):
        config(qc={"require_complete": ["bt_9"]})
    assert set(config(qc={"bits": {"qualind": 1}, "value_flags": {"bt_1": "q1"}}).input_columns()) >= {"qualind", "q1"}


def test_input_qc_on_its_own_columns():
    cfg = config()
    flagged = Input(
        channels={"bt_a": "bt_1", "bt_b": "bt_2"},
        scan=OLD.scan,
        qc={"bits": {"pixel_qc": 2}, "value_flags": {"bt_a": "issue_a"}, "range": {"zen": [0, 90]}},
    )
    old = frame(500, ["2020-01-01 05:00"] * 4, 204, 0, bt_a=210.0, bt_b=220.0).assign(
        pixel_qc=[0, 2, 0, 0], issue_a=[0, 0, 1, 0], zen=[10.0, 10.0, np.nan, 178.0]
    )
    assert flagged.columns(cfg)[-3:] == ["pixel_qc", "issue_a", "zen"]
    df = combine(cfg, [(flagged, old)])
    assert list(df.columns) == cfg.input_columns()  # the flag columns are not passed on
    # a missing zenith passes the range
    assert df["bt_1"].isna().tolist() == [False, True, True, True]
    assert df["bt_2"].isna().tolist() == [False, True, False, True]
    _, keep, counts = qc.apply(cfg.qc, df, cfg.variables.channels)
    assert keep.tolist() == [True, False, True, False]
    assert counts.set_index(["group", "step"])["n"][204, "incomplete"] == 2
    # a row out of the input qc has its scan position unchecked (fov -1 out of range)
    bad_fov = old.assign(fov=[0, 0, 0, -1])
    with pytest.raises(ValueError, match="maps outside"):
        combine(cfg, [(OLD, bad_fov)])
    fenced = Input(channels={"bt_a": "bt_1", "bt_b": "bt_2"}, scan=OLD.scan, qc={"range": {"fov": [0, 10]}})
    df = combine(cfg, [(fenced, bad_fov)])
    assert df["scan_pos"].isna().tolist() == [False, False, False, True]
    with pytest.raises(ValueError, match="no column this input fills"):
        Input(channels={"bt_a": "bt_1"}, scan=OLD.scan, qc={"value_flags": {"bt_b": "q"}}).columns(cfg)
    with pytest.raises(ValueError, match="require_complete"):
        flagged.columns(config(qc={"require_complete": False}))
