# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import earthkit.data as ekd
import numpy as np
import pytest

from anemoi.transform.filters import create_filter_by_name as create_filter

H = datetime.timedelta(hours=1)
# Equinox: the sun is over the equator, local solar noon is at about 11:53 UTC at 0E.
EQUINOX = datetime.datetime(2016, 9, 22)
LAT = np.array([0.0, 0.0, 0.0, 0.0])
LON = np.array([0.0, 90.0, 180.0, -90.0])


def _fields(date: datetime.datetime, **params: float) -> ekd.FieldList:
    """One field per param, constant over a 4-point grid, valid at ``date``."""
    return ekd.from_source(
        "list-of-dicts",
        [
            dict(
                values=np.full(4, value, dtype=np.float32),
                latitudes=LAT,
                longitudes=LON,
                valid_datetime=date.isoformat(),
                param=param,
            )
            for param, value in params.items()
        ],
    )


def _by_param(fields: ekd.FieldList) -> dict[str, np.ndarray]:
    return {f.metadata("param"): f.to_numpy() for f in fields}


def _fake_insolation(table: dict[datetime.datetime, list[float]]):
    """Replacement for ``NightMask._is_dark`` reading insolation from a table instead of computing it."""

    def is_dark(lat, lon, date):
        return np.array(table[date]) <= 0

    return is_dark


def test_masks_only_where_dark_at_both_ends(monkeypatch) -> None:
    t = EQUINOX + 7 * H
    # point 0 always dark, 1 sunset (lit at t-1h only), 2 sunrise (lit at t only), 3 always lit
    table = {t - H: [0, 0.3, 0, 0.5], t: [0, 0, 0.1, 0.5]}
    night_mask = create_filter("night_mask", variables=["ssr"], accumulation_period="1h")
    monkeypatch.setattr(night_mask, "_is_dark", _fake_insolation(table))

    out = _by_param(night_mask.forward(_fields(t, ssr=5.0, ssrd=7.0, t2m=280.0)))

    assert out["ssr"].tolist() == [0.0, 5.0, 5.0, 5.0]  # only the always-dark point is zeroed
    assert out["ssrd"].tolist() == [7.0] * 4  # not listed in `variables`
    assert out["t2m"].tolist() == [280.0] * 4


def test_all_listed_variables_are_masked(monkeypatch) -> None:
    t = EQUINOX + 7 * H
    night_mask = create_filter("night_mask", variables=["ssr", "ssrd", "fdir"], accumulation_period="1h")
    monkeypatch.setattr(night_mask, "_is_dark", _fake_insolation({t - H: [0] * 4, t: [0] * 4}))

    out = _by_param(night_mask.forward(_fields(t, ssr=1.0, ssrd=2.0, fdir=3.0)))

    assert all(np.all(v == 0) for v in out.values())


def test_accumulation_period_sets_the_previous_date(monkeypatch) -> None:
    t = EQUINOX + 6 * H
    seen = []

    def is_dark(lat, lon, date):
        seen.append(date)
        return np.ones(4, dtype=bool)

    night_mask = create_filter("night_mask", variables="ssr", accumulation_period="6h")  # str accepted
    monkeypatch.setattr(night_mask, "_is_dark", is_dark)

    night_mask.forward(_fields(t, ssr=1.0))

    assert sorted(seen) == [EQUINOX, t]


def test_dtype_preserved_and_input_not_modified(monkeypatch) -> None:
    t = EQUINOX + 7 * H
    night_mask = create_filter("night_mask", variables=["ssr"], accumulation_period="1h")
    monkeypatch.setattr(night_mask, "_is_dark", _fake_insolation({t - H: [0] * 4, t: [0] * 4}))
    fields = _fields(t, ssr=5.0)

    out = night_mask.forward(fields)

    assert out[0].to_numpy().dtype == np.float32
    assert fields[0].to_numpy().tolist() == [5.0] * 4


def test_with_the_real_insolation_around_sunset() -> None:
    """Equinox, equator: the sun sets at about 11:53 UTC at 90E, 17:53 UTC at 0E and rises at 11:53 UTC at -90E."""
    night_mask = create_filter("night_mask", variables=["ssr"], accumulation_period="1h")

    def masked(hour: int) -> list[float]:
        return _by_param(night_mask.forward(_fields(EQUINOX + hour * H, ssr=1.0)))["ssr"].tolist()

    #            0E   90E  180E -90E
    assert masked(0) == [0.0, 1.0, 1.0, 1.0]  # midnight at 0E; noon at 180E
    assert masked(12) == [1.0, 1.0, 0.0, 1.0]  # 90E is in its sunset hour and -90E in its sunrise hour: both kept
    assert masked(13) == [1.0, 0.0, 0.0, 1.0]  # 90E now dark at both ends: zeroed
    assert masked(18) == [1.0, 0.0, 1.0, 1.0]  # 0E sunset hour: lit at 17:00, dark at 18:00 -> kept
    assert masked(19) == [0.0, 0.0, 1.0, 1.0]  # 0E dark at 18:00 and 19:00 -> zeroed


def test_backward_is_identity() -> None:
    night_mask = create_filter("night_mask", variables=["ssr"], accumulation_period="1h")
    fields = _fields(EQUINOX + 19 * H, ssr=1.0)
    assert _by_param(night_mask.backward(fields))["ssr"].tolist() == [1.0] * 4


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(variables=[], accumulation_period="1h"), "at least one"),
        (dict(variables=["ssr"], accumulation_period="0h"), "positive"),
        (dict(variables=["ssr"]), "accumulation_period"),
        (dict(accumulation_period="1h"), "variables"),
    ],
)
def test_invalid_configuration(kwargs, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        create_filter("night_mask", **kwargs)
