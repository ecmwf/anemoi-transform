# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import numpy as np
import numpy.testing as npt
import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters.clipper import Clipper


def calc_stats(fieldlist):
    stats = {}
    for param in ("2t", "sp"):
        fields = fieldlist.sel(param=param)
        assert len(fields) == 1
        data = fields[0].to_numpy()
        stats[param] = {"min": np.min(data), "max": np.max(data)}
    return stats


@skip_if_offline
def test_clipper_minimum(fieldlist: ekd.FieldList) -> None:
    before_stats = calc_stats(fieldlist)
    clipper = Clipper(minimum=300.0, param="2t")
    clipped = clipper(fieldlist)
    after_stats = calc_stats(clipped)

    data = fieldlist.sel(param="2t").to_numpy()
    ref = np.clip(data, 300.0, None)

    npt.assert_allclose(clipped.sel(param="2t").to_numpy(), ref)
    npt.assert_allclose(clipped.sel(param="sp").to_numpy(), fieldlist.sel(param="sp").to_numpy())

    assert after_stats["sp"]["min"] == before_stats["sp"]["min"]
    assert after_stats["sp"]["max"] == before_stats["sp"]["max"]

    assert after_stats["2t"]["min"] == pytest.approx(300.0)
    assert after_stats["2t"]["max"] == before_stats["2t"]["max"]


@skip_if_offline
def test_clipper_maximum(fieldlist: ekd.FieldList) -> None:
    before_stats = calc_stats(fieldlist)
    clipper = Clipper(maximum=300.0, param="2t")
    clipped = clipper(fieldlist)
    after_stats = calc_stats(clipped)

    data = fieldlist.sel(param="2t").to_numpy()
    ref = np.clip(data, None, 300.0)

    npt.assert_allclose(clipped.sel(param="2t").to_numpy(), ref)
    npt.assert_allclose(clipped.sel(param="sp").to_numpy(), fieldlist.sel(param="sp").to_numpy())

    assert after_stats["sp"]["min"] == before_stats["sp"]["min"]
    assert after_stats["sp"]["max"] == before_stats["sp"]["max"]

    assert after_stats["2t"]["min"] == before_stats["2t"]["min"]
    assert after_stats["2t"]["max"] == pytest.approx(300.0)


@skip_if_offline
def test_clipper_both(fieldlist: ekd.FieldList) -> None:
    before_stats = calc_stats(fieldlist)
    clipper = Clipper(minimum=300.0, maximum=305.0, param="2t")
    clipped = clipper(fieldlist)
    after_stats = calc_stats(clipped)

    data = fieldlist.sel(param="2t").to_numpy()
    ref = np.clip(data, 300.0, 305.0)

    npt.assert_allclose(clipped.sel(param="2t").to_numpy(), ref)
    npt.assert_allclose(clipped.sel(param="sp").to_numpy(), fieldlist.sel(param="sp").to_numpy())

    assert after_stats["sp"]["min"] == before_stats["sp"]["min"]
    assert after_stats["sp"]["max"] == before_stats["sp"]["max"]

    assert after_stats["2t"]["min"] == pytest.approx(300.0)
    assert after_stats["2t"]["max"] == pytest.approx(305.0)
