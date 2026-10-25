# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Tests for the GRIB-specific routines in :mod:`anemoi.transform.grib`."""

import numpy as np
import pytest

from anemoi.transform import FieldList
from anemoi.transform.grib import grib_handle
from anemoi.transform.grib import new_grib_output


@pytest.fixture
def sample_field():
    """A single GRIB-backed field from the eccodes samples (offline)."""
    return FieldList.from_source("sample", "test.grib")[0]


def test_grib_handle_returns_handle(sample_field):
    """``grib_handle`` returns the codes handle of a GRIB-backed field."""
    handle = grib_handle(sample_field)
    # The handle must be cloneable (consumers mutate a clone) and queryable.
    clone = handle.clone()
    assert clone is not None
    assert handle.get("edition") in (1, 2)


def test_grib_handle_accepts_raw_field(sample_field):
    """``grib_handle`` accepts both a wrapped ``Field`` and a raw earthkit field."""
    from_wrapped = grib_handle(sample_field)
    from_raw = grib_handle(sample_field._field)
    assert from_wrapped.get("edition") == from_raw.get("edition")


def test_grib_handle_raises_for_non_grib():
    """``grib_handle`` raises for a field that is not backed by a GRIB message."""

    class _NotGrib:
        def _get_grib(self, strict=True):
            return None

    with pytest.raises(ValueError, match="not backed by a GRIB message"):
        grib_handle(_NotGrib())


def test_new_grib_output_roundtrip(sample_field, tmp_path):
    """A field written via ``new_grib_output`` reads back with the same values."""
    path = str(tmp_path / "out.grib")

    values = sample_field.to_numpy()

    out = new_grib_output(path)
    out.write(values, template=sample_field)
    out.close()

    read_back = FieldList.from_source("file", path)
    assert len(read_back) == 1
    np.testing.assert_allclose(read_back[0].to_numpy(), values)
