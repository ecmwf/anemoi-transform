# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pandas as pd
import pytest

from tests.utils import create_tabular_filter as create_filter


def test_radiance_to_brightness_temperature():
    config = {
        "input_prefix": "obsvalue_rad_",
        "output_prefix": "obsvalue_rawbt_",
        "mode": "cris_fsr",
    }
    # NB: column should end in an int (channel number)
    df = pd.DataFrame(
        {
            "obsvalue_rad_1": [0.01, 0.1, 1.0],
        }
    )
    radiance_to_brightness_temperature = create_filter("radiance_to_brightness_temperature", **config)
    result = radiance_to_brightness_temperature(df.copy())

    assert isinstance(result, pd.DataFrame)
    assert tuple(result.columns) == ("obsvalue_rawbt_1",)
    assert result.shape == (len(df), len(df.columns))

    expected = np.array([161.429057, 265.852062, 644.158476])
    assert np.allclose(result["obsvalue_rawbt_1"].to_numpy(), expected)


def test_radiance_to_brightness_temperature_missing_columns():
    config = {
        "input_prefix": "obsvalue_rad_",
        "output_prefix": "obsvalue_rawbt_",
        "mode": "cris_fsr",
    }
    df = pd.DataFrame({"foo": [0.01, 0.1, 1.0]})
    radiance_to_brightness_temperature = create_filter("radiance_to_brightness_temperature", **config)
    with pytest.raises(ValueError):
        _ = radiance_to_brightness_temperature(df.copy())


def test_radiance_to_brightness_temperature_invalid_mode():
    config = {
        "input_prefix": "obsvalue_rad_",
        "output_prefix": "obsvalue_rawbt_",
        "mode": "bad_mode",
    }
    with pytest.raises(ValueError):
        _ = create_filter("radiance_to_brightness_temperature", **config)
