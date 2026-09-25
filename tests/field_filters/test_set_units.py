# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy as np
import numpy.testing as npt

from anemoi.transform.filters import create_filter_by_name as create_filter


def _fields():
    return [
        {"param": "tcc", "values": [0.0, 50.0, 100.0], "units": "%"},
        {"param": "hcc", "values": [25.0, 75.0, 100.0], "units": "%"},
        {"param": "2t", "values": [280.0, 290.0, 300.0], "units": "K"},
    ]


def test_convert_units_multiple_params(test_source) -> None:
    """Convert several params in a single filter instance, stamping an explicit units label."""
    fieldlist = test_source(_fields()).forward()

    convert = create_filter(
        "convert_units",
        param=["tcc", "hcc"],
        scale=0.01,
        offset=0.0,
        units="(0 - 1)",
    )
    out = convert.forward(fieldlist)

    result = {f.metadata("param"): f for f in out}

    # selected fields: values divided by 100, units set to the exact label, param preserved
    npt.assert_allclose(result["tcc"].to_numpy(), np.array([0.0, 0.5, 1.0]))
    npt.assert_allclose(result["hcc"].to_numpy(), np.array([0.25, 0.75, 1.0]))
    assert result["tcc"].metadata("units") == "(0 - 1)"
    assert result["hcc"].metadata("units") == "(0 - 1)"

    # unselected field is passed through unchanged
    npt.assert_allclose(result["2t"].to_numpy(), np.array([280.0, 290.0, 300.0]))
    assert result["2t"].metadata("units") == "K"


def test_convert_units_defaults_identity(test_source) -> None:
    """With default scale/offset, only the units label changes."""
    fieldlist = test_source(_fields()).forward()

    convert = create_filter("convert_units", param="tcc", units="(0 - 1)")
    out = convert.forward(fieldlist)
    result = {f.metadata("param"): f for f in out}

    npt.assert_allclose(result["tcc"].to_numpy(), np.array([0.0, 50.0, 100.0]))
    assert result["tcc"].metadata("units") == "(0 - 1)"
