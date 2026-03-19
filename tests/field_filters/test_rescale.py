# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import numpy.testing as npt
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import filter_registry


@skip_if_offline
def test_rescale(fieldlist: ekd.FieldList) -> None:
    """Test rescaling temperature from Kelvin to Celsius and back.

    Parameters
    ----------
    fieldlist : ekd.FieldList
        The fieldlist to use for testing.
    """

    before_filter = {field.metadata("param"): field.to_numpy().copy() for field in fieldlist}

    # rescale from K to °C
    k_to_deg = filter_registry.create("rescale", scale=1.0, offset=-273.15, param="2t")
    rescaled = k_to_deg.forward(fieldlist)
    after_forward = {field.metadata("param"): field.to_numpy().copy() for field in rescaled}

    # and back
    rescaled_back = k_to_deg.backward(rescaled)
    after_backward = {field.metadata("param"): field.to_numpy().copy() for field in rescaled_back}

    for param in ("2t", "sp"):
        npt.assert_allclose(before_filter[param], after_backward[param])

        if param == "2t":
            npt.assert_allclose(before_filter[param] - 273.15, after_forward[param])
        else:
            npt.assert_allclose(before_filter[param], after_forward[param])


@skip_if_offline
def test_convert(fieldlist: ekd.FieldList) -> None:
    """Test converting temperature from Kelvin to Celsius and back.

    Parameters
    ----------
    fieldlist : ekd.FieldList
        The fieldlist to use for testing.
    """
    before_filter = {field.metadata("param"): field.to_numpy().copy() for field in fieldlist}
    # rescale from K to °C
    k_to_deg = filter_registry.create("convert", unit_in="K", unit_out="degC", param="2t")
    rescaled = k_to_deg.forward(fieldlist)
    after_forward = {field.metadata("param"): field.to_numpy().copy() for field in rescaled}

    # and back
    rescaled_back = k_to_deg.backward(rescaled)
    after_backward = {field.metadata("param"): field.to_numpy().copy() for field in rescaled_back}

    for param in ("2t", "sp"):
        npt.assert_allclose(before_filter[param], after_backward[param])

        if param == "2t":
            npt.assert_allclose(before_filter[param] - 273.15, after_forward[param])
        else:
            npt.assert_allclose(before_filter[param], after_forward[param])
