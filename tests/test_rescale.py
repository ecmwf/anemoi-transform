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
from pytest import approx

from anemoi.transform.filters.rescale import Convert
from anemoi.transform.filters.rescale import Rescale


def test_rescale(fieldlist: ekd.FieldList) -> None:
    """Test rescaling temperature from Kelvin to Celsius and back.

    Parameters
    ----------
    fieldlist : ekd.FieldList
        The fieldlist to use for testing.
    """

    fieldlist = fieldlist.sel(param="2t")
    # rescale from K to °C
    k_to_deg = Rescale(scale=1.0, offset=-273.15, param="2t")
    rescaled = k_to_deg.forward(fieldlist)

    npt.assert_allclose(rescaled[0].to_numpy(), fieldlist[0].to_numpy() - 273.15)
    # and back
    rescaled_back = k_to_deg.backward(rescaled)
    npt.assert_allclose(rescaled_back[0].to_numpy(), fieldlist[0].to_numpy())


def test_convert(fieldlist: ekd.FieldList) -> None:
    """Test converting temperature from Kelvin to Celsius and back.

    Parameters
    ----------
    fieldlist : ekd.FieldList
        The fieldlist to use for testing.
    """
    try:
        # rescale from K to °C
        fieldlist = fieldlist.sel(param="2t")
        k_to_deg = Convert(unit_in="K", unit_out="degC", param="2t")
        rescaled = k_to_deg.forward(fieldlist)
        assert rescaled[0].values.min() == fieldlist.values.min() - 273.15
        assert rescaled[0].values.std() == approx(fieldlist.values.std())
        # and back
        rescaled_back = k_to_deg.backward(rescaled)
        assert rescaled_back[0].values.min() == fieldlist.values.min()
        assert rescaled_back[0].values.std() == approx(fieldlist.values.std())
    except FileNotFoundError:
        print("Skipping test_convert because of missing UNIDATA UDUNITS2 library, " "required by cfunits.")


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
