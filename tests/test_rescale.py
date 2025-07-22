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
import pytest
from anemoi.utils.testing import skip_if_offline
from pytest import approx

from anemoi.transform.filters.rescale import Convert
from anemoi.transform.filters.rescale import Rescale


def skip_missing_udunits2():
    """Skip tests if udunits2 package is not available."""
    # Can't use utils.testing.skip_missing_packages because it only fails
    # when cfunits.Units is imported...
    # NB: cfunits depends on udunits2 which is a system library and not installed with pip
    try:
        from cfunits import Units  # noqa: F401

        return lambda f: f
    except FileNotFoundError:
        return pytest.mark.skip(reason="udunits2 not found")


@skip_if_offline
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


@skip_missing_udunits2()
@skip_if_offline
def test_convert(fieldlist: ekd.FieldList) -> None:
    """Test converting temperature from Kelvin to Celsius and back.

    Parameters
    ----------
    fieldlist : ekd.FieldList
        The fieldlist to use for testing.
    """
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


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
