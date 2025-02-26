# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import os
import sys
from pathlib import Path
from typing import Any
from typing import Optional

import earthkit.data as ekd
import numpy.testing as npt
import pytest
from pytest import approx

from anemoi.transform.filters.lambda_filters import EarthkitFieldLambdaFilter
from anemoi.transform.filters.rescale import Convert
from anemoi.transform.filters.rescale import Rescale

sys.path.append(Path(__file__).parents[1].as_posix())

NO_MARS = not os.path.exists(os.path.expanduser("~/.ecmwfapirc"))


def fieldlist_fixture() -> Any:
    """Fixture to create a fieldlist for testing.

    Returns
    -------
    Any
        The created fieldlist.
    """
    return ekd.from_source(
        "mars",
        {
            "param": ["2t", "sp"],
            "levtype": "sfc",
            "dates": ["2023-11-17 00:00:00"],
        },
    )


@pytest.mark.skipif(NO_MARS, reason="No access to MARS")
def test_rescale(fieldlist: Optional[Any] = None) -> None:
    """Test rescaling temperature from Kelvin to Celsius and back.

    Parameters
    ----------
    fieldlist : Optional[Any], optional
        The fieldlist to use for testing, by default None.
    """

    if fieldlist is None:
        fieldlist = fieldlist_fixture()
    fieldlist = fieldlist.sel(param="2t")
    # rescale from K to °C
    k_to_deg = Rescale(scale=1.0, offset=-273.15, param="2t")
    rescaled = k_to_deg.forward(fieldlist)

    npt.assert_allclose(rescaled[0].to_numpy(), fieldlist[0].to_numpy() - 273.15)
    # and back
    rescaled_back = k_to_deg.backward(rescaled)
    npt.assert_allclose(rescaled_back[0].to_numpy(), fieldlist[0].to_numpy())


@pytest.mark.skipif(NO_MARS, reason="No access to MARS")
def test_convert(fieldlist: Optional[Any] = None) -> None:
    """Test converting temperature from Kelvin to Celsius and back.

    Parameters
    ----------
    fieldlist : Optional[Any], optional
        The fieldlist to use for testing, by default None.
    """
    if fieldlist is None:
        fieldlist = fieldlist_fixture()
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


def _do_something(field: Any, a: float) -> Any:
    """Multiply field values by a constant.

    Parameters
    ----------
    field : Any
        The field to modify.
    a : float
        The constant to multiply by.

    Returns
    -------
    Any
        The modified field.
    """
    return field.clone(values=field.values * a)


@pytest.mark.skipif(NO_MARS, reason="No access to MARS")
def test_singlefieldlambda(fieldlist: Optional[Any] = None) -> None:
    """Test the EarthkitFieldLambdaFilter, applying a lambda filter to scale field values and then undoing the operation.

    Parameters
    ----------
    fieldlist : Optional[Any], optional
        The fieldlist to use for testing, by default None.
    """
    if fieldlist is None:
        fieldlist = fieldlist_fixture()

    fieldlist = fieldlist.sel(param="sp")

    def undo_something(field: Any, a: float) -> Any:
        """Divide field values by a constant.

        Parameters
        ----------
        field : Any
            The field to modify.
        a : float
            The constant to divide by.

        Returns
        -------
        Any
            The modified field.
        """
        return field.clone(values=field.values / a)

    something = EarthkitFieldLambdaFilter(
        fn="tests.test_filters._do_something",
        param="sp",
        fn_args=[10],
        backward_fn=undo_something,
    )

    transformed = something.forward(fieldlist)
    npt.assert_allclose(transformed[0].to_numpy(), fieldlist[0].to_numpy() * 10)

    untransformed = something.backward(transformed)
    npt.assert_allclose(untransformed[0].to_numpy(), fieldlist[0].to_numpy())


if __name__ == "__main__":
    fieldlist = fieldlist_fixture()

    test_rescale(fieldlist)
    test_convert(fieldlist)
    test_singlefieldlambda(fieldlist)

    print("All tests passed.")
