# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import sys
from pathlib import Path

import numpy.testing as npt

from anemoi.transform.filters.rescale import Rescale, Convert
from anemoi.transform.filters.lambda_filters import EarthkitFieldLambdaFilter
import earthkit.data as ekd
from pytest import approx

sys.path.append(Path(__file__).parents[1].as_posix())

def test_rescale(fieldlist):
    fieldlist = fieldlist.sel(param="2t")
    # rescale from K to °C
    k_to_deg = Rescale(scale=1.0, offset=-273.15, param="2t")
    rescaled = k_to_deg.forward(fieldlist)
    
    npt.assert_allclose(
        rescaled[0].to_numpy(),
        fieldlist[0].to_numpy() - 273.15
    )
    # and back
    rescaled_back = k_to_deg.backward(rescaled)
    npt.assert_allclose(
        rescaled_back[0].to_numpy(),
        fieldlist[0].to_numpy()
    )
    # rescale from °C to F
    deg_to_far = Rescale(scale=9 / 5, offset=32, param="2t")
    rescaled_farheneit = deg_to_far.forward(rescaled)
    npt.assert_allclose(
        rescaled_farheneit[0].to_numpy(),
        9 / 5 * rescaled[0].to_numpy() + 32
    )
    # rescale from F to K
    rescaled_back = k_to_deg.backward(deg_to_far.backward(rescaled_farheneit))
    npt.assert_allclose(
        rescaled_back[0].to_numpy(),
        fieldlist[0].to_numpy()
    )

def test_convert(fieldlist):
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



# used in the test below
def _do_something(field, a):
    return field.clone(values=field.values * a)

def test_singlefieldlambda(fieldlist):

    fieldlist = fieldlist.sel(param="sp")

    def undo_something(field, a):
        return field.clone(values=field.values / a)
    
    something = EarthkitFieldLambdaFilter(
        fn="tests.test_filters._do_something",
        param="sp",
        args=[10],
        backward_fn=undo_something,
    )

    transformed = something.forward(fieldlist)
    npt.assert_allclose(
        transformed[0].to_numpy(),
        fieldlist[0].to_numpy() * 10
    )

    untransformed = something.backward(transformed)
    npt.assert_allclose(
        untransformed[0].to_numpy(),
        fieldlist[0].to_numpy()
    )



if __name__ == "__main__":

    fieldlist = ekd.from_source(
        "mars",
        {
            "param": ["2t", "sp"],
            "levtype": "sfc",
            "dates": ["2023-11-17 00:00:00"],
        },
    )

    test_rescale(fieldlist)
    try:
        test_convert(fieldlist)
    except FileNotFoundError:
        print(
            "Skipping test_convert because of missing UNIDATA UDUNITS2 library, "
            "required by cfunits."
        )
    test_singlefieldlambda(fieldlist)

    print("All tests passed.")
