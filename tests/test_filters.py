# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import earthkit.data as ekd
from pytest import approx

from anemoi.transform.filters.rescale import Convert
from anemoi.transform.filters.rescale import Rescale


def test_rescale():
    # rescale from K to °C
    temp = ekd.from_source("mars", {"param": "2t", "levtype": "sfc", "dates": ["2023-11-17 00:00:00"]})
    fieldlist = temp.to_fieldlist()
    k_to_deg = Rescale(scale=1.0, offset=-273.15, param="2t")
    rescaled = k_to_deg.forward(fieldlist)
    assert rescaled[0].values.min() == temp.values.min() - 273.15
    assert rescaled[0].values.std() == approx(temp.values.std())
    # and back
    rescaled_back = k_to_deg.backward(rescaled)
    assert rescaled_back[0].values.min() == temp.values.min()
    assert rescaled_back[0].values.std() == approx(temp.values.std())
    # rescale from °C to F
    deg_to_far = Rescale(scale=9 / 5, offset=32, param="2t")
    rescaled_farheneit = deg_to_far.forward(rescaled)
    assert rescaled_farheneit[0].values.min() == 9 / 5 * rescaled[0].values.min() + 32
    assert rescaled_farheneit[0].values.std() == approx((9 / 5) * rescaled[0].values.std())
    # rescale from F to K
    rescaled_back = k_to_deg.backward(deg_to_far.backward(rescaled_farheneit))
    assert rescaled_back[0].values.min() == temp.values.min()
    assert rescaled_back[0].values.std() == approx(temp.values.std())


def test_convert():
    # rescale from K to °C
    temp = ekd.from_source("mars", {"param": "2t", "levtype": "sfc", "dates": ["2023-11-17 00:00:00"]})
    fieldlist = temp.to_fieldlist()
    k_to_deg = Convert(unit_in="K", unit_out="degC", param="2t")
    rescaled = k_to_deg.forward(fieldlist)
    assert rescaled[0].values.min() == temp.values.min() - 273.15
    assert rescaled[0].values.std() == approx(temp.values.std())
    # and back
    rescaled_back = k_to_deg.backward(rescaled)
    assert rescaled_back[0].values.min() == temp.values.min()
    assert rescaled_back[0].values.std() == approx(temp.values.std())


if __name__ == "__main__":
    test_rescale()
    test_convert()
