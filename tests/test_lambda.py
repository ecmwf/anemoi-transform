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

import earthkit.data as ekd
import numpy.testing as npt
from anemoi.utils.testing import skip_if_offline

from anemoi.transform.filters import filter_registry

sys.path.append(Path(__file__).parents[1].as_posix())


def do_something(field: ekd.Field, a: float) -> ekd.Field:
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


def undo_something(field: ekd.Field, a: float) -> ekd.Field:
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


@skip_if_offline
def test_earthkitfieldlambda(fieldlist: ekd.FieldList) -> None:
    """Test the EarthkitFieldLambdaFilter, applying a lambda filter to scale field values and then undoing the operation.

    Parameters
    ----------
    fieldlist : ekd.FieldList
        The fieldlist to use for testing.
    """

    before_filter = {field.metadata("param"): field.to_numpy().copy() for field in fieldlist}
    filter = filter_registry.create(
        "earthkitfieldlambda",
        fn="tests.test_lambda.do_something",
        param="sp",
        fn_args=[10],
        backward_fn="tests.test_lambda.undo_something",
    )

    transformed = filter.forward(fieldlist)
    after_forward = {field.metadata("param"): field.to_numpy().copy() for field in transformed}

    untransformed = filter.backward(transformed)
    after_backward = {field.metadata("param"): field.to_numpy().copy() for field in untransformed}

    for param in ("sp", "2t"):
        # round trip works
        npt.assert_allclose(before_filter[param], after_backward[param])

        if param == "sp":
            # sp fields transformed as expected
            npt.assert_allclose(after_forward[param], before_filter[param] * 10)
        else:
            # non-sp fields are unchanged
            npt.assert_allclose(before_filter[param], after_forward[param])
