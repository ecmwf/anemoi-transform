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
from typing import Any

import earthkit.data as ekd
import numpy.testing as npt

from anemoi.transform.filters.lambda_filters import EarthkitFieldLambdaFilter

sys.path.append(Path(__file__).parents[1].as_posix())


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


def test_singlefieldlambda(fieldlist: ekd.FieldList) -> None:
    """Test the EarthkitFieldLambdaFilter, applying a lambda filter to scale field values and then undoing the operation.

    Parameters
    ----------
    fieldlist : ekd.FieldList
        The fieldlist to use for testing.
    """
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
        fn="tests.test_lambda._do_something",
        param="sp",
        fn_args=[10],
        backward_fn=undo_something,
    )

    transformed = something.forward(fieldlist)
    npt.assert_allclose(transformed[0].to_numpy(), fieldlist[0].to_numpy() * 10)

    untransformed = something.backward(transformed)
    npt.assert_allclose(untransformed[0].to_numpy(), fieldlist[0].to_numpy())


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
