# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import numpy as np
import numpy.testing as npt

from anemoi.transform.filters.clipper import Clipper


def test_clipper_1(fieldlist: ekd.FieldList) -> None:
    fieldlist = fieldlist.sel(param="2t")

    clipper = Clipper(minimum=1.0, param="2t")
    clipped = clipper.forward(fieldlist)

    data = fieldlist[0].to_numpy()
    ref = np.clip(data, 1.0, None)

    npt.assert_allclose(clipped[0].to_numpy(), ref)


def test_clipper_2(fieldlist: ekd.FieldList) -> None:
    fieldlist = fieldlist.sel(param="2t")

    clipper = Clipper(maximum=200.0, param="2t")
    clipped = clipper.forward(fieldlist)

    data = fieldlist[0].to_numpy()
    ref = np.clip(data, None, 200.0)

    npt.assert_allclose(clipped[0].to_numpy(), ref)


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
