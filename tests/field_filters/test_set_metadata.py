# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import numpy.testing as npt
import pytest
from anemoi.utils.testing import skip_if_offline

from anemoi.transform import FieldList
from anemoi.transform.filters import create_filter_by_name as create_filter


@skip_if_offline
def test_set_metadata_selected(fieldlist: FieldList) -> None:
    """Set metadata on the selected fields only; others pass through unchanged.

    Parameters
    ----------
    fieldlist : FieldList
        The fieldlist to use for testing.
    """
    before = {field.parameter.variable(): field.to_numpy().copy() for field in fieldlist}

    f = create_filter(
        "set_metadata",
        selection={"parameter.variable": "2t"},
        metadata={"parameter.units": "degC"},
    )
    result = f.forward(fieldlist)

    assert len(result) == len(fieldlist)
    for field in result:
        param = field.parameter.variable()
        npt.assert_allclose(before[param], field.to_numpy())
        if param == "2t":
            assert field.get("parameter.units") == "degC"
        else:
            assert field.get("parameter.units") != "degC"


@skip_if_offline
def test_set_metadata_all_fields(fieldlist: FieldList) -> None:
    """Without a selection, the metadata is applied to every field.

    Parameters
    ----------
    fieldlist : FieldList
        The fieldlist to use for testing.
    """
    f = create_filter("set_metadata", metadata={"parameter.units": "degC"})
    result = f.forward(fieldlist)

    for field in result:
        assert field.get("parameter.units") == "degC"


@skip_if_offline
def test_set_metadata_legacy_keys(fieldlist: FieldList) -> None:
    """Legacy metadata keys are translated with a deprecation warning.

    Parameters
    ----------
    fieldlist : FieldList
        The fieldlist to use for testing.
    """
    with pytest.warns(DeprecationWarning):
        f = create_filter(
            "set_metadata",
            selection={"param": "2t"},
            metadata={"units": "degC"},
        )
    result = f.forward(fieldlist)

    for field in result:
        if field.parameter.variable() == "2t":
            assert field.get("parameter.units") == "degC"


def test_set_metadata_validation() -> None:
    """The metadata input must be a non-empty dictionary."""
    with pytest.raises(ValueError):
        create_filter("set_metadata", metadata={})

    with pytest.raises(ValueError):
        create_filter("set_metadata", metadata="units=degC")

    with pytest.raises(TypeError):
        create_filter("set_metadata")
