# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import numpy as np
import pytest

from anemoi.transform import Field
from anemoi.transform import FieldList
from anemoi.transform.naming import create_naming
from anemoi.transform.naming import create_naming_from_remapping
from anemoi.transform.naming import variable_naming_from_remapping
from anemoi.transform.naming.param import ParamNaming
from anemoi.transform.naming.param_levelist import ParamLevelNaming
from anemoi.transform.naming.template import TemplateNaming


def _field(param: str, level: int | None = None, **labels) -> Field:
    components = dict(
        values=np.zeros(4),
        parameter={"variable": param},
        time={"valid_datetime": datetime.datetime(2020, 1, 1)},
        geography={"latitudes": np.arange(4.0), "longitudes": np.arange(4.0)},
    )
    if level is not None:
        components["vertical"] = {"level": level, "level_type": "pressure"}
    if labels:
        components["labels"] = labels
    return Field.from_components(**components)


# ---------------------------------------------------------------------------
# create_naming: configuration forms
# ---------------------------------------------------------------------------


def test_create_naming_default_is_param_levelist():
    assert isinstance(create_naming("default"), ParamLevelNaming)
    assert isinstance(create_naming("param_levelist"), ParamLevelNaming)


def test_create_naming_param():
    assert isinstance(create_naming("param"), ParamNaming)


def test_create_naming_template_string():
    naming = create_naming("{param}_{levelist}")
    assert isinstance(naming, TemplateNaming)
    assert naming.template == "{param}_{levelist}"


def test_registered_schemes_are_templates():
    assert create_naming("param").template == "{param}"
    assert create_naming("param_levelist").template == "{param}_{levelist}"


# ---------------------------------------------------------------------------
# Naming semantics
# ---------------------------------------------------------------------------


def test_default_names_level_field():
    naming = create_naming("default")
    assert naming.name(_field("t", level=850)) == "t_850"


def test_default_names_field_without_level():
    # A missing template value drops the preceding separator: "2t", not "2t_".
    naming = create_naming("default")
    assert naming.name(_field("2t")) == "2t"


def test_default_strips_zero_level():
    # In earthkit-data 1.0 surface fields report vertical.level == 0; the
    # "_0" suffix is stripped from the name.
    naming = create_naming("default")
    assert naming.name(_field("10u", level=0)) == "10u"


def test_default_strips_computed_forcing_level():
    # Computed forcings inherit the level of the field they were templated
    # from; any level suffix is stripped from their name.
    naming = create_naming("default")
    assert naming.name(_field("cos_latitude", level=850)) == "cos_latitude"


def test_param_scheme_ignores_level():
    naming = create_naming("param")
    assert naming.name(_field("t", level=850)) == "t"


def test_custom_template_unknown_key_is_dropped_when_missing():
    # Unknown bare keys resolve under metadata.*; a missing value drops the
    # preceding separator.
    naming = create_naming("{param}_{directionNumber}")
    assert naming.name(_field("swh", level=None)) == "swh"


# ---------------------------------------------------------------------------
# Attaching names to fieldlists
# ---------------------------------------------------------------------------


def test_naming_attaches_labels_name():
    fields = FieldList.from_fields([_field("t", level=850), _field("2t")])
    named = create_naming("default")(fields)
    assert [f.name for f in named] == ["t_850", "2t"]


def test_naming_respects_existing_name():
    # Fields explicitly renamed (Field.with_name) keep their name.
    renamed = Field.with_name(_field("tp"), "tp_accum_6h")
    named = create_naming("default")(FieldList.from_fields([renamed, _field("2t")]))
    assert [f.name for f in named] == ["tp_accum_6h", "2t"]


def test_field_name_raises_when_unset():
    with pytest.raises(ValueError, match="no name"):
        _field("t", level=850).name


def test_names_survive_sel_and_order_by():
    named = create_naming("default")(FieldList.from_fields([_field("t", level=850), _field("2t")]))
    subset = named.sel(**{"labels.name": "t_850"})
    assert len(subset) == 1 and subset[0].name == "t_850"
    assert [f.name for f in named.order_by("labels.name")] == ["2t", "t_850"]


# ---------------------------------------------------------------------------
# Legacy remapping compatibility
# ---------------------------------------------------------------------------


def test_variable_naming_from_remapping():
    assert variable_naming_from_remapping({"param_level": "{param}_{levelist}"}) == "{param}_{levelist}"
    assert variable_naming_from_remapping({}) is None
    assert variable_naming_from_remapping(None) is None
    # Non-naming synthetic keys carry no naming information.
    assert variable_naming_from_remapping({"traj_point": "{time.base_datetime}_{time.step}"}) is None


def test_create_naming_from_remapping_equivalent_names():
    # The legacy bare-key form and the earthkit 1.0 component-path form
    # must name fields identically.
    bare = create_naming_from_remapping({"param_level": "{param}_{levelist}"})
    paths = create_naming_from_remapping({"param_level": "{parameter.variable}_{vertical.level}"})
    for field in (_field("t", level=850), _field("2t")):
        assert bare.name(field) == paths.name(field)


def test_create_naming_from_remapping_none_cases():
    assert create_naming_from_remapping({}) is None
    assert create_naming_from_remapping(None) is None
