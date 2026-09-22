# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import earthkit.data as ekd
import pytest

from anemoi.transform.flavour import RuleBasedFlavour


@pytest.fixture
def sample_field():
    return ekd.from_source("sample", "test.grib").to_fieldlist()[0]


@pytest.fixture
def sample_fieldlist():
    return ekd.from_source("sample", "test.grib").to_fieldlist()


def test_flavour_keys():
    """The flavour reports the metadata keys it may override."""
    flavour = RuleBasedFlavour([[{"levtype": "sfc"}, {"levelist": None}]])
    assert list(flavour.keys()) == ["levelist"]


def test_flavour_matching_rule_overrides_metadata(sample_field):
    """A matching rule overrides the field metadata.

    This is the flavour used in the anemoi-datasets documentation and tests:
    surface fields should have no levelist.
    """
    assert sample_field.metadata("levtype") == "sfc"
    assert sample_field.vertical.level() == 0

    flavour = RuleBasedFlavour([[{"levtype": "sfc"}, {"levelist": None}]])
    flavoured = flavour.apply(sample_field)

    assert flavoured.vertical.level() is None
    # unrelated metadata is preserved
    assert flavoured.parameter.variable() == sample_field.parameter.variable()


def test_flavour_can_rename_param(sample_field):
    """A rule matching on raw GRIB keys can override the param."""
    assert sample_field.metadata("paramId") == 167
    assert sample_field.parameter.variable() == "2t"

    flavour = RuleBasedFlavour([[{"paramId": 167}, {"param": "csf"}]])
    flavoured = flavour.apply(sample_field)

    assert flavoured.parameter.variable() == "csf"


def test_flavour_non_matching_rule_leaves_field_unchanged(sample_field):
    """A flavour whose rules do not match returns the field untouched."""
    flavour = RuleBasedFlavour([[{"levtype": "pl"}, {"param": "nope"}]])
    flavoured = flavour.apply(sample_field)

    assert flavoured is sample_field
    assert flavoured.parameter.variable() == "2t"


def test_flavour_map_applies_to_all_fields(sample_fieldlist):
    """map() applies the flavour across a whole fieldlist."""
    flavour = RuleBasedFlavour([[{"levtype": "sfc"}, {"levelist": None}]])
    flavoured = flavour.map(sample_fieldlist)

    assert len(flavoured) == len(sample_fieldlist)
    assert [f.parameter.variable() for f in flavoured] == [f.parameter.variable() for f in sample_fieldlist]
    assert all(f.vertical.level() is None for f in flavoured)


@pytest.fixture
def non_grib_field():
    """A field not backed by GRIB, e.g. one already rebuilt by a filter."""
    import numpy as np

    return ekd.from_source(
        "list-of-dicts",
        [
            {
                "parameter": {"variable": "2t"},
                "data": {"values": np.array([1.0])},
                "geography": {"latitudes": np.array([0.0]), "longitudes": np.array([0.0])},
                "vertical": {"level_type": "surface"},
            }
        ],
    ).to_fieldlist()[0]


def test_flavour_matches_mars_keys_on_non_grib_field(non_grib_field):
    """Rules keyed on MARS-style keys match fields without raw GRIB metadata.

    Such fields have no "param" raw metadata key, only "parameter.variable".
    """
    with pytest.raises(KeyError):
        non_grib_field.metadata("param")

    flavour = RuleBasedFlavour([[{"param": "2t"}, {"levelist": None}]])
    flavoured = flavour.apply(non_grib_field)

    assert flavoured is not non_grib_field
    assert flavoured.vertical.level() is None


def test_flavour_non_matching_mars_key_on_non_grib_field(non_grib_field):
    """A rule that does not match leaves the field untouched."""
    flavour = RuleBasedFlavour([[{"param": "not-2t"}, {"levelist": None}]])
    assert flavour.apply(non_grib_field) is non_grib_field
