# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest

from anemoi.transform.filters import filter_registry

from .utils import collect_fields_by_param


@pytest.fixture
def source(test_source):
    return test_source("anemoi-filters/2t-sp.grib")


def test_remove_mars_metadata_all_params(source):
    keys = ["domain", "step"]
    filter = filter_registry.create("remove_metadata", keys=keys)
    pipeline = source | filter

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check the number of fields for each param is unchanged
    assert set(input_fields) == {"2t", "sp"}
    assert set(input_fields) == set(output_fields)
    for param in input_fields:
        assert len(input_fields[param]) == len(output_fields[param])

    # check all keys exist in input
    for param, fields in input_fields.items():
        for field in fields:
            keys_exist = (k in field.metadata(namespace="mars") for k in keys)
            assert all(keys_exist)

    # check keys are removed for all fields
    for param, fields in output_fields.items():
        for field in fields:
            keys_exist = (k in field.metadata(namespace="mars") for k in keys)
            assert not any(keys_exist)


def test_remove_mars_metadata_single_param(source):
    keys = ["domain", "step"]
    filter = filter_registry.create("remove_metadata", param="2t", keys=keys)
    pipeline = source | filter

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check the number of fields for each param is unchanged
    assert set(input_fields) == {"2t", "sp"}
    assert set(input_fields) == set(output_fields)
    for param in input_fields:
        assert len(input_fields[param]) == len(output_fields[param])

    # check all keys exist in input
    for param, fields in input_fields.items():
        for field in fields:
            keys_exist = (k in field.metadata(namespace="mars") for k in keys)
            assert all(keys_exist)

    # check keys are removed only for matching param
    for param, fields in output_fields.items():
        for field in fields:
            keys_exist = (k in field.metadata(namespace="mars") for k in keys)
            if param == "2t":
                assert not any(keys_exist)
            else:
                assert all(keys_exist)


def test_remove_mars_metadata_list_params(source):
    keys = ["domain", "step"]
    params = ["2t", "sp"]
    filter = filter_registry.create("remove_metadata", param=params, keys=keys)
    pipeline = source | filter

    input_fields = collect_fields_by_param(source)
    output_fields = collect_fields_by_param(pipeline)

    # check the number of fields for each param is unchanged
    assert set(input_fields) == {"2t", "sp"}
    assert set(input_fields) == set(output_fields)
    for param in input_fields:
        assert len(input_fields[param]) == len(output_fields[param])

    # check all keys exist in input
    for param, fields in input_fields.items():
        for field in fields:
            keys_exist = (k in field.metadata(namespace="mars") for k in keys)
            assert all(keys_exist)

    # check keys are removed for both params
    for param, fields in output_fields.items():
        for field in fields:
            keys_exist = (k in field.metadata(namespace="mars") for k in keys)
            if param in params:
                assert not any(keys_exist)
            else:
                raise ValueError(f"Unexpected param {param}")
