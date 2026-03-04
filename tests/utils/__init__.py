# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.source import Source


def collect_fields_by_param(pipeline):
    fields = defaultdict(list)
    for field in pipeline:
        param = field.parameter.variable()
        fields[param].append(field)
    return fields


def assert_fields_equal(field_a, field_b, exclude_keys=None):
    METADATA_KEYS = [
        "parameter.variable",
        "time.valid_datetime",
        "geography.distinct_latitudes",
        "geography.distinct_longitudes",
        "vertical.level",
    ]

    if exclude_keys is None:
        exclude_keys = []
    exclude_keys = set(exclude_keys)

    # TODO: remove this?
    # workaround for unreliable __contains__ in potentially wrapped objects
    def metadata_contains(field, key):
        try:
            field.get(key)
            return True
        except KeyError:
            return False

    for key in set(METADATA_KEYS) - exclude_keys:
        try:
            assert field_a.get(key) == field_b.get(key)
        except ValueError:
            # if ValueError, assume not just scalar values - use numpy for comparison
            assert np.allclose(field_a.get(key), field_b.get(key))
        except KeyError:
            in_a = metadata_contains(field_a, key)
            in_b = metadata_contains(field_b, key)
            if in_a ^ in_b:
                field = "field_a" if in_a else "field_b"
                raise AssertionError(f"Metadata key: {key} only in {field}")
            # not all keys will be in all fields
            continue

    assert np.allclose(field_a.to_numpy(), field_b.to_numpy(), equal_nan=True)


class SelectFieldSource(Source):
    def __init__(self, fields, params=None):
        self._fields = fields
        self.params = params

    def forward(self, *args, **kwargs):
        fields = []
        for f in self._fields:
            if self.params and f.parameter.variable() in self.params:
                fields.append(f)
        return new_fieldlist_from_list(fields)


class SelectAndAddFieldSource(Source):
    def __init__(self, fields, additional_fields, params=None, additional_params=None):
        self._fields = fields
        self._additional_fields = additional_fields
        self.params = params
        if self.params and additional_params:
            self.params += additional_params
        elif additional_params:
            self.params = additional_params

    def forward(self, *args, **kwargs):
        fields = []
        params = []
        for f in self._fields:
            if self.params and f.parameter.variable() in self.params:
                fields.append(f)
                params.append(f.parameter.variable())
        for f in self._additional_fields:
            if self.params and f.parameter.variable() in self.params and f.parameter.variable() not in params:
                fields.append(f)
        return new_fieldlist_from_list(fields)


def compare_npz_files(file1, file2):
    data1 = np.load(file1)
    data2 = np.load(file2)

    assert set(data1.keys()) == set(
        data2.keys()
    ), f"Keys in NPZ files do not match {set(data1.keys())} and {set(data2.keys())}"

    for key in data1.keys():
        assert (data1[key] == data2[key]).all(), f"Data for key {key} does not match between {file1} and {file2}"


def mock_field(**metadata):
    field_spec = {"data.values": np.array([1])} | metadata
    field_spec = group_component_dict(field_spec)
    return ekd.from_source("list-of-dicts", [field_spec]).to_fieldlist()[0]


def create_tabular_filter(name, **kwargs):
    from anemoi.transform.filters.tabular import filter_registry

    return filter_registry.create(name, **kwargs)


def create_fields_filter(name, **kwargs):
    from anemoi.transform.filters.fields import filter_registry

    return filter_registry.create(name, **kwargs)


def create_dispatching_filter(name, **kwargs):
    from anemoi.transform.filters import dispatching_filter_registry as filter_registry

    return filter_registry.create(name, **kwargs)


def group_component_dict(components: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Groups dictionaries in the form {'x.y': 'u', 'x.z': 'v', ...} into {'x': {'y': 'v', 'z': 'v'}, ...}"""

    SEP = "."
    result = {}

    for key, value in components.items():
        if not isinstance(key, str):
            raise TypeError(f"Expected key to be a str, got {type(key)}: {key!r}")

        if key.startswith(SEP):
            raise ValueError(f"Invalid key {key}: cannot start with '{SEP}'")

        head, found, tail = key.partition(SEP)

        if not found:
            # key does not have components - must be a full dict
            if not isinstance(value, dict):
                raise ValueError(f"Value of key {key} must be a dict, got {type(value)}")

            if head in result:
                raise ValueError(f"Duplicate key: {key}")
            result[head] = value
            continue

        # sep was found - therefore key is like "x.", i.e. with no tail
        if not tail:
            raise ValueError(f"Invalid key {key}: empty tail after '{SEP}'")

        # key is in the form "x.y" (assume two levels max)
        if SEP in tail:
            raise ValueError(f"Invalid key: {key}, cannot have more than one '{SEP}'")

        if head not in result:
            result[head] = {}

        if tail in result[head]:
            raise KeyError(f"Duplicate key: {key} already exists")

        result[head][tail] = value
    return result
