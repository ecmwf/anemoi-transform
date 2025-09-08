# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections import defaultdict

import numpy as np

from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.source import Source


def collect_fields_by_param(pipeline):
    fields = defaultdict(list)
    for field in pipeline:
        param = field.metadata("param")
        fields[param].append(field)
    return fields


def assert_fields_equal(field_a, field_b, exclude_keys=None):
    METADATA_KEYS = ["param", "valid_datetime", "latitudes", "longitudes", "levelist"]

    if exclude_keys is None:
        exclude_keys = []
    exclude_keys = set(exclude_keys)

    # workaround for unreliable __contains__ in potentially wrapped objects
    def metadata_contains(field, key):
        try:
            field.metadata(key)
            return True
        except KeyError:
            return False

    for key in set(METADATA_KEYS) - exclude_keys:
        try:
            assert field_a.metadata(key) == field_b.metadata(key)
        except ValueError:
            # if ValueError, assume not just scalar values - use numpy for comparison
            assert np.allclose(field_a.metadata(key), field_b.metadata(key))
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
            if self.params and f.metadata("param") in self.params:
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
            if self.params and f.metadata("param") in self.params:
                fields.append(f)
                params.append(f.metadata("param"))
        for f in self._additional_fields:
            if self.params and f.metadata("param") in self.params and f.metadata("param") not in params:
                fields.append(f)
        return new_fieldlist_from_list(fields)
