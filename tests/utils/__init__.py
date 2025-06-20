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


def collect_fields_by_param(pipeline):
    fields = defaultdict(list)
    for field in pipeline:
        param = field.metadata("param")
        fields[param].append(field)
    return fields


def assert_fields_equal(field_a, field_b):
    METADATA_KEYS = ["param", "valid_datetime", "latitudes", "longitudes", "levelist"]
    for key in METADATA_KEYS:
        try:
            assert field_a.metadata(key) == field_b.metadata(key)
        except ValueError:
            # if ValueError, assume not just scalar values - use numpy for comparison
            assert np.allclose(field_a.metadata(key), field_b.metadata(key))
        except KeyError:
            continue
    assert np.allclose(field_a.to_numpy(), field_b.to_numpy())
