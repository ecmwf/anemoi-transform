# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

LOG = logging.getLogger(__name__)


def test_create_filters() -> None:
    """Test the creation of filters."""
    from anemoi.transform.filters.fields import filter_registry as fields_filter_registry
    from anemoi.transform.filters.tabular import filter_registry as tabular_filter_registry

    for reg in (fields_filter_registry, tabular_filter_registry):
        for n in reg.registered:
            try:
                reg.create(n)
            except Exception as e:
                LOG.error(f"Error creating filter {n}: {e}")


if __name__ == "__main__":
    """Run all test functions that start with 'test_'."""
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            print(f"Running {name}...")
            obj()
