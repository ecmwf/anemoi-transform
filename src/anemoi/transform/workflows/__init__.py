# (C) Copyright 2024 European Centre for Medium-Range Weather Forecasts.
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from ..registry import Registry

registry = Registry(__name__)


def register_workflow(name, maker):
    registry.register(name, maker)


def lookup_workflow(name):
    return registry.lookup(name)


def workflow_factory(name, *args, **kwargs):
    return lookup_workflow(name)(*args, **kwargs)
