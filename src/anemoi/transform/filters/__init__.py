# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Any

from anemoi.utils.registry import Registry

from anemoi.transform.filter import Filter
from anemoi.transform.filters.fields import filter_registry as fields_filter_registry
from anemoi.transform.filters.tabular import filter_registry as tabular_filter_registry

dispatching_filter_registry = Registry(__name__)

filter_registry = fields_filter_registry


def _merge_registries():
    target = filter_registry
    # force loading of the main registry
    _ = target.factories

    SOURCES = (fields_filter_registry, tabular_filter_registry)
    for source in SOURCES:
        for name, factory in source.factories.items():
            try:
                target.register(name, factory, aliases=source.aliases().get(name, None))
            except AssertionError as e:
                raise AssertionError(f"Duplicate filter name: {name} in {source.package} registry") from e


def create_filter_by_name(name: str, *, context: Any = None, **config) -> Filter:
    """Create a filter from the given key and config."""
    filter = filter_registry.create(name, **config)
    filter.context = context
    return filter


def create_filter(context: Any, config: Any) -> Filter:
    """Create a filter from the given configuration.

    Parameters
    ----------
    context : Any
        The context in which the filter is created.
    config : Any
        The configuration for the filter.

    Returns
    -------
    Filter
        The created filter.
    """
    filter = filter_registry.from_config(config)
    filter.context = context
    return filter


__all__ = ["filter_registry", "create_filter", "create_filter_by_name", "dispatching_filter_registry"]
