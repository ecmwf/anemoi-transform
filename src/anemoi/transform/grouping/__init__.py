# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from abc import ABC
from collections import defaultdict
from collections.abc import Callable
from collections.abc import Iterator
from typing import Any

import earthkit.data as ekd

LOG = logging.getLogger(__name__)


def _lost(f: Any) -> None:
    """Raise a ValueError indicating a lost field.

    Parameters
    ----------
    f : Any
        The lost field.
    """
    raise ValueError(f"Lost field {f}")


def _flatten(params: list[Any] | tuple[Any, ...]) -> list[str]:
    """Flatten a list of parameters.

    Parameters
    ----------
    params : list or tuple of Any
        List or tuple of parameters to flatten.

    Returns
    -------
    list of str
        Flattened list of parameters.
    """
    flat = []
    for p in params:
        if isinstance(p, (list, tuple)):
            flat.extend(_flatten(p))
        else:
            flat.append(p)
    return flat


class FieldGrouper(ABC):
    def groups(self, fields: list[ekd.Field]) -> dict:
        return self._group_by(self._group_functions, fields)

    @property
    def _group_functions(self):
        """Return a tuple of functions that generate a key (used for grouping) when given a field."""
        raise NotImplementedError

    @staticmethod
    def _group_by(group_fns: tuple[Callable], fields: list[ekd.Field]) -> dict:
        group_fn = group_fns[0]
        groups = defaultdict(list)

        # current group
        for field in fields:
            key = group_fn(field)
            groups[key].append(field)

        # recursively group on remaining group_fns
        if len(group_fns) > 1:
            return {key: FieldGrouper._group_by(group_fns[1:], group) for key, group in groups.items()}
        return dict(groups)


class GroupByParam(FieldGrouper):
    """Group matching fields by parameters name.

    Parameters
    ----------
    params : str | list[str] | tuple[str]
        Parameters to group by.
    """

    def __init__(self, params: str | list[str] | tuple[str]) -> None:
        if not isinstance(params, (list, tuple)):
            params = [params]
        self.params = _flatten(params)

    @property
    def _group_functions(self) -> tuple[Callable, ...]:
        def base_key(field: ekd.Field):
            key = field.metadata(namespace="mars")
            if not key:
                keys = [k for k in field.metadata().keys() if k not in ("latitudes", "longitudes", "values")]
                key = {k: field.metadata(k) for k in keys}
                if not keys:
                    raise NotImplementedError(f"GroupByParam: {field} has no sufficient metadata")
            return key

        def everything_except_param(field: ekd.Field):
            key = base_key(field)
            key.pop("param", None)
            return frozenset(key.items())

        def param(field: ekd.Field):
            key = base_key(field)
            param = key.pop("param", field.metadata("param"))
            return param

        return (
            everything_except_param,
            param,
        )

    def iterate(self, fields: list[ekd.Field], *, other: Callable[[Any], None] = _lost) -> Iterator[tuple[Any, ...]]:
        """Iterate over the data and group fields by parameters.

        Parameters
        ----------
        fields : list of ekd.Field
            List of data fields to group.
        other : callable, optional
            Function to call for fields that do not match the parameters, by default _lost.

        Returns
        -------
        Iterator[tuple[Any, ...]]
            Iterator yielding tuples of grouped fields.
        """
        for group in self.groups(fields).values():
            missing_params = set(self.params) - set(group.keys())
            if missing_params:
                raise ValueError(f"Missing component. Want {sorted(self.params)}, got {sorted(group.keys())}")

            for param, fields in group.items():
                # ensure only one field per param in this group
                assert len(fields) == 1
                # handle unwanted fields
                if param not in self.params:
                    for field in fields:
                        other(field)
                    continue

            yield tuple(group[param][0] for param in self.params)


class GroupByParamVertical(FieldGrouper):
    """Group matching fields by parameter name and vertical level.

    Parameters
    ----------
    params : list of str
        List of parameters to group by.
    """

    def __init__(self, params: str | list[str] | tuple[str]) -> None:
        if not isinstance(params, (list, tuple)):
            params = [params]
        self.params = _flatten(params)

    @property
    def _group_functions(self) -> tuple[Callable, ...]:
        def everything_except_param_and_levels(field: ekd.Field):
            key = field.metadata(namespace="mars")
            if not key:
                keys = [k for k in field.metadata().keys() if k not in ("latitudes", "longitudes", "values")]
                key = {k: field.metadata(k) for k in keys}
                if not keys:
                    raise NotImplementedError(f"GroupByParamVertical: {field} has no sufficient metadata")
            key.pop("param", None)
            key.pop("levtype", None)
            key.pop("levelist", None)
            return frozenset(key.items())

        return (
            everything_except_param_and_levels,
            lambda field: field.metadata("param"),
        )

    def iterate(self, fields: list[ekd.Field], *, other: Callable[[Any], None] = _lost) -> Iterator[tuple[Any, ...]]:
        """Iterate over the data and group fields by parameter and vertical level.

        Parameters
        ----------
        fields : list of ekd.Field
            List of data fields to group.
        other : callable, optional
            Function to call for fields that do not match the parameters, by default _lost.

        Returns
        -------
        Iterator[tuple[Any, ...]]
            Iterator yielding tuples of grouped fields.
        """
        for group in self.groups(fields).values():
            missing_params = set(self.params) - set(group.keys())
            if missing_params:
                raise ValueError(f"Missing component. Want {sorted(self.params)}, got {sorted(group.keys())}")

            for param, fields in group.items():
                # handle unwanted fields
                if param not in self.params:
                    for field in fields:
                        other(field)
                    continue

            yield tuple(group[param] for param in self.params)
