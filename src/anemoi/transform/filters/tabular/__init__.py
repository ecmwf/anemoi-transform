# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import importlib
import pathlib
from abc import ABC
from abc import abstractmethod

import pandas as pd

__all__ = ["create_filter", "TabularFilter"]


def create_filter(name: str, **kwargs):
    return TabularFilter.create_from_registry(name, **kwargs)


class RegistryMixin:
    _registry = {}

    @staticmethod
    def _get_registry_name(cls):
        return cls.__name__.lower()

    def __init_subclass__(cls, **kwargs):
        # registry_name explicitly passed in as None
        if "registry_name" in kwargs and kwargs["registry_name"] is None:
            return
        # use registry_name if provided, else use default from class
        if not (registry_name := kwargs.get("registry_name", None)):
            registry_name = RegistryMixin._get_registry_name(cls)
        cls._registry[registry_name] = cls

    @classmethod
    def create_from_registry(cls, name: str, **kwargs):
        try:
            factory = cls._registry[name]
        except KeyError as e:
            raise KeyError(f"Key '{name}' missing in '{cls.__name__}' registry") from e
        return factory(**kwargs)

    @classmethod
    def list_registry(cls):
        return list(cls._registry.keys())


class TabularFilter(ABC, RegistryMixin, registry_name=None):
    _registry = {}

    def __repr__(self):
        dict_str = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({dict_str})"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.forward(df)

    @abstractmethod
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()


def import_all_modules():
    current_dir = pathlib.Path(__file__).parent
    for file in current_dir.glob("*.py"):
        if file.stem not in ["__init__"]:
            module_name = f"{__package__}.{file.stem}"
            importlib.import_module(module_name)


# import all modules so that all filters are available through the registry
import_all_modules()
