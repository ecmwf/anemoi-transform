# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from abc import ABC
from abc import abstractmethod

import pandas as pd
from anemoi.utils.registry import Registry

filter_registry = Registry(__name__)

__all__ = ["create_filter", "filter_registry", "TabularFilter"]


def create_filter(name: str, **kwargs):
    return filter_registry.create(name, **kwargs)


class TabularFilter(ABC):
    def __repr__(self):
        dict_str = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({dict_str})"

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.forward(df)

    @abstractmethod
    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError()
