# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import numpy as np
import pandas as pd

from anemoi.transform.filters.tabular import TabularFilter


def safe_log(x):
    return np.log(x + 1e-10)


def sin_deg(x):
    return np.sin(np.deg2rad(x))


def cos_deg(x):
    return np.cos(np.deg2rad(x))


class ColumnTransformation:
    TRANSFORMATIONS = {
        "log": np.log,
        "log1p": np.log1p,
        "safe_log": safe_log,
        "sqrt": np.sqrt,
        "exp": np.exp,
        "abs": np.abs,
        "sin": np.sin,
        "sin_deg": sin_deg,
        "cos": np.cos,
        "cos_deg": cos_deg,
    }

    def __init__(self, source: tuple[str] | tuple[str, str], target: str, transformation: str):
        self.source = source
        self.target = target
        self.transform = self._to_callable(transformation)

    def _to_callable(self, transformation: str):
        if transformation in self.TRANSFORMATIONS:
            return self.TRANSFORMATIONS[transformation]
        try:
            # TODO: replace eval with safer method
            return eval(transformation)
        except Exception as e:
            raise ValueError(f"Invalid transformation: {transformation}") from e

    def apply(self, df: pd.DataFrame) -> None:
        # inplace operation
        try:
            inputs = (df[col] for col in self.source)
        except KeyError as e:
            raise KeyError(f"DataFrame must contain columns {self.source} for transformation.") from e
        df[self.target] = self.transform(*inputs)


class ApplyColumnTransformations(TabularFilter, registry_name="apply_column_transformations"):
    """Apply mathematical transformations to DataFrame columns (including
    multiple columns).

    The configuration should contain a dictionary mapping target column names to
    the desired transformation, which is a dictionary containing a "function"
    key and optionally a "source_column" key (required for transformations
    requiring multiple columns). Note that using a source key which is different
    to the target key can allow for creating new columns from existing ones.
    Without the source key, for a transformation requiring only a single column,
    the column will not be renamed.

    The single-column transformation must be one of: log, log1p, safe_log, sqrt,
    exp, abs, sin, sin_deg, cos, cos_deg.

    Examples
    --------
    .. code-block:: yaml

      input:
        pipe:
          - source:
              ...
          - apply_column_transformations:
              surface_pressure:
                function: log
              precipitation:
                function: abs
              temperature:
                function: "lambda t: t + 273.15"

        Creating a new column from existing columns.
        .. code-block:: yaml

          input:
            pipe:
              - source:
                  ...
              - apply_column_transformations:
                  temperature_in_kelvin:
                    function: "lambda t: t + 273.15"
                    source_column: temperature_in_celsius

    """

    def __init__(self, **config):
        if not config:
            raise ValueError("No columns to transform were specified.")

        # check config is valid
        for target_column, transform_spec in config.items():
            if not isinstance(transform_spec, dict):
                raise ValueError(f"Invalid transformation specification for column {target_column}: {transform_spec}")
            if "function" not in transform_spec:
                raise ValueError(f"Invalid transformation specification for column {target_column}: {transform_spec}")

        # build transformations
        transformations = []
        for target_column, transform_spec in config.items():
            source = transform_spec.get("source_column", (target_column,))
            if isinstance(source, str):
                source = (source,)

            transformation = ColumnTransformation(
                source=source, target=target_column, transformation=transform_spec["function"]
            )
            transformations.append(transformation)
        self.transformations = transformations

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        df_transformed = df.copy()

        # applied in order, so one transformation can be chained from another
        for transformation in self.transformations:
            transformation.apply(df_transformed)
        return df_transformed
