from __future__ import annotations

from typing import Dict
from typing import Iterable
from typing import List

import earthkit.data as ekd
import numpy as np

from anemoi.transform.fields import new_field_from_numpy
from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.filter import Filter
from anemoi.transform.filters import filter_registry


@filter_registry.register("accum_to_interval")
class AccumToInterval(Filter):
    """Convert accumulated-from-start fields into interval accumulations by time differencing.

    This filter:
    - Works per variable (grouped by param, level, levelType) along valid_datetime.
    - Sorts inputs by valid_datetime inside each group before differencing.
    - For the first step, sets zero if `zero_left=True` (default); otherwise keeps the first step unchanged.
    - Passes through non-target variables unchanged.

    Notes
    -----
    - Target variables are matched by their GRIB `param` (e.g. "tp"), not by `param`.
    - Grouping is done by `(param, level, levelType)`, so both variables are considered unique across surface and model/pressure levels.

    Examples
    --------
    .. code-block:: yaml

        input:
          pipe:
            - source:   # e.g. mars / file / netcdf
                param: [tp, t]
            - accum_to_interval:
                variables: ["tp"]  # convert accumulated total precipitation to interval totals
                zero_left: true    # set the first interval to zero

    """

    def __init__(
        self,
        variables: Iterable[str],
        window: str | None = None,  # accepted for YAML, not required by the algorithm
        zero_left: bool = True,
        **kwargs,
    ) -> None:
        self.variables = set(variables)
        self.zero_left = bool(zero_left)
        self.window = window  # kept for API completeness

    def _identifier(self, f):
        # Build a unique key for time series: (name, level)
        param = f.metadata("param")
        level = f.metadata("level", default=None)
        levelType = f.metadata("levelType", default=None)
        return (param, level, levelType)

    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        # Group by identifier (name + level) so it works for sfc and pl/ml variables
        groups: Dict[tuple, List[ekd.Field]] = {}
        for f in fields:
            groups.setdefault(self._identifier(f), []).append(f)

        # Sort each group by valid time
        for k, fl in groups.items():
            groups[k] = sorted(fl, key=lambda x: x.metadata("valid_datetime"))

        out: List[ekd.Field] = []
        for (param_name, level, level_type), fl in groups.items():
            # Only transform targeted variables; pass-through others untouched
            if param_name not in self.variables or len(fl) == 0:
                out.extend(fl)
                continue

            # Convert accum-from-start → interval accumulations by differencing
            if self.zero_left:
                out.append(new_field_from_numpy(np.zeros_like(fl[0].to_numpy()), template=fl[0]))
            else:
                out.append(fl[0])

            # subsequent steps via indexing
            for i in range(1, len(fl)):
                out.append(new_field_from_numpy(fl[i].to_numpy() - fl[i - 1].to_numpy(), template=fl[i]))

        return new_fieldlist_from_list(out)
