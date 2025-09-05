from __future__ import annotations
from typing import Iterable, List, Dict
import numpy as np
import earthkit.data as ekd

from anemoi.transform.fields import new_field_from_numpy, new_fieldlist_from_list
from . import filter_registry  # comes from anemoi.transform.filters

@filter_registry.register("accum_to_interval")
class AccumToInterval:
    """
    Convert accumulated-from-start fields into interval accumulations by time differencing.
    - Works per-variable along valid_datetime.
    - For the first step, sets zero if zero_left=True.
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
        short_name = f.metadata("shortName")
        level = f.metadata("level", default=None)
        levelType = f.metadata("levelType", default=None)
        return (short_name, level, levelType)


    def forward(self, fields: ekd.FieldList) -> ekd.FieldList:
        # Group by identifier (name + level) so it works for sfc and pl/ml variables
        groups: Dict[tuple, List[ekd.Field]] = {}
        for f in fields:
            groups.setdefault(self._identifier(f), []).append(f)

        # Sort each group by valid time
        for k, fl in groups.items():
            groups[k] = sorted(fl, key=lambda x: x.metadata("valid_datetime"))

        out: List[ekd.Field] = []
        for (short_name, level, levelType), fl in groups.items():
            # Only transform targeted variables; pass-through others untouched
            if short_name not in self.variables or len(fl) == 0:
                out.extend(fl)
                continue

            # Convert accum-from-start → interval accumulations by differencing
            for i, f in enumerate(fl):
                if i == 0:
                    if self.zero_left:
                        zero = np.zeros_like(f.to_numpy())
                        out.append(new_field_from_numpy(zero, template=f))
                    else:
                        out.append(f)
                else:
                    prev = fl[i - 1]
                    diff = f.to_numpy() - prev.to_numpy()
                    out.append(new_field_from_numpy(diff, template=f))

        return new_fieldlist_from_list(out)

    def patch_data_request(self, request: dict) -> dict:
        # No changes to requests
        return request