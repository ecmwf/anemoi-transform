from typing import Iterator, Optional
import numpy as np
import earthkit.data as ekd

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching


@filter_registry.register("clipper")
class Clipper(MatchingFieldsFilter):
    """Clip the values of a single field to a specified range [min_value, max_value].

    Parameters
    ----------
    param : str
        Name of the field to clip.
    min_value : float, optional
        Minimum allowed value. Values below this will be set to min_value.
    max_value : float, optional
        Maximum allowed value. Values above this will be set to max_value.
    """

    @matching(select="param", forward=("param",))
    def __init__(
        self,
        *,
        param: str,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        if min_value is None and max_value is None:
            raise ValueError("At least one of min_value or max_value must be specified.")
        self.param = param
        self.min_value = min_value
        self.max_value = max_value

    def forward_transform(self, param: ekd.Field) -> Iterator[ekd.Field]:
        data = param.to_numpy()
        clipped = np.clip(data, self.min_value, self.max_value)
        yield self.new_field_from_numpy(clipped, template=param, param=self.param)
