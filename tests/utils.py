import earthkit.data as ekd

from anemoi.transform.fields import new_fieldlist_from_list
from anemoi.transform.source import Source

ALIAS = ["levelist", "level"]


def get_levelist(v):
    for key in ALIAS:
        try:
            return v.metadata(key)
        except KeyError:
            return ""


def convert_to_dict(output):
    return {f"{v.metadata('param')}{get_levelist(v)}": v.to_numpy() for v in list(output)}


def convert_to_ekd_fieldlist(output):
    ds = ekd.SimpleFieldList()
    for f in output:
        ds.append(f)
    return ds


class ListSource(Source):
    def __init__(self, fields):
        self.fields = convert_to_ekd_fieldlist(fields)

    def forward(self, *args, **kwargs):
        return new_fieldlist_from_list(self.fields)
