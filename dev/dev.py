from earthkit.data import from_source

from anemoi.transform.filters import filter_factory

################
data = from_source(
    "mars",
    param=["u", "v", "t", "q"],
    grid=[1, 1],
    date="20200101/to/20200105",
    levelist=[1000, 850, 500],
)
for f in data:
    print(f)

################

uv_2_ddff = filter_factory("uv_2_ddff")

data = uv_2_ddff.forward(data)
for f in data:
    print(f)


ddff_2_uv = filter_factory("ddff_2_uv")
data = ddff_2_uv.forward(data)
for f in data:
    print(f)
