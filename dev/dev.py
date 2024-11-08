# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.transform.filters import filter_registry
from anemoi.transform.sources import source_registry
from anemoi.transform.workflows import workflow_registry

################

mars = source_registry.create("mars")

r = dict(
    param=["u", "v", "t", "q"],
    grid=[20, 20],
    date="20200101/to/20200105",
    levelist=[1000, 850, 500],
)

data = mars.forward(r)

for f in data:
    print(f)

################

uv_2_ddff = filter_registry.create("uv_2_ddff")

data = uv_2_ddff.forward(data)
for f in data:
    print(f)


ddff_2_uv = filter_registry.create("ddff_2_uv")
data = ddff_2_uv.forward(data)
for f in data:
    print(f)


################

pipeline = workflow_registry.create("pipeline", filters=[mars, uv_2_ddff, ddff_2_uv])
for f in pipeline(r):
    print(f)

################


pipeline = r | mars | uv_2_ddff | ddff_2_uv

for f in pipeline:
    print(f)


# ipipe = pipeline.to_infernece()
