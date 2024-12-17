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
    _class="ea",
    expver="0001",
    stream="oper",
    obsgroup="conv",
    reportype="16001/16002",
    date="20241212",
    type="ofb",
    time="00/06/12/18",
    filter="'select reportype,seqno,date,time,lat,lon,report_status,report_event1,entryno,varno,statid,stalt,obsvalue,lsm@modsurf,biascorr_fg,final_obs_error,datum_status@body,datum_event1@body,vertco_reference_1,vertco_type where ((varno==39 and abs(fg_depar@body)<20) or (varno in (41,42) and abs(fg_depar@body)<15) or (varno==58 and abs(fg_depar@body)<0.4) or (varno == 110 and entryno == 1 and abs(fg_depar@body)<10000)) and time in (000000,030000,060000,090000,120000,150000,180000,210000)'",
)

data = mars.forward(r)

print(data)

################

odb2df = filter_registry.create("reshape_odb_df",
                                predicted_cols=["obsvalue@body"],
                                pivot_cols=["varno@body"],
                                meta_cols=["reportype", "stalt@hdr", "lsm@modsurf"],
                                drop_nans=True)

data = odb2df.forward(data)
print(data)

################

pipeline = workflow_registry.create("pipeline", filters=[mars, odb2df])
print(pipeline)

################

pipeline = r | mars | odb2df
print(pipeline)

# ipipe = pipeline.to_infernece()
