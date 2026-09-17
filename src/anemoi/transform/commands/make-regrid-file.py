# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import argparse
import logging
import os

import numpy as np

from anemoi.transform.commands import Command
from anemoi.transform.constants import L_1_degree_earth_arc_length_km as L_1d_km

LOG = logging.getLogger(__name__)


def _path_to_lat_lon(path):
    import earthkit.data as ekd

    ds = ekd.from_source("file", path)
    field = ds[0]
    lat, lon = field.to_latlon()
    return np.asarray(lat).reshape(-1), np.asarray(lon).reshape(-1)
