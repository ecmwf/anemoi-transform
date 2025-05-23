# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from typing import Union, Generator
import numpy as np

import earthkit.data as ekd
from earthkit.meteo import thermo, vertical

from . import filter_registry
from .matching import MatchingFieldsFilter
from .matching import matching

PREDEFINED_AB = {
    "IFS_137": {
        "A": [
            0.00000000e+00, 2.00036500e+00, 3.10224100e+00, 4.66608400e+00,
            6.82797700e+00, 9.74696600e+00, 1.36054240e+01, 1.86089310e+01,
            2.49857180e+01, 3.29857100e+01, 4.28792420e+01, 5.49554630e+01,
            6.95205760e+01, 8.68958820e+01, 1.07415741e+02, 1.31425507e+02,
            1.59279404e+02, 1.91338562e+02, 2.27968948e+02, 2.69539581e+02,
            3.16420746e+02, 3.68982361e+02, 4.27592499e+02, 4.92616028e+02,
            5.64413452e+02, 6.43339905e+02, 7.29744141e+02, 8.23967834e+02,
            9.26344910e+02, 1.03720117e+03, 1.15685364e+03, 1.28561035e+03,
            1.42377014e+03, 1.57162292e+03, 1.72944898e+03, 1.89751929e+03,
            2.07609595e+03, 2.26543164e+03, 2.46577051e+03, 2.67734814e+03,
            2.90039136e+03, 3.13511938e+03, 3.38174365e+03, 3.64046826e+03,
            3.91149048e+03, 4.19493066e+03, 4.49081738e+03, 4.79914941e+03,
            5.11989502e+03, 5.45299072e+03, 5.79834473e+03, 6.15607422e+03,
            6.52694678e+03, 6.91187061e+03, 7.31186914e+03, 7.72741211e+03,
            8.15935400e+03, 8.60852539e+03, 9.07640039e+03, 9.56268262e+03,
            1.00659785e+04, 1.05846318e+04, 1.11166621e+04, 1.16600674e+04,
            1.22115479e+04, 1.27668730e+04, 1.33246689e+04, 1.38813311e+04,
            1.44321396e+04, 1.49756152e+04, 1.55082568e+04, 1.60261152e+04,
            1.65273223e+04, 1.70087891e+04, 1.74676133e+04, 1.79016211e+04,
            1.83084336e+04, 1.86857188e+04, 1.90312891e+04, 1.93435117e+04,
            1.96200430e+04, 1.98593906e+04, 2.00599316e+04, 2.02196641e+04,
            2.03378633e+04, 2.04123086e+04, 2.04420781e+04, 2.04257188e+04,
            2.03618164e+04, 2.02495117e+04, 2.00870859e+04, 1.98740254e+04,
            1.96085723e+04, 1.92902266e+04, 1.89174609e+04, 1.84897070e+04,
            1.80069258e+04, 1.74718398e+04, 1.68886875e+04, 1.62620469e+04,
            1.55966953e+04, 1.48984531e+04, 1.41733242e+04, 1.34277695e+04,
            1.26682578e+04, 1.19013398e+04, 1.11333047e+04, 1.03701758e+04,
            9.61751562e+03, 8.88045312e+03, 8.16337500e+03, 7.47034375e+03,
            6.80442188e+03, 6.16853125e+03, 5.56438281e+03, 4.99379688e+03,
            4.45737500e+03, 3.95596094e+03, 3.48923438e+03, 3.05726562e+03,
            2.65914062e+03, 2.29424219e+03, 1.96150000e+03, 1.65947656e+03,
            1.38754688e+03, 1.14325000e+03, 9.26507813e+02, 7.34992188e+02,
            5.68062500e+02, 4.24414063e+02, 3.02476563e+02, 2.02484375e+02,
            1.22101563e+02, 6.27812500e+01, 2.28359380e+01, 3.75781300e+00,
            0.00000000e+00, 0.00000000e+00
        ],
        "B": [
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00, 0.00000e+00,
            7.00000e-06, 2.40000e-05, 5.90000e-05, 1.12000e-04, 1.99000e-04,
            3.40000e-04, 5.62000e-04, 8.90000e-04, 1.35300e-03, 1.99200e-03,
            2.85700e-03, 3.97100e-03, 5.37800e-03, 7.13300e-03, 9.26100e-03,
            1.18060e-02, 1.48160e-02, 1.83180e-02, 2.23550e-02, 2.69640e-02,
            3.21760e-02, 3.80260e-02, 4.45480e-02, 5.17730e-02, 5.97280e-02,
            6.84480e-02, 7.79580e-02, 8.82860e-02, 9.94620e-02, 1.11505e-01,
            1.24448e-01, 1.38313e-01, 1.53125e-01, 1.68910e-01, 1.85689e-01,
            2.03491e-01, 2.22333e-01, 2.42244e-01, 2.63242e-01, 2.85354e-01,
            3.08598e-01, 3.32939e-01, 3.58254e-01, 3.84363e-01, 4.11125e-01,
            4.38391e-01, 4.66003e-01, 4.93800e-01, 5.21619e-01, 5.49301e-01,
            5.76692e-01, 6.03648e-01, 6.30036e-01, 6.55736e-01, 6.80643e-01,
            7.04669e-01, 7.27739e-01, 7.49797e-01, 7.70798e-01, 7.90717e-01,
            8.09536e-01, 8.27256e-01, 8.43881e-01, 8.59432e-01, 8.73929e-01,
            8.87408e-01, 8.99900e-01, 9.11448e-01, 9.22096e-01, 9.31881e-01,
            9.40860e-01, 9.49064e-01, 9.56550e-01, 9.63352e-01, 9.69513e-01,
            9.75078e-01, 9.80072e-01, 9.84542e-01, 9.88500e-01, 9.91984e-01,
            9.95003e-01, 9.97630e-01, 1.00000e+0
        ]
    }

}

class HumidityConversionAtHeightLevel(MatchingFieldsFilter):
    """
    A filter to convert specific humidity to relative humidity
    at a specified height level (in meters) with standard thermodynamical formulas/
    """

    @matching(
        select="param",
        forward=(
            "specific_humidity_at_height_level", 
            "temperature_at_height_level", 
            "surface_pressure",
            "specific_humidity_at_model_levels",
            "temperature_at_model_levels",
        ),
        backward=(
            "relative_humidity_at_height_level", 
            "temperature_at_height_level",
            "surface_pressure",
            "specific_humidity_at_model_levels",
            "temperature_at_model_levels",
        ),
    )
    def __init__(
        self,
        *,
        height : float = 2.,
        specific_humidity_at_height_level : str = "2q",
        relative_humidity_at_height_level : str = "2r",
        temperature_at_height_level : str = "2t",
        surface_pressure : str = "sp",
        specific_humidity_at_model_levels : str = "q",
        temperature_at_model_levels : str = "t",
        AB : Union[str,dict] = "IFS_137"
    ):

        self.height = float(height)
        self.q_sl = specific_humidity_at_height_level
        self.rh_sl = relative_humidity_at_height_level
        self.t_sl = temperature_at_height_level
        self.sp = surface_pressure
        self.q_ml = specific_humidity_at_model_levels
        self.t_ml = temperature_at_model_levels

        if isinstance(AB, str):
            AB = AB.upper()
            if AB in PREDEFINED_AB.keys():
                AB = PREDEFINED_AB[AB]
            else:
                KeyError(
                    "%s is not in the list of predefined AB-coefficients. Possible options are %s." % (AB, ", ".join(PREDEFINED_AB.keys()))
                )
        if not isinstance(AB, dict):
            TypeError("AB must be a string or a dictionary.")
        self.A = np.array(AB["A"])
        self.B = np.array(AB["B"])
        
    def forward_transform(
            self,
            specific_humidity_at_height_level: ekd.Field,
            temperature_at_height_level: ekd.Field,
            surface_pressure: ekd.Field,
            specific_humidity_at_model_levels: ekd.FieldList,
            temperature_at_model_levels: ekd.FieldList
        ) -> Generator[ekd.Field, ekd.Field, ekd.Field]:
        """This will return the relative humidity along with temperature from specific humidity and temperature"""

        p_sl = vertical.pressure_at_height_level(
            self.height,
            specific_humidity_at_model_levels.to_numpy(),
            temperature_at_model_levels.to_numpy(),
            surface_pressure.to_numpy(),
            self.A,
            self.B
        )

        # For now We need to go from qv --> td --> rh to take into account
        # the mixed / ice phase when T ~ 0C / T < 0C
        # See https://github.com/ecmwf/earthkit-meteo/issues/15
        td_sl = thermo.dewpoint_from_specific_humidity(
            specific_humidity_at_height_level.to_numpy(),
            p_sl
        )
        rh_sl = thermo.relative_humidity_from_dewpoint(
            temperature_at_height_level,
            td_sl
        )

        yield self.new_field_from_numpy(
            rh_sl,
            template=specific_humidity_at_height_level, 
            param=self.relative_humidity_at_height_level
        )
        yield temperature_at_height_level
        yield specific_humidity_at_height_level

    def backward_transform(self, relative_humidity: ekd.Field, temperature: ekd.Field) -> ekd.Field:
        pass


filter_registry.register("q_2_r_height", HumidityConversionAtHeightLevel)
filter_registry.register("r_2_q_height", HumidityConversionAtHeightLevel.reversed)
