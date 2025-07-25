########
 r_to_d
########

The ``r_to_d`` filter extracts dewpoint temperature from relative humidity and temperature.
This filter must follow a source that provides relative humidity and temperature.
Geometric vertical velocity is removed by the filter, and pressure
vertical velocity is added.  For details regarding the exact formula used please refer 
to `earthkit-meteo <https://github.com/ecmwf/earthkit-meteo/blob/develop/src/earthkit/meteo/thermo/array/thermo.py>`_ ``dewpoint_from_relative_humidity`` formula.



.. literalinclude:: yaml/r_to_d.yaml
   :language: yaml