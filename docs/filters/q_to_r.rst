#####################
 q_to_r (Reversible)
#####################

The ``q_to_r`` filter converts specific humidity to relative humidity.
This filter must follow a source that provides both specific humidity
and temperature. For details regarding the exact formula used please
refer to `earthkit-meteo
<https://github.com/ecmwf/earthkit-meteo/blob/develop/src/earthkit/meteo/thermo/array/thermo.py>`_
``relative_humidity_from_specific_humidity`` function.

.. literalinclude:: yaml/q_to_r.yaml
   :language: yaml
