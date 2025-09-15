#########################
 uv_to_ddff (Reversible)
#########################

The ``uv_to_ddff`` filter converts wind speed and direction from U and V
components, and back. This filter a source that provides the wind (U and
V) components. For details regarding the exact formula used please refer
to `earthkit-meteo
<https://github.com/ecmwf/earthkit-meteo/blob/develop/src/earthkit/meteo/wind/array/wind.py>`_
``xy_to_polar`` formula.

.. literalinclude:: yaml/uv_to_ddff.yaml
   :language: yaml
