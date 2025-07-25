#########
 wz_to_w
#########

The ``wz_to_w`` filter converts  geometric vertical velocity (provided in
m/s) to vertical velocity in pressure coordinates (Pa/s). This filter must follow a source that provides the vertical velocity, humidity and temperature.
The hydrostatic hypothesis is used for this conversion.

.. literalinclude:: yaml/wz_to_w.yaml
   :language: yaml