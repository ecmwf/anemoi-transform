######################
 w_to_wz (Reversible)
######################

The ``w_to_wz`` filter converts geometric vertical velocity (provided in
m/s) to vertical velocity in pressure coordinates (Pa/s). This filter
must follow a source that provides the vertical velocity, humidity and
temperature. The hydrostatic hypothesis is used for this conversion.

.. literalinclude:: yaml/w_to_wz.yaml
   :language: yaml
