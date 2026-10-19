# (C) Copyright 2025-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from anemoi.transform import Field
from anemoi.transform.fields import metadata_key
from anemoi.transform.filter import SingleFieldFilter
from anemoi.transform.filters.fields import filter_registry


@filter_registry.register("set_metadata")
class SetMetadata(SingleFieldFilter):
    """Set metadata keys of the selected fields to the given values.

    Fields matching the ``selection`` (all fields when no selection is
    given) are rebuilt with the ``metadata`` dictionary applied; the other
    fields pass through unchanged. Keys are earthkit-data 1.0 component
    paths (legacy keys such as ``param`` or ``levelist`` are translated
    with a deprecation warning). To rename a variable, use the ``rename``
    filter instead: it attaches the field name (the ``labels.name`` label)
    rather than overwriting metadata.

    Examples
    --------

    .. code-block:: yaml

      input:
        pipe:
          - grib:
              path: /path/to/file.grib
          - set_metadata:
              selection:
                parameter.variable: [2t, skt]
              metadata:
                parameter.units: K
                vertical.level_type: sfc
    """

    required_inputs = ("metadata",)
    optional_inputs = {"selection": {}}

    def prepare_filter(self) -> None:
        """Validate and translate the ``metadata`` dictionary once, at construction."""
        if not isinstance(self.metadata, dict) or not self.metadata:
            raise ValueError(f"'metadata' must be a non-empty dictionary, got {self.metadata!r}")
        self._metadata = {metadata_key(key): value for key, value in self.metadata.items()}

    def forward_select(self) -> dict:
        """Return the selection constraints, translated to component paths.

        Returns
        -------
        dict
            The ``selection`` input with its keys mapped through
            :func:`anemoi.transform.fields.metadata_key`; empty when no
            selection was given (process all fields).
        """
        return {metadata_key(key): value for key, value in self.selection.items()}

    def forward_transform(self, field: Field) -> Field:
        """Return ``field`` with the configured metadata applied.

        Parameters
        ----------
        field : Field
            The field to modify.

        Returns
        -------
        Field
            The new field carrying the configured metadata.
        """
        return Field.with_new_metadata(field, **self._metadata)
