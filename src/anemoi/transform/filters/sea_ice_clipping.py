from typing import Iterator

import earthkit.data as ekd
import numpy as np

from anemoi.transform.filters import filter_registry
from anemoi.transform.filters.matching import MatchingFieldsFilter
from anemoi.transform.filters.matching import matching

PUNY = 10e-6

@filter_registry.register("sea_ice_velocity_masking")
class SeaIceVelocityMasking(MatchingFieldsFilter):
    """A filter to mask sea ice-related variables when sea ice concentration is low.

    Parameters
    ----------
    siue : str, optional
        The name of the eastward sea ice velocity field, by default "avg_siue".
    sivn : str, optional
        The name of the northward sea ice velocity field, by default "avg_sivn".
    siconc : str, optional
        The name of the sea ice concentration field, by default "avg_siconc".
    icesalt : str, optional
        The name of the sea ice salinity field, by default "icesalt".
    sihc : str, optional
        The name of the sea ice heat content field, by default "sihc".
    snhc : str, optional
        The name of the snow heat content field, by default "snhc".
    sipf : str, optional
        The name of the sea ice pressure field, by default "sipf".
    sitemptop : str, optional
        The name of the sea ice top temperature field, by default "sitemptop".
    sntemp : str, optional
        The name of the snow temperature field, by default "sntemp".
    snvol : str, optional
        The name of the snow volume field, by default "snvol".
    sivol : str, optional
        The name of the sea ice volume field, by default "sivol".
    """

    @matching(
        match="param",
        forward=("siue", "sivn", "siconc", "icesalt", "sihc", "snhc", "sipf", "sitemptop", "sntemp", "snvol", "sivol"),
    )
    def __init__(
        self,
        *,
        siue: str = "avg_siue",
        sivn: str = "avg_sivn",
        siconc: str = "avg_siconc",
        icesalt: str = "icesalt",
        sihc: str = "sihc",
        snhc: str = "snhc",
        sipf: str = "sipf",
        sitemptop: str = "sitemptop",
        sntemp: str = "sntemp",
        snvol: str = "snvol",
        sivol: str = "sivol",
    ) -> None:
        self.siue = siue
        self.sivn = sivn
        self.siconc = siconc
        self.icesalt = icesalt
        self.sihc = sihc
        self.snhc = snhc
        self.sipf = sipf
        self.sitemptop = sitemptop
        self.sntemp = sntemp
        self.snvol = snvol
        self.sivol = sivol

    def forward_transform(
        self,
        siue: ekd.Field,
        sivn: ekd.Field,
        siconc: ekd.Field,
        icesalt: ekd.Field,
        sihc: ekd.Field,
        snhc: ekd.Field,
        sipf: ekd.Field,
        sitemptop: ekd.Field,
        sntemp: ekd.Field,
        snvol: ekd.Field,
        sivol: ekd.Field,
    ) -> Iterator[ekd.Field]:
        """Mask sea ice-related variables where concentration is low.

        Parameters
        ----------
        siue : ekd.Field
            Eastward sea ice velocity.
        sivn : ekd.Field
            Northward sea ice velocity.
        siconc : ekd.Field
            Sea ice concentration.
        icesalt : ekd.Field
            Sea ice salinity.
        sihc : ekd.Field
            Sea ice heat content.
        snhc : ekd.Field
            Snow heat content.
        sipf : ekd.Field
            Sea ice pressure field.
        sitemptop : ekd.Field
            Sea ice top temperature.
        sntemp : ekd.Field
            Snow temperature.
        snvol : ekd.Field
            Snow volume.
        sivol : ekd.Field
            Sea ice volume.

        Returns
        -------
        Iterator[ekd.Field]
            Transformed fields with masked values.
        """

        siconc_np = siconc.to_numpy()

        siue_np = siue.to_numpy()
        sivn_np = sivn.to_numpy()
        icesalt_np = icesalt.to_numpy()
        sihc_np = sihc.to_numpy()
        snhc_np = snhc.to_numpy()
        sipf_np = sipf.to_numpy()
        sitemptop_np = sitemptop.to_numpy()
        sntemp_np = sntemp.to_numpy()
        snvol_np = snvol.to_numpy()
        sivol_np = sivol.to_numpy()

        siue_np[siconc_np <= PUNY] = 0
        sivn_np[siconc_np <= PUNY] = 0
        icesalt_np[siconc_np <= PUNY] = 0
        sihc_np[siconc_np <= PUNY] = 0
        snhc_np[siconc_np <= PUNY] = 0
        sipf_np[siconc_np <= PUNY] = 0
        sitemptop_np[siconc_np <= PUNY] = 0
        sntemp_np[siconc_np <= PUNY] = 0
        snvol_np[siconc_np <= PUNY] = 0
        sivol_np[siconc_np <= PUNY] = 0

        # Additional cleanup of interpolation artefacts

        sihc_np[sihc_np >= -PUNY] = 0
        snhc_np[snhc_np >= -PUNY] = 0

        # Convert snow temperature in K to solve archiving error in ORAS6

        sntemp_np = sntemp_np + 273.15

        yield self.new_field_from_numpy(siue_np, template=siue, param=self.siue)
        yield self.new_field_from_numpy(sivn_np, template=sivn, param=self.sivn)
        yield self.new_field_from_numpy(icesalt_np, template=icesalt, param=self.icesalt)
        yield self.new_field_from_numpy(sihc_np, template=sihc, param=self.sihc)
        yield self.new_field_from_numpy(snhc_np, template=snhc, param=self.snhc)
        yield self.new_field_from_numpy(sipf_np, template=sipf, param=self.sipf)
        yield self.new_field_from_numpy(sitemptop_np, template=sitemptop, param=self.sitemptop)
        yield self.new_field_from_numpy(sntemp_np, template=sntemp, param=self.sntemp)
        yield self.new_field_from_numpy(snvol_np, template=snvol, param=self.snvol)
        yield self.new_field_from_numpy(sivol_np, template=sivol, param=self.sivol)

