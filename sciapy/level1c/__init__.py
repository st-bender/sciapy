"""SCIAMACHY level 1c spectra module

Copyright (c) 2014-2017 Stefan Bender

This module contains the class for SCIAMACHY level 1c spectra.
It supports limb and solar specra.

License
-------
This module is part of sciapy.
sciapy is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, version 2.
See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""

__all__ = ["scia_limb_point", "scia_limb_scan", "scia_solar"]

from .scia_limb import scia_limb_point, scia_limb_scan
from .scia_solar import scia_solar

# un-clutter the namespace
del scia_limb_hdf5
del scia_limb_mpl
del scia_limb_nc
del scia_limb_txt
