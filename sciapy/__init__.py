# Copyright (c) 2014-2018 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""(Some) SCIAMACHY tools

Tools to handle SCIAMACHY/Envisat level 1c spectral data,
and to process level 2 number density data (only NO for now).
"""
__version__ = "0.0.7"

from . import level1c
from . import level2
from . import regress
