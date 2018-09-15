# Copyright (c) 2017-2018 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY regression modelling

This module contains regression models and methods to analyse
SCIAMACHY level 2 number density data (only NO for now).
"""

from .load_data import *
from .mcmc import *
from .models_cel import *
