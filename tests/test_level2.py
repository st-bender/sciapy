# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2018 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY level 2 tests
"""

import sciapy.level2


def test_module_structure():
	assert sciapy.level2
	assert sciapy.level2.binning
	assert sciapy.level2.binning.bin_lat_timeavg
	assert sciapy.level2.nrlmsise00
	assert sciapy.level2.nrlmsise00.gtd7
	assert sciapy.level2.nrlmsise00.gtd7d
