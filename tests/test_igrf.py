# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2020 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Test IGRF geomagnetic CD and ED coordinates

Centred-iploe (CD) and eccentric-dipole (ED) coordinates
according to Fraser-Smith 1987
"""
import datetime as dt

import numpy as np
import pytest

from sciapy.level2 import igrf


# From Fraser-Smith 1987 and Laundal and Richmond 2017
@pytest.mark.parametrize(
	"date, cd_n, cd_s, dr, dR",
	[
		(
			dt.datetime(1980, 1, 1),
			(78.81, -70.8), None,
			(-385.4, 247.5, 170.2),
			None,
		),
		(
			dt.datetime(1985, 1, 1),
			# (78.98, 289.1), None,
			(78.98, -70.9), None,
			(-391.9, 257.7, 178.9),
			(-399.1, -286.1, 104.6),
		),
		(
			dt.datetime(2015, 1, 1),
			(90 - 9.69, -72.63), (90. - 170.31, 107.37),
			None,
			None,
		),
	]
)
def test_gmpole(date, cd_n, cd_s, dr, dR):
	cd_np, cd_sp, drp, dRp, B0sqp = igrf.gmpole(date)
	if cd_n is not None:
		np.testing.assert_allclose(cd_np, cd_n, atol=0.05)
	if cd_s is not None:
		np.testing.assert_allclose(cd_sp, cd_s, atol=0.05)
	if dr is not None:
		np.testing.assert_allclose(drp, dr, atol=0.81)
	if dR is not None:
		np.testing.assert_allclose(dRp, dR, atol=0.71)
