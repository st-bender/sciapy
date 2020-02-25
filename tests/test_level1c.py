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
"""SCIAMACHY level 1c tests
"""

import sciapy.level1c


def test_module_object_structure():
	assert sciapy.level1c
	assert sciapy.level1c.scia_limb
	assert sciapy.level1c.scia_limb.scia_limb_point
	assert sciapy.level1c.scia_limb.scia_limb_scan
	assert sciapy.level1c.scia_limb_point
	assert sciapy.level1c.scia_limb_scan
	assert sciapy.level1c.scia_solar


def test_limbmodule_method_structure():
	assert sciapy.level1c.scia_limb.scia_limb_point.from_limb_scan
	assert sciapy.level1c.scia_limb_point.from_limb_scan
	assert sciapy.level1c.scia_limb.scia_limb_scan.assemble_textheader
	assert sciapy.level1c.scia_limb.scia_limb_scan.local_solar_time
	assert sciapy.level1c.scia_limb.scia_limb_scan.parse_textheader
	assert sciapy.level1c.scia_limb.scia_limb_scan.read_from_file
	assert sciapy.level1c.scia_limb.scia_limb_scan.read_from_hdf5
	assert sciapy.level1c.scia_limb.scia_limb_scan.read_from_mpl_binary
	assert sciapy.level1c.scia_limb.scia_limb_scan.read_from_netcdf
	assert sciapy.level1c.scia_limb.scia_limb_scan.read_from_textfile
	assert sciapy.level1c.scia_limb.scia_limb_scan.write_to_mpl_binary
	assert sciapy.level1c.scia_limb.scia_limb_scan.write_to_netcdf
	assert sciapy.level1c.scia_limb.scia_limb_scan.write_to_textfile
	assert sciapy.level1c.scia_limb_scan.assemble_textheader
	assert sciapy.level1c.scia_limb_scan.local_solar_time
	assert sciapy.level1c.scia_limb_scan.parse_textheader
	assert sciapy.level1c.scia_limb_scan.read_from_file
	assert sciapy.level1c.scia_limb_scan.read_from_hdf5
	assert sciapy.level1c.scia_limb_scan.read_from_mpl_binary
	assert sciapy.level1c.scia_limb_scan.read_from_netcdf
	assert sciapy.level1c.scia_limb_scan.read_from_textfile
	assert sciapy.level1c.scia_limb_scan.write_to_mpl_binary
	assert sciapy.level1c.scia_limb_scan.write_to_netcdf
	assert sciapy.level1c.scia_limb_scan.write_to_textfile


def test_solarmodule_method_structure():
	assert sciapy.level1c.scia_solar.read_from_hdf5
	assert sciapy.level1c.scia_solar.read_from_netcdf
	assert sciapy.level1c.scia_solar.read_from_textfile
	assert sciapy.level1c.scia_solar.read_from_file
	assert sciapy.level1c.scia_solar.write_to_netcdf
	assert sciapy.level1c.scia_solar.write_to_textfile
