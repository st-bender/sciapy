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
import os

import numpy as np

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


DATADIR = os.path.join(".", "tests", "data", "l1c")
IFILE = os.path.join(
	DATADIR, "2010", "20100203",
	"SCIA_limb_20100203_015238_1_0_41454.dat.l_mpl_binary",
)


def test_level1c_round_trip_mpl(tmpdir):
	obase = os.path.join(tmpdir, "test_l1c_t")
	ofmpl = obase + ".l_mpl_binary"
	l1c_o = sciapy.level1c.scia_limb_scan()
	l1c_o.read_from_file(IFILE)
	l1c_o.write_to_mpl_binary(ofmpl)
	l1c_t = sciapy.level1c.scia_limb_scan()
	l1c_t.read_from_mpl_binary(ofmpl)
	for _k, _v in l1c_o.__dict__.items():
		assert np.all(_v == l1c_t.__dict__[_k]), "mpl round trip failed"


def test_level1c_round_trip_nc(tmpdir):
	obase = os.path.join(tmpdir, "test_l1c_t")
	ofnc = obase + ".nc"
	l1c_o = sciapy.level1c.scia_limb_scan()
	l1c_o.read_from_file(IFILE)
	l1c_o.write_to_netcdf(ofnc)
	l1c_t = sciapy.level1c.scia_limb_scan()
	l1c_t.read_from_netcdf(ofnc)
	for _k, _v in l1c_o.__dict__.items():
		assert np.all(_v == l1c_t.__dict__[_k]), "netcdf round trip failed"


def test_level1c_round_trip_txt(tmpdir):
	obase = os.path.join(tmpdir, "test_l1c_t")
	oftxt = obase + ".dat"
	l1c_o = sciapy.level1c.scia_limb_scan()
	l1c_o.read_from_file(IFILE)
	l1c_o.write_to_textfile(oftxt)
	l1c_t = sciapy.level1c.scia_limb_scan()
	l1c_t.read_from_textfile(oftxt)
	for _k, _v in l1c_o.__dict__.items():
		assert np.all(_v == l1c_t.__dict__[_k]), "ascii round trip failed"
