# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2019 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY level 2 post processing command line interface tests

Test functions to assure that the command line interface works in
most of the cases.
"""
import os
import sys
from subprocess import Popen

import pytest
from nccmpx import (ncallclose, nccmpattrs, ncequal, ncidentical)
try:
	import netCDF4
	NC_EXT = ".nc"
	NC_FMT = "-4"
	CDL_EXT = ".cdl"
except ImportError:
	NC_EXT = ".nc3"
	NC_FMT = "-3"
	CDL_EXT = ".cdl3"

DATADIR = os.path.join(".", "tests", "data")
IFILE1 = os.path.join(DATADIR, "test_v{0}" + NC_EXT)
IFILE2 = os.path.join(DATADIR, "test_v{0}x" + NC_EXT)


def _gentestfile(tfile, tmpdir):
	tpath = os.path.join(
		tmpdir,
		os.path.basename(tfile)
	)
	p = Popen([
		"ncgen",
		NC_FMT,
		"-o",
		tpath,
		tfile[:-len(NC_EXT)] + CDL_EXT,
	])
	p.communicate()
	p.wait()
	assert p.returncode == 0
	return tpath


def test_pp_help():
	p = Popen(["scia_post_process_l2.py", "-h"])
	p.communicate()
	p.wait()
	assert p.returncode == 0


@pytest.mark.xfail(
	sys.version_info[:2] == (3, 4),
	reason="netcdf file attributes don't work with Python 3.4 compatible xarray.",
)
@pytest.mark.parametrize("revision", ["2.1", "2.2"])
def test_pp_netcdf(revision, tmpdir):
	ifile = _gentestfile(IFILE1.format(revision), tmpdir)
	ofile = os.path.join(tmpdir, "test_v{0}_t.nc".format(revision))
	p = Popen([
		"scia_post_process_l2.py",
		"-A", "The Dude",
		"-M", "2010-02",
		"-R", revision,
		"-p", os.path.join(DATADIR, "l2"),
		"-s", os.path.join(DATADIR, "l1c"),
		"--mlt",
		ofile,
	])
	p.communicate()
	p.wait()
	assert p.returncode == 0
	ncallclose(ifile, ofile)
	nccmpattrs(ifile, ofile, ignore=["creation_time", "software"])


@pytest.mark.xfail(
	sys.version_info[:2] == (3, 4),
	reason="netcdf file attributes don't work with Python 3.4 compatible xarray.",
)
@pytest.mark.parametrize("revision", ["2.1", "2.2"])
def test_pp_xarray(revision, tmpdir):
	ifile = _gentestfile(IFILE2.format(revision), tmpdir)
	ofile = os.path.join(tmpdir, "test_v{0}x_t.nc".format(revision))
	p = Popen([
		"scia_post_process_l2.py",
		"-A", "The Dude",
		"-M", "2010-02",
		"-R", revision,
		"-p", os.path.join(DATADIR, "l2"),
		"-s", os.path.join(DATADIR, "l1c"),
		"--mlt",
		"-X",
		ofile,
	])
	p.communicate()
	p.wait()
	assert p.returncode == 0
	ncallclose(ifile, ofile)
	nccmpattrs(ifile, ofile, ignore=["creation_time", "software"])
