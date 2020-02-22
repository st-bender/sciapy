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

DATADIR = os.path.join(".", "tests", "data")
IFILE1 = os.path.join(DATADIR, "test_v2.2.nc")
IFILE2 = os.path.join(DATADIR, "test_v2.2x.nc")


def test_pp_help():
	p = Popen(["scia_post_process_l2.py", "-h"])
	p.communicate()
	p.wait()
	assert p.returncode == 0


def test_pp_netcdf(tmpdir):
	ofile = os.path.join(tmpdir, "test_v2.2_t.nc")
	p = Popen([
		"scia_post_process_l2.py",
		"-A", "The Dude",
		"-M", "2010-02",
		"-p", os.path.join(DATADIR, "l2"),
		"-s", os.path.join(DATADIR, "l1c"),
		"--mlt",
		ofile,
	])
	p.communicate()
	p.wait()
	assert p.returncode == 0
	ncallclose(IFILE1, ofile)
	nccmpattrs(IFILE1, ofile, ignore=["creation_time"])


@pytest.mark.xfail(
	sys.version_info[:2] == (3, 4),
	reason="netcdf file attributes don't work with Python 3.4 compatible xarray.",
)
def test_pp_xarray(tmpdir):
	ofile = os.path.join(tmpdir, "test_v2.2x_t.nc")
	p = Popen([
		"scia_post_process_l2.py",
		"-A", "The Dude",
		"-M", "2010-02",
		"-p", os.path.join(DATADIR, "l2"),
		"-s", os.path.join(DATADIR, "l1c"),
		"--mlt",
		"-X",
		ofile,
	])
	p.communicate()
	p.wait()
	assert p.returncode == 0
	ncallclose(IFILE2, ofile)
	nccmpattrs(IFILE2, ofile, ignore=["creation_time"])
