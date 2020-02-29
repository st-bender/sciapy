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
"""SCIAMACHY regression module command line interface tests

Test functions to assure that the command line interface works in
most of the cases, if not all.
"""
from os import path
from pytest import mark
from subprocess import Popen

DATA_FILE = path.join(
	".", "tests", "data",
	"scia_mlt_dzmNO_part_2008-2012_v6.2_2.1_akm0.002_geomag10_nw.nc"
)


def test_main_help():
	p = Popen(["python", "-m", "sciapy.regress", "-h"])
	p.communicate()
	p.wait()
	assert p.returncode == 0


@mark.long
def test_main_lin_mean(tmpdir):
	p = Popen(["python", "-m", "sciapy.regress",
			DATA_FILE,
			"-o", tmpdir,
			"-A", "70",
			"-L", "65",
			"-k",
			"-O1",
			"-w", "50",
			"-b", "10",
			"-p", "20",
			"--plot_median",
			"--plot_residuals",
			"--plot_maxlnpres",
			"--random_seed=1234",
			"-q",
			])
	p.communicate()
	p.wait()
	assert p.returncode == 0


@mark.long
def test_main_lin_gp(tmpdir):
	p = Popen(["python", "-m", "sciapy.regress",
			DATA_FILE,
			"-o", tmpdir,
			"-A", "70",
			"-L", "65",
			"-k",
			"-K", "Mat32",
			"-O1",
			"-w", "4",
			"-b", "10",
			"-p", "20",
			"--no-plot_samples",
			"--random_seed=1234",
			"-q",
			])
	p.communicate()
	p.wait()
	assert p.returncode == 0


@mark.long
def test_main_nonlin_gp(tmpdir):
	p = Popen(["python", "-m", "sciapy.regress",
			DATA_FILE,
			"-o", tmpdir,
			"-A", "70",
			"-L", "65",
			"-k",
			"-F", "\"\"",
			"-I", "GM",
			"--fit_annlifetimes", "GM",
			"--positive_proxies", "GM",
			"--lifetime_scan=60",
			"--lifetime_prior=exp",
			"--cnt_threshold=3",
			"--akd_threshold=0.01",
			"-K", "Mat32",
			"-O1",
			"-w", "4",
			"-b", "10",
			"-p", "20",
			"--no-plot_samples",
			"--random_seed=1234",
			"-q",
			])
	p.communicate()
	p.wait()
	assert p.returncode == 0
