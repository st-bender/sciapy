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
"""SCIAMACHY regression module data loading tests
"""
from datetime import datetime
from pkg_resources import resource_filename

from astropy.time import Time
import numpy as np
from pytest import mark, raises
try:
	import netCDF4
	NC_EXT = ".nc"
except ImportError:
	NC_EXT = ".nc3"

import sciapy.regress

DZM_FILE = resource_filename(__name__, "sciaNO_20100203_v6.2.1_geogra30" + NC_EXT)

AEdata = [
		("2000-01-01", "jyear", 1999.9986311, 507.208333),
		("2000-01-01", "jd", 2451544.5, 507.208333),
		("2000-01-01", "mjd", 51544.0, 507.208333),
		("2007-07-01", "jyear", 2007.4948665, 83.208333),
		("2007-07-01", "jd", 2454282.5, 83.208333),
		("2007-07-01", "mjd", 54282.0, 83.208333),
]
Lyadata = [
		("2000-01-01", "jyear", 1999.9986311, 4.59),
		("2000-01-01", "jd", 2451544.5, 4.59),
		("2000-01-01", "mjd", 51544.0, 4.59),
		("2007-07-01", "jyear", 2007.4948665, 3.74),
		("2007-07-01", "jd", 2454282.5, 3.74),
		("2007-07-01", "mjd", 54282.0, 3.74),
]


def test_load_proxyAEfiles():
	AEfile = resource_filename("sciapy",
			"data/indices/AE_Kyoto_1980-2018_daily2_shift12h.dat")
	pAEt, pAEv = sciapy.regress.load_solar_gm_table(AEfile,
			cols=[0, 1], names=["time", "AE"], tfmt="jyear")
	pAEt2, pAEv2 = sciapy.regress.load_data.load_dailymeanAE()
	np.testing.assert_allclose(pAEt, pAEt2)


def test_load_proxyLyafiles():
	Lyafile = resource_filename("sciapy",
			"data/indices/lisird_lya3_1980-2021.dat")
	pLyat, pLyav = sciapy.regress.load_solar_gm_table(Lyafile,
			cols=[0, 1], names=["time", "Lya"], tfmt="jyear")
	pLyat2, pLyav2 = sciapy.regress.load_data.load_dailymeanLya()
	np.testing.assert_allclose(pLyat, pLyat2)


@mark.parametrize("date, tfmt, texp, vexp", AEdata)
def test_load_proxyAEvalues(date, tfmt, texp, vexp):
	pAEt, pAEv = sciapy.regress.load_data.load_dailymeanAE(tfmt=tfmt)
	idx = list(Time(pAEt, format=tfmt).iso).index(date + " 00:00:00.000")
	np.testing.assert_allclose(pAEt[idx], texp)
	np.testing.assert_allclose(pAEv["AE"][idx], vexp)


@mark.parametrize("date, tfmt, texp, vexp", Lyadata)
def test_load_proxyLyavalues(date, tfmt, texp, vexp):
	pLyat, pLyav = sciapy.regress.load_data.load_dailymeanLya(tfmt=tfmt)
	idx = list(Time(pLyat, format=tfmt).iso).index(date + " 00:00:00.000")
	np.testing.assert_allclose(pLyat[idx], texp)
	np.testing.assert_allclose(pLyav["Lya"][idx], vexp)


def test_load_dzm_normal():
	data = sciapy.regress.load_scia_dzm(DZM_FILE, 70., -75.)
	np.testing.assert_allclose(data[0], np.array([2010.09184335]))
	np.testing.assert_allclose(data[1], np.array([25992364.81988303]))
	np.testing.assert_allclose(data[2], np.array([2722294.10593951]))
	np.testing.assert_allclose(data[3], np.array([65.5642548]))


def test_load_dzm_center():
	data = sciapy.regress.load_scia_dzm(DZM_FILE, 70., -45., center=True)
	np.testing.assert_allclose(data[0], np.array([2010.09184335]))
	np.testing.assert_allclose(data[1], np.array([-9293324.84946741]))
	np.testing.assert_allclose(data[2], np.array([2337592.1464543]))
	np.testing.assert_allclose(data[3], np.array([46.02054338]))


def test_load_dzm_spe():
	data = sciapy.regress.load_scia_dzm(DZM_FILE, 70., 15., SPEs=True)
	np.testing.assert_allclose(data[0], np.array([2010.09184335]))
	np.testing.assert_allclose(data[1], np.array([10184144.7669378]))
	np.testing.assert_allclose(data[2], np.array([2633165.7502271]))
	np.testing.assert_allclose(data[3], np.array([45.41338086]))


def test_load_dzm_summerSH():
	data = sciapy.regress.load_scia_dzm(DZM_FILE, 70., 45., season="summerSH")
	np.testing.assert_allclose(data[0], np.array([2010.09184335]))
	np.testing.assert_allclose(data[1], np.array([24484783.29918655]))
	np.testing.assert_allclose(data[2], np.array([2814284.4588219]))
	np.testing.assert_allclose(data[3], np.array([65.04123748]))


def test_load_dzm_summerNH():
	data = sciapy.regress.load_scia_dzm(DZM_FILE, 70., 75., season="summerNH")
	np.testing.assert_equal(data[0], np.array([]))
	np.testing.assert_equal(data[1], np.array([]))
	np.testing.assert_equal(data[2], np.array([]))
	np.testing.assert_equal(data[3], np.array([]))
