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
import os

import numpy as np
import pytest
import xarray as xr

import sciapy.level2


def test_module_structure():
	assert sciapy.level2
	assert sciapy.level2.binning
	assert sciapy.level2.binning.bin_lat_timeavg
	assert sciapy.level2.density
	assert sciapy.level2.density.scia_densities


@pytest.fixture(scope="module")
def ds():
	alts = np.r_[80:121:20.]
	lats = np.r_[-85:86:10.]
	lons = np.r_[15:186:10.]
	ts = np.linspace(3686, 3686.5, 2)
	lons_2d = np.vstack([lons + i * 180. for i in range(ts.size)])
	data = np.arange(ts.size * lats.size * alts.size).reshape(
		(ts.size, lats.size, alts.size)
	)
	ds = xr.Dataset(
		{
			"data": (
				["time", "latitude", "altitude"],
				data,
				{"units": "fir mfur μftn$^{-2}$ fth"}
			),
			"longitude": (
				["time", "latitude"],
				lons_2d,
				{"units": "degrees_east"}
			),
		},
		coords={
			"latitude": lats,
			"altitude": alts,
			"time": (
				["time"], ts, {"units": "days since 2000-01-01 00:00:00+00:00"}
			),
		},
	)
	return ds


def test_binning_aw(ds):
	from sciapy.level2.binning import bin_lat_timeavg
	data = ds.data.values
	lons = ds.longitude.values
	nts = ds.time.size
	wgts = np.cos(np.radians(ds.latitude.values))
	wgtbs = wgts[::2] + wgts[1::2]
	# divide each by the bin sum
	wgts[::2] /= wgtbs
	wgts[1::2] /= wgtbs
	data = data * wgts[None, :, None]
	lons = lons * wgts[None, :]
	# "manual" result
	data_avg = 1.0 / nts * (data[:, ::2, :] + data[:, 1::2, :]).sum(axis=0)
	lons_avg = 1.0 / nts * (lons[:, ::2] + lons[:, 1::2]).sum(axis=0)
	# binning function result
	dst_nw = bin_lat_timeavg(ds, bins=np.r_[-90:91:20])
	np.testing.assert_allclose(dst_nw.data.values, data_avg)
	np.testing.assert_allclose(dst_nw.longitude.values, lons_avg)
	assert dst_nw.data.attrs["units"] == "fir mfur μftn$^{-2}$ fth"
	assert dst_nw.longitude.attrs["units"] == "degrees_east"


def test_binning_nw(ds):
	from sciapy.level2.binning import bin_lat_timeavg
	data = ds.data.values
	lons = ds.longitude.values
	nts = ds.time.size
	# "manual" result
	data_avg = 0.5 / nts * (data[:, ::2, :] + data[:, 1::2, :]).sum(axis=0)
	lons_avg = 0.5 / nts * (lons[:, ::2] + lons[:, 1::2]).sum(axis=0)
	# binning function result
	dst_nw = bin_lat_timeavg(ds, bins=np.r_[-90:91:20], area_weighted=False)
	np.testing.assert_allclose(dst_nw.data.values, data_avg)
	np.testing.assert_allclose(dst_nw.longitude.values, lons_avg)
	assert dst_nw.data.attrs["units"] == "fir mfur μftn$^{-2}$ fth"
	assert dst_nw.longitude.attrs["units"] == "degrees_east"


DATADIR = os.path.join(".", "tests", "data", "l2")
IFILE = os.path.join(
	DATADIR,
	"000NO_orbit_41454_20100203_Dichten.txt",
)


def _assert_class_equal(l, r):
	for _k, _l in l.__dict__.items():
		_r = r.__dict__[_k]
		assert np.all(_l == _r), (_k, _l, _r)


@pytest.mark.parametrize(
	"dirname, version",
	[["level2_v1.2.3", "1.2.3"], ["foo", None]],
)
def test_level2_round_trip_nc(tmpdir, dirname, version):
	from sciapy.level2.density import scia_densities
	odir = os.path.join(tmpdir, dirname)
	if not os.path.exists(odir):
		os.makedirs(odir)
	obase = os.path.join(odir, os.path.basename(IFILE))
	ofnc = obase + ".nc"
	l2_o = scia_densities(data_ver=version)
	l2_o.read_from_file(IFILE)
	l2_o.write_to_netcdf(ofnc)
	l2_t = scia_densities()
	l2_t.read_from_netcdf(ofnc)
	_assert_class_equal(l2_o, l2_t)


@pytest.mark.parametrize(
	"dirname, version",
	[["level2_v1.2.3", "1.2.3"], ["foo", None]],
)
def test_level2_round_trip_txt(tmpdir, dirname, version):
	from sciapy.level2.density import scia_densities
	odir = os.path.join(tmpdir, dirname)
	if not os.path.exists(odir):
		os.makedirs(odir)
	obase = os.path.join(odir, os.path.basename(IFILE))
	oftxt = obase + ".txt"
	l2_o = scia_densities(data_ver=version)
	l2_o.read_from_file(IFILE)
	l2_o.write_to_textfile(oftxt)
	l2_t = scia_densities()
	l2_t.read_from_textfile(oftxt)
	_assert_class_equal(l2_o, l2_t)


@pytest.mark.parametrize(
	"version",
	["0.1", "0.8", "0.9"],
)
def test_oldver_round_trip_txt(tmpdir, version):
	from sciapy.level2.density import scia_densities
	odir = os.path.join(tmpdir, "level2_v{0}".format(version))
	if not os.path.exists(odir):
		os.makedirs(odir)
	obase = os.path.join(odir, os.path.basename(IFILE))
	oftxt = obase + ".txt"
	l2_o = scia_densities(data_ver=version)
	l2_o.read_from_file(IFILE[:-4] + "_v{0}.txt".format(version))
	l2_o.write_to_textfile(oftxt)
	l2_t = scia_densities()
	l2_t.read_from_textfile(oftxt)
	_assert_class_equal(l2_o, l2_t)


@pytest.mark.parametrize(
	"version",
	["0.1", "0.8", "0.9"],
)
def test_oldver_round_trip_nc(tmpdir, version):
	from sciapy.level2.density import scia_densities
	odir = os.path.join(tmpdir, "level2_v{0}".format(version))
	if not os.path.exists(odir):
		os.makedirs(odir)
	obase = os.path.join(odir, os.path.basename(IFILE))
	ofnc = obase + ".nc"
	l2_o = scia_densities(data_ver=version)
	l2_o.read_from_file(IFILE[:-4] + "_v{0}.txt".format(version))
	l2_o.write_to_netcdf(ofnc)
	l2_t = scia_densities()
	l2_t.read_from_netcdf(ofnc)
	_assert_class_equal(l2_o, l2_t)
