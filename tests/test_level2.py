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
