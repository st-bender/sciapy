# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2017 Stefan Bender
#
# This file is part of sciapy.
# sciapy is free software: you can redistribute it or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY level 2 binning
"""

__all__ = ["bin_lat_timeavg"]

import logging

import numpy as np
import xarray as xr

def _bin_stats(ds,
		binvar="latitude", tvar="time",
		area_weighted=True, stacked="stacked_time_latitude",
		set_attrs=True):
	"""Helper function for (weighted) bin statistics
	"""
	if not hasattr(ds, stacked):
		stacked = "stacked_{0}_{1}".format(tvar, binvar)
		ds = ds.stack(**{stacked: (tvar, binvar)})
	_weights = (np.cos(np.radians(ds[binvar])) if area_weighted
				else xr.ones_like(ds[binvar]))
	# normalize weights (sum(weights) = 1)
	_weights /= _weights.sum(dim=stacked)
	_ssqw = (_weights**2).sum(dim=stacked)
	mean_ds = (ds * _weights).sum(dim=stacked) if area_weighted \
			else ds.mean(dim=stacked)
	mean_ds["wsqsum"] = _ssqw
	mean_ds["wsqsum"].attrs = {
			"long_name": "sum of squared weights",
			"units": "1"}
	# unbiased standard deviations
	var_ds = ((_weights * (ds - mean_ds)**2).sum(dim=stacked) /
			(1. - _ssqw)) if area_weighted else ds.var(dim=stacked, ddof=1)
	sdev_ds = var_ds.apply(np.sqrt)
	cnts_ds = ds.count(dim=stacked)
	if set_attrs:
		for var in ds.data_vars:
			sdev_ds[var].attrs = ds[var].attrs
			cnts_ds[var].attrs = ds[var].attrs
			cnts_ds[var].attrs.update({"units": "1"})
			for key in ["long_name", "standard_name"]:
				try:
					sdev_ds[var].attrs.update({
						key: ds[var].attrs[key] + " standard deviation"})
					cnts_ds[var].attrs.update({
						key: ds[var].attrs[key] + " counts"})
				except KeyError:
					pass
	sdev_ds = sdev_ds.rename({v: v + "_std" for v in sdev_ds.data_vars})
	cnts_ds = cnts_ds.rename({v: v + "_cnt" for v in cnts_ds.data_vars})
	return xr.merge([mean_ds, sdev_ds, cnts_ds])

def bin_lat_timeavg(ds, binvar="latitude", tvar="time",
		bins=np.r_[-90:91:5], labels=None, area_weighted=True,
		keep_attrs=True,
		load=True, save_progress=False):
	"""Latitudinally bin and time average xarray dataset(s)

	Time-averages the variables in an :class:`xarray.Dataset` in the given
	latitude bins. This should be applied to daily-binned datasets from
	a groupby object (via .apply()) to yield daily zonal means.

	The type of latitudes is selected by passing the appropriate
	`binvar` and must be a variable in the data set.
	Area weighting (cos(latitude)) is also supported.

	Parameters
	----------
	ds : xarray.Dataset or xarray.DatasetGroupBy instance
		The dataset (or GroupBy) instance to bin latitudinally.
	binvar : str
		The name of the variable used for binning, default: "latitude".
	tvar : str
		The name of the time variable of the GroupBy object,
		default: "time".
	bins : numpy.ndarray
		The (latitudinal) bin edges, default: `np.r_[-90:91:5]`.
	labels : list or None
		The bin labels, if set to `None` (the default), the labels are
		set to the central bin values.
	area_weighted : bool
		Use area weighted averages, default: `True`.
	keep_attrs : bool
		Keep the global and variable attributes from the data set,
		default: `True`.
	load : bool
		Loads the data into memory before binning, speeds it up considerably
		provided that the it fits into memory, default: `True`
	save_progress : bool
		Saves the individual binned files to netcdf, to enable recovering from
		interrupted runs, default: `False`

	Returns
	-------
	ds : xarray.Dataset
		The binned and time-averaged dataset together with the (unbiased)
		standard deviations of the variables as `<variable>_std` and the
		number of averaged values as `<variable>_cnt`.
	"""
	if load:
		# load the chunk into memory to speed up binning
		ds.load()
	# adjust the time variable
	if np.issubdtype(ds[tvar].values[0], np.floating):
		# convert floats to datetime first (probably MLT states)
		try:
			# xarray <= 0.9.6
			date = (xr.conventions.decode_cf_variable(ds[tvar])
				.values[0]
				.astype("datetime64[D]"))
		except TypeError:
			# xarray => 0.10.0
			date = (xr.conventions.decode_cf_variable(tvar, ds[tvar])
				.values[0]
				.astype("datetime64[D]"))
	else:
		date = ds[tvar].values[0].astype('datetime64[D]')
	if not hasattr(ds, binvar):
		# nothing to bin
		logging.warn("skipping %s", date)
		return ds
	logging.info("processing %s", date)
	if labels is None:
		labels = 0.5 * (bins[1:] + bins[:-1])
	# stack and bin and delegate to the statistics helper function
	ds_out = (ds.stack(stacked_time_latitude=("time", "latitude"))
				.groupby_bins(binvar, bins, labels=labels)
				.apply(_bin_stats,
					binvar=binvar, tvar=tvar,
					area_weighted=area_weighted,
					set_attrs=keep_attrs))
	if keep_attrs:
		ds_out.attrs = ds.attrs
		for var in ds.data_vars:
			ds_out[var].attrs = ds[var].attrs
	if save_progress:
		ds_out.to_netcdf("tmp_binavg-{0}.nc".format(date))
	return ds_out
