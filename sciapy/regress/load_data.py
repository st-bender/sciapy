# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2017-2018 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY data interface

Data loading and selection routines for SCIAMACHY data regression fits.
Includes reading the solar and geomagnetic index files used as proxies.
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from astropy.time import Time

__all__ = ["load_solar_gm_table", "load_scia_dzm"]

_NPV_MAJOR, _NPV_MINOR, _NPV_PATCH = np.__version__.split('.')

_seasons = {
	"summerNH": [
		slice("2002-03-20", "2002-09-25"),
		slice("2003-03-20", "2003-10-13"),
		slice("2004-03-20", "2004-10-20"),
		slice("2005-03-20", "2005-10-20"),
		slice("2006-03-20", "2006-10-20"),
		slice("2007-03-21", "2007-10-20"),
		slice("2008-03-20", "2008-10-20"),
		slice("2009-03-20", "2009-10-20"),
		slice("2010-03-20", "2010-10-20"),
		slice("2011-03-20", "2011-08-31"),
		slice("2012-03-20", "2012-04-07"),
	],
	"summerSH": [
		slice("2002-09-13", "2003-03-26"),
		slice("2003-09-13", "2004-04-01"),
		slice("2004-09-14", "2005-04-02"),
		slice("2005-09-13", "2006-04-02"),
		slice("2006-09-13", "2007-04-02"),
		slice("2007-09-13", "2008-04-01"),
		slice("2008-09-14", "2009-04-02"),
		slice("2009-09-03", "2010-04-02"),
		slice("2010-09-13", "2011-04-02"),
		slice("2011-09-13", "2012-04-01"),
	]
}

_SPEs = [pd.date_range("2002-04-20", "2002-05-01"),
		pd.date_range("2002-05-21", "2002-06-02"),
		pd.date_range("2002-07-15", "2002-07-27"),
		pd.date_range("2002-08-23", "2002-09-03"),
		pd.date_range("2002-09-06", "2002-09-17"),
		pd.date_range("2002-11-08", "2002-11-20"),
		pd.date_range("2003-05-27", "2003-06-07"),
		pd.date_range("2003-10-25", "2003-11-15"),
		pd.date_range("2004-07-24", "2004-08-05"),
		pd.date_range("2004-11-06", "2004-11-18"),
		pd.date_range("2005-01-15", "2005-01-27"),
		pd.date_range("2005-05-13", "2005-05-25"),
		pd.date_range("2005-07-13", "2005-08-05"),
		pd.date_range("2005-08-21", "2005-09-02"),
		pd.date_range("2005-09-07", "2005-09-21"),
		pd.date_range("2006-12-05", "2006-12-23"),
		pd.date_range("2012-01-22", "2012-02-07"),
		pd.date_range("2012-03-06", "2012-03-27")]


def load_dailymeanAE(
		filename="data/indices/AE_Kyoto_1980-2018_daily2_shift12h.dat",
		name="AE", tfmt="jyear"):
	from pkg_resources import resource_filename
	AEfilename = resource_filename("sciapy", filename)
	return load_solar_gm_table(AEfilename, cols=[0, 1],
			names=["time", name], tfmt=tfmt)


def load_dailymeanLya(filename="data/indices/lisird_lya3_1980-2021.dat",
		name="Lya", tfmt="jyear"):
	from pkg_resources import resource_filename
	Lyafilename = resource_filename("sciapy", filename)
	return load_solar_gm_table(Lyafilename, cols=[0, 1],
			names=["time", name], tfmt=tfmt)


def load_solar_gm_table(filename, cols, names, sep="\t", tfmt="jyear"):
	"""Load proxy tables from ascii files

	This function wraps :func:`numpy.genfromtxt()` [#]_ with
	pre-defined settings to match the file format.
	It explicitly returns the times as UTC decimal years or julian epochs.

	.. [#] https://docs.scipy.org/doc/numpy/reference/generated/numpy.genfromtxt.html

	Parameters
	----------
	filename: str
		The file to read
	cols: sequence
		The columns in the file to use, passed to `numpy.genfromtxt`'s
		`usecols` keyword. Should be at least of length 2, indicating
		time values in the first column, e.g., `cols=[0, 1]`.
	names: sequence
		The column names, passed as `names` to `numpy.genfromtxt()`.
		Should be at least of length 2, naming the proxy values in the
		second column, e.g., `names=["time", "proxy"]`.
	sep: str, optional
		The column separator character, passed as `sep`.
		Default: `\t`
	tfmt: str, optional
		The astropy.time "Time Format" for the time units,
		for example, "jyear", "decimalyear", "jd", "mjd", etc.
		See:
		http://docs.astropy.org/en/stable/time/index.html#time-format
		Default: "jyear"

	Returns
	-------
	times: array_like
		The proxy times according to the `tfmt` keyword (UTC) as
		:class:`numpy.ndarray` from the first column of "cols".
	values: tuple of array_like
		The proxy values combined into a structured array
		(:class:`numpy.recarray`) with fields set
		according to "names[1:]" and "cols[1:]" passed above.

	See Also
	--------
	:class:`astropy.time.Time`
	:class:`numpy.recarray`
	"""
	if len(cols) < 2 and len(names) < 2:
		raise ValueError(
				"Both `cols` and `names` should be at least of length 2.")
	encoding = {}
	# set the encoding for numpy >= 1.14.0
	if int(_NPV_MAJOR) >= 1 and int(_NPV_MINOR) >= 14:
		encoding = dict(encoding=None)
	tab = np.genfromtxt(filename,
			delimiter=sep,
			dtype=None,
			names=names,
			usecols=cols,
			**encoding)
	times = Time(tab[names[0]], scale="utc")
	ts = getattr(times, tfmt)
	return ts, tab[names[1:]]


def _greedy_select(ds, size, varname="NO_DENS_std", scale=1.):
	logging.info("Greedy subsampling to size: %s", size)
	var = np.ma.masked_array((ds[varname] * scale)**2, mask=[False])
	var_p = np.ma.masked_array((ds[varname] * scale)**2, mask=[False])
	sigma2_i = scale**2
	sigma2_ip = scale**2
	idxs = np.arange(len(var))
	for _ in range(size):
		max_entr_idx = np.ma.argmax(np.log(1. + var * sigma2_i))
		min_entr_idx = np.ma.argmin(np.log(1. + var_p * sigma2_ip))
		sigma2_i += 1. / var[max_entr_idx]
		sigma2_ip += 1. / var_p[min_entr_idx]
		var.mask[max_entr_idx] = True
		var_p.mask[max_entr_idx] = True
	return ds.isel(time=idxs[var.mask])

def _greedy_idxs_post(x, xerr, size):
	logging.info("Greedy subsampling to size: %s", size)
	var = np.ma.masked_array(xerr**2, mask=[False])
	var_p = np.ma.masked_array(xerr**2, mask=[False])
	sigma2_i = 1.
	sigma2_ip = 1.
	idxs = np.arange(len(var))
	for _ in range(size):
		max_entr_idx = np.ma.argmax(np.log(1. + var * sigma2_i))
		min_entr_idx = np.ma.argmin(np.log(1. + var_p * sigma2_ip))
		sigma2_i += 1. / var[max_entr_idx]
		sigma2_ip += 1. / var_p[min_entr_idx]
		var.mask[max_entr_idx] = True
		var_p.mask[max_entr_idx] = True
	return idxs[var.mask]

def load_scia_dzm(filename, alt, lat, tfmt="jyear",
		scale=1, subsample_factor=1, subsample_method="greedy",
		akd_threshold=0.002, cnt_threshold=0,
		center=False, season=None, SPEs=False):
	"""Load SCIAMACHY daily zonal mean data

	Interface function for SCIAMACHY daily zonal mean data files version 6.x.
	Uses :mod:`xarray` [#]_ to load and select the data. Possible selections are by
	hemispheric summer (NH summer ~ SH winter and SH summer ~ NH winter) and
	exclusion of strong solar proton events (SPE).

	.. [#] https://xarray.pydata.org

	Parameters
	----------
	filename: str
		The input filename
	alt: float
		The altitude
	lat: float
		The longitude
	tfmt: string, optional
		The astropy.time "Time Format" for the time units,
		for example, "jyear", "decimalyear", "jd", "mjd", etc.
		See:
		http://docs.astropy.org/en/stable/time/index.html#time-format
		Default: "jyear"
	scale: float, optional
		Scale factor of the data (default: 1)
	subsample_factor: int, optional
		Factor to subsample the data by (see `subsample_method`)
		(default: 1 (no subsampling))
	subsample_method: "equal", "greedy", or "random", optional
		Method for subsampling the data (see `subsample_factor`).
		"equal" for equally spaced subsampling,
		"greedy" for selecting the data based on their uncertainty,
		and "random" for uniform random subsampling.
		(default: "greedy")
	center: bool, optional
		Center the data by subtracting the global mean.
		(default: False)
	season: "summerNH", "summerSH", or `None`, optional
		Select the named season or `None` for all data (default: None)
	SPEs: bool, optional
		Set to `True` to exclude strong SPE events (default: False)

	Returns
	-------
	(times, dens, errs): tuple of (N,) array_like
		The measurement times according to the `tfmt` keyword,
		the number densities, and their uncertainties.
	"""
	logging.info("Opening dataset: '%s'", filename)
	NO_ds = xr.open_mfdataset(filename, decode_times=False,
				chunks={"time": 400, "latitude": 9, "altitude": 11})
	logging.info("done.")
	# Decode time coordinate for selection
	try:
		# xarray <= 0.9.6
		NO_ds["time"] = xr.conventions.decode_cf_variable(NO_ds.time)
	except TypeError:
		# xarray => 0.10.0
		NO_ds["time"] = xr.conventions.decode_cf_variable("time", NO_ds.time)

	NO_mean = 0.
	if center:
		NO_mean = NO_ds.NO_DENS.mean().values
		logging.info("Centering with global mean: %s", NO_mean)
	NO_tds = NO_ds.sel(latitude=lat, altitude=alt)

	# Exclude SPEs first if requested
	if SPEs:
		logging.info("Removing SPEs.")
		for spe in _SPEs:
			NO_tds = NO_tds.drop(
					NO_tds.sel(time=slice(spe[0], spe[-1])).time.values,
					dim="time")

	# Filter by season
	if season in _seasons.keys():
		logging.info("Restricting to season: %s", season)
		NO_tds = xr.concat([NO_tds.sel(time=s) for s in _seasons[season]],
					dim="time")
		NO_tds.load()
	else:
		logging.info("No season selected or unknown season, "
					"using all available data.")

	try:
		NO_counts = NO_tds.NO_DENS_cnt
	except AttributeError:
		NO_counts = NO_tds.counts

	# Select only useful data
	NO_tds = NO_tds.where(
				np.isfinite(NO_tds.NO_DENS) &
				(NO_tds.NO_DENS_std != 0) &
				(NO_tds.NO_AKDIAG > akd_threshold) &
				(NO_counts > cnt_threshold) &
				(NO_tds.NO_MASK == 0),
				drop=True)

	no_dens = scale * NO_tds.NO_DENS
	if center:
		no_dens -= scale * NO_mean
	no_errs = scale * NO_tds.NO_DENS_std / np.sqrt(NO_counts)
	logging.debug("no_dens.shape (ntime,): %s", no_dens.shape)

	no_sza = NO_tds.mean_SZA

	# Convert to astropy.Time for Julian epoch or decimal year
	if NO_tds.time.size > 0:
		no_t = Time(pd.to_datetime(NO_tds.time.values, utc=True).to_pydatetime(),
					format="datetime", scale="utc")
		no_ys = getattr(no_t, tfmt)
	else:
		no_ys = np.empty_like(NO_tds.time.values, dtype=np.float64)

	if subsample_factor > 1:
		new_data_size = no_dens.shape[0] // subsample_factor
		if subsample_method == "random":
			# random subsampling
			_idxs = np.random.choice(no_dens.shape[0],
					new_data_size, replace=False)
		elif subsample_method == "equal":
			# equally spaced subsampling
			_idxs = slice(0, no_dens.shape[0], subsample_factor)
		else:
			# "greedy" subsampling (default fall-back)
			_idxs = _greedy_idxs_post(no_dens, no_errs, new_data_size)
		return (no_ys[_idxs],
				no_dens.values[_idxs],
				no_errs.values[_idxs],
				no_sza.values[_idxs])

	return no_ys, no_dens.values, no_errs.values, no_sza.values
