#!/usr/bin/env python
# vim: set fileencoding=utf-8
"""SCIAMACHY level 2 data post processing

Main script for SCIAMACHY orbital retrieval post processing
and data combining (to netcdf).

Copyright (c) 2018 Stefan Bender

This file is part of sciapy.
sciapy is free software: you can redistribute it or modify it
under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.
See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""

from __future__ import absolute_import, division, print_function

import glob
import os
import argparse as ap
import datetime as dt
import logging
from pkg_resources import resource_filename

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interp1d

from astropy import units
from astropy.time import Time
from astropy.utils import iers
import astropy.coordinates as coord

import sciapy.level1c as sl
from sciapy.level2 import density_pp as sd
from sciapy.level2 import scia_akm as sa
from sciapy.level2.igrf import gmag_igrf
from sciapy.level2.aacgm2005 import gmag_aacgm2005
from sciapy import __version__
try:
	from nrlmsise00 import msise_flat as msise
except ImportError:
	msise = None
try:
	from sciapy.level2.noem import noem_cpp
except ImportError:
	noem_cpp = None

F107_FILE = resource_filename("sciapy", "data/indices/f107_noontime_flux_obs.txt")
F107A_FILE = resource_filename("sciapy", "data/indices/f107a_noontime_flux_obs.txt")
AP_FILE = resource_filename("sciapy", "data/indices/spidr_ap_2000-2012.dat")
F107_ADJ_FILE = resource_filename("sciapy", "data/indices/spidr_f107_2000-2012.dat")
KP_FILE = resource_filename("sciapy", "data/indices/spidr_kp_2000-2012.dat")

PHI_FAC = 11.91
LST_FAC = -0.62

iers.conf.auto_download = False
iers.conf.auto_max_age = None
# Use a mirror page for the astrpy time data, see
# https://github.com/mzechmeister/serval/issues/33#issuecomment-551156361
# and
# https://github.com/astropy/astropy/issues/8981#issuecomment-523984247
iers.conf.iers_auto_url = "https://astroconda.org/aux/astropy_mirror/iers_a_1/finals2000A.all"


def solar_zenith_angle(alt, lat, lon, time):
	atime = Time(time)
	loc = coord.EarthLocation.from_geodetic(
		height=alt * units.km,
		lat=lat * units.deg,
		lon=lon * units.deg,
	)
	altaz = coord.AltAz(location=loc, obstime=atime)
	sun = coord.get_sun(atime)
	return sun.transform_to(altaz).zen.value


def read_spectra(year, orbit, spec_base=None, skip_upleg=True):
	"""Read and examine SCIAMACHY orbit spectra

	Reads the limb spactra and extracts the dates, times, latitudes,
	longitudes to be used to re-assess the retrieved geolocations.

	Parameters
	----------
	year: int
		The measurement year to select the corresponding subdir
		below `spec_base` (see below).
	orbit: int
		SCIAMACHY/Envisat orbit number of the spectra.
	spec_base: str, optional
		The root path to the level 1c spectra. Uses the current
		dir if not set or set to `None` (default).
	skip_upleg: bool, optional
		Skip upleg limb scans, i.e. night time scans. For NO retrievals,
		those are not used and should be not used here.
		Default: True

	Returns
	-------
	(dts, times, lats, lons, mlsts, alsts, eotcorr)
	"""
	fail = (None,) * 7
	if spec_base is None:
		spec_base = os.curdir
	spec_path = os.path.join(spec_base, "{0}".format(year))
	spec_path2 = os.path.join(spec_base, "{0}".format(int(year) + 1))
	logging.debug("spec_path: %s", spec_path)
	logging.debug("spec_path2: %s", spec_path)
	if not (os.path.isdir(spec_path) or os.path.isdir(spec_path2)):
		return fail

	# the star stands for the (optional) date subdir
	# to find all spectra for the orbit
	spfiles = glob.glob(
			'{0}/*/SCIA_limb_*_1_0_{1:05d}.dat.l_mpl_binary'
			.format(spec_path, orbit))
	# sometimes for whatever reason the orbit ends up in the wrong year subdir
	# looks in the subdir for the following year as well.
	spfiles += glob.glob(
			'{0}/*/SCIA_limb_*_1_0_{1:05d}.dat.l_mpl_binary'
			.format(spec_path2, orbit))
	if len(spfiles) < 2:
		return fail

	dts = []
	times = []
	lats = []
	lons = []
	mlsts = []
	alsts = []

	sls = sl.scia_limb_scan()
	for f in sorted(spfiles):
		sls.read_from_file(f)
		# copy the values from the l1c file
		lat, lon = sls.cent_lat_lon[:2]
		mlst, alst, eotcorr = sls.local_solar_time(False)
		tp_lats = sls.limb_data.tp_lat
		date = sls.date
		# debug output if requested
		logging.debug("file: %s", f)
		logging.debug("lat: %s, lon: %s", lat, lon)
		logging.debug("mlst: %s, alst: %s, eotcorr: %s", mlst, alst, eotcorr)
		logging.debug("tp_lats: %s", tp_lats)
		logging.debug("date: %s", date)
		if skip_upleg and ((tp_lats[1] - tp_lats[-2]) < 0.5):
			# Exclude non-downleg measurements where the latitude
			# of the last real tangent point (the last is dark sky)
			# is larger than or too close to the first latitude.
			# Requires an (empirical) separation of +0.5 degree.
			logging.debug("excluding upleg point at: %s, %s", lat, lon)
			continue
		dtdate = pd.to_datetime(dt.datetime(*date), utc=True)
		time_hour = dtdate.hour + dtdate.minute / 60.0 + dtdate.second / 3600.0
		logging.debug("mean lst: %s, apparent lst: %s, EoT: %s", mlst, alst, eotcorr)
		dts.append(dtdate)
		times.append(time_hour)
		lats.append(lat)
		lons.append(lon)
		mlsts.append(mlst)
		alsts.append(alst)

	if len(lats) < 2:
		# interpolation will fail with less than 2 points
		return fail

	return (np.asarray(dts),
			np.asarray(times),
			np.asarray(lats),
			np.asarray(lons),
			np.asarray(mlsts),
			np.asarray(alsts), eotcorr)


class _circ_interp(object):
	"""Interpolation on a circle"""
	def __init__(self, x, y, **kw):
		self.c_intpf = interp1d(x, np.cos(y), **kw)
		self.s_intpf = interp1d(x, np.sin(y), **kw)

	def __call__(self, x):
		return np.arctan2(self.s_intpf(x), self.c_intpf(x))


def process_orbit(
	orbit,
	ref_date="2000-01-01",
	dens_path=None,
	spec_base=None,
	use_msis=True,
):
	"""Post process retrieved SCIAMACHY orbit

	Parameters
	----------
	orbit: int
		SCIAMACHY/Envisat orbit number of the results to process.
	ref_date: str, optional
		Base date to calculate the relative days from,
		of the format "%Y-%m-%d". Default: 2000-01-01
	dens_path: str, optional
		The path to the level 2 data. Uses the current
		dir if not set or set to `None` (default).
	spec_base: str, optional
		The root path to the level 1c spectra. Uses the current
		dir if not set or set to `None` (default).

	Returns
	-------
	(dts0, time0, lst0, lon0, sdd): tuple
		dts0 - days since ref_date at equator crossing (float)
		time0 - utc hour into the day at equator crossing (float)
		lst0 - apparent local solar time at the equator (float)
		lon0 - longitude of the equator crossing (float)
		sdd - `scia_density_pp` instance of the post-processed data
	"""
	def _read_gm(fname):
		return dict(np.genfromtxt(fname, usecols=[0, 2], dtype=None))

	fail = (None,) * 5
	logging.debug("processing orbit: %s", orbit)
	dtrefdate = pd.to_datetime(ref_date, format="%Y-%m-%d", utc=True)

	dfiles = glob.glob(
			"{0}/000NO_orbit_{1:05d}_*_Dichten.txt"
			.format(dens_path, orbit))
	if len(dfiles) < 1:
		return fail
	logging.debug("dfiles: %s", dfiles)
	logging.debug("splits: %s", [fn.split('/') for fn in dfiles])
	ddict = dict([(fn, (fn.split('/')[-3:-1] + fn.split('/')[-1].split('_')[3:4]))
				for fn in dfiles])
	logging.debug("ddict: %s", ddict)
	year = ddict[sorted(ddict.keys())[0]][-1][:4]
	logging.debug("year: %s", year)

	dts, times, lats, lons, mlsts, alsts, eotcorr = \
		read_spectra(year, orbit, spec_base)
	if dts is None:
		# return early if reading the spectra failed
		return fail

	dts = pd.to_datetime(dts, utc=True) - dtrefdate
	dts = np.array([dtd.days + dtd.seconds / 86400. for dtd in dts])
	logging.debug("lats: %s, lons: %s, times: %s", lats, lons, times)

	sdd = sd.scia_densities_pp(ref_date=ref_date)
	sdd.read_from_file(dfiles[0])
	logging.debug("density lats: %s, lons: %s", sdd.lats, sdd.lons)

	# Re-interpolates the location (longitude) and times from the
	# limb scan spectra files along the orbit to determine the values
	# at the Equator and to fill in possibly missing data.
	#
	# y values are unit circle angles in radians (0 < phi < 2 pi or -pi < phi < pi)
	# longitudes
	lons_intpf = _circ_interp(
		lats[::-1], np.radians(lons[::-1]),
		fill_value="extrapolate",
	)
	# apparent local solar time (EoT corrected)
	lst_intpf = _circ_interp(
		lats[::-1], np.pi / 12. * alsts[::-1],
		fill_value="extrapolate",
	)
	# mean local solar time
	mst_intpf = _circ_interp(
		lats[::-1], np.pi / 12. * mlsts[::-1],
		fill_value="extrapolate",
	)
	# utc time (day)
	time_intpf = _circ_interp(
		lats[::-1], np.pi / 12. * times[::-1],
		fill_value="extrapolate",
	)
	# datetime
	dts_retr_interpf = interp1d(lats[::-1], dts[::-1], fill_value="extrapolate")

	# equator values
	lon0 = np.degrees(lons_intpf(0.)) % 360.
	lst0 = (lst_intpf(0.) * 12. / np.pi) % 24.
	mst0 = (mst_intpf(0.) * 12. / np.pi) % 24.
	time0 = (time_intpf(0.) * 12. / np.pi) % 24.
	dts_retr_interp0 = dts_retr_interpf(0.)
	logging.debug("utc day at equator: %s", dts_retr_interp0)
	logging.debug("mean LST at equator: %s, apparent LST at equator: %s", mst0, lst0)

	sdd.utchour = (time_intpf(sdd.lats) * 12. / np.pi) % 24.
	sdd.utcdays = dts_retr_interpf(sdd.lats)

	if sdd.lons is None:
		# recalculate the longitudes
		# estimate the equatorial longitude from the
		# limb scan latitudes and longitudes
		lon0s_tp = lons - PHI_FAC * np.tan(np.radians(lats))
		clon0s_tp = np.cos(np.radians(lon0s_tp))
		slon0s_tp = np.sin(np.radians(lon0s_tp))
		lon0_tp = np.arctan2(np.sum(slon0s_tp[1:-1]), np.sum(clon0s_tp[1:-1]))
		lon0_tp = np.degrees((lon0_tp + 2. * np.pi) % (2. * np.pi))
		logging.info("lon0: %s", lon0)
		logging.info("lon0 tp: %s", lon0_tp)
		# interpolate to the retrieval latitudes
		tg_retr_lats = np.tan(np.radians(sdd.lats))
		calc_lons = (tg_retr_lats * PHI_FAC + lon0) % 360.
		calc_lons_tp = (tg_retr_lats * PHI_FAC + lon0_tp) % 360.
		sdd.lons = calc_lons_tp
		logging.debug("(calculated) retrieval lons: %s, %s",
				calc_lons, calc_lons_tp)
	else:
		# sdd.lons = sdd.lons % 360.
		logging.debug("(original) retrieval lons: %s", sdd.lons)

	sdd.mst = (sdd.utchour + sdd.lons / 15.) % 24.
	sdd.lst = sdd.mst + eotcorr / 60.
	mean_alt_km = sdd.alts.mean()

	dt_date_this = dt.timedelta(np.asscalar(dts_retr_interp0)) + dtrefdate
	logging.info("date: %s", dt_date_this)
	# caclulate geomagnetic coordinates
	sdd.gmlats, sdd.gmlons = gmag_igrf(dt_date_this, sdd.lats, sdd.lons, alt=mean_alt_km)
	logging.debug("geomag. lats: %s, lons: %s", sdd.gmlats, sdd.gmlons)
	sdd.aacgmgmlats, sdd.aacgmgmlons = gmag_aacgm2005(sdd.lats, sdd.lons)
	logging.debug("aacgm geomag. lats: %s, lons: %s",
			sdd.aacgmgmlats, sdd.aacgmgmlons)

	# current day for MSIS input
	f107_data = _read_gm(F107_FILE)
	f107a_data = _read_gm(F107A_FILE)
	ap_data = _read_gm(AP_FILE)
	msis_dtdate = dt.timedelta(np.asscalar(dts_retr_interp0)) + dtrefdate
	msis_dtdate1 = msis_dtdate - dt.timedelta(days=1)
	msis_date = msis_dtdate.strftime("%Y-%m-%d").encode()
	msis_date1 = msis_dtdate1.strftime("%Y-%m-%d").encode()
	msis_f107 = f107_data[msis_date1]
	msis_f107a = f107a_data[msis_date]
	msis_ap = ap_data[msis_date]
	logging.debug("MSIS date: %s, f10.7a: %s, f10.7: %s, ap: %s",
			msis_date, msis_f107a, msis_f107, msis_ap)

	# previous day for NOEM input
	f107_adj = _read_gm(F107_ADJ_FILE)
	kp_data = _read_gm(KP_FILE)
	noem_dtdate = dt.timedelta(np.asscalar(dts_retr_interp0) - 1) + dtrefdate
	noem_date = noem_dtdate.strftime("%Y-%m-%d").encode()
	noem_f107 = f107_adj[noem_date]
	noem_kp = kp_data[noem_date]
	logging.debug("NOEM date: %s, f10.7: %s, kp: %s",
			noem_date, noem_f107, noem_kp)

	if sdd.noem_no is None:
		sdd.noem_no = np.zeros_like(sdd.densities)
	if sdd.temperature is None and msise is None:
		sdd.temperature = np.full_like(sdd.densities, np.nan)
	if sdd.sza is None:
		sdd.sza = np.zeros_like(sdd.lats)
	if sdd.akdiag is None:
		sdd.akdiag = np.zeros_like(sdd.densities)
		akm_filename = glob.glob(
				"{0}/000NO_orbit_{1:05d}_*_AKM*"
				.format(dens_path, orbit))[0]
		logging.debug("ak file: %s", akm_filename)
		ak = sa.read_akm(akm_filename, sdd.nalt, sdd.nlat)
		logging.debug("ak data: %s", ak)
		sdd.akdiag = ak.diagonal(axis1=1, axis2=3).diagonal(axis1=0, axis2=1)

	if msise is not None:
		if sdd.temperature is None or use_msis:
			_msis_d_t = msise(
				msis_dtdate,
				sdd.alts[None, :], sdd.lats[:, None], sdd.lons[:, None] % 360.,
				msis_f107a, msis_f107, msis_ap,
				lst=sdd.lst[:, None],
			)
			sdd.temperature = _msis_d_t[:, :, -1]
		if use_msis:
			sdd.dens_tot = np.sum(_msis_d_t[:, :, np.r_[:5, 6:9]], axis=2)
	for i, lat in enumerate(sdd.lats):
		if noem_cpp is not None:
			sdd.noem_no[i] = noem_cpp(noem_date.decode(), sdd.alts,
					[lat], [sdd.lons[i]], noem_f107, noem_kp)[:]
		else:
			sdd.noem_no[i][:] = np.nan
	sdd.sza[:] = solar_zenith_angle(
			mean_alt_km,
			sdd.lats, sdd.lons,
			(pd.to_timedelta(sdd.utcdays, unit="days") + dtrefdate).to_pydatetime(),
	)
	sdd.vmr = sdd.densities / sdd.dens_tot * 1.e9  # ppb
	return dts_retr_interp0, time0, lst0, lon0, sdd


def get_orbits_from_date(date, mlt=False, path=None, L2_version="v6.2"):
	"""Find SCIAMACHY orbits with retrieved data at a date

	Parameters
	----------
	date: str
		The date in the format "%Y-%m-%d".
	mlt: bool, optional
		Look for MLT mode data instead of nominal mode data.
		Increases the heuristics to find all MLT orbits.
	path: str, optional
		The path to the level 2 data. If `None` tries to infer
		the data directory from the L2 version using
		'./*<L2_version>'. Default: None

	Returns
	-------
	orbits: list
		List of found orbits with retrieved data files
	"""
	logging.debug("pre-processing: %s", date)
	if path is None:
		density_base = os.curdir
		path = "{0}/*{1}".format(density_base, L2_version)
		logging.debug("path: %s", path)

	dfiles = glob.glob("{0}/000NO_orbit_*_{1}_Dichten.txt".format(
				path, date.replace("-", "")))
	orbits = sorted([int(os.path.basename(df).split('_')[2]) for df in dfiles])
	if mlt:
		orbits.append(orbits[-1] + 1)
	return orbits


def combine_orbit_data(orbits,
		ref_date="2000-01-01",
		L2_version="v6.2", file_version="2.2",
		dens_path=None, spec_base=None,
		use_xarray=False, save_nc=False):
	"""Combine post-processed SCIAMACHY retrieved orbit data

	Parameters
	----------
	orbits: list
		List of SCIAMACHY/Envisat orbit numbers to process.
	ref_date: str, optional
		Base date to calculate the relative days from,
		of the format "%Y-%m-%d". Default: 2000-01-01
	L2_version: str, optional
		SCIAMACHY level 2 data version to process
	file_version: str, optional
		Postprocessing format version of the output data
	dens_path: str, optional
		The path to the level 2 data. If `None` tries to infer
		the data directory from the L2 version looking for anything
		in the current directory that ends in <L2_version>: './*<L2_version>'.
		Default: None
	spec_base: str, optional
		The root path to the level 1c spectra. Uses the current
		dir if not set or set to `None` (default).
	use_xarray: bool, optional
		Uses xarray (if available) to combine the orbital data.
	save_nc: bool, optional
		Save the intermediate orbit data sets to netcdf files
		for debugging.

	Returns
	-------
	(sdday, sdday_ds): tuple
		`sdday` contains the combined data as a `scia_density_day` instance,
		`sdday_ds` contains the same data as a `xarray.Dataset`.
	"""
	if dens_path is None:
		# try some heuristics
		density_base = os.curdir
		dens_path = "{0}/*{1}".format(density_base, L2_version)

	sdday = sd.scia_density_day(ref_date=ref_date)
	sddayl = []
	sdday_ds = None
	for orbit in sorted(orbits):
		dateo, timeo, lsto, lono, sdens = process_orbit(orbit,
				ref_date=ref_date, dens_path=dens_path, spec_base=spec_base)
		logging.info(
			"orbit: %s, eq. date: %s, eq. hour: %s, eq. app. lst: %s, eq. lon: %s",
			orbit, dateo, timeo, lsto, lono
		)
		if sdens is not None:
			sdens.version = file_version
			sdens.data_version = L2_version
			sdday.append_data(dateo, orbit, timeo, sdens)
			if use_xarray:
				sd_xr = sdens.to_xarray(dateo, orbit)
				if sd_xr is not None:
					logging.debug("orbit %s dataset: %s", orbit, sd_xr)
					sddayl.append(sd_xr)
			if save_nc:
				sdens.write_to_netcdf(sdens.filename[:-3] + "nc")
	if use_xarray and sddayl:
		sdday_ds = xr.concat(sddayl, dim="time")
	return sdday, sdday_ds


VAR_ATTRS = {
	"2.1": {
		"MSIS_Dens": dict(
			units='cm^{-3}',
			long_name='total number density (NRLMSIS-00)',
		),
		"MSIS_Temp": dict(
			units='K',
			long_name='temperature',
			model="NRLMSIS-00",
		),
	},
	"2.2": {
	},
	"2.3": {
		"aacgm_gm_lats": dict(
			long_name='geomagnetic_latitude',
			model='AACGM2005 at 80 km',
			units='degrees_north',
		),
		"aacgm_gm_lons": dict(
			long_name='geomagnetic_longitude',
			model='AACGM2005 at 80 km',
			units='degrees_east',
		),
		"orbit": dict(
			axis='T', calendar='standard',
			long_name='SCIAMACHY/Envisat orbit number',
			standard_name="orbit",
			units='1',
		),
	},
}
VAR_RENAME = {
	"2.1": {
		# Rename to v2.1 variable names
		"MSIS_Dens": "TOT_DENS",
		"MSIS_Temp": "temperature",
	},
	"2.2": {
	},
	"2.3": {
	},
}
FLOAT_VARS = [
	"altitude", "latitude", "longitude",
	"app_LST", "mean_LST", "mean_SZA",
	"aacgm_gm_lats", "aacgm_gm_lons",
	"gm_lats", "gm_lons",
]


def sddata_xr_set_attrs(
	sdday_xr,
	file_version="2.2",
	ref_date="2000-01-01",
	rename=True,
	species="NO",
):
	"""Customize xarray Dataset variables and attributes

	Changes the variable names to match those exported from the
	`scia_density_day` class.

	Parameters
	----------
	sdday_xr: `xarray.Dataset` instance
		The combined dataset.
	file_version: string "major.minor", optional
		The netcdf file datase version, determines some variable
		names and attributes.
	ref_date: str, optional
		Base date to calculate the relative days from,
		of the format "%Y-%m-%d". Default: 2000-01-01
	rename: bool, optional
		Rename the dataset variables to match the
		`scia_density_day` exported ones.
		Default: True
	species: str, optional
		The name of the level 2 species, used to prefix the
		dataset variables to be named <species>_<variable>.
		Default: "NO".
	"""
	if rename:
		sdday_xr = sdday_xr.rename({
			# 2d vars
			"akm_diagonal": "{0}_AKDIAG".format(species),
			"apriori": "{0}_APRIORI".format(species),
			"density": "{0}_DENS".format(species),
			"density_air": "MSIS_Dens",
			"error_meas": "{0}_ERR".format(species),
			"error_tot": "{0}_ETOT".format(species),
			"temperature": "MSIS_Temp",
			"NOEM_density": "{0}_NOEM".format(species),
			"VMR": "{0}_VMR".format(species),
			# 1d vars and dimensions
			"app_lst": "app_LST",
			"mean_lst": "mean_LST",
			"mean_sza": "mean_SZA",
			"utc_hour": "UTC",
		})
	# relative standard deviation
	sdday_xr["{0}_RSTD".format(species)] = 100.0 * np.abs(
			sdday_xr["{0}_ERR".format(species)] / sdday_xr["{0}_DENS".format(species)])
	# fix coordinate attributes
	sdday_xr["time"].attrs = dict(axis='T', standard_name='time',
		calendar='standard', long_name='equatorial crossing time',
		units="days since {0}".format(
			pd.to_datetime(ref_date, utc=True).isoformat(sep=" ")))
	sdday_xr["altitude"].attrs = dict(axis='Z', positive='up',
		long_name='altitude', standard_name='altitude', units='km')
	sdday_xr["latitude"].attrs = dict(axis='Y', long_name='latitude',
		standard_name='latitude', units='degrees_north')
	# Default variable attributes
	sdday_xr["{0}_DENS".format(species)].attrs = {
			"units": "cm^{-3}",
			"long_name": "{0} number density".format(species)}
	sdday_xr["{0}_ERR".format(species)].attrs = {
			"units": "cm^{-3}",
			"long_name": "{0} density measurement error".format(species)}
	sdday_xr["{0}_ETOT".format(species)].attrs = {
			"units": "cm^{-3}",
			"long_name": "{0} density total error".format(species)}
	sdday_xr["{0}_RSTD".format(species)].attrs = dict(
			units='%',
			long_name='{0} relative standard deviation'.format(species))
	sdday_xr["{0}_AKDIAG".format(species)].attrs = dict(
			units='1',
			long_name='{0} averaging kernel diagonal element'.format(species))
	sdday_xr["{0}_APRIORI".format(species)].attrs = dict(
			units='cm^{-3}', long_name='{0} apriori density'.format(species))
	sdday_xr["{0}_NOEM".format(species)].attrs = dict(
			units='cm^{-3}', long_name='NOEM {0} number density'.format(species))
	sdday_xr["{0}_VMR".format(species)].attrs = dict(
			units='ppb', long_name='{0} volume mixing ratio'.format(species))
	sdday_xr["MSIS_Dens"].attrs = dict(units='cm^{-3}',
			long_name='MSIS total number density',
			model="NRLMSIS-00")
	sdday_xr["MSIS_Temp"].attrs = dict(units='K',
			long_name='MSIS temperature',
			model="NRLMSIS-00")
	sdday_xr["longitude"].attrs = dict(long_name='longitude',
			standard_name='longitude', units='degrees_east')
	sdday_xr["app_LST"].attrs = dict(units='hours',
			long_name='apparent local solar time')
	sdday_xr["mean_LST"].attrs = dict(units='hours',
			long_name='mean local solar time')
	sdday_xr["mean_SZA"].attrs = dict(units='degrees',
			long_name='solar zenith angle at mean altitude')
	sdday_xr["UTC"].attrs = dict(units='hours',
			long_name='measurement utc time')
	sdday_xr["utc_days"].attrs = dict(
			units='days since {0}'.format(
				pd.to_datetime(ref_date, utc=True).isoformat(sep=" ")),
			long_name='measurement utc day')
	sdday_xr["gm_lats"].attrs = dict(long_name='geomagnetic_latitude',
			model='IGRF', units='degrees_north')
	sdday_xr["gm_lons"].attrs = dict(long_name='geomagnetic_longitude',
			model='IGRF', units='degrees_east')
	sdday_xr["aacgm_gm_lats"].attrs = dict(long_name='geomagnetic_latitude',
			# model='AACGM2005 80 km',  # v2.3
			model='AACGM',  # v2.1, v2.2
			units='degrees_north')
	sdday_xr["aacgm_gm_lons"].attrs = dict(long_name='geomagnetic_longitude',
			# model='AACGM2005 80 km',  # v2.3
			model='AACGM',  # v2.1, v2.2
			units='degrees_east')
	sdday_xr["orbit"].attrs = dict(
			axis='T', calendar='standard',
			# long_name='SCIAMACHY/Envisat orbit number',  # v2.3
			long_name='orbit',  # v2.1, v2.2
			standard_name="orbit",
			# units='1',  # v2.3
			units='orbit number',  # v2.1, v2.2
	)
	# Overwrite version-specific variable attributes
	for _v, _a in VAR_ATTRS[file_version].items():
		sdday_xr[_v].attrs = _a
	if rename:
		# version specific renaming
		sdday_xr = sdday_xr.rename(VAR_RENAME[file_version])
	if int(file_version.split(".")[0]) < 3:
		# invert latitudes for backwards-compatitbility
		sdday_xr = sdday_xr.sortby("latitude", ascending=False)
	else:
		sdday_xr = sdday_xr.sortby("latitude", ascending=True)

	sdday_xr.attrs["software"] = "sciapy {0}".format(__version__)

	dateo = pd.to_datetime(
			xr.conventions.decode_cf_variable("date", sdday_xr.time).data[0],
			utc=True,
	).strftime("%Y-%m-%d")
	logging.debug("date %s dataset: %s", dateo, sdday_xr)
	return sdday_xr


def main():
	logging.basicConfig(level=logging.WARNING,
			format="[%(levelname)-8s] (%(asctime)s) "
			"%(filename)s:%(lineno)d %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S %z")

	parser = ap.ArgumentParser()
	parser.add_argument("file", default="SCIA_NO.nc",
			help="the filename of the output netcdf file")
	parser.add_argument("-M", "--month", metavar="YEAR-MM",
			help="infer start and end dates for month")
	parser.add_argument("-D", "--date_range", metavar="START_DATE:END_DATE",
			help="colon-separated start and end dates")
	parser.add_argument("-d", "--dates", help="comma-separated list of dates")
	parser.add_argument("-B", "--base_date",
			metavar="YEAR-MM-DD", default="2000-01-01",
			help="Reference date to base the time values (days) on "
			"(default: %(default)s).")
	parser.add_argument("-f", "--orbit_file",
			help="the file containing the input orbits")
	parser.add_argument("-r", "--retrieval_version", default="v6.2",
			help="SCIAMACHY level 2 data version to process")
	parser.add_argument("-R", "--file_version", default="2.2",
			help="Postprocessing format version of the output file")
	parser.add_argument("-A", "--author", default="unknown",
			help="Author of the post-processed data set "
			"(default: %(default)s)")
	parser.add_argument("-p", "--path", default=None,
			help="path containing the L2 data")
	parser.add_argument("-s", "--spectra", default=None, metavar="PATH",
			help="path containing the L1c spectra")
	parser.add_argument("-m", "--mlt", action="store_true", default=False,
			help="indicate nominal (False, default) or MLT data (True)")
	parser.add_argument("-X", "--xarray", action="store_true", default=False,
			help="use xarray to prepare the dataset"
			" (experimental, default %(default)s)")
	loglevels = parser.add_mutually_exclusive_group()
	loglevels.add_argument("-l", "--loglevel", default="WARNING",
			choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
			help="change the loglevel (default: 'WARNING')")
	loglevels.add_argument("-q", "--quiet", action="store_true", default=False,
			help="less output, same as --loglevel=ERROR (default: False)")
	loglevels.add_argument("-v", "--verbose", action="store_true", default=False,
			help="verbose output, same as --loglevel=INFO (default: False)")
	args = parser.parse_args()
	if args.quiet:
		logging.getLogger().setLevel(logging.ERROR)
	elif args.verbose:
		logging.getLogger().setLevel(logging.INFO)
	else:
		logging.getLogger().setLevel(args.loglevel)

	logging.info("processing L2 version: %s", args.retrieval_version)
	logging.info("writing data file version: %s", args.file_version)

	pddrange = []
	if args.month is not None:
		d0 = pd.to_datetime(args.month + "-01", utc=True)
		pddrange.extend(pd.date_range(d0, d0 + pd.tseries.offsets.MonthEnd()))
	if args.date_range is not None:
		pddrange.extend(pd.date_range(*args.date_range.split(':')))
	if args.dates is not None:
		pddrange.extend(pd.to_datetime(args.dates.split(','), utc=True))
	logging.debug("pddrange: %s", pddrange)

	olist = []
	for date in pddrange:
		try:
			olist += get_orbits_from_date(date.strftime("%Y-%m-%d"),
				mlt=args.mlt, path=args.path, L2_version=args.retrieval_version)
		except:  # handle NaT
			pass
	if args.orbit_file is not None:
		olist += np.genfromtxt(args.orbit_file, dtype=np.int32).tolist()
	logging.debug("olist: %s", olist)

	if not olist:
		logging.warn("No orbits to process.")
		return

	sdlist, sdxr_ds = combine_orbit_data(olist,
			ref_date=args.base_date,
			L2_version=args.retrieval_version, file_version=args.file_version,
			dens_path=args.path, spec_base=args.spectra, use_xarray=args.xarray,
			save_nc=False)
	sdlist.author = args.author

	if args.xarray and sdxr_ds is not None:
		sdxr_ds.attrs["author"] = args.author
		sd_xr = sddata_xr_set_attrs(
			sdxr_ds, ref_date=args.base_date,
			rename=True, file_version=args.file_version,
		)
		sd_xr2 = sdlist.to_xarray()
		# Overwrite version-specific variable attributes
		for _v, _a in VAR_ATTRS[args.file_version].items():
			sd_xr2[_v].attrs = _a
		# version specific renaming
		sd_xr2 = sd_xr2.rename(VAR_RENAME[args.file_version])
		logging.debug(sd_xr)
		logging.debug(sd_xr2)
		logging.debug("equal datasets: %s", sd_xr.equals(sd_xr2))
		xr.testing.assert_allclose(sd_xr, sd_xr2)
		if sd_xr2 is not None:
			logging.debug("xarray dataset: %s", sd_xr2)
			sd_xr2.to_netcdf(args.file, unlimited_dims=["time"])
	else:
		if sdlist.no_dens is not None:
			sdlist.write_to_netcdf(args.file)
		else:
			logging.warn("Processed data is empty.")


if __name__ == "__main__":
	logging.warn(
		"%s: This script is deprecated, use 'python -m sciapy.level2.post_process' instead.",
		DeprecationWarning.__name__,
	)
	main()
