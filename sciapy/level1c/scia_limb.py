# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2014-2018 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY level 1c limb spectra module

This module contains the class for SCIAMACHY level 1c limb spectra.
It include some simple conversion routines: from and to ascii,
from and to binary, from and to netcdf.

A simple import from HDF5 files produced by the SRON nadc_tools
(https://github.com/rmvanhees/nadc_tools) is also supported.
"""
from __future__ import absolute_import, division, print_function

import numpy as np

__all__ = ["scia_limb_point", "scia_limb_scan"]

def _equation_of_time(doy):
	"""Equation of time correction for day of year (doy)

	See:
	https://en.wikipedia.org/wiki/Equation_of_time

	Parameters
	----------
	doy: int
		Day of year, Jan 1 = 1

	Returns
	-------
	eot: float
		Equation of time correction in minutes
	"""
	D = doy - 1  # jan 1 = day zero
	W = 360.0 / 365.242
	A = W * (D + 10)
	B = A + 1.914 * np.sin(np.radians(W * (D - 2)))
	C = (np.radians(A) -
			np.arctan2(np.tan(np.radians(B)),
					np.cos(np.radians(23.44)))) / np.pi
	return 720.0 * (C - round(C))

class scia_limb_point(object):
	"""SCIAMACHY limb tangent point data

	Contains the data from a single tangent point."""
	def __init__(self, ls, i):
		self.date = []
		self.npix = 0
		self.orbit = 0
		self.sub_sat_lat = None
		self.sub_sat_lon = None
		self.tp_lat = None
		self.tp_lon = None
		self.tp_alt = None
		self.tp_sza = None
		self.tp_saa = None
		self.tp_los_zenith = None
		self.toa_sza = None
		self.toa_saa = None
		self.toa_los_zenith = None
		self.sat_sza = None
		self.sat_saa = None
		self.sat_los_zenith = None
		self.sat_alt = None
		self.earthradius = None

		self.wls = []
		self.rads = []
		self.errs = []
		self.limb_data = None
		self.from_limb_scan(ls, i)

	def from_limb_scan(self, ls, i):
		"""Import the spectra from a single tangent point of the limb scan

		Parameters
		----------
		ls : :class:`~sciapy.level1c.scia_limb_scan`
			The SCIAMACHY limb scan from which to extract the spectra.
		i : int
			The number of the tangent point in the limb scan
		"""
		self.date = ls.date
		self.npix = ls.npix
		self.orbit = ls.orbit
		if np.any(ls.limb_data.sub_sat_lat):
			self.sub_sat_lat = ls.limb_data.sub_sat_lat[i]
			self.sub_sat_lon = ls.limb_data.sub_sat_lon[i]
		self.tp_lat = ls.limb_data.tp_lat[i]
		self.tp_lon = ls.limb_data.tp_lon[i]
		self.tp_alt = ls.limb_data.tp_alt[i]
		self.tp_sza = ls.limb_data.tp_sza[i]
		self.tp_saa = ls.limb_data.tp_saa[i]
		self.tp_los_zenith = ls.limb_data.tp_los[i]
		self.toa_sza = ls.limb_data.toa_sza[i]
		self.toa_saa = ls.limb_data.toa_saa[i]
		self.toa_los_zenith = ls.limb_data.toa_los[i]
		self.sat_sza = ls.limb_data.sat_sza[i]
		self.sat_saa = ls.limb_data.sat_saa[i]
		self.sat_los_zenith = ls.limb_data.sat_los[i]
		self.sat_alt = ls.limb_data.sat_alt[i]
		self.earthradius = ls.limb_data.earth_rad[i]

		self.wls = ls.wls
		self.rads = ls.limb_data.rad[i]
		self.errs = ls.limb_data.err[i]
		self.limb_data = ls.limb_data[i]


class scia_limb_scan(object):
	"""SCIAMACHY limb scan data

	Contains the data from all or some selected tangent points.
	The format is inspired by the SCIAMACHY ascii data format.

	Attributes
	----------
	textheader_length : int
		The number of lines of the text header.
	textheader : str
		The header containing the limb scan meta data.
	metadata : dict
		Metadata of the limb scan as parsed by
		:func:`parse_textheader`. Contains:

		datatype_txt: str
			The name of the data type.
		l1b_product: str
			The level 1b product which was calibrated.
		orbit: int
			The Envisat/SCIAMACHY orbit number.
		state_id: int
			The SCIAMACHY state_id, denotes the measurement type
			that was carried out, i.e. nominal limb, MLT limb,
			nadir, sun or moon occultation etc.
		software_version: str
			The software used for calibration.
		keyfile_version: str
			The keyfile version used in the calibration process.
		mfactor_version: str
			The M-factor version used in the calibration process.
		init_version: str
			The init version used in the calibration process.
		decont_flags: str
			The decont flags used in the calibration process.
		calibration: str
			The calibrations that were applied to the level 1b data
			to produce the level 1c data.
		date: str
			The measurement data of the limb scan as "%Y%m%d"
		nr_profile: int
			The number of profiles in the scan.
		act_profile: int
			The number of the current profile.
	nalt : int
		The number of tangent points.
	npix : int
		The number of spectral points.
	orbit_state : tuple or list of ints
		Orbit state data containing
		(orbit number, state in orbit, state id,
		number of profiles per state (usually one),
		the actual profile number).
	date : tuple or list of ints
		The limb scan's date (year, month, day, hour, minute, second).
	cent_lat_lon : tuple or list of float
		The centre latitude and longitude of the scan followed by the
		four corner latitude and longitude:
		(lat_centre, lon_centre, lat_corner0, lon_corner0, ...,
		lat_corner3, lon_corner3).
	orbit_phase : float
		The orbital phase of the limb scan.
	wls: (N,) array_like
		The spectral wavelengths.
	limb_data : numpy.recarray
		The limb data containing the following records:

		sub_sat_lat: (M,) array_like
			The latitudes of the satellite ground points (M = nalt).
		sub_sat_lon: (M,) array_like
			The longitudes of the satellite ground points (M = nalt).
		tp_lat: (M,) array_like
			The latitudes of the tangent points (M = nalt).
		tp_lon: (M,) array_like
			The longitudes of the tangent points (M = nalt).
		tp_alt: (M,) array_like
			The tangent altitudes (M = nalt).
		tp_sza: (M,) array_like
			The solar zenith angles at the tangent points (M = nalt).
		tp_saa: (M,) array_like
			The solar azimuth angles at the tangent points (M = nalt).
		tp_los: (M,) array_like
			The line-of-sight zenith angles at the tangent points (M = nalt).
		toa_sza: (M,) array_like
			The solar zenith angles at the top-of-atmosphere points (M = nalt).
		toa_saa: (M,) array_like
			The solar azimuth angles at the top-of-atmosphere points (M = nalt).
		toa_los: (M,) array_like
			The line-of-sight zenith angles at the top-of-atmosphere points (M = nalt).
		sat_sza: (M,) array_like
			The solar zenith angles at the satellite points (M = nalt).
		sat_saa: (M,) array_like
			The solar azimuth angles at the satellite points (M = nalt).
		sat_los: (M,) array_like
			The line-of-sight zenith angles at the satellite points (M = nalt).
		sat_alt: (M,) array_like
			The satellite altitudes (M = nalt).
		earth_rad: (M,) array_like
			The earth radii at the tangent ground points (M = nalt).
		rad: (M, N) array_like
			The radiances at the tangent points, M = nalt, N = len(wls).
		err: (M, N) array_like
			The relative radiance uncertainties at the tangent points,
			M = nalt, N = len(wls).
	"""
	from .scia_limb_nc import read_from_netcdf, write_to_netcdf
	from .scia_limb_txt import read_from_textfile, write_to_textfile
	from .scia_limb_mpl import read_from_mpl_binary, write_to_mpl_binary
	from .scia_limb_hdf5 import (read_hdf5_limb_state_common_data,
								read_hdf5_limb_state_spectral_data,
								read_from_hdf5)

	def __init__(self):
		self._limb_data_dtype = None
		self.textheader_length = 0
		self.textheader = ""
		self.metadata = {}
		self.nalt = 0
		self.npix = 0
		self.orbit_state = []
		(self.orbit, self.state_in_orbit, self.state_id,
			self.profiles_per_state, self.profile_in_state) = (0, 0, 0, 0, 0)
		self.date = []
		self.cent_lat_lon = []
		self.orbit_phase = 0.

		self.limb_data = None

		self.wls = []

	def parse_textheader(self):
		"""Parses the ASCII header metadata

		The ASCII header text contains metadata about the current limb scan.
		This function reads this metadata into the :attr:`metadata` dictionary.
		"""
		from parse import parse
		split_header = self.textheader.split('\n')
		line = 0
		res = parse("#Data type          : {txt}", split_header[line])
		self.metadata["datatype_txt"] = res["txt"]
		line += 1
		res = parse("#L1b product        : {product}", split_header[line])
		self.metadata["l1b_product"] = res["product"]
		line += 1
		res = parse("#Orbit nr.,State ID : {orbit:05d} {state_id:2d}", split_header[line])
		self.metadata["orbit"] = res["orbit"]
		self.metadata["state_id"] = res["state_id"]
		line += 1
		res = parse("#Ver. Proc/Key/M/I/D: {soft}{:s}{key}  {mf}  {init}  {decont}",
				split_header[line])
		self.metadata["software_version"] = res["soft"]
		self.metadata["keyfile_version"] = res["key"]
		self.metadata["mfactor_version"] = res["mf"]
		self.metadata["init_version"] = res["init"]
		self.metadata["decont_flags"] = res["decont"]
		line += 1
		res = parse("#Calibr. appl. (0-8): {cal}", split_header[line])
		self.metadata["calibration"] = res["cal"]
		line += 1
		res = parse("#State Starttime    : {date}", split_header[line])
		self.metadata["date"] = res["date"]
		line += 1
		res = parse("#Nr Profiles / act. : {np:3d} {ap:3d}", split_header[line])
		self.metadata["nr_profile"] = res["np"]
		self.metadata["act_profile"] = res["ap"]

	def assemble_textheader(self):
		"""Combines the metadata to ASCII header

		Tranfers the :attr:`metadata` dictionary back to ASCII form
		for writing to disk.
		"""
		# Prepare the header
		meta = self.metadata
		if not meta:
			return
		n_header = 30
		line = n_header + 2
		header = ("#Data type          : {0[datatype_txt]}\n".format(meta))
		header += ("#L1b product        : {0[l1b_product]}\n".format(meta))
		header += ("#Orbit nr.,State ID : {0:05d} {1:2d}\n".format(meta["orbit"], meta["state_id"]))
		header += ("#Ver. Proc/Key/M/I/D: {0[software_version]:14s}  "
				"{0[keyfile_version]}  {0[mfactor_version]}  "
				"{0[init_version]}  {0[decont_flags]}\n"
				.format(meta))
		header += ("#Calibr. appl. (0-8): {0[calibration]}\n".format(meta))
		header += ("#State Starttime    : {0[date]}\n".format(meta))
		header += ("#Nr Profiles / act. : {0[nr_profile]:3d} {0[act_profile]:3d}\n".format(meta))
		header += ("# Angles TOA\n")
		header += ("#L.{0:2d} : Number_of_altitudes Number_of_pixels\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Orbit State_in_orbit/file State-ID Profiles_per_state Profile_in_State\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Date Time : yyyy mm dd hh mm ss\n".format(line))
		line += 1

		header += ("#L.{0:2d} : Sub satellite point lat\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Sub satellite point lon\n".format(line))
		line += 1
		header += ("#L.{0:2d} : orbit phase [0..1]\n".format(line))
		line += 1

		header += ("#L.{0:2d} : Center(lat/lon) 4*Corners(lat/lon)\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Tangent ground point lat\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Tangent ground point lon\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Tangent height\n".format(line))
		line += 1

		header += ("#L.{0:2d} : tangent pnt: Solar Zenith angle\n".format(line))
		line += 1
		header += ("#L.{0:2d} : tangent pnt: rel. Solar Azimuth angle\n".format(line))
		line += 1
		header += ("#L.{0:2d} : tangent pnt: LOS zenith\n".format(line))
		line += 1
		header += ("#L.{0:2d} : TOA: Solar Zenith angle\n".format(line))
		line += 1
		header += ("#L.{0:2d} : TOA: rel Solar Azimuth angle\n".format(line))
		line += 1
		header += ("#L.{0:2d} : TOA: LOS zenith\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Sat: Solar Zenith angle\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Sat: rel Solar Azimuth angle\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Sat: LOS zenith\n".format(line))
		line += 1

		header += ("#L.{0:2d} : Sat. height\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Earth radius\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Npix lines : wavelength  n_altitude x radiance".format(line))
		self.textheader_length = n_header
		self.textheader = header

	def read_from_file(self, filename):
		"""SCIAMACHY level 1c limb scan general file import

		Tries `netcdf` first, the custom binary format second if
		netcdf fails, and finally ASCII import if that also fails.
		"""
		try:
			# try netcdf first
			self.read_from_netcdf(filename)
		except:
			try:
				# fall back to mpl binary
				self.read_from_mpl_binary(filename)
			except:
				# fall back to text file as a last resort
				self.read_from_textfile(filename)

	def local_solar_time(limb_scan, debug=True):
		"""Local solar time at limb scan footprint centre

		Returns
		-------
		(mean_lst, apparent_lst, eot_correction): tuple
			* mean_lst - mean local solar time
			* apparent_lst - apparent local solar time, equation of time corrected
			* eot_correction - equation of time correction in minutes
		"""
		import datetime as dt
		import logging
		dtime = dt.datetime(*limb_scan.date)
		doy = int(dtime.strftime("%j"))
		eot_correction = _equation_of_time(doy)
		hours, mins, secs = limb_scan.date[3:]
		clat, clon = limb_scan.cent_lat_lon[:2]
		if clon > 180.0:
			clon = clon - 360.0
		mean_lst = hours + mins / 60. + secs / 3600. + clon / 15.
		apparent_lst = mean_lst + eot_correction / 60.0

		if debug:
			logging.debug("%d %d %02d",
					limb_scan.orbit, limb_scan.state_in_orbit, doy)
			logging.debug("%s", limb_scan.orbit_state)
			logging.debug("%02d %02d %02d", hours, mins, secs)
			logging.debug("%.3f %.3f %.6f %.6f %.6f", clat, clon,
					mean_lst % 24, apparent_lst % 24, eot_correction)
		return mean_lst % 24, apparent_lst % 24, eot_correction
