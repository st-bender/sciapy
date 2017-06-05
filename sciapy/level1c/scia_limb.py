#!/usr/bin/env python
# vim: set fileencoding=utf-8
"""SCIAMACHY level 1c limb spectra module

Copyright (c) 2014-2017 Stefan Bender

This module contains the class for SCIAMACHY level 1c limb spectra.
It include some simple conversion routines: from and to ascii,
from and to binary, from and to netcdf.

A simple import from HDF5 files produced by the SRON nadc_tools
(https://github.com/rmvanhees/nadc_tools) is also supported.

License
-------
This module is part of sciapy.
sciapy is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, version 2.
See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""
from __future__ import absolute_import, division, print_function

__all__ = ["scia_limb_point", "scia_limb_scan", "__doc__"]

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

	def from_limb_scan(self, ls, i):
		self.date = ls.date
		self.npix = ls.npix
		self.orbit = ls.orbit
		if list(ls.sub_sat_lat_list):
			self.sub_sat_lat = ls.sub_sat_lat_list[i]
			self.sub_sat_lon = ls.sub_sat_lon_list[i]
		self.tp_lat = ls.tp_lat_list[i]
		self.tp_lon = ls.tp_lon_list[i]
		self.tp_alt = ls.tp_alt_list[i]
		self.tp_sza = ls.tp_sza_list[i]
		self.tp_saa = ls.tp_saa_list[i]
		self.tp_los_zenith = ls.tp_los_zenit_list[i]
		self.toa_sza = ls.toa_sza_list[i]
		self.toa_saa = ls.toa_saa_list[i]
		self.toa_los_zenith = ls.toa_los_zenit_list[i]
		self.sat_sza = ls.sat_sza_list[i]
		self.sat_saa = ls.sat_saa_list[i]
		self.sat_los_zenith = ls.sat_los_zenit_list[i]
		self.sat_alt = ls.sat_alt_list[i]
		self.earthradius = ls.earthradii[i]

		self.wls = ls.wls
		self.rads = ls.rad_list[i]
		self.errs = ls.err_list[i]


class scia_limb_scan(object):
	"""SCIAMACHY limb scan data

	Contains the data from all or some selected tangent points.
	The format is inspired by the SCIAMACHY ascii data format.

	Attributes
	----------
	textheader_length : int
		The number of lines of the text header.
	textheader : string
		The header containing the limb scan meta data.
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
	sub_sat_lat_list : (M,) array_like
		The latitudes of the satellite ground points (M = nalt).
	sub_sat_lon_list : (M,) array_like
		The longitudes of the satellite ground points (M = nalt).
	tp_lat_list : (M,) array_like
		The latitudes of the tangent points (M = nalt).
	tp_lon_list : (M,) array_like
		The longitudes of the tangent points (M = nalt).
	tp_alt_list : (M,) array_like
		The tangent altitudes (M = nalt).
	tp_sza_list : (M,) array_like
		The solar zenith angles at the tangent points (M = nalt).
	tp_saa_list : (M,) array_like
		The solar azimuth angles at the tangent points (M = nalt).
	tp_los_zenith_list : (M,) array_like
		The line-of-sight zenith angles at the tangent points (M = nalt).
	toa_sza_list : (M,) array_like
		The solar zenith angles at the top-of-atmosphere points (M = nalt).
	toa_saa_list : (M,) array_like
		The solar azimuth angles at the top-of-atmosphere points (M = nalt).
	toa_los_zenith_list : (M,) array_like
		The line-of-sight zenith angles at the top-of-atmosphere points (M = nalt).
	sat_sza_list : (M,) array_like
		The solar zenith angles at the satellite points (M = nalt).
	sat_saa_list : (M,) array_like
		The solar azimuth angles at the satellite points (M = nalt).
	sat_los_zenith_list : (M,) array_like
		The line-of-sight zenith angles at the satellite points (M = nalt).
	sat_alt_list : (M,) array_like
		The satellite altitudes (M = nalt).
	earthradii : (M,) array_like
		The earth radii at the tangent ground points (M = nalt).
	wls : (N,) array_like
		The spectral wavelengths.
	rad_list : (M, N) array_like
		The radiances at the tangent points, M = nalt, N = len(wls).
	err_list : (M, N) array_like
		The relative radiance uncertainties at the tangent points,
		M = nalt, N = len(wls).
	"""
	from .scia_limb_nc import read_from_netcdf, write_to_netcdf
	from .scia_limb_txt import read_from_textfile, write_to_textfile
	from .scia_limb_mpl import read_from_mpl_binary, write_to_mpl_binary
	from .scia_limb_hdf5 import (read_hdf5_limb_state_common_data,
								read_hdf5_limb_state_spectral_data)

	def __init__(self):
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

		self.sub_sat_lat_list = []
		self.sub_sat_lon_list = []
		self.tp_lat_list = []
		self.tp_lon_list = []
		self.tp_alt_list = []
		self.tp_sza_list = []
		self.tp_saa_list = []
		self.tp_los_zenith_list = []
		self.toa_sza_list = []
		self.toa_saa_list = []
		self.toa_los_zenith_list = []
		self.sat_sza_list = []
		self.sat_saa_list = []
		self.sat_los_zenith_list = []
		self.sat_alt_list = []
		self.earthradii = []

		self.wls = []
		self.rad_list = []
		self.err_list = []

	def parse_textheader(self):
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

	def read_from_file(self, filename):
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
