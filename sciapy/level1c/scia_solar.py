# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2014-2017 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY level 1c solar spectra module

This module contains the python class for SCIAMACHY level 1c solar spectra.
It include some simple conversion routines, from and to ascii and from and to netcdf.

A simple import from SRON nadc_tools (https://github.com/rmvanhees/nadc_tools)
produced HDF5 is also supported.
"""
from __future__ import absolute_import, division, print_function

__all__ = ["scia_solar", "__doc__"]

import datetime
import logging

import numpy as np
try:
	from netCDF4 import Dataset as netcdf_file
	_fmtargs = {"format": "NETCDF4"}
except ImportError:
	try:
		from scipy.io.netcdf import netcdf_file
		_fmtargs = {"version": 1}
	except ImportError:
		from pupynere import netcdf_file
		_fmtargs = {"version": 1}

from ._types import _try_decode

logging.basicConfig(level=logging.INFO,
		format="[%(levelname)-8s] (%(asctime)s) "
		"%(filename)s:%(lineno)d %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S %z")

class scia_solar(object):
	"""SCIAMACHY solar reference spectrum class

	Contains the SCIAMACHY level 1c solar reference spectrum.
	The format is inspired by the SCIAMACHY ascii data format.

	Attributes
	----------
	textheader_length : int
		The number of lines of the text header.
	textheader : str
		The header containing the solar spectrum meta data.
	npix : int
		The number of spectral points.
	solar_id : str
		The solar reference spectrum ID,
		choices: "D0", "D1", "D2", "E0", "E1", "A0", "A1",
		"N1", "N2", "N3", "N4", "N5".
	orbit : int
		The SCIAMACHY/Envisat orbit number.
	time : :class:`datetime.datetime`
		The sensing start time of the (semi-)orbit.
	wls : (M,) array_like
		The spectral wavelengths.
	rads : (M,) array_like
		The radiances at the spectral points, M = len(wls).
	errs : (M,) array_like
		The relative radiance uncertainties at the tangent points, M = len(wls).
	"""

	def __init__(self):
		self.textheader_length = 0
		self.textheader = ""
		self.npix = 0
		self.solar_id = ""
		self.orbit = -1
		self.time = None
		self.wls = np.array([])
		self.rads = np.array([])
		self.errs = None

	def read_from_netcdf(self, filename):
		"""SCIAMACHY level 1c solar reference netcdf import

		Parameters
		----------
		filename : str
			The netcdf filename to read the data from.

		Returns
		-------
		nothing
		"""
		ncf = netcdf_file(filename, 'r')
		self.textheader_length = ncf.textheader_length
		self.textheader = _try_decode(ncf.textheader)
		if self.textheader_length > 6:
			self.solar_id = _try_decode(ncf.solar_id)
			self.orbit = ncf.orbit
			_time = _try_decode(ncf.time)
			self.time = datetime.datetime.strptime(_time, '%Y-%m-%d %H:%M:%S %Z')
		self.wls = ncf.variables['wavelength'][:].copy()
		self.rads = ncf.variables['radiance'][:].copy()
		self.npix = self.wls.size
		try:
			self.errs = ncf.variables['radiance errors'][:].copy()
		except KeyError:
			self.errs = None
		ncf.close()

	def read_from_textfile(self, filename):
		"""SCIAMACHY level 1c solar reference text import

		Parameters
		----------
		filename : str
			The (plain) ascii table filename to read the data from.

		Returns
		-------
		nothing
		"""
		if hasattr(filename, 'seek'):
			f = filename
		else:
			f = open(filename, 'r')
		h_list = []
		try:
			nh = int(f.readline())
		except:
			nh = 6
			f.seek(0)
		for i in range(0, nh):
			h_list.append(f.readline().rstrip())
		self.textheader_length = nh
		self.textheader = '\n'.join(h_list)
		self.npix = int(f.readline())
		if nh > 6:
			self.solar_id = f.readline().rstrip()
			self.orbit = int(f.readline())
			self.time = datetime.datetime.strptime(
				f.readline().strip('\n') + " UTC",
				"%Y %m %d %H %M %S %Z",
			)
			self.wls, self.rads = np.genfromtxt(filename, skip_header=nh + 5, unpack=True)
			self.errs = None
		else:
			self.wls, self.rads, self.errs = np.genfromtxt(filename, skip_header=7, unpack=True)

	def read_from_hdf5(self, hf, ref="D0"):
		"""SCIAMACHY level 1c solar reference HDF5 import

		Parameters
		----------
		hf : opened file
			Pointer to the opened level 1c HDF5 file
		ref : str
			The solar reference spectra id name,
			choose from: "D0", "D1", "D2", "E0", "E1", "A0", "A1",
			"N1", "N2", "N3", "N4", "N5". Defaults to "D0".

		Returns
		-------
		success : int
			0 on success, 1 if an error occured.
		"""
		product = hf.get("/MPH")["product_name"][0].decode()
		soft_ver = hf.get("/MPH")["software_version"][0].decode()
		key_ver = hf.get("/SPH")["key_data_version"][0].decode()
		mf_ver = hf.get("/SPH")["m_factor_version"][0].decode()
		init_version = hf.get("/SPH")["init_version"][0].decode().strip()
		init_ver, decont = init_version.split(' ')
		decont = decont.lstrip("DECONT=")
		start_date = hf.get("/MPH")["sensing_start"][0].decode().rstrip('"')
		# fill some class variables
		self.time = (datetime.datetime.strptime(start_date, "%d-%b-%Y %H:%M:%S.%f")
					.replace(tzinfo=datetime.timezone.utc))
		self.orbit = hf.get("/MPH")["abs_orbit"][0]
		self.solar_id = ref

		logging.debug("product: %s, orbit: %s", product, self.orbit)
		logging.debug("soft_ver: %s, key_ver: %s, mf_ver: %s, init_ver: %s, "
				"decont_ver: %s", soft_ver, key_ver, mf_ver, init_ver, decont)

		# Prepare the header
		datatype_txt = "SCIAMACHY solar mean ref."
		n_header = 10
		line = n_header + 2
		header = ("#Data type          : {0}\n".format(datatype_txt))
		header += ("#L1b product        : {0}\n".format(product))
		header += ("#Orbit nr.          : {0:05d}\n".format(self.orbit))
		header += ("#Ver. Proc/Key/M/I/D: {0}  {1}  {2}  {3}  {4}\n"
				.format(soft_ver, key_ver, mf_ver, init_ver, decont))
		header += ("#Starttime          : {0}\n".format(start_date))
		header += ("#L.{0:2d} : Number_of_pixels\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Solar ID\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Orbit\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Date Time : yyyy mm dd hh mm ss\n".format(line))
		line += 1
		header += ("#L.{0:2d} : Npix lines : wavelangth  irradiance  accuracy".format(line))

		sun_ref = hf.get("/GADS/SUN_REFERENCE")
		this_ref = sun_ref[sun_ref["sun_spec_id"] == ref]

		# fill the remaining class variables
		self.textheader_length = n_header
		self.textheader = header
		self.wls = this_ref["wvlen_sun"][:]
		self.npix = len(self.wls)
		self.rads = this_ref["mean_sun"][:]
		self.errs = this_ref["accuracy_sun"][:]
		return 0

	def read_from_file(self, filename):
		"""SCIAMACHY level 1c solar reference data import

		Convenience function to read the reference spectrum.
		Tries to detect the file format automatically trying netcdf first,
		and if that fails falls back to the text reader.
		Currently no HDF5 support.

		Parameters
		----------
		filename : str
			The filename to read from.

		Returns
		-------
		nothing
		"""
		try:
			# try netcdf first
			self.read_from_netcdf(filename)
		except:
			# fall back to text file
			self.read_from_textfile(filename)

	def write_to_netcdf(self, filename):
		"""SCIAMACHY level 1c solar reference netcdf export

		Parameters
		----------
		filename : str
			The netcdf filename to write the data to.

		Returns
		-------
		nothing
		"""
		ncf = netcdf_file(filename, 'w', **_fmtargs)
		ncf.textheader_length = self.textheader_length
		ncf.textheader = self.textheader
		ncf.solar_id = self.solar_id
		ncf.orbit = self.orbit
		ncf.time = self.time.strftime('%Y-%m-%d %H:%M:%S UTC')

		ncf.createDimension('wavelength', self.npix)
		wavs = ncf.createVariable('wavelength', np.dtype('float64').char, ('wavelength',))
		wavs.units = 'nm'
		wavs[:] = self.wls

		rads = ncf.createVariable('radiance', np.dtype('float64').char, ('wavelength',))
		rads.units = 'ph / s / cm^2 / nm'
		rads[:] = self.rads

		if self.errs is not None:
			errs = ncf.createVariable('radiance errors', np.dtype('float64').char, ('wavelength',))
			errs.units = 'ph / s / cm^2 / nm'
			errs[:] = self.errs

		ncf.close()

	def write_to_textfile(self, filename):
		"""SCIAMACHY level 1c solar reference text export

		Parameters
		----------
		filename : str
			The (plain) ascii table filename to write the data to.
			Passing sys.STDOUT writes to the console.

		Returns
		-------
		nothing
		"""
		if hasattr(filename, 'seek'):
			f = filename
		else:
			f = open(filename, 'w')
		if self.textheader_length > 6:
			print("{0:2d}".format(self.textheader_length), file=f)
		print(self.textheader, file=f)
		print(self.npix, file=f)
		if self.textheader_length > 6:
			print(self.solar_id, file=f)
			print(self.orbit, file=f)
			print("%4d %2d %2d %2d %2d %2d" % (self.time.year, self.time.month,
					self.time.day, self.time.hour, self.time.minute, self.time.second),
				file=f)
		for i in range(self.npix):
			#output = []
			#output.append(self.wls[i])
			#output.append(self.rads[i])
			#output.append(self.errs[i])
			#print('\t'.join(map(str, output)), file=f)
			if self.errs is not None:
				print(
					"{0:9.4f}  {1:12.5e}  {2:12.5e}".format(
						self.wls[i], self.rads[i], self.errs[i],
					),
					file=f,
				)
			else:
				print(
					"{0:9.4f}  {1:12.5e}".format(
						self.wls[i], self.rads[i],
					),
					file=f,
				)
