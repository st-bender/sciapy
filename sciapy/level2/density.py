# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2015-2018 Stefan Bender
#
# This file is part of sciapy.
# sciapy is free software: you can redistribute it or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY level 2 number density retrieval results interface

Interface classes for the level 2 retrieval results from text (ascii)
files and netcdf files for further processing.
"""

from __future__ import absolute_import, division, print_function

import os
import sys
import datetime as dt

import numpy as np
try:
	from netCDF4 import Dataset as netcdf_file
	fmtargs = {"format": "NETCDF4"}
except ImportError:
	try:
		from scipy.io.netcdf import netcdf_file
		fmtargs = {"version": 1}
	except ImportError:
		from pupynere import netcdf_file
		fmtargs = {"version": 1}

__all__ = ["scia_density", "scia_densities"]

try:
	_UTC = dt.timezone.utc
except AttributeError:
	# python 2.7
	class UTC(dt.tzinfo):
		def utcoffset(self, d):
			return dt.timedelta(0)
		def tzname(self, d):
			return "UTC"
		def dst(self, d):
			return dt.timedelta(0)
	_UTC = UTC()

_meas_dtypes = [[('gp_id', int),
		('alt_max', float), ('alt', float), ('alt_min', float),
		('lat_max', float), ('lat', float), ('lat_min', float),
		('density', float), ('dens_err_meas', float),
		('dens_err_tot', float), ('dens_tot', float)],
	[('gp_id', int),
		('alt_max', float), ('alt', float), ('alt_min', float),
		('lat_max', float), ('lat', float), ('lat_min', float),
		('longitude', float),
		('density', float), ('dens_err_meas', float),
		('dens_err_tot', float), ('dens_tot', float)],
	[('gp_id', int),
		('alt_max', float), ('alt', float), ('alt_min', float),
		('lat_max', float), ('lat', float), ('lat_min', float),
		('longitude', float),
		('density', float), ('dens_err_meas', float),
		('dens_err_tot', float), ('dens_tot', float),
		('apriori', float)],
	[('gp_id', int),
		('alt_max', float), ('alt', float), ('alt_min', float),
		('lat_max', float), ('lat', float), ('lat_min', float),
		('longitude', float),
		('density', float), ('dens_err_meas', float),
		('dens_err_tot', float), ('dens_tot', float),
		('apriori', float), ('akdiag', float)]]


def _unique_values(vals):
	ldum = []
	[ldum.append(i) for i in vals if not ldum.count(i)]
	return np.asarray(ldum).flatten()


class scia_density(object):
	"""SCIAMACHY single scan retrieved number densities"""
	def __init__(self):
		self.gp_id = 0
		self.alt_min = 0.
		self.alt = 0.
		self.alt_max = 0.
		self.lat_min = 0.
		self.lat = 0.
		self.lat_max = 0.
		self.lon = 0.
		self.density = 0.
		self.dens_err_meas = 0.
		self.dens_err_tot = 0.
		self.dens_tot = 0.
		self.akdiag = 0.
		self.apriori = 0.


class scia_densities(object):
	"""SCIAMACHY orbital retrieved number densities

	Class interface to orbit-wise SCIAMACHY retrieval results.
	The attributes are based on the text file layout and are
	tied to the NO retrieval for now.

	Parameters
	----------
	ref_date: str, optional
		The reference date on which to base the date calculations on.
		Default: "2000-01-01"
	ver: str, optional
		Explicit density version, used for exporting the data.
		Not used if set to `None`.
		Default: `None`
	data_ver: str, optional
		Level 2 data version to use, as "ver" used for exporting.
		Not used if set to `None`.
		Default: `None`

	Attributes
	----------
	version
		file version string
	data_version
		level 2 data version
	date0
		reference date
	nalt
		number of altitudes in the orbit
	nlat
		number of latitudes in the orbit
	nlon
		number of longitudes in the orbit, if longitudes are available
	orbit
		SCIAMACHY/Envisat orbit number
	date
		number of days of the orbit counting from the reference date
		date0
	alts_min
	alts
	alts_max
		the altitude bins: minimum, central, and maximum altitude
	lats_min
	lats
	lats_max
		the latitude bins: minimum, central, and maximum latitude
	lons:
		the central longitude of the bins, only used if available

	densities
		NO number densities in the bins, (nlat, nalt) array_like
	dens_err_meas
		NO number densities measurement uncertainty,
		(nlat, nalt) array_like
	dens_err_tot
		NO number densities total uncertainty, (nlat, nalt) array_like
	dens_tot
		total number densities calculated and interpolated NRLMSIS-00
		values, (nlat, nalt) array_like

	apriori
		prior NO number densities, (nlat, nalt) array_like if available,
		otherwise `None`
	akdiag
		diagonal element of the averaging kernel matrix at the retrieval
		grid point. (nlat, nalt) array_like if available otherwise `None`

	Methods
	-------
	read_from_textfile
	read_from_netcdf
	read_from_file
	write_to_textfile
	write_to_netcdf

	Note
	----
	The variables are empty when initialized, use one of the
	read_from_...() methods to fill with actual data.
	"""
	def __init__(self, ref_date="2000-01-01", ver=None, data_ver=None):
		self.version = ver
		self.data_version = data_ver
		self.date0 = dt.datetime.strptime(ref_date, "%Y-%m-%d").replace(tzinfo=_UTC)
		self.nalt = 0
		self.nlat = 0
		self.nlon = 0
		self.orbit = -1
		self.date = -1
		self.alts_min = np.array([])
		self.alts = np.array([])
		self.alts_max = np.array([])
		self.lats_min = np.array([])
		self.lats = np.array([])
		self.lats_max = np.array([])
		self.lons = np.array([])
		self.akdiag = None
		self.apriori = None

	def read_from_textfile(self, filename):
		"""Read NO densities from ascii table file

		Parameters
		----------
		filename: str, file object or io.TextIOBase.buffer
			The filename or stream to read the data from. For example
			to read from stdin in python 3, pass `sys.stdin.buffer`.
		"""
		if hasattr(filename, 'seek'):
			f = filename
		else:
			f = open(filename, 'rb')
			# example filename:000NO_orbit_41467_20100203_Dichten.txt
			fn_fields = os.path.basename(filename).split('_')
			self.orbit = int(fn_fields[2])
			self.date = (dt.datetime.strptime(fn_fields[3], "%Y%m%d")
						.replace(tzinfo=_UTC) - self.date0).days
			if self.data_version is None:
				# try some heuristics to find the level 2 data version
				self.data_version = os.path.dirname(filename).split('v')[-1]
		line = f.readline()
		data = line.split()
		mydtype = _meas_dtypes[len(data) - 13]
		marr = np.genfromtxt(f, dtype=mydtype)
		f.close()

		# unique altitudes
		self.alts_min = _unique_values(marr['alt_min'])
		self.alts = _unique_values(marr['alt'])
		self.alts_max = _unique_values(marr['alt_max'])

		# unique latitudes
		self.lats_min = _unique_values(marr['lat_min'])
		self.lats = _unique_values(marr['lat'])
		self.lats_max = _unique_values(marr['lat_max'])

		# unique longitudes if available
		try:
			self.lons = _unique_values(marr['longitude'])
		except:
			pass

		self.nalt = len(self.alts)
		self.nlat = len(self.lats)
		self.nlon = len(self.lons)

		# reorder by latitude first, then altitude
		self.densities = marr['density'].flatten().reshape(self.nalt, self.nlat).transpose()
		self.dens_err_meas = marr['dens_err_meas'].flatten().reshape(self.nalt, self.nlat).transpose()
		self.dens_err_tot = marr['dens_err_tot'].flatten().reshape(self.nalt, self.nlat).transpose()
		self.dens_tot = marr['dens_tot'].flatten().reshape(self.nalt, self.nlat).transpose()

		# apriori if available
		try:
			self.apriori = marr['apriori'].flatten().reshape(self.nalt, self.nlat).transpose()
		except:
			pass
		# akdiag if available
		try:
			self.akdiag = marr['akdiag'].flatten().reshape(self.nalt, self.nlat).transpose()
		except:
			pass

	def write_to_textfile(self, filename):
		"""Write NO densities to ascii table files

		Parameters
		----------
		filename: str or file object or io.TextIOBase.buffer
			The filename or stream to write the data to. For writing to
			stdout in python 3, pass `sys.stdout.buffer`.
		"""
		if hasattr(filename, 'seek'):
			f = filename
		else:
			f = open(filename, 'w')

		header = "%5s %13s %12s %13s %14s  %12s %14s   %12s  %12s %12s %12s %12s" % ("GP_ID",
				"Max_Hoehe[km]", "Hoehe[km]", "Min_Hoehe[km]",
				"Max_Breite[°]", "Breite[°]", "Min_Breite[°]",
				"Laenge[°]",
				"Dichte[cm^-3]", "Fehler Mess[cm^-3]",
				"Fehler tot[cm^-3]", "Gesamtdichte[cm^-3]")
		if self.apriori is not None:
			header = header + " %12s" % ("apriori[cm^-3]",)
		if self.akdiag is not None:
			header = header + " %12s" % ("AKdiag",)
		print(header, file=f)

		oformat = "%5i  %+1.5E %+1.5E  %+1.5E  %+1.5E %+1.5E  %+1.5E  %+1.5E   %+1.5E       %+1.5E      %+1.5E        %+1.5E"
		oformata = "  %+1.5E"

		for i, a in enumerate(self.alts):
			for j, b in enumerate(self.lats):
				print(oformat % (i * self.nlat + j,
					self.alts_max[i], a, self.alts_min[i],
					self.lats_max[j], b, self.lats_min[j], self.lons[j],
					self.densities[j, i], self.dens_err_meas[j, i],
					self.dens_err_tot[j, i], self.dens_tot[j, i]),
					end="", file=f)
				if self.apriori is not None:
					print(" " + oformata % self.apriori[j, i], end="", file=f)
				if self.akdiag is not None:
					print(" " + oformata % self.akdiag[j, i], end="", file=f)
				print("", file=f)

	def write_to_netcdf(self, filename, close=True):
		"""Write NO densities to netcdf files

		This function has no stream, i.e. file object, support.
		Uses single precision floats (32 bit) to save space.

		Parameters
		----------
		filename: str
			The name of the file to write the data to.
		close: bool, optional
			Whether or not to close the file after writing.
			Setting to `False` enables appending further data
			to the same file.
			Default: True

		Returns
		-------
		Nothing if `close` is True. If `close` is False, returns either an
		`netCDF4.Dataset`, `scipy.io.netcdf.netcdf_file` or
		`pupynere.netcdf_file` instance depending on availability.
		"""
		alts_min_out = np.asarray(self.alts_min).reshape(self.nalt)
		alts_out = np.asarray(self.alts).reshape(self.nalt)
		alts_max_out = np.asarray(self.alts_max).reshape(self.nalt)

		lats_min_out = np.asarray(self.lats_min).reshape(self.nlat)
		lats_out = np.asarray(self.lats).reshape(self.nlat)
		lats_max_out = np.asarray(self.lats_max).reshape(self.nlat)

		ncf = netcdf_file(filename, 'w', **fmtargs)

		if self.version is not None:
			ncf.version = self.version
		if self.data_version is not None:
			ncf.L2_data_version = self.data_version
		#ncf.creation_time = dt.datetime.utcnow().replace(tzinfo=_UTC).strftime("%a %b %d %Y %H:%M:%S %z (%Z)")
		ncf.creation_time = dt.datetime.utcnow().strftime("%a %b %d %Y %H:%M:%S +00:00 (UTC)")
		ncf.author = "Firstname Lastname"

		# create netcdf file
		ncf.createDimension('altitude', self.nalt)
		ncf.createDimension('latitude', self.nlat)
		ncf.createDimension('time', None)

		forbit = ncf.createVariable('orbit', np.dtype('int32').char, ('time',))
		ftime = ncf.createVariable('time', np.dtype('int32').char, ('time',))

		falts_min = ncf.createVariable('alt_min', np.dtype('float32').char, ('altitude',))
		falts = ncf.createVariable('altitude', np.dtype('float32').char, ('altitude',))
		falts_max = ncf.createVariable('alt_max', np.dtype('float32').char, ('altitude',))
		flats_min = ncf.createVariable('lat_min', np.dtype('float32').char, ('latitude',))
		flats = ncf.createVariable('latitude', np.dtype('float32').char, ('latitude',))
		flats_max = ncf.createVariable('lat_max', np.dtype('float32').char, ('latitude',))

		falts_min.units = 'km'
		falts_min.positive = 'up'
		falts.units = 'km'
		falts.positive = 'up'
		falts_max.units = 'km'
		falts_max.positive = 'up'
		flats_min.units = 'degrees_north'
		flats.units = 'degrees_north'
		flats_max.units = 'degrees_north'

		forbit.units = '1'
		forbit.long_name = 'SCIAMACHY/Envisat orbit number'
		ftime.units = 'days since {0}'.format(self.date0.isoformat(sep=' '))
		ftime.standard_name = 'time'

		fdens = ncf.createVariable('density', np.dtype('float32').char, ('time', 'latitude', 'altitude'))
		fdens.units = 'cm^{-3}'
		fdens.standard_name = 'number_concentration_of_nitrogen_monoxide_molecules_in_air'
		fdens_err_meas = ncf.createVariable('error_meas', np.dtype('float32').char, ('time', 'latitude', 'altitude'))
		fdens_err_meas.units = 'cm^{-3}'
		fdens_err_meas.long_name = 'NO number density measurement error'
		fdens_err_tot = ncf.createVariable('error_tot', np.dtype('float32').char, ('time', 'latitude', 'altitude'))
		fdens_err_tot.units = 'cm^{-3}'
		fdens_err_tot.long_name = 'NO number density total error'
		fdens_tot = ncf.createVariable('density_air', np.dtype('float32').char, ('time', 'latitude', 'altitude'))
		fdens_tot.units = 'cm^{-3}'
		fdens_tot.long_name = 'approximate overall number concentration of air molecules (NRLMSIS-00)'

		ftime[:] = self.date
		forbit[:] = self.orbit

		falts_min[:] = alts_min_out
		falts[:] = alts_out
		falts_max[:] = alts_max_out
		flats_min[:] = lats_min_out
		flats[:] = lats_out
		flats_max[:] = lats_max_out
		# reorder by latitude first, then altitude
		fdens[0, :] = self.densities
		# reorder by latitude first, then altitude
		fdens_err_meas[0, :] = self.dens_err_meas
		fdens_err_tot[0, :] = self.dens_err_tot
		fdens_tot[0, :] = self.dens_tot

		# longitudes if they are available
		if self.nlon > 0:
			lons_out = np.asarray(self.lons).reshape(self.nlon)
			flons = ncf.createVariable('longitude', np.dtype('float32').char, ('time', 'latitude',))
			flons.units = 'degrees_east'
			flons[0, :] = lons_out

		if self.apriori is not None:
			fapriori = ncf.createVariable('apriori',
					np.dtype('float32').char, ('time', 'latitude', 'altitude'))
			fapriori.units = 'cm^{-3}'
			fapriori.long_name = 'apriori NO number density'
			fapriori[0, :] = self.apriori

		if self.akdiag is not None:
			fakdiag = ncf.createVariable('akm_diagonal',
					np.dtype('float32').char, ('time', 'latitude', 'altitude'))
			fakdiag.units = '1'
			fakdiag.long_name = 'averaging kernel matrix diagonal element'
			fakdiag[0, :] = self.akdiag
		if close:
			ncf.close()
		else:
			return ncf

	def read_from_netcdf(self, filename, close=True):
		"""Read NO densities from netcdf files

		This function has no stream, i.e. file object support.

		Parameters
		----------
		filename: str
			The filename to read the data from.
		close: bool, optional
			Whether or not to close the file after reading.
			Setting to `False` enables reading further data
			from the same file.
			Default: True

		Returns
		-------
		Nothing if `close` is True. If `close` is False, returns either an
		`netCDF4.Dataset`, `scipy.io.netcdf.netcdf_file` or
		`pupynere.netcdf_file` instance depending on availability.
		"""
		ncf = netcdf_file(filename, 'r')

		self.nalt = ncf.dimensions['altitude']
		self.nlat = ncf.dimensions['latitude']

		self.alts_min = ncf.variables['alt_min'][:]
		self.alts = ncf.variables['altitude'][:]
		self.alts_max = ncf.variables['alt_max'][:]
		self.lats_min = ncf.variables['lat_min'][:]
		self.lats = ncf.variables['latitude'][:]
		self.lats_max = ncf.variables['lat_max'][:]

		self.date = ncf.variable['time'][:]
		self.orbit = ncf.variable['orbit'][:]

		self.densities = ncf.variables['density'][:]
		self.dens_err_meas = ncf.variables['error_meas'][:]
		self.dens_err_tot = ncf.variables['error_tot'][:]
		self.dens_tot = ncf.variables['density_air'][:]

		# longitudes if they are available
		try:
			self.nlon = ncf.dimensions['longitude']
			self.lons = ncf.variables['longitude'][:]
		except:
			pass

		# apriori
		try:
			self.apriori = ncf.variables['apriori'][:]
		except:
			pass

		# akm diagonal elements
		try:
			self.akdiag = ncf.variables['akm_diagonal'][:]
		except:
			pass

		if close:
			ncf.close()
		else:
			return ncf

	def read_from_file(self, filename):
		"""Wrapper to read NO desnities from files

		Simple wrapper to delegate reading the data from either netcdf
		or ascii files. Poor man's logic: simply try netcdf first, and
		if that fails, read as ascii.

		Parameters
		----------
		filename: str
			The filename to read the data from.
		"""
		try:
			# try netcdf first
			self.read_from_netcdf(filename)
		except:
			# fall back to text file as a last resort
			self.read_from_textfile(filename)


def main(*args):
	argc = len(sys.argv)
	if argc < 2:
		print("Not enough arguments, Usage:\n"
			"{0} [input] output [< input]".format(sys.argv[0]))
		sys.exit(1)
	elif argc < 3:
		try:
			infile = sys.stdin.buffer  # Python 3
		except AttributeError:
			infile = sys.stdin
		outfile = sys.argv[1]
	else:
		infile = sys.argv[1]
		outfile = sys.argv[2]
	sdl = scia_densities()
	sdl.read_from_file(infile)
	sdl.write_to_netcdf(outfile)


if __name__ == "__main__":
	sys.exit(main())
