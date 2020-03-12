# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2018 Stefan Bender
#
# This file is part of sciapy.
# sciapy is free software: you can redistribute it or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY level 2 post-processed number densities interface

Interface classes for the level 2 post-processed retrieval results.
"""

from __future__ import absolute_import, division, print_function

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

from .density import scia_densities, _UTC
from .. import __version__

__all__ = ["scia_densities_pp", "scia_density_day"]


class scia_densities_pp(scia_densities):
	"""Post-processed SCIAMACHY number densities

	Extends `scia_densities` with additional post-processing
	attributes such as (MSIS) temperature and density, local
	solar time, solar zenith angle, and geomagnetic latitudes
	and longitudes.

	This class only supports writing ascii files but reading to
	and writing from netcdf.

	Attributes
	----------
	temperature
		NRLMSISE-00 temperatures
	noem_no
		NOEM NO nuimber densities
	vmr
		NO vmr using the NRLMSISE-00 total air number densities
	lst
		Apparent local solar times
	mst
		Mean local solar times
	sza
		Solar zenith angles
	utchour
		UTC hours into measurement day
	utcdays
		Number of days since reference date
	gmlats
		IGRF-12 geomagentic latitudes
	gmlons
		IGRF-12 geomagentic longitudes
	aacgmgmlats
		AACGM geomagentic latitudes
	aacgmgmlons
		AACGM geomagentic longitudes

	Methods
	-------
	write_to_textfile
	write_to_netcdf
	read_from_netcdf
	to_xarray
	"""
	def __init__(self, ref_date="2000-01-01",
			ver=None, data_ver=None):
		self.filename = None
		self.temperature = None
		self.noem_no = None
		self.vmr = None
		self.lst = None
		self.mst = None
		self.sza = None
		self.utchour = None
		self.utcdays = None
		self.gmlats = None
		self.gmlons = None
		self.aacgmgmlats = None
		self.aacgmgmlons = None
		super(scia_densities_pp, self).__init__(
				ref_date=ref_date, ver=ver, data_ver=data_ver)

	def write_to_textfile(self, filename):
		"""Write the variables to ascii table files

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
		if self.temperature is not None:
			header = header + " %12s" % ("T[K]",)
		if self.noem_no is not None:
			header = header + " %12s" % ("NOEM_NO[cm^-3]",)
		if self.akdiag is not None:
			header = header + " %12s" % ("AKdiag",)
		if self.lst is not None:
			header = header + " %12s" % ("LST",)
		if self.mst is not None:
			header = header + " %12s" % ("MST",)
		if self.sza is not None:
			header = header + " %12s" % ("SZA",)
		if self.utchour is not None:
			header = header + " %12s" % ("Hour",)
		if self.utcdays is not None:
			header = header + " %12s" % ("Days",)
		if self.gmlats is not None:
			header = header + " %12s" % ("GeomagLat",)
		if self.gmlons is not None:
			header = header + " %12s" % ("GeomagLon",)
		if self.aacgmgmlats is not None:
			header = header + " %12s" % ("AACGMGeomagLat",)
		if self.aacgmgmlons is not None:
			header = header + " %12s" % ("AACGMGeomagLon",)
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
				if self.temperature is not None:
					print(" " + oformata % self.temperature[j, i], end="", file=f)
				if self.noem_no is not None:
					print(" " + oformata % self.noem_no[j, i], end="", file=f)
				if self.akdiag is not None:
					print(" " + oformata % self.akdiag[j, i], end="", file=f)
				if self.lst is not None:
					print(" " + oformata % self.lst[j], end="", file=f)
				if self.mst is not None:
					print(" " + oformata % self.mst[j], end="", file=f)
				if self.sza is not None:
					print(" " + oformata % self.sza[j], end="", file=f)
				if self.utchour is not None:
					print(" " + oformata % self.utchour[j], end="", file=f)
				if self.utcdays is not None:
					print(" " + oformata % self.utcdays[j], end="", file=f)
				if self.gmlats is not None:
					print(" " + oformata % self.gmlats[j], end="", file=f)
				if self.gmlons is not None:
					print(" " + oformata % self.gmlons[j], end="", file=f)
				if self.aacgmgmlats is not None:
					print(" " + oformata % self.aacgmgmlats[j], end="", file=f)
				if self.aacgmgmlons is not None:
					print(" " + oformata % self.aacgmgmlons[j], end="", file=f)
				print("", file=f)

	def write_to_netcdf(self, filename, close=True):
		"""Write variables to netcdf files

		This function has no stream, i.e. file object, support.

		Parameters
		----------
		filename: str
			The name of the file to write the data to.
		"""
		# write the base variables first and keep the file open for appending
		ncf = scia_densities.write_to_netcdf(self, filename, close=False)

		if self.temperature is not None:
			ftemp = ncf.createVariable('temperature',
					np.dtype('float64').char, ('time', 'latitude', 'altitude'))
			ftemp.units = 'K'
			ftemp.long_name = 'temperature'
			ftemp.model = 'NRLMSIS-00'
			ftemp[0, :] = self.temperature

		if self.noem_no is not None:
			fnoem_no = ncf.createVariable('NOEM_density',
					np.dtype('float64').char, ('time', 'latitude', 'altitude'))
			fnoem_no.units = 'cm^{-3}'
			fnoem_no.long_name = 'NOEM NO number density'
			fnoem_no[0, :] = self.noem_no

		if self.vmr is not None:
			fvmr = ncf.createVariable('VMR',
					np.dtype('float64').char, ('time', 'latitude', 'altitude'))
			fvmr.units = 'ppb'
			fvmr.long_name = 'volume mixing ratio'
			fvmr[0, :] = self.vmr

		if self.lst is not None:
			flst = ncf.createVariable('app_lst',
					np.dtype('float64').char, ('time', 'latitude',))
			flst.units = 'hours'
			flst.long_name = 'apparent local solar time'
			flst[0, :] = self.lst

		if self.mst is not None:
			fmst = ncf.createVariable('mean_lst',
					np.dtype('float64').char, ('time', 'latitude',))
			fmst.units = 'hours'
			fmst.long_name = 'mean local solar time'
			fmst[0, :] = self.mst

		if self.sza is not None:
			fsza = ncf.createVariable('mean_sza',
					np.dtype('float64').char, ('time', 'latitude',))
			fsza.units = 'degrees'
			fsza.long_name = 'mean solar zenith angle'
			fsza[0, :] = self.sza

		if self.utchour is not None:
			futc = ncf.createVariable('utc_hour',
					np.dtype('float64').char, ('time', 'latitude',))
			futc.units = 'hours'
			futc.long_name = 'measurement utc time'
			futc[0, :] = self.utchour

		if self.utcdays is not None:
			futcd = ncf.createVariable('utc_days',
					np.dtype('float64').char, ('time', 'latitude',))
			futcd.long_name = 'measurement day'
			futcd.units = 'days since {0}'.format(self.date0.isoformat(sep=' '))
			futcd[0, :] = self.utcdays

		if self.gmlats is not None:
			fgmlats = ncf.createVariable('gm_lats',
					np.dtype('float64').char, ('time', 'latitude',))
			fgmlats.long_name = 'geomagnetic_latitude'
			fgmlats.model = 'IGRF'
			fgmlats.units = 'degrees_north'
			fgmlats[0, :] = self.gmlats

		if self.gmlons is not None:
			fgmlons = ncf.createVariable('gm_lons',
					np.dtype('float64').char, ('time', 'latitude',))
			fgmlons.long_name = 'geomagnetic_longitude'
			fgmlons.model = 'IGRF'
			fgmlons.units = 'degrees_east'
			fgmlons[0, :] = self.gmlons

		if self.aacgmgmlats is not None:
			faacgmgmlats = ncf.createVariable('aacgm_gm_lats',
					np.dtype('float64').char, ('time', 'latitude',))
			faacgmgmlats.long_name = 'geomagnetic_latitude'
			faacgmgmlats.model = 'AACGM'
			faacgmgmlats.units = 'degrees_north'
			faacgmgmlats[0, :] = self.aacgmgmlats

		if self.aacgmgmlons is not None:
			faacgmgmlons = ncf.createVariable('aacgm_gm_lons',
					np.dtype('float64').char, ('time', 'latitude',))
			faacgmgmlons.long_name = 'geomagnetic_longitude'
			faacgmgmlons.model = 'AACGM'
			faacgmgmlons.units = 'degrees_east'
			faacgmgmlons[0, :] = self.aacgmgmlons

		if close:
			ncf.close()
		else:
			return ncf

	def read_from_netcdf(self, filename, close=True):
		"""Read post-processed level 2 orbit files

		Parameters
		----------
		filename: str
			The name of the netcdf file.
		"""
		# read the base variables first and keep the file open for reading
		ncf = scia_densities.read_from_netcdf(self, filename, close=False)

		# additional data...
		# MSIS temperature
		try:
			self.temperature = ncf.variables['temperature'][:]
		except:
			pass
		# NOEM density
		try:
			self.noem_no = ncf.variables['NOEM_density'][:]
		except:
			pass
		# calculated vmr
		try:
			self.vmr = ncf.variables['VMR'][:]
		except:
			pass
		# apparent local solar time
		try:
			self.lst = ncf.variables['app_lst'][:]
		except:
			pass
		# mean local solar time
		try:
			self.mst = ncf.variables['mean_lst'][:]
		except:
			pass
		# mean solar zenith angle
		try:
			self.sza = ncf.variables['mean_sza'][:]
		except:
			pass
		# utc hours
		try:
			self.utchour = ncf.variables['utc_hour'][:]
		except:
			pass
		# utc days
		try:
			self.utcdays = ncf.variables['utc_days'][:]
		except:
			pass

		if close:
			ncf.close()
		else:
			return ncf

	def to_xarray(self, dateo, orbit):
		"""Convert the data to :class:`xarray.Dataset`

		This is a very simple approach, it dumps the data to a temporary
		netcdf file and reads that using :func:`xarray.open_dataset()`.

		Parameters
		----------
		dateo: float
			The days since the reference date at the equator
			crossing of the orbit. Used to set the `time`
			dimension of the dataset.
		orbit: int
			The SCIAMACHY/Envisat orbit number of the retrieved data.

		Returns
		-------
		dataset: xarray.Dataset
		"""
		import tempfile
		try:
			import xarray as xr
		except ImportError:
			print("Error: xarray not available!")
			return None
		with tempfile.NamedTemporaryFile() as tf:
			self.write_to_netcdf(tf.name)
			with xr.open_dataset(tf.name, decode_cf=False) as sdorb:
				sdorb = sdorb.drop(["alt_min", "alt_max", "lat_min", "lat_max"])
				sdorb["time"] = np.array([dateo], dtype=np.float64)
				sdorb["orbit"] = orbit
				sdorb.load()
		return sdorb


class scia_density_day(object):
	"""SCIAMACHY daily number densities combined

	Contains a stacked version of the post-processed orbit data
	for multiple orbits on a day. Used to combine the results.

	Parameters
	----------
	name: str
		Name of the retrieved species, default: "NO".
		Used to name the netcdf variables appropriately.
	ref_date: str, optional
		The reference date on which to base the date calculations on.
		Default: "2000-01-01"

	Attributes
	----------
	alts
		Retrieval grid altitudes
	lats
		Retrieval grid latitudes
	no_dens
		Retrieved number densities
	no_errs
		Measurement uncertainty
	no_etot
		Total uncertainty
	no_rstd
		Relative measurement uncertainty
	no_akd
		Averaging kernel diagonal elements
	no_apri
		Prior number density
	temperature
		NRLMSISE-00 temperatures
	noem_no
		NOEM NO nuimber densities
	vmr
		NO vmr using the NRLMSISE-00 total air number densities
	lst
		Apparent local solar times
	mst
		Mean local solar times
	sza
		Solar zenith angles
	utchour
		UTC hours into measurement day
	utcdays
		Number of days since reference date
	gmlats
		IGRF-12 geomagentic latitudes
	gmlons
		IGRF-12 geomagentic longitudes
	aacgmgmlats
		AACGM geomagentic latitudes
	aacgmgmlons
		AACGM geomagentic longitudes
	"""
	def __init__(self, name="NO", ref_date="2000-01-01", author="unknown"):
		self.date0 = (dt.datetime.strptime(ref_date, "%Y-%m-%d")
					.replace(tzinfo=_UTC))
		self.alts = None
		self.lats = None
		self.version = None
		self.data_version = None
		self.name = name
		self.author = author
		self.date = []
		self.time = []
		self.orbit = []
		self.no_dens = None
		self.no_errs = None
		self.no_etot = None
		self.no_rstd = None
		self.no_akd = None
		self.no_apri = None
		self.no_noem = None
		self.temperature = None
		self.tot_dens = None
		self.no_vmr = None
		self.lons = None
		self.lst = None
		self.mst = None
		self.sza = None
		self.utchour = None
		self.utcdays = None
		self.gmlats = None
		self.gmlons = None
		self.aacgmgmlats = None
		self.aacgmgmlons = None

	def append(self, cdata):
		"""Append (stack) the data from one orbit

		Parameters
		----------
		cdata: :class:`scia_densities_pp` instance
			Post-processed level 2 orbital data.
		"""
		self.time.extend(cdata.time)
		self.date.extend(cdata.date)
		self.no_dens = np.ma.dstack((self.no_dens, cdata.no_dens))
		self.no_errs = np.ma.dstack((self.no_errs, cdata.no_errs))
		self.no_etot = np.ma.dstack((self.no_etot, cdata.no_etot))
		self.no_rstd = np.ma.dstack((self.no_rstd, cdata.no_rstd))
		self.no_akd = np.ma.dstack((self.no_akd, cdata.no_akd))
		self.no_apri = np.ma.dstack((self.no_apri, cdata.no_apri))
		self.no_noem = np.ma.dstack((self.no_noem, cdata.no_noem))
		self.tot_dens = np.ma.dstack((self.tot_dens, cdata.tot_dens))
		self.no_vmr = np.ma.dstack((self.no_vmr, cdata.no_vmr))
		self.lons = np.ma.dstack((self.lons, cdata.lons))
		self.lst = np.ma.dstack((self.lst, cdata.lst))
		self.mst = np.ma.dstack((self.mst, cdata.mst))
		self.sza = np.ma.dstack((self.sza, cdata.sza))
		self.utchour = np.ma.dstack((self.utchour, cdata.utchour))
		self.utcdays = np.ma.dstack((self.utcdays, cdata.utcdays))
		self.gmlats = np.ma.dstack((self.gmlats, cdata.gmlats))
		self.gmlons = np.ma.dstack((self.gmlons, cdata.gmlons))
		self.aacgmgmlats = np.ma.dstack((self.aacgmgmlats, cdata.aacgmgmlats))
		self.aacgmgmlons = np.ma.dstack((self.aacgmgmlons, cdata.aacgmgmlons))

	def append_data(self, date, orbit, equtime, scia_dens):
		"""Append (stack) the data from one orbit

		Updates the data in place.

		Parameters
		----------
		date: float
			Days since `ref_date` for the time coordinate
		orbit: int
			SCIAMACHY/Envisat orbit number
		equtime: float
			UTC hour into the day at the equator
		scia_dens: :class:`scia_densities_pp` instance
			The post-processed orbit data set
		"""
		def _vstack_or_new(a, b):
			# Check if we 'stack' for the first time (a is None),
			# in that case we assign first.
			if a is None:
				return b[None]
			return np.ma.vstack((a, b[None]))

		self.version = scia_dens.version
		self.data_version = scia_dens.data_version
		self.date.append(date)
		self.orbit.append(orbit)
		self.time.append(equtime)
		_dens = scia_dens.densities
		_errs = scia_dens.dens_err_meas
		_etot = scia_dens.dens_err_tot
		_r_std = np.abs(_errs / _dens) * 100.0
		if self.alts is None:
			# we need altitudes and latitudes only once
			self.alts = scia_dens.alts
			self.lats = scia_dens.lats
		self.no_dens = _vstack_or_new(self.no_dens, _dens)
		self.no_errs = _vstack_or_new(self.no_errs, _errs)
		self.no_etot = _vstack_or_new(self.no_etot, _etot)
		self.no_rstd = _vstack_or_new(self.no_rstd, _r_std)
		self.no_akd = _vstack_or_new(self.no_akd, scia_dens.akdiag)
		self.no_apri = _vstack_or_new(self.no_apri, scia_dens.apriori)
		self.temperature = _vstack_or_new(self.temperature, scia_dens.temperature)
		self.no_noem = _vstack_or_new(self.no_noem, scia_dens.noem_no)
		self.tot_dens = _vstack_or_new(self.tot_dens, scia_dens.dens_tot)
		self.no_vmr = _vstack_or_new(self.no_vmr, scia_dens.vmr)
		self.lons = _vstack_or_new(self.lons, scia_dens.lons)
		self.lst = _vstack_or_new(self.lst, scia_dens.lst)
		self.mst = _vstack_or_new(self.mst, scia_dens.mst)
		self.sza = _vstack_or_new(self.sza, scia_dens.sza)
		self.utchour = _vstack_or_new(self.utchour, scia_dens.utchour)
		self.utcdays = _vstack_or_new(self.utcdays, scia_dens.utcdays)
		self.gmlats = _vstack_or_new(self.gmlats, scia_dens.gmlats)
		self.gmlons = _vstack_or_new(self.gmlons, scia_dens.gmlons)
		self.aacgmgmlats = _vstack_or_new(self.aacgmgmlats, scia_dens.aacgmgmlats)
		self.aacgmgmlons = _vstack_or_new(self.aacgmgmlons, scia_dens.aacgmgmlons)

	def write_to_netcdf(self, filename):
		"""Write variables to netcdf files

		Parameters
		----------
		filename: str
			The name of the file to write the data to.
		"""
		_var_dicts = {
			"2.1": {
				"dens_tot": {
					"name": "TOT_DENS",
					"long_name": "total number density (NRLMSIS-00)",
					"model": None,
				},
				"temperature": {
					"name": "temperature",
					"long_name": "temperature",
					"model": "NRLMSIS-00",
				},
			},
			"2.2": {
				"dens_tot": {
					"name": "MSIS_Dens",
					"long_name": "MSIS total number density",
					"model": "NRLMSIS-00",
				},
				"temperature": {
					"name": "MSIS_Temp",
					"long_name": "MSIS temperature",
					"model": "NRLMSIS-00",
				},
			},
		}

		ncf = netcdf_file(filename, 'w', **fmtargs)
		o = np.asarray(self.orbit)
		d = np.asarray(self.date)
		t = np.asarray(self.time)

		if self.version is not None:
			ncf.version = self.version
		if self.data_version is not None:
			ncf.L2_data_version = self.data_version
		ncf.software = "sciapy {0}".format(__version__)
		ncf.creation_time = dt.datetime.utcnow().strftime("%a %b %d %Y %H:%M:%S +00:00 (UTC)")
		ncf.author = self.author

		ncf.createDimension('time', None)
		ncf.createDimension('altitude', self.alts.size)
		ncf.createDimension('latitude', self.lats.size)
		forbit = ncf.createVariable('orbit', np.dtype('int32').char, ('time',))
		forbit.axis = 'T'
		forbit.calendar = 'standard'
		forbit.long_name = 'orbit'
		forbit.standard_name = 'orbit'
		forbit.units = 'orbit number'
		# the time coordinate is actually called "date" here
		ftime = ncf.createVariable('time', 'f8', ('time',))
		ftime.axis = 'T'
		ftime.calendar = 'standard'
		ftime.long_name = 'equatorial crossing time'
		ftime.standard_name = 'time'
		ftime.units = 'days since {0}'.format(self.date0.isoformat(sep=' '))
		#ftime.units = 'days since {0}'.format(self.date0.strftime('%Y-%m-%d %H:%M:%S%z (%Z)'))
		#fdate = ncf.createVariable('date', np.dtype('float64').char, ('time',))
		#fdate.axis = 'T'
		#fdate.calendar = 'standard'
		#fdate.long_name = 'date'
		#fdate.standard_name = 'date'
		#fdate.units = 'days since 1950-01-01 00:00:00'
		falts = ncf.createVariable('altitude', 'f8', ('altitude',))
		falts.axis = 'Z'
		falts.long_name = 'altitude'
		falts.standard_name = 'altitude'
		falts.units = 'km'
		falts.positive = 'up'
		flats = ncf.createVariable('latitude', 'f8', ('latitude',))
		flats.axis = 'Y'
		flats.long_name = 'latitude'
		flats.standard_name = 'latitude'
		flats.units = 'degrees_north'
		flons = ncf.createVariable('longitude', 'f8', ('time', 'latitude',))
		flons.long_name = 'longitude'
		flons.standard_name = 'longitude'
		flons.units = 'degrees_east'
		fdens = ncf.createVariable('%s_DENS' % self.name, 'f8', ('time', 'latitude', 'altitude'))
		fdens.units = 'cm^{-3}'
		fdens.long_name = '%s number density' % self.name
		ferrs = ncf.createVariable('%s_ERR' % self.name, 'f8', ('time', 'latitude', 'altitude'))
		ferrs.units = 'cm^{-3}'
		ferrs.long_name = '%s density measurement error' % self.name
		fetot = ncf.createVariable('%s_ETOT' % self.name, 'f8', ('time', 'latitude', 'altitude'))
		fetot.units = 'cm^{-3}'
		fetot.long_name = '%s density total error' % self.name
		frstd = ncf.createVariable('%s_RSTD' % self.name, 'f8', ('time', 'latitude', 'altitude'))
		frstd.units = '%'
		frstd.long_name = '%s relative standard deviation' % self.name
		fakd = ncf.createVariable('%s_AKDIAG' % self.name, 'f8', ('time', 'latitude', 'altitude'))
		fakd.units = '1'
		fakd.long_name = '%s averaging kernel diagonal element' % self.name
		fapri = ncf.createVariable('%s_APRIORI' % self.name, 'f8', ('time', 'latitude', 'altitude'))
		fapri.units = 'cm^{-3}'
		fapri.long_name = '%s apriori density' % self.name
		ftemp = ncf.createVariable(_var_dicts[self.version]["temperature"]["name"],
				'f8', ('time', 'latitude', 'altitude'))
		ftemp.long_name = _var_dicts[self.version]["temperature"]["long_name"]
		ftemp.model = 'NRLMSIS-00'
		ftemp.units = 'K'
		fdens_tot = ncf.createVariable(_var_dicts[self.version]["dens_tot"]["name"],
				'f8', ('time', 'latitude', 'altitude'))
		fdens_tot.long_name = _var_dicts[self.version]["dens_tot"]["long_name"]
		fdens_tot.units = 'cm^{-3}'
		if _var_dicts[self.version]["dens_tot"]["model"] is not None:
			fdens_tot.model = _var_dicts[self.version]["dens_tot"]["model"]
		fnoem = ncf.createVariable('%s_NOEM' % self.name, 'f8', ('time', 'latitude', 'altitude'))
		fnoem.units = 'cm^{-3}'
		fnoem.long_name = 'NOEM %s number density' % self.name
		fvmr = ncf.createVariable('%s_VMR' % self.name, 'f8', ('time', 'latitude', 'altitude'))
		fvmr.units = 'ppb'
		fvmr.long_name = '%s volume mixing ratio' % self.name
		flst = ncf.createVariable('app_LST', 'f8', ('time', 'latitude'))
		flst.units = 'hours'
		flst.long_name = 'apparent local solar time'
		fmst = ncf.createVariable('mean_LST', 'f8', ('time', 'latitude'))
		fmst.units = 'hours'
		fmst.long_name = 'mean local solar time'
		fsza = ncf.createVariable('mean_SZA', 'f8', ('time', 'latitude'))
		fsza.units = 'degrees'
		fsza.long_name = 'solar zenith angle at mean altitude'
		futc = ncf.createVariable('UTC', 'f8', ('time', 'latitude'))
		futc.units = 'hours'
		futc.long_name = 'measurement utc time'
		futcd = ncf.createVariable('utc_days', 'f8', ('time', 'latitude'))
		futcd.long_name = 'measurement utc day'
		futcd.units = 'days since {0}'.format(self.date0.isoformat(sep=' '))

		fgmlats = ncf.createVariable('gm_lats', 'f8', ('time', 'latitude',))
		fgmlats.long_name = 'geomagnetic_latitude'
		fgmlats.model = 'IGRF'
		fgmlats.units = 'degrees_north'

		fgmlons = ncf.createVariable('gm_lons', 'f8', ('time', 'latitude',))
		fgmlons.long_name = 'geomagnetic_longitude'
		fgmlons.model = 'IGRF'
		fgmlons.units = 'degrees_east'

		faacgmgmlats = ncf.createVariable('aacgm_gm_lats', 'f8', ('time', 'latitude',))
		faacgmgmlats.long_name = 'geomagnetic_latitude'
		faacgmgmlats.model = 'AACGM'
		faacgmgmlats.units = 'degrees_north'

		faacgmgmlons = ncf.createVariable('aacgm_gm_lons', 'f8', ('time', 'latitude',))
		faacgmgmlons.long_name = 'geomagnetic_longitude'
		faacgmgmlons.model = 'AACGM'
		faacgmgmlons.units = 'degrees_east'

		forbit[:] = o
		ftime[:] = d
		falts[:] = self.alts
		flats[:] = self.lats
		flons[:] = self.lons
		fdens[:] = np.ma.atleast_3d(self.no_dens[:])
		ferrs[:] = np.ma.atleast_3d(self.no_errs[:])
		fetot[:] = np.ma.atleast_3d(self.no_etot[:])
		frstd[:] = np.ma.atleast_3d(self.no_rstd[:])
		fakd[:] = np.ma.atleast_3d(self.no_akd[:])
		fapri[:] = np.ma.atleast_3d(self.no_apri[:])
		ftemp[:] = np.ma.atleast_3d(self.temperature[:])
		fnoem[:] = np.ma.atleast_3d(self.no_noem[:])
		fdens_tot[:] = np.ma.atleast_3d(self.tot_dens[:])
		fvmr[:] = np.ma.atleast_3d(self.no_vmr[:])
		flst[:] = np.ma.atleast_2d(self.lst[:])
		fmst[:] = np.ma.atleast_2d(self.mst[:])
		fsza[:] = np.ma.atleast_2d(self.sza[:])
		futc[:] = np.ma.atleast_2d(self.utchour[:])
		futcd[:] = np.ma.atleast_2d(self.utcdays[:])
		fgmlats[:] = np.ma.atleast_2d(self.gmlats[:])
		fgmlons[:] = np.ma.atleast_2d(self.gmlons[:])
		faacgmgmlats[:] = np.ma.atleast_2d(self.aacgmgmlats[:])
		faacgmgmlons[:] = np.ma.atleast_2d(self.aacgmgmlons[:])
		ncf.close()

	def to_xarray(self):
		"""Convert the combined orbit data to :class:`xarray.Dataset`

		Exports the data using the same data variables as
		when writing to netcdf.

		Returns
		-------
		dataset: xarray.Dataset
		"""
		try:
			import xarray as xr
		except ImportError:
			print("Error: xarray not available!")
			return None
		o = np.asarray(self.orbit)
		d = np.asarray(self.date)

		xr_dens = xr.DataArray(self.no_dens, coords=[d, self.lats, self.alts],
				dims=["time", "latitude", "altitude"],
				attrs={"units": "cm^{-3}",
						"long_name": "{0} number density".format(self.name)},
				name="{0}_DENS".format(self.name))

		xr_errs = xr.DataArray(self.no_errs, coords=[d, self.lats, self.alts],
				dims=["time", "latitude", "altitude"],
				attrs={"units": "cm^{-3}",
						"long_name": "{0} density measurement error".format(self.name)},
				name="{0}_ERR".format(self.name))

		xr_etot = xr.DataArray(self.no_etot, coords=[d, self.lats, self.alts],
				dims=["time", "latitude", "altitude"],
				attrs={"units": "cm^{-3}",
						"long_name": "{0} density total error".format(self.name)},
				name="{0}_ETOT".format(self.name))

		xr_rstd = xr.DataArray(self.no_rstd, coords=[d, self.lats, self.alts],
				dims=["time", "latitude", "altitude"],
				attrs=dict(units='%',
						long_name='{0} relative standard deviation'.format(self.name)),
				name="{0}_RSTD".format(self.name))

		xr_akd = xr.DataArray(self.no_akd, coords=[d, self.lats, self.alts],
				dims=["time", "latitude", "altitude"],
				attrs=dict(units='1',
						long_name='{0} averaging kernel diagonal element'.format(self.name)),
				name="{0}_AKDIAG".format(self.name))

		xr_apri = xr.DataArray(self.no_apri, coords=[d, self.lats, self.alts],
				dims=["time", "latitude", "altitude"],
				attrs=dict(units='cm^{-3}', long_name='{0} apriori density'.format(self.name)),
				name="{0}_APRIORI".format(self.name))

		xr_noem = xr.DataArray(self.no_noem, coords=[d, self.lats, self.alts],
				dims=["time", "latitude", "altitude"],
				attrs=dict(units='cm^{-3}', long_name='NOEM {0} number density'.format(self.name)),
				name="{0}_NOEM".format(self.name))

		xr_vmr = xr.DataArray(self.no_vmr, coords=[d, self.lats, self.alts],
				dims=["time", "latitude", "altitude"],
				attrs=dict(units='ppb', long_name='{0} volume mixing ratio'.format(self.name)),
				name="{0}_VMR".format(self.name))

		xr_dtot = xr.DataArray(self.tot_dens, coords=[d, self.lats, self.alts],
				dims=["time", "latitude", "altitude"],
				attrs=dict(units='cm^{-3}', long_name='MSIS total number density',
					model='NRLMSIS-00'),
				name="MSIS_Dens")

		xr_temp = xr.DataArray(self.temperature, coords=[d, self.lats, self.alts],
				dims=["time", "latitude", "altitude"],
				attrs=dict(units='K', long_name='MSIS temperature',
						model="NRLMSIS-00"),
				name="MSIS_Temp")

		xr_lons = xr.DataArray(self.lons, coords=[d, self.lats],
				dims=["time", "latitude"],
				attrs=dict(long_name='longitude', standard_name='longitude',
						units='degrees_east'),
				name='longitude')

		xr_lst = xr.DataArray(self.lst, coords=[d, self.lats],
				dims=["time", "latitude"],
				attrs=dict(units='hours', long_name='apparent local solar time'),
				name="app_LST")

		xr_mst = xr.DataArray(self.mst, coords=[d, self.lats],
				dims=["time", "latitude"],
				attrs=dict(units='hours', long_name='mean local solar time'),
				name="mean_LST")

		xr_sza = xr.DataArray(self.sza, coords=[d, self.lats],
				dims=["time", "latitude"],
				attrs=dict(units='degrees',
						long_name='solar zenith angle at mean altitude'),
				name="mean_SZA")

		xr_utch = xr.DataArray(self.utchour, coords=[d, self.lats],
				dims=["time", "latitude"],
				attrs=dict(units='hours',
						long_name='measurement utc time'),
				name="UTC")

		xr_utcd = xr.DataArray(self.utcdays, coords=[d, self.lats],
				dims=["time", "latitude"],
				attrs=dict(units='days since {0}'.format(self.date0.isoformat(sep=' ')),
						long_name='measurement utc day'),
				name="utc_days")

		xr_gmlats = xr.DataArray(self.gmlats, coords=[d, self.lats],
				dims=["time", "latitude"],
				attrs=dict(long_name='geomagnetic_latitude',
						model='IGRF', units='degrees_north'),
				name="gm_lats")

		xr_gmlons = xr.DataArray(self.gmlons, coords=[d, self.lats],
				dims=["time", "latitude"],
				attrs=dict(long_name='geomagnetic_longitude',
						model='IGRF', units='degrees_east'),
				name="gm_lons")

		xr_aacgmgmlats = xr.DataArray(self.aacgmgmlats, coords=[d, self.lats],
				dims=["time", "latitude"],
				attrs=dict(long_name='geomagnetic_latitude',
						model='AACGM', units='degrees_north'),
				name="aacgm_gm_lats")

		xr_aacgmgmlons = xr.DataArray(self.aacgmgmlons, coords=[d, self.lats],
				dims=["time", "latitude"],
				attrs=dict(long_name='geomagnetic_longitude',
						model='AACGM', units='degrees_east'),
				name="aacgm_gm_lons")

		xr_orbit = xr.DataArray(o, coords=[d], dims=["time"],
				attrs=dict(axis='T', calendar='standard', long_name='orbit',
						standard_name='orbit', units='orbit number'),
				name="orbit")

		xr_ds = xr.Dataset({da.name: da for da in
				[xr_dens, xr_errs, xr_etot, xr_rstd, xr_akd, xr_apri, xr_noem,
					xr_vmr, xr_dtot, xr_temp, xr_lons, xr_lst, xr_mst, xr_sza,
					xr_utch, xr_utcd, xr_gmlats, xr_gmlons, xr_aacgmgmlats,
					xr_aacgmgmlons, xr_orbit]})

		xr_ds["time"].attrs = dict(axis='T', calendar='standard',
						long_name='equatorial crossing time',
						standard_name='time',
						units='days since {0}'.format(self.date0.isoformat(sep=' ')))

		xr_ds["altitude"].attrs = dict(axis='Z', long_name='altitude',
				standard_name='altitude', units='km', positive='up')

		xr_ds["latitude"].attrs = dict(axis='Y', long_name='latitude',
				standard_name='latitude', units='degrees_north')

		if self.version is not None:
			xr_ds.attrs["version"] = self.version
		if self.data_version is not None:
			xr_ds.attrs["L2_data_version"] = self.data_version
		xr_ds.attrs["software"] = "sciapy {0}".format(__version__)
		xr_ds.attrs["creation_time"] = dt.datetime.utcnow().strftime("%a %b %d %Y %H:%M:%S +00:00 (UTC)")
		xr_ds.attrs["author"] = self.author

		return xr_ds
