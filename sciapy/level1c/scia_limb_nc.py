# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2014-2017 Stefan Bender
#
# This file is part of sciapy.
# sciapy is free software: you can redistribute it or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY level 1c limb spectra netcdf interface
"""

from __future__ import absolute_import, division, print_function

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

from ._types import _limb_data_dtype, _try_decode

def read_from_netcdf(self, filename):
	"""SCIAMACHY level 1c limb scan netcdf import

	Parameters
	----------
	filename : str
		The netcdf filename to read the data from.

	Returns
	-------
	nothing
	"""
	import numpy.lib.recfunctions as rfn

	ncf = netcdf_file(filename, 'r')

	self.textheader_length = ncf.textheader_length
	self.textheader = _try_decode(ncf.textheader)

	self.orbit_state = ncf.orbit_state
	(self.orbit, self.state_in_orbit, self.state_id,
		self.profiles_per_state, self.profile_in_state) = self.orbit_state
	self.date = ncf.date
	self.cent_lat_lon = ncf.cent_lat_lon
	self.orbit_phase = ncf.orbit_phase

	try:
		self.nalt = ncf.dimensions['limb'].size
		self.npix = ncf.dimensions['wavelength'].size
	except:
		self.nalt = ncf.dimensions['limb']
		self.npix = ncf.dimensions['wavelength']

	self.wls = ncf.variables['wavelength'][:].copy()

	# pre-set the limb_data
	if self._limb_data_dtype is None:
		self._limb_data_dtype = _limb_data_dtype[:]
	self.limb_data = np.zeros((self.nalt), dtype=self._limb_data_dtype)

	self.limb_data["sub_sat_lat"] = ncf.variables['sub_sat_lat'][:].copy()
	self.limb_data["sub_sat_lon"] = ncf.variables['sub_sat_lon'][:].copy()
	self.limb_data["tp_lat"] = ncf.variables['TP latitude'][:].copy()
	self.limb_data["tp_lon"] = ncf.variables['TP longitude'][:].copy()
	self.limb_data["tp_alt"] = ncf.variables['TP altitude'][:].copy()
	self.limb_data["tp_sza"] = ncf.variables['TP SZA'][:].copy()
	self.limb_data["tp_saa"] = ncf.variables['TP SAA'][:].copy()
	self.limb_data["tp_los"] = ncf.variables['TP LOS Zenith'][:].copy()
	self.limb_data["toa_sza"] = ncf.variables['TOA SZA'][:].copy()
	self.limb_data["toa_saa"] = ncf.variables['TOA SAA'][:].copy()
	self.limb_data["toa_los"] = ncf.variables['TOA LOS Zenith'][:].copy()
	self.limb_data["sat_sza"] = ncf.variables['SAT SZA'][:].copy()
	self.limb_data["sat_saa"] = ncf.variables['SAT SAA'][:].copy()
	self.limb_data["sat_los"] = ncf.variables['SAT LOS Zenith'][:].copy()
	self.limb_data["sat_alt"] = ncf.variables['SAT altitude'][:].copy()
	self.limb_data["earth_rad"] = ncf.variables['earthradius'][:].copy()

	tmp_rad_arr = list(ncf.variables['radiance'][:].copy())
	tmp_err_arr = list(ncf.variables['radiance errors'][:].copy())

	# save to limb_data recarray
	rads = np.rec.fromarrays([tmp_rad_arr],
				dtype=np.dtype([("rad", 'f4', (self.npix,))]))
	errs = np.rec.fromarrays([tmp_err_arr],
				dtype=np.dtype([("err", 'f4', (self.npix,))]))
	self.limb_data = rfn.merge_arrays([self.limb_data, rads, errs],
			usemask=False, asrecarray=True, flatten=True)
	self._limb_data_dtype = self.limb_data.dtype

	if hasattr(ncf, "_attributes"):
		# scipy.io.netcdf / pupynere
		ncattrs = ncf._attributes.keys()
	else:
		# netcdf4
		ncattrs = ncf.ncattrs()
	for _k in ncattrs:
		if _k.startswith("metadata"):
			_meta_key = _k.split("::")[1]
			_att = getattr(ncf, _k)
			self.metadata[_meta_key] = _try_decode(_att)
	ncf.close()

def write_to_netcdf(self, filename):
	"""SCIAMACHY level 1c limb scan netcdf export

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

	ncf.orbit_state = self.orbit_state
	ncf.date = self.date
	ncf.cent_lat_lon = self.cent_lat_lon
	ncf.orbit_phase = self.orbit_phase

	ncf.createDimension('limb', self.nalt)
	ncf.createDimension('wavelength', self.npix)

	wavs = ncf.createVariable('wavelength', np.dtype('float32').char, ('wavelength',))
	wavs.units = 'nm'
	wavs[:] = np.asarray(self.wls)

	sslat = ncf.createVariable('sub_sat_lat', np.dtype('float32').char, ('limb',))
	sslat.units = 'deg'
	sslat[:] = np.asarray(self.limb_data["sub_sat_lat"])
	sslon = ncf.createVariable('sub_sat_lon', np.dtype('float32').char, ('limb',))
	sslon.units = 'deg'
	sslon[:] = np.asarray(self.limb_data["sub_sat_lon"])
	tp_lats = ncf.createVariable('TP latitude', np.dtype('float32').char, ('limb',))
	tp_lats.units = 'deg'
	tp_lats[:] = np.asarray(self.limb_data["tp_lat"])
	tp_lons = ncf.createVariable('TP longitude', np.dtype('float32').char, ('limb',))
	tp_lons.units = 'deg'
	tp_lons[:] = np.asarray(self.limb_data["tp_lon"])
	tp_alts = ncf.createVariable('TP altitude', np.dtype('float32').char, ('limb',))
	tp_alts.units = 'km'
	tp_alts[:] = np.asarray(self.limb_data["tp_alt"])
	tp_szas = ncf.createVariable('TP SZA', np.dtype('float32').char, ('limb',))
	tp_szas.units = 'deg'
	tp_szas[:] = np.asarray(self.limb_data["tp_sza"])
	tp_saas = ncf.createVariable('TP SAA', np.dtype('float32').char, ('limb',))
	tp_saas.units = 'deg'
	tp_saas[:] = np.asarray(self.limb_data["tp_saa"])
	tp_los_zeniths = ncf.createVariable('TP LOS Zenith', np.dtype('float32').char, ('limb',))
	tp_los_zeniths.units = 'deg'
	tp_los_zeniths[:] = np.asarray(self.limb_data["tp_los"])
	toa_szas = ncf.createVariable('TOA SZA', np.dtype('float32').char, ('limb',))
	toa_szas.units = 'deg'
	toa_szas[:] = np.asarray(self.limb_data["toa_sza"])
	toa_saas = ncf.createVariable('TOA SAA', np.dtype('float32').char, ('limb',))
	toa_saas.units = 'deg'
	toa_saas[:] = np.asarray(self.limb_data["toa_saa"])
	toa_los_zeniths = ncf.createVariable('TOA LOS Zenith', np.dtype('float32').char, ('limb',))
	toa_los_zeniths.units = 'deg'
	toa_los_zeniths[:] = np.asarray(self.limb_data["toa_los"])
	sat_szas = ncf.createVariable('SAT SZA', np.dtype('float32').char, ('limb',))
	sat_szas.units = 'deg'
	sat_szas[:] = np.asarray(self.limb_data["sat_sza"])
	sat_saas = ncf.createVariable('SAT SAA', np.dtype('float32').char, ('limb',))
	sat_saas.units = 'deg'
	sat_saas[:] = np.asarray(self.limb_data["sat_saa"])
	sat_los_zeniths = ncf.createVariable('SAT LOS Zenith', np.dtype('float32').char, ('limb',))
	sat_los_zeniths.units = 'deg'
	sat_los_zeniths[:] = np.asarray(self.limb_data["sat_los"])
	sat_alts = ncf.createVariable('SAT altitude', np.dtype('float32').char, ('limb',))
	sat_alts.units = 'km'
	sat_alts[:] = np.asarray(self.limb_data["sat_alt"])
	eradii_alts = ncf.createVariable('earthradius', np.dtype('float32').char, ('limb',))
	eradii_alts.units = 'km'
	eradii_alts[:] = np.asarray(self.limb_data["earth_rad"])

	try:
		rads = ncf.createVariable('radiance', np.dtype('float32').char,
				('limb', 'wavelength'), zlib=True, complevel=1)
		errs = ncf.createVariable('radiance errors', np.dtype('float32').char,
				('limb', 'wavelength'), zlib=True, complevel=1)
	except TypeError:
		rads = ncf.createVariable('radiance', np.dtype('float32').char,
				('limb', 'wavelength'))
		errs = ncf.createVariable('radiance errors', np.dtype('float32').char,
				('limb', 'wavelength'))
	rads.units = 'ph / s / cm^2 / nm'
	errs.units = 'ph / s / cm^2 / nm'
	rads[:] = np.asarray(self.limb_data["rad"]).reshape(self.nalt, self.npix)
	errs[:] = np.asarray(self.limb_data["err"]).reshape(self.nalt, self.npix)

	for _k, _v in self.metadata.items():
		setattr(ncf, "metadata::" + _k, _v)
	ncf.close()
