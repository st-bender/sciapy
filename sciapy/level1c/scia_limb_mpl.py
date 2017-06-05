#!/usr/bin/env python
# vim: set fileencoding=utf-8
"""SCIAMACHY level 1c limb spectra binary interface

Copyright (c) 2014-2017 Stefan Bender

This file is part of sciapy.
sciapy is free software: you can redistribute it or modify it
under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.
See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""

from __future__ import absolute_import, division, print_function

import numpy as np

_float_type = np.float32

def _write_padded_string(fp, s, padding):
	s = s.encode('ascii', "ignore")
	count = padding - len(s)
	fp.write(s)
	# pad
	fp.write(b'\x00' * count)

def _read_single_float(fp):
	ret = np.fromstring(fp.read(4), dtype=_float_type)[0]
	return ret

def _write_int_to_binary(fp, a):
	fp.write(np.asarray(a, dtype=np.int32).tostring())

def _write_float_to_binary(fp, a):
	fp.write(np.asarray(a, dtype=_float_type).tostring())


def read_from_mpl_binary(self, filename):
	"""SCIAMACHY level 1c limb scan binary import

	Parameters
	----------
	filename : string
		The binary filename to read the data from.

	Returns
	-------
	nothing
	"""
	if hasattr(filename, 'seek'):
		f = filename
	else:
		f = open(filename, 'rb')
	hlen = 100
	# the first bytes of the first 100 header bytes are
	# the number of header lines that follow
	nline = ""
	j = 0
	flag = 0
	while j < hlen:
		char = bytes(f.read(1))
		if char == b'\n':
			# we have a usual text file, abort binary reading.
			raise ValueError
		j += 1
		if char and char != b'\x00' and flag == 0:
			nline += char.decode()
		else:
			flag = 1

	self.textheader_length = int(''.join(nline))

	h_list = []
	for i in range(self.textheader_length):
		line = ""
		j = 0
		flag = 0
		while j < hlen:
			char = bytes(f.read(1))
			j += 1
			if char and char != b'\x00' and flag == 0:
				line += char.decode()
			else:
				flag = 1
		h_list.append(line.rstrip())

	self.textheader = '\n'.join(h_list)

	# global data
	self.nalt = np.fromstring(f.read(4), dtype=np.int32)[0]
	self.npix = np.fromstring(f.read(4), dtype=np.int32)[0]
	self.orbit_state = np.fromstring(f.read(4 * 5), dtype=np.int32)
	(self.orbit, self.state_in_orbit, self.state_id,
		self.profiles_per_state, self.profile_in_state) = self.orbit_state
	self.date = np.fromstring(f.read(4 * 6), dtype=np.int32)
	self.cent_lat_lon = np.fromstring(f.read(4 * 10), dtype=_float_type)
	if self.textheader_length > 29:
		self.orbit_phase = np.fromstring(f.read(4), dtype=_float_type)[0]

	self.wls = np.fromstring(f.read(4 * self.npix), dtype=_float_type)

	for i in range(self.nalt):
		if self.textheader_length > 27:
			self.sub_sat_lat_list.append(_read_single_float(f))
			self.sub_sat_lon_list.append(_read_single_float(f))
		self.tp_lat_list.append(_read_single_float(f))
		self.tp_lon_list.append(_read_single_float(f))
		self.tp_alt_list.append(_read_single_float(f))
		self.tp_sza_list.append(_read_single_float(f))
		self.tp_saa_list.append(_read_single_float(f))
		self.tp_los_zenith_list.append(_read_single_float(f))
		self.toa_sza_list.append(_read_single_float(f))
		self.toa_saa_list.append(_read_single_float(f))
		self.toa_los_zenith_list.append(_read_single_float(f))
		self.sat_sza_list.append(_read_single_float(f))
		self.sat_saa_list.append(_read_single_float(f))
		self.sat_los_zenith_list.append(_read_single_float(f))
		self.sat_alt_list.append(_read_single_float(f))
		self.earthradii.append(_read_single_float(f))

		self.rad_list.append(np.fromstring(f.read(4 * self.npix), dtype=_float_type))
		self.err_list.append(np.fromstring(f.read(4 * self.npix), dtype=_float_type))

def write_to_mpl_binary(self, filename):
	"""SCIAMACHY level 1c limb scan binary export

	Parameters
	----------
	filename : string
		The binary filename to write the data to.

	Returns
	-------
	nothing
	"""
	if hasattr(filename, 'seek'):
		f = filename
	else:
		f = open(filename, 'wb')

	# write out the padded header first
	bufsize = 100
	_write_padded_string(f, str(self.textheader_length), bufsize)
	h_list = self.textheader.split('\n')
	for h_line in h_list:
		_write_padded_string(f, h_line, bufsize)

	_write_int_to_binary(f, self.nalt)
	_write_int_to_binary(f, self.npix)

	_write_int_to_binary(f, self.orbit_state)
	_write_int_to_binary(f, self.date)
	_write_float_to_binary(f, self.cent_lat_lon)
	if self.textheader_length > 29:
		_write_float_to_binary(f, self.orbit_phase)

	_write_float_to_binary(f, self.wls)

	for i in range(self.nalt):
		if self.textheader_length > 27:
			_write_float_to_binary(f, self.sub_sat_lat_list[i])
			_write_float_to_binary(f, self.sub_sat_lon_list[i])
		_write_float_to_binary(f, self.tp_lat_list[i])
		_write_float_to_binary(f, self.tp_lon_list[i])
		_write_float_to_binary(f, self.tp_alt_list[i])
		_write_float_to_binary(f, self.tp_sza_list[i])
		_write_float_to_binary(f, self.tp_saa_list[i])
		_write_float_to_binary(f, self.tp_los_zenith_list[i])
		_write_float_to_binary(f, self.toa_sza_list[i])
		_write_float_to_binary(f, self.toa_saa_list[i])
		_write_float_to_binary(f, self.toa_los_zenith_list[i])
		_write_float_to_binary(f, self.sat_sza_list[i])
		_write_float_to_binary(f, self.sat_saa_list[i])
		_write_float_to_binary(f, self.sat_los_zenith_list[i])
		_write_float_to_binary(f, self.sat_alt_list[i])
		_write_float_to_binary(f, self.earthradii[i])

		_write_float_to_binary(f, self.rad_list[i])
		_write_float_to_binary(f, self.err_list[i])

	f.close()
