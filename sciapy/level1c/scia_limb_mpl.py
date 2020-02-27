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
"""SCIAMACHY level 1c limb spectra binary interface
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from ._types import _float_type, _int_type, _limb_data_dtype

def _write_padded_string(fp, s, padding):
	s = s.encode('ascii', "ignore")
	count = padding - len(s)
	fp.write(s)
	# pad
	fp.write(b'\x00' * count)

def _read_single_float(fp):
	ret = np.frombuffer(fp.read(4), dtype=_float_type)[0]
	return ret

def _write_int_to_binary(fp, a):
	fp.write(np.asarray(a, dtype=_int_type).tostring())

def _write_float_to_binary(fp, a):
	fp.write(np.asarray(a, dtype=_float_type).tostring())


def read_from_mpl_binary(self, filename):
	"""SCIAMACHY level 1c limb scan binary import

	Parameters
	----------
	filename : str
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
	for _ in range(self.textheader_length):
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
	self.parse_textheader()

	# global data
	self.nalt = np.frombuffer(f.read(4), dtype=_int_type)[0]
	self.npix = np.frombuffer(f.read(4), dtype=_int_type)[0]
	self.orbit_state = np.frombuffer(f.read(4 * 5), dtype=_int_type)
	(self.orbit, self.state_in_orbit, self.state_id,
		self.profiles_per_state, self.profile_in_state) = self.orbit_state
	self.date = np.frombuffer(f.read(4 * 6), dtype=_int_type)
	self.cent_lat_lon = np.frombuffer(f.read(4 * 10), dtype=_float_type)
	if self.textheader_length > 29:
		self.orbit_phase = np.frombuffer(f.read(4), dtype=_float_type)[0]

	self.wls = np.frombuffer(f.read(4 * self.npix), dtype=_float_type)

	if self._limb_data_dtype is None:
		self._limb_data_dtype = _limb_data_dtype[:]
		if self.textheader_length < 28:
			self._limb_data_dtype.remove(("sub_sat_lat", _float_type))
			self._limb_data_dtype.remove(("sub_sat_lon", _float_type))

		self._limb_data_dtype.append(("rad", _float_type, (self.npix)))
		self._limb_data_dtype.append(("err", _float_type, (self.npix)))

	self.limb_data = np.fromfile(f, dtype=np.dtype(self._limb_data_dtype),
			count=self.nalt).view(type=np.recarray)

def write_to_mpl_binary(self, filename):
	"""SCIAMACHY level 1c limb scan binary export

	Parameters
	----------
	filename : str
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

	# write the data as is, the dtype should take care of
	# all the formatting.
	self.limb_data.tofile(f)

	f.close()
