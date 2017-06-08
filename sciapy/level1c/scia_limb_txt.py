#!/usr/bin/env python
# vim: set fileencoding=utf-8
"""SCIAMACHY level 1c limb spectra text interface

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
_int_type = np.int32

def _print_indent(fp, indent):
	for i in range(indent):
		print(" ", end="", file=fp)

def _print_array(fp, a, indent):
	_print_indent(fp, indent)
	print('\t'.join(map(str, a)), file=fp)

# format strings taken from scia_L1C_ascii.c
def _print_array1(fp, a, padding=0):
	if padding > 0:
		_print_indent(fp, padding)
	print(''.join(map(lambda x: str('%12.3f ' % x), a)), file=fp)

def _print_array2(fp, a):
	# print(' '.join(map(lambda x: str('% -7.5e' % x), a)), file=fp)
	print(''.join(map(lambda x: str('%12.5e ' % x), a)), file=fp)


def read_from_textfile(self, filename):
	"""SCIAMACHY level 1c limb scan text import

	Parameters
	----------
	filename : string
		The (plain ascii) table text filename to read the data from.

	Returns
	-------
	nothing
	"""
	if hasattr(filename, 'seek'):
		f = filename
	else:
		f = open(filename, 'rb')
	h_list = []
	nh = int(f.readline())
	for i in range(0, nh):
		h_list.append(bytes(f.readline()).decode().rstrip())
	self.textheader_length = nh
	self.textheader = '\n'.join(h_list)
	self.parse_textheader()
	self.nalt, self.npix = np.fromstring(f.readline(), dtype=_int_type, sep=' ')

	self.orbit_state = np.fromstring(f.readline(), dtype=_int_type, sep=' ')
	(self.orbit, self.state_in_orbit, self.state_id,
		self.profiles_per_state, self.profile_in_state) = self.orbit_state
	self.date = np.fromstring(f.readline(), dtype=_int_type, sep=' ')
	if nh > 27:
		self.sub_sat_lat_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
		self.sub_sat_lon_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
		if nh > 29:
			self.orbit_phase = np.fromstring(f.readline(), dtype=_float_type, sep=' ')[0]
	self.cent_lat_lon = np.fromstring(f.readline(), dtype=_float_type, sep=' ')

	self.tp_lat_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.tp_lon_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.tp_alt_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.tp_sza_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.tp_saa_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.tp_los_zenith_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.toa_sza_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.toa_saa_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.toa_los_zenith_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.sat_sza_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.sat_saa_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.sat_los_zenith_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.sat_alt_list = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.earthradii = np.fromstring(f.readline(), dtype=_float_type, sep=' ')

	tmp_list = []
	for i in range(self.npix):
		input = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
		self.wls.append(input[0])
		tmp_list.append(input[1:])
	tmp_rad_arr = np.asarray(tmp_list).reshape(self.npix, self.nalt).transpose()
	tmp_list[:] = []
	line = f.readline().strip()
	if bytes(line) == b"ERRORS":
		for i in range(self.npix):
			input = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
			tmp_list.append(input[1:])
	else:
		for i in range(self.npix):
			tmp_list.append(np.zeros(self.nalt))
	tmp_err_arr = np.asarray(tmp_list).reshape(self.npix, self.nalt).transpose()
	for i in range(self.nalt):
		self.rad_list.append(tmp_rad_arr[i])
		self.err_list.append(tmp_err_arr[i])

def write_to_textfile(self, filename):
	"""SCIAMACHY level 1c limb scan text export

	Parameters
	----------
	filename : string
		The (plain ascii) table text filename to write the data to.

	Returns
	-------
	nothing
	"""
	if hasattr(filename, 'seek'):
		f = filename
	else:
		f = open(filename, 'w')
	print(self.textheader_length, file=f)
	print(self.textheader, file=f)
	# format strings taken from scia_L1C_ascii.c
	print("%2d %4d" % (self.nalt, self.npix), file=f)
	print("%05d %2d %2d %2d %2d" % tuple(self.orbit_state), file=f)
	print("%4d %2d %2d %2d %2d %2d" % tuple(self.date), file=f)
	if self.textheader_length > 27:
		_print_array1(f, self.sub_sat_lat_list, 9)
		_print_array1(f, self.sub_sat_lon_list, 9)
	if self.textheader_length > 29:
		_print_indent(f, 9)
		print("%12.3f " % (self.orbit_phase,), file=f)
	_print_indent(f, 9)
	print("%8.3f %8.3f    %8.3f %8.3f  %8.3f %8.3f  "
			"%8.3f %8.3f %8.3f %8.3f" % tuple(self.cent_lat_lon), file=f)
	_print_array1(f, self.tp_lat_list, 9)
	_print_array1(f, self.tp_lon_list, 9)
	_print_array1(f, self.tp_alt_list, 9)
	_print_array1(f, self.tp_sza_list, 9)
	_print_array1(f, self.tp_saa_list, 9)
	_print_array1(f, self.tp_los_zenith_list, 9)
	_print_array1(f, self.toa_sza_list, 9)
	_print_array1(f, self.toa_saa_list, 9)
	_print_array1(f, self.toa_los_zenith_list, 9)
	_print_array1(f, self.sat_sza_list, 9)
	_print_array1(f, self.sat_saa_list, 9)
	_print_array1(f, self.sat_los_zenith_list, 9)
	_print_array1(f, self.sat_alt_list, 9)
	_print_array1(f, self.earthradii, 9)

	rads = np.asarray(self.rad_list).reshape(self.nalt, self.npix).transpose()
	errs = np.asarray(self.err_list).reshape(self.nalt, self.npix).transpose()

	# format strings taken from scia_L1C_ascii.c
	for i in range(self.npix):
		print("%9.4f" % self.wls[i], end=" ", file=f)
		_print_array2(f, rads[i])

	print("ERRORS", file=f)

	for i in range(self.npix):
		print("%9.4f" % self.wls[i], end=" ", file=f)
		_print_array2(f, errs[i])

	f.close()
