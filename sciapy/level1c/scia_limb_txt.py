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
"""SCIAMACHY level 1c limb spectra text interface
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from ._types import _float_type, _int_type, _limb_data_dtype

def _print_indent(fp, indent):
	for _ in range(indent):
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
	filename : str
		The (plain ascii) table text filename to read the data from.

	Returns
	-------
	nothing
	"""
	import numpy.lib.recfunctions as rfn
	if hasattr(filename, 'seek'):
		f = filename
	else:
		f = open(filename, 'rb')
	h_list = []
	nh = int(f.readline())
	for _ in range(nh):
		h_list.append(bytes(f.readline()).decode().rstrip())
	self.textheader_length = nh
	self.textheader = '\n'.join(h_list)
	self.parse_textheader()
	self.nalt, self.npix = np.fromstring(f.readline(), dtype=_int_type, sep=' ')

	self.orbit_state = np.fromstring(f.readline(), dtype=_int_type, sep=' ')
	(self.orbit, self.state_in_orbit, self.state_id,
		self.profiles_per_state, self.profile_in_state) = self.orbit_state
	self.date = np.fromstring(f.readline(), dtype=_int_type, sep=' ')
	# pre-set the limb_data
	if self._limb_data_dtype is None:
		self._limb_data_dtype = _limb_data_dtype[:]
	self.limb_data = np.zeros((self.nalt), dtype=self._limb_data_dtype)
	if nh > 27:
		self.limb_data["sub_sat_lat"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
		self.limb_data["sub_sat_lon"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
		if nh > 29:
			self.orbit_phase = np.fromstring(f.readline(), dtype=_float_type, sep=' ')[0]
	self.cent_lat_lon = np.fromstring(f.readline(), dtype=_float_type, sep=' ')

	self.limb_data["tp_lat"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["tp_lon"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["tp_alt"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["tp_sza"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["tp_saa"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["tp_los"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["toa_sza"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["toa_saa"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["toa_los"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["sat_sza"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["sat_saa"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["sat_los"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["sat_alt"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
	self.limb_data["earth_rad"] = np.fromstring(f.readline(), dtype=_float_type, sep=' ')

	tmp_list = []
	for _ in range(self.npix):
		input = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
		self.wls.append(input[0])
		tmp_list.append(input[1:])
	tmp_rad_arr = np.asarray(tmp_list).reshape(self.npix, self.nalt).transpose()
	tmp_list[:] = []
	line = f.readline().strip()
	if bytes(line) == b"ERRORS":
		for _ in range(self.npix):
			input = np.fromstring(f.readline(), dtype=_float_type, sep=' ')
			tmp_list.append(input[1:])
	else:
		for _ in range(self.npix):
			tmp_list.append(np.zeros(self.nalt))
	tmp_err_arr = np.asarray(tmp_list).reshape(self.npix, self.nalt).transpose()

	# save to limb_data recarray
	rads = np.rec.fromarrays([tmp_rad_arr],
				dtype=np.dtype([("rad", 'f4', (self.npix,))]))
	errs = np.rec.fromarrays([tmp_err_arr],
				dtype=np.dtype([("err", 'f4', (self.npix,))]))
	self.limb_data = rfn.merge_arrays([self.limb_data, rads, errs],
			usemask=False, asrecarray=True, flatten=True)
	self._limb_data_dtype = self.limb_data.dtype

def write_to_textfile(self, filename):
	"""SCIAMACHY level 1c limb scan text export

	Parameters
	----------
	filename : str
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
		_print_array1(f, self.limb_data["sub_sat_lat"], 9)
		_print_array1(f, self.limb_data["sub_sat_lon"], 9)
	if self.textheader_length > 29:
		_print_indent(f, 9)
		print("%12.3f " % (self.orbit_phase,), file=f)
	_print_indent(f, 9)
	print("%8.3f %8.3f    %8.3f %8.3f  %8.3f %8.3f  "
			"%8.3f %8.3f %8.3f %8.3f" % tuple(self.cent_lat_lon), file=f)
	# print the limb data
	_print_array1(f, self.limb_data["tp_lat"], 9)
	_print_array1(f, self.limb_data["tp_lon"], 9)
	_print_array1(f, self.limb_data["tp_alt"], 9)
	_print_array1(f, self.limb_data["tp_sza"], 9)
	_print_array1(f, self.limb_data["tp_saa"], 9)
	_print_array1(f, self.limb_data["tp_los"], 9)
	_print_array1(f, self.limb_data["toa_sza"], 9)
	_print_array1(f, self.limb_data["toa_saa"], 9)
	_print_array1(f, self.limb_data["toa_los"], 9)
	_print_array1(f, self.limb_data["sat_sza"], 9)
	_print_array1(f, self.limb_data["sat_saa"], 9)
	_print_array1(f, self.limb_data["sat_los"], 9)
	_print_array1(f, self.limb_data["sat_alt"], 9)
	_print_array1(f, self.limb_data["earth_rad"], 9)

	rads = np.asarray(self.limb_data["rad"]).reshape(self.nalt, self.npix).transpose()
	errs = np.asarray(self.limb_data["err"]).reshape(self.nalt, self.npix).transpose()

	# format strings taken from scia_L1C_ascii.c
	for i in range(self.npix):
		print("%9.4f" % self.wls[i], end=" ", file=f)
		_print_array2(f, rads[i])

	print("ERRORS", file=f)

	for i in range(self.npix):
		print("%9.4f" % self.wls[i], end=" ", file=f)
		_print_array2(f, errs[i])

	f.close()
