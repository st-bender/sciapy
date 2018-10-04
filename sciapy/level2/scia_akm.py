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
"""SCIAMACHY level 2 averaging kernel interface
"""
from __future__ import print_function

import numpy as np
import xarray as xr


def read_akm(filename, nalt, nlat):
	"""Read SCIAMACHY level 2 averaging kernels into numpy array

	Supports plain ascii (text) tables using :func:`numpy.genfromtxt`
	and netcdf files using :mod:`xarray`.

	Parameters
	----------
	filename: str
		Filename of the averaging kernel elements
	nalt: int
		Number of altitude bins of the retrieval
	nlat: int
		Number of latitude bins of the retrieval

	Returns
	-------
	akm: numpy.ndarray of shape (nalt, nlat, nalt, nlat)
		The averaging kernel matrix elements.
	"""
	try:
		akm = np.genfromtxt(filename)
	except UnicodeDecodeError:
		# most probably a netcdf file
		akm = xr.open_dataarray(filename).data
	return akm.reshape(nalt, nlat, nalt, nlat)
