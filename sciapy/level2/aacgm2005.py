# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
"""AACGM 2005 geomagnetic model at 80 km

Copyright (c) 2018 Stefan Bender

This file is part of sciapy.
sciapy is free software: you can redistribute it or modify it
under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2.
See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""
from __future__ import absolute_import, division, print_function

import logging
import os

import numpy as np
from scipy.interpolate import RectBivariateSpline
import xarray as xr

__all__ = ['gmag_aacgm2005']


def gmag_aacgm2005(lat, lon, aacgm_name="AACGM2005_80km_grid.nc"):
	"""Fixed 2005 AACGM geomagnetic coordinates at 80 km

	Geomagnetic coordinates according to the AACGM model but
	with fixed parameters for the 2005 epoch.

	Parameters
	----------
	lat: array_like
		Geographic latitude(s) in degrees north
	lon: array_like
		Geographic longitude(s) in degrees east
	aacgm_name: str, optional
		Filename of the AACGM grid, relating geographic latitude
		and longitude to AACGM geomagnetic latitude and longitude.
		The default is the prepared grid file for 2005 and at 80 km.

	Returns
	-------
	aacgmlat: numpy.ndarray or float
		The AACGM 2005 geomagnetic latitude(s)
	aacgmlon: numpy.ndarray or float
		The AACGM 2005 geomagnetic longitude(s)
	"""
	aacgm_file = os.path.join(
			os.path.realpath(os.path.dirname(__file__)),
			aacgm_name)
	logging.debug("aacgm_file: %s", aacgm_file)
	# Fix longitudes to +- 180
	lon = np.asarray(lon) % -180.
	aacgm_ds = xr.open_dataset(aacgm_file)
	lats = aacgm_ds["Latitude"]
	lons = aacgm_ds["Longitude"]
	glats = aacgm_ds["Geomagnetic_latitude"]
	glons = aacgm_ds["Geomagnetic_longitude"]
	splglat = RectBivariateSpline(lats, lons, glats)
	splglon = RectBivariateSpline(lats, lons, glons)
	return (splglat.ev(lat, lon), splglon.ev(lat, lon))
