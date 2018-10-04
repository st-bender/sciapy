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
"""NRLMSISE-00 python wrapper for the C version [#]_

.. [#] https://www.brodo.de/space/nrlmsise
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import datetime as dt

from sciapy.level2.nrlmsise00 import gtd7

def msise_gtd7(date, alt, lat, lon, lst, f107a, f107, ap):
	"""Direct interface to `gtd7()` [#]_

	Uses only the standard flags and uses only the daily ap index.

	.. [#] https://git.linta.de/?p=~brodo/nrlmsise-00.git;a=blob;f=nrlmsise-00.c#l916

	Parameters
	----------
	date: str
		Date in the format "%Y-%m-%d".
	alt: float
		Altitude in km.
	lat: float
		Latitude in degrees north.
	lon: float
		Longitude in degrees east.
	lst: float
		Local solar time in hours.
	f107: float
		The observed f107 value at date.
	f107a: float
		The observed f107a (81-day running average of f107) value at date.
	ap: float
		The ap value at date.

	Returns
	-------
	nr_densities: list of floats
		0. He number density [cm^-3]
		1. O number density [cm^-3]
		2. N2 number density [cm^-3]
		3. O2 number density [cm^-3]
		4. AR number density [cm^-3]
		5. total mass density [g cm^-3] (includes d[8] in td7d)
		6. H number density [cm^-3]
		7. N number density [cm^-3]
		8. Anomalous oxygen number density [cm^-3]
	temperatures: list of floats
		0. exospheric temperature [K]
		1. temperature at alt [K]
	"""
	dtdate = dt.datetime.strptime(date, "%Y-%m-%d")
	year = int(dtdate.strftime("%Y"))
	doy = int(dtdate.strftime("%j"))
	hour = lst - lon / 15.0

	return gtd7(input=dict(year=year, doy=doy, sec=hour * 3600., alt=alt,
				g_lat=lat, g_long=lon, lst=lst, f107A=f107a, f107=f107, ap=ap))

def msise(date, alt, lat, lon, lst, f107a, f107, ap):
	"""Temperature and total air number and mass densities

	Shortcut to the temperature at altitude and the total densities,
	number and mass densities.

	Parameters
	----------
	date: str
		Date in the format "%Y-%m-%d".
	alt: float
		Altitude in km.
	lat: float
		Latitude in degrees north.
	lon: float
		Longitude in degrees east.
	lst: float
		Local solar time in hours.
	f107: float
		The observed f107 value at date.
	f107a: float
		The observed f107a (81-day running average of f107) value at date.
	ap: float
		The ap value at date.

	Returns
	-------
	temperature: float
		Temperature at altitude `alt` [K].
	density_air: float
		Total number density of air at the location [cm^-3].
	mass_density: float
		Total mass density of air at the location [g cm^-3].
	"""
	nr_densities, temperatures = msise_gtd7(
			date, alt, lat, lon, lst, f107a, f107, ap)
	nr_dens_total = np.sum(nr_densities[:5]) + np.sum(nr_densities[6:])
	return temperatures[1], nr_dens_total, nr_densities[5]
