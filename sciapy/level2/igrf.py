# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2017-2018 Stefan Bender
#
# This file is part of sciapy.
# sciapy is free software: you can redistribute it or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""IGRF geomagnetic coordinates

This is a python (mix) version of GMPOLE and GMCOORD from
http://www.ngdc.noaa.gov/geomag/geom_util/utilities_home.shtml
to transform geodetic to geomagnetic coordinates.
It uses the IGRF 2012 model and coefficients [#]_.

.. [#] Thébault et al. 2015,
	International Geomagnetic Reference Field: the 12th generation.
	Earth, Planets and Space, 67 (79)
	http://nora.nerc.ac.uk/id/eprint/511258
	https://doi.org/10.1186/s40623-015-0228-9
"""
from __future__ import absolute_import, division, print_function

import logging
from collections import namedtuple
from pkg_resources import resource_filename

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import lpmn

__all__ = ['gmpole', 'gmag_igrf']

# The WGS84 reference ellipsoid
Earth_ellipsoid = {
	"a": 6378.137,  # semi-major axis of the ellipsoid in km
	"b": 6356.7523142,  # semi-minor axis of the ellipsoid in km
	"fla": 1. / 298.257223563,  # flattening
	"re": 6371.2  # Earth's radius in km
}

def _ellipsoid(ellipsoid_data=Earth_ellipsoid):
	# extends the dictionary with the eccentricities
	ell = namedtuple('ellip', ["a", "b", "fla", "eps", "epssq", "re"])
	ell.a = ellipsoid_data["a"]
	ell.b = ellipsoid_data["b"]
	ell.fla = ellipsoid_data["fla"]
	ell.re = ellipsoid_data["re"]
	# first eccentricity squared
	ell.epssq = 1. - ell.b**2 / ell.a**2
	# first eccentricity
	ell.eps = np.sqrt(ell.epssq)
	return ell

def _date_to_frac_year(year, month, day):
	# fractional year by dividing the day of year by the overall
	# number of days in that year
	extraday = 0
	if ((year % 4 == 0) and (year % 100 != 0)) or (year % 400 == 0):
		extraday = 1
	month_days = [0, 31, 28 + extraday, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
	doy = np.sum(month_days[:month]) + day
	return year + (doy - 1) / (365.0 + extraday)

def _load_igrf_file(filename="IGRF.tab"):
	"""Load IGRF coefficients

	Parameters
	----------
	filename: str, optional
		The file with the IGRF coefficients.

	Returns
	-------
	interpol: `scipy.interpolate.interp1d` instance
		Interpolator instance, called with the fractional
		year to obtain the IGRF coefficients for the epoch.
	"""
	igrf_tab = np.genfromtxt(filename, skip_header=3, dtype=None)

	sv = igrf_tab[igrf_tab.dtype.names[-1]][1:].astype(np.float)

	years = np.asarray(igrf_tab[0].tolist()[3:-1])
	years = np.append(years, [years[-1] + 5])

	coeffs = []
	for i in range(1, len(igrf_tab)):
		coeff = np.asarray(igrf_tab[i].tolist()[3:-1])
		coeff = np.append(coeff, np.array([5]) * sv[0] + coeff[-1])
		coeffs.append(coeff)

	return interp1d(years, coeffs)

def _geod_to_spher(phi, lon, Ellip, HeightAboveEllipsoid=0.):
	"""Convert geodetic to spherical coordinates

	Converts geodetic coordinates on the WGS-84 reference ellipsoid
	to Earth-centered Earth-fixed Cartesian coordinates,
	and then to spherical coordinates.
	"""
	CosLat = np.cos(np.radians(phi))
	SinLat = np.sin(np.radians(phi))

	# compute the local radius of curvature on the WGS-84 reference ellipsoid
	rc = Ellip.a / np.sqrt(1.0 - Ellip.epssq * SinLat**2)

	# compute ECEF Cartesian coordinates of specified point (for longitude=0)
	xp = (rc + HeightAboveEllipsoid) * CosLat
	zp = (rc * (1.0 - Ellip.epssq) + HeightAboveEllipsoid) * SinLat

	# compute spherical radius and angle phi of specified point
	rg = np.sqrt(xp**2 + zp**2)
	# geocentric latitude
	phig = np.degrees(np.arcsin(zp / rg))
	return phig, lon, rg

def _igrf_model(coeffs, Lmax, r, theta, phi, R_E=Earth_ellipsoid["re"]):
	"""Evaluates the IGRF model function at the given location
	"""
	rho = R_E / r
	sin_theta = np.sin(theta)
	cos_theta = np.cos(theta)
	Plm, dPlm = lpmn(Lmax, Lmax, cos_theta)
	logging.debug("R_E: %s, r: %s, rho: %s", R_E, r, rho)
	logging.debug("rho: %s, theta: %s, sin_theta: %s, cos_theta: %s",
			rho, theta, sin_theta, cos_theta)
	Bx, By, Bz = 0., 0., 0.  # Btheta, Bphi, Br
	idx = 0
	rho_l = rho
	K_l1 = 1.
	for l in range(1, Lmax + 1):
		rho_l *= rho  # rho^(l+1)
		# m = 0 values, h_l^m = 0
		Bxl = rho_l * coeffs[idx] * dPlm[0, l] * (-sin_theta)
		Byl = 0.
		Bzl = -(l + 1) * rho_l * coeffs[idx] * Plm[0, l]
		if l > 1:
			K_l1 *= np.sqrt((l - 1) / (l + 1))
		idx += 1
		K_lm = -K_l1
		for m in range(1, l + 1):
			cfi = K_lm * rho_l * (coeffs[idx] * np.cos(m * phi)
						+ coeffs[idx + 1] * np.sin(m * phi))
			Bxl += cfi * dPlm[m, l] * (-sin_theta)
			Bzl += -(l + 1) * cfi * Plm[m, l]
			if sin_theta != 0:
				Byl += K_lm * rho_l * m * Plm[m, l] * (
						- coeffs[idx] * np.sin(m * phi) +
						coeffs[idx + 1] * np.cos(m * phi))
			else:
				Byl += 0.
			if m < l:
				# K_lm for the next m
				K_lm /= -np.sqrt((l + m + 1) * (l - m))
			idx += 2
		Bx += Bxl
		By += Byl
		Bz += Bzl
	Bx = rho * Bx
	By = -rho * By / sin_theta
	Bz = rho * Bz
	return Bx, By, Bz

def igrf_mag(date, lat, lon, alt, filename="IGRF.tab"):
	"""Evaluate the local magnetic field using the IGRF model

	Evaluates the IGRF coefficients to calculate the Earth's
	magnetic field at the given location.
	The results agree to within a few decimal places with
	https://ngdc.noaa.gov/geomag-web/

	Parameters
	----------
	date: `datetime.date` or `datetime.datetime` instance
		The date for the evaluation.
	lat: float
		Geographic latitude in degrees north
	lon: float
		Geographic longitude in degrees east
	alt: float
		Altitude above ground in km.
	filename: str, optional
		File containing the IGRF coefficients.

	Returns
	-------
	Bx: float
		Northward component of the magnetic field, B_N.
	By: float
		Eastward component of the magnetic field, B_E.
	Bz: float
		Downward component of the magnetic field, B_D.
	"""
	ellip = _ellipsoid()
	# date should be datetime.datetime or datetime.date instance,
	# or something else that provides .year, .month, and .day attributes
	frac_year = _date_to_frac_year(date.year, date.month, date.day)
	glat, glon, grad = _geod_to_spher(lat, lon, ellip, alt)
	sin_theta = np.sin(np.radians(90. - glat))
	cos_theta = np.cos(np.radians(90. - glat))

	rho = np.sqrt((ellip.a * sin_theta)**2 + (ellip.b * cos_theta)**2)
	r = np.sqrt(alt**2 + 2 * alt * rho +
			(ellip.a**4 * sin_theta**2 + ellip.b**4 * cos_theta**2) / rho**2)
	cd = (alt + rho) / r
	sd = (ellip.a**2 - ellip.b**2) / rho * cos_theta * sin_theta / r
	logging.debug("rho: %s, r: %s, (alt + rho) / r: %s, R_E / (R_E + h): %s, R_E / r: %s",
			rho, r, cd, ellip.re / (ellip.re + alt), ellip.re / r)

	cos_theta, sin_theta = cos_theta*cd - sin_theta*sd, sin_theta*cd + cos_theta*sd
	logging.debug("r: %s, spherical coordinates (radius, rho, theta, lat): %s, %s, %s, %s",
			r, grad, rho, np.degrees(np.arccos(cos_theta)), 90. - glat)

	# evaluate the IGRF model in spherical coordinates
	igrf_file = resource_filename(__name__, filename)
	igrf_coeffs = _load_igrf_file(igrf_file)(frac_year)
	Bx, By, Bz = _igrf_model(igrf_coeffs, 13, r, np.radians(90. - glat), np.radians(glon))
	logging.debug("spherical geomagnetic field (Bx, By, Bz): %s, %s, %s", Bx, By, Bz)
	logging.debug("spherical dip coordinates: lat %s, lon %s",
			np.degrees(np.arctan2(0.5 * Bz,  np.sqrt(Bx**2 + By**2))),
			np.degrees(np.arctan2(-By, Bz)))
	# back to geodetic coordinates
	Bx, Bz = cd * Bx + sd * Bz, cd * Bz - sd * Bx
	logging.debug("geodetic geomagnetic field (Bx, By, Bz): %s, %s, %s", Bx, By, Bz)
	logging.debug("geodetic dip coordinates: lat %s, lon %s",
			np.degrees(np.arctan2(0.5 * Bz, np.sqrt(Bx**2 + By**2))),
			np.degrees(np.arctan2(-By, Bz)))
	return Bx, By, Bz

def gmpole(date, r_e=Earth_ellipsoid["re"], filename="IGRF.tab"):
	"""Centered dipole geomagnetic pole coordinates

	Parameters
	----------
	date: datetime.datetime or datetime.date
	r_e: float, optional
		Earth radius to evaluate the dipole's off-centre shift.
	filename: str, optional
		File containing the IGRF parameters.

	Returns
	-------
	(lat_n, phi_n): tuple of floats
		Geographic latitude and longitude of the centered dipole
		magnetic north pole.
	(lat_s, phi_s): tuple of floats
		Geographic latitude and longitude of the centered dipole
		magnetic south pole.
	(dx, dy, dz): tuple of floats
		Magnetic variations in Earth-centered Cartesian coordinates
		for shifting the dipole off-center.
	(dX, dY, dZ): tuple of floats
		Magnetic variations in centered-dipole Cartesian coordinates
		for shifting the dipole off-center.
	B_0: float
		The magnitude of the magnetic field.
	"""
	igrf_file = resource_filename(__name__, filename)
	gh_func = _load_igrf_file(igrf_file)

	frac_year = _date_to_frac_year(date.year, date.month, date.day)
	logging.debug("fractional year: %s", frac_year)
	g10, g11, h11, g20, g21, h21, g22, h22 = gh_func(frac_year)[:8]

	# This function finds the location of the north magnetic pole in spherical coordinates.
	# The equations are from Wallace H. Campbell's "Introduction to Geomagnetic Fields"
	# and Fraser-Smith 1987, Eq. (5).
	# For the minus signs see also
	# Laundal and Richmond, Space Sci. Rev. (2017) 206:27--59,
	# doi:10.1007/s11214-016-0275-y: (p. 31)
	# "The Earth’s field is such that the dipole axis points roughly southward,
	# so that the dipole North Pole is really in the Southern Hemisphere (SH).
	# However convention dictates that the axis of the geomagnetic dipole is
	# positive northward, hence the negative sign in the definition of mˆ."
	# Note that hence Phi_N in their Eq. (14) is actually Phi_S.
	B_0_sq = g10**2 + g11**2 + h11**2
	theta_n = np.arccos(-g10 / np.sqrt(B_0_sq))
	phi_n = np.arctan2(-h11, -g11)
	lat_n = 0.5 * np.pi - theta_n
	logging.debug("centered dipole north pole coordinates "
			"(lat, theta, phi): %s, %s, %s",
			np.degrees(lat_n), np.degrees(theta_n), np.degrees(phi_n))

	# dipole offset according to Fraser-Smith, 1987
	L_0 = 2 * g10 * g20 + np.sqrt(3) * (g11 * g21 + h11 * h21)
	L_1 = -g11 * g20 + np.sqrt(3) * (g10 * g21 + g11 * g22 + h11 * h22)
	L_2 = -h11 * g20 + np.sqrt(3) * (g10 * h21 - h11 * g22 + g11 * h22)
	E = (L_0 * g10 + L_1 * g11 + L_2 * h11) / (4 * B_0_sq)

	# dipole offset in geodetic Cartesian coordinates
	xi = (L_0 - g10 * E) / (3 * B_0_sq)
	eta = (L_1 - g11 * E) / (3 * B_0_sq)
	zeta = (L_2 - h11 * E) / (3 * B_0_sq)
	dx = eta * r_e
	dy = zeta * r_e
	dz = xi * r_e
	logging.debug("dx, dy, dz: %s, %s, %s", dx, dy, dz)

	# dipole offset in geodetic spherical coordinates
	# Fraser-Smith 1987, Eq. (24)
	delta = np.sqrt(dx**2 + dy**2 + dz**2)
	theta_d = np.arccos(dz / delta)
	lambda_d = 0.5 * np.pi - theta_d
	phi_d = np.arctan2(dy, dx)
	logging.debug(
		"delta: %s, theta_d: %s, phi_d: %s",
		delta, np.degrees(theta_d), np.degrees(phi_d),
	)

	# dipole offset in centred-dipole spherical coordindates
	sin_lat_ed = (np.sin(lambda_d) * np.sin(lat_n)
				+ np.cos(lambda_d) * np.cos(lat_n) * np.cos(phi_d - phi_n))
	lat_ed = np.arcsin(sin_lat_ed)
	theta_ed = 0.5 * np.pi - lat_ed
	sin_lon_ed = np.sin(theta_d) * np.sin(phi_d - phi_n) / np.sin(theta_ed)
	lon_ed = np.pi - np.arcsin(sin_lon_ed)
	logging.debug("eccentric dipole offset in CD coordinates "
			"(lat, theta, lon): %s, %s, %s",
			np.degrees(theta_ed), np.degrees(lat_ed), np.degrees(lon_ed))

	# dipole offset in centred-dipole Cartesian coordindates
	dX = delta * np.sin(theta_ed) * np.cos(lon_ed)
	dY = delta * np.sin(theta_ed) * np.sin(lon_ed)
	dZ = delta * np.cos(theta_ed)
	logging.debug("magnetic variations (dX, dY, dZ): %s, %s, %s", dX, dY, dZ)

	# North pole, south pole coordinates
	return ((np.degrees(lat_n), np.degrees(phi_n)),
			(-np.degrees(lat_n), np.degrees(phi_n + np.pi)),
			(dx, dy, dz),
			(dX, dY, dZ),
			np.sqrt(B_0_sq))

def gmag_igrf(date, lat, lon, alt=0.,
		centered_dipole=False,
		igrf_name="IGRF.tab"):
	"""Centered or eccentric dipole geomagnetic coordinates

	Parameters
	----------
	date: datetime.datetime
	lat: float
		Geographic latitude in degrees north
	lon: float
		Geographic longitude in degrees east
	alt: float, optional
		Altitude in km. Default: 0.
	centered_dipole: bool, optional
		Returns the centered dipole geomagnetic coordinates
		if set to True, returns the eccentric dipole
		geomagnetic coordinates if set to False.
		Default: False
	igrf_name: str, optional
		Default: "IGRF.tab"

	Returns
	-------
	geomag_latitude: numpy.ndarray or float
		Geomagnetic latitude in eccentric dipole coordinates,
		centered dipole coordinates if `centered_dipole` is True.
	geomag_longitude: numpy.ndarray or float
		Geomagnetic longitude in eccentric dipole coordinates,
		centered dipole coordinates if `centered_dipole` is True.
	"""
	ellip = _ellipsoid()
	glat, glon, grad = _geod_to_spher(lat, lon, ellip, alt)
	(lat_GMP, lon_GMP), _, _, (dX, dY, dZ), B_0 = gmpole(date, ellip.re, igrf_name)
	latr, lonr = np.radians(glat), np.radians(glon)
	lat_GMPr, lon_GMPr = np.radians(lat_GMP), np.radians(lon_GMP)
	sin_lat_gmag = (np.sin(latr) * np.sin(lat_GMPr)
				+ np.cos(latr) * np.cos(lat_GMPr) * np.cos(lonr - lon_GMPr))
	lon_gmag_y = np.cos(latr) * np.sin(lonr - lon_GMPr)
	lon_gmag_x = (np.cos(latr) * np.sin(lat_GMPr) * np.cos(lonr - lon_GMPr)
				- np.sin(latr) * np.cos(lat_GMPr))
	lat_gmag = np.arcsin(sin_lat_gmag)
	lat_gmag_geod = np.arctan2(np.tan(lat_gmag), (1. - ellip.epssq))

	B_r = -2. * B_0 * sin_lat_gmag / (grad / ellip.re)**3
	B_th = -B_0 * np.cos(lat_gmag) / (grad / ellip.re)**3
	logging.debug("B_r: %s, B_th: %s, dip lat: %s",
			-B_r, B_th, np.degrees(np.arctan2(0.5 * B_r, B_th)))

	lon_gmag = np.arctan2(lon_gmag_y, lon_gmag_x)
	logging.debug("centered dipole coordinates: "
			"lat_gmag: %s, lon_gmag: %s",
			np.degrees(lat_gmag), np.degrees(lon_gmag))
	logging.debug("lat_gmag_geod: %s, lat_GMPr: %s",
			np.degrees(lat_gmag_geod), np.degrees(lat_GMPr))
	if centered_dipole:
		return (np.degrees(lat_gmag), np.degrees(lon_gmag))

	# eccentric dipole coordinates (shifted dipole)
	phi_ed = np.arctan2(
		(grad * np.cos(lat_gmag) * np.sin(lon_gmag) - dY),
		(grad * np.cos(lat_gmag) * np.cos(lon_gmag) - dX)
	)
	theta_ed = np.arctan2(
		(grad * np.cos(lat_gmag) * np.cos(lon_gmag) - dX),
		(grad * np.sin(lat_gmag) - dZ) * np.cos(phi_ed)
	)
	lat_ed = 0.5 * np.pi - theta_ed
	lat_ed_geod = np.arctan2(np.tan(lat_ed), (1. - ellip.epssq))
	logging.debug("lats ed: %s", np.degrees(lat_ed))
	logging.debug("lats ed geod: %s", np.degrees(lat_ed_geod))
	logging.debug("phis ed: %s", np.degrees(phi_ed))
	return (np.degrees(lat_ed_geod), np.degrees(phi_ed))
