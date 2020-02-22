# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2017 Stefan Bender
#
# This file is part of sciapy.
# sciapy is free software: you can redistribute it or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY level 1c limb spectra hdf5 interface
"""

from __future__ import absolute_import, division, print_function

import logging

import numpy as np
from astropy.time import Time

from ._types import _limb_data_dtype

logging.basicConfig(level=logging.INFO,
		format="[%(levelname)-8s] (%(asctime)s) "
		"%(filename)s:%(lineno)d %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S %z")

def _calc_angles(sza_in, los_in, saa_in, z_in, z_out, earth_radius):
	"""SCIAMACHY limb scan angle setup

	Calculates the solar zenith, solar azimuth, and line-of-sight angles
	to the tangent point (tp), to the top of the atmosphere (toa),
	and to the satellite (sat). All angles are in degrees.

	Parameters
	----------
	sza_in : ndarray
		The solar zenith angles (per tangent point).
	los_in : ndarray
		The line-of-sight zenith angles (per tangent point).
	saa_in : ndarray
		The relative solar azimuth angles (per tangent point).
	z_in : ndarray
		The input altitudes, can be tangent heights, top of
		atmosphere, or satellite altitudes.
	z_out : ndarray
		The output altitude, can be tangent heights, top of
		atmosphere, or satellite altitudes.
	earth_radius : ndarray
		The earth radii (per tangent point).

	Returns
	-------
	sza_out : ndarray
		The calculated solar zenith angles for the output altitudes.
	los_out : ndarray
		The calculated line-of-sight angles for the output altitudes.
	saa_out : ndarray
		The calculated solar azimuth angles for the output altitudes.
	"""
	def sign(x, y):
		xa = np.full_like(y, np.abs(x))
		xa[np.where(y < 0.)] *= -1
		return xa

	cos_psi_in = np.cos(np.radians(sza_in))
	mju_in = np.cos(np.radians(los_in))
	cos_phi_0 = np.cos(np.radians(saa_in))
	sin_phi_0 = np.sin(np.radians(saa_in))

	# /* start original calculation in angles.f */
	r = earth_radius + z_in
	sin_1 = np.sqrt(1.0 - mju_in**2)
	h_0 = r * (sin_1 - 1.0) + z_in

	z_out[np.where(z_out < h_0)] = h_0[np.where(z_out < h_0)]

	delta = (np.sqrt((2.0 * earth_radius + z_in + h_0) * (z_in - h_0)) -
			np.sqrt((2.0 * earth_radius + z_out + h_0) * (z_out - h_0)))
	mju_out = ((mju_in * r - delta) /
			np.sqrt((mju_in * r - delta)**2 + (r * sin_1)**2))
	sin_out = r * sin_1 / (earth_radius + z_out)
	sin_psi = np.sqrt(1.0 - cos_psi_in**2)
	zeta_0 = mju_in * cos_psi_in - sin_1 * sin_psi * cos_phi_0
	ksi_0 = mju_in * sin_psi + sin_1 * cos_psi_in * cos_phi_0

	cos_psi_out = ((cos_psi_in * r - delta * zeta_0) /
			np.sqrt((r * cos_psi_in - delta * zeta_0)**2 +
				(r * sin_psi - delta * ksi_0)**2 +
				(delta * sin_1 * sin_phi_0)**2))
	sin_psi_out = np.sqrt(1.0 - np.clip(cos_psi_out * cos_psi_out, 1.0, np.inf))
	#sin_psi_out = np.sqrt(1.0 - np.clip(cos_psi_out*cos_psi_out, -np.inf, 1.0));

	eta_0 = ((r - delta * mju_in) * sin_psi * cos_phi_0 - delta * sin_1 * cos_psi_in) / (earth_radius + z_out)

	eta_0 = eta_0 + 1.0e-39  # /*! numerical stabilization*/

	s1 = eta_0 / np.sqrt(eta_0**2 + (sin_psi * sin_phi_0)**2)
	s1 = s1 - sign(1.0e-13, s1)  # /* ! numerical stabilization*/

	sd = r * sin_psi * sin_1 * sin_phi_0 / (earth_radius + z_out) / (sin_psi_out * sin_out + 1.0e-78)

	phi_out = sign(np.arccos(s1), sd)

	sza_out = np.degrees(np.arccos(cos_psi_out))
	los_out = np.degrees(np.arccos(mju_out))
	saa_out = np.degrees(phi_out)
	# /* set back direction of line of sight
	# if (z_out > z_in)
	# *saa_out -= PI; */
	logging.debug("calculated tangent_height: %s", h_0)
	return (sza_out, los_out, saa_out)

def _middle_coord(lat1, lon1, lat2, lon2):
	sin_lat = 0.5 * (np.sin(np.radians(lat1)) + np.sin(np.radians(lat2)))
	cos_lat = 0.5 * (np.cos(np.radians(lat1)) + np.cos(np.radians(lat2)))
	sin_lon = 0.5 * (np.sin(np.radians(lon1)) + np.sin(np.radians(lon2)))
	cos_lon = 0.5 * (np.cos(np.radians(lon1)) + np.cos(np.radians(lon2)))
	return (np.degrees(np.arctan2(sin_lat, cos_lat)),
			np.degrees(np.arctan2(sin_lon, cos_lon)))

_state_txt = {
	27: "SCIAMACHY limb eclips",
	28: "SCIAMACHY limb",
	29: "SCIAMACHY limb",
	30: "SCIAMACHY limb",
	31: "SCIAMACHY limb",
	32: "SCIAMACHY limb",
	33: "SCIAMACHY limb",
	55: "SCIAMACHY limb mesosp",
}

def read_hdf5_limb_state_common_data(self, hf, lstate_id, state_in_orbit, cl_id):
	"""SCIAMACHY level 1c common data

	Parameters
	----------
	hf : opened file
		Pointer to the opened level 1c HDF5 file
	lstate_id : int
		The limb state id.
	state_in_orbit : int
		The number in this batch of states for the header.
	cl_id : int
		The spectral cluster number.

	Returns
	-------
	success : int
		0 on success,
		1 if an error occurred, for example if the measurement data
		set for the requested limb and cluster ids is empty.
	"""
	# MDS = measurement data set
	cl_mds_group_name = "/MDS/limb_{0:02d}/cluster_{1:02d}".format(lstate_id, cl_id + 1)
	cl_mds_group = hf.get(cl_mds_group_name)
	if cl_mds_group is None:
		return 1
	# Load meta data
	self.metadata["calibration"] = hf.attrs["Calibration"].decode()
	self.metadata["l1b_product"] = hf.get("/MPH")["product_name"][0].decode()
	self.metadata["orbit"] = hf.get("/MPH")["abs_orbit"][0]
	self.metadata["state_id"] = hf.get("/ADS/STATES")["state_id"][lstate_id]
	self.metadata["software_version"] = hf.get("/MPH")["software_version"][0].decode()
	self.metadata["keyfile_version"] = hf.get("/SPH")["key_data_version"][0].decode()
	self.metadata["mfactor_version"] = hf.get("/SPH")["m_factor_version"][0].decode()
	init_ver = hf.get("/SPH")["init_version"][0].decode().strip()
	self.metadata["init_version"], decont = init_ver.split()
	self.metadata["decont_flags"] = decont.lstrip("DECONT=")
	orb_phase = hf.get("/ADS/STATES")["orb_phase"][lstate_id]
	j_day_0 = 2451544.5  # 2000-01-01
	dsr_d, dsr_s, dsr_us = hf.get("/ADS/STATES")["dsr_time"][lstate_id]
	state_dt = Time(dsr_d + j_day_0 + dsr_s / 86400. + dsr_us / (86400. * 1e6),
			format="jd").datetime
	self.metadata["date"] = state_dt.strftime("%d-%b-%Y %H:%M:%S.%f")

	logging.debug("applied calibrations: %s", self.metadata["calibration"])
	logging.debug("product: %s, orbit_nr: %s, state_id: %s, orb_phase: %s",
			self.metadata["l1b_product"], self.metadata["orbit"],
			self.metadata["state_id"], orb_phase)
	logging.debug("soft_ver: %s, key_ver: %s, mf_ver: %s, init_ver: %s, "
			"decont_ver: %s", self.metadata["software_version"],
			self.metadata["keyfile_version"], self.metadata["mfactor_version"],
			self.metadata["init_version"], self.metadata["decont_flags"])

	ads_state = hf.get("/ADS/STATES")[lstate_id]
	cl_n_readouts = ads_state["clus_config"]["num_readouts"][cl_id]
	cl_intg_time = ads_state["clus_config"]["intg_time"][cl_id]
	self.metadata["nr_profile"] = 24 // (cl_intg_time * cl_n_readouts)
	self.metadata["act_profile"] = 0  # always zero for now

	# Prepare the header
	try:
		self.metadata["datatype_txt"] = _state_txt[self.metadata["state_id"]]
	except KeyError:
		logging.warn("State id %s not supported.", self.metadata["state_id"])
		return 1
	self.assemble_textheader()
	logging.debug("header:\n%s", self.textheader)

	# parse geolocation data
	gr_scia_geo = cl_mds_group.get("geoL_scia")
	tan_h = gr_scia_geo["tan_h"]
	# lat and lon are integers in degrees * 10^6
	lats_all = gr_scia_geo["tang_ground_point"]["lat"]
	lons_all = gr_scia_geo["tang_ground_point"]["lon"]
	sza_all = gr_scia_geo["sun_zen_ang"]
	saa_all = (gr_scia_geo["sun_azi_ang"] - gr_scia_geo["los_azi_ang"])
	sat_h_all = gr_scia_geo["sat_h"]
	earth_rad_all = gr_scia_geo["earth_rad"]
	# lat and lon are integers in degrees * 10^6
	subsatlat_all = gr_scia_geo["sub_sat_point"]["lat"]
	subsatlon_all = gr_scia_geo["sub_sat_point"]["lon"]
	# fix longitudes to [0°, 360°)
	lons_all[np.where(lons_all < 0)] += 360000000

	if cl_n_readouts > 2:
		tangent_heights = 0.5 * (tan_h[1::cl_n_readouts, 2] + tan_h[2::cl_n_readouts, 0])
		tp_lats = 0.5 * (lats_all[1::cl_n_readouts, 2] + lats_all[2::cl_n_readouts, 0]) * 1e-6
		tp_lons = 0.5 * (lons_all[1::cl_n_readouts, 2] + lons_all[2::cl_n_readouts, 0]) * 1e-6
		sza_toa = 0.5 * (sza_all[1::cl_n_readouts, 2] + sza_all[2::cl_n_readouts, 0])
		saa_toa = 0.5 * (saa_all[1::cl_n_readouts, 2] + saa_all[2::cl_n_readouts, 0])
		sat_hs = sat_h_all.reshape((-1, cl_n_readouts)).mean(axis=1)
		earth_rads = earth_rad_all.reshape((-1, cl_n_readouts)).mean(axis=1)
		subsatlat = subsatlat_all.reshape((-1, cl_n_readouts)).mean(axis=1) * 1e-6
		subsatlon = subsatlon_all.reshape((-1, cl_n_readouts)).mean(axis=1) * 1e-6
	else:
		tangent_heights = tan_h[::cl_n_readouts, 1]
		tp_lats = lats_all[::cl_n_readouts, 1] * 1e-6
		tp_lons = lons_all[::cl_n_readouts, 1] * 1e-6
		sza_toa = sza_all[::cl_n_readouts, 1]
		saa_toa = saa_all[::cl_n_readouts, 1]
		sat_hs = sat_h_all[::cl_n_readouts]
		earth_rads = earth_rad_all[::cl_n_readouts]
		subsatlat = subsatlat_all[::cl_n_readouts] * 1e-6
		subsatlon = subsatlon_all[::cl_n_readouts] * 1e-6

	logging.debug("tangent altitudes: %s", tangent_heights)
	nalt = len(tangent_heights)

	centre = _middle_coord(lats_all[0, 1] * 1e-6, lons_all[0, 1] * 1e-6,
			lats_all[nalt - 2, 1] * 1e-6, lons_all[nalt - 2, 1] * 1e-6)

	cent_lat_lon = (centre[0],
			# fix longitudes to [0, 360.)
			centre[1] if centre[1] >= 0. else 360. + centre[1],
			lats_all[0, 0] * 1e-6, lons_all[0, 0] * 1e-6,
			lats_all[0, 2] * 1e-6, lons_all[0, 2] * 1e-6,
			lats_all[nalt - 2, 0] * 1e-6, lons_all[nalt - 2, 0] * 1e-6,
			lats_all[nalt - 2, 2] * 1e-6, lons_all[nalt - 2, 2] * 1e-6)

	toa = 100.
	# to satellite first
	los_calc = np.degrees(np.arccos(0.0))
	sza_tp_h = sza_toa.copy()
	saa_tp_h = saa_toa.copy()
	los_tp_h = np.full_like(tangent_heights, los_calc)
	los_toa_h = np.full_like(tangent_heights, los_calc)
	sza_sat_h, los_sat_h, saa_sat_h = _calc_angles(
			sza_toa, los_calc, saa_toa,
			tangent_heights, sat_hs, earth_rads)
	# angles toa
	los_calc = np.degrees(np.arcsin((tangent_heights + earth_rads) / (toa + earth_rads)))
	# to tangent point
	los_toa_l = np.full_like(tangent_heights, los_calc)
	sza_tp_l, los_tp_l, saa_tp_l = _calc_angles(
			sza_toa, los_calc, saa_toa,
			np.full_like(tangent_heights, toa), tangent_heights,
			earth_rads)
	# to satellite
	sza_sat_l, los_sat_l, saa_sat_l = _calc_angles(
			sza_toa, los_calc, saa_toa,
			np.full_like(tangent_heights, toa), sat_hs,
			earth_rads)

	sza_sat_h[np.where(tangent_heights <= toa)] = 0.
	sza_sat_l[np.where(tangent_heights > toa)] = 0.
	saa_sat_h[np.where(tangent_heights <= toa)] = 0.
	saa_sat_l[np.where(tangent_heights > toa)] = 0.
	los_sat_h[np.where(tangent_heights <= toa)] = 0.
	los_sat_l[np.where(tangent_heights > toa)] = 0.

	sza_tp_h[np.where(tangent_heights <= toa)] = 0.
	sza_tp_l[np.where(tangent_heights > toa)] = 0.
	saa_tp_h[np.where(tangent_heights <= toa)] = 0.
	saa_tp_l[np.where(tangent_heights > toa)] = 0.
	los_tp_h[np.where(tangent_heights <= toa)] = 0.
	los_tp_l[np.where(tangent_heights > toa)] = 0.

	los_toa_h[np.where(tangent_heights <= toa)] = 0.
	los_toa_l[np.where(tangent_heights > toa)] = 0.

	sza_sat = sza_sat_h + sza_sat_l
	saa_sat = saa_sat_h + saa_sat_l
	los_sat = los_sat_h + los_sat_l

	sza_tp = sza_tp_h + sza_tp_l
	saa_tp = saa_tp_h + saa_tp_l
	los_tp = los_tp_h + los_tp_l

	los_toa = los_toa_h + los_toa_l

	logging.debug("TP sza, saa, los: %s, %s, %s", sza_tp, saa_tp, los_tp)
	logging.debug("TOA sza, saa, los: %s, %s, %s", sza_toa, saa_toa, los_toa)
	logging.debug("SAT sza, saa, los: %s, %s, %s", sza_sat, saa_sat, los_sat)

	# save the data to the limb scan class
	self.nalt = nalt
	self.orbit_state = (self.metadata["orbit"], state_in_orbit,
			self.metadata["state_id"],
			self.metadata["nr_profile"], self.metadata["act_profile"])
	self.date = (state_dt.year, state_dt.month, state_dt.day,
			state_dt.hour, state_dt.minute, state_dt.second)
	self.orbit_phase = orb_phase
	self.cent_lat_lon = cent_lat_lon
	# pre-set the limb_data
	if self._limb_data_dtype is None:
		self._limb_data_dtype = _limb_data_dtype[:]
	self.limb_data = np.zeros((self.nalt), dtype=self._limb_data_dtype)
	self.limb_data["sub_sat_lat"] = subsatlat
	self.limb_data["sub_sat_lon"] = subsatlon
	self.limb_data["tp_lat"] = tp_lats
	self.limb_data["tp_lon"] = tp_lons
	self.limb_data["tp_alt"] = tangent_heights
	self.limb_data["tp_sza"] = sza_tp
	self.limb_data["tp_saa"] = saa_tp
	self.limb_data["tp_los"] = los_tp
	self.limb_data["toa_sza"] = sza_toa
	self.limb_data["toa_saa"] = saa_toa
	self.limb_data["toa_los"] = los_toa
	self.limb_data["sat_sza"] = sza_sat
	self.limb_data["sat_saa"] = saa_sat
	self.limb_data["sat_los"] = los_sat
	self.limb_data["sat_alt"] = sat_hs
	self.limb_data["earth_rad"] = earth_rads
	return 0

def read_hdf5_limb_state_spectral_data(self, hf, lstate_id, cl_id):
	"""SCIAMACHY level 1c spectral data

	Parameters
	----------
	hf : opened file
		Pointer to the opened level 1c HDF5 file
	lstate_id : int
		The limb state id.
	cl_id : int
		The spectral cluster number.

	Returns
	-------
	success : int
		0 on success,
		1 if an error occurred, for example if the measurement data
		set for the requested limb and cluster ids is empty.
	"""
	cl_mds_group_name = "/MDS/limb_{0:02d}/cluster_{1:02d}".format(lstate_id, cl_id + 1)
	cl_mds_group = hf.get(cl_mds_group_name)
	if cl_mds_group is None:
		return 1

	ads_state = hf.get("/ADS/STATES")[lstate_id]
	cl_n_readouts = ads_state["clus_config"]["num_readouts"][cl_id]

	pwl = cl_mds_group.get("pixel_wavelength")[:]
	signal = cl_mds_group.get("pixel_signal")[:]
	sig_errs = cl_mds_group.get("pixel_signal_error")[:]
	if cl_n_readouts > 1:
		# coadd data
		signal_coadd = signal.reshape((-1, cl_n_readouts, len(pwl))).sum(axis=1)
		sig_errs = np.sqrt(((sig_errs * signal)**2)
					.reshape((-1, cl_n_readouts, len(pwl)))
					.sum(axis=1)) / np.abs(signal_coadd)
		signal = signal_coadd / cl_n_readouts
	if np.any(self.wls):
		# apparently we already have some data, so concatenate
		self.wls = np.concatenate([self.wls, pwl], axis=0)
		self._rad_arr = np.concatenate([self._rad_arr, signal], axis=1)
		self._err_arr = np.concatenate([self._err_arr, sig_errs], axis=1)
	else:
		# this seems to be the first time we fill the arrays
		self.wls = pwl
		self._rad_arr = signal
		self._err_arr = sig_errs
	return 0

def read_from_hdf5(self, hf, limb_state_id, state_in_orbit, cluster_ids):
	"""SCIAMACHY level 1c HDF main interface

	This should be the function to be used for HDF5 reading.
	It reads the common data and iterates over the spectral
	clusters to fill the class data.

	Parameters
	----------
	hf : opened file
		Pointer to the opened level 1c HDF5 file
	limb_state_id : int
		The limb state id.
	state_in_orbit : int
		The number in this batch of states for the header.
	cluster_ids : int or sequence of ints
		The spectral cluster numbers.

	Returns
	-------
	success : int
		0 on success,
		1 if an error occurred, for example if the measurement data
		set for the requested limb and cluster ids is empty.
	"""
	import numpy.lib.recfunctions as rfn
	if not hasattr(cluster_ids, '__getitem__'):
		cluster_ids = [cluster_ids]

	if self.read_hdf5_limb_state_common_data(hf, limb_state_id,
			state_in_orbit, cluster_ids[0]):
		return 1

	for cl_id in cluster_ids:
		self.read_hdf5_limb_state_spectral_data(hf, limb_state_id, cl_id)

	self.npix = len(self.wls)

	rads = np.rec.fromarrays([self._rad_arr],
				dtype=np.dtype([("rad", 'f4', (self.npix,))]))
	errs = np.rec.fromarrays([self._err_arr],
				dtype=np.dtype([("err", 'f4', (self.npix,))]))
	self.limb_data = rfn.merge_arrays([self.limb_data, rads, errs],
			usemask=False, asrecarray=True, flatten=True)
	self._limb_data_dtype = self.limb_data.dtype

	return 0
