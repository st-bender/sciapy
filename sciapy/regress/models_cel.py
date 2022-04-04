# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2017-2018 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY regression models (celerite version)

Model classes for SCIAMACHY data regression fits using the
:mod:`celerite` [#]_ modelling protocol.

.. [#] https://celerite.readthedocs.io
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.interpolate import interp1d

from celerite.modeling import Model, ModelSet, ConstantModel

__all__ = ["ConstantModel",
		"HarmonicModelCosineSine", "HarmonicModelAmpPhase",
		"ProxyModel", "TraceGasModelSet",
		"setup_proxy_model_with_bounds", "trace_gas_model"]


class HarmonicModelCosineSine(Model):
	"""Model for harmonic terms

	Models harmonic terms using a cosine and sine part.
	The total amplitude and phase can be inferred from that.

	Parameters
	----------
	freq : float
		The frequency in years^-1
	cos : float
		The amplitude of the cosine part
	sin : float
		The amplitude of the sine part
	"""
	parameter_names = ("freq", "cos", "sin")

	def get_value(self, t):
		t = np.atleast_1d(t)
		return (self.cos * np.cos(self.freq * 2 * np.pi * t) +
				self.sin * np.sin(self.freq * 2 * np.pi * t))

	def get_amplitude(self):
		return np.sqrt(self.cos**2 + self.sin**2)

	def get_phase(self):
		return np.arctan2(self.sin, self.cos)

	def compute_gradient(self, t):
		t = np.atleast_1d(t)
		dcos = np.cos(self.freq * 2 * np.pi * t)
		dsin = np.sin(self.freq * 2 * np.pi * t)
		df = 2 * np.pi * t * (self.sin * dcos - self.cos * dsin)
		return np.array([df, dcos, dsin])


class HarmonicModelAmpPhase(Model):
	"""Model for harmonic terms

	Models harmonic terms using a cosine and sine part.
	The total amplitude and phase can be inferred from that.

	Parameters
	----------
	freq : float
		The frequency in years^-1
	amp : float
		The amplitude of the harmonic term
	phase : float
		The phase of the harmonic part
	"""
	parameter_names = ("freq", "amp", "phase")

	def get_value(self, t):
		t = np.atleast_1d(t)
		return self.amp * np.cos(self.freq * 2 * np.pi * t + self.phase)

	def get_amplitude(self):
		return self.amp

	def get_phase(self):
		return self.phase

	def compute_gradient(self, t):
		t = np.atleast_1d(t)
		damp = np.cos(self.freq * 2 * np.pi * t + self.phase)
		dphi = -self.amp * np.sin(self.freq * 2 * np.pi * t + self.phase)
		df = 2 * np.pi * t * dphi
		return np.array([df, damp, dphi])


class ProxyModel(Model):
	"""Model for proxy terms

	Models proxy terms with a finite and (semi-)annually varying life time.

	Parameters
	----------
	proxy_times : (N,) array_like
		The data times of the proxy values
	proxy_vals : (N,) array_like
		The proxy values at `proxy_times`
	amp : float
		The amplitude of the proxy term
	lag : float
		The lag of the proxy value in years.
	tau0 : float
		The base life time of the proxy
	taucos1 : float
		The amplitude of the cosine part of the annual life time variation.
	tausin1 : float
		The amplitude of the sine part of the annual life time variation.
	taucos2 : float
		The amplitude of the cosine part of the semi-annual life time variation.
	tausin2 : float
		The amplitude of the sine part of the semi-annual life time variation.
	ltscan : float
		The number of days to sum the previous proxy values. If it is
		negative, the value will be set to three times the maximal lifetime.
		No lifetime adjustemets are calculated when set to zero.
	center : bool, optional
		Centers the proxy values by subtracting the overall mean. The mean is
		calculated from the whole `proxy_vals` array and is stored in the
		`mean` attribute.
		Default: False
	sza_intp : scipy.interpolate.interp1d() instance, optional
		When not `None`, cos(sza) and sin(sza) are used instead
		of the time to model the annual variation of the lifetime.
		Semi-annual variations are not used in that case.
		Default: None
	fit_phase : bool, optional
		Fit the phase shift directly instead of using sine and cosine
		terms for the (semi-)annual lifetime variations. If True, the fitted
		cosine parameter is the amplitude and the sine parameter the phase.
		Default: False (= fit sine and cosine terms)
	lifetime_prior : str, optional
		The prior probability density for each coefficient of the lifetime.
		Possible types are "flat" or `None` for a flat prior, "exp" for an
		exponential density ~ :math:`\\text{exp}(-|\\tau| / \\text{metric})`,
		and "normal" for a normal distribution
		~ :math:`\\text{exp}(-\\tau^2 / (2 * \\text{metric}^2))`.
		Default: None (= flat prior).
	lifetime_metric : float, optional
		The metric (scale) of the lifetime priors in days, see `prior`.
		Default 1.
	days_per_time_unit : float, optional
		The number of days per time unit, used to normalize the lifetime
		units. Use 365.25 if the times are in fractional years, or 1 if
		they are in days.
		Default: 365.25
	"""
	parameter_names = ("amp", "lag", "tau0",
			"taucos1", "tausin1", "taucos2", "tausin2",
			"ltscan")

	def __init__(self, proxy_times, proxy_vals,
			center=False,
			sza_intp=None, fit_phase=False,
			lifetime_prior=None, lifetime_metric=1.,
			days_per_time_unit=365.25,
			*args, **kwargs):
		self.mean = 0.
		if center:
			self.mean = np.nanmean(proxy_vals)
		self.times = proxy_times
		self.dt = 1.
		self.values = proxy_vals - self.mean
		self.sza_intp = sza_intp
		self.fit_phase = fit_phase
		self.days_per_time_unit = days_per_time_unit
		self.omega = 2 * np.pi * days_per_time_unit / 365.25
		self.lifetime_prior = lifetime_prior
		self.lifetime_metric = lifetime_metric
		# Makes "(m)jd" and "jyear" compatible for the lifetime
		# seasonal variation. The julian epoch (the default)
		# is slightly offset with respect to (modified) julian days.
		self.t_adj = 0.
		if self.days_per_time_unit == 1:
			# discriminate between julian days and modified julian days,
			# 1.8e6 is year 216 in julian days and year 6787 in
			# modified julian days. It should be pretty safe to judge on
			# that for most use cases.
			if self.times[0] > 1.8e6:
				# julian days
				self.t_adj = 13.
			else:
				# modified julian days
				self.t_adj = -44.25
		super(ProxyModel, self).__init__(*args, **kwargs)

	def _lt_corr(self, t, tau, tmax=60.):
		"""Lifetime corrected values

		Corrects for a finite lifetime by summing over the last `tmax`
		days with an exponential decay given of lifetime(s) `taus`.
		"""
		bs = np.arange(self.dt, tmax + self.dt, self.dt)
		yp = np.zeros_like(t)
		tauexp = np.exp(-self.dt / tau)
		taufac = np.ones_like(tau)
		for b in bs:
			taufac *= tauexp
			yp += np.interp(t - self.lag - b / self.days_per_time_unit,
					self.times, self.values, left=0., right=0.) * taufac
		return yp * self.dt

	def _lt_corr_grad(self, t, tau, tmax=60.):
		"""Lifetime corrected gradient

		Corrects for a finite lifetime by summing over the last `tmax`
		days with an exponential decay given of lifetime(s) `taus`.
		"""
		bs = np.arange(self.dt, tmax + self.dt, self.dt)
		ypg = np.zeros_like(t)
		tauexp = np.exp(-self.dt / tau)
		taufac = np.ones_like(tau)
		for b in bs:
			taufac *= tauexp
			ypg += np.interp(t - self.lag - b / self.days_per_time_unit,
					self.times, self.values, left=0., right=0.) * taufac * b
		return ypg * self.dt / tau**2

	def get_value(self, t):
		t = np.atleast_1d(t)
		proxy_val = np.interp(t - self.lag,
				self.times, self.values, left=0., right=0.)
		if self.ltscan == 0:
			# no lifetime, nothing else to do
			return self.amp * proxy_val
		# annual variation of the proxy lifetime
		if self.sza_intp is not None:
			# using the solar zenith angle
			tau_cs = (self.taucos1 * np.cos(np.radians(self.sza_intp(t)))
					+ self.tausin1 * np.sin(np.radians(self.sza_intp(t))))
		elif self.fit_phase:
			# using time (cos) and phase (sin)
			tau_cs = (self.taucos1 * np.cos(1 * self.omega * (t + self.t_adj) + self.tausin1)
					+ self.taucos2 * np.cos(2 * self.omega * (t + self.t_adj) + self.tausin2))
		else:
			# using time
			tau_cs = (self.taucos1 * np.cos(1 * self.omega * (t + self.t_adj))
					+ self.tausin1 * np.sin(1 * self.omega * (t + self.t_adj))
					+ self.taucos2 * np.cos(2 * self.omega * (t + self.t_adj))
					+ self.tausin2 * np.sin(2 * self.omega * (t + self.t_adj)))
		tau_cs = np.maximum(0., tau_cs)  # clip to zero
		tau = self.tau0 + tau_cs
		if self.ltscan > 0:
			_ltscn = int(np.floor(self.ltscan))
		else:
			# infer the scan time from the maximal lifetime
			_ltscn = 3 * int(np.ceil(self.tau0 +
						np.sqrt(self.taucos1**2 + self.tausin1**2)))
		if np.all(tau > 0):
			proxy_val += self._lt_corr(t, tau, tmax=_ltscn)
		return self.amp * proxy_val

	def compute_gradient(self, t):
		t = np.atleast_1d(t)
		proxy_val = np.interp(t - self.lag,
				self.times, self.values, left=0., right=0.)
		proxy_val_grad0 = proxy_val.copy()
		# annual variation of the proxy lifetime
		if self.sza_intp is not None:
			# using the solar zenith angle
			dtau_cos1 = np.cos(np.radians(self.sza_intp(t)))
			dtau_sin1 = np.sin(np.radians(self.sza_intp(t)))
			dtau_cos2 = np.zeros_like(t)
			dtau_sin2 = np.zeros_like(t)
			tau_cs = self.taucos1 * dtau_cos1 + self.tausin1 * dtau_sin1
		elif self.fit_phase:
			# using time (cos) and phase (sin)
			dtau_cos1 = np.cos(1 * self.omega * (t + self.t_adj) + self.tausin1)
			dtau_sin1 = -self.taucos1 * np.sin(1 * self.omega * t + self.tausin1)
			dtau_cos2 = np.cos(2 * self.omega * (t + self.t_adj) + self.tausin2)
			dtau_sin2 = -self.taucos2 * np.sin(2 * self.omega * t + self.tausin2)
			tau_cs = self.taucos1 * dtau_cos1 + self.taucos2 * dtau_cos2
		else:
			# using time
			dtau_cos1 = np.cos(1 * self.omega * (t + self.t_adj))
			dtau_sin1 = np.sin(1 * self.omega * (t + self.t_adj))
			dtau_cos2 = np.cos(2 * self.omega * (t + self.t_adj))
			dtau_sin2 = np.sin(2 * self.omega * (t + self.t_adj))
			tau_cs = (self.taucos1 * dtau_cos1 + self.tausin1 * dtau_sin1 +
					self.taucos2 * dtau_cos2 + self.tausin2 * dtau_sin2)
		tau_cs = np.maximum(0., tau_cs)  # clip to zero
		tau = self.tau0 + tau_cs
		if self.ltscan > 0:
			_ltscn = int(np.floor(self.ltscan))
		else:
			# infer the scan time from the maximal lifetime
			_ltscn = 3 * int(np.ceil(self.tau0 +
						np.sqrt(self.taucos1**2 + self.tausin1**2)))
		if np.all(tau > 0):
			proxy_val += self._lt_corr(t, tau, tmax=_ltscn)
			proxy_val_grad0 += self._lt_corr_grad(t, tau, tmax=_ltscn)
		return np.array([proxy_val,
				# set the gradient wrt lag to zero for now
				np.zeros_like(t),
				self.amp * proxy_val_grad0,
				self.amp * proxy_val_grad0 * dtau_cos1,
				self.amp * proxy_val_grad0 * dtau_sin1,
				self.amp * proxy_val_grad0 * dtau_cos2,
				self.amp * proxy_val_grad0 * dtau_sin2,
				# set the gradient wrt lifetime scan to zero for now
				np.zeros_like(t)])

	def _log_prior_normal(self):
		l_prior = super(ProxyModel, self).log_prior()
		if not np.isfinite(l_prior):
			return -np.inf
		for n, p in self.get_parameter_dict().items():
			if n.startswith("tau"):
				# Gaussian prior for the lifetimes
				l_prior -= 0.5 * (p / self.lifetime_metric)**2
		return l_prior

	def _log_prior_exp(self):
		l_prior = super(ProxyModel, self).log_prior()
		if not np.isfinite(l_prior):
			return -np.inf
		for n, p in self.get_parameter_dict().items():
			if n.startswith("tau"):
				# exponential prior for the lifetimes
				l_prior -= np.abs(p / self.lifetime_metric)
		return l_prior

	def log_prior(self):
		_priors = {"exp": self._log_prior_exp,
				"normal": self._log_prior_normal}
		if self.lifetime_prior is None or self.lifetime_prior == "flat":
			return super(ProxyModel, self).log_prior()
		return _priors[self.lifetime_prior]()


class TraceGasModelSet(ModelSet):
	"""Combined model class for trace gases (and probably other data)

	Inherited from :class:`celerite.ModelSet`, provides `get_value()`
	and `compute_gradient()` methods.
	"""
	def get_value(self, t):
		t = np.atleast_1d(t)
		v = np.zeros_like(t)
		for m in self.models.values():
			v += m.get_value(t)
		return v

	def compute_gradient(self, t):
		t = np.atleast_1d(t)
		grad = []
		for m in self.models.values():
			grad.extend(list(m.compute_gradient(t)))
		return np.array(grad)


def setup_proxy_model_with_bounds(times, values,
		max_amp=1e10, max_days=100,
		**kwargs):
	# extract setup from `kwargs`
	center = kwargs.get("center", False)
	fit_phase = kwargs.get("fit_phase", False)
	lag = kwargs.get("lag", 0.)
	lt_metric = kwargs.get("lifetime_metric", 1)
	lt_prior = kwargs.get("lifetime_prior", "exp")
	lt_scan = kwargs.get("lifetime_scan", 60)
	positive = kwargs.get("positive", False)
	sza_intp = kwargs.get("sza_intp", None)
	time_format = kwargs.get("time_format", "jyear")

	return ProxyModel(times, values,
			center=center,
			sza_intp=sza_intp,
			fit_phase=fit_phase,
			lifetime_prior=lt_prior,
			lifetime_metric=lt_metric,
			days_per_time_unit=1 if time_format.endswith("d") else 365.25,
			amp=0.,
			lag=lag,
			tau0=0,
			taucos1=0, tausin1=0,
			taucos2=0, tausin2=0,
			ltscan=lt_scan,
			bounds=dict([
				("amp", [0, max_amp] if positive else [-max_amp, max_amp]),
				("lag", [0, max_days]),
				("tau0", [0, max_days]),
				("taucos1", [0, max_days] if fit_phase else [-max_days, max_days]),
				("tausin1", [-np.pi, np.pi] if fit_phase else [-max_days, max_days]),
				# semi-annual cycles for the life time
				("taucos2", [0, max_days] if fit_phase else [-max_days, max_days]),
				("tausin2", [-np.pi, np.pi] if fit_phase else [-max_days, max_days]),
				("ltscan", [0, 200])])
			)


def _default_proxy_config(tfmt="jyear"):
	from .load_data import load_dailymeanLya, load_dailymeanAE
	proxy_config = {}
	# Lyman-alpha
	plyat, plyadf = load_dailymeanLya(tfmt=tfmt)
	proxy_config.update({"Lya": {
		"times": plyat,
		"values": plyadf["Lya"],
		"center": False,
		"positive": False,
		"lifetime_scan": 0,
		}}
	)
	# AE index
	paet, paedf = load_dailymeanAE(name="GM", tfmt=tfmt)
	proxy_config.update({"GM": {
		"times": paet,
		"values": paedf["GM"],
		"center": False,
		"positive": True,
		"lifetime_scan": 60,
		}}
	)
	return proxy_config


def trace_gas_model(constant=True, freqs=None, proxy_config=None, **kwargs):
	"""Trace gas model setup

	Sets up the trace gas model for easy access. All parameters are optional,
	defaults to an offset, no harmonics, proxies are uncentered and unscaled
	Lyman-alpha and AE. AE with only positive amplitude and a seasonally
	varying lifetime.

	Parameters
	----------
	constant : bool, optional
		Whether or not to include a constant (offset) term, default is True.
	freqs : list, optional
		Frequencies of the harmonic terms in 1 / a^-1 (inverse years).
	proxy_config : dict, optional
		Proxy configuration if different from the standard setup.
	**kwargs : optional
		Additional keyword arguments, all of them are also passed on to
		the proxy setup. For now, supported are the following which are
		also passed along to the proxy setup with
		`setup_proxy_model_with_bounds()`:

		* fit_phase : bool
			fit amplitude and phase instead of sine and cosine
		* scale : float
			the factor by which the data is scaled, used to constrain
			the maximum and minimum amplitudes to be fitted.
		* time_format : string
			The `astropy.time.Time` format string to setup the time axis.
		* days_per_time_unit : float
			The number of days per time unit, used to normalize the frequencies
			for the harmonic terms. Use 365.25 if the times are in fractional years,
			1 if they are in days. Default: 365.25
		* max_amp : float
			Maximum magnitude of the coefficients, used to constrain the
			parameter search.
		* max_days : float
			Maximum magnitude of the lifetimes, used to constrain the
			parameter search.

	Returns
	-------
	model : :class:`TraceGasModelSet` (extends :class:`celerite.ModelSet`)
	"""
	fit_phase = kwargs.get("fit_phase", False)
	scale = kwargs.get("scale", 1e-6)
	tfmt = kwargs.get("time_format", "jyear")
	delta_t = kwargs.get("days_per_time_unit", 365.25)

	max_amp = kwargs.pop("max_amp", 1e10 * scale)
	max_days = kwargs.pop("max_days", 100)

	offset_model = []
	if constant:
		offset_model = [("offset",
				ConstantModel(value=0.,
						bounds={"value": [-max_amp, max_amp]}))]

	freqs = freqs or []
	harmonic_models = []
	for freq in freqs:
		if not fit_phase:
			harm = HarmonicModelCosineSine(freq=freq * delta_t / 365.25,
					cos=0, sin=0,
					bounds=dict([
						("cos", [-max_amp, max_amp]),
						("sin", [-max_amp, max_amp])])
			)
		else:
			harm = HarmonicModelAmpPhase(freq=freq * delta_t / 365.25,
					amp=0, phase=0,
					bounds=dict([
						("amp", [0, max_amp]),
						("phase", [-np.pi, np.pi])])
			)
		harm.freeze_parameter("freq")
		harmonic_models.append(("f{0:.0f}".format(freq), harm))

	proxy_config = proxy_config or _default_proxy_config(tfmt=tfmt)
	proxy_models = []
	for pn, conf in proxy_config.items():
		if "max_amp" not in conf:
			conf.update(dict(max_amp=max_amp))
		if "max_days" not in conf:
			conf.update(dict(max_days=max_days))
		kw = kwargs.copy()  # don't mess with the passed arguments
		kw.update(conf)
		proxy_models.append(
			(pn, setup_proxy_model_with_bounds(**kw))
		)

	return TraceGasModelSet(offset_model + harmonic_models + proxy_models)
