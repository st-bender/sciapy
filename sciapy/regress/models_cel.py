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
`celerite`[1] modeling protocol.

[1](https://celerite.readthedocs.io)
"""
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.interpolate import interp1d

from celerite.modeling import Model, ModelSet, ConstantModel

__all__ = ["ConstantModel",
		"HarmonicModelCosineSine", "HarmonicModelAmpPhase",
		"ProxyModel", "CeleriteModelSet"]

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
		return (self.cos * np.cos(self.freq * 2 * np.pi * t) +
				self.sin * np.sin(self.freq * 2 * np.pi * t))

	def get_amplitude(self):
		return np.sqrt(np.cos**2 + np.sin**2)

	def get_phase(self):
		return np.arctan2(np.sin, np.cos)

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
		return self.amp * np.cos(self.freq * 2 * np.pi * t + self.phase)

	def get_amplitude(self):
		return self.amp

	def get_phase(self):
		return self.phase

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
		The number of days to sum the previous proxy values. If it is zero or
		negative, the value will be set to three times the maximal lifetime.
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
		exponential density ~ exp(-|tau| / metric), and "normal" for a normal
		distribution ~ exp(-tau^2 / (2 * metric^2)). The distributions are
		normalized according to the parameter bounds.
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
		self.intp = interp1d(proxy_times, proxy_vals - self.mean,
				fill_value="extrapolate")
		self.sza_intp = sza_intp
		self.fit_phase = fit_phase
		self.days_per_time_unit = days_per_time_unit
		self.omega = 2 * np.pi * days_per_time_unit / 365.25
		self.lifetime_prior = lifetime_prior
		self.lifetime_metric = lifetime_metric
		super(ProxyModel, self).__init__(*args, **kwargs)

	def get_value(self, t):
		proxy_val = self.intp(t - self.lag)
		# annual variation of the proxy lifetime
		if self.sza_intp is not None:
			# using the solar zenith angle
			tau_cs = (self.taucos1 * np.cos(np.radians(self.sza_intp(t)))
					+ self.tausin1 * np.sin(np.radians(self.sza_intp(t))))
		elif self.fit_phase:
			# using time (cos) and phase (sin)
			tau_cs = (self.taucos1 * np.cos(1 * self.omega * t + self.tausin1)
					+ self.taucos2 * np.cos(2 * self.omega * t + self.tausin2))
		else:
			# using time
			tau_cs = (self.taucos1 * np.cos(1 * self.omega * t)
					+ self.tausin1 * np.sin(1 * self.omega * t)
					+ self.taucos2 * np.cos(2 * self.omega * t)
					+ self.tausin2 * np.sin(2 * self.omega * t))
		tau_cs[tau_cs < 0] = 0.  # clip to zero
		tau = self.tau0 + tau_cs
		if self.ltscan > 0:
			_ltscn = int(np.floor(self.ltscan))
		else:
			# infer the scan time from the maximal lifetime
			_ltscn = 3 * int(np.ceil(self.tau0 +
						np.sqrt(self.taucos1**2 + self.tausin1**2)))
		if np.all(tau > 0):
			tauexp = np.exp(-1. / tau)
			taufac = 1.
			for b in range(1, _ltscn + 1):
				taufac *= tauexp
				proxy_val += self.intp(t - self.lag - b / self.days_per_time_unit) * taufac
		return self.amp * proxy_val

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


class CeleriteModelSet(ModelSet):

	def __init__(self, times, data, errs,
			models=None,
			*args, **kwargs):
		self.t, self.f, self.fe = times, data, errs
		super(CeleriteModelSet, self).__init__(
				models, *args, **kwargs)

	def get_value(self, t):
		v = np.zeros_like(t)
		for m in self.models.values():
			v += m.get_value(t)
		return v
