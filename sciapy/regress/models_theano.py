# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2022 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY regression models (theano/pymc3 version)

Model classes for SCIAMACHY data regression fits using
:mod:`theano` for :mod:`pymc3`.

This interface is still experimental, it is available
when installing ``sciapy``'s ``pymc3`` or ``theano`` extras:

.. code-block:: bash

	pip install "sciapy[pymc3]"

The classes can be imported as usual, e.g. via:

.. code-block:: python

	from sciapy.regress.models_theano import ProxyModel

"""
from __future__ import absolute_import, division, print_function
from warnings import warn

import numpy as np

try:
	import aesara_theano_fallback.tensor as tt
except ImportError as err:
	raise ImportError(
		"The `aesara_theano_fallback` package is required for the `theano` model interface."
	).with_traceback(err.__traceback__)
try:
	import pymc3 as pm
except ImportError as err:
	raise ImportError(
		"The `pymc3` package is required for the `theano` model interface."
	).with_traceback(err.__traceback__)

__all__ = [
	"HarmonicModelCosineSine", "HarmonicModelAmpPhase",
	"LifetimeModel",
	"ProxyModel",
	"ModelSet",
	"setup_proxy_model_theano",
	"trace_gas_modelset",
]


class HarmonicModelCosineSine:
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
	def __init__(self, freq, cos, sin):
		self.omega = tt.as_tensor_variable(2 * np.pi * freq).astype("float64")
		self.cos = tt.as_tensor_variable(cos).astype("float64")
		self.sin = tt.as_tensor_variable(sin).astype("float64")

	def get_value(self, t):
		t = tt.as_tensor_variable(t).astype("float64")
		return (
			self.cos * tt.cos(self.omega * t)
			+ self.sin * tt.sin(self.omega * t)
		)

	def get_amplitude(self):
		return tt.sqrt(self.cos**2 + self.sin**2)

	def get_phase(self):
		return tt.arctan2(self.cos, self.sin)

	def compute_gradient(self, t):
		t = tt.as_tensor_variable(t).astype("float64")
		dcos = tt.cos(self.omega * t)
		dsin = tt.sin(self.omega * t)
		df = 2 * np.pi * t * (self.sin * dcos - self.cos * dsin)
		return (df, dcos, dsin)


class HarmonicModelAmpPhase:
	"""Model for harmonic terms

	Models harmonic terms using amplitude and phase of a sine.

	Parameters
	----------
	freq : float
		The frequency in years^-1
	amp : float
		The amplitude of the harmonic term
	phase : float
		The phase of the harmonic part
	"""
	def __init__(self, freq, amp, phase):
		self.omega = tt.as_tensor_variable(2 * np.pi * freq).astype("float64")
		self.amp = tt.as_tensor_variable(amp).astype("float64")
		self.phase = tt.as_tensor_variable(phase).astype("float64")

	def get_value(self, t):
		t = tt.as_tensor_variable(t).astype("float64")
		return self.amp * tt.sin(self.omega * t + self.phase)

	def get_amplitude(self):
		return self.amp

	def get_phase(self):
		return self.phase

	def compute_gradient(self, t):
		t = tt.as_tensor_variable(t).astype("float64")
		damp = tt.sin(self.omega * t + self.phase)
		dphi = self.amp * tt.cos(self.omega * t + self.phase)
		df = 2 * np.pi * t * dphi
		return (df, damp, dphi)


class LifetimeModel:
	"""Model for variable lifetime

	Returns the positive values of the sine/cosine.

	Parameters
	----------
	harmonics : HarmonicModelCosineSine or HarmonicModelAmpPhase or list
	"""
	def __init__(self, harmonics, lower=0.):
		if not hasattr(harmonics, "getitem"):
			harmonics = [harmonics]
		self.harmonics = harmonics
		self.lower = tt.as_tensor_variable(lower).astype("float64")

	def get_value(self, t):
		t = tt.as_tensor_variable(t).astype("float64")
		tau_cs = tt.zeros(t.shape[:-1], dtype="float64")
		for h in self.harmonics:
			tau_cs += h.get_value(t)
		return tt.maximum(self.lower, tau_cs)


def _interp(x, xs, ys, fill_value=0.):
	idx = xs.searchsorted(x)
	out_of_bounds = tt.zeros(x.shape[:-1], dtype=bool)
	out_of_bounds |= (idx < 1) | (idx >= xs.shape[0])
	idx = tt.clip(idx, 1, xs.shape[0] - 1)
	dl = x - xs[idx - 1]
	dr = xs[idx] - x
	d = dl + dr
	wl = dr / d
	ret = tt.zeros(x.shape[:-1], dtype="float64")
	ret += wl * ys[idx - 1] + (1 - wl) * ys[idx]
	ret = tt.switch(out_of_bounds, fill_value, ret)
	return ret


class ProxyModel:
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
	lag : float, optional
		The lag of the proxy value in days.
	tau0 : float, optional
		The base life time of the proxy
	tau_harm : LifetimeModel, optional
		The lifetime harmonic model with a lower limit.
	tau_scan : float, optional
		The number of days to sum the previous proxy values. If it is
		negative, the value will be set to three times the maximal lifetime.
		No lifetime adjustemets are calculated when set to zero.
	days_per_time_unit : float, optional
		The number of days per time unit, used to normalize the lifetime
		units. Use 365.25 if the times are in fractional years, or 1 if
		they are in days.
		Default: 365.25
	"""
	def __init__(
		self, ptimes, pvalues, amp,
		lag=0.,
		tau0=0.,
		tau_harm=None,
		tau_scan=0,
		days_per_time_unit=365.25,
	):
		# data
		self.times = tt.as_tensor_variable(ptimes).astype("float64")
		self.values = tt.as_tensor_variable(pvalues).astype("float64")
		# parameters
		self.amp = tt.as_tensor_variable(amp).astype("float64")
		self.days_per_time_unit = tt.as_tensor_variable(days_per_time_unit).astype("float64")
		self.lag = tt.as_tensor_variable(lag / days_per_time_unit).astype("float64")
		self.tau0 = tt.as_tensor_variable(tau0).astype("float64")
		self.tau_harm = tau_harm
		self.tau_scan = tau_scan
		dt = 1.0
		bs = np.arange(dt, tau_scan + dt, dt) / days_per_time_unit
		self.bs = tt.as_tensor_variable(bs).astype("float64")
		self.dt = tt.as_tensor_variable(dt).astype("float64")
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
		self.t_adj = tt.as_tensor_variable(self.t_adj).astype("float64")

	def _lt_corr(self, t, tau):
		"""Lifetime corrected values

		Corrects for a finite lifetime by summing over the last `tmax`
		days with an exponential decay given of lifetime(s) `tau`.
		"""
		yp = tt.zeros(t.shape[:-1], dtype="float64")
		tauexp = tt.exp(-self.dt / tau)
		taufac = tt.ones(tau.shape[:-1], dtype="float64")
		for b in self.bs:
			taufac *= tauexp
			yp += taufac * _interp(
				t - b,
				self.times, self.values,
			)
		return yp * self.dt

	def get_value(self, t):
		t = tt.as_tensor_variable(t).astype("float64")
		tp = t - self.lag
		proxy_val = _interp(
			tp,
			self.times, self.values,
		)
		if self.tau_scan == 0:
			# no lifetime, nothing else to do
			return self.amp * proxy_val
		tau = self.tau0
		if self.tau_harm is not None:
			tau_cs = self.tau_harm.get_value(t + self.t_adj)
			tau += tau_cs
		proxy_val += self._lt_corr(tp, tau)
		return self.amp * proxy_val


class ModelSet:
	def __init__(self, models):
		self.models = models

	def get_value(self, t):
		t = tt.as_tensor_variable(t).astype("float64")
		v = tt.zeros(t.shape[:-1], dtype="float64")
		for m in self.models:
			v += m.get_value(t)
		return v


def setup_proxy_model_theano(
	model, name,
	times, values,
	max_amp=1e10, max_days=100,
	**kwargs
):
	warn(
		"This method to set up the `theano`/`pymc3` interface is experimental, "
		"and the interface will most likely change in future versions. "
		"It is recommended to use the `ProxyModel` class instead."
	)
	# extract setup from `kwargs`
	fit_lag = kwargs.get("fit_lag", False)
	lag = kwargs.get("lag", 0.)
	lifetime_scan = kwargs.get("lifetime_scan", 0)
	positive = kwargs.get("positive", False)
	time_format = kwargs.get("time_format", "jyear")
	days_per_time_unit = kwargs.get(
		"days_per_time_unit",
		1. if time_format.endswith("d") else 365.25
	)
	harm_freq = days_per_time_unit / 365.25

	with pm.Model(model=model, name=name):
		if positive:
			log_amp = pm.Normal("log_amp", mu=0.0, sd=np.log(max_amp))
			amp = pm.Deterministic("amp", pm.math.exp(log_amp))
		else:
			amp = pm.Normal("amp", mu=0.0, sd=max_amp)
		if fit_lag:
			log_lag = pm.Normal("log_lag", mu=0.0, sd=np.log(max_days))
			lag = pm.Deterministic("lag", pm.math.exp(log_lag))
		if lifetime_scan > 0:
			log_tau0 = pm.Normal("log_tau0", mu=0.0, sd=np.log(max_days))
			tau0 = pm.Deterministic("tau0", pm.math.exp(log_tau0))
			cos1 = pm.Normal("tau_cos1", mu=0.0, sd=max_amp)
			sin1 = pm.Normal("tau_sin1", mu=0.0, sd=max_amp)
			harm1 = HarmonicModelCosineSine(harm_freq, cos1, sin1)
			tau1 = LifetimeModel(harm1, lower=0)
		else:
			tau0 = 0.
			tau1 = None
		proxy = ProxyModel(
			times, values,
			amp,
			lag=lag,
			tau0=tau0,
			tau_harm=tau1,
			tau_scan=lifetime_scan,
			days_per_time_unit=days_per_time_unit,
		)
	return proxy


def _default_proxy_config(tfmt="jyear"):
	from .load_data import load_dailymeanLya, load_dailymeanAE
	proxy_config = {}
	# Lyman-alpha
	plyat, plyadf = load_dailymeanLya(tfmt=tfmt)
	proxy_config.update({
		"Lya": {
			"times": plyat,
			"values": plyadf["Lya"],
			"lifetime_scan": 0,
			"positive": False,
		}
	})
	# AE index
	paet, paedf = load_dailymeanAE(name="GM", tfmt=tfmt)
	proxy_config.update({
		"GM": {
			"times": paet,
			"values": paedf["GM"],
			"lifetime_scan": 30,
			"positive": True,
		}
	})
	return proxy_config


def trace_gas_modelset(constant=True, freqs=None, proxy_config=None, **kwargs):
	"""Trace gas model set

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
	model, ModelSet, offset : tuple
		The :class:`pymc3.Model` containing the random variables,
		the :class:`ModelSet` with entries of type :class:`ProxyModel` as setup up
		via ``proxy_config`` or with a default set. The offset is included
		to keep pro-forma compatibility with the ``celerite`` model setup.
	"""
	warn(
		"This method to set up the `theano`/`pymc3` interface is experimental, "
		"and the interface will most likely change in future versions. "
		"It is recommended to use the `ProxyModel` class instead."
	)
	fit_phase = kwargs.get("fit_phase", False)
	scale = kwargs.get("scale", 1e-6)
	tfmt = kwargs.get("time_format", "jyear")
	delta_t = kwargs.get("days_per_time_unit", 365.25)

	max_amp = kwargs.pop("max_amp", 1e10 * scale)
	max_days = kwargs.pop("max_days", 100)

	freqs = freqs or []
	proxy_config = proxy_config or _default_proxy_config(tfmt=tfmt)

	with pm.Model() as model:
		offset = 0.
		if constant:
			offset = pm.Normal("offset", mu=0.0, sd=max_amp)

		modelset = []
		for freq in freqs:
			if not fit_phase:
				cos = pm.Normal("cos{0}".format(freq), mu=0., sd=max_amp)
				sin = pm.Normal("sin{0}".format(freq), mu=0., sd=max_amp)
				harm = HarmonicModelCosineSine(
					freq * delta_t / 365.25,
					cos, sin,
				)
			else:
				amp = pm.Normal("amp{0}".format(freq), mu=0., sd=max_amp)
				phase = pm.Normal("phase{0}".format(freq), mu=0., sd=max_amp)
				harm = HarmonicModelAmpPhase(
					freq * delta_t / 365.25,
					amp, phase,
				)
			modelset.append(harm)

		for pn, conf in proxy_config.items():
			if "max_amp" not in conf:
				conf.update(dict(max_amp=max_amp))
			if "max_days" not in conf:
				conf.update(dict(max_days=max_days))
			kw = kwargs.copy()  # don't mess with the passed arguments
			kw.update(conf)
			modelset.append(
				setup_proxy_model_theano(model, pn, **kw)
			)

	return model, ModelSet(modelset), offset
