# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2018-2022 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY regression module tests
"""
import numpy as np

import pytest

try:
	import pymc3 as pm
except ImportError:
	pytest.skip("Theano/PyMC3 packages not installed", allow_module_level=True)

try:
	from sciapy.regress.models_theano import (
		HarmonicModelCosineSine,
		HarmonicModelAmpPhase,
		LifetimeModel,
		ProxyModel,
	)
except ImportError:
	pytest.skip("Theano/PyMC3 interface not installed", allow_module_level=True)


@pytest.fixture(scope="module")
def xs():
	_xs = np.linspace(0., 11.1, 2048)
	return np.ascontiguousarray(_xs, dtype=np.float64)


def ys(xs, c, s):
	_ys = c * np.cos(2 * np.pi * xs) + s * np.sin(2 * np.pi * xs)
	return np.ascontiguousarray(_ys, dtype=np.float64)


@pytest.mark.parametrize(
	"c, s",
	[
		(0.5, 2.0),
		(1.0, 0.5),
		(1.0, 1.0),
	]
)
def test_harmonics_theano(xs, c, s):
	# Initialize random number generator
	np.random.seed(93457)
	yp = ys(xs, c, s)
	yp += 0.5 * np.random.randn(xs.shape[0])

	with pm.Model() as model1:
		cos = pm.Normal("cos", mu=0.0, sigma=4.0)
		sin = pm.Normal("sin", mu=0.0, sigma=4.0)
		harm1 = HarmonicModelCosineSine(1., cos, sin)
		wave1 = harm1.get_value(xs)
		# add amplitude and phase for comparison
		pm.Deterministic("amp", harm1.get_amplitude())
		pm.Deterministic("phase", harm1.get_phase())
		pm.Normal("obs", mu=wave1, observed=yp)
		trace1 = pm.sample(tune=400, draws=400, chains=2, return_inferencedata=True)

	with pm.Model() as model2:
		amp2 = pm.HalfNormal("amp", sigma=4.0)
		phase2 = pm.Normal("phase", mu=0.0, sigma=4.0)
		harm2 = HarmonicModelAmpPhase(1., amp2, phase2)
		wave2 = harm2.get_value(xs)
		pm.Normal("obs", mu=wave2, observed=yp)
		trace2 = pm.sample(tune=400, draws=400, chains=2, return_inferencedata=True)

	np.testing.assert_allclose(
		trace1.posterior.median(dim=("chain", "draw"))[["cos", "sin"]].to_array(),
		(c, s),
		atol=1e-2,
	)
	np.testing.assert_allclose(
		trace1.posterior.median(dim=("chain", "draw"))[["amp", "phase"]].to_array(),
		trace2.posterior.median(dim=("chain", "draw"))[["amp", "phase"]].to_array(),
		atol=4e-3,
	)


def _test_data(xs, values, f, c, s):
	amp = 3.
	lag = 2.
	tau0 = 1.
	harm0 = HarmonicModelCosineSine(f, c, s)
	tau_lt0 = LifetimeModel(harm0, lower=0.)
	proxy0 = ProxyModel(
		xs, values,
		amp=amp,
		lag=lag,
		tau0=tau0,
		tau_harm=tau_lt0,
		tau_scan=10,
		days_per_time_unit=f * 365.25,
	)
	return proxy0.get_value(xs).eval()


def _yy(x, c, s):
	_ys = np.zeros_like(x)
	_ys[10::20] = 10.
	return np.ascontiguousarray(_ys, dtype=np.float64)


@pytest.mark.long
def test_proxy_theano(xx, f=1, c=3.0, s=1.0):
	# Initialize random number generator
	np.random.seed(93457)

	# proxy "values"
	values = _yy(xs, c, s)

	yp = _test_data(xs, values, f, c, s)
	yp += 0.5 * np.random.randn(xs.shape[0])

	# using "name" prefixes all variables with <name>_
	with pm.Model(name="proxy") as model:
		# amplitude
		pamp = pm.Normal("amp", mu=0.0, sigma=4.0)
		# lag
		plag = pm.Lognormal("lag", mu=0.0, sigma=4.0, testval=1.0)
		# lifetime
		ptau0 = pm.Lognormal("tau0", mu=0.0, sigma=4.0, testval=1.0)
		cos1 = pm.Normal("tau_cos1", mu=0.0, sigma=10.0)
		sin1 = pm.Normal("tau_sin1", mu=0.0, sigma=10.0)
		harm1 = HarmonicModelCosineSine(f, cos1, sin1)
		tau1 = LifetimeModel(harm1, lower=0)

		proxy = ProxyModel(
			xs, values,
			amp=pamp,
			lag=plag,
			tau0=ptau0,
			tau_harm=tau1,
			tau_scan=10,
			days_per_time_unit=f * 365.25,
		)
		prox1 = proxy.get_value(xs)
		# Include "jitter"
		log_jitter = pm.Normal("log_jitter", mu=0.0, sigma=4.0)
		pm.Normal("obs", mu=prox1, sigma=pm.math.exp(log_jitter), observed=yp)

		maxlp0 = pm.find_MAP()
		trace = pm.sample(
			chains=2,
			draws=400,
			tune=400,
			init="jitter+adapt_full",
			random_seed=[286923464, 464329682],
			return_inferencedata=True,
			start=maxlp0,
			target_accept=0.9,
		)

	medians = trace.posterior.median(dim=("chain", "draw"))
	var_names = [
		model.name_for(n)
		for n in [
			"amp", "lag", "tau0", "tau_cos1", "tau_sin1", "log_jitter",
		]
	]
	np.testing.assert_allclose(
		medians[var_names].to_array(),
		(3., 2., 1., c, s, np.log(0.5)),
		atol=3e-2, rtol=1e-2,
	)
