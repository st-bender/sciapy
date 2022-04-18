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
	import arviz as az
except (ImportError, ModuleNotFoundError):
	pytest.skip("Theano/PyMC3 packages not installed", allow_module_level=True)

try:
	from sciapy.regress.models_theano import (
		HarmonicModelCosineSine,
		HarmonicModelAmpPhase,
	)
except (ImportError, ModuleNotFoundError):
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
		cos = pm.Normal("cos", mu=0.0, sd=4.0)
		sin = pm.Normal("sin", mu=0.0, sd=4.0)
		harm1 = HarmonicModelCosineSine(1., cos, sin)
		wave1 = harm1.get_value(xs)
		# add amplitude and phase for comparison
		pm.Deterministic("amp", harm1.get_amplitude())
		pm.Deterministic("phase", harm1.get_phase())
		resid1 = yp - wave1
		pm.Normal("obs", mu=0.0, observed=resid1)
		trace1 = pm.sample(tune=800, draws=800, chains=2, return_inferencedata=True)

	with pm.Model() as model2:
		amp2 = pm.HalfNormal("amp", sigma=4.0)
		phase2 = pm.Normal("phase", mu=0.0, sd=4.0)
		harm2 = HarmonicModelAmpPhase(1., amp2, phase2)
		wave2 = harm2.get_value(xs)
		resid2 = yp - wave2
		pm.Normal("obs", mu=0.0, observed=resid2)
		trace2 = pm.sample(tune=800, draws=800, chains=2, return_inferencedata=True)

	np.testing.assert_allclose(
		trace1.posterior.median(dim=("chain", "draw"))[["cos", "sin"]].to_array(),
		(c, s),
		atol=1e-2,
	)
	np.testing.assert_allclose(
		trace1.posterior.median(dim=("chain", "draw"))[["amp", "phase"]].to_array(),
		trace2.posterior.median(dim=("chain", "draw"))[["amp", "phase"]].to_array(),
		atol=3e-3,
	)
