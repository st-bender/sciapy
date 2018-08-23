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
"""SCIAMACHY data regression MCMC sampler

Markov Chain Monte Carlo (MCMC) routines to sample
the posterior probability of SCIAMACHY data regression fits.
Uses the `emcee.EnsembleSampler` [1] do do the real work.

[1](https://emcee.readthedocs.io)
"""

import logging
from multiprocessing import Pool

import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver

import celerite
import george
import emcee

__all__ = ["mcmc_sample_model"]


def _lpost(p, model, y=None, beta=1.):
	model.set_parameter_vector(p)
	lprior = model.log_prior()
	if not np.isfinite(lprior):
		return -np.inf
	return beta * (model.log_likelihood(y, quiet=True) + lprior)


def mcmc_sample_model(model, y, beta=1.,
		nwalkers=100, nburnin=200, nprod=800,
		nthreads=1, optimized=False,
		bounds=None,
		return_logpost=False,
		show_progress=False, progress_mod=10):
	"""Markov Chain Monte Carlo sampling interface

	MCMC sampling interface to sample posterior probabilities using the
	`emcee.EnsembleSampler`[1].

	[1](https://emcee.readthedocs.io)

	Arguments
	---------
	model : celerite.GP, george.GP or sciapy.regress.RegressionModel instance
		The model to draw posterior samples from. It should provide either
		`log_likelihood()` and `log_prior()` functions be directly callable
		via `__call__()`.
	y : (N,) array_like
		The data to condition the probabilities on.
	beta : float, optional
		Tempering factor for the probability, default: 1.
	nwalkers : int, optional
		The number of MCMC walkers (default: 100). If this number is smaller
		than 4 times the number of parameters, it is multiplied by the number
		of parameters. Otherwise it specifies the number of parameters directly.
	nburnin : int, optional
		The number of burn-in samples to draw, default: 200.
	nprod : int, optional
		The number of production samples to draw, default: 800.
	nthreads : int, optional
		The number of threads to use with a `multiprocessing.Pool`,
		used as `pool` for `emcee.EnsembleSampler`. Default: 1.
	optimized : bool, optional
		Indicate whether the actual (starting) position was determined with an
		optimization algorithm beforehand. If `False` (the default), a
		pre-burn-in run optimizes the starting position. Sampling continues
		from there with the normal burn-in and production runs.
		In that case, latin hypercube sampling is used to distribute the walker
		starting positions equally in parameter space.
	bounds : iterable, optional
		The parameter bounds as a list of (min, max) entries.
		Default: None
	return_logpost : bool, optional
		Indicate whether or not to  return the sampled log probabilities as well.
		Default: False
	show_progress : bool, optional
		Print the percentage of samples every `progress_mod` samples.
		Default: False
	progress_mod : int, optional
		Interval in samples to print the percentage of samples.
		Default: 10

	Returns
	-------
	samples or (samples, logpost) : array_like or tuple
		(nwalkers * nprod, ndim) array of the sampled parameters from the
		production run if return_logpost is `False`.
		A tuple of an (nwalkers * nprod, ndim) array (the same as above)
		and an (nwalkers,) array with the second entry containing the
		log posterior probabilities if return_logpost is `True`.
	"""
	v = model.get_parameter_vector()
	ndim = len(v)
	if nwalkers < 4 * ndim:
		nwalkers *= ndim
	logging.info("MCMC parameters: %s walkers, %s burn-in samples, "
				"%s production samples using %s threads.",
				nwalkers, nburnin, nprod, nthreads)

	if isinstance(model, celerite.GP) or isinstance(model, george.GP):
		mod_func = _lpost
		mod_args = (model, y, beta)
	else:
		mod_func = model
		mod_args = (beta,)

	# Initialize the walkers.
	if not optimized:
		# scipy.optimize's DifferentialEvolutionSolver uses
		# latin hypercube sampling as starting positions.
		# We just use their initialization to avoid duplicating code.
		if bounds is None:
			bounds = model.get_parameter_bounds()
		de_solver = DifferentialEvolutionSolver(_lpost,
					bounds=bounds,
					popsize=nwalkers // ndim)
		# The initial population should reflect latin hypercube sampling
		p0 = de_solver.population
	else:
		p0 = np.array([v + 1e-2 * np.random.randn(ndim) for _ in range(nwalkers)])

	# set up the sampling pool
	pool = Pool(processes=nthreads)
	sampler = emcee.EnsembleSampler(nwalkers, ndim, mod_func, args=mod_args,
			pool=pool)

	logging.info("Running burn-in ({0} samples)".format(nburnin))
	for i, result in enumerate(sampler.sample(p0, iterations=nburnin)):
		if show_progress and (i + 1) % progress_mod == 0:
			logging.info("{0:5.1%}".format(float(i + 1) / nburnin))
			pp, lnpp, _ = result
			logging.debug("lnpmax: %s, p(lnpmax): %s",
					np.max(lnpp), pp[np.argmax(lnpp)])
	p0, lnp0, rst0 = result
	logging.info("Burn-in finished.")

	p = p0[np.argmax(lnp0)]
	logging.info("burn-in max logpost: %s, params: %s, exp(params): %s",
				np.max(lnp0), p, np.exp(p))
	model.set_parameter_vector(p)
	logging.debug("params: %s", model.get_parameter_dict())
	logging.debug("log_likelihood: %s", model.log_likelihood(y))
	sampler.reset()

	if not optimized:
		p0 = [p + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)]
		logging.info("Running second burn-in ({0} samples)".format(nburnin))
		for i, result in enumerate(
				sampler.sample(p0, lnp0, rstate0=rst0, iterations=nburnin)):
			if show_progress and (i + 1) % progress_mod == 0:
				logging.info("{0:5.1%}".format(float(i + 1) / nburnin))
		p0, lnp0, rst0 = result
		sampler.reset()
		logging.info("Second burn-in finished.")

	logging.info("Running production chain ({0} samples)".format(nprod))
	for i, result in enumerate(
			sampler.sample(p0, lnp0, rstate0=rst0, iterations=nprod)):
		if show_progress and (i + 1) % progress_mod == 0:
			logging.info("{0:5.1%}".format(float(i + 1) / nprod))
	logging.info("Production run finished.")

	samples = sampler.flatchain
	lnp = sampler.flatlnprobability
	logging.info("total samples: %s", samples.shape)

	samplmean = np.mean(samples, axis=0)
	logging.info("mean: %s, exp(mean): %s, sqrt(exp(mean)): %s",
			samplmean, np.exp(samplmean), np.sqrt(np.exp(samplmean)))

	samplmedian = np.median(samples, axis=0)
	logging.info("median: %s, exp(median): %s, sqrt(exp(median)): %s",
			samplmedian, np.exp(samplmedian), np.sqrt(np.exp(samplmedian)))

	logging.info("max logpost: %s, params: %s, exp(params): %s",
			np.max(lnp), samples[np.argmax(lnp)],
			np.exp(samples[np.argmax(lnp)]))

	logging.info("AIC: %s", 2 * ndim - 2 * np.max(lnp))
	logging.info("BIC: %s", np.log(len(y)) * ndim - 2 * np.max(lnp))
	logging.info("poor man's evidence 1 sum: %s, mean: %s", np.sum(np.exp(lnp)), np.mean(np.exp(lnp)))
	logging.info("poor man's evidence 2 max: %s, std: %s", np.max(np.exp(lnp)), np.std(np.exp(lnp)))
	logging.info("poor man's evidence 3: %s", np.max(np.exp(lnp)) / np.std(np.exp(lnp)))

	if return_logpost:
		return samples, lnp
	return samples
