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
Uses the :class:`emcee.EnsembleSampler` [#]_ do do the real work.

.. [#] https://emcee.readthedocs.io
"""

import logging
from multiprocessing import Pool

import numpy as np
from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
from scipy.special import logsumexp

import celerite
import george
import emcee

try:
	from tqdm import tqdm
	have_tqdm = True
except ImportError:
	have_tqdm = False


__all__ = ["mcmc_sample_model"]


def _lpost(p, model, y=None, beta=1.):
	model.set_parameter_vector(p)
	lprior = model.log_prior()
	if not np.isfinite(lprior):
		return (-np.inf, [np.nan])
	log_likelihood = model.log_likelihood(y, quiet=True)
	return (beta * (log_likelihood + lprior), [log_likelihood])


def _sample_mcmc(sampler, nsamples, p0, rst0,
		show_progress, progress_mod, debug=False):
	smpl = sampler.sample(p0, rstate0=rst0, iterations=nsamples)
	if have_tqdm and show_progress:
		smpl = tqdm(smpl, total=nsamples, disable=None)
	for i, result in enumerate(smpl):
		if show_progress and (i + 1) % progress_mod == 0:
			if not have_tqdm and not debug:
				logging.info("%5.1f%%", 100 * (float(i + 1) / nsamples))
			if debug:
				_pos, _logp, _, _ = result
				logging.debug("%5.1f%% lnpmax: %s, p(lnpmax): %s",
					100 * (float(i + 1) / nsamples),
					np.max(_logp), _pos[np.argmax(_logp)])
	return result


def mcmc_sample_model(model, y, beta=1.,
		nwalkers=100, nburnin=200, nprod=800,
		nthreads=1, optimized=False,
		bounds=None,
		return_logpost=False,
		show_progress=False, progress_mod=10):
	"""Markov Chain Monte Carlo sampling interface

	MCMC sampling interface to sample posterior probabilities using the
	:class:`emcee.EnsembleSampler` [#]_.

	.. [#] https://emcee.readthedocs.io

	Arguments
	---------
	model : celerite.GP, george.GP, or sciapy.regress.RegressionModel instance
		The model to draw posterior samples from. It should provide either
		`log_likelihood()` and `log_prior()` functions or be directly callable
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
		# fill up to full size in case the number of walkers is not a
		# multiple of the number of parameters
		missing = nwalkers - p0.shape[0]
		p0 = np.vstack([p0] +
			[v + 1e-2 * np.random.randn(ndim) for _ in range(missing)])
	else:
		p0 = np.array([v + 1e-2 * np.random.randn(ndim) for _ in range(nwalkers)])

	# set up the sampling pool
	if nthreads > 1:
		pool = Pool(processes=nthreads)
	else:
		pool = None
	sampler = emcee.EnsembleSampler(nwalkers, ndim, mod_func, args=mod_args,
			pool=pool)

	rst0 = np.random.get_state()

	if not optimized:
		logging.info("Running MCMC fit (%s samples)", nburnin)
		p0, lnp0, rst0, _ = _sample_mcmc(sampler, nburnin, p0, rst0,
				show_progress, progress_mod, debug=True)
		logging.info("MCMC fit finished.")

		p = p0[np.argmax(lnp0)]
		logging.info("Fit max logpost: %s, params: %s, exp(params): %s",
					np.max(lnp0), p, np.exp(p))
		model.set_parameter_vector(p)
		logging.debug("params: %s", model.get_parameter_dict())
		logging.debug("log_likelihood: %s", model.log_likelihood(y))
		p0 = [p + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)]
		sampler.reset()

	logging.info("Running burn-in (%s samples)", nburnin)
	p0, lnp0, rst0, _ = _sample_mcmc(sampler, nburnin, p0, rst0,
			show_progress, progress_mod)
	logging.info("Burn-in finished.")

	p = p0[np.argmax(lnp0)]
	logging.info("burn-in max logpost: %s, params: %s, exp(params): %s",
				np.max(lnp0), p, np.exp(p))
	model.set_parameter_vector(p)
	logging.debug("params: %s", model.get_parameter_dict())
	logging.debug("log_likelihood: %s", model.log_likelihood(y))
	sampler.reset()

	logging.info("Running production chain (%s samples)", nprod)
	_sample_mcmc(sampler, nprod, p0, rst0, show_progress, progress_mod)
	logging.info("Production run finished.")

	samples = sampler.flatchain
	lnp = sampler.flatlnprobability
	# first column in the blobs are the log likelihoods
	lnlh = np.array(sampler.blobs)[..., 0].ravel().astype(float)
	post_expect_loglh = np.nanmean(np.array(lnlh))
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
	logging.info("poor man's evidence 1 sum: %s, mean: %s",
			np.sum(np.exp(lnp)), np.mean(np.exp(lnp)))
	logging.info("poor man's evidence 2 max: %s, std: %s",
			np.max(np.exp(lnp)), np.std(np.exp(lnp)))
	logging.info("poor man's evidence 3: %s",
			np.max(np.exp(lnp)) / np.std(np.exp(lnp)))
	logging.info("poor man's evidence 4 sum: %s",
			logsumexp(lnp, b=1. / lnp.shape[0], axis=0))

	# mode
	model.set_parameter_vector(samples[np.argmax(lnp)])
	log_lh = model.log_likelihood(y)
	# Use the likelihood instead of the posterior
	# https://doi.org/10.3847/1538-3881/aa9332
	logging.info("BIC lh: %s", np.log(len(y)) * ndim - 2 * log_lh)
	# DIC
	sample_deviance = -2 * np.max(lnp)
	deviance_at_sample = -2 * (model.log_prior() + log_lh)
	pd = sample_deviance - deviance_at_sample
	dic = 2 * sample_deviance - deviance_at_sample
	logging.info("max logpost log_lh: %s, AIC: %s, DIC: %s, pd: %s",
			model.log_likelihood(y), 2 * ndim - 2 * log_lh, dic, pd)
	# mean
	model.set_parameter_vector(samplmean)
	log_lh = model.log_likelihood(y)
	log_lh_mean = log_lh
	# DIC
	sample_deviance = -2 * np.nanmean(lnp)
	deviance_at_sample = -2 * (model.log_prior() + log_lh)
	pd = sample_deviance - deviance_at_sample
	dic = 2 * sample_deviance - deviance_at_sample
	logging.info("mean log_lh: %s, AIC: %s, DIC: %s, pd: %s",
			model.log_likelihood(y), 2 * ndim - 2 * log_lh, dic, pd)
	# median
	model.set_parameter_vector(samplmedian)
	log_lh = model.log_likelihood(y)
	# DIC
	sample_deviance = -2 * np.nanmedian(lnp)
	deviance_at_sample = -2 * (model.log_prior() + log_lh)
	dic = 2 * sample_deviance - deviance_at_sample
	pd = sample_deviance - deviance_at_sample
	logging.info("median log_lh: %s, AIC: %s, DIC: %s, pd: %s",
			model.log_likelihood(y), 2 * ndim - 2 * log_lh, dic, pd)
	# (4)--(6) in Ando2011 doi:10.1080/01966324.2011.10737798
	pd_ando = 2 * (log_lh_mean - post_expect_loglh)
	ic5 = - 2 * post_expect_loglh + 2 * pd_ando
	ic6 = - 2 * post_expect_loglh + 2 * ndim
	logging.info("Ando2011: pd: %s, IC(5): %s, IC(6): %s",
			pd_ando, ic5, ic6)

	if return_logpost:
		return samples, lnp
	return samples
