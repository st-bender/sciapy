# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2017-2019 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY MCMC statistic tools

Statistical functions for MCMC sampled parameters.
"""

import logging

import numpy as np
from scipy import linalg

__all__ = ["mcmc_statistics", "waic_loo"]


def _log_prob(resid, var):
	return -0.5 * (np.log(2 * np.pi * var) + resid**2 / var)


def _log_lh_pt(gp, times, data, errs, s):
	gp.set_parameter_vector(s)
	resid = data - gp.mean.get_value(times)
	ker = gp.get_matrix(times, include_diagonal=True)
	ker[np.diag_indices_from(ker)] += errs**2
	ll, lower = linalg.cho_factor(
			ker, lower=True, check_finite=False, overwrite_a=True)
	linv_r = linalg.solve_triangular(
			ll, resid, lower=True, check_finite=False, overwrite_b=True)
	return -0.5 * (np.log(2. * np.pi) + linv_r**2) - np.log(np.diag(ll))


def _log_pred_pt(gp, train_data, times, data, errs, s):
	gp.set_parameter_vector(s)
	gppred, pvar_gp = gp.predict(train_data, t=times, return_var=True)
	# the predictive covariance should include the data variance
	# if noisy_targets and errs is not None:
	if errs is not None:
		pvar_gp += errs**2
	# residuals
	resid_gp = gppred - data  # GP residuals
	return _log_prob(resid_gp, pvar_gp)


def _log_prob_pt_samples_dask(log_p_pt, samples,
		nthreads=1, cluster=None):
	from dask.distributed import Client, LocalCluster, progress
	import dask.bag as db

	# calculate the point-wise probabilities and stack them together
	if cluster is None:
		# start local dask cluster
		_cl = LocalCluster(n_workers=nthreads, threads_per_worker=1)
	else:
		# use provided dask cluster
		_cl = cluster

	with Client(_cl):
		_log_pred = db.from_sequence(samples).map(log_p_pt)
		progress(_log_pred)
		ret = np.stack(_log_pred.compute())

	if cluster is None:
		_cl.close()

	return ret


def _log_prob_pt_samples_mt(log_p_pt, samples, nthreads=1):
	from multiprocessing import pool
	try:
		from tqdm.autonotebook import tqdm
	except ImportError:
		tqdm = None

	# multiprocessing.pool
	_p = pool.Pool(processes=nthreads)

	_mapped = _p.imap_unordered(log_p_pt, samples)
	if tqdm is not None:
		_mapped = tqdm(_mapped, total=len(samples))

	ret = np.stack(list(_mapped))

	_p.close()
	_p.join()

	return ret


def waic_loo(model, times, data, errs,
		samples,
		method="likelihood",
		train_data=None,
		noisy_targets=True,
		nthreads=1,
		use_dask=False,
		dask_cluster=None,
		):
	"""Watanabe-Akaike information criterion (WAIC) and LOO IC of the (GP) model

	Calculates the WAIC and leave-one-out (LOO) cross validation scores and
	information criteria (IC) from the MCMC samples of the posterior parameter
	distributions. Uses the posterior point-wise (per data point)
	probabilities and the formulae from [1]_ and [2]_.

	.. [1] Vehtari, Gelman, and Gabry, Stat Comput (2017) 27:1413–1432,
		doi: 10.1007/s11222-016-9696-4

	.. [2] Vehtari and Gelman, (unpublished)
		http://www.stat.columbia.edu/~gelman/research/unpublished/waic_stan.pdf
		http://www.stat.columbia.edu/~gelman/research/unpublished/loo_stan.pdf

	Parameters
	----------
	model : `celerite.GP`, `george.GP` or `CeleriteModelSet` instance
		The model instance whose parameter distribution was drawn.
	times : (M,) array_like
		The test coordinates to predict or evaluate the model on.
	data : (M,) array_like
		The test data to test the model against.
	errs : (M,) array_like
		The errors (variances) of the test data.
	samples : (K, L) array_like
		The `K` MCMC samples of the `L` parameter distributions.
	method : str ("likelihood" or "predict"), optional
		The method to "predict" the data, the default uses the (log)likelihood
		in the same way as is done when fitting (training) the model.
		"predict" uses the actual GP prediction, might be useful if the IC
		should be estimated for actual test data that was not used to train
		the model.
	train_data : (N,) array_like, optional
		The data on which the model was trained, needed if method="predict" is
		used, otherwise None is the default and the likelihood is used.
	noisy_targets : bool, optional
		Include the given errors when calculating the predictive probability.
	nthreads : int, optional
		Number of threads to distribute the point-wise probability
		calculations to (default: 1).
	use_dask : boot, optional
		Use `dask.distributed` to distribute the point-wise probability
		calculations to `nthreads` workers. The default is to use
		`multiprocessing.pool.Pool()`.
	dask_cluster: str, or `dask.distributed.Cluster` instance, optional
		Will be passed to `dask.distributed.Client()`
		This can be the address of a Scheduler server like a string
		'127.0.0.1:8786' or a cluster object like `dask.distributed.LocalCluster()`.

	Returns
	-------
	waic, waic_se, p_waic, loo_ic, loo_se, p_loo : tuple
		The WAIC and its standard error as well as the
		estimated effective number of parameters, p_waic.
		The LOO IC, its standard error, and the estimated
		effective number of parameters, p_loo.
	"""
	from functools import partial
	from scipy.special import logsumexp

	# the predictive covariance should include the data variance
	# set to a small value if we don't want to account for them
	if not noisy_targets or errs is None:
		errs = 1.123e-12

	# point-wise posterior/predictive probabilities
	_log_p_pt = partial(_log_lh_pt, model, times, data, errs)
	if method == "predict" and train_data is not None:
		_log_p_pt = partial(_log_pred_pt, model, train_data, times, data, errs)

	# calculate the point-wise probabilities and stack them together
	if nthreads > 1 and use_dask:
		log_pred = _log_prob_pt_samples_dask(_log_p_pt, samples,
				nthreads=nthreads, cluster=dask_cluster)
	else:
		log_pred = _log_prob_pt_samples_mt(_log_p_pt, samples,
				nthreads=nthreads)

	lppd_i = logsumexp(log_pred, b=1. / log_pred.shape[0], axis=0)
	p_waic_i = np.nanvar(log_pred, ddof=1, axis=0)
	if np.any(p_waic_i > 0.4):
		logging.warn("""For one or more samples the posterior variance of the
		log predictive densities exceeds 0.4. This could be indication of
		WAIC starting to fail see http://arxiv.org/abs/1507.04544 for details
		""")
	elpd_i = lppd_i - p_waic_i
	waic_i = -2. * elpd_i
	waic_se = np.sqrt(len(waic_i) * np.nanvar(waic_i, ddof=1))
	waic = np.nansum(waic_i)
	p_waic = np.nansum(p_waic_i)
	if 2. * p_waic > len(waic_i):
		logging.warn("""p_waic > n / 2,
		the WAIC approximation is unreliable.
		""")
	logging.info("WAIC: %s, waic_se: %s, p_w: %s", waic, waic_se, p_waic)

	# LOO
	loo_ws = 1. / np.exp(log_pred - np.nanmax(log_pred, axis=0))
	loo_ws_n = loo_ws / np.nanmean(loo_ws, axis=0)
	loo_ws_r = np.clip(loo_ws_n, None, np.sqrt(log_pred.shape[0]))
	elpd_loo_i = logsumexp(log_pred,
			b=loo_ws_r / np.nansum(loo_ws_r, axis=0),
			axis=0)
	p_loo_i = lppd_i - elpd_loo_i
	loo_ic_i = -2 * elpd_loo_i
	loo_ic_se = np.sqrt(len(loo_ic_i) * np.nanvar(loo_ic_i))
	loo_ic = np.nansum(loo_ic_i)
	p_loo = np.nansum(p_loo_i)
	logging.info("loo IC: %s, se: %s, p_loo: %s", loo_ic, loo_ic_se, p_loo)

	# van der Linde, 2005, Statistica Neerlandica, 2005
	# https://doi.org/10.1111/j.1467-9574.2005.00278.x
	hy1 = -np.nanmean(lppd_i)
	hy2 = -np.nanmedian(lppd_i)
	logging.info("H(Y): mean %s, median: %s", hy1, hy2)

	return waic, waic_se, p_waic, loo_ic, loo_ic_se, p_loo


def mcmc_statistics(model, times, data, errs,
		train_times, train_data, train_errs,
		samples, lnp,
		median=False):
	"""Statistics for the (GP) model against the provided data

	Statistical information about the model and the sampled parameter
	distributions with respect to the provided data and its variance.

	Sends the calculated values to the logger, includes the mean
	standardized log loss as described in R&W, 2006, section 2.5, (2.34),
	and some slightly adapted $\\chi^2_{\\text{red}}$ and $R^2$ scores.

	Parameters
	----------
	model : `celerite.GP`, `george.GP` or `CeleriteModelSet` instance
		The model instance whose parameter distribution was drawn.
	times : (M,) array_like
		The test coordinates to predict or evaluate the model on.
	data : (M,) array_like
		The test data to test the model against.
	errs : (M,) array_like
		The errors (variances) of the test data.
	train_times : (N,) array_like
		The coordinates on which the model was trained.
	train_data : (N,) array_like
		The data on which the model was trained.
	train_errs : (N,) array_like
		The errors (variances) of the training data.
	samples : (K, L) array_like
		The `K` MCMC samples of the `L` parameter distributions.
	lnp : (K,) array_like
		The posterior log probabilities of the `K` MCMC samples.
	median : bool, optional
		Whether to use the median of the sampled distributions or
		the maximum posterior sample (the default) to evaluate the
		statistics.

	Returns
	-------
	nothing
	"""
	ndat = len(times)
	ndim = len(model.get_parameter_vector())
	mdim = len(model.mean.get_parameter_vector())
	samples_max_lp = np.max(lnp)
	if median:
		sample_pos = np.nanmedian(samples, axis=0)
	else:
		sample_pos = samples[np.argmax(lnp)]
	model.set_parameter_vector(sample_pos)
	# calculate the GP predicted values and covariance
	gppred, gpcov = model.predict(train_data, t=times, return_cov=True)
	# the predictive covariance should include the data variance
	gpcov[np.diag_indices_from(gpcov)] += errs**2
	# residuals
	resid_mod = model.mean.get_value(times) - data  # GP mean model
	resid_gp = gppred - data  # GP prediction
	resid_triv = np.nanmean(train_data) - data  # trivial model
	_const = ndat * np.log(2.0 * np.pi)
	test_logpred = -0.5 * (resid_gp.dot(linalg.solve(gpcov, resid_gp))
			+ np.trace(np.log(gpcov))
			+ _const)
	# MSLL -- mean standardized log loss
	# as described in R&W, 2006, section 2.5, (2.34)
	var_mod = np.nanvar(resid_mod, ddof=mdim)  # mean model variance
	var_gp = np.nanvar(resid_gp, ddof=ndim)  # gp model variance
	var_triv = np.nanvar(train_data, ddof=1)  # trivial model variance
	logpred_mod = _log_prob(resid_mod, var_mod)
	logpred_gp = _log_prob(resid_gp, var_gp)
	logpred_triv = _log_prob(resid_triv, var_triv)
	logging.info("MSLL mean: %s", np.nanmean(-logpred_mod + logpred_triv))
	logging.info("MSLL gp: %s", np.nanmean(-logpred_gp + logpred_triv))
	# predictive variances
	logpred_mod = _log_prob(resid_mod, var_mod + errs**2)
	logpred_gp = _log_prob(resid_gp, var_gp + errs**2)
	logpred_triv = _log_prob(resid_triv, var_triv + errs**2)
	logging.info("pred MSLL mean: %s", np.nanmean(-logpred_mod + logpred_triv))
	logging.info("pred MSLL gp: %s", np.nanmean(-logpred_gp + logpred_triv))
	# cost values
	cost_mod = np.sum(resid_mod**2)
	cost_triv = np.sum(resid_triv**2)
	cost_gp = np.sum(resid_gp**2)
	# chi^2 (variance corrected costs)
	chisq_mod_ye = np.sum((resid_mod / errs)**2)
	chisq_triv = np.sum((resid_triv / errs)**2)
	chisq_gpcov = resid_mod.dot(linalg.solve(gpcov, resid_mod))
	# adjust for degrees of freedom
	cost_gp_dof = cost_gp / (ndat - ndim)
	cost_mod_dof = cost_mod / (ndat - mdim)
	cost_triv_dof = cost_triv / (ndat - 1)
	# reduced chi^2
	chisq_red_mod_ye = chisq_mod_ye / (ndat - mdim)
	chisq_red_triv = chisq_triv / (ndat - 1)
	chisq_red_gpcov = chisq_gpcov / (ndat - ndim)
	# "generalized" R^2
	logp_triv1 = np.sum(_log_prob(resid_triv, errs**2))
	logp_triv2 = np.sum(_log_prob(resid_triv, var_triv))
	logp_triv3 = np.sum(_log_prob(resid_triv, var_triv + errs**2))
	log_lambda1 = test_logpred - logp_triv1
	log_lambda2 = test_logpred - logp_triv2
	log_lambda3 = test_logpred - logp_triv3
	gen_rsq1a = 1 - np.exp(-2 * log_lambda1 / ndat)
	gen_rsq1b = 1 - np.exp(-2 * log_lambda1 / (ndat - ndim))
	gen_rsq2a = 1 - np.exp(-2 * log_lambda2 / ndat)
	gen_rsq2b = 1 - np.exp(-2 * log_lambda2 / (ndat - ndim))
	gen_rsq3a = 1 - np.exp(-2 * log_lambda3 / ndat)
	gen_rsq3b = 1 - np.exp(-2 * log_lambda3 / (ndat - ndim))
	# sent to the logger
	logging.info("train max logpost: %s", samples_max_lp)
	logging.info("test log_pred: %s", test_logpred)
	logging.info("1a cost mean model: %s, dof adj: %s", cost_mod, cost_mod_dof)
	logging.debug("1c cost gp predict: %s, dof adj: %s", cost_gp, cost_gp_dof)
	logging.debug("1b cost triv model: %s, dof adj: %s", cost_triv, cost_triv_dof)
	logging.info("1d var resid mean model: %s, gp model: %s, triv: %s",
			var_mod, var_gp, var_triv)
	logging.info("2a adjR2 mean model: %s, adjR2 gp predict: %s",
			1 - cost_mod_dof / cost_triv_dof, 1 - cost_gp_dof / cost_triv_dof)
	logging.info("2b red chi^2 mod: %s / triv: %s = %s",
			chisq_red_mod_ye, chisq_red_triv, chisq_red_mod_ye / chisq_red_triv)
	logging.info("2c red chi^2 mod (gp cov): %s / triv: %s = %s",
			chisq_red_gpcov, chisq_red_triv, chisq_red_gpcov / chisq_red_triv)
	logging.info("3a stand. red chi^2: %s", chisq_red_gpcov / chisq_red_triv)
	logging.info("3b 1 - stand. red chi^2: %s",
			1 - chisq_red_gpcov / chisq_red_triv)
	logging.info("5a generalized R^2: 1a: %s, 1b: %s",
			gen_rsq1a, gen_rsq1b)
	logging.info("5b generalized R^2: 2a: %s, 2b: %s",
			gen_rsq2a, gen_rsq2b)
	logging.info("5c generalized R^2: 3a: %s, 3b: %s",
			gen_rsq3a, gen_rsq3b)
	try:
		# celerite
		logdet = model.solver.log_determinant()
	except TypeError:
		# george
		logdet = model.solver.log_determinant
	logging.debug("5 logdet: %s, const 2: %s", logdet, _const)
