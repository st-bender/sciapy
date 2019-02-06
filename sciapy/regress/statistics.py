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

__all__ = ["mcmc_statistics"]


def _log_prob(resid, var):
	return -0.5 * (np.log(2 * np.pi * var) + resid**2 / var)


def mcmc_statistics(model, times, data, errs,
		train_times, train_data, train_errs,
		samples, lnp,
		median=False):
	"""Statistics for the (GP) model against the provided data

	Statistical information about the model and the sampled parameter
	distributions with respect to the provided data and its variance.
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
	test_logpred = -0.5 * (resid_gp.dot(np.linalg.solve(gpcov, resid_gp))
			+ np.trace(np.log(gpcov))
			+ _const)
	# MSLL
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
	chisq_gpcov = resid_mod.dot(np.linalg.solve(gpcov, resid_mod))
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
