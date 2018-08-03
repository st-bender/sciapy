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
"""SCIAMACHY regression tool plotting helpers

Plot (helper) functions for the regression command line tool.
"""

import logging

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_single_sample_and_residuals(model, times, data, errs,
		sample, filename):
	"""Plot one sample and residuals in one figure

	Parameters
	----------
	model: `celerite.Model`, `george.Model`, or `celerite.ModelSet`
		The Gaussian Process model to plot samples from.
	times: array_like
		Time axis values.
	data: array_like
		Data values.
	errs: array_like
		Data uncertainties for the errorbars.
	sample: array_like
		The (MCMC) sample of the parameter vector.
	filename: string
		Output filename of the figure file.
	"""
	# Set up the GP for this sample.
	logging.debug("sample values: %s", sample)
	model.set_parameter_vector(sample)
	log_lh = model.log_likelihood(data)
	logging.debug("sample log likelihood: %s", log_lh)
	logging.debug("half data variance: %s", 0.5 * np.var(data))
	t = np.sort(np.append(times,
			np.linspace(times.min(), times.max(), 1000)))
	# Plot
	fig = plt.figure()
	gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
	ax = plt.subplot(gs[0])
	ax2 = plt.subplot(gs[1])
	# Plot the data.
	ax.errorbar(times, data, yerr=2 * errs, fmt=".k", zorder=8)
	# Compute the prediction conditioned on the observations and plot it.
	mum = model.predict(data, t=times, return_cov=False)
	mup = model.mean.get_value(t)
	mum2 = model.mean.get_value(times)
	mu, cov = model.predict(data, t=t)
	try:
		cov[np.diag_indices_from(cov)] += model.kernel.jitter
	except AttributeError:
		pass
	std = np.sqrt(np.diagonal(cov))
	ax.plot(t, mup, alpha=0.75, color="C1", zorder=12,
			label="mean_model")
	ax.plot(t, mu, alpha=0.75, color="C0", zorder=10,
			label="logl: {0:.3f}".format(log_lh))
	ax.fill_between(t, mu - 2. * std, mu + 2. * std, color="C0", alpha=0.3, zorder=10)
	ax2.errorbar(times, data - mum, yerr=2 * errs, fmt=".k", zorder=12, alpha=0.5)
	ax2.errorbar(times, data - mum2, yerr=2 * errs, fmt=".C1", zorder=8, ms=4)
	ax2.axhline(y=0, color='k', alpha=0.5)
	ax.legend()
	fig.savefig(filename, transparent=True)


def plot_residual(model, times, data, errs, sample, scale, filename):
	"""Plot the residuals of one sample

	Parameters
	----------
	model: `celerite.Model`, `george.Model`, or `celerite.ModelSet`
		The Gaussian Process model to plot samples from.
	times: array_like
		Time axis values.
	data: array_like
		Data values.
	errs: array_like
		Data uncertainties for the errorbars.
	sample: array_like
		The (MCMC) sample of the parameter vector.
	scale: float
		The scale factor of the data to adjust the value axis.
	filename: string
		Output filename of the figure file.
	"""
	# Set up the GP for this sample.
	logging.debug("sample values: %s", sample)
	model.set_parameter_vector(sample)
	logging.debug("sample log likelihood: %s", model.log_likelihood(data))
	# Plot
	fig, ax = plt.subplots()
	# Plot the data.
	# Compute the prediction conditioned on the observations
	mu = model.predict(data, t=times, return_cov=False)
	# Plot the residuals with error bars
	ax.errorbar(times, data - mu, yerr=2 * errs, fmt=".k", zorder=8)
	ax.axhline(y=0, color='k', alpha=0.5)
	ax.set_xlabel("time [years]")
	ax.set_ylabel("number density residual [10$^{{{0:.0f}}}$ cm$^{{-3}}$]"
				.format(-np.log10(scale)))
	fig.savefig(filename, transparent=True)


def plot_single_sample(model, times, data, errs, sample, filename):
	"""Plot a sample and data

	Parameters
	----------
	model: `celerite.Model`, `george.Model`, or `celerite.ModelSet`
		The Gaussian Process model to plot samples from.
	times: array_like
		Time axis values.
	data: array_like
		Data values.
	errs: array_like
		Data uncertainties for the errorbars.
	sample: array_like
		The (MCMC) sample of the parameter vector.
	filename: string
		Output filename of the figure file.
	"""
	# Set up the GP for this sample.
	model.set_parameter_vector(sample)
	t = np.linspace(times.min(), times.max(), 1000)
	# Plot
	fig, ax = plt.subplots()
	# Plot the data.
	ax.errorbar(times, data, yerr=2 * errs, fmt="o", ms=4, zorder=4)
	# Compute the prediction conditioned on the observations and plot it.
	mu, cov = model.predict(data, t=t)
	_sample = np.random.multivariate_normal(mu, cov, 1)[0]
	ax.plot(t, _sample, alpha=0.75, zorder=2)
	fig.savefig(filename, transparent=True)


def plot_random_samples(model, times, data, errs,
		samples, scale, filename, size=8,
		extra_years=[0, 0]):
	"""Plot random samples and data

	Parameters
	----------
	model: `celerite.Model`, `george.Model`, or `celerite.ModelSet`
		The Gaussian Process model to plot samples from.
	times: array_like
		Time axis values.
	data: array_like
		Data values.
	errs: array_like
		Data uncertainties for the errorbars.
	samples: array_like
		The (MCMC) sample of the parameter vector.
	scale: float
		The scale factor of the data to adjust the value axis.
	filename: string
		Output filename of the figure file.
	size: int, optional
		Number of samples to plot.
	extra_years: list or tuple, optional
		Extend the prediction period extra_years[0] into the past
		and extra_years[1] into the future.
	"""
	t = np.linspace(times.min() - extra_years[0],
					times.max() + extra_years[1], 2000)
	fig, ax = plt.subplots()
	# Plot the data.
	ax.errorbar(times, data, yerr=2 * errs, fmt=".k", zorder=8)
	ax.set_xlim(np.min(t), np.max(t))
	ax.set_xlabel("time [years]")
	ax.set_ylabel("number density [10$^{{{0:.0f}}}$ cm$^{{-3}}$]"
				.format(-np.log10(scale)))
	for i in np.random.randint(len(samples), size=size):
		logging.info("plotting random sample %s.", i)
		logging.debug("sample values: %s", samples[i])
		model.set_parameter_vector(samples[i])
		logging.debug("sample log likelihood: %s", model.log_likelihood(data))
		# Compute the prediction conditioned on the observations and plot it.
		mu, cov = model.predict(data, t=t)
		try:
			cov[np.diag_indices_from(cov)] += model.kernel.jitter
		except AttributeError:
			pass
		sample = np.random.multivariate_normal(mu, cov, 1)[0]
		ax.plot(t, sample, color="C0", alpha=0.25, zorder=12)
	# fig.subplots_adjust(left=0.2, bottom=0.2, top=0.9, right=0.9)
	fig.savefig(filename, transparent=True)
