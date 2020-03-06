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
"""SCIAMACHY data regression command line interface

Command line main program for regression analysis of SCIAMACHY
daily zonal mean time series (NO for now).
"""

import ctypes
import logging
from os import path

import numpy as np
import scipy.optimize as op
from scipy.interpolate import interp1d

import george
import celerite

import matplotlib as mpl
# switch off X11 rendering
mpl.use("Agg")

from .load_data import load_solar_gm_table, load_scia_dzm
from .models_cel import trace_gas_model
from .mcmc import mcmc_sample_model
from .statistics import mcmc_statistics

from ._gpkernels import (george_solvers,
		setup_george_kernel, setup_celerite_terms)
from ._plot import (plot_single_sample_and_residuals,
		plot_residual, plot_random_samples)
from ._options import parser


def save_samples_netcdf(filename, model, alt, lat, samples,
		scale=1e-6,
		lnpost=None, compressed=False):
	from xarray import Dataset
	smpl_ds = Dataset(dict([(pname, (["lat", "alt", "sample"],
				samples[..., i].reshape(1, 1, -1)))
				for i, pname in enumerate(model.get_parameter_names())]
				# + [("lpost", (["lat", "alt", "sample"], lnp.reshape(1, 1, -1)))]
			),
			coords={"lat": [lat], "alt": [alt]})

	for modn in model.mean.models:
		modl = model.mean.models[modn]
		if hasattr(modl, "mean"):
			smpl_ds.attrs[modn + ":mean"] = modl.mean

	units = {"kernel": {
				"log": "log(10$^{{{0:.0f}}}$ cm$^{{-3}}$)"
						.format(-np.log10(scale))},
			"mean": {
				"log": "log(10$^{{{0:.0f}}}$ cm$^{{-3}}$)"
						.format(-np.log10(scale)),
				"cos": "10$^{{{0:.0f}}}$ cm$^{{-3}}$".format(-np.log10(scale)),
				"sin": "10$^{{{0:.0f}}}$ cm$^{{-3}}$".format(-np.log10(scale)),
				"val": "10$^{{{0:.0f}}}$ cm$^{{-3}}$".format(-np.log10(scale)),
				"amp": "10$^{{{0:.0f}}}$ cm$^{{-3}}$".format(-np.log10(scale)),
				"tau": "d"}}
	for pname in smpl_ds.data_vars:
		_pp = pname.split(':')
		for _n, _u in units[_pp[0]].items():
			if _pp[-1].startswith(_n):
				logging.debug("units for %s: %s", pname, _u)
				smpl_ds[pname].attrs["units"] = _u

	smpl_ds["alt"].attrs = {"long_name": "altitude", "units": "km"}
	smpl_ds["lat"].attrs = {"long_name": "latitude", "units": "degrees_north"}

	_encoding = None
	if compressed:
		_encoding = {var: {"zlib": True, "complevel": 1}
					for var in smpl_ds.data_vars}
	try:
		smpl_ds.to_netcdf(filename, encoding=_encoding)
	except ValueError:
		smpl_ds.to_netcdf(filename)
	smpl_ds.close()


def _train_test_split(times, data, errs, train_frac,
		test_frac, randomize):
	# split the data into training and test subsets according to the
	# fraction given (default is 1, i.e. no splitting)
	ndata = len(times)
	train_size = int(ndata * train_frac)
	test_size = min(ndata - train_size, int(ndata * test_frac))
	# randomize if requested
	if randomize:
		permut_idx = np.random.permutation(np.arange(ndata))
	else:
		permut_idx = np.arange(ndata)
	train_idx = np.sort(permut_idx[:train_size])
	test_idx = np.sort(permut_idx[train_size:train_size + test_size])
	times_train = times[train_idx]
	data_train = data[train_idx]
	errs_train = errs[train_idx]
	if test_size > 0:
		times_test = times[test_idx]
		data_test = data[test_idx]
		errs_test = errs[test_idx]
	else:
		times_test = times
		data_test = data
		errs_test = errs
	logging.info("using %s of %s samples for training.", len(times_train), ndata)
	logging.info("using %s of %s samples for testing.", len(times_test), ndata)
	return (times_train, data_train, errs_train,
			times_test, data_test, errs_test)


def _r_sun_earth(time, tfmt="jyear"):
	"""First order approximation of the Sun-Earth distance

	The Sun-to-Earth distance can be used to (un-)normalize proxies
	to the actual distance to the Sun instead of 1 AU.

	Parameters
	----------
	time : float
		Time value in the units given by 'tfmt'.
	tfmt : str, optional
		The units of 'time' as supported by the
		astropy.time time formats. Default: 'jyear'.

	Returns
	-------
	dist : float
		The Sun-Earth distance at the given day of year in AU.
	"""
	from astropy.time import Time
	tdoy = Time(time, format=tfmt)
	tdoy.format = "yday"
	doy = int(tdoy.value.split(':')[1])
	return 1 - 0.01672 * np.cos(2 * np.pi / 365.256363 * (doy - 4))


def main():
	logging.basicConfig(level=logging.WARNING,
			format="[%(levelname)-8s] (%(asctime)s) "
				"%(filename)s:%(lineno)d %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S %z")

	args = parser.parse_args()

	logging.info("command line arguments: %s", args)
	if args.quiet:
		logging.getLogger().setLevel(logging.ERROR)
	elif args.verbose:
		logging.getLogger().setLevel(logging.INFO)
	else:
		logging.getLogger().setLevel(args.loglevel)

	from numpy.distutils.system_info import get_info
	try:
		ob_lib_dirs = get_info("openblas")["library_dirs"]
	except KeyError:
		ob_lib_dirs = []
	for oblas_path in ob_lib_dirs:
		oblas_name = "{0}/libopenblas.so".format(oblas_path)
		logging.info("Trying %s", oblas_name)
		try:
			oblas_lib = ctypes.cdll.LoadLibrary(oblas_name)
			oblas_cores = oblas_lib.openblas_get_num_threads()
			oblas_lib.openblas_set_num_threads(args.openblas_threads)
			logging.info("Using %s/%s Openblas thread(s).",
					oblas_lib.openblas_get_num_threads(), oblas_cores)
		except:
			logging.info("Setting number of openblas threads failed.")

	if args.random_seed is not None:
		np.random.seed(args.random_seed)

	if args.proxies:
		proxies = args.proxies.split(',')
		proxy_dict = dict(_p.split(':') for _p in proxies)
	else:
		proxy_dict = {}
	lag_dict = {pn: 0 for pn in proxy_dict.keys()}

	# Post-processing of arguments...
	# List of proxy lag fits from csv
	fit_lags = args.fit_lags.split(',')
	# List of proxy lifetime fits from csv
	fit_lifetimes = args.fit_lifetimes.split(',')
	fit_annlifetimes = args.fit_annlifetimes.split(',')
	# List of proxy lag times from csv
	lag_dict.update(dict(_ls.split(':') for _ls in args.lag_times.split(',')))
	# List of cycles (frequencies in 1/year) from argument list (csv)
	try:
		freqs = list(map(float, args.freqs.split(',')))
	except ValueError:
		freqs = []
	args.freqs = freqs
	# List of initial parameter values
	initial = None
	if args.initial is not None:
		try:
			initial = list(map(float, args.initial.split(',')))
		except ValueError:
			pass
	# List of GP kernels from argument list (csv)
	kernls = args.kernels.split(',')

	lat = args.latitude
	alt = args.altitude
	logging.info("location: %.0f°N %.0f km", lat, alt)

	no_ys, no_dens, no_errs, no_szas = load_scia_dzm(args.file, alt, lat,
			tfmt=args.time_format,
			scale=args.scale,
			#subsample_factor=args.random_subsample,
			#subsample_method="random",
			akd_threshold=args.akd_threshold,
			cnt_threshold=args.cnt_threshold,
			center=args.center_data,
			season=args.season,
			SPEs=args.exclude_spe)

	(no_ys_train, no_dens_train, no_errs_train,
		no_ys_test, no_dens_test, no_errs_test) = _train_test_split(
				no_ys, no_dens, no_errs, args.train_fraction,
				args.test_fraction, args.random_train_test)

	sza_intp = interp1d(no_ys, no_szas, bounds_error=False)

	max_amp = 1e10 * args.scale
	max_days = 100

	proxy_config = {}
	for pn, pf in proxy_dict.items():
		pt, pp = load_solar_gm_table(path.expanduser(pf),
				cols=[0, 1], names=["time", pn], tfmt=args.time_format)
		pv = pp[pn]
		# use log of proxy values if desired
		if pn in args.log_proxies.split(','):
			pv = np.log(pv)
		# normalize to sun--earth distance squared
		if pn in args.norm_proxies_distSEsq.split(','):
			rad_sun_earth = np.vectorize(_r_sun_earth)(pt, tfmt=args.time_format)
			pv /= rad_sun_earth**2
		# normalize by cos(SZA)
		if pn in args.norm_proxies_SZA.split(',') and sza_intp is not None:
			pv *= np.cos(np.radians(sza_intp(pt)))
		proxy_config.update({pn:
			dict(times=pt, values=pv,
				center=pn in args.center_proxies.split(','),
				positive=pn in args.positive_proxies.split(','),
				lag=float(lag_dict[pn]),
				max_amp=max_amp, max_days=max_days,
				sza_intp=sza_intp if args.use_sza else None,
			)}
		)

	model = trace_gas_model(constant=args.fit_offset,
			proxy_config=proxy_config, **vars(args))

	logging.debug("model dict: %s", model.get_parameter_dict())
	model.freeze_all_parameters()
	# thaw parameters according to requested fits
	for pn in proxy_dict.keys():
		model.thaw_parameter("{0}:amp".format(pn))
		if pn in fit_lags:
			model.thaw_parameter("{0}:lag".format(pn))
		if pn in fit_lifetimes:
			model.set_parameter("{0}:tau0".format(pn), 1e-3)
			model.thaw_parameter("{0}:tau0".format(pn))
			if pn in fit_annlifetimes:
				model.thaw_parameter("{0}:taucos1".format(pn))
				model.thaw_parameter("{0}:tausin1".format(pn))
		else:
			model.set_parameter("{0}:ltscan".format(pn), 0)
	for freq in freqs:
		if not args.fit_phase:
			model.thaw_parameter("f{0:.0f}:cos".format(freq))
			model.thaw_parameter("f{0:.0f}:sin".format(freq))
		else:
			model.thaw_parameter("f{0:.0f}:amp".format(freq))
			model.thaw_parameter("f{0:.0f}:phase".format(freq))
	if args.fit_offset:
		#model.set_parameter("offset:value", -100.)
		#model.set_parameter("offset:value", 0)
		model.thaw_parameter("offset:value")

	if initial is not None:
		model.set_parameter_vector(initial)
	# model.thaw_parameter("GM:ltscan")
	logging.debug("params: %s", model.get_parameter_dict())
	logging.debug("param names: %s", model.get_parameter_names())
	logging.debug("param vector: %s", model.get_parameter_vector())
	logging.debug("param bounds: %s", model.get_parameter_bounds())
	#logging.debug("model value: %s", model.get_value(no_ys))
	#logging.debug("default log likelihood: %s", model.log_likelihood(model.vector))

	# setup the Gaussian Process kernel
	kernel_base = (1e7 * args.scale)**2
	ksub = args.name_suffix

	solver = "basic"
	skwargs = {}
	if args.HODLR_Solver:
		solver = "HODLR"
		#skwargs = {"tol": 1e-3}

	if args.george:
		gpname, kernel = setup_george_kernel(kernls,
				kernel_base=kernel_base, fit_bias=args.fit_bias)
		gpmodel = george.GP(kernel, mean=model,
			white_noise=1.e-25, fit_white_noise=args.fit_white,
			solver=george_solvers[solver], **skwargs)
		# the george interface does not allow setting the bounds in
		# the kernel initialization so we prepare simple default bounds
		kernel_bounds = [(-0.3 * max_amp, 0.3 * max_amp)
				for _ in gpmodel.kernel.get_parameter_names()]
		bounds = gpmodel.mean.get_parameter_bounds() + kernel_bounds
	else:
		gpname, cel_terms = setup_celerite_terms(kernls,
				fit_bias=args.fit_bias, fit_white=args.fit_white)
		gpmodel = celerite.GP(cel_terms, mean=model,
			fit_white_noise=args.fit_white,
			fit_mean=True)
		bounds = gpmodel.get_parameter_bounds()
	gpmodel.compute(no_ys_train, no_errs_train)
	logging.debug("gpmodel params: %s", gpmodel.get_parameter_dict())
	logging.debug("gpmodel bounds: %s", bounds)
	logging.debug("initial log likelihood: %s", gpmodel.log_likelihood(no_dens_train))
	if isinstance(gpmodel, celerite.GP):
		logging.info("(GP) jitter: %s", gpmodel.kernel.jitter)
	model_name = "_".join(gpmodel.mean.get_parameter_names()).replace(':', '')
	gpmodel_name = model_name + gpname
	logging.info("GP model name: %s", gpmodel_name)

	pre_opt = False
	if args.optimize > 0:
		def gpmodel_mean(x, *p):
			gpmodel.set_parameter_vector(p)
			return gpmodel.mean.get_value(x)

		def gpmodel_res(x, *p):
			gpmodel.set_parameter_vector(p)
			return (gpmodel.mean.get_value(x) - no_dens_train) / no_errs_train

		def lpost(p, y, gp):
			gp.set_parameter_vector(p)
			return gp.log_likelihood(y, quiet=True) + gp.log_prior()

		def nlpost(p, y, gp):
			lp = lpost(p, y, gp)
			return -lp if np.isfinite(lp) else 1e25

		def grad_nlpost(p, y, gp):
			gp.set_parameter_vector(p)
			grad_ll = gp.grad_log_likelihood(y)
			if isinstance(grad_ll, tuple):
				# celerite
				return -grad_ll[1]
			# george
			return -grad_ll

		jacobian = grad_nlpost if gpmodel.kernel.vector_size else None
		if args.optimize == 1:
			resop_gp = op.minimize(
				nlpost,
				gpmodel.get_parameter_vector(),
				args=(no_dens_train, gpmodel),
				bounds=bounds,
				# method="l-bfgs-b", options=dict(disp=True, maxcor=100, eps=1e-9, ftol=2e-15, gtol=1e-8))
				method="l-bfgs-b", jac=jacobian)
				# method="tnc", options=dict(disp=True, maxiter=500, xtol=1e-12))
				# method="nelder-mead", options=dict(disp=True, maxfev=100000, fatol=1.49012e-8, xatol=1.49012e-8))
				# method="Powell", options=dict(ftol=1.49012e-08, xtol=1.49012e-08))
		elif args.optimize == 2:
			resop_gp = op.differential_evolution(
				nlpost,
				bounds=bounds,
				args=(no_dens_train, gpmodel),
				popsize=2 * args.walkers, tol=0.01)
		elif args.optimize == 3:
			resop_bh = op.basinhopping(
				nlpost,
				gpmodel.get_parameter_vector(),
				niter=200,
				minimizer_kwargs=dict(
					args=(no_dens_train, gpmodel),
					bounds=bounds,
					# method="tnc"))
					# method="l-bfgs-b", options=dict(maxcor=100)))
					method="l-bfgs-b", jac=jacobian))
					# method="Nelder-Mead"))
					# method="BFGS"))
					# method="Powell", options=dict(ftol=1.49012e-08, xtol=1.49012e-08)))
			logging.debug("optimization result: %s", resop_bh)
			resop_gp = resop_bh.lowest_optimization_result
		elif args.optimize == 4:
			resop, cov_gp = op.curve_fit(
				gpmodel_mean,
				no_ys_train, no_dens_train, gpmodel.get_parameter_vector(),
				bounds=tuple(np.array(bounds).T),
				# method='lm',
				# absolute_sigma=True,
				sigma=no_errs_train)
			resop_gp = op.OptimizeResult(dict(
				x=resop,
				success=True,
				message="Curve fit successful.",
			))
			logging.debug("curve fit %s, std %s:", resop, np.sqrt(np.diag(cov_gp)))
		else:
			logging.warn("unsupported optimization method: %s", args.optimize)
			resop_gp = op.OptimizeResult(dict(
				x=gpmodel.get_parameter_vector(),
				success=False,
				message="unsupported optimization method: {0}".format(args.optimize),
			))
		logging.info("%s", resop_gp.message)
		logging.debug("optimization result: %s", resop_gp)
		logging.info("gpmodel dict: %s", gpmodel.get_parameter_dict())
		logging.info("log posterior trained: %s", lpost(gpmodel.get_parameter_vector(), no_dens_train, gpmodel))
		gpmodel.compute(no_ys_test, no_errs_test)
		logging.info("log posterior test: %s", lpost(gpmodel.get_parameter_vector(), no_dens_test, gpmodel))
		gpmodel.compute(no_ys, no_errs)
		logging.info("log posterior all: %s", lpost(gpmodel.get_parameter_vector(), no_dens, gpmodel))
		# cross check to make sure that the gpmodel parameter vector is really
		# set to the fitted parameters
		logging.info("opt. model vector: %s", resop_gp.x)
		gpmodel.compute(no_ys_train, no_errs_train)
		logging.debug("opt. log posterior trained 1: %s", lpost(resop_gp.x, no_dens_train, gpmodel))
		gpmodel.compute(no_ys_test, no_errs_test)
		logging.debug("opt. log posterior test 1: %s", lpost(resop_gp.x, no_dens_test, gpmodel))
		gpmodel.compute(no_ys, no_errs)
		logging.debug("opt. log posterior all 1: %s", lpost(resop_gp.x, no_dens, gpmodel))
		logging.debug("opt. model vector: %s", gpmodel.get_parameter_vector())
		gpmodel.compute(no_ys_train, no_errs_train)
		logging.debug("opt. log posterior trained 2: %s", lpost(gpmodel.get_parameter_vector(), no_dens_train, gpmodel))
		gpmodel.compute(no_ys_test, no_errs_test)
		logging.debug("opt. log posterior test 2: %s", lpost(gpmodel.get_parameter_vector(), no_dens_test, gpmodel))
		gpmodel.compute(no_ys, no_errs)
		logging.debug("opt. log posterior all 2: %s", lpost(gpmodel.get_parameter_vector(), no_dens, gpmodel))
		pre_opt = resop_gp.success
	try:
		logging.info("GM lt: %s", gpmodel.get_parameter("mean:GM:tau0"))
	except ValueError:
		pass
	logging.info("(GP) model: %s", gpmodel.kernel)
	if isinstance(gpmodel, celerite.GP):
		logging.info("(GP) jitter: %s", gpmodel.kernel.jitter)

	bestfit = gpmodel.get_parameter_vector()
	filename_base = path.join(
		args.output_path,
		"NO_regress_fit_{0}_{1:.0f}_{2:.0f}_{{0}}_{3}"
		.format(gpmodel_name, lat * 10, alt, ksub),
	)

	if args.mcmc:
		gpmodel.compute(no_ys_train, no_errs_train)
		samples, lnp = mcmc_sample_model(gpmodel,
				no_dens_train,
				beta=1.0,
				nwalkers=args.walkers, nburnin=args.burn_in,
				nprod=args.production, nthreads=args.threads,
				show_progress=args.progress,
				optimized=pre_opt, bounds=bounds, return_logpost=True)

		if args.train_fraction < 1. or args.test_fraction < 1.:
			logging.info("Statistics for the test samples")
			mcmc_statistics(gpmodel,
					no_ys_test, no_dens_test, no_errs_test,
					no_ys_train, no_dens_train, no_errs_train,
					samples, lnp,
			)
		logging.info("Statistics for all samples")
		mcmc_statistics(gpmodel,
				no_ys, no_dens, no_errs,
				no_ys_train, no_dens_train, no_errs_train,
				samples, lnp,
		)

		sampl_percs = np.percentile(samples, [2.5, 50, 97.5], axis=0)
		if args.plot_corner:
			import corner
			# Corner plot of the sampled parameters
			fig = corner.corner(samples,
					quantiles=[0.025, 0.5, 0.975],
					show_titles=True,
					labels=gpmodel.get_parameter_names(),
					truths=bestfit,
					hist_args=dict(normed=True))
			fig.savefig(filename_base.format("corner") + ".pdf", transparent=True)

		if args.save_samples:
			if args.samples_format in ["npz"]:
				# save the samples compressed to save space.
				np.savez_compressed(filename_base.format("sampls") + ".npz",
					samples=samples)
			if args.samples_format in ["nc", "netcdf4"]:
				save_samples_netcdf(filename_base.format("sampls") + ".nc",
					gpmodel, alt, lat, samples, scale=args.scale, compressed=True)
			if args.samples_format in ["h5", "hdf5"]:
				save_samples_netcdf(filename_base.format("sampls") + ".h5",
					gpmodel, alt, lat, samples, scale=args.scale, compressed=True)
		# MCMC finished here

	# set the model times and errors to use the full data set for plotting
	gpmodel.compute(no_ys, no_errs)
	if args.save_model:
		try:
			# python 2
			import cPickle as pickle
		except ImportError:
			# python 3
			import pickle
		# pickle and save the model
		with open(filename_base.format("model") + ".pkl", "wb") as f:
			pickle.dump((gpmodel), f, -1)

	if args.plot_samples and args.mcmc:
		plot_random_samples(gpmodel, no_ys, no_dens, no_errs,
				samples, args.scale,
				filename_base.format("sampls") + ".pdf",
				size=4, extra_years=[4, 2])

	if args.plot_median:
		plot_single_sample_and_residuals(gpmodel, no_ys, no_dens, no_errs,
				sampl_percs[1],
				filename_base.format("median") + ".pdf")
	if args.plot_residuals:
		plot_residual(gpmodel, no_ys, no_dens, no_errs,
				sampl_percs[1], args.scale,
				filename_base.format("medres") + ".pdf")
	if args.plot_maxlnp:
		plot_single_sample_and_residuals(gpmodel, no_ys, no_dens, no_errs,
				samples[np.argmax(lnp)],
				filename_base.format("maxlnp") + ".pdf")
	if args.plot_maxlnpres:
		plot_residual(gpmodel, no_ys, no_dens, no_errs,
				samples[np.argmax(lnp)], args.scale,
				filename_base.format("mlpres") + ".pdf")

	labels = gpmodel.get_parameter_names()
	logging.info("param percentiles [2.5, 50, 97.5]:")
	for pc, label in zip(sampl_percs.T, labels):
		median = pc[1]
		pc_minus = median - pc[0]
		pc_plus = pc[2] - median
		logging.debug("%s: %s", label, pc)
		logging.info("%s: %.6f (- %.6f) (+ %.6f)", label,
				median, pc_minus, pc_plus)

	logging.info("Finished successfully.")


if __name__ == "__main__":
	main()
