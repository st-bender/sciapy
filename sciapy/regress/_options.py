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
"""SCIAMACHY regression tool command line options

Command line options for the command line tool.
"""

import argparse
from distutils.util import strtobool

from ._gpkernels import george_kernels, celerite_terms

__all__ = ["parser"]


class Range(object):
	"""Ranges for floats in command line arguments

	Helps to properly notify the user if the command line argument
	is out of range[1].

	[1](https://stackoverflow.com/questions/12116685/)
	"""
	def __init__(self, start, end):
		self.start = start
		self.end = end

	def __eq__(self, other):
		return self.start <= other <= self.end

	def __getitem__(self, index):
		if index == 0:
			return self
		else:
			raise IndexError()

	def __repr__(self):
		return '{0}--{1}'.format(self.start, self.end)


parser = argparse.ArgumentParser(description="SCIAMACHY data regression",
		prog="scia_regress")
parser.add_argument("file", default="SCIA_NO.nc",
		help="The filename of the input netcdf file")
parser.add_argument("-o", "--output_path", default=".",
		metavar="PATH",
		help="The directory for the output files (figures and MCMC samples) "
		"(default: %(default)s)")
parser.add_argument("-m", "--name_suffix", default="",
		help="The suffix for the figure plot files (default: \"\")")
parser.add_argument("--proxies", metavar="NAME1:FILE1,NAME2:FILE2,...",
		default="Sol:data/indices/lisird_lya_version3_daily_full.dat,"
		"GM:data/indices/AE_Kyoto_1980-2018_daily2_shift12h.dat",
		help="Comma separated list of (solar and geomagnetic or other) "
		"proxies as 'name:file' (default: %(default)s)")
parser.add_argument("-T", "--fit_lags", default="", type=str,
		metavar="NAME1,NAME2",
		help="Fit the proxy lag time "
		"(comma separated proxy names, e.g. Sol,GM) "
		"(default: %(default)s)")
parser.add_argument("-I", "--fit_lifetimes", default="", type=str,
		metavar="NAME1,NAME2",
		help="Fit the proxy life time "
		"(comma separated proxy names, e.g. Sol,GM), "
		"sets the proxy lag time to zero (default: %(default)s)")
parser.add_argument("--fit_annlifetimes", default="", type=str,
		metavar="NAME1,NAME2",
		help="Fit the proxy annual life time variations "
		"(comma separated proxy names, e.g. Sol,GM) "
		"(default: %(default)s)")
parser.add_argument("--fit_phase", action="store_true", default=False,
		help="Fit the phase of the harmonic terms directly "
		"instead of using separate cosine and sine terms "
		"(default: %(default)s)")
parser.add_argument("--use_sza", action="store_true", default=False,
		help="Fit the proxy annual life time variations "
		"using the (cosine and sine of the) of the solar zenith angle "
		"instead of the time (default: %(default)s)")
parser.add_argument("-t", "--lag_times",
		default="Sol:0,GM:0", type=str,
		metavar="NAME1:LAG1,NAME2:LAG2",
		help="Comma-separated list of name:value pairs of fixed proxy lags "
		"(in fractional years) (default: %(default)s)")
parser.add_argument("--center_proxies", default="", type=str,
		metavar="NAME1,NAME2",
		help="Comma-separated list of proxies to center "
		"by subtracting the mean (default: %(default)s)")
parser.add_argument("--log_proxies", default="", type=str,
		metavar="NAME1,NAME2",
		help="Comma-separated list of proxies to take the logarithm of "
		"for fitting (default: %(default)s)")
parser.add_argument("--positive_proxies", default="", type=str,
		metavar="NAME1,NAME2",
		help="Comma-separated list of proxies with positive cofficients. "
		"Changes the parameter bounds for these proxies accordingly "
		"(default: %(default)s)")
parser.add_argument("--norm_proxies_distSEsq", default="", type=str,
		metavar="NAME1,NAME2",
		help="Comma-separated list of proxies to be normalized by the "
		"Sun-Earth distance squared, for example the Lyman-alpha radiation "
		"(default: %(default)s)")
parser.add_argument("--norm_proxies_SZA", default="", type=str,
		metavar="NAME1,NAME2",
		help="Comma-separated list of proxies to be normalized by the "
		"the solar zenith angle, for example to adjust the Lyman-alpha "
		"radiation for the seasonal effects at different latitudes "
		"(default: %(default)s)")
parser.add_argument("--time_format", default="jyear", type=str,
		choices=['jyear', 'decimalyear', 'jd', 'mjd'],
		help="Treat the time units (proxy and data) according to the given "
		"astropy.time time format. (default: %(default)s)")
parser.add_argument("-k", "--fit_offset", action="store_true", default=False,
		help="Fit an additional offset via regression (default: %(default)s)")
parser.add_argument("-F", "--freqs", default="1, 2", type=str,
		metavar="FREQ1,FREQ2",
		help="Comma separated list of frequencies (in inverse years) to fit "
		"(default: %(default)s)")
parser.add_argument("--lifetime_scan", default=0, type=int,
		help="Number of days to go back to estimate the lifetime. "
		"If set to zero or negative, the scan range will be set to "
		"three times the maximum lifetime, including the annual variation "
		"(default: %(default)s)")
parser.add_argument("--lifetime_prior", default=None, type=str,
		choices=[None, 'flat', 'exp', 'normal'],
		help="The prior probability density for the lifetimes "
		"(default: %(default)s)")
parser.add_argument("--lifetime_metric", default=1, type=float,
		help="The prior probability density metric for the lifetimes in days "
		"(default: %(default)s)")
parser.add_argument("--center_data", action="store_true", default=False,
		help="Center the data by subtracting a global mean (default: %(default)s)")
parser.add_argument("--initial", metavar="values", default=None, type=str,
		help="Comma separated list of initial parameter values "
		"(default: %(default)s)")
parser.add_argument("-i", "--linearise", action="store_true", default=False,
		help="Use the linearised version of the model (default: %(default)s).")
parser.add_argument("-A", "--altitude", metavar="km",
		type=float, default=72,
		help="Altitude bin [km] (default: %(default)s)")
parser.add_argument("-L", "--latitude", metavar="degN",
		type=float, default=62.5,
		help="Latitude bin [°N] (default: %(default)s)")
parser.add_argument("--season", default=None,
		choices=[None, 'summerNH', 'summerSH'],
		help="Select a particular season (default: %(default)s)")
parser.add_argument("--exclude_spe", action="store_true", default=False,
		help="Exclude pre-defined SPE events (default: %(default)s)")
parser.add_argument("--akd_threshold", default=0.002, type=float,
		metavar="VALUE",
		help="Exclude data with an averaging kernel diagonal element "
		"smaller than the given threshold (default: %(default)s)")
parser.add_argument("--cnt_threshold", default=0, type=int,
		metavar="VALUE",
		help="Exclude data with less than the given number of measurement points "
		"in the averaged bin (default: %(default)s)")
parser.add_argument("-s", "--scale", metavar="factor",
		type=float, default=1e-6,
		help="Scale the data by factor prior to fitting (default: %(default)s)")
parser.add_argument("-r", "--random_subsample", metavar="factor",
		type=int, default=1,
		help="Randomly subsample the data by the given factor "
		"(default: 1, no subsampling)")
parser.add_argument("--train_fraction", metavar="factor",
		type=float, default=1, choices=Range(0., 1.),
		help="Use the given fraction of the data points to train the model "
		"(default: 1, train on all points)")
parser.add_argument("--test_fraction", metavar="factor",
		type=float, default=1, choices=Range(0., 1.),
		help="Use the given fraction of the data points to test the model "
		"(default: test on (1 - train_fraction) or all points)")
parser.add_argument("--random_train_test", dest="random_train_test",
		action="store_true")
parser.add_argument("--no-random_train_test", dest="random_train_test",
		action="store_false",
		help="Randomize the data before splitting into train and test sets "
		"(default: %(default)s).")
parser.set_defaults(random_train_test=False)
parser.add_argument("--scheduler_address", metavar="address:port",
		default=None,
		help="Connect to dask scheduler at address:port "
		"(default: %(default)s)")
parser.add_argument("--scheduler_file", metavar="file",
		default=None,
		help="Connect to dask scheduler at using the scheduler file "
		"(default: %(default)s)")
parser.add_argument("-O", "--optimize", type=int, default="1",
		choices=range(5),
		help="Optimize the parameters before MCMC run with method no.: "
		"0: no optimization, 1: Powell, "
		"2: differential evolution with latin hypercube initialization, "
		"3: basin hopping (experimental), and "
		"4: least squares curve fitting (experimental) "
		"(default: %(default)s)")
parser.add_argument("-N", "--openblas_threads", metavar="N",
		type=int, default=1,
		help="Use N OpenBlas threads (default: %(default)s)")
parser.add_argument("-S", "--random_seed", metavar="VALUE",
		type=int, default=None,
		help="Use a particular random seed to obtain "
		"reproducible results (default: %(default)s)")
group_mcmc = parser.add_argument_group(title="MCMC parameters",
		description="Fine-tuning of the (optional) MCMC run.")
group_mcmc.add_argument("-M", "--mcmc", type=strtobool, default="true",
		help="Fit the parameters with MCMC (default: %(default)s)")
group_mcmc.add_argument("-w", "--walkers", metavar="N",
		type=int, default=100,
		help="Use N MCMC walkers (default: %(default)s)")
group_mcmc.add_argument("-b", "--burn_in", metavar="N",
		type=int, default=200,
		help="Use N MCMC burn-in samples "
		"(run twice if --optimize is False) (default: %(default)s)")
group_mcmc.add_argument("-p", "--production", metavar="N",
		type=int, default=800,
		help="Use N MCMC production samples (default: %(default)s)")
group_mcmc.add_argument("-n", "--threads", metavar="N",
		type=int, default=1,
		help="Use N MCMC threads (default: %(default)s)")
group_mcmc.add_argument("-P", "--progress", action="store_true", default=False,
		help="Show MCMC sampler progress (default: %(default)s)")
group_gp = parser.add_argument_group(title="GP parameters",
		description="Fine-tuning of the (optional) Gaussian Process parameters.")
group_gp.add_argument("-g", "--george", action="store_true", default=False,
		help="Optimize a Gaussian Process model of the correlations "
		"using the `celerite` (not set) or `george` GP packages "
		"(default: %(default)s)")
group_gp.add_argument("-K", "--kernels", default="", type=str,
		help="Comma separated list of Gaussian Process kernels to use. "
		"They will be combined linearly (default: %(default)s) "
		"Possible choices are: {0} for george (-g) and {1} for celerite"
		.format(sorted(map(str, george_kernels.keys())),
			sorted(map(str, celerite_terms.keys()))))
group_gp.add_argument("-B", "--fit_bias", action="store_true", default=False,
		help="Fit bias using a constant kernel (default: %(default)s)")
group_gp.add_argument("-W", "--fit_white", action="store_true", default=False,
		help="Fit additional white noise (default: %(default)s)")
group_gp.add_argument("-H", "--HODLR_Solver", action="store_true", default=False,
		help="Use the HODLR solver for the GP fit (default: %(default)s)")
group_save = parser.add_argument_group(title="Output options",
		description="Diagnostic output and figures.")
group_save.add_argument("--save_model", dest="save_model", action="store_true")
group_save.add_argument("--no-save_model", dest="save_model", action="store_false",
		help="Saves a pickled version of the Model (default: %(default)s).")
group_save.add_argument("--save_samples", dest="save_samples", action="store_true")
group_save.add_argument("--no-save_samples", dest="save_samples", action="store_false",
		help="Saves the MCMC samples to disk (see --sample_format) "
		"(default: %(default)s).")
group_save.add_argument("--samples_format", default="netcdf4",
		choices=['npz', 'h5', 'hdf5', 'nc', 'netcdf4'],
		help="File format for the samples, compressed .npz or netcdf4 (hdf5) "
		"(h5 and hdf5 will also save to netcdf4 files but named \".h5\") "
		"(default: %(default)s).")
group_save.add_argument("--plot_corner", dest="plot_corner", action="store_true")
group_save.add_argument("--no-plot_corner", dest="plot_corner", action="store_false",
		help="Plot the fitted parameter distributions as a corner plot "
		"(default: %(default)s).")
group_save.add_argument("--plot_samples", dest="plot_samples", action="store_true")
group_save.add_argument("--no-plot_samples", dest="plot_samples", action="store_false",
		help="Plot sample predictions using the fitted parameters "
		"(default: %(default)s).")
group_save.add_argument("--plot_median", dest="plot_median", action="store_true")
group_save.add_argument("--no-plot_median", dest="plot_median", action="store_false",
		help="Plot median prediction and the residuals combined "
		"(default: %(default)s).")
group_save.add_argument("--plot_residuals", dest="plot_residuals", action="store_true")
group_save.add_argument("--no-plot_residuals", dest="plot_residuals", action="store_false",
		help="Plot standalone median prediction residuals "
		"(default: %(default)s).")
group_save.add_argument("--plot_maxlnp", dest="plot_maxlnp", action="store_true")
group_save.add_argument("--no-plot_maxlnp", dest="plot_maxlnp", action="store_false",
		help="Plot the maximum posterior prediction and the residuals combined "
		"(default: %(default)s).")
group_save.add_argument("--plot_maxlnpres", dest="plot_maxlnpres", action="store_true")
group_save.add_argument("--no-plot_maxlnpres", dest="plot_maxlnpres", action="store_false",
		help="Plot standalone maximum posterior prediction residuals "
		"(default: %(default)s).")
group_save.set_defaults(save_model=False, save_samples="netcdf4",
		plot_corner=True, plot_samples=True, plot_median=False,
		plot_residuals=False, plot_maxlnp=True, plot_maxlnpres=False)
loglevels = parser.add_mutually_exclusive_group()
loglevels.add_argument("-q", "--quiet", action="store_true", default=False,
		help="less output, same as --loglevel=ERROR (default: %(default)s)")
loglevels.add_argument("-v", "--verbose", action="store_true", default=False,
		help="verbose output, same as --loglevel=INFO (default: %(default)s)")
loglevels.add_argument("-l", "--loglevel", default="WARNING",
		choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
		help="change the loglevel (default: %(default)s)")
