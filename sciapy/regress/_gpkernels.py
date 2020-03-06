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
"""SCIAMACHY regression tool Gaussian process kernel options

Gaussian process kernel standard parameters for the command line tool.
"""

import numpy as np

import george
from george import kernels

from celerite import terms

__all__ = ["george_kernels", "george_solvers",
		"celerite_terms",
		"setup_george_kernel", "setup_celerite_terms"]

george_kernels = {
	"Exp2": kernels.ExpSquaredKernel(10**2),
	"Exp2ESin2": (kernels.ExpSquaredKernel(10**2) *
				kernels.ExpSine2Kernel(2 / 1.3**2, 1.0)),
	"ESin2": kernels.ExpSine2Kernel(2 / 1.3**2, 1.0),
	"27dESin2": kernels.ExpSine2Kernel(2 / 1.3**2, 27.0 / 365.25),
	"RatQ": kernels.RationalQuadraticKernel(0.8, 0.1**2),
	"Mat32": kernels.Matern32Kernel((0.5)**2),
	"Exp": kernels.ExpKernel((0.5)**2),
	# "W": kernels.WhiteKernel,  # deprecated, delegated to `white_noise`
	"B": kernels.ConstantKernel,
}
george_solvers = {
	"basic": george.BasicSolver,
	"HODLR": george.HODLRSolver,
}

celerite_terms = {
	"N": terms.Term(),
	"B": terms.RealTerm(log_a=-6., log_c=-np.inf,
				bounds={"log_a": [-30, 30],
						"log_c": [-np.inf, np.inf]}),
	"W": terms.JitterTerm(log_sigma=-25,
				bounds={"log_sigma": [-30, 30]}),
	"Mat32": terms.Matern32Term(
				log_sigma=1.,
				log_rho=1.,
				bounds={"log_sigma": [-30, 30],
						# The `celerite` version of the Matern-3/2
						# kernel has problems with very large `log_rho`
						# values. -7.4 is empirical.
						"log_rho": [-7.4, 16]}),
	"SHO0": terms.SHOTerm(log_S0=-6, log_Q=1.0 / np.sqrt(2.), log_omega0=0.,
				bounds={"log_S0": [-30, 30],
						"log_Q": [-30, 30],
						"log_omega0": [-30, 30]}),
	"SHO1": terms.SHOTerm(log_S0=-6, log_Q=-2., log_omega0=0.,
				bounds={"log_S0": [-10, 10],
						"log_omega0": [-10, 10]}),
	"SHO2": terms.SHOTerm(log_S0=-6, log_Q=0.5, log_omega0=0.,
				bounds={"log_S0": [-10, 10],
						"log_Q": [-10, 10],
						"log_omega0": [-10, 10]}),
	# see Foreman-Mackey et al. 2017, AJ 154, 6, pp. 220
	# doi: 10.3847/1538-3881/aa9332
	# Eq. (53)
	"SHO3": terms.SHOTerm(log_S0=-6, log_Q=0., log_omega0=0.,
				bounds={"log_S0": [-15, 5],
						"log_Q": [-10, 10],
						"log_omega0": [-10, 10]}) *
			terms.SHOTerm(log_S0=1, log_Q=1.0 / np.sqrt(2.), log_omega0=0.,
				bounds={"log_omega0": [-5, 5]}),
}
_celerite_terms_freeze_params = {
	"N": [],
	"B": ["log_c"],
	"W": [],
	"Mat32": [],
	"SHO0": ["log_Q", "log_omega0"],
	"SHO1": ["log_Q"],
	"SHO2": [],
	"SHO3": ["k2:log_S0", "k2:log_Q"],
}


def setup_george_kernel(kernelnames, kernel_base=1, fit_bias=False):
	"""Setup the Gaussian Process kernel for george

	Parameters
	----------
	kernelnames : list of str
		List of abbreviated names for the kernels, choices:
		'Exp2' for a `ExpSquaredKernel`, 'ESin2' for a `ExpSine2Kernel`,
		'Exp2ESin2' for a `ExpSquaredKernel` multiplied by a `ExpSine2Kernel`,
		'RatQ' for a `RationalQuadraticKernel`, 'Mat32' for a `Matern32Kernel`,
		'Exp' for `ExpKernel`, and 'B' for a `ConstantKernel` (bias).
	kernel_base : float, optional
		The initial "strength" of the kernels.
	fit_bias : bool, optional
		Adds a `ConstantKernel` if kernel does not already contain one.

	Returns
	-------
	name : The kernel names concatenated by an underscore and prepended by '_gp'
	kernel : The covariance kernel for use with george.GP
	"""
	kernel = None
	kname = "_gp"
	for kn in kernelnames:
		krn = george_kernels.get(kn, None)
		if krn is None:
			# not found in the list of available kernels
			continue
		if kn in ["B", "W"]:
			# don't scale the constant or white kernels
			krnl = krn(0.25 * kernel_base)
		else:
			krnl = kernel_base * krn
		kernel = kernel + krnl if hasattr(kernel, "is_kernel") else krnl
		kname += "_" + kn

	if fit_bias and "B" not in kernelnames:
		krnl = kernels.ConstantKernel(kernel_base)
		kernel = kernel + krnl if hasattr(kernel, "is_kernel") else krnl
		kname += "_b"

	return kname, kernel


def setup_celerite_terms(termnames, fit_bias=False, fit_white=False):
	"""Setup the Gaussian Process terms for celerite

	Parameters
	----------
	termnames : list of str
		List of abbreviated names for the `celerite.terms`, choices:
		'N' for an empty `Term`, 'B' for a constant term (bias),
		'W' for a `JitterTerm` (white noise), 'Mat32' for a `Matern32Term`,
		'SHO0'...'SHO3' for `SHOTerm`s (harmonic oscillators) with different
		frozen parameters.
	fit_bias : bool, optional
		Adds a constant term (`RealTerm` with log_c fixed to -np.inf) if the
		terms do not already contain one.
	fit_white : bool, optional
		Adds a `JitterTerm` if not already included.

	Returns
	-------
	name : The term names concatenated by an underscore and prepended by '_cel'
	terms : The covariance terms for use with celerite.GP
	"""
	# celerite terms
	cel_terms = None
	cel_name = "_cel"
	for tn in termnames:
		trm = celerite_terms.get(tn, None)
		if trm is None:
			# not found in the list of available terms
			continue
		for freeze_p in _celerite_terms_freeze_params[tn]:
			trm.freeze_parameter(freeze_p)
		cel_terms = cel_terms + trm if cel_terms is not None else trm
		cel_name += "_" + tn

	# avoid double initialisation of White and Constant kernels
	# if already set above
	if fit_white and "W" not in termnames:
		trm = celerite_terms["W"]
		cel_terms = cel_terms + trm if cel_terms is not None else trm
		cel_name += "_w"
	if fit_bias and "B" not in termnames:
		trm = celerite_terms["B"]
		trm.freeze_parameter("log_c")
		cel_terms = cel_terms + trm if cel_terms is not None else trm
		cel_name += "_b"

	if cel_terms is None:
		cel_terms = terms.Term()

	return cel_name, cel_terms
