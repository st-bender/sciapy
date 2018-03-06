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
		"celerite_terms", "celerite_terms_freeze_params",
		"celerite_terms_params_bounds"]

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
	"B": terms.RealTerm(log_a=-6., log_c=-np.inf),
	"W": terms.JitterTerm(log_sigma=-25),
	"Mat32": terms.Matern32Term(
				log_sigma=-6,
				log_rho=2. * np.log(0.5)),
	"SHO": terms.SHOTerm(-6, 1.0 / np.sqrt(2.), 0.),
	"SHO2": terms.SHOTerm(-6, -2.24, -0.6)
}
celerite_terms_freeze_params = {
	"N": [],
	"B": ["log_c"],
	"W": [],
	"Mat32": [],
	"SHO": ["log_Q", "log_omega0"],
	"SHO2": []
}
celerite_terms_params_bounds = {
	"N": [],
	"B": [[-30, 30]],
	"W": [[-30, 30]],
	"Mat32": [[-30, 30], [-30, 30]],
	"SHO": [[-30, 30]],
	"SHO2": [[-30, 30], [-30, 30], [-30, 30]]
}
