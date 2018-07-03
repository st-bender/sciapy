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
		"celerite_terms", "celerite_terms_freeze_params"]

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
				log_sigma=-6,
				log_rho=2. * np.log(0.5),
				bounds={"log_sigma": [-30, 30],
						"log_rho": [-30, 30]}),
	"SHO0": terms.SHOTerm(log_S0=-6, log_Q=1.0 / np.sqrt(2.), log_omega0=0.,
				bounds={"log_S0": [-30, 30],
						"log_Q": [-30, 30],
						"log_omega0": [-30, 30]}),
	"SHO1": terms.SHOTerm(log_S0=-6, log_Q=1.0 / 2., log_omega0=0.,
				bounds={"log_S0": [-10, 10],
						"log_omega0": [-10, 10]}),
	"SHO2": terms.SHOTerm(log_S0=-6, log_Q=-2.24, log_omega0=-0.6,
				bounds={"log_S0": [-30, 30],
						"log_Q": [-30, 30],
						"log_omega0": [-30, 30]}),
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
celerite_terms_freeze_params = {
	"N": [],
	"B": ["log_c"],
	"W": [],
	"Mat32": [],
	"SHO0": ["log_Q", "log_omega0"],
	"SHO1": ["log_Q"],
	"SHO2": [],
	"SHO3": ["k2:log_S0", "k2:log_Q"],
}
