# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2018 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY regression module tests
"""

import sciapy.regress


def test_module_structure():
	assert sciapy.regress
	assert sciapy.regress.load_data
	assert sciapy.regress.mcmc


def test_loaddatamodule_method_structure():
	assert sciapy.regress.load_scia_dzm
	assert sciapy.regress.load_solar_gm_table
	assert sciapy.regress.load_data.load_scia_dzm
	assert sciapy.regress.load_data.load_solar_gm_table
	assert sciapy.regress.load_data.load_dailymeanAE
	assert sciapy.regress.load_data.load_dailymeanLya


def test_mcmcmodule_method_structure():
	assert sciapy.regress.mcmc_sample_model
	assert sciapy.regress.mcmc.mcmc_sample_model
