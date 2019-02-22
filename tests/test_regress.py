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
	assert sciapy.regress.models_cel


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


def test_modelmodule_object_structure():
	assert sciapy.regress.TraceGasModelSet
	assert sciapy.regress.ConstantModel
	assert sciapy.regress.HarmonicModelAmpPhase
	assert sciapy.regress.HarmonicModelCosineSine
	assert sciapy.regress.ProxyModel
	assert sciapy.regress.models_cel.TraceGasModelSet
	assert sciapy.regress.models_cel.ConstantModel
	assert sciapy.regress.models_cel.HarmonicModelAmpPhase
	assert sciapy.regress.models_cel.HarmonicModelCosineSine
	assert sciapy.regress.models_cel.ProxyModel


def test_modelmodule_method_structure():
	assert sciapy.regress.TraceGasModelSet.get_value
	assert sciapy.regress.TraceGasModelSet.compute_gradient
	assert sciapy.regress.HarmonicModelAmpPhase.get_value
	assert sciapy.regress.HarmonicModelAmpPhase.get_amplitude
	assert sciapy.regress.HarmonicModelAmpPhase.get_phase
	assert sciapy.regress.HarmonicModelAmpPhase.compute_gradient
	assert sciapy.regress.HarmonicModelCosineSine.get_value
	assert sciapy.regress.HarmonicModelCosineSine.get_amplitude
	assert sciapy.regress.HarmonicModelCosineSine.get_phase
	assert sciapy.regress.HarmonicModelCosineSine.compute_gradient
	assert sciapy.regress.ProxyModel.get_value
	assert sciapy.regress.ProxyModel.compute_gradient
	assert sciapy.regress.setup_proxy_model_with_bounds
	assert sciapy.regress.trace_gas_model
	assert sciapy.regress.models_cel.TraceGasModelSet.get_value
	assert sciapy.regress.models_cel.TraceGasModelSet.compute_gradient
	assert sciapy.regress.models_cel.HarmonicModelAmpPhase.get_value
	assert sciapy.regress.models_cel.HarmonicModelAmpPhase.get_amplitude
	assert sciapy.regress.models_cel.HarmonicModelAmpPhase.get_phase
	assert sciapy.regress.models_cel.HarmonicModelAmpPhase.compute_gradient
	assert sciapy.regress.models_cel.HarmonicModelCosineSine.get_value
	assert sciapy.regress.models_cel.HarmonicModelCosineSine.get_amplitude
	assert sciapy.regress.models_cel.HarmonicModelCosineSine.get_phase
	assert sciapy.regress.models_cel.HarmonicModelCosineSine.compute_gradient
	assert sciapy.regress.models_cel.ProxyModel.get_value
	assert sciapy.regress.models_cel.ProxyModel.compute_gradient
	assert sciapy.regress.models_cel.setup_proxy_model_with_bounds
	assert sciapy.regress.models_cel.trace_gas_model
