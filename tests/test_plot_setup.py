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
"""SCIAMACHY plot setup tests
"""

import sciapy
from sciapy import plot_setup


def test_module_structure():
	assert sciapy.plot_setup.plot_setup
	assert plot_setup.plot_setup


def test_plot_setup():
	from matplotlib import rcParams
	plot_setup.plot_setup()
	assert rcParams["figure.dpi"] == 96
	assert rcParams["figure.figsize"] == [8, 5]
	assert rcParams["figure.constrained_layout.use"]
	assert rcParams["font.size"] == 16
	assert rcParams['mathtext.default'] == 'regular'
	assert rcParams['savefig.dpi'] == 600
	assert rcParams['pdf.compression'] == 0
	assert rcParams['axes.linewidth'] == 1.5
	assert rcParams['lines.linewidth'] == 1.5
	# visible ticks
	assert rcParams["xtick.minor.visible"]
	assert rcParams["ytick.minor.visible"]
	assert rcParams["xtick.top"]
	assert rcParams["ytick.right"]
	# tick sizes and padding
	assert rcParams["xtick.major.width"] == 1.5
	assert rcParams["xtick.major.size"] == 6
	assert rcParams["xtick.major.pad"] == 8
	assert rcParams["ytick.major.width"] == 1.5
	assert rcParams["ytick.major.size"] == 6
	assert rcParams["ytick.major.pad"] == 8
	assert rcParams["xtick.minor.size"] == 3
	assert rcParams["ytick.minor.size"] == 3
	# turn off axis spines
	assert not rcParams["axes.spines.left"]
	assert not rcParams["axes.spines.bottom"]
	assert not rcParams["axes.spines.top"]
	assert not rcParams["axes.spines.right"]


def test_grid_lines():
	from matplotlib import rcParams
	rcParams.update(plot_setup.GRID_STYLE)
	assert rcParams["axes.grid"]
	assert rcParams["axes.grid.axis"] == "y"
	assert rcParams["grid.alpha"] == 0.5


def test_line_spines():
	from matplotlib import rcParams
	rcParams.update(plot_setup.LINE_SPINES)
	assert rcParams["axes.spines.left"]
	assert rcParams["axes.spines.bottom"]


def test_line_ticks():
	from matplotlib import rcParams
	rcParams.update(plot_setup.LINE_TICKS)
	assert not rcParams["xtick.top"]
	assert not rcParams["ytick.right"]
