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
"""Matplotlib figure set up
"""

from matplotlib import rcParams

__all__ = ["plot_setup", "GRID_STYLE", "LINE_SPINES", "LINE_TICKS"]

GRID_STYLE = {"axes.grid": True,
		"axes.grid.axis": "y",
		"grid.alpha": 0.5}

LINE_SPINES = {"axes.spines.left": True,
		"axes.spines.bottom": True}

LINE_TICKS = {"xtick.top": False,
		"ytick.right": False}


def plot_setup():
	rcParams["figure.dpi"] = 96
	rcParams["figure.figsize"] = [8, 5]
	rcParams["font.size"] = 16
	rcParams['mathtext.default'] = 'regular'
	rcParams['savefig.dpi'] = 600
	rcParams['pdf.compression'] = 0
	rcParams['axes.linewidth'] = 1.5
	rcParams['lines.linewidth'] = 1.5
	# visible ticks
	rcParams["xtick.minor.visible"] = True
	rcParams["ytick.minor.visible"] = True
	rcParams["xtick.top"] = True
	rcParams["ytick.right"] = True
	# tick sizes and padding
	rcParams["xtick.major.width"] = 1.5
	rcParams["xtick.major.size"] = 6
	rcParams["xtick.major.pad"] = 8
	rcParams["ytick.major.width"] = 1.5
	rcParams["ytick.major.size"] = 6
	rcParams["ytick.major.pad"] = 8
	rcParams["xtick.minor.size"] = 3
	rcParams["ytick.minor.size"] = 3
	# turn off axis spines
	rcParams["axes.spines.left"] = False
	rcParams["axes.spines.bottom"] = False
	rcParams["axes.spines.top"] = False
	rcParams["axes.spines.right"] = False
	# use constrained layout if available (matplotlib >= 2.2)
	try:
		rcParams["figure.constrained_layout.use"] = True
	except KeyError:
		pass
