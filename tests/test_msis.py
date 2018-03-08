# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
import os

import numpy as np

from sciapy.level2.nrlmsise00 import gtd7


def test_gtd7():
	# standard inputs
	inputs_base = dict(
			doy=172,
			year=0,  # /* without effect */
			sec=29000,
			alt=400,
			g_lat=60,
			g_long=-70,
			lst=16,
			f107A=150,
			f107=150,
			ap=4,
			ap_a=[])
	# high ap values
	aph = [100.] * 7
	# standard flags
	flags = [0] + [1] * 23
	# update input dicts with values to test
	inputs = [inputs_base.copy() for _ in range(17)]
	inputs[1]["doy"] = 81
	inputs[2]["sec"] = 75000
	inputs[2]["alt"] = 1000
	inputs[3]["alt"] = 100
	inputs[10]["alt"] = 0
	inputs[11]["alt"] = 10
	inputs[12]["alt"] = 30
	inputs[13]["alt"] = 50
	inputs[14]["alt"] = 70
	inputs[16]["alt"] = 100
	inputs[4]["g_lat"] = 0
	inputs[5]["g_long"] = 0
	inputs[6]["lst"] = 4
	inputs[7]["f107A"] = 70
	inputs[8]["f107"] = 180
	inputs[9]["ap"] = 40
	inputs[15]["ap_a"] = aph
	inputs[16]["ap_a"] = aph
	# MSIS test outputs from the documentation
	test_file = os.path.join(
			os.path.realpath(os.path.dirname(__file__)),
			"msis_testoutput.txt")
	test_output = np.genfromtxt(test_file)

	outputs = []
	for inp in inputs[:15]:
		ds, ts = gtd7(input=inp, flags=flags)
		outputs.append(ds + ts)
	flags[9] = -1
	for inp in inputs[15:17]:
		ds, ts = gtd7(input=inp, flags=flags)
		outputs.append(ds + ts)
	# Compare results
	np.testing.assert_allclose(np.asarray(outputs), test_output, rtol=1e-6)
