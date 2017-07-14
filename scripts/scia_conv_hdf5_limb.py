#!/usr/bin/env python
# coding: utf-8
"""SCIAMACHY l1c hdf5 to ascii conversion

Copyright (c) 2017 Stefan Bender

This program is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published
by the Free Software Foundation, version 2.
See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.

This program converts SCIAMACHY level 1c HDF5 spectra, as produced
by the SRON nadc_tools (https://github.com/rmvanhees/nadc_tools),
to plain ascii text files or to binary files ready to be used in
trace gas retrievals.

Usage
-----
$ python sron_hdf5_spectra.py [l1c_hdf5] [-C|--cat categories] [-c|--clus clusters]

Arguments
---------
l1c_hdf5 : filename
	The input HDF5 file.
categories : int or tuple
	The measurement categories to extract, can be a single number or a
	comma separated list of numbers, for example 26,27 for the MLT states.
clusters : int or tuple
	The spectral clusters to extract, can be a single number or a
	comma separated list of numbers, for example 2,3,4 for channel 1.

Example
-------
$ python sron_hdf5_spectra.py SCI_NL__1PYDPA20100203_031030_000060632086_00319_41455_0002.ch1.h5 --cat 26,27 --clus 2,3,4
"""

from __future__ import absolute_import, division, print_function

import argparse as ap
import logging

import h5py
import numpy as np

import sciapy.level1c as slvl1c

def main():
	logging.basicConfig(level=logging.WARN,
			format="[%(levelname)-8s] (%(asctime)s) "
			"%(filename)s:%(lineno)d %(message)s",
			datefmt="%Y-%m-%d %H:%M:%S %z")
	parser = ap.ArgumentParser()
	parser.add_argument("file", help="The input HDF5 file.",
			default="SCI_NL__1PYDPA20100203_031030_000060632086_00319_41455_0002.ch1.h5")
	parser.add_argument("-C", "--cat", help="The categories to extract, either a "
			"single number or a comma-separated list of numbers (default: %(default)s)",
			default="26,27")
	parser.add_argument("-c", "--clus", help="The spectral clusters to extract, either a "
			"single number or a comma-separated list of numbers (default: %(default)s)",
			default="2,3,4")
	parser.add_argument("-z", "--solar_id", default="D0",
			choices=["D0", "D1", "D2", "E0", "E1", "A0", "A1", "N1", "N2", "N3", "N4", "N5"],
			help="The solar reference ID to extract (default: %(default)s).")
	loglevels = parser.add_mutually_exclusive_group()
	loglevels.add_argument("-q", "--quiet", action="store_true", default=False,
			help="less output, same as --loglevel=ERROR (default: %(default)s)")
	loglevels.add_argument("-v", "--verbose", action="store_true", default=False,
			help="verbose output, same as --loglevel=INFO (default: %(default)s)")
	loglevels.add_argument("-l", "--loglevel", default="WARNING",
			choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
			help="change the loglevel (default: %(default)s)")
	args = parser.parse_args()
	if args.quiet:
		logging.getLogger().setLevel(logging.ERROR)
	elif args.verbose:
		logging.getLogger().setLevel(logging.INFO)
	else:
		logging.getLogger().setLevel(args.loglevel)

	cats = [n for n in map(int, args.cat.split(','))]
	cl_ids = [n - 1 for n in map(int, args.clus.split(','))]
	logging.debug("categories: %s", cats)
	logging.debug("cluster ids: %s", cl_ids)

	hf = h5py.File(args.file, "r")

	mlt_idxs = np.array([], dtype=int)
	for cat in cats:
		meas_cats = hf.get("/ADS/STATES")["meas_cat"]
		mlt_idxs = np.append(mlt_idxs, np.where(meas_cats == cat)[0])
	logging.info("limb state indexes: %s", mlt_idxs)

	for sid, lstate_id in enumerate(sorted(mlt_idxs)):
		logging.info("processing limb state nr. %s (%s)...", lstate_id, sid)
		slsc = slvl1c.scia_limb_scan()
		# read and continue to the next state if reading failed
		if slsc.read_from_hdf5(hf, lstate_id, sid, cl_ids):
			continue
		logging.debug("final shapes: %s (wls), %s (signal)",
				slsc.wls.shape, slsc.limb_data["rad"].shape)
		filename = "SCIA_limb_{0:04d}{1:02d}{2:02d}_{3:02d}{4:02d}{5:02d}_{6}_{7}_{8:05d}".format(
				slsc.date[0], slsc.date[1], slsc.date[2],
				slsc.date[3], slsc.date[4], slsc.date[5],
				slsc.orbit_state[3], slsc.orbit_state[4],
				slsc.orbit_state[0])

		slsc.write_to_textfile("{0}.dat".format(filename))
		logging.info("limb state nr. %s written to %s",
				lstate_id, "{0}.dat".format(filename))
		slsc.write_to_mpl_binary("{0}.l_mpl_binary".format(filename))
		logging.info("limb state nr. %s written to %s",
				lstate_id, "{0}.l_mpl_binary".format(filename))
		del slsc

	sol = slvl1c.scia_solar()
	sol.read_from_hdf5(hf, args.solar_id)
	sol_filename = ("SCIA_solar_{0:%Y%m%d}_{1:%H%M%S}_{2}_{3:05d}".format(
					sol.time, sol.time, sol.solar_id, sol.orbit))
	sol.write_to_textfile("{0}.dat".format(sol_filename))
	logging.info("solar reference %s written to %s",
			sol.solar_id, "{0}.dat".format(sol_filename))
	del sol

	hf.close()


if __name__ == "__main__":
	main()
