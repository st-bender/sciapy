#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2017-2018 Stefan Bender
#
# This file is part of sciapy.
# sciapy is free software: you can redistribute it or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""SCIAMACHY level 2 binning

Command line interface to SCIAMACHY level 2 data binning,
both in time (daily) and (geomagnetic) latitude.
"""

import argparse as ap
import datetime as dt
import logging
from os import path

import numpy as np
import pandas as pd
import xarray as xr
try:
	from dask import compute, delayed
	from dask.distributed import Client
except ImportError:
	delayed = None

from sciapy.level2.binning import bin_lat_timeavg

# non-sensible vairables to drop
_drop_vars = ["NO_ERR_std", "NO_ETOT_std", "NO_RSTD_std",
		"NO_ERR_cnt", "NO_ETOT_cnt", "NO_RSTD_cnt"]


if __name__ == "__main__":
	logging.basicConfig(level=logging.WARNING,
		format="[%(levelname)-8s] (%(asctime)s) %(filename)s:%(lineno)d %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S %z")

	parser = ap.ArgumentParser()
	parser.add_argument("file", default="SCIA_NO.nc",
			help="the filename of the input netcdf file")
	parser.add_argument("-a", "--area_weighted",
			action="store_true", default=True,
			help="calculate the area-weighted mean within the bins")
	parser.add_argument("-u", "--unweighted",
			dest="area_weighted", action="store_false",
			help="calculate the equally weighted mean within the bins")
	parser.add_argument("-g", "--geomagnetic",
			dest="geomag", action="store_true", default=False,
			help="bin according to geomagnetic latitude instead of "
			"geographic latitude (turns off area weighting). "
			"(default: %(default)s)")
	parser.add_argument("-G", "--bin_var", type=str, default=None,
			help="bin according to the variable given instead of "
				"geographic latitude (turns off area weighting).")
	parser.add_argument("-b", "--bins", metavar="START:END:SIZE",
			default="-90:90:5",
			help="bins from START to END (inclusive both) with into SIZE sized bins "
			"(default: %(default)s)")
	parser.add_argument("-B", "--binn", metavar="START:END:NUM",
			default=None,
			help="bins from START to END (inclusive both) into NUM bins "
			"(default: not used)")
	parser.add_argument("-m", "--mlt", action="store_true", default=False,
			help="indicate whether to deal with nominal or MLT data (default: False)")
	parser.add_argument("-o", "--output", help="filename of the output file")
	parser.add_argument("-t", "--akm_threshold", type=float, default=0.002,
			help="the averaging kernel diagonal element threshold "
			"for the mask calculation "
			"(default: %(default)s)")
	parser.add_argument("-j", "--jobs", metavar="N", type=int, default=1,
			help="Use N parallel threads for binning "
			"(default: %(default)s)")
	loglevels = parser.add_mutually_exclusive_group()
	loglevels.add_argument("-l", "--loglevel", default="WARNING",
			choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
			help="change the loglevel "
			"(default: %(default)s)")
	loglevels.add_argument("-q", "--quiet", action="store_true", default=False,
			help="less output, same as --loglevel=ERROR "
			"(default: %(default)s)")
	loglevels.add_argument("-v", "--verbose", action="store_true", default=False,
			help="verbose output, same as --loglevel=INFO "
			"(default: %(default)s)")
	args = parser.parse_args()
	if args.quiet:
		logging.getLogger().setLevel(logging.ERROR)
	elif args.verbose:
		logging.getLogger().setLevel(logging.INFO)
	else:
		logging.getLogger().setLevel(args.loglevel)

	orbit_filename = args.file

	# geomagnetic/geographic setup
	if args.geomag:
		logging.debug("using default geomagnetic latitudes")
		binvar = "gm_lats"
		args.area_weighted = False
	elif args.bin_var is not None:
		logging.debug("using custom latitude variable")
		binvar = args.bin_var
		args.area_weighted = False
	else:
		logging.debug("using default geographic latitudes")
		binvar = "latitude"
	logging.info("binning according to \"%s\"", binvar)
	lats_rename_dict = {"{0}_bins".format(binvar): "latitude"}

	if args.area_weighted:
		logging.info("area weighted bins")
	else:
		logging.info("equally weighted bins")

	if args.binn is None:
		bin0, bin1, binstep = list(map(float, args.bins.split(':')))
		bins = np.r_[bin0:bin1 + 0.5 * binstep:binstep]
	else:
		bin0, bin1, binnum = list(map(float, args.binn.split(':')))
		bins = np.linspace(bin0, bin1, binnum + 1)
		binstep = bins.diff[0]
	logging.debug("using %s deg sized bins: %s", binstep, bins)

	if args.output is None:
		output = ("scia_dzm_{0}_akm{1:.3f}_{2}{3:.0f}_{4}.nc"
				.format("".join(c if c.isalnum() else '_'
								for c in path.basename(orbit_filename[:-3])),
					args.akm_threshold,
					"geomag" if args.geomag else "geogra",
					binstep,
					"aw" if args.area_weighted else "nw"))
	else:
		output = args.output
	logging.info("saving to: %s", output)

	ds = xr.open_mfdataset(orbit_filename, decode_times=False,
			chunks={"time": 820, "latitude": 18, "altitude": 17})
	ds["longitude"].values = ds.longitude.values % 360.

	if args.mlt:
		# construct the time (day) bin edges from jumps in the time variable
		# works reliably only for the MLT data
		time_rename_dict = {"time_bins": "time"}
		tbin_edges = np.concatenate([[ds.time.values[0] - 0.5],
			ds.time.values[np.where(np.diff(ds.time) > 1)] + 0.01,
			[ds.time.values[-1] + 0.5]])
		tbin_labels = ds.time.groupby_bins("time", tbin_edges).mean("time")
		ds_bins_daily_gb = ds.groupby_bins("time", tbin_edges, labels=tbin_labels)
	else:
		time_rename_dict = {"date": "time"}
		ds["time"] = xr.conventions.decode_cf_variable("time", ds.time)
		# ds.groupby("time.date") does not work anymore :(
		ds_bins_daily_gb = ds.groupby(
				xr.DataArray(
					pd.to_datetime(pd.DatetimeIndex(ds.time.data).date),
					coords=[ds.time], dims=["time"], name="date"))

	if args.jobs > 1 and delayed is not None:
		# use dask.delayed and dask.compute to distribute the binning
		logging.info("multi-threaded binning with dask using %s threads",
					args.jobs)
		binned = (delayed(bin_lat_timeavg)(
						ds, binvar=binvar,
						bins=bins, area_weighted=args.area_weighted)
					for _, ds in iter(ds_bins_daily_gb))
		client = Client()
		logging.info("dask.distributed client: %s", client)
		ds_bins_daily = (ds_bins_daily_gb
				._combine(compute(*binned, num_workers=args.jobs))
				.rename(lats_rename_dict)
				.drop(_drop_vars))
	else:
		# serial binning with .apply()
		logging.info("single-threaded binning")
		ds_bins_daily = (ds_bins_daily_gb
				.apply(bin_lat_timeavg,
					binvar=binvar, bins=bins,
					area_weighted=args.area_weighted)
				.rename(lats_rename_dict)
				.drop(_drop_vars))
	logging.info("finished binning.")
	del ds_bins_daily_gb

	ds_bins_daily = ds_bins_daily.rename(time_rename_dict)
	ds_bins_daily["time"].attrs = ds["time"].attrs
	ds_bins_daily["time"].attrs.update(
		{"axis": "T", "long_name": "measurement date"})

	# construct tha mask from the averaging kernel diagonal elements
	ds_bins_daily["NO_MASK"] = (ds_bins_daily.NO_AKDIAG < args.akm_threshold)
	ds_bins_daily["NO_MASK"].attrs = {"long_name": "density mask", "units": "1"}

	# copy coordinate attributes
	# "time" was already set above
	for var in filter(lambda c: c != "time", ds.coords):
		logging.debug("copying coordinate attributes for: %s", var)
		ds_bins_daily[var].attrs = ds[var].attrs
	if args.geomag:
		ds_bins_daily["latitude"].attrs.update(
			{"long_name": "geomagnetic_latitude"})
	# copy global attributes
	ds_bins_daily.attrs = ds.attrs
	# note binning time
	ds_bins_daily.attrs["binned_on"] = (dt.datetime.utcnow()
			.replace(tzinfo=dt.timezone.utc)
			.strftime("%a %b %d %Y %H:%M:%S %Z"))
	ds_bins_daily.attrs["latitude_bin_type"] = \
		"geomagnetic" if args.geomag else "geographic"

	ds_bins_daily.to_netcdf(output, unlimited_dims=["time"])
