# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8
#
# Copyright (c) 2020 Stefan Bender
#
# This module is part of sciapy.
# sciapy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""Compare netcdf files

Compare netcdf files, testing the variable attributes and values
to ensure format compatibility.
"""
import sys
import xarray as xr

__all__ = ["nccmpattrs", "ncallclose", "ncequal", "ncidentical"]


def cmpvarattrs(v1, v2):
	"""Compare variable attribute values"""
	msg = ""
	same = True
	for a in v1.attrs:
		a1 = getattr(v1, a, None)
		a2 = getattr(v2, a, None)
		if a2 is None or a2 != a1:
			msg = "{0}\nL\t{1}\t{2}\t{3}\nR\t{1}\t{2}\t{4}".format(
				msg, v1.name, a, a1, a2,
			)
			same = False
	return same, msg


def nccmpattrs(file1, file2, ignore=[]):
	"""Compare variable attributes and global attributes"""
	msg = ""
	same = True
	with xr.open_dataset(file1) as ds1:
		ds2 = xr.open_dataset(file2)
		for v in ds1.variables:
			same, vmsg = cmpvarattrs(ds1[v], ds2[v])
			if not same:
				msg = "{0}{1}".format(msg, vmsg)
		for attr in ds1.attrs:
			lattr = getattr(ds1, attr, None)
			rattr = getattr(ds2, attr, None)
			if lattr != rattr and attr not in ignore:
				msg = "{0}\nL\t{1}\t{2}\nR\t{1}\t{3}".format(
					msg, attr, lattr, rattr,
				)
				same = False
	assert same, msg


def ncallclose(file1, file2):
	with xr.open_dataset(file1) as ds1:
		ds2 = xr.open_dataset(file2)
		xr.testing.assert_allclose(ds1, ds2)


def ncequal(file1, file2):
	with xr.open_dataset(file1) as ds1:
		ds2 = xr.open_dataset(file2)
		xr.testing.assert_equal(ds1, ds2)


def ncidentical(file1, file2, ignore=[]):
	with xr.open_dataset(file1) as ds1:
		ds2 = xr.open_dataset(file2)
		for ign in ignore:
			del ds1.attrs[ign]
			del ds2.attrs[ign]
		xr.testing.assert_identical(ds1, ds2)


if __name__ == "__main__":
	nccmpattrs(*sys.argv[1:])
	ncallclose(*sys.argv[1:])
	ncequal(*sys.argv[1:])
	ncidentical(*sys.argv[1:])
