# SCIAMACHY data tools

[![builds](https://github.com/st-bender/sciapy/actions/workflows/ci_build_and_test.yml/badge.svg?branch=master)](https://github.com/st-bender/sciapy/actions/workflows/ci_build_and_test.yml)
[![docs](https://rtfd.org/projects/sciapy/badge/?version=latest)](https://sciapy.rtfd.io/en/latest/?badge=latest)
[![coveralls](https://coveralls.io/repos/github/st-bender/sciapy/badge.svg)](https://coveralls.io/github/st-bender/sciapy)
[![scrutinizer](https://scrutinizer-ci.com/g/st-bender/sciapy/badges/quality-score.png?b=master)](https://scrutinizer-ci.com/g/st-bender/sciapy/?branch=master)

[![doi code](https://zenodo.org/badge/DOI/10.5281/zenodo.1401370.svg)](https://doi.org/10.5281/zenodo.1401370)
[![doi mcmc samples](https://zenodo.org/badge/DOI/10.5281/zenodo.1342701.svg)](https://doi.org/10.5281/zenodo.1342701)

## Overview

These SCIAMACHY tools are provided as convenience tools for handling
SCIAMACHY level 1c limb spectra and retrieved level 2 trace-gas densities.

More extensive documentation is provided on [sciapy.rtfd.io](https://sciapy.rtfd.io).

### Level 1c tools

The `sciapy.level1c` submodule provides a few
[conversion tools](sciapy/level1c/README.md) for [SCIAMACHY](http://www.sciamachy.org)
level 1c calibrated spectra, to be used as input for trace gas retrieval with
[scia\_retrieval\_2d](https://github.com/st-bender/scia_retrieval_2d).

**Note that this is *not* a level 1b to level 1c calibration tool.**

For calibrating level 1b spectra (for example SCI\_NL\_\_1P version 8.02
provided by ESA via the
[ESA data browser](https://earth.esa.int/web/guest/data-access/browse-data-products))
to level 1c spectra, use the
[SciaL1C](https://earth.esa.int/web/guest/software-tools/content/-/article/scial1c-command-line-tool-4073)
command line tool or the free software
[nadc\_tools](https://github.com/rmvanhees/nadc_tools).
The first produces `.child` files, the second can output to HDF5 (`.h5`).

**Further note**: `.child` files are currently not supported.

### Level 2 tools

The `sciapy.level2` submodule provides
post-processing tools for trace-gas densities retrieved from SCIAMACHY limb scans.
Support simple operations as combining files into *netcdf*, calculating and noting
local solar time at the retrieval grid points, geomagnetic latitudes, etc.

The level 2 tools also include a simple binning algorithm.

### Regression

The `sciapy.regress` submodule can be used for regression analysis of SCIAMACHY
level 2 trace gas density time series, either directly or as daily zonal means.
It uses the [`regressproxy`](https://regressproxy.readthedocs.io) package
for modelling the proxy input with lag and lifetime decay.
The regression tools support various parameter fitting methods using
[`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
and uncertainty evaluation using Markov-Chain Monte-Carlo sampling with
[`emcee`](https://emcee.readthedocs.io).
Further supports covariance modelling via
[`celerite`](https://celerite.readthedocs.io)
and [`george`](https://george.readthedocs.io).

## Install

### Prerequisites

Sciapy uses features from a lot of different packages.
All dependencies will be automatically installed when using
`pip install` or `python setup.py`, see below.
However, to speed up the install or for use
within a `conda` environment, it may be advantageous to
install some of the important packages beforehand:

- `numpy` at least version 1.13.0 for general numerics,
- `scipy` at least version 0.17.0 for scientific numerics,
- `matplotlib` at least version 2.2 for plotting,
- `netCDF4` for the low level netcdf4 interfaces,
- `h5py` for the low level hdf5 interfaces,
- `dask`,
- `toolz`,
- `pandas` and
- `xarray` for the higher level data interfaces,
- `astropy` for (astronomical) time conversions,
- `parse` for ASCII text parsing in `level1c`,
- `pybind11` C++ interface needed by `celerite`
- `celerite` at least version 0.3.0 and
- `george` for Gaussian process modelling,
- `emcee` for MCMC sampling and
- `corner` for the sample histogram plots,
- `regressproxy` for the regression proxy modelling.

Out of these packages, `numpy` is probably the most important one
to be installed first because at least `celerite` needs it for setup.
It may also be a good idea to install
[`pybind11`](https://pybind11.readthedocs.io)
because both `celerite` and `george` use its interface,
and both may fail to install without `pybind11`.

Depending on the setup, `numpy` and `pybind11` can be installed
via `pip`:
```sh
pip install numpy pybind11
```
or [`conda`](https://conda.io):
```sh
conda install numpy pybind11
```

### sciapy

Official releases are available as `pip` packages from the main package repository,
to be found at <https://pypi.org/project/sciapy/>, and which can be installed with:
```sh
$ pip install sciapy
```
The latest development version of
sciapy can be installed with [`pip`](https://pip.pypa.io) directly
from github (see <https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support>
and <https://pip.pypa.io/en/stable/reference/pip_install/#git>):
```sh
$ pip install [-e] git+https://github.com/st-bender/sciapy.git
```

The other option is to use a local clone:
```sh
$ git clone https://github.com/st-bender/sciapy.git
$ cd sciapy
```
and then using `pip` (optionally using `-e`, see
<https://pip.pypa.io/en/stable/reference/pip_install/#install-editable>):
```sh
$ pip install [-e] .
```

or using `setup.py`:
```sh
$ python setup.py install
```

## Usage

The whole module as well as the individual submodules can be loaded as usual:
```python
>>> import sciapy
>>> import sciapy.level1c
>>> import sciapy.level2
>>> import sciapy.regress
```

Basic class and method documentation is accessible via `pydoc`:
```sh
$ pydoc sciapy
```

The submodules' documentation can be accessed with `pydoc` as well:
```sh
$ pydoc sciapy.level1c
$ pydoc sciapy.level2
$ pydoc sciapy.regress
```

## License

This python package is free software: you can redistribute it or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 2 (GPLv2), see [local copy](./LICENSE)
or [online version](http://www.gnu.org/licenses/gpl-2.0.html).
