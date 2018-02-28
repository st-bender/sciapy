# SCIAMACHY data tools

These SCIAMACHY tools are provided as convenience tools for handling
SCIAMACHY level 1c limb spectra and retrieved level 2 trace-gas densities.

## Overview

### Level 1c tools

The `sciapy.level1c` submodule provides a few
[conversion tools](sciapy/level1c/README.md) for [SCIAMACHY](www.sciamachy.org)
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

**Furtger note**: `.child` files are currently not supported.

### Level 2 tools

The `sciapy.level2` submodule provides
post-processing tools for trace-gas densities retrieved from SCIAMACHY limb scans.
Support simple operations as combining files into *netcdf*, calculating and noting
local solar time at the retrieval grid points, geomagnetic latitudes, etc.

The level 2 tools also include a simple binning algorithm.

### Regression

The `sciapy.regress` submodule can be used for regression analysis of SCIAMACHY
level 2 trace gas density time series, either directly or as daily zonal means.
The regression tools support various parameter fitting methods using
[`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html)
and uncertainty evaluation using Markov-Chain Monte-Carlo sampling with
[`emcee`](https://emcee.readthedocs.io).
Furhter supports covariance modelling via
[`celerite`](https://celerite.readthedocs.io)
and [`george`](https://george.readthedocs.io).

## Install
  
This package can be installed with [pip](https://pip.pypa.io) directly
from github (see https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support
and https://pip.pypa.io/en/stable/reference/pip_install/#git):
```sh
$ pip install [-e] git+https://github.com/st-bender/sciapy.git
```

Or install from a local clone:
```sh
$ git clone https://github.com/st-bender/sciapy.git
$ cd sciapy
```
and then using `pip` (use `-e` at your own risk, see
https://pip.pypa.io/en/stable/reference/pip_install/#install-editable):
```sh
$ pip install -e .
```
or using `setup.py`:
```sh
$ python setup.py install
```

## Usage

The whole module as well as the individual submodules can be loaded as usual:
```py
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
```
