# SCIAMACHY level 1c tools

Conversion tools for [SCIAMACHY](http://www.sciamachy.org) level 1c
calibrated spectra to be used for trace gas retrieval with
[scia\_retrieval\_2d](https://github.com/st-bender/scia_retrieval_2d).

This package currently supports the following level 1c conversions
for limb and solar reference spectra
(`.l_mpl_binary` does not apply to the solar spectra):

 type            | read | write
-----------------|------|-------
 `.dat`          | yes  | yes
 `.l_mpl_binary` | yes  | yes
 `.nc`           | yes  | yes
 `.h5`           | yes  | no

**This is no level 1b to level 1c calibration tool!**

For calibrating level 1b spectra (for example SCI\_NL\_\_1P version 8.02
provided by ESA via the
[ESA data browser](https://earth.esa.int/web/guest/data-access/browse-data-products))
to level 1c spectra, use the
[SciaL1C](https://earth.esa.int/web/guest/software-tools/content/-/article/scial1c-command-line-tool-4073)
command line tool or the free software
[nadc\_tools](https://github.com/rmvanhees/nadc_tools).
The first produces `.child` files, the second can output to HDF5 (`.h5`).

**Note**: `.child` files are currently not supported.

## Install

The sciapy level 1c module is part of `sciapy` whose installation
is described in the main [README](../../README.md).

## Usage

A simple documentation it provided using `pydoc`:
```sh
$ pydoc sciapy.level1c
```

This packages provides a submodule `level1c` with the following classes
- `scia_limb_scan` to handle calibrated SCIAMACHY level 1c limb scan spectra
- `scia_solar` to handle calibrated SCIAMACHY solar reference spectra

The submodule documentation can also be accessed with `pydoc`:
```sh
$ pydoc sciapy.level1c
$ pydoc sciapy.level1c.scia_limb_scan
$ pydoc sciapy.level1c.scia_solar
```

## Examples

Convert a level1c ascii limb spectral file to binary format for
[scia\_retrieval\_2d](https://github.com/st-bender/scia_retrieval_2d):
```py
>>> import sciapy
>>> scia_limb_scan = sciapy.level1c.scia_limb_scan()
>>> scia_limb_scan.read_from_textfile("/path/to/limb_state_filename.dat")
>>> scia_limb_scan.write_to_mpl_binary("/path/to/limb_state_filename.l_mpl_binary")
```

Using [nadc\_tools](https://github.com/rmvanhees/nadc_tools) to convert
to HDF5 and then save to ascii spectral files:
```sh
# calibrate and save to HDF5 first
$ /path/to/nadc_tools/bin/scia_nl1 -limb -no_gads -no_ads --cat=26,27 --channel=1 --cal=1,2,4,5+,6,7,9,E,N -hdf5 -compress /path/to/L1b_v8.02/SCI_NL__1PYDPA.N1 --output=SCI_NL__1PYDPA.N1.ch1.h5
```
within python the rough steps are:
```py
>>> import h5py
>>> import sciapy
>>> h5file = h5py.File(args.file, "r")
>>> scia_limb_scan = sciapy.level1c.scia_limb_scan()
>>> scia_limb_scan.read_from_hdf5(h5file, limb_state_id, id, cluster_ids)
>>> scia_limb_scan.write_to_textfile("/path/to/limb_state_filename.dat")
```
For a more complete example which also extracts the solar reference spectrum,
see the accompanying [scia\_conv\_hdf5\_limb.py](https://github.com/st-bender/sciapy/blob/master/scripts/scia_conv_hdf5_limb.py) script.

A simple conversion tool is provided with the
[scia\_binary\_util.py](https://github.com/st-bender/sciapy/blob/master/scripts/scia_binary_util.py) script.
