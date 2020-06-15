from codecs import open
from os import path
import re
# Always prefer setuptools over distutils
from setuptools import find_packages, setup

name = "sciapy"
meta_path = path.join(name, "__init__.py")
here = path.abspath(path.dirname(__file__))

extras_require = {
	"msis": ["nrlmsise00"],
	"tests": ["nrlmsise00", "pytest"],
}
extras_require["all"] = sorted(
	{v for req in extras_require.values() for v in req},
)


# Approach taken from
# https://packaging.python.org/guides/single-sourcing-package-version/
# and the `attrs` package https://www.attrs.org/
# https://github.com/python-attrs/attrs
def read(*parts):
	"""
	Builds an absolute path from *parts* and and return the contents of the
	resulting file.  Assumes UTF-8 encoding.
	"""
	with open(path.join(here, *parts), "rb", "utf-8") as f:
		return f.read()


def find_meta(meta, *path):
	"""
	Extracts __*meta*__ from *path* (can have multiple components)
	"""
	meta_file = read(*path)
	meta_match = re.search(
		r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M,
	)
	if not meta_match:
		raise RuntimeError("__{meta}__ string not found.".format(meta=meta))
	return meta_match.group(1)


# Get the long description from the README file
long_description = read("README.md")
version = find_meta("version", meta_path)

if __name__ == "__main__":
	setup(
		name=name,
		version=version,
		description='Python tools for (some) SCIAMACHY data',
		long_description=long_description,
		long_description_content_type="text/markdown",
		url='http://github.com/st-bender/sciapy',
		author='Stefan Bender',
		author_email='stefan.bender@ntnu.no',
		packages=find_packages(),
		scripts=['scripts/scia_binary_util.py',
			'scripts/scia_conv_hdf5_limb.py',
			'scripts/scia_daily_zonal_mean.py',
			'scripts/scia_post_process_l2.py'],
		package_data={
			'sciapy.level2': ['IGRF.tab', 'AACGM2005_80km_grid.nc'],
			'sciapy': ['data/indices/*.dat', 'data/indices/*.txt'],
		},
		install_requires=[
			'numpy>=1.13.0',
			'scipy>=0.17.0',
			'matplotlib>=2.2',
			'netCDF4',
			'h5py',
			'dask',
			'toolz',
			'astropy',
			'pandas',
			'xarray',
			'parse',
			'autograd',
			'celerite>=0.3.0',
			'corner',
			'george',
			'emcee',
		],
		extras_require=extras_require,
		license='GPLv2',
		classifiers=[
			"Development Status :: 3 - Alpha",
			"Intended Audience :: Science/Research",
			"License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
			"Programming Language :: Python",
			"Programming Language :: Python :: 2",
			"Programming Language :: Python :: 2.7",
			"Programming Language :: Python :: 3",
			"Programming Language :: Python :: 3.4",
			"Programming Language :: Python :: 3.5",
			"Programming Language :: Python :: 3.6",
			"Programming Language :: Python :: 3.7",
			"Programming Language :: Python :: 3.8",
		],
		entry_points={'console_scripts':
			[
				'scia_regress = sciapy.regress.__main__:main',
				'scia_post_process_l2 = sciapy.level2.post_process:main',
			]
		},
		options={
			"bdist_wheel": {"universal": True},
		},
		zip_safe=False,
	)
