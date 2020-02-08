from codecs import open
from os import path
from sys import version_info
# Always prefer setuptools over distutils
from setuptools import find_packages, setup
from subprocess import check_call
from distutils.core import Extension

here = path.abspath(path.dirname(__file__))

extnrlmsise00 = Extension(
		name='sciapy.level2.nrlmsise00',
		sources=[
			'sciapy/level2/nrlmsise00module.c',
			'sciapy/level2/nrlmsise00/nrlmsise-00.c',
			'sciapy/level2/nrlmsise00/nrlmsise-00_data.c'
		],
		include_dirs=['sciapy/level2/nrlmsise00'])

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
	long_description = f.read()

if __name__ == "__main__":
	# Approach copied from dfm (celerite, emcee, and george)
	# Hackishly inject a constant into builtins to enable importing of the
	# package before the library is built.
	if version_info[0] < 3:
		import __builtin__ as builtins
	else:
		import builtins
	builtins.__SCIAPY_SETUP__ = True
	from sciapy import __version__

	# update git submodules
	if path.exists(".git"):
		check_call(["git", "submodule", "update", "--init", "--recursive"])

	setup(name='sciapy',
		version=__version__,
		description='Python tools for (some) SCIAMACHY data',
		long_description=long_description,
		url='http://github.com/st-bender/sciapy',
		author='Stefan Bender',
		author_email='stefan.bender@ntnu.no',
		packages=find_packages(),
		scripts=['scripts/scia_binary_util.py',
			'scripts/scia_conv_hdf5_limb.py',
			'scripts/scia_daily_zonal_mean.py',
			'scripts/scia_post_process_l2.py'],
		package_data={'sciapy.level2': ['IGRF.tab',
				'AACGM2005_80km_grid.nc'],
			'sciapy': ['data/indices/*.dat', 'data/indices/*.txt']},
		install_requires=[
			'numpy>=1.13.0',
			'scipy>=0.17.0',
			'matplotlib>=2.2',
			'netCDF4',
			'h5py',
			'dask',
			'toolz',
			'astropy<4.0',
			'pandas',
			'xarray',
			'pysolar',
			'parse',
			'autograd',
			'celerite>=0.3.0',
			'corner',
			'george',
			'emcee'],
		license='GPLv2',
		classifiers=[
			"Development Status :: 3 - Alpha",
			"Intended Audience :: Science/Research",
			"License :: OSI Approved :: GNU General Public License v2 (GPLv2)",
			"Programming Language :: Python",
			"Programming Language :: Python :: 2",
			"Programming Language :: Python :: 3",
		],
		ext_modules=[extnrlmsise00],
		entry_points={'console_scripts':
			['scia_regress = sciapy.regress.__main__:main']
		},
		zip_safe=False)
