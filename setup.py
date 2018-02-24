from codecs import open
from os import path
# Always prefer setuptools over distutils
from setuptools import setup
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

setup(name='sciapy',
		version='0.0.1',
		description='Python tools for (some) SCIAMACHY data',
		long_description=long_description,
		url='http://github.com/st-bender/sciapy',
		author='Stefan Bender',
		author_email='stefan.bender@kit.edu',
		packages=['sciapy'],
		scripts=['scripts/scia_binary_util.py',
			'scripts/scia_conv_hdf5_limb.py',
			'scripts/scia_post_process_l2.py'],
		package_data={'sciapy': ['sciapy/level2/IGRF.tab',
				'sciapy/level2/AACGM2005_80km_grid.nc']},
		install_requires=[
			'numpy',
			'netCDF4',
			'h5py',
			'astropy',
			'pandas',
			'xarray',
			'pysolar',
			'parse'],
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
		zip_safe=False)
