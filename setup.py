from setuptools import setup

setup(name='sciapy',
		version='0.0.1',
		description='Python tools for (some) SCIAMACHY data',
		url='http://github.com/st-bender/sciapy',
		author='Stefan Bender',
		author_email='stefan.bender@kit.edu',
		packages=['sciapy'],
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
		zip_safe=False)
