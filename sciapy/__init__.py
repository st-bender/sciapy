"""(Some) SCIAMACHY tools

Tools to handle SCIAMACHY/Envisat level 1c spectral data,
and to process level 2 number density data (only NO for now).
"""
__version__ = "0.0.5"

try:
	__SCIAPY_SETUP__
except NameError:
	__SCIAPY_SETUP__ = False

if not __SCIAPY_SETUP__:
	from . import level1c
	from . import level2
	from . import regress
