Changelog
=========

v0.0.7 (2022-04-04)
-------------------

### New

- CI support for Python 3.8, 3.9, and 3.10

### Changes

- Fixed and updated tests to increase code coverage
- Updated AE index and Lyman-alpha data files
- Updated docs
- Uses Github actions for CI and CD
- Removed Python 3.4 from CI setup, support status unclear
- Code style is more `black`-like now


v0.0.6 (2020-02-09)
-------------------

### New

- Documentation on `readthedocs` <https://sciapy.readthedocs.io>
  with example notebooks
- Extensive MCMC sampler statistics

### Changes

- The local MSIS module has been extracted to its own package
  called `pynrlmsise00` <https://github.com/st-bender/pynrlmsise00>
- Increased test coverage


v0.0.5 (2018-08-21)
-------------------

### New

- Enables the proxies to be scaled by cos(SZA)
- Enables the data to be split into (optionally randomized) training and test sets
- Continuous integration with https://travis-ci.org on https://travis-ci.org/st-bender/sciapy
- Includes tests, far from complete yet
- Installing with `pip`

### Other changes

- Code clean up and resource handling


v0.0.4 (2018-08-12)
-------------------

First official alpha release.
