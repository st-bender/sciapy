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
"""Sciapy test fixtures

Test fixtures to run tests in a clean environment.
"""
import shutil
import tempfile

import pytest


@pytest.fixture(scope="session")
def tmpdir():
	tmpdir = tempfile.mkdtemp()
	yield tmpdir
	shutil.rmtree(tmpdir)
