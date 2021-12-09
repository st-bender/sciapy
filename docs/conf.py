# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from os import getenv

from sciapy import __version__

project = u'sciapy'
copyright = u'2018, Stefan Bender'
author = u'Stefan Bender'

version = __version__
release = __version__

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_markdown_tables',
    'recommonmark',
    'numpydoc'
]
if getenv('SPELLCHECK'):
    extensions += 'sphinxcontrib.spelling',
    spelling_show_suggestions = True
    spelling_lang = 'en_GB'

source_suffix = ['.md', '.rst']
master_doc = 'index'

language = None

htmlhelp_basename = 'sciapydoc'

autosummary_generate = True

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "11pt",
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'sciapy.tex', u'sciapy Documentation',
     u'Stefan Bender', 'manual'),
]

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'sciapy', u'sciapy Documentation',
     [author], 1)
]

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'sciapy', u'sciapy Documentation',
     author, 'sciapy', 'SCIAMACHY data tools.',
     'Miscellaneous'),
]

# on_rtd is whether we are on readthedocs.org
on_rtd = getenv("READTHEDOCS", None) == "True"
if not on_rtd:
    import sphinx_rtd_theme
    html_theme = "sphinx_rtd_theme"
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

exclude_patterns = [u'_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']
pygments_style = 'default'
templates_path = ['_templates']
html_use_smartypants = True
html_last_updated_fmt = '%b %d, %Y'
html_split_index = True
html_sidebars = {
    '**': ['searchbox.html', 'localtoc.html', 'relations.html',
          'globaltoc.html', 'sourcelink.html'],
}
html_short_title = '%s-%s' % (project, version)
html_context = dict(
    display_github=True,
    github_user="st-bender",
    github_repo="sciapy",
    github_version="master",
    conf_py_path="/docs/",
)
html_static_path = ["_static"]
# Switch to old behavior with html4, for a good display of references,
# as described in https://github.com/sphinx-doc/sphinx/issues/6705
html4_writer = True

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'iris': ('http://scitools.org.uk/iris/docs/latest/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'numba': ('https://numba.pydata.org/numba-doc/latest/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'astropy': ('http://docs.astropy.org/en/stable/', None),
    'xarray': ('https://xarray.pydata.org/en/stable/', None),
    'celerite': ('https://celerite.readthedocs.io/en/stable/', None),
    'george': ('https://george.readthedocs.io/en/stable/', None),
    'emcee': ('https://emcee.readthedocs.io/en/stable/', None),
}
