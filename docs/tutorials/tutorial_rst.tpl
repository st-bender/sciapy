{%- extends 'rst.tpl' -%}

{% block header %}
.. module:: sciapy

.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../_static/notebooks/{{ resources.metadata.name }}.ipynb>`_.
          Try a live version: |binderbadge|. |nbviewer|__.

.. |binderbadge| image:: https://mybinder.org/badge_logo.svg
    :target: https://mybinder.org/v2/gh/st-bender/sciapy/master?filepath=docs/_static/notebooks/{{ resources.metadata.name }}.ipynb

.. |nbviewer| replace:: View in *nbviewer*
__ https://nbviewer.jupyter.org/github/st-bender/sciapy/tree/master/docs/_static/notebooks/{{ resources.metadata.name }}.ipynb

.. _{{resources.metadata.name}}:
{% endblock %}

{% block any_cell %}
{%- if (cell.metadata.nbsphinx != 'hidden')
    and (not cell.metadata.hide_input)
    and ("hide" not in cell.metadata.tags)
    and ((cell.metadata.slideshow is not defined)
        or (cell.metadata.slideshow.slide_type != 'skip')) -%}
{{ super() }}
{%- endif -%}
{% endblock any_cell %}
