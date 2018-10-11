{%- extends 'rst.tpl' -%}

{% block header %}
.. module:: sciapy

.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../_static/notebooks/{{ resources.metadata.name }}.ipynb>`_.
          Try a live version: |binderbadge|.

.. |binderbadge| image:: https://mybinder.org/badge.svg
    :target: https://mybinder.org/v2/gh/st-bender/sciapy/master?filepath=docs/_static/notebooks/{{ resources.metadata.name }}.ipynb

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
