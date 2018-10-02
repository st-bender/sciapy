{%- extends 'rst.tpl' -%}

{% block header %}
.. module:: sciapy

.. note:: This tutorial was generated from an IPython notebook that can be
          downloaded `here <../../_static/notebooks/{{ resources.metadata.name }}.ipynb>`_.
          Try a live version: |binderbadge|.

.. |binderbadge| image:: https://mybinder.org/badge.svg
    :target: https://mybinder.org/v2/gh/st-bender/sciapy/master?filepath=docs/_static/notebooks/{{ resources.metadata.name }}.ipynb

.. _{{resources.metadata.name}}:
{% endblock %}

{% block any_cell %}
{%- if cell.metadata.nbsphinx != 'hidden' or not cell.metadata.hide_input -%}
{{ super() }}
{%- endif -%}
{% endblock any_cell %}
