{{ fullname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {%- set listed = methods | reject("equalto", "__init__") | list %}
   {% if listed %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in listed %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: Attributes

   .. autosummary::
   {% for item in attributes %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
