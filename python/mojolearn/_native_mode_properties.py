# SPDX-License-Identifier: Apache-2.0
"""Fixed property recipes that reuse an existing complete native operation."""
from types import BuiltinFunctionType, ModuleType


def connect(context, package):
    # glue: inspect only already-loaded built-in declarations; no imports or
    # property callbacks are needed when a public attribute is read later.
    import sys
    module = sys.modules.get(package.__name__ + '._arima_impl')
    if type(module) is not ModuleType:
        return
    fields = vars(module)
    entry = fields.get('_TSA')
    cls = fields.get('ARIMA')
    if type(entry) is not BuiltinFunctionType or type(cls) is not type:
        return
    recipes = context.setdefault('native_property_recipes', {})
    for name in ('mu_', 'beta_', 'ar_', 'ma_', 'sar_', 'sma_', 'sigma2_'):  # glue: fixed native ARIMA parameter properties
        member = vars(cls)[name]
        if type(member) is not property:
            raise TypeError('ARIMA native parameter property changed during connection')
        kind = 'native_arima_' + name
        context['property_schemas'].setdefault(member, kind)
        recipes.setdefault(kind, dict(entry=entry, operation='arima_parameter',
                                      arguments=(name,), context=fields['_TSA_CONTEXT']))
