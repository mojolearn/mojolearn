# SPDX-License-Identifier: Apache-2.0
"""Connect fixed metric/scorer ABIs once, without product Python callbacks."""
from types import BuiltinFunctionType, FunctionType, GetSetDescriptorType, ModuleType

from ._native_msel_metric_plans import METRIC_PLANS


def connect(context, selection):
    from . import _backend, _metrics_impl, _expansion_metrics
    metrics = {}
    for name, recipe in METRIC_PLANS.items():  # glue: fixed native metric operation metadata
        # Some fixed public metrics are exported by the expansion module only;
        # the metrics alias does not re-export every connected entry. Read the
        # two owned namespaces directly without invoking module attribute hooks.
        function = vars(_metrics_impl).get(name)
        if function is None:
            function = vars(_expansion_metrics).get(name)
        if type(function) is not FunctionType:
            raise TypeError(f'native scorer metric {name} is not the connected public function')
        fields = function.__globals__
        entry = fields['_NATIVE']
        metrics[function] = dict(recipe, name=name, code=function.__code__, entry=entry if type(entry) is BuiltinFunctionType else None,
                                 context=fields['_NATIVE_CONTEXT'])
    bindings = {}
    for mode in ('identical', 'fast'):  # glue: retain existing compiled metric bindings by mode once
        try:
            module = _metrics_impl._get_binding(mode)
            if type(module) is _backend._HostBinding:
                module = _backend.load_host_module(module._HostBinding__basename)  # cpu-route: import-time connection to the actual CPU metric extension
            if type(module) is not ModuleType:
                raise TypeError('scorer metrics require a compiled extension module')
            bindings[mode] = module
        except (ImportError, AttributeError, TypeError, RuntimeError, ValueError) as error:
            bindings[mode] = type(error).__name__ + ': ' + str(error)
    cls = selection._Scorer
    descriptor = vars(cls)['__dict__']
    if type(descriptor) is not GetSetDescriptorType:
        raise TypeError('native scorers require native instance dictionary storage')
    hooks = tuple((name, vars(cls).get(name)) for name in ('__call__', '__init__', '__repr__', '__getattribute__',  # glue: fixed scorer class method/protocol identities
                                                        '__getattr__', '__setattr__', '__del__', '__dict__'))
    context.update(family_msel_metric_plans=metrics, family_msel_metric_bindings=bindings,
                   family_msel_metric_function_type=FunctionType, family_msel_scorer_fields=descriptor.__get__,
                   family_msel_scorer_hooks=hooks,
                   family_msel_binary_proba=frozenset(selection._BINARY_PROBA))
