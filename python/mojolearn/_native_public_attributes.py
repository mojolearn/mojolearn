# SPDX-License-Identifier: Apache-2.0
"""Import-time native attribute slots for classes with connected descriptors."""
from types import BuiltinFunctionType, ClassMethodDescriptorType, MethodType, ModuleType

_CONTEXT = None


def _resolved(cls, name, missing):
    # glue: class dictionaries, without invoking a descriptor or getter
    return next((vars(base)[name] for base in type.__dict__['__mro__'].__get__(cls)  # glue: import-time native MRO metadata
                 if name in vars(base)), missing)


def connect_classes(package):
    # glue: establish native slots once, before dependent estimator/parameter
    # tables capture method identities; lazy imports admit only new classes.
    global _CONTEXT
    import sys
    from . import _mode, _cholesky_impl, _array
    from ._native_classical_bridge import _layout, _numpy_free_context
    from .tokenizer import _binding

    if _CONTEXT is None:
        module = _binding()._module
        entry = module.public_attribute_native_call
        factory = module.classical_attribute_descriptor
        if type(entry) is not BuiltinFunctionType or type(factory) is not BuiltinFunctionType:
            raise TypeError('public attributes require compiled native entries')
        context = _numpy_free_context()
        context['attribute_array_dtypes'] = _array.SUPPORTED_DTYPES
        array_class = context['array_class']
        context['attribute_array_slots'] = {name: vars(array_class)[name]  # glue: native storage slots, never Array protocols
                                            for name in ('_store', '_addr', '_readonly', 'shape', 'dtype', 'order')}  # glue: fixed import-time native slot identities
        context.update(native_module_type=ModuleType, plans={}, pending={}, missing=object(),
                       property_schemas={vars(_cholesky_impl.Cholesky)['logdet_']: 'cholesky_logdet'},
                       stl_connected=False)
        context['descriptor'] = factory(MethodType(entry, context))
        _CONTEXT = context
    context = _CONTEXT
    stl_module = sys.modules.get(package.__name__ + '._x_sequence_stl')
    if not context['stl_connected'] and type(stl_module) is ModuleType:
        context['property_schemas'][vars(vars(stl_module)['STL'])['config']] = 'stl_config'
        context['stl_connected'] = True
    candidates = set()
    for name, module in tuple(sys.modules.items()):  # glue: loaded product declarations only; no optional module imports
        if (name == package.__name__ or name.startswith(package.__name__ + '.')) and not name.startswith(package.__name__ + '.tests'):
            if type(module) is ModuleType:
                candidates.update(value for value in vars(module).values()  # glue: exact product class identities
                                  if type(value) is type and value.__module__.startswith(package.__name__ + '.')
                                  and not value.__module__.startswith(package.__name__ + '.tests'))
    # Compute eligibility before installing any slot: installing a base must
    # never silently convert a subclass with an unported property or callback.
    eligible = []
    original = vars(_mode.PublicOutputsMixin)['__getattribute__']
    for cls in candidates:  # glue: verify complete descriptor coverage before class mutation
        if cls in context['plans'] or cls in context['pending']:
            continue
        if _resolved(cls, '__getattribute__', context['missing']) not in (original, context['descriptor']):
            continue
        if cls in (_mode.PublicOutputsMixin, _mode.ParamsMixin, _mode.NumericModeMixin):
            continue
        try:
            if _resolved(cls, '__getattr__', context['missing']) is not context['missing']:
                raise TypeError('Python __getattr__ requires a native owner')
            plan = _layout(cls, context['descriptor'])
            names = {}
            for base in plan['mro']:  # glue: descriptor precedence is frozen at connection
                for name, member in vars(base).items():  # glue: exact class namespace metadata
                    names.setdefault(name, member)
            properties = {}
            for name, member in names.items():  # glue: native property schema admission
                if type(member) is property:
                    kind = context['property_schemas'].get(member)
                    if kind is None:
                        raise TypeError('unconnected property: ' + name)
                    properties[name] = (member, kind)
                elif type(member) is ClassMethodDescriptorType and member.__objclass__ is object:
                    plan['bindings'][name] = (member, member.__get__)
                elif _resolved(type(member), '__get__', context['missing']) is not context['missing'] and name not in plan['bindings']:
                    raise TypeError('unconnected descriptor: ' + name)
            plan['properties'] = properties
            eligible.append((cls, plan))
        except TypeError as error:
            context['pending'][cls] = str(error)
    # Each candidate gets its own slot. Ineligible built-in subclasses retain
    # the old source slot explicitly if an eligible ancestor is also installed.
    for cls in candidates:  # glue: preserve visible unported inherited paths
        if cls in context['pending'] and _resolved(cls, '__getattribute__', context['missing']) in (original, context['descriptor']):
            cls.__getattribute__ = original
    for cls, plan in eligible:  # glue: install only complete native descriptor plans
        cls.__getattribute__ = context['descriptor']
        plan['bindings']['__getattribute__'] = (context['descriptor'], context['descriptor'].__get__)
        context['plans'][cls] = plan
