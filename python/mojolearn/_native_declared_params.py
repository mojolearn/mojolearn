# SPDX-License-Identifier: Apache-2.0
"""Import-time plans for shallow, state-preserving parameter assignments."""
from types import BuiltinFunctionType
from .tokenizer import _binding

ENTRY = _binding()._module.declared_params_native_call
if type(ENTRY) is not BuiltinFunctionType:
    raise TypeError('declared parameter assignment requires a compiled native entry')
CONTEXT = dict(ready=False, plans={}, missing=object())


def connect(package, estimator_context, parameter_metadata):
    # glue: immutable built-in schemas, storage hooks and native owner identities
    # Re-run when a family loads (lane lazy-family-import): plans are added
    # for the classes of the LOADED declared families once; the ownership
    # context is connected once; the retained owners grow with the modules.
    import sys
    from types import ModuleType
    from ._native_classical_bridge import _numpy_free_context
    from ._native_owner_lifetime import connect_owner_context
    from ._native_public_attributes import _archive_finalizers, _archive_finalizer_binding, _CONTEXT as attributes

    missing = CONTEXT['missing']

    def resolved(cls, name):
        # glue: raw namespace lookup during import, without descriptor calls
        return next((vars(base)[name] for base in type.__dict__['__mro__'].__get__(cls)  # glue: fixed class ancestry metadata
                     if name in vars(base)), missing)

    plans = CONTEXT['plans']  # filled in place; native callers keep reading this table
    hooks = ('set_params', '__setattr__', '__getattribute__', '__getattr__', '__dict__', '_parameters')
    prep = sys.modules.get(__package__ + '._expansion_prep')
    decomp = sys.modules.get(__package__ + '._expansion_decomp')
    families = tuple((module, base, style) for module, base, style in (
        (prep, '_PrepBase', 'prep'), (decomp, '_Base', 'decomp')) if module is not None)
    for module, base, style in families:  # glue: two explicit shallow-assignment contracts
        base = vars(module)[base]
        for name in module.__all__:  # glue: public built-in class connection
            cls = vars(module)[name]
            if type(cls) is not type or cls in plans or resolved(cls, 'set_params') is not vars(base)['set_params']:
                continue
            metadata = parameter_metadata['plans'][cls]
            if metadata['kind'] != 'declared' or metadata['recurse']:
                raise TypeError('shallow parameter assignment requires a declared native schema')
            if resolved(cls, '__setattr__') is not object.__setattr__:
                raise TypeError('shallow parameter assignment requires ordinary native instance storage')
            plans[cls] = dict(names=metadata['names'], style=style,
                              pinned=tuple((key, resolved(cls, key)) for key in hooks))  # glue: exact setter/storage identities

    ownership = CONTEXT.get('ownership')
    if ownership is None:
        ownership = _numpy_free_context()
        ownership['native_module_type'] = ModuleType
        connect_owner_context(ownership)
    archives = _archive_finalizers(package)
    release_plans = CONTEXT.get('release_plans', {})
    for cls, plan in estimator_context['family_msel_estimators'].items():  # glue: existing native built-in composition layouts
        if cls in release_plans:
            continue
        finalizer = resolved(cls, '__del__')
        if any(vars(base).get('__slots__') for base in plan['mro']):  # glue: no unenumerated slot owners
            continue
        if finalizer is not missing and _archive_finalizer_binding(cls, finalizer, attributes, archives) is None:
            continue
        release_plans[cls] = dict(constructor=plan, finalizer=finalizer)

    # These owners are independently pinned by this import-time context, so a
    # parameter replacement cannot destroy them or run a finalizer/weakref hook.
    # This includes native resident binding modules and the connected score
    # functions already supported by SelectKBest, not arbitrary user callbacks.
    retained = tuple(value for value in tuple(sys.modules.values()) if type(value) is ModuleType)  # glue: loaded native module identities
    if prep is not None:
        retained += tuple(prep._ENCODE_CONTEXT['prep_encode_scores'])
    retained += tuple(estimator_context['family_msel_estimators'])
    CONTEXT.update(plans=plans, parameter_metadata=parameter_metadata,
                   ownership=ownership, release_plans=release_plans,
                   retained_owners=retained, ready=True)
