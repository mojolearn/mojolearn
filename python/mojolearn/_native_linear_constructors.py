# SPDX-License-Identifier: Apache-2.0
"""Import-time schemas for the field-only native linear constructors."""
from types import BuiltinFunctionType
from .tokenizer import _binding
from . import _expansion_linear as linear

_ENTRY = _binding()._module.linear_constructor_native_call
if type(_ENTRY) is not BuiltinFunctionType:
    raise TypeError('linear constructors require a compiled native entry')
_CONTEXT = dict(plans={}, missing=object(), ready=False)
_CLASSES = (
    linear.SGDClassifier, linear.SGDRegressor, linear.PoissonRegressor, linear.GammaRegressor,
    linear.TweedieRegressor, linear.HuberRegressor, linear.BayesianRidge, linear.ARDRegression,
    linear.Lars, linear.LassoLars, linear.QuantileRegressor, linear.Perceptron,
    linear.PassiveAggressiveClassifier, linear.PassiveAggressiveRegressor, linear.RidgeClassifier,
    linear.RidgeCV, linear.LassoCV, linear.ElasticNetCV, linear.LogisticRegressionCV,
)


def _initializer():
    def __init__(self, *args, **kwargs):
        return _ENTRY(self, args, kwargs, _CONTEXT)
    return __init__


def install():
    # glue: install before every class-layout snapshot. Signature/default
    # metadata is copied at import; no original constructor is retained/called.
    import inspect
    for cls in _CLASSES:  # glue: exact vetted field-only constructor declarations
        original = vars(cls)['__init__']
        signature = inspect.signature(original)
        parameters = tuple(signature.parameters.values())[1:]
        if any(p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) for p in parameters):  # glue: validate import-time constructor metadata
            raise TypeError('linear native constructor schema requires declared ordinary parameters')
        function = _initializer()
        function.__name__ = '__init__'
        function.__qualname__ = original.__qualname__
        function.__module__ = original.__module__
        function.__doc__ = original.__doc__
        function.__signature__ = signature
        cls.__init__ = function
        names = tuple(p.name for p in parameters)  # glue: ordered public constructor schema
        # These original bodies write a field before the other signature
        # parameters. Preserve observable instance-dictionary insertion order.
        first = ('power', 'link') if cls is linear.TweedieRegressor else (
            ('batch_size',) if cls in (linear.Perceptron, linear.PassiveAggressiveClassifier,
                                      linear.PassiveAggressiveRegressor) else ())
        order = first + tuple(name for name in names if name not in first)  # glue: fixed field assignment order
        _CONTEXT['plans'][cls] = dict(
            names=names, order=order,
            positional=tuple(p.name for p in parameters if p.kind is p.POSITIONAL_OR_KEYWORD),  # glue: native argument schema
            defaults={p.name: p.default for p in parameters if p.default is not p.empty},  # glue: borrow exact defaults
            label=original.__qualname__, function=function)


def connect(setter_context):
    # glue: share the vetted constructor/storage and reference-release schemas
    # after CV and setters have captured their final native method identities.
    from ._native_params_setter import _CLASSES as names
    from ._native_public_attributes import _resolved
    if tuple(cls.__name__ for cls in _CLASSES) != names:  # glue: exact supported subset
        raise TypeError('linear constructor and setter schemas differ')
    for cls in _CLASSES:  # glue: fixed post-install layout snapshots
        owner = setter_context['plans'][cls]
        constructor = owner['constructor']
        plan = _CONTEXT['plans'][cls]
        if (constructor['init'] is not None or constructor['forest'] is not None
                or constructor.get('initial_fields') or tuple(constructor['parameters']) != plan['names']):
            raise TypeError('linear constructor requires its vetted field-only parameter schema')
        plan['field_hooks'] = tuple((base, name, vars(base).get(name))
            for base in constructor['mro'] for name in plan['names'])  # glue: refuse changed parameter descriptors before runtime writes
        plan.update(mro=constructor['mro'], layout_hooks=constructor['hooks'], protected=constructor['protected'],
            inherited_hooks=tuple((name, _resolved(cls, name, _CONTEXT['missing']))
                for name in constructor['protected']))  # glue: inherited constructor/storage identities, allowing passive subclasses
    _CONTEXT.update(setters=setter_context, ready=True)
