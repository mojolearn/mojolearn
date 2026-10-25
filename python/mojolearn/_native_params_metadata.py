# SPDX-License-Identifier: Apache-2.0
"""Import-time schemas for native parameter reads and nested built-in values."""
from types import BuiltinFunctionType, ModuleType


def connect():
    from .tokenizer import _binding
    entry = _binding()._module.params_metadata_native_call
    if type(entry) is not BuiltinFunctionType:
        raise TypeError('parameter metadata requires a compiled native entry')
    return entry, dict(plans={}, missing=object(), pending={}, ready=False)


def connect_classes(context, package):
    # glue: fixed schemas are connected after all built-in class installers
    import inspect
    import sys
    from . import _mode, _expansion_decomp, _expansion_prep, preprocessing
    from . import _gbdt_adapters, _forest_protocol, linear_model, model_selection, _training_impl

    def resolved(cls, name):
        # glue: inspect native class namespaces without descriptor invocation
        return next((vars(base)[name] for base in type.__dict__['__mro__'].__get__(cls)  # glue: import-time namespace metadata, no runtime descriptor call
                     if name in vars(base)), context['missing'])

    schemas = {
        _mode.ParamsMixin.get_params: ('signature', True),
        _expansion_decomp._Base.get_params: ('declared', False),
        _expansion_prep._PrepBase.get_params: ('declared', False),
        preprocessing._ScalerProtocol.get_params: ('declared', False),
        preprocessing.StandardScaler.get_params: ('native_declared', False),
        preprocessing.MinMaxScaler.get_params: ('native_declared', False),
        _expansion_prep.RobustScaler.get_params: ('native_declared', False),
        _expansion_prep.MaxAbsScaler.get_params: ('native_declared', False),
        _expansion_prep.LinearDiscriminantAnalysis.get_params: ('native_declared', False),
        linear_model.LinearRegression.get_params: ('signature', False),
        linear_model.Ridge.get_params: ('signature', False),
        linear_model.LogisticRegression.get_params: ('signature', False),
        _forest_protocol.ForestProtocol.get_params: ('forest', False),
        _gbdt_adapters._GBDTAdapter.get_params: ('gbdt', False),
        model_selection._Splitter.get_params: ('splitter', False),
        model_selection._Repeated.get_params: ('repeated', False),
        model_selection._BaseSearch.get_params: ('search', False),
        _training_impl._Optimizer.get_params: ('optimizer', False),
    }
    classes = set()
    for name, module in tuple(sys.modules.items()):  # glue: loaded built-in class identities only
        if name == package.__name__ or name.startswith(package.__name__ + '.'):
            if type(module) is ModuleType:
                classes.update(value for value in vars(module).values() if type(value) is type)  # glue: fixed class declarations
    for cls in classes:  # glue: native parameter schemas, never product-time introspection
        if cls in context['plans'] or cls in context['pending']:
            continue  # Never bless changes to a previously pinned class.
        getter = resolved(cls, 'get_params')
        if getter is context['missing']:
            continue
        if getter not in schemas:
            context['pending'][cls] = 'unconnected built-in parameter getter'
            continue
        kind, recurse = schemas[getter]
        try:
            initializer = resolved(cls, '__init__')
            if recurse and resolved(cls, '_get_param_names') is not resolved(_mode.ParamsMixin, '_get_param_names'):
                raise TypeError('custom Python parameter-name callbacks are not supported by the native schema')
            if kind in ('declared', 'native_declared'):
                names = tuple(resolved(cls, '_parameters'))
            elif kind == 'forest':
                names = tuple(resolved(cls, '_parameter_names'))
            elif kind == 'search':
                names = model_selection._NATIVE_CONTEXT['family_msel_search_layouts'][cls]['parameters']
            elif kind in ('splitter', 'repeated'):
                names = ()
            elif kind == 'optimizer':
                names = (('accumulation_steps', 'dampening', 'lr', 'lr_schedule', 'maximize', 'momentum',
                          'nesterov', 'numeric_mode', 'params', 'resident', 'weight_decay')
                         if issubclass(cls, _training_impl.SGD) else
                         ('accumulation_steps', 'betas', 'eps', 'lr', 'lr_schedule', 'maximize',
                          'numeric_mode', 'params', 'resident', 'weight_decay'))
            else:
                parameters = inspect.signature(initializer).parameters.values()
                names = tuple(p.name for p in parameters if p.name != 'self'  # glue: constructor signature metadata
                              and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))
                if kind == 'signature':
                    if initializer is not object.__init__ and issubclass(cls, _mode.NumericModeMixin) and 'numeric_mode' not in names:
                        names += ('numeric_mode',)
                    names = tuple(sorted(names))  # glue: fixed import-time constructor-name order, never a product-time reduction
            if any(type(name) is not str for name in names):  # glue: native schema field identities
                raise TypeError('parameter names must be exact strings')
            protected = ('__init__', '_get_param_names', 'get_params', '__getattribute__', '__getattr__',
                         '_parameters', '_parameter_names')
            pinned = tuple((name, resolved(cls, name)) for name in protected + names)  # glue: freeze parameter descriptors and method identities
            context['plans'][cls] = dict(kind=kind, recurse=recurse, names=names, pinned=pinned,
                                        class_name=cls.__name__,
                                        fields_only=kind in ('native_declared', 'forest', 'search', 'optimizer') or (kind == 'signature' and not recurse),
                                        missing_error=('KeyError' if kind in ('native_declared', 'search', 'optimizer') else
                                                       'TypeError' if kind == 'signature' else 'AttributeError'),
                                        gbdt_init=_gbdt_adapters._GBDTAdapter.__init__ if kind == 'gbdt' else None)
        except (TypeError, ValueError, KeyError) as error:
            context['pending'][cls] = 'native parameter schema is incomplete: ' + str(error)
    context['ready'] = True
