# SPDX-License-Identifier: Apache-2.0
"""Import-time native attribute slots for classes with connected descriptors."""
from types import BuiltinFunctionType, ClassMethodDescriptorType, FunctionType, MethodType, ModuleType

_CONTEXT = None


def _resolved(cls, name, missing):
    # glue: class dictionaries, without invoking a descriptor or getter
    return next((vars(base)[name] for base in type.__dict__['__mro__'].__get__(cls)  # glue: import-time native MRO metadata
                 if name in vars(base)), missing)


def _archive_finalizers(package):
    # glue: only existing archive-owner connections, never optional imports or
    # arbitrary native callable/descriptor admission.
    import sys
    archives = []
    for suffix, key in (('._native_archive_bridge', '_CONTEXT'), ('._classical_host', '_CLASSICAL')):  # glue: the two fixed archive owners
        module = sys.modules.get(package.__name__ + suffix)
        if type(module) is ModuleType:
            owner = vars(module).get(key)
            archive = owner.get('host_archive_context') if type(owner) is dict else None
            if type(archive) is dict:
                archives.append(archive)
    return archives


def _archive_finalizer_binding(cls, member, context, archives):
    # glue: PyInstanceMethod binding itself is native. Admit only the exact
    # archive finalizer installed for this class (or its registered ancestor),
    # with the known compiled entry and the original owner's context.
    if type(member) is not type(context['descriptor']):
        return None
    for archive in archives:  # glue: exact registered native lifetime ownership
        if member is not archive.get('host_archive_finalizer_descriptor'):
            continue
        bound = member.__func__
        layouts = archive.get('host_archive_lifetime_layouts')
        if (type(bound) is MethodType and bound.__func__ is context['archive_finalizer_entry']
                and bound.__self__ is archive and type(layouts) is dict
                and any(base in layouts for base in type.__dict__['__mro__'].__get__(cls))):  # glue: same native finalizer resolution as its owner
            return member, member.__get__
    return None


def _prep_availability(package, missing):
    # glue: both declared-parameter families use the same built-in
    # hasattr(transform) rule. Pin each exact predicate; no arbitrary
    # available_if callback or other predicate is admitted by this schema.
    import sys
    result = {}
    for module_name, base_name, predicates in (  # glue: two fixed import-time class families
            ('_expansion_prep', '_PrepBase', ('_PrepBase', '_Classifier')),
            ('_expansion_decomp', '_Base', ('_Base', 'MinCovDet', 'AlternatingLeastSquares'))):
        module = sys.modules.get(package.__name__ + '.' + module_name)
        if type(module) is not ModuleType:
            continue
        declarations = vars(module)
        getter = vars(declarations[base_name])['get_params']
        checks = tuple(vars(vars(declarations[name])['fit_transform'])['check']  # glue: exact built-in predicate identities
                       for name in predicates)  # glue: fixed import-time predicate declarations
        result.update({declarations[name]: checks for name in declarations['__all__']  # glue: exact public class schema
                       if type(declarations[name]) is type
                       and _resolved(declarations[name], 'get_params', missing) is getter})
    return result


def _availability_plan(member, checks, missing):
    # glue: snapshot descriptor data without calling the availability check,
    # its __get__, or the wrapped function. Runtime validates these identities.
    fields = vars(member)
    check, function = fields.get('check'), fields.get('fn')
    if type(check) is not FunctionType or not any(check is expected for expected in checks):  # glue: built-in predicate identity only
        raise TypeError('unconnected availability predicate')
    if type(function) is not FunctionType:
        raise TypeError('unconnected availability method')
    return dict(member=member, check=check, code=check.__code__, function=function,
                attribute_name=fields.get('attribute_name', missing),
                mro=type.__dict__['__mro__'].__get__(type(member)),
                hooks=tuple((name, _resolved(type(member), name, missing))  # glue: descriptor protocol identities
                            for name in ('__get__', '__set__', '__delete__', '__getattribute__', '__getattr__')))  # glue: fixed import-time protocol metadata


def _neighbor_availability(package):
    # glue: the four fixed LOF declarations are data at import; runtime
    # evaluates the native novelty word without invoking these predicates.
    import sys
    module = sys.modules.get(package.__name__ + '._expansion_neighbors')
    if type(module) is not ModuleType:
        return {}
    cls = vars(module).get('LocalOutlierFactor')
    if type(cls) is not type:
        return {}
    result = {}
    for name in ('fit_predict', 'score_samples', 'decision_function', 'predict'):  # glue: fixed public availability declarations
        member = vars(cls)[name]
        condition = 'not_novelty' if name == 'fit_predict' else 'novelty'
        result[name] = ((vars(member)['check'],), condition)
    return {cls: result}


def _hierarchy_availability(package):
    # glue: pin the hierarchy prediction predicate without invoking it.
    import sys
    module = sys.modules.get(package.__name__ + '._hierarchy_impl')
    if type(module) is not ModuleType:
        return {}
    cls = vars(module).get('AgglomerativeClustering')
    if type(cls) is not type:
        return {}
    member = vars(cls)['predict']
    return {cls: {'predict': ((vars(member)['check'],), 'hierarchy_predict')}}


def _density_availability(package):
    # glue: the declared DBSCAN predicate is frozen import metadata only.
    import sys
    module = sys.modules.get(package.__name__ + '.density')
    if type(module) is not ModuleType:
        return {}
    cls = vars(module).get('DBSCAN')
    if type(cls) is not type:
        return {}
    member = vars(cls)['predict']
    return {cls: {'predict': ((vars(member)['check'],), 'density_predict')}}


def _linear_spectral_availability(package):
    # glue: fixed availability declarations, captured once without calling predicates
    import sys
    result = {}
    for module_name, class_name, member_name, condition in (  # glue: two native availability schemas
            ('_expansion_linear', 'SGDClassifier', 'predict_proba', 'sgd_probability'),
            ('_spectral_impl', 'SpectralClustering', 'predict', 'spectral_predict')):
        module = sys.modules.get(package.__name__ + '.' + module_name)
        if type(module) is ModuleType:
            cls = vars(module)[class_name]
            member = vars(cls)[member_name]
            result[cls] = {member_name: ((vars(member)['check'],), condition)}
    return result


def _inherited_host_availability(package, declared, missing):
    # glue: only archive-registered package host classes inherit these rules.
    # The method descriptor must be the exact admitted base descriptor; a
    # replacement predicate, wrapper or application subclass gains no schema.
    import sys
    module = sys.modules.get(package.__name__ + '._classical_host')
    if type(module) is not ModuleType:
        return {}
    result = {}
    for classes in vars(module).get('_FORMATS', {}).values():  # glue: registered archive classes at package import
        for cls in classes.values():  # glue: fixed Host class setup inventory
            if type(cls) is not type or vars(cls).get('__module__') != module.__name__:
                continue
            methods = {}
            for base in cls.__mro__[1:]:  # glue: inherited descriptor identities captured at setup
                for name, rule in declared.get(base, {}).items():  # glue: fixed availability schema declarations
                    if (name not in methods
                            and _resolved(cls, name, missing) is _resolved(base, name, missing)):
                        methods[name] = rule
            if methods:
                result[cls] = methods
    return result


def connect_classes(package):
    # glue: establish native slots once, before dependent estimator/parameter
    # tables capture method identities; lazy imports admit only new classes.
    global _CONTEXT
    import sys
    from . import _mode, _cholesky_impl, _svm_impl
    from ._native_classical_bridge import _layout, _numpy_free_context
    from .tokenizer import _binding

    if _CONTEXT is None:
        module = _binding()._module
        entry = module.public_attribute_native_call
        factory = module.classical_attribute_descriptor
        finalizer = module.host_archive_finalize_call
        if any(type(value) is not BuiltinFunctionType for value in (entry, factory, finalizer)):  # glue: fixed native entry identities
            raise TypeError('public attributes require compiled native entries')
        context = _numpy_free_context()
        context.update(native_module_type=ModuleType, plans={}, pending={}, missing=object(),
                       property_schemas={vars(_cholesky_impl.Cholesky)['logdet_']: 'cholesky_logdet',
                                         vars(_svm_impl.SVC)['coef_']: 'svm_svc_coef',
                                         vars(_svm_impl.SVR)['coef_']: 'svm_svr_coef'},
                       svm_owner=(_svm_impl._NATIVE, _svm_impl._NATIVE_CONTEXT),
                       stl_connected=False, archive_finalizer_entry=finalizer)
        context['descriptor'] = factory(MethodType(entry, context))
        _CONTEXT = context
        # The generic slot (_mode.PublicOutputsMixin.__getattribute__) uses
        # the same Array/NumPy admission metadata.
        _mode._GENERIC_ATTRIBUTE_CONTEXT['attributes'] = context
        # Native hierarchy/cluster/kernel lifecycle admission consumes this
        # same context through the canonical module registry. Connect after
        # its one-time initialization; lazy class discovery mutates its plans
        # in place, while replacing the owner must still refuse at native use.
        from ._native_loaded_modules import connect_module
        connect_module(__name__, sys.modules[__name__], ('_CONTEXT',))
    context = _CONTEXT
    from ._native_mode_properties import connect as connect_mode_properties
    connect_mode_properties(context, package)
    archives = _archive_finalizers(package)
    prep_availability = _prep_availability(package, context['missing'])
    neighbor_availability = _neighbor_availability(package)
    neighbor_availability.update(_hierarchy_availability(package))
    neighbor_availability.update(_density_availability(package))
    neighbor_availability.update(_linear_spectral_availability(package))
    neighbor_availability[_svm_impl.SVC] = {
        name: ((vars(vars(_svm_impl.SVC)[name])['check'],), 'svm_probability')
        for name in ('predict_proba', 'predict_log_proba')  # glue: two fixed native availability schemas
    }
    neighbor_availability.update(_inherited_host_availability(
        package, neighbor_availability, context['missing']))
    stl_module = sys.modules.get(package.__name__ + '._x_sequence_stl')
    if not context['stl_connected'] and type(stl_module) is ModuleType:
        context['property_schemas'][vars(vars(stl_module)['STL'])['config']] = 'stl_config'
        context['stl_connected'] = True
    boost_module = sys.modules.get(package.__name__ + '._expansion_trees')
    if type(boost_module) is ModuleType:
        bagging = vars(boost_module).get('_BaggingBase')
        if type(bagging) is type:
            context['property_schemas'][vars(bagging)['estimators_']] = 'bagging_estimators'
    candidates = set()
    for name, module in tuple(sys.modules.items()):  # glue: loaded product declarations only; no optional module imports
        if (name == package.__name__ or name.startswith(package.__name__ + '.')) and not name.startswith(package.__name__ + '.tests'):
            if type(module) is ModuleType:
                # Raw declared metadata: native types can lack __module__, and
                # descriptors or non-string metadata must not run protocols.
                candidates.update(value for value in vars(module).values()  # glue: exact product class identities
                                  if type(value) is type
                                  and type(owner := vars(value).get('__module__')) is str
                                  and owner.startswith(package.__name__ + '.')
                                  and not owner.startswith(package.__name__ + '.tests'))
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
            # The fallback getter is now a known native descriptor, so the
            # generic FunctionType admission in _layout no longer covers it.
            # Admit only this import-connected identity before examining names.
            if _resolved(cls, '__getattribute__', context['missing']) is original:
                plan['bindings']['__getattribute__'] = (original, original.__get__)
            names = {}
            for base in plan['mro']:  # glue: descriptor precedence is frozen at connection
                for name, member in vars(base).items():  # glue: exact class namespace metadata
                    names.setdefault(name, member)
            properties = {}
            availability = {}
            for name, member in names.items():  # glue: native property schema admission
                finalizer = (_archive_finalizer_binding(cls, member, context, archives)
                             if name == '__del__' else None)
                if type(member) is property:
                    kind = context['property_schemas'].get(member)
                    if kind is None:
                        raise TypeError('unconnected property: ' + name)
                    properties[name] = (member, kind)
                elif finalizer is not None:
                    plan['bindings'][name] = finalizer
                elif (cls in prep_availability and name == 'fit_transform'
                      and type(member) is _mode.available_if):
                    availability[name] = _availability_plan(member, prep_availability[cls], context['missing'])
                elif (cls in neighbor_availability and name in neighbor_availability[cls]
                      and type(member) is _mode.available_if):
                    checks, condition = neighbor_availability[cls][name]
                    availability[name] = _availability_plan(member, checks, context['missing'])
                    availability[name]['condition'] = condition
                elif type(member) is ClassMethodDescriptorType and member.__objclass__ is object:
                    plan['bindings'][name] = (member, member.__get__)
                elif _resolved(type(member), '__get__', context['missing']) is not context['missing'] and name not in plan['bindings']:
                    raise TypeError('unconnected descriptor: ' + name)
            plan['properties'] = properties
            plan['availability'] = availability
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
