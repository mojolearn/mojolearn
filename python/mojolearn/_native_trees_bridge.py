# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the compiled tree-ensemble owner.

The public `fit`, `predict`, `predict_proba` and `score` methods of the
forests (RandomForest*, ExtraTrees*, DecisionTree*) and the other tree
ensembles are one call each into the compiled classical owner
(`bindings/classical_native.mojo`, operations `trees_*`, implemented in
`bindings/trees_public_*.mojo`). That native operation validates the
parameters, reads X and y, encodes the labels, checks the weights, chooses
the fit entry and the inference engine, launches the kernels and assigns the
fitted state.

Python here only connects fixed metadata once, at import: the shared
context, the compiled entry points of every tree binding in both numeric
modes, the base binding's input helpers, and the classes the owner serves.
Nothing here runs when a public method is called.
"""

_ENTRY = None
_CONTEXT = None

# Only these schemas use forest_estimator's constructor metadata and the
# trees_public_lifecycle_native envelope. Other registered tree owners retain
# their own native constructor/public-operation connections.
_PUBLIC_FOREST_KINDS = ("rf_classifier", "rf_regressor", "et_classifier", "et_regressor",
                        "dt_classifier", "dt_regressor")

#: The tree bindings the owner launches, by numeric mode.
_TREE_BINDINGS = ("_mojolearn_rf", "_mojolearn_trees", "_mojolearn_x_trees", "_mojolearn_metrics",
                  "_mojolearn_svm", "_mojolearn_gbdt", "_mojolearn_estimators", "_mojolearn_x_neighbors")
#: The base binding's input and label helpers (`_buffer._native`), always the
#: identical binary (they are byte moves and comparisons with no tier).
_BASE_HELPERS = ("cast_f64_to_f32", "cast_colmajor_f64_to_f32", "transpose_f32", "all_finite_f32",
                 "all_finite_f64", "prepare_targets", "bincount_i64", "class_ratio_f64",
                 "gather_rows_bytes", "argmax_rows_f32", "argmax_rows_f64", "gather_i64", "gather_f64",
                 "unique_inverse", "reduce_stat", "cast_elements", "gather_i32", "fsum_f64",
                 "check_indices_i64", "equal_elements")


def _unavailable(error):
    # glue: an install without the compiled owner keeps the package
    # importable; the tree methods then explain what is missing.
    message = f"mojolearn: the tree ensembles need the compiled host binding: {error}"

    def unavailable(*args):
        raise ModuleNotFoundError(message, name=getattr(error, "name", None))
    return unavailable


def _refusal(error):
    # glue: a refusal recorded at import as "<Class>: <message>", the class
    # the native owner raises again (the nearest builtin base the owner's
    # error prefixes name)
    for base in (ImportError, AttributeError, TypeError, RuntimeError, ValueError):  # glue: the fixed refusal classes
        if isinstance(error, base):
            return f"{base.__name__}: {error}"
    return f"RuntimeError: {error}"


def _entries(module):
    # glue: retain the binding's compiled entry points once at connection
    from types import BuiltinFunctionType, ModuleType
    from . import _backend
    if type(module) is _backend._HostBinding:
        # A CPU-only install: the family's compiled host binding stands
        # behind the proxy; its entries are read once here, not through the
        # proxy's Python lookup at call time.
        module = _backend.load_host_module(module._HostBinding__basename)  # cpu-route: a CPU-only install connects its compiled host binding once
    if type(module) is not ModuleType:
        # A missing tier or a refusing stub: touching it states the reason.
        try:
            getattr(module, "trees_owner_probe")
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            return _refusal(error)
        return "ImportError: this binding is not a compiled extension module"
    table = {}
    for name in dir(module):  # glue: the binding's exported names, once at import
        entry = getattr(module, name, None)
        if isinstance(entry, BuiltinFunctionType):
            table[name] = entry
    return table


def _bindings():
    # glue: connect each tree binding of both numeric modes once at import
    from . import _backend
    selected = _backend.default_mode()
    connected = {}
    for mode in (selected, "fast" if selected == "identical" else "identical"):  # glue: the two numeric modes
        per_mode = {}
        for name in _TREE_BINDINGS:  # glue: the fixed tree binding names
            try:
                per_mode[name] = _entries(_backend.binding(name, mode))
            except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
                per_mode[name] = _refusal(error)
        connected[mode] = per_mode
    return connected


def _helpers():
    # glue: the base binding's helpers, resolved once at import
    from ._buffer import _native
    table = {}
    for name in _BASE_HELPERS:  # glue: the fixed helper names
        try:
            table[name] = _native(name)
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            table[name] = _refusal(error)
    return table


def connect():
    """`(entry, context)`: the compiled tree owner and the context every
    connected class shares. Connected once; later calls return the same
    pair."""
    global _ENTRY, _CONTEXT
    if _CONTEXT is not None:
        return _ENTRY, _CONTEXT
    from types import BuiltinFunctionType
    try:
        from .tokenizer import _binding
        from ._native_classical_bridge import _state_context
        module = _binding()._module
        entry = module.classical_native_call
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError("the tree ensembles require the compiled classical_native_call entry")
        context = _state_context()
    except ImportError as error:
        _ENTRY, _CONTEXT = _unavailable(error), {"trees_unavailable": str(error)}
        return _ENTRY, _CONTEXT
    import numbers
    import types
    from ._labels import ClassLabels
    from ._native_labels_bridge import label_context
    # glue: public regressor outputs use the shared native label cast helpers.
    context.update(label_context())
    from . import _backend
    _backend.default_mode()  # glue: settle the process default the native mode falls back to
    context.update(
        trees_bindings=_bindings(), trees_helpers=_helpers(), trees_classes={}, trees_defaults={},
        trees_boost_classes={}, trees_meta_classes={},
        trees_constructors={}, trees_gbdt_adapter_constructors={}, trees_state_layouts={},
        trees_missing=object(), trees_namespace=types.SimpleNamespace,
        trees_real_type=numbers.Real, trees_integral_type=numbers.Integral,
        classifier_labels_class=ClassLabels,
        classifier_labels_namespace=dict(vars(ClassLabels)),
        classifier_labels_mro=type.__dict__["__mro__"].__get__(ClassLabels),
    )
    from ._native_owner_lifetime import connect_owner_context
    connect_owner_context(context)
    tables = tuple(table for modes in context['trees_bindings'].values()  # glue: fixed native binding tables retained independently of model lifetime
                   for table in modes.values() if type(table) is dict)  # glue: exclude import-time refusal strings
    context['trees_state_retained'] = tables
    context['trees_state_releases'] = tuple(table[name] for table in tables  # glue: native forest/GBDT/IsolationForest resident release identities, once at connection
                                           for name in ('forest_release_gpu', 'gbdt_resident_release', 'iforest_resident_release')  # glue: existing compiled release ABIs
                                           if name in table)
    _ENTRY, _CONTEXT = entry, context
    return _ENTRY, _CONTEXT


def bindings(name=None):
    """The connected tree binding tables by numeric mode (for the archive
    owner's loaders), or None without the compiled owner. With `name`, the
    entries of that one binding by numeric mode (what a loader reads as
    `tables[mode][entry]`)."""
    connected = connect()[1].get("trees_bindings")
    if connected is None or name is None:
        return connected
    return {mode: per_mode.get(name) for mode, per_mode in connected.items()}  # glue: the two numeric modes


def connect_cpu_prediction():
    # glue: connect CPU CTR prediction metadata once, after package classes
    # exist. The native tree owner also serves plain GradientBoosting, whose
    # CPU categorical predict needs the same registry as explicit HostGBDT.
    # Waiting for a saved-model API import left that native route unconnected.
    # GPU installs connect it through their existing explicit host API import.
    if _CONTEXT is None or _CONTEXT.get("vendor") != "cpu":
        return
    from . import _forest_host  # noqa: F401 - registers the native host tree/archive context


def register(cls, kind, **tables):
    """Connect one public tree class at its definition: `kind` names the
    native owner's schema for it ("rf_classifier", "et_regressor", ...);
    `tables` are its fixed lookup tables (criteria, refusal texts)."""
    _, context = connect()
    if "trees_unavailable" in context:
        return
    public_forest = kind in _PUBLIC_FOREST_KINDS
    if public_forest and cls not in context["trees_constructors"]:
        raise TypeError("native public forests require constructor metadata before registration")
    import inspect
    context["trees_classes"][cls] = kind
    # The constructor defaults, for an estimator the owner assembles itself
    # (a splitter='random' tree's inner forest).
    signature = inspect.signature(cls.__init__)
    context["trees_defaults"][cls] = {
        name: p.default for name, p in signature.parameters.items()  # glue: the constructor's parameters, once at definition
        if name != "self" and p.default is not inspect.Parameter.empty}
    for key, value in tables.items():  # glue: the class's fixed lookup tables, once at definition
        context.setdefault(key, value)
    if public_forest:
        _connect_public_forest(cls, context)
    elif kind == 'gbdt':
        from ._native_classical_bridge import _layout
        context['trees_state_layouts'][cls] = {
            'classical_public_layouts': {cls: _layout(cls, None)}, 'classical_public_installed': True}
    elif kind == "iforest":
        _connect_public_iforest(cls)


def _connect_public_iforest(cls):
    # glue: import-time removal of the former validation/output decorators;
    # iforest_public_lifecycle owns the complete public operation natively.
    import inspect
    for name in ('fit', 'fit_predict', 'score_samples', 'decision_function', 'predict'):  # glue: fixed native public doors
        setattr(cls, name, inspect.unwrap(getattr(cls, name)))
    # The native archive loader already publishes fitted state; no Python
    # loader callback is needed after the one archive binding call.
    loader = inspect.getattr_static(cls, 'load')
    cls.load = classmethod(inspect.unwrap(loader.__func__))
    cls._mojolearn_native_public = True


def connect_gbdt_parameters(base, classes, metadata):
    # glue: install the native public constructor only after both adapter classes
    # exist, before shared parameter/attribute plans capture their final methods.
    import functools
    import inspect
    from ._native_classical_bridge import _layout
    entry, context = connect()
    original = base.__init__
    signature = inspect.signature(original)

    @functools.wraps(original)
    def initialize(self, *args, **kwargs):
        return entry(self, 'trees_gbdt_adapter_construct', (base, args, kwargs), context)

    initialize.__signature__ = signature
    base.__init__ = initialize

    if 'trees_unavailable' in context:
        return
    rows = tuple((p.name, int(p.kind), p.default is inspect.Parameter.empty,
                  None if p.default is inspect.Parameter.empty else p.default)
                 for p in signature.parameters.values() if p.name != 'self')  # glue: fixed constructor signature
    if any(row[1] not in (0, 1, 3) for row in rows):  # glue: native binder's supported named-argument kinds
        raise TypeError('native GBDT constructors require named parameters')
    for cls in (base,) + tuple(classes):  # glue: built-in adapter storage plans
        layout = _layout(cls, None)
        namespaces = tuple(type.__dict__['__dict__'].__get__(parent)  # glue: raw native class namespaces
                           for parent in type.__dict__['__mro__'].__get__(cls))  # glue: fixed class ancestry
        # Native attribute slots install later; constructor writes never invoke
        # __getattribute__. Setter/dictionary/finalizer protocols stay pinned.
        hooks = tuple((name, next((row[name] for row in namespaces if name in row), None))  # glue: fixed inherited write/lifetime hooks
                      for name in ('__setattr__', '__getattr__', '__del__'))  # glue: no user setter/finalizer callbacks
        context['trees_gbdt_adapter_constructors'][cls] = (rows, layout['fields_get'], hooks)
        context['trees_state_layouts'][cls] = {
            'classical_public_layouts': {cls: layout}, 'classical_public_installed': True}
    context['trees_gbdt_parameter_metadata'] = metadata
    # glue: each adapter's public owner now performs validation, tracking and
    # output conversion; remove only the former package-installed decorators.
    from types import BuiltinFunctionType, MethodType
    from .tokenizer import _binding
    module = _binding()._module
    attribute, factory = module.standard_scaler_attribute_call, module.classical_attribute_descriptor
    if type(attribute) is not BuiltinFunctionType or type(factory) is not BuiltinFunctionType:
        raise TypeError('native GBDT adapters require compiled attribute entries')
    descriptor = factory(MethodType(attribute, context))
    for cls in (base,) + tuple(classes):  # glue: fixed adapter public classes
        for name in ('fit', 'predict', 'predict_proba', 'decision_function', 'score'):  # glue: public native doors
            method = getattr(cls, name, None)
            if inspect.isfunction(method):
                setattr(cls, name, inspect.unwrap(method))
        loader = inspect.getattr_static(cls, 'load', None)
        if isinstance(loader, classmethod):
            cls.load = classmethod(inspect.unwrap(loader.__func__))
        cls.__getattribute__ = descriptor
        cls._mojolearn_native_public = True
        layout = _layout(cls, descriptor)
        context.setdefault('classical_public_layouts', {})[cls] = layout
        context['trees_state_layouts'][cls] = {
            'classical_public_layouts': {cls: layout}, 'classical_public_installed': True}
    context['classical_public_installed'] = True
    context['classical_attribute_descriptor'] = descriptor
    context['trees_gbdt_adapter_public_classes'] = {cls: cls._estimator_type for cls in classes}  # glue: fixed native classifier/regressor identity


def _connect_public_forest(cls, context):
    # Connect the existing native fit/predict owner to the exact built-in
    # classes. Unwrap only the known mode/validation/output decorators: their
    # semantics now live in trees_public_lifecycle_native, not Python frames.
    rows, getter, previous = context['trees_constructors'][cls]
    import inspect
    from types import BuiltinFunctionType, MethodType
    from .tokenizer import _binding
    from ._native_classical_bridge import _layout
    for name in ('fit', '_fit_with_tree_start', 'predict', 'predict_proba', 'score', 'apply', 'predict_with_leaves', 'staged_predict'):  # glue: fixed public tree doors
        method = getattr(cls, name, None)
        if inspect.isfunction(method):
            setattr(cls, name, inspect.unwrap(method))
    module = _binding()._module
    attribute = module.standard_scaler_attribute_call
    factory = module.classical_attribute_descriptor
    if type(attribute) is not BuiltinFunctionType or type(factory) is not BuiltinFunctionType:
        raise TypeError('native forests require compiled attribute entries')
    descriptor = factory(MethodType(attribute, context))
    cls.__getattribute__ = descriptor
    cls._mojolearn_native_public = True
    context.setdefault('classical_public_layouts', {})[cls] = _layout(cls, descriptor)
    context['classical_public_installed'] = True
    context['classical_attribute_descriptor'] = descriptor
    # Constructor metadata was captured before the native attribute slot was
    # installed. Retain its final hook identities without runtime inspection.
    ancestry = type.__dict__['__mro__'].__get__(cls)
    namespaces = tuple(type.__dict__['__dict__'].__get__(base) for base in ancestry)  # glue: final connected storage layout
    hooks = tuple((name, next((row[name] for row in namespaces if name in row), None))  # glue: import-time inherited hook lookup
                  for name, _ in previous)  # glue: fixed final constructor hook identities
    context['trees_constructors'][cls] = (rows, getter, hooks)


def register_boost(cls, kind, **tables):
    """Connect one bagged or boosted tree ensemble class at its definition:
    `kind` names its schema in bindings/trees_public_boost.mojo
    ("bagging_classifier", "ada_regressor", "dart_classifier", "rte", ...);
    `tables` are its fixed lookup tables and the member classes it builds."""
    _, context = connect()
    if "trees_unavailable" in context:
        return
    context["trees_boost_classes"][cls] = kind
    for key, value in tables.items():  # glue: the class's fixed lookup tables, once at definition
        context.setdefault(key, value)


def register_meta(cls, kind):
    """Connect one meta-estimator class at its definition: `kind` names its
    schema in bindings/trees_public_meta.mojo ("voting_classifier",
    "stacking_regressor", "multioutput_classifier", "ovr", "calibrated", ...)."""
    _, context = connect()
    if "trees_unavailable" in context:
        return
    context["trees_meta_classes"][cls] = kind
    # These classes already completed their constructor/numeric-mode setup.
    # Replace the old fit/validation/output decorators only on these public
    # methods; trees_public_meta_lifecycle now owns their runtime semantics.
    # Keep constructors and the still-unported parameter setter visible.
    import inspect
    for name in ('fit', 'fit_transform', 'predict', 'predict_proba', 'transform', 'score'):  # glue: fixed public meta-estimator doors
        method = getattr(cls, name, None)
        if inspect.isfunction(method):
            setattr(cls, name, inspect.unwrap(method))
    cls._mojolearn_native_public = True
