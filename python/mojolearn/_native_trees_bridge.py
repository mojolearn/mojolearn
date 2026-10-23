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

#: The tree bindings the owner launches, by numeric mode.
_TREE_BINDINGS = ("_mojolearn_rf", "_mojolearn_trees", "_mojolearn_x_trees", "_mojolearn_svm")
#: The base binding's input and label helpers (`_buffer._native`), always the
#: identical binary (they are byte moves and comparisons with no tier).
_BASE_HELPERS = ("cast_f64_to_f32", "cast_colmajor_f64_to_f32", "transpose_f32", "all_finite_f32",
                 "all_finite_f64", "prepare_targets", "bincount_i64", "class_ratio_f64",
                 "gather_rows_bytes", "argmax_rows_f32", "argmax_rows_f64", "gather_i64", "gather_f64",
                 "unique_inverse", "reduce_stat", "cast_elements")


def _unavailable(error):
    # glue: an install without the compiled owner keeps the package
    # importable; the tree methods then explain what is missing.
    message = f"mojolearn: the tree ensembles need the compiled host binding: {error}"

    def unavailable(*args):
        raise ModuleNotFoundError(message, name=getattr(error, "name", None))
    return unavailable


def _entries(module):
    # glue: retain the binding's compiled entry points once at connection
    from types import BuiltinFunctionType, ModuleType
    if type(module) is not ModuleType:
        # A missing tier or a refusing stub: touching it states the reason.
        try:
            getattr(module, "trees_owner_probe")
        except Exception as error:  # noqa: BLE001 - recorded, raised at the first call
            return f"{type(error).__name__}: {error}"
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
                per_mode[name] = f"{type(error).__name__}: {error}"
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
            table[name] = f"{type(error).__name__}: {error}"
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
    from . import _backend
    _backend.default_mode()  # glue: settle the process default the native mode falls back to
    context.update(
        trees_bindings=_bindings(), trees_helpers=_helpers(), trees_classes={}, trees_defaults={},
        trees_missing=object(),
        trees_real_type=numbers.Real, trees_integral_type=numbers.Integral,
    )
    _ENTRY, _CONTEXT = entry, context
    return _ENTRY, _CONTEXT


def register(cls, kind, **tables):
    """Connect one public tree class at its definition: `kind` names the
    native owner's schema for it ("rf_classifier", "et_regressor", ...);
    `tables` are its fixed lookup tables (criteria, refusal texts)."""
    _, context = connect()
    if "trees_unavailable" in context:
        return
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
