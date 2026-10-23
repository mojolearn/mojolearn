# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the compiled owner of the classical estimator
families' public methods (bindings/family_native.mojo): k-means,
neighbors, kernels, model selection and the rest of lane
migrate-classical-b. A public method of a connected class is one call,
`_NATIVE(self, "family_<op>", args, _NATIVE_CONTEXT)`; the native owner
validates, prepares, runs the family's compiled kernels and stores or
returns the result. Python here only retains the compiled entry and the
shared admission context, once, at import; each module adds its fixed
lookup tables to the context at its own import."""

_ENTRY = None
_CONTEXT = None


def _unavailable(error):
    # glue: an install without the compiled owner keeps the package
    # importable; a connected method then explains what is missing.
    message = f"mojolearn: this estimator needs the compiled host binding: {error}"

    def unavailable(*args):
        raise ModuleNotFoundError(message, name=getattr(error, "name", None))
    return unavailable


def shared_context():
    """Return the family owners' shared admission and metadata context."""
    global _CONTEXT
    if _CONTEXT is None:
        from types import BuiltinFunctionType
        from ._native_classical_bridge import _state_context
        context = _state_context()
        context.update(family_builtin_type=BuiltinFunctionType)
        _CONTEXT = context
    return _CONTEXT


def connect():
    """`(entry, context)`: the compiled family owner and the context every
    connected class shares. Connected once; later calls return the same
    pair."""
    global _ENTRY, _CONTEXT
    if _ENTRY is not None:
        return _ENTRY, shared_context()
    from types import BuiltinFunctionType
    try:
        from .tokenizer import _binding
        entry = _binding()._module.classical_native_call
        if not isinstance(entry, BuiltinFunctionType):
            raise TypeError("the family owner requires the compiled classical_native_call entry")
        context = shared_context()
    except ImportError as error:
        _ENTRY = _unavailable(error)
        if _CONTEXT is None:
            _CONTEXT = {"family_unavailable": str(error)}
        return _ENTRY, _CONTEXT
    _ENTRY, _CONTEXT = entry, context
    return _ENTRY, _CONTEXT


def tables(**fixed):
    """Add one module's fixed lookup tables (its `family_*` names) to the
    shared context, once, at that module's import."""
    context = shared_context()
    context.update(fixed)
