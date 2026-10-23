# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the native johnson_lindenstrauss_min_dim owner."""


def connect_public():
    """The compiled `jl_native_call` and its (empty) context, connected once
    at import; an install without it keeps the package importable and the
    function explains what is missing."""
    # glue: retain the compiled entry once at import
    from types import BuiltinFunctionType
    try:
        from .tokenizer import _binding
        entry = _binding()._module.jl_native_call
        if not isinstance(entry, BuiltinFunctionType):
            raise ImportError("johnson_lindenstrauss_min_dim requires a compiled jl_native_call")
        return entry, {}
    except (ImportError, AttributeError) as error:
        message = f"mojolearn: johnson_lindenstrauss_min_dim is unavailable: {error}"

        def unavailable(*args):
            raise ModuleNotFoundError(message, name=getattr(error, "name", None))
        return unavailable, {"jl_unavailable": message}
