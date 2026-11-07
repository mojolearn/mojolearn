# SPDX-License-Identifier: Apache-2.0
"""Import-time connection of the classical owner's integer operation codes
(bindings/native_operation_codes.mojo). A shell passes `_OP.<name>` where it
passed the operation name: the code from the one native table, read once
here, so the native door neither copies the name nor matches it against a
chain of names. Nothing here runs when a public method is called."""

_CODES = []


class _Codes:
    """One attribute per operation name, its integer code. A name the table
    does not hold answers as itself (the str route of the same door)."""

    def __init__(self, names):
        self.__dict__.update(zip(names, range(len(names))))  # glue: the fixed operation names, once at import

    def __getattr__(self, name):
        return name


def codes():
    """The process-wide operation codes; read from the compiled door once.
    Without the compiled binding every name answers as itself, so each
    shell's unavailable entry raises its own message."""
    if not _CODES:
        try:
            from .tokenizer import _binding
            names = _binding()._module.classical_native_call(None, "operation_codes", (), {})
        except ImportError:
            names = ()
        _CODES.append(_Codes(names))
    return _CODES[0]
