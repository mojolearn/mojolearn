# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the complete native public numeric-profile operations."""
_CALL = None
_CONTEXT = None


def _profiles():
    return _CALL("profiles", (), _CONTEXT)


def _measured(profile, vendor=None):
    return _CALL("measured", (profile, vendor), _CONTEXT)


def _default_profile(use="inference"):
    return _CALL("default", (use,), _CONTEXT)


def _set_default_profile(name, use="inference"):
    return _CALL("set_default", (name, use), _CONTEXT)


def connect(package):
    # glue: replace the bootstrap definitions only after all native imports exist
    global _CALL, _CONTEXT
    import os
    from types import BuiltinFunctionType
    from . import _numeric_profile
    from .tokenizer import _binding
    entry = _binding()._module.numeric_profile_native_call
    if type(entry) is not BuiltinFunctionType:
        raise TypeError("numeric profiles require the compiled native owner")
    _CALL = entry
    _CONTEXT = dict(fields=vars(_numeric_profile), defaults=_numeric_profile._defaults,
                    os_fields=vars(os), environ=os.environ, environ_fields=vars(os.environ))
    for name, public, replacement in (  # glue: fixed public profile aliases installed once
        ("profiles", "numeric_profiles", _profiles),
        ("measured", "numeric_profile_measured", _measured),
        ("default_profile", "numeric_profile", _default_profile),
        ("set_default_profile", "set_numeric_profile", _set_default_profile),
    ):
        original = vars(_numeric_profile)[name]
        replacement.__doc__ = original.__doc__
        replacement.__name__ = replacement.__qualname__ = name
        replacement.__module__ = _numeric_profile.__name__
        setattr(_numeric_profile, name, replacement)
        setattr(package, public, replacement)
