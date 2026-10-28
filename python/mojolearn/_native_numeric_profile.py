# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the complete native public numeric-profile operations."""
from .tokenizer import _binding

# Imported by the package after its native imports exist: the compiled entry is
# bound once here; connect() supplies the context and installs the public names.
_CALL = _binding()._module.numeric_profile_native_call
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
    global _CONTEXT
    import os
    from types import BuiltinFunctionType
    from . import _numeric_profile
    if type(_CALL) is not BuiltinFunctionType:
        raise TypeError("numeric profiles require the compiled native owner")
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
