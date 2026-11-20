# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for the complete native numeric-profile operations."""
from .tokenizer import _binding

# Imported by the package after its native imports exist: the compiled entry is
# bound once here; connect() supplies the context to `_numeric_profile`, whose
# functions are each one call into it.
_CALL = _binding()._module.numeric_profile_native_call


def connect(package):
    # glue: bind the compiled entry and its context once, after the backend
    # metadata connection (the speed warning reads its vendor natively)
    import os
    from types import BuiltinFunctionType
    from . import _native_backend_public, _numeric_profile
    if type(_CALL) is not BuiltinFunctionType:
        raise TypeError("numeric profiles require the compiled native owner")
    if type(_native_backend_public._CONTEXT) is not dict:
        raise TypeError("numeric profiles require the connected native backend metadata")
    _numeric_profile._CONTEXT = dict(
        fields=vars(_numeric_profile), defaults=_numeric_profile._defaults,
        os_fields=vars(os), environ=os.environ, environ_fields=vars(os.environ),
        speed_warning=_numeric_profile.NumericProfileSpeedWarning,
        backend_context=_native_backend_public._CONTEXT)
    _numeric_profile._CALL = _CALL
