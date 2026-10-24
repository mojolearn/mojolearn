# SPDX-License-Identifier: Apache-2.0
"""Import-time native array storage metadata; no product operation runs here."""
from array import array as ARRAY_TYPE, typecodes as _TYPECODES
from types import WrapperDescriptorType
import sys
import warnings

from ._native_loaded_modules import connect_module

# The native allocator uses the public sequence-repeat API only with this exact
# standard-library type. Never connect an overridden Python sequence protocol.
if (type(ARRAY_TYPE) is not type or type(ARRAY_TYPE.__mul__) is not WrapperDescriptorType
        or ARRAY_TYPE.__mul__.__objclass__ is not ARRAY_TYPE):
    raise TypeError("native array storage requires the standard-library array type")

# Fixed one-word zero seeds, independent of workloads. Every operation still
# allocates/copies its own requested storage inside the compiled owner.
ZERO_STORES = {code: ARRAY_TYPE(code, [0])  # glue: fixed native storage type metadata at import
               for code in ("b", "B", "h", "H", "i", "I", "l", "L", "q", "Q", "f", "d")}

# Constructor-parameter cloning and worker records preserve Unicode arrays
# too. Their fixed native owners belong to setup, just like the numeric seeds.
# Python 3.13 added 'w'; 'u' remains available on older supported interpreters.
if 'w' in _TYPECODES:
    ZERO_STORES['w'] = ARRAY_TYPE('w', '\0')
if 'u' in _TYPECODES:
    # This deprecated constructor is our own import-time implementation detail.
    # Match only its exact warning in this private module, including under
    # -Werror, then restore the caller's filter state before any operation.
    # Audit hooks remain enabled throughout setup and every product operation.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',
            message=r"^The 'u' type code is deprecated and will be removed in Python 3\.16$",
            category=DeprecationWarning, module=r'^mojolearn\._native_array_storage$')
        ZERO_STORES['u'] = ARRAY_TYPE('u', '\0')

connect_module(__name__, sys.modules[__name__], ('ARRAY_TYPE', 'ZERO_STORES'))  # glue: canonical import-only native metadata
