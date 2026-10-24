# SPDX-License-Identifier: Apache-2.0
"""Fixed native-type bootstrap followed by one compiled worker-loop call."""
import os
import sys

# glue: the native parent selects one fixed ABI connection before task receipt;
# no Python request interpretation, parameter conversion or task dispatch occurs.
if os.environ.get('MOJOLEARN_NATIVE_CV_NUMPY') == '1':
    import numpy  # noqa: F401 - connects existing native scalar/ndarray types

from . import model_selection as _selection

_selection._NATIVE(None, 'family_msel_parallel_worker', (sys.argv,), _selection._NATIVE_CONTEXT)
