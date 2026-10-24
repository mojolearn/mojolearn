# SPDX-License-Identifier: Apache-2.0
"""Fixed native-type bootstrap followed by one compiled worker-loop call."""
import os
import sys

# glue: the native parent selects one fixed ABI connection before task receipt;
# no Python request interpretation, parameter conversion or task dispatch occurs.
if os.environ.get('MOJOLEARN_NATIVE_CV_NUMPY') == '1':
    # connects existing native scalar/ndarray types (the parent had NumPy)
    from ._optional_numpy import require_numpy
    require_numpy('the parallel cross-validation worker')

from . import model_selection as _selection

_selection._NATIVE(None, 'family_msel_parallel_worker', (sys.argv,), _selection._NATIVE_CONTEXT)
