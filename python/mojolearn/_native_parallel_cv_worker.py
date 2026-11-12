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

# Lazy family import (lane lazy-family-import): a worker runs on the loaded
# set the package had before it, every family and unit, as its parent may hand
# it any of them.
from . import _families as _worker_families
_worker_families.load_all()
from . import model_selection as _selection

_selection._NATIVE(None, 'family_msel_parallel_worker', (sys.argv,), _selection._NATIVE_CONTEXT)
