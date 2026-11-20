# SPDX-License-Identifier: Apache-2.0
"""Evaluation metrics: the public name of `mojolearn._metrics_impl`.

`import mojolearn.metrics`, `from mojolearn.metrics import accuracy_score` and
`mojolearn.metrics.accuracy_score` all reach the same module object.
"""
# glue: the package used to register the private module under this name at
# import (Sep 22 pip smoke); with per-family import on first use it is
# registered when the name is first imported (lane lazy-family-import).
import sys as _sys
from . import _metrics_impl as _impl

_sys.modules[__name__] = _impl
