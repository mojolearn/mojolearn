# SPDX-License-Identifier: Apache-2.0
"""The CausalLM session lease (lanes causal-lease, causal-doors): a second caller of one model while the lease is
held gets a private session and completes with its single-thread output; it is never refused.

The tests are the wheel's own `mojolearn.tests.test_causal_lease_two_threads` (reused, not copied). The resident
sessions are GPU state, so this file runs with `-m gpu` only; it sets MOJOLEARN_GENERATION_DEVICE=gpu (the
module's explicit opt-in) when a GPU is present and the variable is unset.
"""
import os

import pytest

from _proving import has_gpu, reexport_installed

if has_gpu():
    os.environ.setdefault("MOJOLEARN_GENERATION_DEVICE", "gpu")

reexport_installed(globals(), "mojolearn.tests.test_causal_lease_two_threads")
pytestmark = pytest.mark.gpu
