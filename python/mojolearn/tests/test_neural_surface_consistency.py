# SPDX-License-Identifier: Apache-2.0
"""Host-only consistency checks for the neural surfaces (pip smoke, 2026-09-22).

No GPU and no native binding: these read signatures and the pure-Python input
checks only.
"""
import inspect

import pytest

from mojolearn import _byte_lm_host, _byte_lm_impl


def _defaults(fn, names):
    params = inspect.signature(fn).parameters
    return {n: params[n].default for n in names}


# TOMBSTONE: test_host_trainer_optimizer_defaults_match_the_gpu_trainer deleted 2026-11-22 by lane
# neural-consolidate: LanguageModelHostTrainer is gone; CPU training is the GPU trainer's own class
# (device="cpu"), so there is one set of defaults.


# test_mlp_batch_refuses_a_list_by_type_not_by_shape exercised the dead Python
# `_mlp_impl._batch`; the refusal it pinned ("SmallMLPTrainer X must be a
# float32 array of shape (batch, 8) ... got list") is the native owner's
# (bindings/small_mlp_native_door.mojo `_batch_in`), lane py-lm-neural.
