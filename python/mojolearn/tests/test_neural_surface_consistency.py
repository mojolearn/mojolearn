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


def test_host_trainer_optimizer_defaults_match_the_gpu_trainer():
    # LanguageModelHostTrainer is the CPU twin of the GPU byte LM trainer. With
    # weight_decay defaulting to 0.0 on the CPU and 0.01 on the GPU, the same
    # parameters and ids gave two different first steps at library defaults.
    names = ("lr", "betas", "eps", "weight_decay")
    host = _defaults(_byte_lm_host.LanguageModelHostTrainer.__init__, names)
    gpu = _defaults(_byte_lm_impl.SmallByteLanguageModelTrainer.__init__, names)
    assert host == gpu


# test_mlp_batch_refuses_a_list_by_type_not_by_shape exercised the dead Python
# `_mlp_impl._batch`; the refusal it pinned ("SmallMLPTrainer X must be a
# float32 array of shape (batch, 8) ... got list") is the native owner's
# (bindings/small_mlp_native_door.mojo `_batch_in`), lane py-lm-neural.
