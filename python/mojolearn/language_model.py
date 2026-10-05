# SPDX-License-Identifier: Apache-2.0
"""Configured FP32 decoder language-model training and compatibility aliases.

`LanguageModelTrainer(parameters, *, data_schedule, lr, betas, eps,
weight_decay, shape=None, resident=None, step_result=None)`. On a GPU
install `resident` defaults to True and the state lives on the device between
steps (DEVIATION 2514; default since cpu2-l11-neural); `resident=False` is the
stateless path (the CPU-only install's route). `train_step` returns the full
dict or, with `step_result='lean'`,
only the loss, step and flags; `export_state()`, `export_gradients()` and
`export_checkpoint()` copy the device state out on demand.

`LanguageModelInference.from_checkpoint(path)` runs the same model forward on
the CPU with no GPU (DEVIATION 2610); docs/BYTE_LM_CPU_INFERENCE.md lists the
CPUs it is certified on.

`LanguageModelHostTrainer(parameters, *, m, v, completed_steps, lr, betas, eps,
weight_decay)` runs ONE TRAINING STEP on the CPU with no GPU (DEVIATION 2680):
forward, backward and the AdamW update. `train_step(ids)` returns the loss's
IEEE-754 bits and exposes `gradient_`, `parameters_`, `m_` and `v_`.
docs/BYTE_LM_CPU_TRAINING.md records what it is certified against, which is
every one of the 128 steps of the retained three-vendor capture, byte for byte,
on seven CPUs. Reference path only, one profile, one batch shape; identity is
per shape because the weight gradients contract over the token count.
"""
from ._byte_lm_config import ByteLanguageModelConfig
from ._byte_lm_impl import SmallByteLanguageModelTrainer

# The two CPU classes live in the CPU-side `_byte_lm_host` and load on first
# touch (cpu-gpu-cleanup n-pyneural, 2026-10-02), so a GPU install importing
# this module pulls in no CPU-side module.
_CPU_SIDE = {'LanguageModelHostTrainer': '._byte_lm_host',
             'LanguageModelInference': '._byte_lm_host'}

LanguageModelConfig = ByteLanguageModelConfig
LanguageModelTrainer = SmallByteLanguageModelTrainer

__all__ = ['LanguageModelConfig', 'LanguageModelTrainer', 'LanguageModelInference',
           'LanguageModelHostTrainer', 'SmallByteLanguageModelTrainer',
           'ByteLanguageModelConfig']


def __getattr__(name):
    if name in _CPU_SIDE:
        import importlib
        value = getattr(importlib.import_module(_CPU_SIDE[name], __package__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
