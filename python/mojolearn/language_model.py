# SPDX-License-Identifier: Apache-2.0
"""Configured FP32 decoder language-model training and compatibility aliases.

`LanguageModelTrainer(parameters, *, data_schedule, lr, betas, eps,
weight_decay, shape=None, resident=False, step_result='full')`. With
`resident=True` the state lives on the device between steps (DEVIATION
2514): `train_step` returns the full dict or, with `step_result='lean'`,
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
from ._byte_lm_host import LanguageModelHostTrainer, LanguageModelInference
from ._byte_lm_impl import SmallByteLanguageModelTrainer

LanguageModelConfig = ByteLanguageModelConfig
LanguageModelTrainer = SmallByteLanguageModelTrainer

__all__ = ['LanguageModelConfig', 'LanguageModelTrainer', 'LanguageModelInference',
           'LanguageModelHostTrainer', 'SmallByteLanguageModelTrainer',
           'ByteLanguageModelConfig']
