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
"""
from ._byte_lm_config import ByteLanguageModelConfig
from ._byte_lm_host import LanguageModelInference
from ._byte_lm_impl import SmallByteLanguageModelTrainer

LanguageModelConfig = ByteLanguageModelConfig
LanguageModelTrainer = SmallByteLanguageModelTrainer

__all__ = ['LanguageModelConfig', 'LanguageModelTrainer', 'LanguageModelInference',
           'SmallByteLanguageModelTrainer', 'ByteLanguageModelConfig']
