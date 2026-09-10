# SPDX-License-Identifier: Apache-2.0
"""Configured FP32 decoder language-model training and compatibility aliases."""
from ._byte_lm_config import ByteLanguageModelConfig
from ._byte_lm_impl import SmallByteLanguageModelTrainer

LanguageModelConfig = ByteLanguageModelConfig
LanguageModelTrainer = SmallByteLanguageModelTrainer

__all__ = ['LanguageModelConfig', 'LanguageModelTrainer',
           'SmallByteLanguageModelTrainer', 'ByteLanguageModelConfig']
