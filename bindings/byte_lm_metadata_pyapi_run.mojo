# SPDX-License-Identifier: Apache-2.0
"""The byte LM configuration field names, in their published order (RUN)."""


def _config_names() -> List[String]:
    return [String("batch"), String("length"), String("d_model"), String("n_heads"), String("n_kv"),
            String("head_dim"), String("intermediate"), String("n_layers"), String("vocab_size")]
