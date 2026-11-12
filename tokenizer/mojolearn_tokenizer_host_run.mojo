# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/_mojolearn_tokenizer_host.mojo` (wave 8 run split): the
loaded-handle refusal, typed Mojo only. Imports no Python
(tools/python_free_dirs.sh); body moved unchanged (lane
rs-sequence-tokenizer, 2026-11-12)."""


def _require_loaded(loaded: Bool) raises:
    if not loaded:
        raise Error("tokenizer host: the handle holds no tokenizer; bpe_load it")
