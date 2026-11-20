# SPDX-License-Identifier: Apache-2.0
"""RUN of `bindings/tsa_arima_resident.mojo` (wave 8 run split): the eleven
plan words of `arima_resident_prepare`, typed Mojo only. Imports no Python
(tools/python_free_dirs.sh); body moved unchanged (lane rs-sequence-tokenizer,
2026-11-12)."""


def arima_resident_words(batch: Int, n_obs: Int, order: List[Int], k: Int, n_exog: Int) -> List[Int]:
    """`arima_resident_prepare`'s eleven plan words, in its order."""
    return [batch, n_obs, order[0], order[1], order[2], order[3], order[4], order[5], order[6], k, n_exog]
