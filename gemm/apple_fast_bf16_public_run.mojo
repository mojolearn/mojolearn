# SPDX-License-Identifier: Apache-2.0
"""The compiled-candidate refusal of the Apple fast BF16 doors (RUN)."""
from gemm.afn_apple_fast import AFN_GEMM_BF16_MMA


def _require_candidate() raises:
    comptime if not AFN_GEMM_BF16_MMA:
        raise Error("Apple fast BF16 requires a fast Apple GPU binding with MOJOLEARN_AFN_GEMM_BF16_MMA enabled (default 1; explicit 0 disables it); no identical fallback is used")
