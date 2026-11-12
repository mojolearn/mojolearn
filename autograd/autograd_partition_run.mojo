# SPDX-License-Identifier: Apache-2.0
"""RUN of `providers/python/autograd_partition_pyapi.mojo` (wave 8 run
split): the fit's loss-name code, typed Mojo only. Imports no Python
(tools/python_free_dirs.sh). The statements are the former door's,
unchanged (lane rs-sequence-tokenizer, 2026-11-12)."""


def partition_loss_kind(name: String) raises -> Int:
    """The loss code of the program fit: output 0, mse 1, mae 2, huber 3,
    bce_with_logits 4, cross_entropy 5."""
    var kind = -1
    if name == "output":
        kind = 0
    elif name == "mse":
        kind = 1
    elif name == "mae":
        kind = 2
    elif name == "huber":
        kind = 3
    elif name == "bce_with_logits":
        kind = 4
    elif name == "cross_entropy":
        kind = 5
    else:
        raise Error("partitioned fit: unknown loss")
    return kind
