# SPDX-License-Identifier: Apache-2.0
"""RUN of model selection's input reader (bindings/msel_inputs.mojo): the
final dtype of a materialized pandas frame from its promoted block kinds. No
Python import (tools/python_free_dirs.sh). Lane rs-shared-infra-b
(2026-11-12): moved word for word; one decision per frame, nothing to split
across cores."""


def _pandas_promoted_dtype(mut kind: String, mut item: Int, objects: Bool, booleans: Bool,
                           numbers: Bool) -> String:
    """`_pandas_materialize`'s output dtype after the block walk promoted
    `kind` and `item`; updates both to the output's and returns its typestr."""
    if objects or (booleans and numbers):
        # The old materializer reads pandas' object export as int64 when
        # every scalar is integral, float64 otherwise.
        item = 8
        kind = "f" if kind == "f" else "i"
    if kind == "i" and item == 2:
        item = 4  # the established Array ABI widens signed 16-bit buffers to int32
    if kind == "b":
        kind = "u"
    if kind == "u" and item == 8:
        kind = "i"
    var dtype = "<" + kind + String(item)
    return dtype
