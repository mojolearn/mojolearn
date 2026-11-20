# SPDX-License-Identifier: Apache-2.0
"""RUN of the low-bit weight provider (providers/python/lowbit_weight_pyapi.mojo):
the empty-input refusal from typed shape words. No Python import
(tools/python_free_dirs.sh). Lane rs-shared-infra-b (2026-11-12): moved word
for word."""
from bindings.tensor_view import shape_text


def low_empty(name: String, shape: List[Int]) raises:
    var what = "feature(s)" if len(shape) == 2 and shape[0] != 0 else "sample(s)"
    var text = shape_text(shape)
    raise Error("mojolearn: " + name + " is empty, shape " + text + ": Found array with 0 " + what
        + " (shape=" + text + ") while a minimum of 1 is required.")
