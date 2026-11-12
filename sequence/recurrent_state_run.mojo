# SPDX-License-Identifier: Apache-2.0
"""RUN of `providers/python/recurrent_state_pyapi.mojo` (wave 8 run split):
the recurrent model's named parameter layout (names, shapes, element counts,
total), typed Mojo only. Imports no Python (tools/python_free_dirs.sh). The
statements are the former layout loops of `recurrent_state` unchanged; the
door keeps the field admission and the state-dict copies (lane
rs-sequence-tokenizer, 2026-11-12)."""


def recurrent_state_layout(gates: Int, hidden: Int, layers: Int, features: Int, outputs: Int,
                           mut names: List[String], mut dimensions: List[List[Int]],
                           mut counts: List[Int]) -> Int:
    """Appends each parameter's name, shape and element count in packing
    order; returns the total element count."""
    var total = 0
    for layer in range(layers):  # driver-loop: over recurrent parameter block layout
        var input_width = features if layer == 0 else hidden
        names.append("weight_ih_l" + String(layer))
        dimensions.append(List[Int]([gates * hidden, input_width]))
        names.append("weight_hh_l" + String(layer))
        dimensions.append(List[Int]([gates * hidden, hidden]))
        names.append("bias_ih_l" + String(layer))
        dimensions.append(List[Int]([gates * hidden]))
        names.append("bias_hh_l" + String(layer))
        dimensions.append(List[Int]([gates * hidden]))
    names.append("head.weight")
    dimensions.append(List[Int]([outputs, hidden]))
    names.append("head.bias")
    dimensions.append(List[Int]([outputs]))
    for i in range(len(names)):  # driver-loop: over named parameter buffers
        var count = 1
        for j in range(len(dimensions[i])):  # small-loop(parameter shape words)
            count *= dimensions[i][j]
        counts.append(count)
        total += count
    return total
