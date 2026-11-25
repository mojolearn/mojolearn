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
    order; returns the total element count (one direction, with biases)."""
    var emitted = List[Bool]()
    return recurrent_state_layout(gates, hidden, layers, features, outputs, names, dimensions, counts, 1, True,
                                  emitted)


def recurrent_state_layout(gates: Int, hidden: Int, layers: Int, features: Int, outputs: Int,
                           mut names: List[String], mut dimensions: List[List[Int]],
                           mut counts: List[Int], dirs: Int, bias: Bool, mut emitted: List[Bool]) -> Int:
    """The layout with `dirs` directions (lane rnn-parity: 2 is bidirectional, each layer's reverse block after
    its forward one under PyTorch's `_reverse` names; layer l > 0 reads dirs * hidden features, the head too)
    and `bias` (False: the recurrent biases keep their zero slots in the packed block but are not named, as
    PyTorch's bias=False module has none; `emitted` marks the named entries). Every entry, named or not, is in
    names / dimensions / counts in packing order."""
    var total = 0
    for layer in range(layers):  # driver-loop: over recurrent parameter block layout
        var input_width = features if layer == 0 else dirs * hidden
        for d in range(dirs):  # driver-loop: over directions
            var suffix = String(layer) + ("_reverse" if d == 1 else "")
            names.append("weight_ih_l" + suffix)
            dimensions.append(List[Int]([gates * hidden, input_width]))
            emitted.append(True)
            names.append("weight_hh_l" + suffix)
            dimensions.append(List[Int]([gates * hidden, hidden]))
            emitted.append(True)
            names.append("bias_ih_l" + suffix)
            dimensions.append(List[Int]([gates * hidden]))
            emitted.append(bias)
            names.append("bias_hh_l" + suffix)
            dimensions.append(List[Int]([gates * hidden]))
            emitted.append(bias)
    names.append("head.weight")
    dimensions.append(List[Int]([outputs, dirs * hidden]))
    emitted.append(True)
    names.append("head.bias")
    dimensions.append(List[Int]([outputs]))
    emitted.append(True)
    for i in range(len(names)):  # driver-loop: over named parameter buffers
        var count = 1
        for j in range(len(dimensions[i])):  # small-loop(parameter shape words)
            count *= dimensions[i][j]
        counts.append(count)
        total += count
    return total
