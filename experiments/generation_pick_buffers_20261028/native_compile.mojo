# SPDX-License-Identifier: Apache-2.0
# Native-only compile root. No Python initialization, extension or wrapper imports.
from core.generation_policy_impl import generation_pick
from transformer.native_generation_engine import NativeGenerationModel, NativeGenerationProgress, native_generation_read


@export
def native_pick(values: MutPointer[Float32, MutUntrackedOrigin], n: Int, greedy: Bool) abi("C") -> Int:
    try:
        var logits = List[Float32](capacity=n)
        for i in range(n):
            logits.append(values[i])
        return generation_pick(logits, greedy, 1.0, 0, 1.0, 0, 0, 0)
    except:
        return -1


@export
def native_read(model: MutPointer[NativeGenerationModel, MutUntrackedOrigin],
                progress: MutPointer[NativeGenerationProgress, MutUntrackedOrigin],
                count: Int, maximum: Int, batch: Int, greedy: Bool) abi("C") -> Int:
    try:
        var finished = List[Bool](length=batch, fill=False)
        var lengths = List[Int](length=batch, fill=0)
        native_generation_read(model[], progress[], count, maximum, finished, lengths, [],
                               0, greedy, 1.0, 0, 1.0, 0)
        return progress[].wrote
    except:
        return -1
