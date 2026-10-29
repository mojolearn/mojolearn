# SPDX-License-Identifier: Apache-2.0
"""Host generation policy, shared by every accelerator and the CPU.

Only logits cross this boundary. All vendors use this one ordered sampler:
descending logits, lower token id first on ties, portable FP32 exp, sequential
FP64 sums, and a SplitMix64 counter indexed by (seed, batch row, output index).
Chunk boundaries never enter the counter. No process RNG or vendor math is used.
"""
from std.math import isfinite
from checks.numerics import portable_expf
from core.generation_sampling_f32 import sampling_pick_f32, sampling_random


def generation_random(seed: UInt64, row: Int, position: Int) -> UInt64:
    return sampling_random(seed, row, position)


struct GenerationSamplingScratch(Movable):
    # Private to one operation; no logits or caller buffers escape into scratch.
    # Greedy never sizes these lists. Sampling overwrites every retained element.
    var order: List[Int]
    var merge: List[Int]
    var weights: List[Float64]
    var use_f32_policy: Bool
    var order_f32: List[Int32]
    var merge_f32: List[Int32]
    var weights_f32: List[Float32]

    def __init__(out self):
        self.order = List[Int]()
        self.merge = List[Int]()
        self.weights = List[Float64]()
        self.use_f32_policy = False
        self.order_f32 = List[Int32]()
        self.merge_f32 = List[Int32]()
        self.weights_f32 = List[Float32]()


def _before(logits: MutPointer[Float32, MutUntrackedOrigin], a: Int, b: Int) -> Bool:
    return logits[a] > logits[b] or (logits[a] == logits[b] and a < b)


def generation_pick_row(
    logits: MutPointer[Float32, MutUntrackedOrigin], n: Int, greedy: Bool, temperature: Float32,
    top_k: Int, top_p: Float64, seed: UInt64, row: Int, position: Int,
    mut workspace: GenerationSamplingScratch,
) raises -> Int:
    # Caller retains the row owner until this synchronous native scan returns.
    if n == 0:
        raise Error("generation: empty vocabulary")
    if temperature <= 0 or not isfinite(temperature):
        raise Error("generation: temperature must be finite and positive")
    if top_k < 0 or top_k > n or top_p <= 0 or top_p > 1 or not isfinite(top_p):
        raise Error("generation: top_k must be in [0, vocabulary]; top_p in (0, 1]")
    if workspace.use_f32_policy and not greedy:
        if n > 2147483647:
            raise Error("generation: vocabulary exceeds native sampler indexing")
        workspace.order_f32.resize(n, 0)
        workspace.merge_f32.resize(n, 0)
        workspace.weights_f32.resize(n if top_k == 0 else top_k, 0)
        var selected = sampling_pick_f32(
            MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(logits)),
            MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(workspace.order_f32.unsafe_ptr())),
            MutPointer[Int32, MutAnyOrigin](unsafe_from_address=Int(workspace.merge_f32.unsafe_ptr())),
            MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(workspace.weights_f32.unsafe_ptr())),
            n, temperature, top_k, Float32(top_p), seed, row, position)
        _ = workspace
        if selected[1] >= 0:
            raise Error("generation: non-finite logit at token " + String(selected[1]))
        return selected[0]
    var best = 0
    for i in range(n):
        if not isfinite(logits[i]):
            raise Error("generation: non-finite logit at token " + String(i))
        if logits[i] > logits[best]:
            best = i
    if greedy:
        return best
    # Stable bottom-up mergesort: O(V log V), independent of vocabulary or
    # benchmark dimensions. The explicit token tie break is part of the policy.
    workspace.order.resize(n, 0)
    workspace.merge.resize(n, 0)
    ref order = workspace.order
    ref scratch = workspace.merge
    for i in range(n):
        order[i] = i
    var width = 1
    while width < n:
        var start = 0
        while start < n:
            var mid = min(start + width, n)
            var end = min(start + width * 2, n)
            var a = start
            var b = mid
            for dest in range(start, end):
                if a < mid and (b == end or _before(logits, order[a], order[b])):
                    scratch[dest] = order[a]
                    a += 1
                else:
                    scratch[dest] = order[b]
                    b += 1
            start = end
        for i in range(n):
            order[i] = scratch[i]
        width *= 2
    var keep = n if top_k == 0 else top_k
    workspace.weights.resize(keep, 0)
    ref weights = workspace.weights
    var total = Float64(0)
    var maximum = logits[order[0]]
    for i in range(keep):
        var weight = Float64(portable_expf((logits[order[i]] - maximum) / temperature))
        weights[i] = weight
        total += weight
    # Nucleus truncation includes the first token that reaches the requested
    # mass. Top-k is applied first; top-p is relative to that distribution.
    var limit = top_p * total
    var retained = Float64(0)
    var count = 0
    while count < keep:
        retained += weights[count]
        count += 1
        if retained >= limit:
            break
    # Exactly 53 random bits, in [0, 1), are representable on every host.
    var uniform = Float64(generation_random(seed, row, position) >> 11) * Float64(1.0 / 9007199254740992.0)
    var target = uniform * retained
    var prefix = Float64(0)
    for i in range(count):
        prefix += weights[i]
        if target < prefix:
            return order[i]
    return order[count - 1]


def generation_pick(
    logits: List[Float32], greedy: Bool, temperature: Float32,
    top_k: Int, top_p: Float64, seed: UInt64, row: Int, position: Int,
) raises -> Int:
    var workspace = GenerationSamplingScratch()
    var result = generation_pick_row(
        MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(logits.unsafe_ptr())),
        len(logits), greedy, temperature, top_k, top_p, seed, row, position, workspace)
    _ = logits  # Retain the borrowed row owner through its final pointer read.
    return result
