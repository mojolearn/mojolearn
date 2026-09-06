# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The host FP32 oracle of profile `mojolearn.identical.embedding.fp32.v1`. The contract is `embedding/IDENTICAL_EMBEDDING_CONTRACT.md` and this file is that contract in code -- the two must be read together and every function below cites its section."""

from checks.numerics import ftz, identical_mul_add



comptime EMB_MAX_POSITIONS = 1073741823

comptime EMB_NO_PADDING_IDX = -1

comptime EMB_POS_INF_BITS = UInt32(0x7F800000)

comptime EMB_KEY_MASK = UInt64(0xFFFFFFFF)


@fieldwise_init
struct EmbConfig(Copyable, Movable):
    """One embedding call's configuration. **A caller that carries must fill exactly once, must not fill again, and must present microbatches in ASCENDING `t`.** Contract 7.4's price."""

    var vocab: Int
    var width: Int
    var padding_idx: Int
    var accumulate: Bool

    @staticmethod
    def llama(vocab: Int, width: Int) -> Self:
        """A fresh gradient with no `padding_idx`, which is what `LlamaModel.embed_tokens` is configured as."""
        return Self(vocab, width, EMB_NO_PADDING_IDX, False)

    def has_padding(self) -> Bool:
        """Whether `padding_idx` names a real row."""
        return self.padding_idx >= 0 and self.padding_idx < self.vocab




def refuse_nonfinite(name: String, values: List[Float32]) raises:
    """IDENTITY_PATHS row 39: a NaN or an infinity in an input is REFUSED BY NAME before any recorded stage. All four must stay the same shape."""
    for i in range(len(values)):
        var au = rebind[UInt32](values[i].to_bits()) & UInt32(0x7FFFFFFF)
        if au > EMB_POS_INF_BITS:
            raise Error(
                String("embedding: NaN in ")
                + name
                + " at flat index "
                + String(i)
                + " REFUSED (row 39: NaN payloads are vendor-shaped; no"
                + " stage may record one)"
            )
        if au == EMB_POS_INF_BITS:
            raise Error(
                String("embedding: infinity in ")
                + name
                + " at flat index "
                + String(i)
                + " REFUSED (contract 9.1)"
            )


def emb_refuse_shape(cfg: EmbConfig, n_positions: Int) raises:
    """Contract section 3's shape refusals and section 8's degenerate cases."""
    if cfg.vocab <= 0:
        raise Error(
            String("embedding: V = ")
            + String(cfg.vocab)
            + " REFUSED; every id would be out of range (contract 8)"
        )
    if cfg.width < 0:
        raise Error(
            String("embedding: d = ") + String(cfg.width) + " REFUSED"
        )
    if n_positions < 0:
        raise Error(
            String("embedding: T = ") + String(n_positions) + " REFUSED"
        )
    if n_positions > EMB_MAX_POSITIONS:
        raise Error(
            String("embedding: T = ")
            + String(n_positions)
            + " exceeds EMB_MAX_POSITIONS = "
            + String(EMB_MAX_POSITIONS)
            + " (contract 3, the key packing's own bound)"
        )


def emb_refuse_ids(ids: List[Int32], cfg: EmbConfig) raises:
    """Contract section 8. An id below zero or at or past `V` is REFUSED BY NAME, with its position and its value in the message."""
    for t in range(len(ids)):
        var v = Int(ids[t])
        if v < 0 or v >= cfg.vocab:
            raise Error(
                String("embedding: id ")
                + String(v)
                + " at position "
                + String(t)
                + " is outside [0, "
                + String(cfg.vocab)
                + ") REFUSED (contract 8; never clamped)"
            )




def emb_counts(ids: List[Int32], cfg: EmbConfig) -> List[Int32]:
    """`counts[v]`, the length of run `v`."""
    var counts = List[Int32](capacity=cfg.vocab)
    for _ in range(cfg.vocab):
        counts.append(Int32(0))
    for t in range(len(ids)):
        var v = Int(ids[t])
        if v == cfg.padding_idx:
            continue
        if v < 0 or v >= cfg.vocab:
            continue  # `emb_refuse_ids` has already raised; belt only.
        counts[v] = counts[v] + Int32(1)
    return counts^


def emb_run_begin(counts: List[Int32]) -> List[Int32]:
    """The exclusive prefix sum of `counts`, length `V + 1`."""
    var begin = List[Int32](capacity=len(counts) + 1)
    var acc = Int32(0)
    for v in range(len(counts)):
        begin.append(acc)
        acc = acc + counts[v]
    begin.append(acc)
    return begin^


def emb_perm_by_scan(ids: List[Int32], cfg: EmbConfig) -> List[Int32]:
    """**THE NORMATIVE PERMUTATION.** Pass R3 of contract 6.1, `PLAN_SCAN`. For each `v` in ascending order, the positions with `ids[t] == v` in ASCENDING `t`, written into `[run_begin[v], run_begin[v+1])`."""
    var counts = emb_counts(ids, cfg)
    var begin = emb_run_begin(counts)
    var total = Int(begin[len(begin) - 1])
    var perm = List[Int32](capacity=total)
    for _ in range(total):
        perm.append(Int32(0))
    for v in range(cfg.vocab):
        if v == cfg.padding_idx:
            continue
        var w = Int(begin[v])
        for t in range(len(ids)):
            if Int(ids[t]) == v:
                perm[w] = Int32(t)
                w += 1
    return perm^


def pack_emb_key(token_id: Int32, position: Int) -> UInt64:
    """Contract 6.2(a), DEVIATION 1303. That converts "the sort must be stable", a property of an implementation, into "the sort must be correct", a property anybody can check."""
    var hi = UInt64(Int(token_id)) & EMB_KEY_MASK
    var lo = UInt64(position) & EMB_KEY_MASK
    return (hi << 32) | lo


def merge_sort_u64_with_index(mut keys: List[UInt64], mut idx: List[Int32]):
    """Bottom-up merge sort of `keys` carrying `idx` along. Deterministic and STABLE, so a caller who packs a NON-total key still gets a defined (discovery) order among ties."""
    var n = len(keys)
    if n < 2:
        return
    var tk = List[UInt64](capacity=n)
    var ti = List[Int32](capacity=n)
    for _ in range(n):
        tk.append(UInt64(0))
        ti.append(Int32(0))
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = lo + width
            if mid > n:
                mid = n
            var hi = lo + 2 * width
            if hi > n:
                hi = n
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if keys[j] < keys[i]:
                    tk[k] = keys[j]
                    ti[k] = idx[j]
                    j += 1
                else:
                    tk[k] = keys[i]
                    ti[k] = idx[i]
                    i += 1
                k += 1
            while i < mid:
                tk[k] = keys[i]
                ti[k] = idx[i]
                i += 1
                k += 1
            while j < hi:
                tk[k] = keys[j]
                ti[k] = idx[j]
                j += 1
                k += 1
            lo += 2 * width
        for t in range(n):
            keys[t] = tk[t]
            idx[t] = ti[t]
        width *= 2


def emb_perm_by_total_order_key(
    ids: List[Int32], cfg: EmbConfig
) -> List[Int32]:
    """**DIAGNOSTIC, NOT NORMATIVE.** The same permutation as `emb_perm_by_scan`, reached by sorting `pack_emb_key(id, t)`. This exists so that contract 6.2's determinism argument is CODE a check can run rather than prose a reader has to accept."""
    var keys = List[UInt64]()
    var idx = List[Int32]()
    for t in range(len(ids)):
        var v = Int(ids[t])
        if v == cfg.padding_idx:
            continue
        if v < 0 or v >= cfg.vocab:
            continue
        keys.append(pack_emb_key(ids[t], t))
        idx.append(Int32(t))
    merge_sort_u64_with_index(keys, idx)
    return idx^




def emb_backward_cell(
    dy: List[Float32],
    perm: List[Int32],
    run_lo: Int,
    run_hi: Int,
    j: Int,
    width: Int,
    seed: Float32,
) -> Float32:
    """**THE CONTRACT'S FOLD.** One output cell, contract 5.1. acc = seed (+0.0, or 7.4's carry) for r in [run_lo, run_hi) ASCENDING sorted position, which IS ascending absolute t acc = ftz( ftz(acc) + ftz(dy[perm[r] * width + j]) ) return ftz(acc) Six clauses live in those four lines and each is separately falsifiable."""
    var acc = seed
    for r in range(run_lo, run_hi):
        var t = Int(perm[r])
        acc = ftz(ftz(acc) + ftz(dy[t * width + j]))
    return ftz(acc)


def emb_fold_balanced_tree_diagnostic(
    contributions: List[Float32],
) -> Float32:
    """**DIAGNOSTIC AND REFUSED.** `gemm/IDENTICAL_FP32_CONTRACT.md` 7.2's fixed balanced tree over one run's contributions, adjacent pairing, odd tail carried, no seed and no padding. Fixture F-TREE4 of contract 11.2 is the separator, by bits -- {0x3F800000, 0x33800000, 0x33800000, 0x33800000} chain -> 0x3F800000 (1.0 + 2^-24 is the exact midpoint of 1.0 and 1+2^-23; round-half-to-EVEN picks 1.0, three times over) tree..."""
    var n = len(contributions)
    if n == 0:
        return Float32(0.0)
    var current = contributions.copy()
    while len(current) > 1:
        var w = len(current)
        var pairs = w // 2
        var nxt = List[Float32]()
        for q in range(pairs):
            nxt.append(ftz(ftz(current[2 * q]) + ftz(current[2 * q + 1])))
        if w % 2 != 0:
            nxt.append(current[w - 1])
        current = nxt^
    return ftz(current[0])


def emb_fold_via_fma_diagnostic(
    contributions: List[Float32], seed: Float32
) -> Float32:
    """**DIAGNOSTIC.** The contract's own chain, spelled with `identical_mul_add(dy, 1.0, acc)` instead of a plain add. **It must agree with `emb_backward_cell` at EVERY input, and that is a LEMMA rather than a choice** (contract 4.1, DEVIATION 1317)."""
    var acc = seed
    for i in range(len(contributions)):
        acc = ftz(
            identical_mul_add(ftz(contributions[i]), Float32(1.0), ftz(acc))
        )
    return ftz(acc)




def emb_forward_oracle(
    w: List[Float32], ids: List[Int32], cfg: EmbConfig
) raises -> List[Float32]:
    """**THE NORMATIVE FORWARD.** `Y[t, j] = ftz(ftz(W[ids[t], j]))`, row-major `[T, d]`."""
    emb_refuse_shape(cfg, len(ids))
    emb_refuse_ids(ids, cfg)
    refuse_nonfinite(String("W"), w)
    var y = List[Float32](capacity=len(ids) * cfg.width)
    for t in range(len(ids)):
        var v = Int(ids[t])
        var base = v * cfg.width
        for j in range(cfg.width):
            y.append(ftz(ftz(w[base + j])))
    return y^




def emb_backward_seed(
    cfg: EmbConfig, dw_prev: List[Float32]
) raises -> List[Float32]:
    """The `emb.dw_seed` stage, contract section 10. **THE FILL IS NOT AN IMPLEMENTATION DETAIL AND IT MAY NOT BE SKIPPED.** Contract 5.5 -- the empty run's value is STATED, not derived, and an implementation must write it rather than leave whatever was in the buffer."""
    var cells = cfg.vocab * cfg.width
    if not cfg.accumulate:
        var fresh = List[Float32](capacity=cells)
        for _ in range(cells):
            fresh.append(Float32(0.0))
        return fresh^
    if len(dw_prev) != cells:
        raise Error(
            String("embedding: accumulate=True needs dW_prev of ")
            + String(cells)
            + " cells, got "
            + String(len(dw_prev))
            + " (contract 7.4)"
        )
    var carried = List[Float32](capacity=cells)
    for i in range(cells):
        carried.append(ftz(dw_prev[i]))
    return carried^


def emb_backward_oracle(
    dy: List[Float32],
    ids: List[Int32],
    cfg: EmbConfig,
    dw_prev: List[Float32],
) raises -> List[Float32]:
    """**THE NORMATIVE BACKWARD ANSWER of `mojolearn.identical.embedding.fp32.v1`.** dW[v, :] = sum over every t with ids[t] == v, ASCENDING t, of dY[t, :], seeded +0.0, every seam flushed Row-major `[V, d]`. This is the value the device kernel must reproduce bit for bit, on Apple, NVIDIA and AMD, at every launch geometry and under every execution plan."""
    emb_refuse_shape(cfg, len(ids))
    emb_refuse_ids(ids, cfg)
    refuse_nonfinite(String("dY"), dy)

    var dw = emb_backward_seed(cfg, dw_prev)

    if cfg.has_padding() and cfg.width > 0:
        var pad_base = cfg.padding_idx * cfg.width
        for j in range(cfg.width):
            dw[pad_base + j] = Float32(0.0)

    if cfg.width == 0 or len(ids) == 0:
        return dw^

    var counts = emb_counts(ids, cfg)
    var begin = emb_run_begin(counts)
    var perm = emb_perm_by_scan(ids, cfg)

    for v in range(cfg.vocab):
        var lo = Int(begin[v])
        var hi = Int(begin[v + 1])
        if lo == hi:
            continue
        var base = v * cfg.width
        for j in range(cfg.width):
            var seed = dw[base + j]
            dw[base + j] = emb_backward_cell(
                dy, perm, lo, hi, j, cfg.width, seed
            )
    return dw^




struct EmbStages(Movable):
    """Every recorded stage of one embedding call, in the card's order. A check that hashes an empty list records a zero-length stage, which the differ must treat as "absent" rather than as "agreeing with an absent one" -- `IdentityTrace.record_device`'s own length hazard, pointed the other way."""

    var ids: List[Int32]          # [T]       as given
    var weight: List[Float32]     # [V, d]    forward only, as given
    var fwd: List[Float32]        # [T, d]    G1, G2
    var dy: List[Float32]         # [T, d]    backward only, as given
    var counts: List[Int32]       # [V]       R1
    var run_begin: List[Int32]    # [V + 1]   R2
    var perm: List[Int32]         # [T]       R3
    var dw_seed: List[Float32]    # [V, d]    the fill, or the carried dW
    var dw: List[Float32]         # [V, d]    E0 through E4

    def __init__(out self):
        self.ids = List[Int32]()
        self.weight = List[Float32]()
        self.fwd = List[Float32]()
        self.dy = List[Float32]()
        self.counts = List[Int32]()
        self.run_begin = List[Int32]()
        self.perm = List[Int32]()
        self.dw_seed = List[Float32]()
        self.dw = List[Float32]()


def emb_forward_stages(
    w: List[Float32], ids: List[Int32], cfg: EmbConfig
) raises -> EmbStages:
    """The forward's three card stages. Contract section 10."""
    var st = EmbStages()
    st.ids = ids.copy()
    st.weight = w.copy()
    st.fwd = emb_forward_oracle(w, ids, cfg)
    return st^


def emb_backward_stages(
    dy: List[Float32],
    ids: List[Int32],
    cfg: EmbConfig,
    dw_prev: List[Float32],
) raises -> EmbStages:
    """The backward's six card stages, in card order."""
    var st = EmbStages()
    st.ids = ids.copy()
    st.dy = dy.copy()
    st.counts = emb_counts(ids, cfg)
    st.run_begin = emb_run_begin(st.counts)
    st.perm = emb_perm_by_scan(ids, cfg)
    st.dw_seed = emb_backward_seed(cfg, dw_prev)
    st.dw = emb_backward_oracle(dy, ids, cfg, dw_prev)
    return st^


def emb_max_run_length(counts: List[Int32]) -> Int:
    """`R_max`, the length of the longest run. A check must assert `emb_max_run_length >= 4` on the fixtures that are supposed to separate the fold, and assert the INERT MASK on the ones that are not."""
    var best = 0
    for v in range(len(counts)):
        var c = Int(counts[v])
        if c > best:
            best = c
    return best
