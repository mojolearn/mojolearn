# Asks for Modular

Features we wait for instead of working around (CLAUDE.md, "Wait for Mojo and Modular"). One section per ask:
what we need, where it would be used, what we do meanwhile.

## Masked (subset) warp reduction and native grouping on AMD (2026-10-07, lane trees-hist-ideas)

**Need.** Warp-aggregated histogram atomics in their general form: lanes that hit the same histogram cell
(`std.gpu.primitives.warp.match_any`) add their values once. That needs a reduction over an ARBITRARY lane
subset, the group mask `match_any` returns:

- NVIDIA: `redux.sync.add.s32 dst, src, membermask` (sm_80+), or `__reduce_add_sync(mask, v)`. Mojo's `warp`
  module exposes `sum`/`reduce` over the whole warp or a lane group of fixed width, not over a member mask.
- AMD (CDNA, wave64): no hardware match or masked reduce. `match_any` builds for gfx942 (probed 2026-10-07),
  but we do not know whether it lowers to a per-distinct-value loop; its cost on a 256-bin key is the question.
  A documented cost model, or a native lowering, would let us decide.

**Where.** `gbdt/methods/greedy_subsets_searcher/kernel/hist_2_one_byte_8bit.mojo` (`h8_add_point_wave`), and the
same pattern in RF class-count histograms and categorical/CTR counters (low-cardinality bins).

**Meanwhile.** `MOJOLEARN_TREES_HIST_WARP_AGG` aggregates only the whole-wave case (every lane on the same cell:
`warp.broadcast` + `vote` + `warp.sum`), which needs no subset reduce. Partial groups take one atomic per lane,
as before. No hand-written intrinsic or inline assembly.
