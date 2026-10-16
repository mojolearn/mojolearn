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

## DeviceContext memory manager reserves 90% of free VRAM per process on AMD (2026-10-16, amd-investigate)

**What we see.** On an AMD Instinct MI300X VF (ROCm 7.2.4, gfx942), the first non-small allocation of a process makes MAX's
DeviceContext memory manager take one chunk of 90% of FREE VRAM (`hipMalloc` of 184.9 GB, about 4.2 s) plus 1.28 GiB of
pinned host memory. A following process waits about 5 s for that chunk to be released and cleared (ttm delayed delete and
SDMA clear), and a second concurrent process on the same GPU starves: allocations fail, later HIP calls return
`hipErrorUnknown`/`hipErrorNoBinaryForGpu`, and `libamdhip64.so.7.2.70204` segfaults in its allocation-failure path
(`segfault at 8 ... lock xadd`).

**The ask.** A default that does not assume one process owns the GPU: a bounded or growing chunk (the binary already has
`growing_chunk_size_to` and `min_chunk_size`), or documented guidance for libraries that may share a device. Also, the
libamdhip64 null dereference on allocation failure belongs to AMD's CLR (reproducer: two concurrent processes with the
default manager on one VF).

**Where.** `python/mojolearn/_backend.py` `_bound_amd_memory_manager` (all AMD processes); `tools/gemm_remote_leg.sh` (AMD leg).

**Meanwhile.** mojolearn sets `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE=1073741824` on AMD hosts with `setdefault`
(a caller's own setting wins), which MAX supports: fresh processes 0.78 s instead of 5.9 s, concurrent processes pass, no
bit changes (`verify --compare` against NVIDIA: 1747 agree, 0 differ). NVIDIA keeps MAX's defaults.
