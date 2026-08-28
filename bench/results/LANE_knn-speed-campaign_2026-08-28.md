# LANE: knn speed campaign (DEV 1921-1923) — code only, NO timings run

2026-08-28, builder lane on `integrate/dev1921-knn-campaign`. NO tests,
benchmarks or binding builds were run in this lane; `mojo build --emit
object` compile checks only (apple / nvidia / amd columns, FAST and
IDENTICAL, all clean). The orchestrator gates and times centrally.

The target row: H100 FAST legs 2026-08-25/26
(`bench/results/fast_speed/2026-08-25_235543-nvidia-classical.md`,
`2026-08-26_040100-nvidia-classical.md`): knn 400,000 x 32, 4,000
queries, k = 10 — ours 233.8-243.9 ms vs `cuml-gpu` 10.2-10.8 ms,
**21.6-23.9x SLOWER**, the worst classical row on the board.

## DEV 1921 — route the distance GEMM through MAX linalg: ALREADY SHIPPED, no code

The campaign brief asked whether our knn path can call MAX's matmul on
device buffers we already hold. VERDICT: it can, it does, and it has
since before this lane.

- `core/gemm.mojo::gemm_nt` calls `linalg.matmul.matmul[transpose_b=True,
  target="gpu"]` on `TileTensor(DeviceBuffer, row_major(...))` views under
  FAST; `tiled_brute_force_knn` reaches it through a `create_sub_buffer`
  window (no copy, no layout change). Under IDENTICAL the same call site
  takes `pinned_distance_tile_kernel` (DEVIATION 505/526), byte for byte
  as before.
- The substitution is legitimate ONLY on the tiled arm and
  `VENDOR_LIBS.md`'s banner is the authority: cuVS's dispatch for the
  bench parameters (`knn_brute_force.cuh:443`, k <= 64 row-major L2) is
  `fusedL2Knn`, which calls NO vendor primitive — a fused tile kernel
  with a register-resident FAISS WarpSelect queue. When they fuse, they
  hand-write; a device-wide matmul cannot be fused and forces the
  distance matrix to be materialized. That fused kernel is ported
  (`neighbors/ported/neighbors/detail/fused_l2_knn.mojo`).

So the 21.6x deficit is NOT "we hand-write a GEMM cuBLAS would win" — the
FAST tiled arm already runs MAX's matmul. The deficit is (a) WHICH ARM
AUTO picks on NVIDIA and (b) WHICH SELECTOR the tiled arm runs, which is
what 1922/1923 change. No code was written under this number.

## DEV 1922 — warpsort selection on the tiled path (commit 195b8e3)

Kernel-matrix row `knn_warpsort_select_for` + `_warpsort_select_tile` in
`neighbors/ported/neighbors/detail/knn_brute_force.mojo`. RAFT's own
`select_k` sends `2 < k <= 256` to warpsort and only `k > 256` to radix
(`select_k-inl.cuh:38`); we ran the ported radix alone across that band.
The ported `warpsort_topk_block_kernel` (single-pass block form, grid
`(1, rows)`, capacity 32/64/128/256 with a floor of 32) now selects on
columns the row admits: NVIDIA FAST only. Apple FAST keeps radix byte
for byte (never measured on Metal — the row is where it flips); AMD/CDNA
comptime-EXCLUDED (`WARP_LANES = 32` geometry does not exist at 64
lanes, the DEVIATION 1910 shape); IDENTICAL keeps
`select_radix_identical` (DEVIATIONS 500/501) untouched everywhere.

The warp-primitive situation, for the record: `std.gpu.primitives.warp`
ships `shuffle_xor/shuffle_idx/shuffle_down/lane_id` with NO width
parameter; the ported warpsort needs only `shuffle_xor` with
`stride < warp_width` over power-of-two aligned subwarps, where the
width argument cannot change the answer (the file's DEVIATION 2). The
`warp_sort_filtered/_distributed` queues stay unported for exactly the
missing-width reason recorded in that file's docstring.

Open item (declared, not hidden): RAFT's `calc_launch_parameter` splits
long rows over several blocks and merges with a second launch
(`select_k_:1106-1120`); ours is the one-block-per-row form. If one
block per row underfills a device, that caller loop is the next port.

## DEV 1923 — AUTO follows their dispatch on NVIDIA (commit d962d35)

Kernel-matrix row `knn_auto_follows_their_dispatch_for`: NVIDIA FAST
AUTO now takes `fusedL2Knn` unconditionally for k <= 64 row-major L2,
x-split and mutex merge included — cuVS's own default. DEVIATION 36's
`grid_x == 1` gate was measured entirely on Apple (the Metal x-split
mutex cliff, 0.19x); on the H100 bench shape the NVIDIA occupancy inputs
give `grid_x > 1`, so AUTO was taking the tiled arm on the one vendor
whose own library never takes it there. Apple keeps DEVIATION 36 byte
for byte; AMD moot (fused refuses at 64 lanes, DEVIATION 512);
IDENTICAL pinned tiled everywhere (DEVIATION 509). Also deleted
`brute_force_knn_impl`'s false "defaults to KNN_METHOD_TILED" sentence
(the signature has said AUTO since the AUTO round) per
fix-docs-on-discovery.

## FAST numeric effect (both rows, NVIDIA column only)

- Selected distance VALUES: same multiset — both selectors take the k
  smallest of the same materialized tile; the fused arm computes
  distances in a different (fused) order, which is the already-declared
  FAST arm difference of DEVIATION 36.
- Ties: which equidistant index survives, and its slot, may move (radix
  arrival-order atomics vs bitonic positional merge vs FAISS queue's
  distance-only comparator). All inside FAST's declared unpinned tie set
  (IDENTITY_PATHS row 11). IDENTICAL is untouched on every column — the
  identical selector, arm pin and pinned distance kernel did not move.
- Apple FAST: byte-for-byte unchanged (both rows False there; the only
  code motion is the radix enqueue hoisted into `_radix_select_tile`,
  same enqueue).

## What the orchestrator must gate

1. NVIDIA FAST correctness: `check-knn` family on an H100 leg (the fused
   arm's checks exist: `check_fused_l2_knn`, `check_fused_griddimx_merge`,
   `check_fused_edge_shapes`; warpsort vs radix vs host oracle:
   `neighbors/mojo_only/warpsort_check.mojo`, `knn_check.mojo`).
2. The knn speed row, three arms (`ours` / `ours-fused` / `ours-tiled`,
   already in `run_knn`): AUTO should now match `ours-fused` on H100.
   If `ours-fused` does NOT beat `ours-tiled` there, DEV 1923's row
   flips back — the rows are the flip point, one line each.
3. The tiled arm's selector A/B on the same leg (radix vs warpsort at
   k = 10): DEV 1922's row is the flip point.
4. IDENTICAL cards unchanged (Apple at minimum; the E1U/E2U knn cells).
5. Mac FAST board: expected UNCHANGED — any movement on Apple rows is a
   defect in this lane, not a result.
