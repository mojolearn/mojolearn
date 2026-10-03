# lane/apple-fast-graph: connected_components (CSR) under FAST on Apple

Written without a Mojo toolchain (cloud peer); the first M3 build of `x_neighbors` (`bindings/build_x_neighbors.sh`,
FAST, `-D MOJOLEARN_CC_FAST`) is the compile check. Default OFF; IDENTICAL compiles main's code unchanged.

Gap: connected-components taxi 88 ms vs networkx 7.3 ms, istella 93 vs 7.0 ms. The board clocks
`connected_components(csr, directed=False)` on the 20,000-node k=2 kNN graph (54,528 edges). networkx is pure Python,
so the 90 ms is overhead: main's `_cc_csr_device` (x_neighbors/iter_device.mojo) creates a pinned staging buffer and
waits for it, uploads three times, then per round a memset, two launches, a one-word download and a host wait; Python
builds the identity labels through a per-item walk (`_i32(list(range(n)))`) and relabels through a dict loop
(`_cc_relabel`) over 20,000 labels.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_CC_FAST` | define, `CC_FAST` comptime in `x_neighbors/iter_device.mojo` | `op_cc_iterate_csr` -> `_cc_csr_fast` (new), kernels `cc_hook_round_kernel`, `cc_root_count_kernel`, `cc_root_rank_kernel`, `cc_relabel_kernel`; the scan stages reuse `nan_group_sum_kernel` / `nan_top_scan_kernel` / `nan_down_scan_kernel` from `x_neighbors/nan_cells_device.mojo` | CSR and start labels uploaded straight from the caller's memory (no staging buffer, no wait for it). `CC_FAST_BATCH = 4` hook + jump rounds between two reads of the change word; the word is never cleared: every writer of a round stores the round number, so after a batch the word equals the batch's last round iff that round still lowered a label (a round that changes nothing is the fixed point, so the spare rounds of a batch are free of effect). Relabel on the device: root flags (`lab[v] == v`) per block -> group sums -> top scan (the count) -> block offsets -> each root's rank -> every node its root's rank; one download of the labels and the count (`info[1] = n_components + 1`). |
| Python (`python/mojolearn/_expansion_neighbors.py`, `connected_components`, `_cc_fast_tier`) | FAST + metal branch, no switch | the CSR path only | identity start labels from `Array._from_flat(range(n), ...)` (array.array, C speed), a two-word info; the device's labels and count are taken when info[1] > 0, else main's `_cc_relabel`. Both A/B arms run this Python (the binding decides); arm A's Mojo is main's. |

Same labels: the min-label fixed point names every component by its minimum node, which is the first node (in node order)
that carries the component's label, so numbering the roots by "roots below me" is exactly `_cc_relabel`'s order of
first appearance. The board compares the component count and ARI vs networkx; both unchanged.

## Risky compile sites

- `x_neighbors/iter_device.mojo` `_cc_csr_fast`: `enqueue_fill(ctx, d_c, Int32(0))` (core/device_zero, the form used at
  `op_svgp_stats`); `ctx.enqueue_copy(dst_ptr=hflag.unsafe_ptr(), src_buf=d_c)` with a `List[Int32]` destination (the
  form of `nan_cells_device`); `for _ in range(CC_FAST_BATCH)` (the form of core/shuffle_iterator.mojo); the import of
  `nan_*_kernel` and `NC_*` constants from `x_neighbors/nan_cells_device.mojo` (module-level defs; that module imports
  only `x_neighbors/device_ops.mojo`, no cycle).
- Kernel pointer parameters are `IP` (`MutPointer[Int32, MutAnyOrigin]`) launched with `DeviceBuffer.unsafe_ptr()`,
  exactly as the sibling `cc_hook_kernel` launches in `_cc_csr_device`; no new `MutAnyOrigin` parameters.
- `cc_hook_round_kernel`'s `_ = Atomic[DType.int32].min(lab + Int(hi), lo)` is the dense kernel's form.
- `comptime assert NC_SMEM_FITS` at the top of `_cc_csr_fast` (the form of `nan_cells_device`).

## Not covered

The dense-adjacency path (`op_cc_iterate`) keeps main's per-round waits; the board's connected-components lane uses the
CSR path. istella follows after taxi wins.
