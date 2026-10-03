# OPTICS on the Apple GPU: profile of one fit (lane/apple-fast-optics2, 2026-10-03)

Board shape (tools/bench_board_algos.py:289): `OPTICS(min_samples=10, xi=0.05)`, euclidean (metric code -1,
the squared-distance route), `max_eps=inf`, cluster_method xi, n = 10,000 rows (SUB["quad"]), d = 220 (istella)
or the taxi width. M3 Ultra: FAST 412 ms taxi / 514 ms istella, IDENTICAL 417 / 534: FAST has never diverged
from IDENTICAL here, so every ms below is main's path (`x_cluster/optics.mojo` `optics_graph`,
`x_cluster/device_ops.mojo` `DeviceOps.optics_order`, `x_cluster/device_post.mojo` the optics kernels).

## Per fit, in stream order (GPU binding, `DeviceOps`)

| # | call (file:line) | launches | syncs | host copies | device memory |
|---|---|---|---|---|---|
| 1 | `ops.put(x)` (optics.mojo:159) | 0 | 1 (n*d >= 1 MB: synchronous upload, device_ops.mojo:1484) | upload n*d*4 B (8.8 MB istella) | buffer n*d |
| 2 | `dist_slot` -> `ops.zeros(n*n)` (optics.mojo:50, 171) | 1 memset of 400 MB | 0 | 0 | buffer n*n (400 MB) |
| 3 | `ops.sqdist` tiled (device_ops.mojo:1604) | 1 (157 x 157 blocks of 256) | 0 | 0 | writes 400 MB |
| 4 | `ops.sqrt(dm, n*n)` (optics.mojo:173; device_ops.mojo:1624) | 1 (781,250 blocks of 128) | 0 | 0 | reads + writes 400 MB |
| 5 | `ops.zeros(n)` core slot (optics.mojo:174) | 1 memset | 0 | 0 | buffer n |
| 6 | `ops.kth` (device_ops.mojo:1631, `_kth_kernel` :225) | 1 (n blocks of 256, 4 radix passes over each row) | 0 | 0 | reads 4 x 400 MB |
| 7 | `ops.zeros_i(n)` x2, `ops.zeros(n)` (optics.mojo:188-190) | 3 memsets | 0 | 0 | 3 buffers n |
| 8 | `optics_order` (device_ops.mojo:2388): `zeros_i(n)` done, `_keys(nb)` partials, `optics_init_kernel` | 1 memset + 1 | 0 | 0 | 2 buffers |
| 9 | the ordering loop, n steps: `optics_part_kernel` (nb = n/1024 blocks of 256: the block-min key over the unprocessed rows) then `optics_step_kernel` (79 blocks of 128: every thread re-reduces the nb partials, thread 0 books the point, one row relaxed per thread) | **2n = 20,000** | 0 | 0 | each step reads one 40 KB distance row, reach, done |
| 10 | `get_if(os_, rs)` (optics.mojo:194) | 0 | 1 | 2 x 40 KB down | |
| 11 | `gets([cs])` (optics.mojo:195) | 0 | 1 | 40 KB down | |
| 12 | `get_i(ps)` (optics.mojo:197) | 0 | 1 | 40 KB down | |
| 13 | xi extraction on the host (`optics_xi_clusters`, `optics_xi_labels`, optics.mojo:243-377) | 0 | 0 | 0 | a serial data-dependent walk over the n-length plot: plot and four steep flags O(n), the steep-area state machine O(n) amortized (`_extend_region` moves `index` forward), the predecessor correction a short inner scan per candidate cluster |

Totals: ~20,009 launches, 4 synchronizes (1 upload, 3 readbacks), 6 memsets, 9 live Metal buffers for the fit,
~2.4 GB of n^2 traffic (memset 0.4, sqdist write 0.4, sqrt 0.8, kth 1.6 reads... 3.2 GB counting the kth passes).

## Where the time is

- The ordering loop is the fit. A Metal launch costs ~20 us and each step is TWO launches that cannot overlap
  (the step kernel needs the partial mins, the next partials need the relaxed reachabilities):
  20,000 launches x ~20 us = ~400 ms, which is the whole measured 412 ms on taxi. Per-step GPU work is tiny
  (one 40 KB row, 10,000 compares).
- The n^2 passes (2 to 6) are a few ms together on the M3 Ultra (~3.2 GB at several hundred GB/s): the 400 MB
  memset of `dist_slot` (`XC_ALLOC` is opt-in, so the default zeroes a matrix `sqdist` fills completely), the
  sqrt pass over n^2 cells, the four radix passes of `kth`.
- The three readback syncs after the loop are ~0.2 ms each with an idle stream.
- The host xi walk at n = 10,000 is well under a millisecond's order of magnitude of the fit (O(n) with small
  constants, no n^2 term) and is serial by nature (each steep area depends on the previous `index` and `mib`):
  on the device it would be a one-thread kernel plus a readback of the same size, which the GPU rules forbid.

## Experiments (each its own define; docs/apple-fast/ab/optics2.md explains them for the manager)

1. `MOJOLEARN_OPTICS_STEP_BATCH`: the ordering loop as ONE threadgroup of 1024 threads running 512 steps per
   launch (`optics_batch_kernel`): thread t owns rows t, t+1024, ... (their reach, pred and done words are
   written and read by that thread only), the step's point is a block-wide min of `order_key(reach, i)` (warp
   shuffles then one 32-entry threadgroup pass: 3 barriers a step), the relaxation reads the point's distance
   row coalesced. n/512 launches instead of 2n (20 vs 20,000 at 10k rows); each launch stays far under the 4 s
   command-buffer cut (512 steps of a few us).
2. `MOJOLEARN_OPTICS_FRONTIER_DEVICE`: one launch per step (`optics_fused_kernel`): a block relaxes its 1024
   rows against the step's point AND emits the block's min key over its still unprocessed rows for the next
   step; the partials are double-buffered (step s reads part[s % 2], writes part[(s+1) % 2]) so no block can
   overwrite a partial another block has not read. n + 2 launches instead of 2n + 1. An alternative to 1 (1 wins
   when both are set).
3. `MOJOLEARN_OPTICS_CORE_SQ`: no sqrt pass over the n^2 matrix. `kth` runs on the squared distances (sqrt is
   monotone and correctly rounded, so the k-th smallest of sqrt(x) is sqrt of the k-th smallest x: same bits),
   `ops.sqrt` runs over the n core values instead of n^2 cells, and the relaxation applies `sqrt_cell`'s
   expression to the squared cell at use (same input, same function, same bits as the pre-rooted cell).
   Euclidean route only (metric -1); the pdist metrics and precomputed matrices keep main's path.
4. `MOJOLEARN_OPTICS_LIVEBUF`: no memset of slots a kernel fills completely (the 400 MB matrix, the core slot,
   the ordering, reach, pred and done buffers), the four outputs laid out in TWO paired buffers
   ([ordering | pred] ints, [reach | core] floats, the core copy written by the init kernel) so ONE synchronize
   reads everything back instead of three.
5. `MOJOLEARN_OPTICS2_ALL`: 1 + 3 + 4 (2 is 1's alternative).

Not done, and why: the brief's register top-k core distance without the n x n matrix (CORE_TILE) saves at most
the kth passes (~1% of the fit) while the ordering loop needs the resident matrix anyway; the xi extraction on
the device (EXTRACT_DEVICE) is a serial state machine that would need a one-thread launch.

## Tie and bit contract (what every variant preserves)

The point of a step is the unprocessed row with the lowest reachability, the LOWEST INDEX on a tie
(`post_bodies.order_key`: value bits then index; the host column's `point < 0 or pr[i] < best` walk). A row
within `max_eps` of the point takes `max(dist, core)` when strictly lower (`optics_relax_cell`). +inf
reachabilities (`Float32.MAX * 2`) are the "never reached" value and sort last, lowest index first. All
variants compute exactly these picks and updates; only launch shape and buffer layout change.
