# lane/mutex-extratrees-knn

**The two claim sites the random forest lane never measured.** `claim_device_mutex`
(`core/device_mutex.mojo`) is called from three subsystems. Only the forest has ever been
measured. `docs/lanes/RF_MUTEX_RECONCILIATION_2026-09-16.md` section 5 item 4 says so:
"ExtraTrees and fused kNN have no measurement at all."

This lane owns the other two. It stays off `rf-*` entirely.

## HEADLINE, established before any box was rented

**The two subsystems are NOT symmetric with the forest, and they are not symmetric with
each other. Both have a REACHABILITY GATE that the forest does not have, and the gates
are in completely different places.** Neither gate was named in the reconciliation, which
treated all three sites as "the same protocol, therefore the same exposure".

| site | grid that contends one mutex | contends when | on gfx942 |
|---|---|---|---|
| forest `_publish_to_global` | column blocks, cap `N_BLKS_FOR_COLS` = 10 | `n_cols >= 11` | **measured, 4.3% to 5.3% of fits** |
| ExtraTrees `split_reduce_kernel` | `bpn = ceildiv(k, TPB)` blocks per node | `k > TPB` | `TPB = 512`, so **`max_features * n_cols > 512`** |
| fused kNN producer/consumer | `grid_dim.x` column blocks | `grid_x > 1` | **never, in any shipped build** |

### fused L2 kNN: the claim is UNREACHABLE in every build that ships

Two independent facts, either one sufficient:

1. `neighbors/impl/detail/fused_l2_knn.mojo:877` pins `grid_x = 1` under
   `comptime if PIN_DETERMINISM`. `checks/numerics.mojo:19` makes `PIN_DETERMINISM` true
   for both IDENTICAL and DETERMINISTIC. At `grid_dim.x == 1` the kernel takes the
   `if gdx == 1` arm at `:564` and the mutex array is never touched.
2. `bindings/build.sh:278` REFUSES any `MOJOLEARN_NUMERIC_MODE` other than `identical`
   for this binding, by DEVIATION 2490: "only the tree lanes (gbdt, rf, trees) ship fast
   and deterministic". The fused kNN lives in `_mojolearn.so`, which is that binding. So
   there is no shipped FAST arm in which the pin is off.

`git grep -n "fused_l2_knn_launch("` outside `bench/results` returns four call sites: the
definition, the one call inside `fused_l2_knn` (which applies the pin), and three inside
`neighbors/checks/knn_check.mojo`. **No shipped path reaches the mutex.**

So the kNN claim is not a second LIVE instance of the shipped defect. It is a latent one:
correct to have repaired, reachable only by a check or by a FAST build that no build
script will produce. The measurement below is therefore of the check path, which is the
only path that reaches it, and a null there is a null about a path no user runs.

Note also that at `grid_x > 1` the kNN merge is ALREADY documented nondeterministic for a
second, unrelated reason: the FAISS comparator compares the distance only, so which of
several equidistant neighbours survives is decided by mutex order
(`fused_l2_knn.mojo:879-895`). Any kNN experiment must use a tie-free fixture or it
cannot separate the ordering defect from that acknowledged order-dependence.

### ExtraTrees: reachable, but only above 512 sampled columns on AMD

`extratrees/impl/decisiontree/batched_levelalgo/builder.mojo:3860` and `:5144`:

    var bpn = ceildiv(Int(k), TPB)
    ...
    grid_dim=(bpn, n_nodes, 1)

`k` is the sampled column count, `max(1, max_features * n_cols)`
(`builder.mojo:1033-1039`). `TPB` is `DEVICE_TPB`, and `_device_tpb()` at `:2922` returns
`128 if WARP_SIZE <= 32 else 512`. Its own docstring states the consequence outright:
**"the reduce runs one block per node at every `k <= TPB`"**.

One block per node is one claimant per mutex. The ordering hole needs two holders of the
SAME mutex, so below the threshold the defect cannot fire however long it runs.

- **gfx942 (WARP_SIZE 64): `TPB = 512`. ExtraTrees needs more than 512 sampled columns.**
- NVIDIA and Apple (WARP_SIZE 32): `TPB = 128`, threshold 128.

The forest's threshold is ELEVEN columns. ExtraTrees' is FIVE HUNDRED AND TWELVE on the
one column where the defect has ever been seen. That is a 47x difference in exposure
between two sites the reconciliation described as carrying "the identical protocol".

**Consequence for the experiment.** The forest lane's `wide` fixture is 16 columns. Run
ExtraTrees on it and `bpn == 1`, the mutex is claimed by exactly one block per node, and
the arm returns 0/300 no matter what the memory model does. That null would be worth
nothing and would have cost a box. An ExtraTrees fixture must carry **more than 512
columns at `max_features=1.0`** to contend at all on gfx942.

`-D MOJOLEARN_ET_TPB_128` (`builder.mojo:2922`, an existing documented A/B define, set by
no build script) forces the 32-lane width on a 64-lane device and drops the threshold to
128. It is the exposure dial, the way `N_BLKS_FOR_COLS` was the forest's.

## Status

Reachability established locally, no box spent. Measurement pending.
