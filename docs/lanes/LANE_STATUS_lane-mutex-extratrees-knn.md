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

## THE POWER, COMPUTED BEFORE THE BOX REPORTED

Written and committed while the MI325X leg was still bootstrapping, at 2026-09-16T18:02Z,
so it cannot have been chosen to suit a result. The rate to beat is the forest's measured
4.3% to 5.3% of fits on the shipped 0.8.5 wheel.

| arm | N | E[moves] at 4.3% | P(0 moves) at 4.3% | P(0) at 1% | 95% bound on a 0/N |
|---|---|---|---|---|---|
| ET stock, 2048 cols | 600 | 25.8 | 3.5e-12 | 0.0024 | 0.50% |
| ET stock, 2048 cols, if the deadline truncates to ~280 | 280 | 12.0 | 4.5e-06 | 0.06 | 1.07% |
| ET control, 256 cols, bpn = 1 | 300 | 12.9 | 1.9e-06 | 0.049 | 1.00% |
| fused kNN, launches | 300 | 12.9 | 1.9e-06 | 0.049 | 1.00% |

The cells are deadline bounded, not count bounded, so the achieved N is whatever the
lease allowed and `runs` in the JSON records it. Read the achieved N, not the planned one.
An ExtraTrees fit of this configuration measured 2.5 s on the M4 at 2048 columns, so a
700 s cell is about 280 fits if the MI325X is no faster.

**What a null would and would not say.** A 0/280 excludes the forest's rate at p about
5e-6. It does NOT clear the protocol: it bounds ExtraTrees' rate at roughly 1% by the 95%
rule of three, and the forest's own rate at the LOW end of its range was 1.3% in one
configuration. So a null at this N is consistent with ExtraTrees carrying the same defect
at a rate this leg cannot see, and the honest sentence is the bound, not the zero.

**The 0/100 trap, named because this lane family has already fallen into it.**
`docs/lanes/LANE_STATUS_lane-rf-score-weighted-nondeterminism.md` records a 0/100 that was
flagged as ~7% likely by luck and used to retract a hypothesis IN THE SAME MESSAGE; the
same cell returned 4/300 on the next leg. No cell here is read below 250.

## THE RESULT. Both subsystems are NULLS, and the two nulls mean different things

**DigitalOcean MI325X VF, gfx942, 2026-09-16T17:59:34Z to 18:22:36Z, 23 minutes of lease.
Droplet 601138387, DELETE returned 204 and the post-destroy GET returned 404.** Evidence at
`/Users/andrewhendel/mojolearn-evidence/mutex-extratrees-knn/gfx942-2026-09-16/` and
`bench/results/e1g/2026-09-16_175653-amd-mi325x-et-knn-mutex/remote/`.

### ExtraTrees, gfx942

| arm | cols | bpn | contends | fits | moved | distinct | P(0) at the forest's 4.3% | 95% bound |
|---|---|---|---|---|---|---|---|---|
| **stock, no acquire** | 2048 | 4 | YES | **545** | **0** | 1 | **4e-11** | **0.55%** |
| stock, same binary | 256 | 1 | no, one claimant | 300 | 0 | 1 | 1.9e-06 | 1.00% |
| repaired (ships) | 2048 | 4 | YES | 545 | 0 | 1 | 4e-11 | 0.55% |
| repaired (ships) | 256 | 1 | no, one claimant | 300 | 0 | 1 | 1.9e-06 | 1.00% |

`distinct = 1` in every cell: not one of the 1690 fits produced a model that differed in
any of the five exported arrays from the first fit of its cell.

### fused L2 kNN, gfx942, at a grid no shipped build ever takes

| arm | grid_x | producers per mutex | launches | wrong vs oracle | moved vs first |
|---|---|---|---|---|---|
| **stock, no acquire** | 16 (computed) | 16 | 300 | **0** | **0** |
| repaired (ships) | 16 | 16 | 300 | 0 | 0 |
| stock, no acquire | 8 (forced) | 8 | 200 | 0 | 0 |
| repaired | 8 (forced) | 8 | 200 | 0 | 0 |
| control, mutex never touched | 1 | n/a | 76 and 51 | 0 | n/a |
| **positive control, sabotaged handoff** | 16 and 8 | | 1 each | **150** and **250** | |

### Apple M4, Metal, the same two probes

| subsystem | arm | N | moved |
|---|---|---|---|
| fused kNN | stock | 300 launches at 16 producers per mutex | 0 |
| fused kNN | repaired | 300 | 0 |
| ExtraTrees | repaired | 300 fits at 2048 cols, **bpn = 16** (TPB is 128 on a 32-lane warp) | 0 |

## WHAT THE GATES DID, because a count is unreadable without them

**The arms were two programs, checked by SECTION on the target before either was trusted,
with both controls.** On gfx942:

- NEGATIVE CONTROL. Two builds of identical source, same directory: `ALL SECTIONS
  IDENTICAL` for the kNN probe and `26 sections compared, 0 moved` for the trees binding.
  So `build_trees.sh`'s fresh `mktemp` does NOT leak into any compared section, and the
  comparator is not noisy.
- POSITIVE CONTROL. `-D MOJOLEARN_MUTEX_SECTION_SABOTAGE=1` moved `ELF,.text` and
  `ELF,.rodata`. The comparator can report DIFFER on this target.
- THE ARMS. Repaired against stock moved `ELF,.text` and `ELF,.rodata`, with `.rodata`
  **128 bytes smaller** in the stock arm, which is the acquire spin the repair adds. The
  trees binding's arms moved `ELF,.rodata` and `ELF,.strtab`.

**The kNN probe's own controls behaved on every run.** The `grid_x = 1` arm, which is the
same binary with the mutex never touched, matched the host oracle in every slot; the
sabotaged handoff moved 150 slots at `grid_x = 16` and 250 at `grid_x = 8`. So the check
CAN fail and the merge IS carrying the output.

**And the positive control failed the first time it was run, on this desk, before the
box.** `$def` unquoted in a zsh loop does not word-split, so
`-D MOJOLEARN_MUTEX_SECTION_SABOTAGE=1` reached `mojo` as one argv entry and was dropped;
the two "arms" were one program and the comparator correctly said SAME. That is the fourth
appearance of this failure class in this lane family and the first time it was caught
before a box was rented.

## READ THESE TWO NULLS DIFFERENTLY

**ExtraTrees is a real null with a real bound.** The stock claim, the exact pre-repair
protocol that 0.8.5 ships, ran 545 fits at a configuration that contends four ways per
node and did not move a bit. That excludes the forest's 4.3% at p = 4e-11 and bounds
ExtraTrees' own rate at **0.55%** by the 95% rule of three. **It is not a clearance.** The
forest's rate in its quietest measured configuration was 1.3%, and this leg would have
caught that; its rate at bpn = 4 could still be anything below 0.55%. The protocol was
invalid and the repair is right on the argument. What this measures is that ExtraTrees was
not visibly losing candidates at the rate the forest was.

**The fused kNN null is about a path no user can take, and that is the finding.** The
mutex is not reachable in any shipped build (the reachability section above), so the 0/500
launches say only that the defect does not fire on the check path. A user was never exposed
here. This is the sharper half of the lane: the reconciliation listed the fused kNN as one
of three subsystems "repaired by the argument" and carrying the same risk, and it does not
carry the same risk, because `PIN_DETERMINISM` compiles the caller down to `grid_x = 1` and
`bindings/build.sh` refuses every mode in which the pin is off.

## WHY EXTRATREES MIGHT NOT MOVE WHERE THE FOREST DOES

Stated as hypotheses, none of them measured here.

1. **Four claimants against the forest's ten.** `N_BLKS_FOR_COLS` caps the forest at 10
   column blocks per node and the forest's rate rose with launch count; ExtraTrees at 2048
   columns has `ceildiv(2048, 512) = 4`. Fewer claimants is less window.
2. **Different payload.** The forest merges histogram-derived candidates where near-ties
   are common at depth; ExtraTrees' candidates are random thresholds, and losing one only
   moves the output when the lost one was the winner.
3. **The forest's rate is itself unexplained.** Section 0.2 of the reconciliation says the
   mechanism is NOT DEMONSTRATED and the primitive check has returned a null twice at 4096
   claims. A null here is the third null from an instrument that is not the forest.

## STILL OWED BY THIS LANE

1. **A RUNTIME WITNESS that the ExtraTrees mutex was contended in the measured cells.**
   `bpn = 4` is arithmetic from the source, not an observation. The instrument exists and
   is already in the repository: `extratrees/checks/split_reduce_check.mojo` carries a
   `SPLIT_SAB_NO_LOCK` arm whose stated prediction is that publishing with no lock at all
   "moves only cells served by more than one block, and at least one of them". Running that
   check on gfx942 is the witness; it was not run inside this lease.
2. **ExtraTrees at more claimants.** `-D MOJOLEARN_ET_TPB_128` forces the 32-lane width on
   a 64-lane device and turns 2048 columns into `bpn = 16`. If the rate scales with
   claimants the way the forest's scaled with launches, that is where to look next, and it
   is one define and one more cell.
3. **The kNN producer at a FAST build.** Not owed for the shipped product, since no build
   script will produce one.
