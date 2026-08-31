# Isolation Forest

A mirror port of cuML's Isolation Forest, pinned at
[rapidsai/cuml](https://github.com/rapidsai/cuml) v26.08.00
(`265b9da6a0e75dbef071a3168398b993a5ff6f0e`), read from
`/Users/andrewhendel/CascadeProjects/upstream/cuml-v26.08.00`.

> The algorithm does NOT exist in cuML 25.08. If you are reading
> `/Users/andrewhendel/CascadeProjects/upstream/cuml` (branch-25.08) there is
> no `cpp/src/isolation_forest` there at all. Use the v26.08.00 mirror.

The RNG is cuRAND's XORWOW, read from
`/Users/andrewhendel/CascadeProjects/upstream/curand-headers`.

## Where the identity actually lives

Not in the tree math. Isolation Forest splits on a **random feature** at a
**random threshold**, so the whole forest is a pure function of the RNG
stream. If the stream diverges by one draw, every downstream bit is
different and no amount of arithmetic pinning saves it. The float seams
(one `log`, one multiply-add, one `pow`) matter, but they matter *second*.

So the lane is built around the stream:

1. `xorwow_reference.py` rebuilds cuRAND's two 32x800 skip-ahead tables from
   the XORWOW step matrix and checks them against `curand_precalc.h`'s own
   25,600 constants, per table, word for word. Both tables match.
2. It emits two reference TSVs of raw draws.
3. `check_if_xorwow_matches_curand` runs the ported RNG on the **host**
   against those TSVs, draw for draw, including the uniform's float bits.
4. `check_if_xorwow_on_device` runs the same triples **through a kernel**,
   because the stream the forest is actually built from walks
   `_curand_matvec_inplace` 160 bit-tests deep against a global-memory table
   inside a GPU compiler's optimizer.
5. Only then does `check_if_device_equals_oracle` compare whole forests.

## The one thing that is not settled

**DEVIATION 750.** cuML's `curand_u64` is

```c
return (static_cast<uint64_t>(curand(rng_state)) << 32) | curand(rng_state);
```

The two `curand()` calls are operands of `|`, and C++ does not sequence them
relative to each other. Which draw becomes the high 32 bits is the
compiler's choice and **both choices conform**. Every index in the forest
comes out of `sample_bounded`, and `sample_bounded` is built out of
`curand_u64`, so the two readings give two *different forests* from the same
seed. Not a rounding difference: a different tree.

This port takes the first draw as the high word, by name, in two locals so
that nothing is left to a compiler. Whether that is what nvcc does has
**not been checked against a cuML binary**. Until it is, every gate in this
lane is us agreeing with us.

Closing it needs one number off a real GPU. Fit cuML with `seed = 42`,
`n_estimators = 1`, `bootstrap = true`, `max_samples = 1` on 1000 rows and
read `tree_sample_indices[0]`:

| `tree_sample_indices[0]` | verdict |
| --- | --- |
| `408` | high word first, this port is right |
| `23` | low word first, flip `curand_u64` |

The swapped reading is built in as `-D MOJOLEARN_IF_SABOTAGE_U64_SWAP=1`.

## Running the gate

One build at a time.

```sh
tools/with_build_lock.sh     pixi run mojo run -I . isolation_forest/checks/if_check.mojo
tools/with_identical_mode.sh pixi run mojo run -I . isolation_forest/checks/if_check.mojo
```

Nine checks: the two xorwow gates, refusals, oracle semantics, device vs
oracle, launch invariance, signed zero, predict thresholds, the card. Under
`IDENTICAL` on an Apple M4 all nine are green and device equals the host
oracle on **37,496 structure / path-length / score cells, bit for bit**.

Under `FAST` `check_if_device_equals_oracle` is a REPORT, not a gate: the
device and the host disagree in the last ulp of some scores, which is what
FAST means.

## The oracle

`checks/if_oracle.mojo` is a second, independent transcription, written
from the `.cuh` again rather than by calling the device function on the
host. It recurses over `List`s where the device walks an explicit stack over
four flat arrays. That is deliberate: the device builder's pointer
arithmetic is exactly where a mis-port hides, and a gate that runs the same
function twice cannot see it.

It shares exactly one thing with the port, the RNG, which is why the RNG is
gated separately and first.

It reproduces their **in-place swap** partition rather than a stable one,
because which of two equal values a node's strict `<` fold calls its min
depends on the order rows reach the next level.

## Sabotage

`-D MOJOLEARN_IF_SABOTAGE_U64_SWAP=1` takes `curand_u64`'s two draws in the
other order. Measured, both arms:

- `check_if_device_equals_oracle` goes **red** under IDENTICAL, on tree 0 of
  the first fixture, on `n_nodes` (41 vs 35). The forest changes shape, not
  just its last bits, which is the point: an RNG perturbation is not a
  rounding perturbation.
- The two xorwow gates stay **green**, because they do not go through
  `curand_u64` at all.

That split is the per-branch reach evidence: the sabotage lands where it
should and nowhere else.

## Deviations

| # | what |
| --- | --- |
| 680 | non-finite inputs refused by name on the host (theirs accept NaN and split on it) |
| 681 | the score's `pow(2, y)` is `identical_pow`; `c(n)` uses `identical_log64` |
| 682 | the builder's and scorer's float seams: `identical_log`, `identical_mul_add`, `ftz` at every stored float |
| **1942** | **the feature matrix is flushed at `_upload_f32` under the pin (2026-08-29). Until then 682's seams covered every float the builder STORED and none it READ, and the AMD MI325X leg scored subnormal features raw where Apple's hardware flushed them, and `score_samples` diverged on `identity_break`'s `denormal` fixture. Bit-inert on Apple; the AMD and NVIDIA re-measurements are owed** |
| 683 | the two precalc tables are rebuilt from the step matrix, not embedded, and live in global memory |
| 684 | `max_depth` auto is an integer `ceil(log2(n))`, not a libm `log2` |
| 685 | node storage is four arrays, not an array of `IFNode` |
| 686 | `size_t` indices are `Int64`; `StackEntry` is four `Int32` words |
| **750** | **`curand_u64` draws the high word first, by choice, because theirs does not say** |
| **751** | **the offset skip-ahead table was built, uploaded and never stepped; it now has its own reference and its own device gate** |

680-686 were assigned by an earlier brief and are outside this lane's
current range (750-769). They are unique in the repository and were left at
their original numbers rather than renumbered, since they are already
written into five files.

### DEVIATION 751, and why it mattered

cuML always calls `curand_init(seed, tree_id, 0, ...)`. The `offset` is
always zero, so `_skipahead_inplace` runs its loop **zero times** and
`precalc_xorwow_offset_matrix` is dead weight in every kernel launch.

That is *reached but inert*: the table was computed, uploaded, and passed as
a kernel argument, and no gate in the lane could have told a correct offset
table from garbage. Half of DEVIATION 683's claim ("the rebuilt tables equal
the header's") was unverified, because the Python reference checked only the
sequence table.

Closed rather than deleted, since the table is part of the ported
`curand_init` contract: the reference now checks both tables against the
header, asserts that skipping *k* equals stepping *k* times, and the device
gate runs 6 x 256 draws at nonzero offsets and confirms that offset 1 is
**exactly a one-draw advance** of offset 0. That is the single-draw
perturbation this whole lane is built to detect, planted deliberately and
observed.

## The dangling-node branch is UNREACHABLE (adjudicated 2026-08-24)

The repo-wide card audit (`CARD_GAPS.md:232-236`) flagged
`isolation_tree_builder.mojo`'s `if node_idx >= max_nodes_per_tree:
continue` as a possible defect: a node silently dropped, left at the poison
fill, with a parent still pointing at it and no gate able to see it.
Adjudicated here with the three questions that decide it.

**Is upstream the same?** Yes, verbatim:
`isolation_tree_builder.cuh:158` is `if (node_idx >= max_nodes_per_tree) {
continue; }`. We did not invent it. So DEVIATION policy is not engaged:
there is no bug of theirs to fix and nothing of ours to number.

**Is it reachable?** No, and the proof is three lines.
`max_nodes_per_tree = min(2*max_samples - 1, 2^(max_depth+1) - 1)` with
`max_samples >= 1` and `max_depth >= 0` both enforced by a raise, so it is
at least 1. Exactly three sites ever push a node index: the root push of
`0`, and the two child pushes. `0 < max_nodes_per_tree` because the bound
is at least 1. The child pushes sit past the capacity arm of the stopping
condition, so when they run `n_nodes + 2 <= max_nodes_per_tree`, and they
push `n_nodes` and `n_nodes + 1`, both `< max_nodes_per_tree`. An index
does not change after it is pushed. Therefore every popped `node_idx` is
below the bound and the `continue` never runs.

**What would happen if it did?** It would be a real defect, not a hash gap.
Under the all-zero poison the dangling child reads feature 0, threshold
0.0, left 0, right 0, and a traversal that reaches it loops back to the
root forever: a hang, not a wrong answer. Under the NaN poison
(`0xffc0dead`) the feature word is negative, the node reads as a leaf, and
the traversal silently returns the poison as a path length: a wrong score.

The audit's secondary claim, that the card could not see it, **is wrong**.
A dropped index is always `< n_nodes`, because it was produced by
incrementing `n_nodes`, and the card records `for i in range(n_used)` with
`n_used = n_nodes`. A poisoned dangling child lands inside the hashed range
and would diverge from the oracle on `structure.feat` at that index.

**Verdict: UNREACHABLE and CLOSED.** Not "ungated" -- calling it ungated
invites the next reader to write a gate that can never fire. The proof is
kept in the source beside the branch so it is re-checked whenever the bound
or the push sites change.

## The seed-truncation trap does not apply here

cuML's Random Forest is not reproducible across GPU *models* above
`n_sampled_rows > 4 * SM * 256`, because it truncates a 64-bit seed to 32
bits. Checked here rather than assumed: Isolation Forest calls
`curand_init(seed, tree_id, 0)` and `_curand_init_inplace` consumes **both**
halves of the seed (`s0` from the low 32, `s1` from the high 32). And
`subsequence = tree_id = blockIdx.x` is a semantic key, one block per tree,
not an SM-count-dependent one. There is no analogue of the RF defect to fix.

## What the card records, and the three washers it now sees behind

Per tree: `rows`, `features`, `meta`, `structure.{feat,thr,left,right}`,
and, added 2026-08-24 in answer to `CARD_GAPS.md`:

| stage | dtype | what |
| --- | --- | --- |
| `split.bounds` | Float32 | `min_val`, `max_val`, `rand_frac` per node |
| `split.choice` | Int32 | `feature_start`, `flags`, `local_feature` per node |
| `rng.final` | Int32 | the tree's finishing XORWOW state, `d` and `v0..v4` |

`structure.thr` was a hash taken on the far side of **two** washers:

1. **Absorption.** `threshold = fma(rand_frac, ftz(max_val - min_val),
   min_val)`. When `(max - min)` is small against `|min|` the whole product
   is absorbed and a divergence in *either* bound yields the identical
   recorded threshold. `split.bounds` records the two bounds and the draw,
   so the divergence has somewhere to land.
2. **Repair.** When the drawn threshold falls on an endpoint the fallback
   sets `threshold = max_val` and repartitions, overwriting what was
   recorded. Whether that repair fired was recorded nowhere. It is now bit
   32 of `flags`.

`flags` also carries which stopping arm fired at each leaf (bits 2, 4, 8,
and 16 for "no feature had `min < max`"). Every arm is evaluated, not just
the first one `or` stops at, because when two hold the leaf written is
**byte-identical either way**. That is the `CRIT_ORDER` shape holtwinters
measured to move zero cells and to be catchable only by a recorded
decision.

`rng.final` closes the other half of the RNG question. `if.rng.probe`
verifies the **port** (that our XORWOW is cuRAND's). It says nothing about
the stream **position** any tree actually reached. A vendor that took one
extra rejection in `sample_bounded`, or one fewer, lands in `rng.final`
and nowhere else.

All of it is a decision the **algorithm** makes, never one the scheduler
makes: every word is a pure function of `(seed, tree_id, data bits)`, and
the node count is a pure function of `(max_depth, max_samples)`. None of it
is the machine-sized scratch `core/identity_trace.mojo` rule 3 forbids.

### One kernel argument, not three: Metal caps a kernel at 31

`build_isolation_trees_global_kernel` stands at **25 arguments** and is
unchanged at 25 after this work. Six new output buffers would have put it
at exactly 31, which is where holtwinters died with a metallib error naming
no argument and pointing at line 1 of an unrelated file.

So the new outputs **reuse a slice**. The existing per-tree scratch buffer
is widened and carved into three disjoint regions:

```
[0,        4*mn)          the stack, 4 Int32 per entry
[4*mn,    10*mn)          the per-node decision records, 6 Int32 per node
[10*mn,   10*mn + 6)      this tree's final RNG state
```

where `mn = max_nodes_per_tree`. Stride per tree is `mn * 10 + 6`.

## What is owed

- The DEVIATION 750 probe above, on an NVIDIA box.
- NVIDIA and AMD legs. Everything here is Apple M4 only.
- **A gate run for the card stages added 2026-08-24. THEY HAVE NEVER BEEN
  COMPILED.** They were written in a read-only slot, so `if_check.mojo` has
  not been built since. Three things must be checked the moment a slot
  opens: that it compiles; that `check_if_device_equals_oracle` still shows
  37,496 cells bit-equal (recording already-computed locals must not move a
  single bit, and if it does, the instrument is itself an identity hazard
  and must come back out); and that `check_if_card_is_emitted` sees the new
  stage count.
- The oracle half of those stages. `if_oracle.mojo` has every one of these
  values as a local (`min_val`, `max_val`, `frac`, `feature_start`,
  `local_feature`, and the repartition test at its `if left_end == start or
  left_end == end`), but does not record them, so the new stages are
  cross-vendor instruments only and are not yet checked against the
  independent transcription locally.
- A second sabotage on the split tie-break (each sabotage needs its own
  rebuild; only one was run).
- The shape sweep and the repeat-launch gate beyond what
  `check_if_launch_invariance` already covers.
- A `pixi.toml` task line and an `IDENTITY_PATHS.md` row; this lane does not
  edit those files.
