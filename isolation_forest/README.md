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
tools/with_build_lock.sh     pixi run mojo run -I . isolation_forest/mojo_only/if_check.mojo
tools/with_identical_mode.sh pixi run mojo run -I . isolation_forest/mojo_only/if_check.mojo
```

Nine checks: the two xorwow gates, refusals, oracle semantics, device vs
oracle, launch invariance, signed zero, predict thresholds, the card. Under
`IDENTICAL` on an Apple M4 all nine are green and device equals the host
oracle on **37,496 structure / path-length / score cells, bit for bit**.

Under `FAST` `check_if_device_equals_oracle` is a REPORT, not a gate: the
device and the host disagree in the last ulp of some scores, which is what
FAST means.

## The oracle

`mojo_only/if_oracle.mojo` is a second, independent transcription, written
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

## The seed-truncation trap does not apply here

cuML's Random Forest is not reproducible across GPU *models* above
`n_sampled_rows > 4 * SM * 256`, because it truncates a 64-bit seed to 32
bits. Checked here rather than assumed: Isolation Forest calls
`curand_init(seed, tree_id, 0)` and `_curand_init_inplace` consumes **both**
halves of the seed (`s0` from the low 32, `s1` from the high 32). And
`subsequence = tree_id = blockIdx.x` is a semantic key, one block per tree,
not an SM-count-dependent one. There is no analogue of the RF defect to fix.

## What is owed

- The DEVIATION 750 probe above, on an NVIDIA box.
- NVIDIA and AMD legs. Everything here is Apple M4 only.
- A second sabotage on the split tie-break (each sabotage needs its own
  rebuild; only one was run).
- The shape sweep and the repeat-launch gate beyond what
  `check_if_launch_invariance` already covers.
- A `pixi.toml` task line and an `IDENTITY_PATHS.md` row; this lane does not
  edit those files.
