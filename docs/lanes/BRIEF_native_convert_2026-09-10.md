# LANE BRIEF, native host converters (cast and cast-plus-transpose)

Written 2026-09-10. Self-contained. Read this file and nothing else is needed.

**SUPERSEDED IN ONE RESPECT, 2026-09-10 (later the same day):** the
pure-Python fallback arms this brief demanded below ("THE FALLBACK IS NOT
OPTIONAL", gate step 2) were REMOVED from the package. The package cannot
import without its binding, so a missing symbol can only mean a binary on
disk older than the Python beside it, and `_buffer._native` now raises by
name with the rebuild command instead of running a slower copy of the same
arithmetic. The converter tests compare against NumPy alone; the nonzero
scan test (DEVIATION 2489) keeps its retired Python loop as its oracle.
The rest of this brief stands as the record of what was built.

**EXECUTED 2026-09-10, same day, on `main`: e9f40d69 (the two helpers),
313ce4a1 (anonymous-mapping output store, flat cast matches NumPy),
485caa24 (float32 transpose, the conditional third helper, which the gate
showed was needed). Gate: `python/mojolearn/tests/test_native_convert.py`,
119 passed. Numbers:
`bench/results/native_convert_2026-09-10/run5_m4_warmup.txt`, reproduced
by `time_converters.py` beside it. Outcome against the bar below: flat
cast EQUALS NumPy (2.95 vs 2.88 ms, 12.07 vs 12.05 ms); every layout flip
BEATS it (5.06 vs 12.81, 24.5 vs 128.1, 4.57 vs 7.23, 20.0 vs 53.7 ms).
Two things the brief got wrong, fixed in place: the colmajor oracle needed
`tobytes(order="F")`, and the timing protocol needed a warm-up round
because a cold first NumPy sample voided windows the box was not heating.
One thing the brief did not foresee: the flat cast could not match NumPy
until the OUTPUT ALLOCATION stopped zero-filling; `_buffer._AnonStore`
(an anonymous mapping, 3.12+) is that fix.**

## Why this lane exists

`numpy-free-0.7` replaced NumPy in the Python layer with PURE PYTHON. That is
the wrong substitution. It gives up NumPy's C-level cast and buys only the
dependency removal. Measured on the M4 at 2,000,000 rows by 20 columns,
float64 C-order input:

| operation | NumPy | the pure-Python converter |
|---|---|---|
| cast only, `as_f32_c` | 5.7 ms | 1,235 ms |
| cast and transpose, `as_f32_colmajor` | 119.4 ms | 1,424 ms |

Float32 input already in the target order is a ZERO-COPY BORROW on both
sides and costs nothing. The gap above is float64 input only, which is what
sklearn and pandas hand a caller by default.

This was never caught because the bench harness does
`X = np.asarray(X, dtype=np.float32)` and the surface tests are float32
nearly everywhere, so every conversion the branch already did was exercised
only on the input shape where the change is free.

Your job is to close the gap in Mojo, so the Python layer can drop NumPy
without paying for it.

## What you are building

Two host helpers in `bindings/_mojolearn.mojo`, plus the Python side that
calls them.

**`cast_f64_to_f32_binding(src_addr, dst_addr, n)`** reads `n` float64 values
at `src_addr` and writes `n` float32 values at `dst_addr`. Returns 0.

**`cast_colmajor_f64_to_f32_binding(src_addr, dst_addr, rows, cols)`** reads a
C-contiguous float64 `[rows, cols]` matrix at `src_addr` and writes a
COLUMN-MAJOR float32 matrix at `dst_addr`, so that
`dst[c * rows + r] == Float32(src[r * cols + c])`. Returns 0. Do this as ONE
fused pass, not a cast followed by a transpose.

Add float32 source variants of both if and only if the gate below shows they
are needed. Do not add anything else.

## The template to follow, exactly

READ THIS PART CAREFULLY, THE TEMPLATE IS NOT ON `main`.

The three host helpers from DEVIATION 2303 (`all_finite_f32`,
`all_finite_f64`, `column_mean_f64`) exist ONLY on branch `numpy-free-0.7`,
which has not landed. `origin/main` has none of them. Read them with:

    git show numpy-free-0.7:bindings/_mojolearn.mojo

and look at `all_finite_f64_binding` and `column_mean_f64_binding`. Copy
their shape: the negative-count refusal spelled by name, the
`with GILReleased(Python()):` block around the loop, and the `PythonObject`
return. Match that style, including the refusal wording.

`_f32_ptr` DOES exist on `origin/main` and you should use it as-is.
`_f64_ptr` DOES NOT. It is about six lines on the branch and your helpers
need it, so port that one function across as part of this lane. Port it
BYTE-FOR-BYTE, same name, same placement right after `_f32_ptr`. The
numpy-free branch's Mojo commits will be replayed onto `main` later and
they add the same function; an identical copy resolves as a no-op, a
reworded one becomes a conflict for that lane to clean up.

Symbol names are yours alone: `cast_f64_to_f32_binding` and
`cast_colmajor_f64_to_f32_binding` exist on no branch today, checked
2026-09-10 across every local and remote ref. No open worktree has
uncommitted edits to either of your two files.

DO NOT port the three DEVIATION 2303 helpers themselves. They are separate
owed work and dragging them in widens this lane for no gain. You need
`_f64_ptr` and nothing else from that branch.

Assign these DEVIATION numbers: 2470 for the flat cast, 2471 for the fused
column-major cast.

## The Python side

`python/mojolearn/_buffer.py` already has the lookup pattern. Read
`_native_all_finite`. It asks `_backend.binding("_mojolearn")` for the symbol,
caches it in `_NATIVE`, and returns `None` when the binding is not built.

Wire the two new helpers the same way, into `_convert` and the
`as_f32_colmajor` path. THE FALLBACK IS NOT OPTIONAL. When the symbol is
missing the existing pure-Python path must still run and still produce the
same bytes. That is what makes this lane additive and unable to break an
unbuilt checkout.

## The gate, in this order

**1. Bit exactness, before any timing.** The helper's output must be
byte-identical to NumPy's, not close to it. The float64 to float32 cast is
one IEEE round-to-nearest-even per element and Mojo's `Float32(x)` is that
same hardware cast, so this should pass on the first build. If it does not,
STOP and report, do not add a tolerance.

    ref = np.ascontiguousarray(x, dtype=np.float32)
    assert bytes(ours) == ref.tobytes()

    ref = np.asfortranarray(x, dtype=np.float32)
    assert bytes(ours_colmajor) == ref.tobytes(order="F")

(`ndarray.tobytes()` serializes in C order WHATEVER the memory layout, so
without `order="F"` that second line compares column-major bytes against
row-major bytes and fails on every matrix with both dimensions above 1.
The first draft of this brief had exactly that bug and the first gate run
showed 30 failures that were all the oracle's.)

Cover, for both helpers: float64 input, an empty buffer, a single element, a
single row, a single column, values that round exactly halfway (so
round-to-nearest-EVEN is actually exercised and a round-half-away
implementation fails), values that overflow float32 to +/- inf, subnormals,
NaN and both infinities. A halfway case that passes under both rounding
rules proves nothing, so pick ties whose two candidate answers differ.

**2. The fallback agrees.** With the binding present and with it forced
absent, the same input must produce the same bytes.

**3. Only then, timing.** See the protocol below.

## The timing protocol, and why it is written out

The M4 drifts about 1.7x within twenty minutes because heat pins the GPU
governor at minimum. A before number taken cold and an after number taken
warm is not a comparison. So:

  * Alternate the arms INSIDE one window. Run NumPy, then ours, then NumPy,
    then ours, and keep going for at least five pairs.
  * Report the MINIMUM of each arm, never the mean. The minimum is the
    least thermally damaged sample.
  * Report the pair count and the spread between first and last NumPy
    sample. If that spread is more than about 20 percent, the box was
    drifting and the window is void. Say so and take it again.
  * Shapes: 1,000,000 x 10 and 2,000,000 x 20, float64 C-order, for both
    helpers. Add the float32 F-order case to confirm the zero-copy borrow
    still costs nothing.

MEASURE THE CONVERTER DIRECTLY, not an estimator fit. `_buffer.py` is on
`main` but NOTHING IMPORTS IT YET, so no estimator will get faster from this
lane and looking for that is a wasted afternoon. Call `as_f32_c` and
`as_f32_colmajor` yourself on the shapes above and time those calls.

## Targets, and what counts as success

  * Flat cast: MATCH NumPy's 5.7 ms, within about 1.5x. It runs at 84 GB/s,
    which is memory bandwidth, so there is nothing to win. Matching is the
    whole goal.
  * Fused column-major: BEAT NumPy's 119.4 ms. This one is winnable and it
    is the point of the lane. NumPy's fused call is slower than doing the
    two steps separately (5.7 ms cast plus 46.3 ms transpose is 52 ms
    against its 119.4 ms), so its fused path is leaving a lot on the floor.
    The cost there is cache behavior, not bandwidth. Copy in TILES, roughly
    a 256 KB working set, so the strided writes stay cache-local.
  * A result slower than the pure-Python path it replaces means the wiring
    is wrong, not that the idea is wrong. Report it, do not ship it.

This also restores something. `main`'s `_arrays.as_f32_colmajor` has a
row-tile arm for large C-order inputs, DEVIATION 1887, that the pure-Python
converter does not carry. A tiled native helper puts that behavior back in a
better place.

## Constraints

  * WORK IN YOUR OWN GIT WORKTREE off current `origin/main`. Do not touch
    `/Users/andrewhendel/CascadeProjects/mojolearn`. That checkout has
    uncommitted work in `checks/` and `bench/results/` and about forty other
    worktrees hang off it.
  * You touch exactly two files, `bindings/_mojolearn.mojo` and
    `python/mojolearn/_buffer.py`, plus one new test. Neither is contested
    against `origin/main`. Do not edit `_arrays.py`, `_byte_lm_impl.py`,
    `_training_impl.py`, `_samba_impl.py`, `extratrees.py`, `randomforest.py`
    or `ensemble.py`. Those are contested or are separate owed work.
  * Do not convert any estimator in this lane. This lane makes the
    conversion cheap. The conversions themselves are a later stage.
  * Do not change any default. Nothing selects the native path on its own
    until the gate above is green and the numbers are recorded.
  * Build with `bindings/build.sh`. Read the resulting arch back rather than
    assuming it.
  * Report a RUN OWED line for anything you did not execute, naming the
    exact command.

## Background you may want

  * `python/mojolearn/NUMPY_FREE_CONTRACT.md`, what the buffer layer promises.
  * `docs/lanes/NUMPY_FREE_RESIDUAL_2026-09-10.md`, the staged landing plan
    and what each stage owes.
  * `python/mojolearn/_array.py`, `_buffer.py`, `_bufcheck.py`, `_labels.py`,
    landed 2026-09-10 and currently unused by anything.
