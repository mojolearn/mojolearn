# lane/umap-batch-determinism, 2026-09-16

## The question

A source review reported that UMAP's saved-model transform is batch dependent,
so changing request batching can change a row's embedding, and asked whether
that is a determinism defect in a shipped inference path.

**It is true, and it was already true and already written down.** What this
lane adds is not the fact but its SHAPE: the four couplings the repo declares
are not equal, two of them behave nothing like the declaration implies, and
the effect is structural rather than float chaos. That last contrast is the
part that decides how seriously to take it.

## The measurement

`umap/checks/batch_determinism_check.mojo` on the CPU host route
(`umap/host/umap_oracle.mojo::host_umap_transform`, the route a saved model
serves on a CPU-only install). `umap/checks/batch_determinism_device_check.mojo`
runs the same arms on the shipped device route
(`umap/transform.mojo::transform` through a `DeviceContext`).

    MODULAR_HOME=<pixi default env>/share/max \
    MAC_SLOTS=4 bash ~/mojolearn-evidence/tools/mac_slot.sh run nice -n 19 \
      mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . \
      umap/checks/batch_determinism_check.mojo

Log: `~/mojolearn-evidence/umap-batch-determinism/cpu-host-nsr.log`.

Fixture, chosen against the standing rule that uniform data hides permutation
effects: 48 training rows in 3 features as four separated clusters with
jitter, a 2D training embedding whose clusters sit about 11 units apart, and
8 queries with real spread including a far outlier at (20, 20, 20). k = 5,
`n_epochs` unset (so 100 refinement epochs), seed 7, IDENTICAL mode, one core.
Embeddings are compared as UInt32 bit patterns, never by tolerance, and every
differing value is printed.

### The comparison was made able to fail before it was trusted

`REPEAT` runs the same batch twice and is bitwise equal on all 8 queries. That
alone proves nothing, so a `ULP` ladder doubles a perturbation of query 0's
first feature until the comparison reports a difference:

    ARM ULP nsr5 inert at 1 ulps      ... inert at 2, 4, 8, 16, 32, 64, 128,
                                          256, 512, 1024, 2048, 4096, 8192
    ARM ULP nsr5 fired at 16384 ulps; other rows moved: False
    DIFF ULP16384.nsr5 query 0 component 0 a -5.8975425 b -5.89251

The `FLOOR` arm below returns a null, so it carries its own control that is
seen to fire. Three checks in this repo on 2026-09-16 were found unable to
fail; neither of these two is.

### The result, at the shipped default negative_sample_rate=5

| arm | what changed | moved? | largest difference |
|---|---|---|---|
| REPEAT | nothing | no | bitwise equal |
| SOLO | each query alone vs inside the batch of 8 | **yes, all 8** | query 4 component 1, `6.9431734` -> `4.974316`, delta `-1.9688573` |
| ORDER | the same 8 queries reversed, mapped back | **yes, all 8** | query 1 component 0, `-5.2577224` -> `-4.8005333`, delta `0.45718908` |
| COMPANY | query 3 at the SAME position in two groups | **yes** | `6.1502852` -> `6.220155`, delta `0.069869995` |

`COMPANY` is the decisive arm: query 3 sits at index 3 of both `[0,1,2,3]` and
`[4,5,6,3]`, so its in-batch ordinal is identical and only its company
changes, and it still moves.

The 1.97 is not noise. The training embedding's clusters are about 11 units
apart, so the row lands about a sixth of the way toward a different cluster.
It is still a visibly different coordinate for the same row, returned by the
same saved model, with no signal that anything changed. Whether a shift that
size flips a downstream decision was NOT tested here, and on this fixture no
query changed which training cluster it sits nearest to.

### The mechanisms, separated rather than listed

`tools/identity_break.py:4945-4968` declares four couplings. Running every arm
a second time at `negative_sample_rate=0`, which never consults the RNG,
separates them.

**1. The batch-position RNG ordinal (`umap/transform.mojo:146,154`). LIVE, and
it dominates.** `ORDER` is bitwise identical on every row at nsr=0 and moves
every row at nsr=5, so all of the reorder sensitivity, and the largest
numbers in the table above, are this one coupling. The counter is
`seed ^ (epoch * K1) ^ (edge * K2) ^ (slot - 1)` with `edge = row * k + j`,
and `row` is a position in the request, not a property of the query.

**2. The batch maximum edge weight (`:141`, used at `:147`). LIVE ALWAYS, at
milli-unit size here.** At nsr=0, `SOLO` moves 7 of 8 rows by `3e-4` to
`5e-3`. The eighth is query 2, bitwise unchanged, and the reason is exact:
`solo2` reports `max_weight_bits 1061399533`, the same bits as the full
batch. Query 2 IS the batch argmax, so it is the one row for which
`weights[edge] / maximum` does not move. That is positive attribution, not
elimination.

**3. The sigma floor's batch mean (`:44,66`). EFFECTIVELY INERT.** The `FLOOR`
arm holds the maximum at exactly 1.0 (a query that copies a training row has
a zero-distance neighbor) with nsr=0, so only the mean can act:

    FLOOR_SCALARS near  rows 2 mean    0.5835664629936218  max_weight 1.0
    FLOOR_SCALARS far   rows 2 mean   11.19422015249729    max_weight 1.0
    ARM FLOOR same maximum, different mean, moves the row: False

A 19-fold change in the batch mean moved no bit. `sigma = max(sigma, 0.001 * mean)`
binds only when the batch mean exceeds a thousand times the row's own sigma,
which the arm's own control then demonstrates:

    FLOOR_SCALARS absurd rows 2 mean 2591.928204527497 max_weight 1.0
    DIFF FLOOR_CONTROL query 0 component 0 a -5.2506504 b -5.2277403
    ARM FLOOR_CONTROL the sigma floor does bind and does move the row: True

So the declaration lists the sigma floor first, and it is the coupling that
almost never acts.

**4. The epoch count from `n_queries` (`:222-224`). NOT MEASURED HERE.** With
`n_epochs` unset the count is 100 at or below 10,000 queries and 30 above, so
it bites only at that one boundary. This lane read it in the code and did not
cross the boundary; it is a code fact in this write-up, not a measurement.

### This is structural, not float chaos

The easy dismissal is that floating point is chaotic, so of course batching
moves things. The ladder refuses that reading. Moving query 0's own feature by
8192 ULPs moved no output bit at all, and at 16384 ULPs it moved only query
0's own row by `0.005`. Putting a row in different company moved it by up to
`1.97`. The sensitivity to a row's own value is tiny; the sensitivity to its
neighbors in the request is four hundred times larger. The coupling is put
there by the algorithm, not by rounding.

## Refused by name, or silently tolerated?

**Neither, exactly, and the distinction matters.**

It is documented, thoroughly and accurately, in five places: the module
docstring of `umap/transform.mojo`, `UMAP.transform`, `UMAP.save` and
`UMAP.load` in `python/mojolearn/_umap_impl.py`,
`docs/lanes/CPU_INFERENCE_BOUNDARY_2026-09-15.md:111-115`, the declaration at
`tools/identity_break.py:4945-4968` with the reference library's coupling
cited line by line, and the generated `docs/VERIFICATION_MATRIX.md` rows for
`umap` and `par-graph-umap`. Every cited line number in that declaration was
re-checked against the current file and is correct. The headline effect was
already measured on the M4 on 2026-09-14.

At RUNTIME it is silent. Nothing raises, nothing warns, no parameter asks for
a batch-invariant answer, and there is no `warnings` call anywhere in the UMAP
surface. A serving system that changes its request batching gets different
embeddings for the same row and no signal of any kind.

That is the gap worth naming. The prose is complete; the code says nothing.

## Is this a violation of the repository's claim?

No, and saying so plainly is the point. The claim is that the same inputs give
the same bits on CPU, Apple, NVIDIA and AMD. The batch IS part of the input,
and all four columns agree for the same batch. Nothing here shows a column
disagreeing with another.

What it is instead is a hole in the practical promise a saved model makes to
an inference server, in the one direction users will actually hit: the same
row, asked twice, in two differently sized requests.

## The fix, scoped and NOT applied

Making `transform` exactly row separable, so that a batch of N is the
concatenation of N batches of one, is three one-line changes, each applied to
both spellings (`umap/transform.mojo` and the host restatement
`umap/host/umap_oracle.mojo:594,614,689,698,706`, which must stay bit
identical to it):

1. `:154` key the negative-sample counter on something batch invariant
   instead of `edge = row * k + j`. The batch-invariant data available inside
   `refine_transform` is the row's own neighbor indices and weights, so a hash
   over that row's k `(index, weight)` pairs is the contained spelling. Keying
   on `indices[row * k]` alone is not enough; two queries can share a nearest
   neighbor.
2. `:141` compute `maximum` per row, over that row's k weights, instead of
   over the whole batch.
3. `:44,66` compute the sigma floor's `mean` per row instead of over the whole
   batch. Measured above as almost never binding, so this one is for
   completeness rather than effect.

**The code change is contained. Its consequence is not.** All three move every
recorded UMAP transform cell on every column, which is a numeric contract
change to a shipped inference path and needs the Apple, NVIDIA and AMD columns
re-recorded. Under [[gpu-records-only-for-releases]] that happens once per
PyPI release, so this is a release decision and not a lane merge. It also
diverges deliberately from the reference implementation, whose own docstring
calls `transform()` stochastic; the standing rule is not to reproduce a
reference library's bug, and whether this counts as a bug or as UMAP's design
is exactly the decision being asked for.

The gate is already written. After the fix, `SOLO`, `ORDER` and `COMPANY` in
`umap/checks/batch_determinism_check.mojo` must all report `False` where they
report `True` today, while `REPEAT` stays equal and the `ULP` ladder still
fires. That is a sabotage arm divergent before and after, in one file, with no
new harness.

## Verification scope

`python3 tools/verify_lanes.py --changed-since origin/main --fixtures base --plan`
printed:

    # umap/checks/batch_determinism_check.mojo: NOT ATTRIBUTABLE: no lane's
    #   derived source set names it, so every lane
    # FALLING BACK TO EVERY LANE: the blast radius of the paths above could
    #   not be determined. This is a full sweep, not a narrow run.
    # 212 of 212 lanes selected

That is a full sweep and was NOT run. The fallback is a selector gap worth
handing to `lane/lane-selector`: a new file under `<algorithm>/checks/` is not
in any lane's derived source set, so adding a check file selects everything.

The narrow run, `--lane umap --lane par-graph-umap --fixtures base`, selected
2 lanes and REFUSED for want of a built identical binding
(`_backend.select()` raised `ImportError`). Per this repo's rule 0 a refused
lane is not a checked lane, so that is not evidence either way. This lane's
diff adds two check files and changes no product code, so no lane cell can
move.
