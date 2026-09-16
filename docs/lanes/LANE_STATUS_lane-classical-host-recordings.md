# lane/classical-host-recordings

Two things were owed and they shared a box: the k-means NVIDIA sabotage arm
that `lane/kmeans-save` reported honestly as never having run, and the
`bench/results/classical_host/` recording that `kmeans` waited on as the last
entry of `SAVED_MODEL_INFERENCE_OWED`. The AMD column two lanes had written
down as owed, and `spectral`'s x86 CPU column at 512 rows, came with them.

## What was in the tree before anything was rented

`tools/classical_host_gate.py`'s `LANES` did not carry a single k-means lane,
and `host_surface`'s core family declared none for inference. That is the same
code half `lane/saved-model-reference-gaps` had to close for the four predict
lanes: **a recording cannot exist for a lane the gate does not know**, so
`SAVED_MODEL_INFERENCE_OWED` could say "waiting on one thing, a GPU recording"
and no box could have answered it.

Six lanes are declared: `kmeans`, `kmeans-random`, `kmeans-array`,
`kmeans-weighted`, `kmeans-sqrt`, `kmeans-classic-pp`. NOT `kmeans-cosine`:
its fit is refused by name (`cluster/impl/kmeans_params.mojo::validate`), so
there is no model to save and no saved-model cell to record. One format,
`mojolearn-kmeans-1`, carries every metric and every start.

The probe is `identity_break`'s `_km_probe` pair, `(predict, transform)`, over
the WHOLE held-out draw rather than its first 256 rows; `do_record` refuses a
probe whose hash is not the lane's own `infer` cell, which is what would have
caught a 256-row mistake. Two surfaces are recorded beside the pair, `labels`
and `predict_training_rows`, because `_km_probe` asserts that a fitted model's
predict over the TRAINING rows is `labels_` bit for bit and a saved model
should be asked the same question.

The `record` crash fix from the previous lane was re-verified on both sides
BEFORE renting rather than rediscovered from a refusing box: the fixed tool
prints `record must run on a GPU box; this install is CPU-only`, and putting
`args.lane_rule_only` back raises `AttributeError`
(`~/mojolearn-evidence/classical-host-recordings/record-crash-two-sided.txt`).

## THE REHEARSAL PAID FOR ITSELF BEFORE THE FIRST POD

A full CPU rehearsal recording was made on the Mac, 54 cells, by running the
real `do_record` with only the vendor guard bypassed by name. Then its
`--every-fixture` sabotage arm, against a core host binding built
`-D MOJOLEARN_HOST_SABOTAGE=1`, read

    gate verdict SABOTAGE NOT CAUGHT ON FIXTURES <all 54>, exit 1
    unmoved: all 54

**The negative control for the gate this lane exists to build did not exist.**
`MOJOLEARN_HOST_SABOTAGE` has exactly one k-means arm, in `host_accumulate`,
and only the FIT walks that function; a saved model's `predict` and
`transform` never reach it. Renting on the strength of that arm would have
bought a green `--every-fixture` line that proved nothing.

### The fix, and the second measurement that shaped it

The first version ORed the family define into a label arm, and that turned
`lane/kmeans-save`'s clean identity reading into

    summary (infer/model): N/A=2, ONE-COLUMN=12     <- every cell REFUSED
    summary (batch): DIVERGENT=6, N/A=1

because `_km_probe` asserts `predict(X) == labels_` before it hashes, so an
arm that moves `predict` makes the assertion raise and the cells read REFUSED.
That is the exact reading `lane/kmeans-save` had to report as worthless. Two
arms now:

| define | moves | why |
|---|---|---|
| `MOJOLEARN_KMEANS_PREDICT_SABOTAGE` | label AND transform | the strongest arm, for a saved-model gate |
| `MOJOLEARN_HOST_SABOTAGE` (family) | transform only | enough for every recorded cell to move, because the cell hashes the pair, and it leaves `predict` equal to `labels_` so the harness's own assertion still holds |

Readings, on the Mac, one core, on rebuilt binaries: both gate arms `EXPECTED
MISMATCH SEEN` with `unmoved` EMPTY on 54; identity arms D and E
`DIVERGENT=12` (infer/model) and `DIVERGENT=6` (batch) with NO REFUSED cell;
and the production binding, rebuilt from the same source with a changed
sha256, still `gate verdict IDENTICAL (54 fixtures, exit 0)`. INERT is a
reading off the new file, not a claim about a comptime flag.

## The NVIDIA box

RunPod **A100-SXM4-80GB (sm_80)**, pod `ujwcpglxvl4lo7`, 18:14 to 18:22 UTC,
about eight minutes. Four cheaper specs answered "There are no instances
currently available" and billed nothing; **nothing was pinned**, which is why
there was a box at all.

| arm | verdict |
|---|---|
| `record` | 54 fixtures, exit 0 |
| `check`, x86-64 on the box | `gate verdict IDENTICAL (54 fixtures, exit 0)` |
| `check`, **arm64** on the M4 | `gate verdict IDENTICAL (54 fixtures, exit 0)` |
| sabotage `--every-fixture`, family define, x86-64 and arm64 | `EXPECTED MISMATCH SEEN`, `unmoved` EMPTY |
| sabotage `--every-fixture`, predict define, x86-64 and arm64 | `EXPECTED MISMATCH SEEN`, `unmoved` EMPTY |
| sabotage `--every-lane` | `EXPECTED MISMATCH SEEN` |

The arm64 line is the claim in one sentence: a k-means model fitted on an
A100, written to a file and loaded on an Apple M4 with the GPU out of reach,
answers predict and transform with the A100's bits on all 54.

### The arms lane/kmeans-save could not take

Arm C, GPU fit -> saved file -> `mojolearn.host_model` -> predict and
transform: `train IDENTICAL=7, infer 12, model 12, batch 6`, which is the L40S
result reproduced on an A100. Arm E, its control, which on the L40S leg did
not exist because the sabotage build exited 127 and every cell read REFUSED:

    summary (infer/model): DIVERGENT=6, N/A=2, RELOAD-MOVED=6
    summary (batch): IDENTICAL=6, N/A=1

**No REFUSED cell.** RELOAD-MOVED is the harness saying the reloaded file no
longer predicts what the model in memory predicted, which is a detection. The
batch cells are IDENTICAL because that protocol runs on the GPU, which this
arm does not touch; their own control is the batch sabotage column,
`BATCH_MOVED = 6 of 6`. Arm F, the boundary in bytes: five fitted shapes all
IDENTICAL on predict, transform and predict-over-training-rows against
`labels_`, and the one-float32-ULP rewrite of `centers[0]` fired on all five,
`cases=5 failed=0`, NAMING `predict` as unmoved each time rather than
reporting only the movers.

### spectral's CPU column, at the published size

`cells=18 stable=18 moved=0 refused=0`, and against the two NVIDIA columns and
the Metal column of `2026-09-16_predict-nvidia`: `summary (infer/model):
IDENTICAL=36`, `summary (batch): IDENTICAL=18`, with no exclusion note on any
of the nine fixtures. It came off the same box from a package copy with
`identical/` removed, which routes every call to the same
`MOJOLEARN_TARGET_COLUMN=cpu` binaries a CPU pod would build; the leg asserts
`vendor() == 'cpu'` on that copy first and refuses otherwise. That saved a
second rental.

## The AMD box, and the one thing it found

RunPod **MI300X (gfx942)**, pod `hx7egqocyrkc9p`, 18:31 to 18:36 UTC.

K-means is clean and complete: `cells=63 stable=63 moved=0 refused=0`, train
`IDENTICAL=63`, infer/model `IDENTICAL=108`, batch `IDENTICAL=54` against the
A100 and the two Apple columns; its recording checks IDENTICAL on 54 with the
sabotage arm caught on 54.

The predict lanes gave `cells=54 stable=31 moved=0 refused=23`. **MOVED IS
ZERO**; every cell that ran is IDENTICAL x3 against NVIDIA and the CPU column,
and agglomerative, spectral and spectral-precomputed are complete and clean.
The 23 refusals are all `hipErrorOutOfMemory` in `dbscan_fit_core`, on a
192 GB card fitting 6000 rows of four columns. They are ORDERED, not shaped:
four DBSCAN fits succeeded and every fit after them raised, including shapes
that had just worked, and the same fixtures ran `refused=0` on an A100 with
LESS memory. NOT claimed: that it is a leak rather than fragmentation, which
allocation it is, or that it reproduces; this is one observation, one card,
one process, no instrumentation, no replication.

## Spend

Two pods, about 13 minutes of paid time in total (A100 SXM ~8 min, MI300X
~5 min), roughly $0.45. Four NVIDIA creates starved and billed nothing. Both
pods terminated and verified gone (HTTP 404), and the account lists no pod at
all at the end.

## Verification scope

This lane's own lanes only, never a sweep. The k-means and predict lanes on
two GPU columns plus an x86 CPU column, the k-means gate on two host
architectures, `test_host_surface` + `test_host_model_kmeans` (175 passed) and
`docs_facts --check`. Heavy local work went through `MAC_SLOTS=4 mac_slot.sh
run`; the one Metal job, the leg's Apple reference card, went through
`mac_slot.sh metal` and waited its turn behind two other lanes. That card came
out BYTE-IDENTICAL to the one `lane/kmeans-save` generated at a different
commit, which is a small free check that the two commits agree on it.

Two assertions were watched FAILING before they were trusted: dropping
`kmeans-sqrt` from the manifest fails
`test_inference_lanes_are_classical_gate_lanes` and the k-means registry test,
and re-adding a `kmeans` note to the owed dict fails three tests
(`unfixed-side-arm1.txt`, `unfixed-side-arm2.txt`).

## Still owed

* **The AMD column for the DBSCAN lanes**, again, and deliberately: the OOM
  above, not a divergence. A second AMD box should be spent on that failure on
  purpose rather than as a side effect of a recording run.
* The Apple/Metal recording of the k-means lanes under
  `bench/results/classical_host/`, at the next release record. Apple is the
  scarce column and this lane pushed everything it could to rented boxes.
* The AMD recording of the four predict lanes beyond the three fixtures at
  `bench/results/classical_host/2026-09-16-amd-predict-partial`.
