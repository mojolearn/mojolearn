# The k-means saved-model recording (2026-09-16)

`lane/classical-host-recordings`. Six lanes x nine fixtures = 54 fixture
directories, recorded on a RunPod **NVIDIA A100-SXM4-80GB (sm_80)**, driver
580.126.16, pod `ujwcpglxvl4lo7`, 2026-09-16 18:14 to 18:22 UTC, terminated and
VERIFIED gone (HTTP 404). Commit `8511c8c518088829c86cd052a1dee0125cd1939e`,
shipped as a `git archive` at that sha so the box ran the commit and not a
working tree.

`kmeans`, `kmeans-random`, `kmeans-array`, `kmeans-weighted`, `kmeans-sqrt`,
`kmeans-classic-pp`. NOT `kmeans-cosine`: its fit is refused by name
(`cluster/impl/kmeans_params.mojo::validate`), so there is no model to save and
no saved-model cell to record. The one format `mojolearn-kmeans-1` carries
every metric and every start, so all six load through `HostKMeans` on
`_mojolearn_core_host`.

## What each directory holds

`expected.json` records, per fixture, the sha256, dtype and shape of every
surface, the identity_break hash of the lane's own `infer` cell, and the saved
file's sha256. The surfaces are

| surface | what it is |
|---|---|
| `predict` | the identity probe's first element, `predict` on the whole held-out draw |
| `transform` | its second element |
| `identity_hash` | the pair, hashed as `tools/identity_break.py` hashes it |
| `labels` | `labels_`, the fit's own assignment, which the file carries |
| `predict_training_rows` | `predict` over the TRAINING rows, rebuilt from the fixture |

The last two are the point of a SAVED k-means model. `identity_break`'s
`_km_probe` asserts that a fitted model's `predict` over the training rows is
`labels_` bit for bit; `predict_training_rows` is that same statement asked of
a model that has been through a file and a different machine. It holds on all
54.

## The verdicts

| arm | verdict |
|---|---|
| `check`, x86-64 host bindings on the recording box | `gate verdict IDENTICAL (54 fixtures, exit 0)` |
| `check`, **arm64** host bindings on the M4 | `gate verdict IDENTICAL (54 fixtures, exit 0)` |
| sabotage `--every-fixture`, family define, x86-64 | `EXPECTED MISMATCH SEEN`, `unmoved` EMPTY |
| sabotage `--every-lane`, family define, x86-64 | `EXPECTED MISMATCH SEEN` |
| sabotage `--every-fixture`, `MOJOLEARN_KMEANS_PREDICT_SABOTAGE`, x86-64 | `EXPECTED MISMATCH SEEN`, `unmoved` EMPTY |
| sabotage `--every-fixture`, family define, **arm64** | `EXPECTED MISMATCH SEEN`, `unmoved` EMPTY |
| sabotage `--every-fixture`, predict define, **arm64** | `EXPECTED MISMATCH SEEN`, `unmoved` EMPTY |

The arm64 line is the claim in one sentence: a k-means model fitted on an
A100, written to a file and loaded on an Apple M4 with the GPU out of reach,
answers `predict` and `transform` with the A100's bits on all 54.

`unmoved` EMPTY is the sentence that matters on the sabotage lines.
`--every-fixture` is the only rule that asks every recorded cell to move rather
than one per lane, and all 54 are named individually in the reports.

## THE CONTROL DID NOT EXIST UNTIL THIS LANE, AND THE REHEARSAL IS WHY

The first `--every-fixture` arm, rehearsed on the Mac on a CPU-made recording
before anything was rented, read

    gate verdict SABOTAGE NOT CAUGHT ON FIXTURES <all 54>, exit 1
    unmoved: all 54

`MOJOLEARN_HOST_SABOTAGE` had exactly one arm in `cluster/host/kmeans_oracle
.mojo`, inside `host_accumulate`, and only the FIT walks that function. A saved
model's `predict` and `transform` never reach it. The gate this recording
exists for could not fail. `KMEANS_PREDICT_HOST_SABOTAGE` and
`KMEANS_TRANSFORM_HOST_SABOTAGE` are the arms that fixed it; the file's
comments say why the label half has a define of its own.

## Two other things this box did

* The k-means identity arms `lane/kmeans-save` left owed. Its L40S leg's
  sabotage build exited 127 and every cell of its arm D read REFUSED. Retaken
  here: arm C `train IDENTICAL=7, infer 12, model 12, batch 6` (the L40S
  numbers, reproduced on an A100), and arm E now DIVERGENT on 6 infer cells and
  RELOAD-MOVED on 6 model cells, with **no REFUSED cell anywhere**.
* `spectral`'s x86 CPU identity column at the published 512-row size, which
  `lane/saved-model-reference-gaps` left owed. Both under
  `bench/results/identity_break/2026-09-16_kmeans-and-spectral-cpu/`.

## Still owed

The AMD column and the AMD recording, for these lanes and for the four predict
lanes; `tools/predict_kmeans_amd_leg.sh` is the body for it.
