# The k-means saved-model recording, AMD (2026-09-16)

`lane/classical-host-recordings`. The same six lanes and nine fixtures as
`2026-09-16-nvidia-kmeans`, recorded on a RunPod **AMD Instinct MI300X
(gfx942)**, pod `hx7egqocyrkc9p`, 18:31 to 18:36 UTC, terminated and VERIFIED
gone (HTTP 404). Commit `27c5baf023c79a3e98f260e5889237ebbf1312b7`, shipped as
a `git archive` at that sha.

| arm | verdict |
|---|---|
| `check`, x86-64 host bindings on the recording box | `gate verdict IDENTICAL (54 fixtures, exit 0)` |
| sabotage `--every-fixture`, `MOJOLEARN_KMEANS_PREDICT_SABOTAGE` | `EXPECTED MISMATCH SEEN (54 fixtures, exit 0)` |

The AMD identity column beside it is
`bench/results/identity_break/2026-09-16_amd-mi300x/amd-mi300x-gfx942.kmeans.json`:
`cells=63 stable=63 moved=0 refused=0`, and against the NVIDIA A100 column and
the two Apple columns of `2026-09-15_kmeans-transform`,

    summary: IDENTICAL=63                     (train)
    summary (infer/model): IDENTICAL=108, N/A=18
    summary (batch): IDENTICAL=54, N/A=9

The N/A cells are `kmeans-cosine`, whose fit is refused by name on every
vendor, which is what its cells are supposed to say.

The recorded `expected.json` files are byte-comparable with the NVIDIA ones:
every `identity_hash`, `predict`, `transform`, `labels` and
`predict_training_rows` digest matches, which is the diff above stated per
fixture.
