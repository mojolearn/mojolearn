# The AMD column for the k-means and predict lanes (2026-09-16)

`lane/classical-host-recordings`. One RunPod **AMD Instinct MI300X (gfx942)**,
AMD EPYC 9474F host, pod `hx7egqocyrkc9p`, 18:31 to 18:36 UTC, terminated and
VERIFIED gone (HTTP 404). Commit `27c5baf023c79a3e98f260e5889237ebbf1312b7`.

Two lanes had left an AMD column owed and both said so in writing:
`lane/saved-model-reference-gaps` for dbscan, agglomerative and spectral, and
`lane/kmeans-save` for k-means. AMD had been left alone by a standing
instruction; that instruction is lifted.

## k-means: clean

`amd-mi300x-gfx942.kmeans.json`, seven lanes, nine fixtures, two repeats.

    cells=63 stable=63 moved=0 refused=0

Against the NVIDIA A100 (sm_80) column of `2026-09-16_kmeans-and-spectral-cpu`
and the two Apple M4 columns (Metal and the M4's own CPU host) of
`2026-09-15_kmeans-transform`:

    summary: IDENTICAL=63                      (train)
    summary (infer/model): IDENTICAL=108, N/A=18
    summary (batch): IDENTICAL=54, N/A=9

Its saved-model recording is `bench/results/classical_host/2026-09-16-amd-kmeans`:
`check` IDENTICAL on 54, the predict-define sabotage arm `EXPECTED MISMATCH
SEEN` on 54 under `--every-fixture`.

## The predict lanes: 31 cells, 23 REFUSED, and the refusal is the finding

`amd-mi300x-gfx942.predict.json`, six lanes, nine fixtures, two repeats.

    cells=54 stable=31 moved=0 refused=23

**MOVED IS ZERO.** Nothing diverged. Every one of the 31 cells that ran is
IDENTICAL x3 against the NVIDIA A100 column of `2026-09-16_predict-nvidia` and
the x86 CPU column of `2026-09-16_kmeans-and-spectral-cpu`
(`diff.predict-amd-vs-nvidia-cpu.txt`: `summary: IDENTICAL=31, ONE-COLUMN=23`,
and the 23 are exactly the cells this column REFUSED). The agglomerative,
spectral and spectral-precomputed lanes are complete and clean on all nine
fixtures.

The 23 refusals are all the same exception, all in the DBSCAN family:

    Exception: At max/mojo/max/gpu/host/device_context.mojo:4073:35:
    HIP call failed: hipErrorOutOfMemory (out of memory)
      density.py:325  dbscan_fit_core

### CORRECTED 2026-09-16 BY lane/amd-dbscan-oom: THE CARD WAS NOT 192 GB

Everything below this heading was written before anyone read the card's free
memory, and two of its sentences are wrong. They are left standing, because a
correction that deletes what it supersedes leaves nothing to check it against.

**The refusals are NOT ordered.** This file's own JSON says `dbscan` fitted
`base`, `ties` and `hashed`, refused `wide`, `denormal` and `denormal_ftz`,
then FITTED `dupes`, then refused `odd` and `negative`, and that eighteen
`agglomerative` and `spectral` cells ran clean afterwards. "Every fit after
them raised" is not what happened, and a fit that succeeds between three that
failed is not an exhausted card.

**The card was not 192 GB free.** A second RunPod MI300X (pod o3paueazuvo87t,
gfx942) reproduced the same `hipErrorOutOfMemory`, and its card had **360.4 MiB
of 196592.0 MiB free before `import mojolearn` ran**, with 178.6 GiB held by a
kfd pid from outside the container. Across our first fit the card's used memory
FELL by 12031.2 MiB, which a fit of ours cannot do. The pod sees eight cards
through `/sys/class/drm` and owns one of them, `card33`, not `card0`.

**It is not an AMD path property.** A dedicated DigitalOcean MI325X (gfx942,
the same architecture) ran the same three DBSCAN lanes at `cells=27 stable=27
moved=0 refused=0`, with device memory flat at a 1.5 GiB plateau across 54
fits in one process and a `wide` fit in 0.44 s against 31.21 s here.

`odd` refusing while `base` succeeded was never explainable by the data: they
are the same shape and within 3% of the same edge count. Arrival time is the
only thing that separated them, and what was arriving was a neighbour's
allocation. The full reading is
`docs/lanes/LANE_STATUS_lane-amd-dbscan-oom.md`.

### What is established, and what is not

ESTABLISHED, from `logs/identity-predict.log` and `logs/record-predict.log`:

* The failures are ordered, not shaped. `dbscan` fitted `base`, `ties`,
  `hashed` and `dupes` and then raised on `wide`, `denormal`, `denormal_ftz`,
  `odd` and `negative`; after that EVERY fit in the process raised, including
  `dbscan-brute-l1/base` and `dbscan-weighted/base`, which are the same 6000 x
  4 shape that had just succeeded four times. A size threshold does not
  produce that order. Something the DBSCAN fit allocates on the device is not
  being returned when the fit ends.
* It is not the fixture data: the same fixtures, the same lane bodies and the
  same two repeats ran to `cells=54 stable=54 refused=0` on an NVIDIA A100
  with 80 GB (`2026-09-16_predict-nvidia`), a card with LESS memory than the
  192 GB MI300X here. That contrast is why this is written down as an AMD-path
  observation rather than a data-size one.
* `record` died at the same call on its own fourth fixture, independently of
  the identity run, so it is not a single unlucky allocation.

NOT ESTABLISHED, and not claimed:

* That it is a leak rather than fragmentation, or which allocation it is. No
  allocator instrumentation was run.
* That it reproduces. This is ONE observation on ONE MI300X in ONE process.
  Nothing here was replicated, and a second AMD box has not been rented for
  it.
* Anything about gfx1100, MI325X or any other AMD architecture.

The honest reading is: the AMD column for the three predict families is OWED
AGAIN for the DBSCAN lanes, and the reason is a resource failure on this
vendor's fit path that a second box should be spent on deliberately rather
than as a side effect of a recording run.

## Files

| file | what it is |
|---|---|
| `amd-mi300x-gfx942.kmeans.json` | the k-means column |
| `amd-mi300x-gfx942.predict.json` | the predict column, with its 23 refusals and their exception text |
| `diff.kmeans-amd-vs-nvidia-apple.txt` | k-means against NVIDIA and the two Apple columns |
| `diff.predict-amd-vs-nvidia-cpu.txt` | predict against NVIDIA and the x86 CPU column |
| `leg-gate.txt`, `leg-status.tsv`, `logs/` | the box's own record |

The leg's OUTER exit was 1, and that is the GEMM payload's arm, not this one:
`device_check_exit=245` and `card_exit=1` belong to
`tools/gemm_remote_leg.sh`'s own gemm card stage, which this lane rides but
does not use. `extra_exit=0`: this body ran to its end and its phases are in
`leg-status.tsv`.
