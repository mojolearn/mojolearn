# The inference lanes built out on 2026-09-16, against four questions

Of each lane merged today by `lane/kmeans-save`, `lane/saved-model-reference-gaps`
and `lane/stateful-cpu-decoding`:

1. is the algorithm actually built out;
2. do the CPU and GPU paths exist and are they BITWISE IDENTICAL;
3. is the lane in the PUBLICLY EXPOSED verification harness, so a user can
   check it;
4. does it have BATCH INVARIANCE testing where that is relevant.

Every cell below comes from something read in the code or run here, never from
a lane's own summary. Evidence:
`~/mojolearn-evidence/inference-coverage-complete/`.

## How each column was measured, and one probe that must not be reused

**Question 3** is `python/mojolearn/host_surface.py::public_reference_lanes()`
read by IMPORT, with `PUBLIC_PENDING_LANES` read for the reason.

**Question 2** is a reference table REBUILT here from every column JSON
committed under `bench/results/identity_break/`, through
`mojolearn._verify_reference.build_table`, then read per cell for which device
classes carry the same hash. It is rebuilt rather than read from
`python/mojolearn/verify_reference/table.json` because that shipped table was
last regenerated on 2026-09-15 16:55 (`429a8da9a`) and is a day behind main.
Counts below say `real` for a cell whose reference is a hash and `n/a` for one
whose reference is itself a declared `n/a`, because a column "agreeing" that a
part is absent is not evidence of anything.

**Question 4** is `tools/identity_break.py::BATCH`, the module-level
declaration dict, read by IMPORT, plus `STEPFULL` for the decode lanes.

> The probe `"batch" in inspect.getsource(lane_fn)` answers a different
> question and reports "no batch part" for almost every lane. The batch
> declarations are deliberately kept OUT of the lane bodies, so that no train,
> infer or model hash can move because of them
> (`tools/identity_break.py:4551`). Read `BATCH`, never the lane's source.

## The table

`cols` is the set of device classes whose hash equals the reference, `xN` the
number of fixtures at that size, out of nine.

| lane | 1. built out | 2. CPU and GPU, bitwise | 3. public | 4. batch |
|---|---|---|---|---|
| **kmeans** | yes. `cluster.py` `fit`/`predict`/`transform`/`save`/`load`; `_KMEANS_FORMAT` dispatched in `_classical_host._FORMATS` to `HostKMeans` | infer and batch `apple+cpu` x4, the other five fixtures `n/a:no-predict`. model `cpu` x1 (recorded here) | **yes** | yes, `predict` and `transform` row calls |
| **kmeans-sqrt** | same class, `algorithm` member | same as `kmeans` | **no, deliberate: `own record`** | yes, same declaration |
| **kmeans-random** | same, `init='random'` | same as `kmeans` | **yes** | yes |
| **kmeans-classic-pp** | same, classic k-means++ | same as `kmeans` | **yes** | yes |
| **kmeans-weighted** | same, `sample_weight` | same as `kmeans` | **yes** | yes |
| **kmeans-array** | same, array `init` | same as `kmeans` | **yes** | yes |
| **kmeans-cosine** | REFUSED BY NAME in `cluster/impl/kmeans_params.mojo::validate`; the lane's cell is the hash of the refusal sentence | every part `n/a` on every fixture, by design | **yes** | **n/a, correctly.** `n/a:fit-refused`: there is no fitted model to ask |
| **dbscan** | yes. `density.py` `fit`/`predict`/`save`/`load`; `_DBSCAN_FORMAT` -> `HostDBSCAN` | infer, model and batch `apple+cpu+nvidia` x4, `nvidia` x5. AMD `n/a:transductive` from a record that predates predict | **yes** | yes, `predict` on 64 held-out rows |
| **dbscan-brute-l1** | same class, `algorithm='brute'`, L1 | same as `dbscan` | **yes** | yes |
| **dbscan-weighted** | same, `sample_weight` | same as `dbscan` | **yes** | yes |
| **agglomerative** | yes. `_hierarchy_impl.py` `fit`/`predict`/`save`/`load`; `_AGGLOMERATIVE_FORMAT` -> `HostAgglomerativeClustering` | same shape as `dbscan` | **yes** | yes |
| **spectral** | yes. `_spectral_impl.py` `fit`/`predict`/`save`/`load` (Nystrom extension); `_SPECTRAL_FORMAT` -> `HostSpectralClustering` | **`cpu+nvidia` x7, `apple+cpu+nvidia` x2, on all three parts.** The CPU column was owed and is taken here | **no, deliberate: `stale reference`** | yes |
| **spectral-precomputed** | same class, `affinity='precomputed'` | same as `spectral` | **yes** | yes, `predict` on 64 held-out affinity rows |
| **transformer** | yes. `neural_inference.py` `allocate_state`/`step`, `forward` no longer overridden | infer and batch `amd+apple+cpu+nvidia` x9. model `n/a` (a block has no save) | **yes** | yes, plus `stepfull` |
| **transformer-window** | same, sliding window | same as `transformer` | **yes** | yes, plus `stepfull` |
| **mamba1** | yes, `_RecurrentBlockInference` | same as `transformer` | **yes** | yes, plus `stepfull` |
| **mamba2** | yes | same as `transformer` | **yes** | yes, plus `stepfull` |
| **mamba2-dtlimit** | yes, same class with `dt_limit` | **`cpu` x1 only**, recorded here. Its pre-shrink columns are gone | **no: `stale reference`, and the reason is understated** | yes, plus `stepfull` |
| **mamba3** | yes | same as `transformer` | **yes** | yes, plus `stepfull` |
| **samba** | yes. `SambaInference` gained `allocate_state`, a stateful `forward` and `step` | **`cpu` x1 only**, recorded here | **no: `stale reference`, understated** | yes, plus `stepfull` |
| **samba-untied-dropout-accum** | yes, untied embeddings, dropout, four accumulation microbatches | **`cpu` x1 only**, recorded here | **no: `stale reference`, understated** | yes, plus `stepfull` |
| **umap** | yes. `_umap_impl.py` `transform`/`save`/`load`; `_UMAP_FORMAT` -> `HostUMAP` | infer `amd+apple+cpu+nvidia` x9. model `cpu` x9 (GPU records say `n/a:no-save`) | **yes** | **has a declaration and it says the property does NOT hold**: `n/a:batch-dependent-by-contract`, four couplings cited line by line |

## Question 3, lane by lane: which exclusions are deliberate

Five scope lanes are outside `public_reference_lanes()`. All five carry a
stated reason in `PUBLIC_PENDING_LANES`, so none is a silent oversight. The
reasons are not equally true.

**`kmeans-sqrt`: deliberate and structural.** Its reason is `own record`, and
the condition is exact: it is in `TRAINING_FIX_LANES`
(`host_surface.py:168`), so the gate diffs it against `TRAINING_FIX_COLUMNS`
rather than against the release record, and the public set is derived from
`record_covered_lanes()`. It is not about its arithmetic. It joins the public
set when a release record covers it in the ordinary scope.

**`spectral`: deliberate today, and one regeneration from promotable.**
Its reason is `stale reference`: its fixture shrank to 512 rows this morning
(`f4e589395`, `LANE_REVISIONS["spectral"] = "rows-512-1"`), and the shipped
table predates that. Rebuilt from the records now on main it is FRESH, with
nine cells on all four parts, and with the CPU column taken in this lane it is
a three-column agreement at the published size. It is the only one of the
thirteen `stale reference` lanes that a regeneration promotes.

**`mamba2-dtlimit`, `samba`, `samba-untied-dropout-accum`: the stated reason
understates the gap, and the difference is what is owed.** `stale reference`
reads as "the references exist and describe the old bytes, so a regeneration
fixes it". Measured: a regeneration does not fix it. Every committed record
that carried these three was taken BEFORE the shrink
(`2026-09-15_cpu-samba` at 09:18, `2026-09-15_cpu-mamba` at 08:19, against
`f4e589395` at 06:28 the next morning), so `build_table` correctly drops their
cells and they end with ZERO. Their true reason is `no reference`, and what
is owed is a RECORD at the current fixture revisions, not a table rebuild.

Eleven other `stale reference` lanes outside this lane's scope are in the same
position: `holtwinters`, `gbdt-nan-modes`, `gbdt-parametric-losses`,
`gbdt-lossguide-newtoncosine`, `gbdt-pair-logit`, `hdbscan`, `hdbscan-leaf`,
`byte-lm`, `byte-lm-resident`, plus the two above. This is the fixture shrink's
bill and it falls due at the next release record.

## The rule that could never have reported this, and its repair

`test_public_reference_lanes_are_derived_and_every_pending_reason_is_true`
checked the `stale reference` reason by asking `lane in LANE_REVISIONS`. That
entry never goes away, and the test's last assertion then REQUIRES any lane
with one to stay pending. So a lane whose fixture is ever shrunk was barred
from the public set FOREVER, whatever a later table said, and nothing could
notice a regeneration.

The rule now reads the table's own `lane_revisions` against the harness's,
which is what `_verify_reference.stale_reference_lanes` already applies at
verify time (`_verify_all.py:1343`), and separates the three outcomes a
regeneration produces.

**Seen to fail before it was trusted.** Against a table rebuilt from main's
records, the OLD rule passes this branch while `stale_reference_lanes` returns
`[]`; the new rule fails and names all thirteen, each with its own verdict:

    spectral: NOT stale. the table carries cells at the current revision
      'rows-512-1', so the hold no longer applies: promote it once a CPU-only
      `verify --all` has been watched to read IDENTICAL for it
    samba: NOT stale. the table carries NO cell for it at all, so its reason is
      'no reference' and what is owed is a record at the current fixture
      revision 'steps-1-1'

Against the shipped table all 155 tests pass, so nothing changes today. The
rule is now self-clearing.

## Question 4, and where it is not the question

**There are no missing batch parts in this lane's scope.** Every lane listed
carries a declaration, and the two that decline give a reason that holds:

* `kmeans-cosine`: `n/a:fit-refused`. The lane body refuses `metric='cosine'`
  by name and its cell is the hash of the refusal sentence, so there is no
  fitted model to ask. Adding a part here would be adding a part for its own
  sake.
* `umap`: `n/a:batch-dependent-by-contract`, with four couplings cited to the
  line in `umap/transform.mojo` and matched against cuML's.

**`umap` is the reason "has batch testing" and "passes batch testing" must be
kept apart.** `lane/umap-batch-determinism` (NOT merged, awaiting a decision)
measured the property and it does not hold: at the shipped
`negative_sample_rate=5` the same row moves by up to `1.97` on a map whose
clusters sit about 11 apart depending only on its company, and the epoch count
falls from 100 to 30 above 10,000 queries, so adding ONE row to a request of
ten thousand moves another row by `1.36` while adding one at 9,999 moves no
bit. It also established what this is NOT: the effect is identical on the CPU
host route and on Metal, bit for bit, so it is the algorithm and not a column
disagreeing. The repository's claim is intact; the practical promise to an
inference server is not.

**For the transductive lanes the question is different and was answered.**
`dbscan`, `agglomerative` and `spectral` were transductive until 2026-09-15;
they have `predict` now, and the batch part asks it on 64 held-out rows, which
is the right question. Their AMD cells still read `n/a:transductive` because
the AMD record predates the capability, which is an owed recording and not a
declaration.

**For the decode lanes the analogous property is `stepfull`, and it is
proved but not publicly checkable.** One fresh-state forward pass over a
sequence against the same sequence decoded one token at a time with a carried
state, compared bitwise per position. Two separate gaps:

* `_verify_reference.PARTS` is `("train", "infer", "model", "batch")`.
  `stepfull`, `rlpair` and `ragged` are not in it and `_verify_all.py` never
  names them, so `python -m mojolearn verify --all` cannot compare `stepfull`
  and a user cannot check the decode property at all.
* Scanned over the 323 committed identity_break column files on main, the
  number carrying a real `stepfull` cell was ZERO before this lane. The part
  lived in the harness and in one lane's evidence directory.

## What this lane closed

Three records, each with both sabotage arms seen to fire.

1. **`bench/results/identity_break/2026-09-16_stateful-decode-cpu/`** -- the
   decode lanes' CPU column at the shrunk fixtures, preserved from
   `lane/stateful-cpu-decoding`'s own worktree rather than re-run. It is the
   only committed record with a `stepfull` cell, and it is the only reference
   `samba`, `samba-untied-dropout-accum` and `mamba2-dtlimit` have at the
   current revision. The lane's own arm covered five of the eight; the other
   three (`mamba2`, `mamba2-dtlimit`, `samba-untied-dropout-accum`) were run
   here. The clean arm reproduced all eighteen recorded hashes exactly from a
   DIFFERENT build of the host bindings, which is what makes the sabotage arm
   a valid control; the sabotage arm then moved every `batch` and `stepfull`
   cell, printing both bit patterns.
2. **`bench/results/identity_break/2026-09-16_kmeans-saved-model-cpu/`** --
   the first reference of any kind for the k-means saved-model bytes. Before
   it, zero committed columns carried a real `model` hash for any k-means
   lane; all read `n/a:no-save`, because `KMeans.save` landed after the last
   k-means recording. A sabotage core host build moved 24 of 24 comparable
   cells. Its first attempt read REFUSED on every cell, which is not a
   control, for a reason worth keeping: the guard refuses a sabotage build
   outside the gate unless `MOJOLEARN_HOST_ALLOW_SABOTAGE=1`.
3. **`bench/results/identity_break/2026-09-16_spectral-cpu-column/`** --
   `spectral`'s owed CPU column at the 512-row size, nine fixtures. All 72
   cell parts equal the reference the NVIDIA A100 and Apple Metal columns
   already carry. A sabotage metrics host build moved 70 of 72; the two it did
   not are named and explained rather than rounded away.

And one rule: the `stale reference` check now reads the table.

## What remains owed, with its cost

| owed | for | cost |
|---|---|---|
| A record of `samba`, `samba-untied-dropout-accum`, `mamba2-dtlimit` on the other eight fixtures, and on Apple and NVIDIA | the three lanes to leave `no reference` and become promotable | one CPU run per column; the Apple one needs the Metal slot, the NVIDIA one a box |
| The Apple `stepfull` column | the decode property on a second column | a rerun. `lane/stateful-cpu-decoding` took one, but from the SHARED checkout's package and Metal bindings while its commit field names the branch, so it is not recordable as-is |
| NVIDIA and AMD `stepfull` confirmation | the decode property cross-vendor | a box each, at the release record |
| `stepfull` in `_verify_reference.PARTS` and in `_verify_all.run_cell` | a user being able to check the decode property | small in code; it must land WITH a table regeneration, or every stepfull cell of all 212 lanes reads OWED |
| A regeneration of `python/mojolearn/verify_reference/table.json` | thirteen lanes to stop reading `stale reference`; `spectral` to become promotable | one command, no GPU (`verify --all --emit-reference`). Measured here: 86 records, 1656 cells, 0 conflicts. It is a RELEASE action, not a lane action: it also rewrites references far outside this scope |
| `tools/classical_host_gate.py record` for `kmeans` on a GPU box, then declaring `kmeans` an inference lane | `KMeans.predict` from a saved model to be gate-covered; it is the last entry in `SAVED_MODEL_INFERENCE_OWED` | one NVIDIA rental. `tools/kmeans_save_nvidia_leg.sh` exists and its `env FOO=1 -u BAR` ordering bug is already fixed (`1647059a0`); the leg has not been rerun |
| The AMD columns for the four predict lanes and for k-means `infer`/`batch`/`model` | the fourth column | the next release record. AMD is left alone by instruction |
| NVIDIA and AMD `infer` and `batch` cells for the k-means lanes on all nine fixtures | five fixtures currently read `n/a:no-predict` on every column | the next release record |

## A note on this lane's own verification

`tools/verify_lanes.py --changed-since origin/main` falls back to all 212
lanes, and the reason line names the cause:

    python/mojolearn/tests/test_host_surface.py: NOT ATTRIBUTABLE: no lane's
      derived source set names it, so every lane

No sweep was run. `tools/lane_select.py::test_module_inert` calls a test module
inert only while its name appears nowhere outside `python/mojolearn/tests/`;
`test_host_surface` is named in `.github/workflows/cpu-identity-gate.yml`,
which RUNS it. A workflow that runs a test is not a path by which a test can
change a lane's arithmetic, so this is a conservatism, and the selector's own
comment says so ("over-firing here only costs a sweep"). It is written down
here rather than fixed, because the selector's map is another lane's. The
appropriate verification for this change set is the test module itself, run
both ways: 155 passed against the shipped table, and the new branch seen to
fail against the regenerated one.
