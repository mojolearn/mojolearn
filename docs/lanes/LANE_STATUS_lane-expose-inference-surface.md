# Lane status: lane/expose-inference-surface (2026-09-16)

Andrew's question: **do we have algorithms that are built but not exposed to
PyPI, and can we expose them?**

Short answer. Almost everything a user would want to INFER with is already
shipped and reachable; the gap is not in what the wheel carries, it is in what
a user can CHECK. The wheel ships fifteen host families serving 79 declared
inference lanes and seven forest kinds, and `python -m mojolearn verify --all`
on a CPU-only install runs **nine lanes and four portable models**. Everything
else the wheel ships is taken on faith. Twenty-seven more lanes pass every
static condition for that list and would grow the wheel by nothing.

This lane made the gaps explicit in the manifest and left the promotion itself
behind a run that needs a host build. The machine was stopped mid-lane for
load (53 MB free of 16 GB, load average 47), so the build half is parked below
with exact commands.

## The three populations, counted

Read from the code, not the prose. Sources: `tools/identity_break.py` (the
harness), `python/mojolearn/host_surface.py` (the manifest),
`python/mojolearn/verify_reference/table.json` (what a wheel can judge
against), `python/mojolearn/_classical_host.py` and `_forest_host.py` (the
`host_model` door).

| population | count | what it is |
|---|---|---|
| harness lanes | 176 | `@lane(...)` registrations in tools/identity_break.py |
| host families | 32 | FAMILIES in the manifest |
| saved-model formats | 28 | `mojolearn-*-N` strings; **all 28 are dispatched by `mojolearn.host_model()`** |
| manifest covered (CPU training) lanes | 169 | `covered_lanes()` |
| manifest inference lanes | 79 | `inference_lanes()`, equal to the classical gate's LANES |
| forest kinds | 7 | `forest_kinds()` |
| families that ship in the wheel | 15 of 32 | `wheel_families()` |
| **lanes a CPU-only wheel user can verify** | **9** | `public_reference_lanes()` |
| portable models shipped | 4 | `verify_reference/models/` (179 KB) |

The 83-lanes-with-no-public-inference figure from Sep 15 did not reproduce.
Re-measured, the population that matters is different and smaller: 94 covered
training lanes are served by a family that does not ship, and that is correct
by design (training is internal). The real gap is the 9.

### Nothing is missing from the `host_model` door

All 28 saved-model formats resolve. `_classical_host._FORMATS` dispatches 28
of them across 31 `Host*` subclasses, `_forest_host` the forest pair, and
`_gbdt_host` the boosting one. There is no format a user can write with
`save()` that a CPU install cannot load. That part of "train on a GPU, infer
anywhere" holds.

## Gap 1: shipped but unverifiable (the big one)

`public_reference_lanes()` is nine: gemm-pinned, kde, ols, ridge, knn, svc,
pca, cholesky, tokenizer. A user who installs the wheel on a CPU box and runs
`verify --all` exercises those nine and the four portable models. The other 70
declared inference lanes, and every shipped family beyond the four the
portable models touch, are asserted in the README and checked by nothing the
user can run.

**27 lanes are addable at zero wheel cost.** Each one, measured:

* is a lane `tools/identity_break.py` defines;
* is in `record_covered_lanes()`, so it diffs against TRAINING_GPU_COLUMNS;
* has a real (not `n/a`, not conflicted) train reference in the shipped
  `table.json` on **all nine fixtures**, so it can read IDENTICAL, not OWED;
* already carries a `cpu` column in that table, so a CPU box has reproduced
  it at least once;
* is served ONLY by families with `ships_in_wheel=True`, so an inference
  wheel already has every binding it needs.

They are now `PUBLIC_REFERENCE_CANDIDATES` in the manifest, with
`--public-reference-candidates`:

    core        kmeans kmeans-random kmeans-array kmeans-weighted
                kmeans-classic-pp kmeans-cosine knn-clf-distance
                knn-reg-distance radius
    estimators  dbscan dbscan-brute-l1 dbscan-weighted kde-weighted
                pca-full-whiten ols-no-intercept ols-weighted
                ridge-no-intercept logistic-l1 logistic-elasticnet
                logistic-unpenalized-no-intercept
    svm         svc-linear svc-poly svr svr-linear iforest iforest-tuned
    metrics     umap

Promoting all 27 takes `public_reference_lanes()` from 9 to 36 and the wheel
grows by **zero bytes**: no new binary, no new model file, no new table entry.
The reference hashes are already in the shipped table; they are simply never
consulted, because the lane is never run.

They are NOT promoted on this branch. The reason is the rule, not caution:
nobody has watched them pass. Promoting a lane without that run would ship a
claim no one has seen succeed, and would turn a user's `verify` into REFUSED
or DIVERGENT if it were wrong. The run is parked below.

## Gap 2: implemented saved-model inference that no gate covers

`mojolearn.host_model()` loads and predicts from these today, and the manifest
does not declare them, so no gate covers them and the README does not mention
them. Now `SAVED_MODEL_INFERENCE_OWED` in the manifest,
`--saved-model-inference-owed`:

| lane | state |
|---|---|
| `dbscan` | `DBSCAN.predict` shipped (DEVIATION 2740), `mojolearn-dbscan-1` dispatched; GPU recording owed |
| `agglomerative` | `AgglomerativeClustering.predict` shipped, `mojolearn-agglomerative-1` dispatched; GPU recording owed |
| `spectral` | `SpectralClustering.predict` shipped (DEVIATION 2860); GPU recording owed |
| `spectral-precomputed` | the same on a precomputed affinity; GPU recording owed |
| `kmeans` | `KMeans.predict` shipped, but the class has **no `save`**, so there is no format to record |

Every one of the first four is waiting on exactly one thing: a recording made
by `tools/classical_host_gate.py record` on a GPU box. That tool refuses a
CPU-only install by design, so the host binding can never record its own
answer as its own reference. This lane could not make them; they need a GPU
leg, and GPU legs are release-record only.

`agglomerative` is worth calling out: its cells in `table.json` already carry
a `cpu` column on train, infer, model and batch. A CPU box has reproduced its
saved-model inference. It is served by the `solver` family, which does not
ship, so unlike the 27 above it cannot be promoted without a packaging
decision as well as a recording.

## Gap 3: silent exclusions, now written down

Seventeen families do not ship. Before this lane, fifteen of the seventeen
gave no reason anywhere in the file; two carried a one-line comment. Every
family now carries a `wheel_note` (`--wheel-notes`), and
`test_every_family_says_why_it_ships_or_does_not` fails if one is missing,
too short, or does not start with "Ships:" / "Does not ship".

Sixteen of the seventeen are settled and correct: the family holds a `fit`,
and the inference a user actually needs is served by a shipping family
(preprocessing and kernel_methods through `estimators`; trees, rf and gbdt
through `forest`; gp through `gp_infer`; mixture through `mixture_infer`;
hdbscan through `hdbscan_infer`; arima and tsa through `forecast`; embedding
through `embedding_infer`; ivf through `ivf_search`; mamba, transformer and
the MLP/Samba forwards through `neural`), or it is training-only by
definition (`training`).

**Two are not settled, and they are flagged OPEN in the file:**

* `resample`: `bootstrap`, `permutation_test`, `monte_carlo_integrate`
* `tsa`: `kpss_test`

These compute an answer from a user's own data and a user's own function, with
no fitted model and nothing to save. They fit neither side of the saved-model
inference boundary, so no shipped family carries them and they refuse on a
CPU-only install. Whether an inference wheel should carry them is a decision
for Andrew, not a boundary anyone drew. I did not expose them because doing so
would require a new inference-only binding and a recording, and because the
question is genuinely open.

## A finding worth Andrew's attention: `verify` can pass while refusing

Measured, not inferred. On a CPU-only install whose host bindings were stale,
`verify --all --full` printed:

    | all | 13 | 44 | 0 | 0 | 288 | 0 |
    RESULT: VERIFIED (44 identical, 0 divergent, 0 owed, 288 refused, 0 n/a
    cell parts). 17.7s. exit 0

288 of 332 cell parts REFUSED and the verdict is VERIFIED, exit 0. This is
documented behavior, not a bug hiding: `docs/VERIFY.md` line 109 says exit 0
means "no part DIVERGENT and at least one IDENTICAL (OWED and REFUSED parts
are counted, not passed)", and `test_verdict_exit_codes` asserts it. So it is
a deliberate policy.

It is still worth revisiting, and it matters more the moment the 27 candidates
land: a user with a partial or broken install would get a green exit while
four fifths of the surface never ran. I did not change it here, because
changing exit semantics is a cross-cutting decision and not this lane's call.
The options, cheapest first: keep exit 0 but print `VERIFIED (INCOMPLETE)` in
the headline when any part REFUSED; or add `--strict` that exits non-zero on
any REFUSED; or make REFUSED exit 4 outright, which would change CI.

## What changed on this branch

Static only. No kernel, no binding, no recording, no claim about arithmetic.

* `python/mojolearn/host_surface.py`
  * `wheel_note` on all 32 families, plus `wheel_notes()` and `--wheel-notes`.
  * `PUBLIC_REFERENCE_CANDIDATES` (27 lanes) and
    `public_reference_candidates()` / `--public-reference-candidates`.
  * `SAVED_MODEL_INFERENCE_OWED` (5 lanes) and
    `saved_model_inference_owed()` / `--saved-model-inference-owed`.
  * `--public-reference-lanes`, and the four new keys in `as_dict()`.
  * `public_reference_lanes()`' docstring now states the 9-against-79 gap.
  * `markdown_table()` is deliberately UNCHANGED, so the marked spans in
    SUPPORT_MATRIX.md and docs/BYTE_LM_CPU_TRAINING.md do not move.
* `python/mojolearn/tests/test_host_surface.py`: five new tests.
* `CHANGELOG.md`: one entry under Unreleased, marked for 0.8.7.

Nothing on the frozen `release/0.8.6` path was touched. `db9047b9f` and
`release/0.8.6` are untouched.

### Gates

    docs_facts --check                      OK: 13 facts, 12 marked spans
    wheel_ci.py pins .                      OK: 56 build scripts
    wheel_ci.py inventory python/mojolearn  OK: 85 modules

(Baseline taken on the clean tree before any edit, and again after; see the
commit message for the after values.)

### Wheel size effect

**Zero.** No binary, model file or table entry was added. Promoting all 27
candidates later is also zero: every serving family already ships and every
reference hash is already in `table.json`. For scale, the host binaries
already in the wheel run 250 KB to 590 KB each (`_mojolearn_linalg_host.so`
253,192 bytes and `_mojolearn_core_host.so` 584,536 bytes, built on this Mac
at this commit).

## NEEDS A BUILD, DEFERRED FOR MACHINE LOAD

Stopped mid-flight on 2026-09-16 at 05:33 ET when the Mac hit 53 MB free RAM
and load average 47. My build was killed; no process of mine survived; other
agents' Mojo processes were left alone. `linalg` and `core` finished before
the stop and are at
`/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/e52730fc-0ee7-4bf3-8741-5fa0e0f1876f/scratchpad/hostbuild/prod/`
(that path is session scratch and will not survive; rebuild rather than trust
it).

**Do not take the Metal lock. None of this needs a GPU.** One core, one
process at a time, `MOJOLEARN_BUILD_JOBS=1`, `nice -n 19`.

### Step 1: build the host families the candidates need (~90 s each)

    cd /Users/andrewhendel/mojolearn-wt/lane-expose-inference
    OUT=$HOME/mojolearn-evidence/expose-inference/hostprod && mkdir -p "$OUT"
    for fam in core estimators svm metrics linalg tokenizer forest; do
      MOJOLEARN_HOST_OUTDIR="$OUT" MOJOLEARN_BUILD_JOBS=1 \
        nice -n 19 sh "bindings/build_${fam}_host.sh"
    done

`build_host_family.sh` never overwrites an existing output, so use a fresh
directory. Never build into the shared checkout.

### Step 2: the CPU-only install, without touching a GPU

The worktree carries no GPU `.so` (they are gitignored build outputs), so with
host bindings present the package selects the CPU-only path on its own and
`mojolearn.vendor()` reads `cpu`. Confirmed on this box already.

    cd /Users/andrewhendel/mojolearn-wt/lane-expose-inference/python
    MOJOLEARN_HOST_DIR="$OUT" MOJOLEARN_NUMERIC_MODE=identical \
      nice -n 19 python3 -c "import mojolearn as ml; print(ml.vendor())"   # cpu

### Step 3: the proof, production build (expect IDENTICAL)

    CAND=$(python3 ../python/mojolearn/host_surface.py --public-reference-candidates)
    MOJOLEARN_HOST_DIR="$OUT" MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_CPU_THREADS=1 \
      nice -n 19 python3 ../tools/identity_break.py --lanes "$CAND" --repeats 2 \
      --json ~/mojolearn-evidence/expose-inference/cpu-candidates.json
    nice -n 19 python3 ../tools/identity_break.py --diff \
      $(python3 ../python/mojolearn/host_surface.py --training-gpu-columns) \
      ~/mojolearn-evidence/expose-inference/cpu-candidates.json --require-columns 4

Every candidate's train cell must read IDENTICAL x4 on all nine fixtures. A
lane that does not is dropped from the list, not argued with.

### Step 4: the sabotage, which must FAIL (this is the part that can fail)

Build the same families with the sabotage define into a SEPARATE directory and
re-run. Every candidate must move.

    SAB=$HOME/mojolearn-evidence/expose-inference/hostsab && mkdir -p "$SAB"
    for fam in core estimators svm metrics linalg; do
      MOJOLEARN_HOST_OUTDIR="$SAB" MOJOLEARN_BUILD_JOBS=1 \
        MOJOLEARN_BUILD_EXTRA_DEFINES="$(python3 python/mojolearn/host_surface.py --sabotage-build-defines $fam)" \
        nice -n 19 sh "bindings/build_${fam}_host.sh"
    done
    MOJOLEARN_HOST_DIR="$SAB" MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
      MOJOLEARN_NUMERIC_MODE=identical nice -n 19 python3 ../tools/identity_break.py \
      --lanes "$CAND" --json ~/mojolearn-evidence/expose-inference/cpu-candidates-sab.json
    nice -n 19 python3 ../tools/identity_break.py --diff \
      ~/mojolearn-evidence/expose-inference/cpu-candidates.json \
      ~/mojolearn-evidence/expose-inference/cpu-candidates-sab.json

Run the sabotage arm BEFORE trusting the production arm. A diff that shows
nothing moving means the sabotage did not reach these lanes, which invalidates
the negative control rather than passing it.

### Step 5: promote, then re-check the whole command

Move the surviving lanes out of `PUBLIC_REFERENCE_CANDIDATES` into
`public_reference_lanes()`, then:

    cd /Users/andrewhendel/mojolearn-wt/lane-expose-inference/python
    MOJOLEARN_HOST_DIR="$OUT" MOJOLEARN_NUMERIC_MODE=identical \
      nice -n 19 python3 -m mojolearn verify --all --full

and update the sentence in `docs/VERIFY.md` (line 39 spells the nine lanes out
by hand) and the timing table (line 95 says "8 lanes, 9 fixtures, 4 models").
`test_cpu_training_misc.py::test_identity_command_runs_public_reference_probes_on_a_cpu`
and `test_verify_all.py::test_full_and_cpu_lane_sets` both read the list and
must stay green.

### Step 6: tests and gates

    /Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python -m pytest \
      python/mojolearn/tests/test_host_surface.py \
      python/mojolearn/tests/test_cpu_inference_boundary.py \
      python/mojolearn/tests/test_cpu_training_misc.py -q
    python3 tools/docs_facts.py --check
    python3 packaging/wheel_ci.py pins .
    python3 packaging/wheel_ci.py inventory python/mojolearn

## Resume

    git worktree list | grep expose-inference
    cd /Users/andrewhendel/mojolearn-wt/lane-expose-inference
    git log --oneline -3          # branch lane/expose-inference-surface
    python3 python/mojolearn/host_surface.py --public-reference-candidates
    python3 python/mojolearn/host_surface.py --saved-model-inference-owed
    python3 python/mojolearn/host_surface.py --wheel-notes

## Done

- [x] the three-population comparison, re-measured from code
- [x] the gap list, with the reason each item is not exposed, in the file
- [x] `wheel_note` on all 32 families, test-enforced, no silent exclusion
- [x] the two OPEN questions named as open (`resample`, `kpss_test`)
- [x] 27 promotable lanes measured and recorded with their conditions
- [x] gates green, branch pushed
- [ ] **NEEDS A BUILD**: steps 1 to 6 above, then promote and merge
- [ ] a decision from Andrew on `resample` and `kpss_test`
- [ ] a GPU recording for dbscan, agglomerative and spectral predict (next
      release record; not a lane of its own)
- [ ] a decision on `verify` reading VERIFIED with REFUSED parts

## Not merged, deliberately

This branch is pushed but NOT merged. The manifest and test changes are green
and standalone, but the lane's point is the promotion in step 5, and unrun
code stays on its branch. Merging the documentation half alone is fine if
Andrew wants it now; it changes no behavior.
