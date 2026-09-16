# Lane status: lane/expose-inference-surface (2026-09-16)

Andrew's question: **do we have algorithms that are built but not exposed to
PyPI, and can we expose them?**

Short answer. Two were, and they are now exposed. Otherwise almost everything
a user would want to INFER with already shipped; the real gap was not what the
wheel carries but **what a user can check**. The wheel ships host families
serving 79 declared inference lanes and seven forest kinds, and
`python -m mojolearn verify --all` on a CPU-only install ran **nine lanes and
four portable models**. Everything else was taken on faith.

Three things came out of the lane: two genuinely unreachable functions are now
shipped, a verification that could not fail is fixed, and 30 lanes are measured
ready to quadruple the checkable surface at zero wheel cost, held only on the
fixture shrink.

## 1. The three populations, counted

Read from the code, not the prose.

| population | count | source |
|---|---|---|
| harness lanes | 176 | `@lane(...)` in `tools/identity_break.py` |
| host families | 32 | `FAMILIES` in `host_surface.py` |
| saved-model formats | 28 | **all 28 dispatched by `mojolearn.host_model()`** |
| covered CPU training lanes | 169 | `covered_lanes()` |
| declared inference lanes | 79 | `inference_lanes()` |
| families shipping in the wheel | **16** of 32 (was 15) | `wheel_families()` |
| **lanes a CPU wheel user can verify** | **9** | `public_reference_lanes()` |
| portable models shipped | 4 | `verify_reference/models/` (179 KB) |

The Sep 15 "83 lanes with no public CPU inference" figure did not reproduce.
94 covered training lanes are served by families that do not ship, which is
correct by design: training is internal. The number that mattered was the 9.

**No format is missing from the door.** All 28 `mojolearn-*-N` formats resolve
through `_classical_host._FORMATS` (28), `_forest_host` and `_gbdt_host`. There
is nothing a user can `save()` that a CPU install cannot load.

## 2. Exposed: the two functions that were genuinely unreachable

Andrew's call, 2026-09-16. The inference boundary exists to keep CPU **training
of models** internal so users train on a GPU and infer anywhere. These compute
a statistic from the caller's own data, train no model, and have nothing to
save, so the boundary never had a side for them to fall on and they refused on
a CPU-only install. A user who installs a library with a time series module and
finds `kpss_test` refusing on their laptop reasonably concludes it is broken.

- **`resample.bootstrap`, `resample.permutation_test`,
  `resample.monte_carlo_integrate`.** The `resample` family now ships
  (`ships_in_wheel=True`), taking the wheel from 15 host bindings to 16. Its
  binding already registered the three entries **and no fit**, and `_backend`
  already routed `_mojolearn_resample` to it, so shipping it was the whole
  change.
- **`kpss_test`.** Its family, `tsa`, holds `holtwinters_fit`, so shipping
  that family would have shipped a fit. Instead `kpss_test_binding` moved into
  a new shared module `bindings/kpss_host_test.mojo`, which **both** the tsa
  reference binding and the shipped `forecast` inference binding register, so
  the two binaries answer through one source and cannot drift. The
  `_mojolearn_tsa` route already falls back to `_mojolearn_forecast_host` on a
  CPU-only install, so `_tsa_impl.py` reaches it unchanged. This is the
  established pattern (`holtwinters_host_predict.mojo`,
  `mixture_host_scoring.mojo`).

`forecast_host_sabotage` now also reports `KPSS_ORACLE_HOST_SABOTAGE`, so a
sabotage build of the shipped binding is refused outside the gate whichever arm
was raised.

**Evidence: PENDING the prover run** (section 7). The functions must be seen
refusing before the binding exists and running after, and their lanes must read
IDENTICAL against the three GPU columns and DIVERGENT under sabotage.

## 3. Fixed: a verification that could not fail

Measured on an Apple M4 CPU-only install whose host bindings were stale:

    | all | 13 | 44 | 0 | 0 | 288 | 0 |
    RESULT: VERIFIED (44 identical, 0 divergent, 0 owed, 288 refused, 0 n/a). exit 0

288 of 332 cell parts REFUSED and the public command printed **VERIFIED, exit
0**. A user ran our verification, saw VERIFIED, and had checked 13 percent of
what they believed they checked. It was documented policy
(`docs/VERIFY.md` line 109) and asserted by `test_verdict_exit_codes`, which
made it deliberate rather than defensible.

A REFUSED part is a part that **did not run**. It can never be evidence of
success, and no number of parts that did run makes up for it, so there is no
threshold below which refusals are tolerable.

- `verdict()` now reads DIVERGENT first (a wrong answer still outranks an
  absent one), then **any** REFUSED gives `INCOMPLETE` and exit 4. Only a run
  with nothing refused may print VERIFIED.
- New `detail_line()` leads with how much of the run was checked:
  `verified 44 of 332 cell parts (0 divergent, 27 owed, 288 refused, 0 n/a)`.
- `docs/VERIFY.md` updated: the REFUSED row, the exit table, and a note on what
  changed.

**Seen failing first, as required.** Against the unmodified code the new tests
failed with `AssertionError: a run with refused parts must not print VERIFIED`
and `AttributeError: module 'mojolearn._verify_all' has no attribute
'detail_line'`, and the old behavior printed plainly as
`verdict(44 identical, 288 refused) = (0, 'VERIFIED')`.

**Not over-strict.** On a healthy CPU-only install the same command still
passes: `VERIFIED (verified 278 of 332 cell parts (0 divergent, 27 owed, 0
refused, 27 n/a))`, exit 0.

`identity`, the sibling command, does **not** share the flaw: its `_judge`
treats any non-`IDENTICAL x4` verdict and any missing row as bad.

## 4. Measured ready, and deliberately held: 30 lanes

`PUBLIC_REFERENCE_CANDIDATES` in the manifest, with
`--public-reference-candidates`. Each one:

* is a lane `tools/identity_break.py` defines;
* is in `record_covered_lanes()`;
* has a real train reference in the shipped `table.json` on **all nine
  fixtures**, so it can read IDENTICAL rather than OWED;
* already carries a `cpu` column there;
* has all nine fixtures on **all three** `TRAINING_GPU_COLUMNS`, so
  `--require-columns 4` can be met;
* is reachable from a binding that **ships**, so promoting adds no binary.

Promoting all 30 takes the checkable surface from 9 lanes to 39 and grows the
wheel by **zero bytes**: the reference hashes are already shipped, they are
simply never consulted because the lane is never run.

**Measured, 2026-09-16 (the run, not the plan).** The 27-lane production column
was taken on this Mac's CPU, 9 fixtures, 2 repeats, one core, and diffed
against the three `TRAINING_GPU_COLUMNS` with `--require-columns 4` the way the
CPU gate does it (`--owed-json`):

    summary: IDENTICAL=270
    summary (infer/model): IDENTICAL=252, N/A=135, OWED=153
    summary (batch):       IDENTICAL=153, N/A=45,  OWED=72
    summary (owed): OWED=225; the next release record owes exactly these parts

- **Zero DIVERGENT, anywhere.**
- **26 of 27 read IDENTICAL x4 on train across all nine fixtures** (234 cells
  = 26 x 9), against the Apple, NVIDIA and AMD columns plus this CPU column.
- The one exception is `svc-poly`, all 9 of its train cells, and it is exactly
  the lane the column count had already removed from the list. Nothing else
  was short on train.
- The 225 OWED parts are infer, model and batch cells no committed GPU record
  hashes yet (dbscan's three parts, the kmeans family's two, umap, svr, radius
  and iforest's one). That is the ordinary state the gate absorbs, not a
  disagreement; they are listed in `C_owed_cells.json`.

Two notes against over-reading this. First, an earlier run of the same diff
reported "261 short" because I omitted `--owed-json`, which the gate passes;
that was my flag, not arithmetic. Second, a re-run of mine failed outright
because zsh does not word-split `$GPUCOLS` and it tried to open all three
column paths as one filename. The authoritative numbers above come from the
chain, which runs under bash.

**Two of my own stated criteria were wrong and a check caught each.** Both
corrections are now written into the file:

- **`svc-poly` was dropped.** It met every criterion I had written and sat in
  the list until the columns were counted: its cells rest on two columns only
  (apple and cpu, from `2026-09-15_inference-svm`), so it can never meet
  `--require-columns 4`. Its NVIDIA and AMD recordings are owed to the next
  release record, as `CLASSICAL_RECORDED` already noted. Added the
  three-GPU-column criterion.
- **`kpss` was wrongly rejected.** My criterion said every declaring family
  must ship; `kpss` is declared by `tsa`, which does not. But it is reachable
  because the shipped `forecast` binding serves `_mojolearn_tsa`. Added the
  route-served criterion.

### HELD: the fixture shrink

**Nothing is promoted on this branch.** Another agent is shrinking oversized
fixtures and will publish `docs/lanes/FIXTURE_SHRINK_SCOPE.md` in three buckets
(will change, leave big, undecided). That file **does not exist yet** — not in
this worktree, not on `origin/main`, and there is no such branch. I cannot
certify a lane is clear of a list I have not seen, so every candidate is
effectively "undecided" and all 30 are held.

This matters more than it sounds. These lanes' references ship **in the
wheel's table**. Promoting a lane whose fixture then changes would ship a
reference a user's `verify` fails against, breaking the exact command this lane
exists to make trustworthy. An unpromoted lane with a stale reference is
latent; a promoted one is a user-visible failure.

**To finish**: read the scope list, promote the lanes in "leave big", hold the
rest, and record which were held and why.

## 5. Owed: implemented saved-model inference no gate covers

`SAVED_MODEL_INFERENCE_OWED`, with `--saved-model-inference-owed`. Each is
implemented and already dispatched by `mojolearn.host_model()`:

| lane | state |
|---|---|
| `dbscan` | `DBSCAN.predict` shipped (DEVIATION 2740); GPU recording owed |
| `agglomerative` | `AgglomerativeClustering.predict` shipped; GPU recording owed |
| `spectral` | `SpectralClustering.predict` shipped (DEVIATION 2860); GPU recording owed |
| `spectral-precomputed` | same on a precomputed affinity; GPU recording owed |
| `kmeans` | `KMeans.predict` shipped, but the class has **no `save`** |

The first four wait on one thing: a recording from
`tools/classical_host_gate.py record`, which refuses a CPU-only install by
design so a host binding can never record its own answer as its own reference.
That needs a GPU leg, and GPU legs are release-record only.

`agglomerative` is worth noting: its cells already carry a `cpu` column on all
four parts. It is served by `solver`, which does not ship, so unlike the 30 it
needs a packaging decision as well as a recording.

## 6. No silent exclusions

Every one of the 32 families now carries a `wheel_note` saying why it does or
does not ship (`--wheel-notes`), enforced by
`test_every_family_says_why_it_ships_or_does_not`, which fails on a missing or
too-short note. Before this lane, 15 of the 17 non-shipping families gave no
reason anywhere.

Sixteen ship. The sixteen that do not each name the shipping family serving
their inference (preprocessing and kernel_methods through `estimators`; trees,
rf and gbdt through `forest`; gp through `gp_infer`; mixture through
`mixture_infer`; hdbscan through `hdbscan_infer`; arima and tsa through
`forecast`; embedding through `embedding_infer`; ivf through `ivf_search`;
mamba, transformer and the MLP/Samba forwards through `neural`), or state they
are training-only (`training`).

## 7. Evidence

Host bindings built on this Mac at this commit, one core, `nice -n 19`,
`MOJOLEARN_BUILD_JOBS=1`. **No Metal was taken and no GPU was used**; the lock
was left free for other agents. Nothing was rented.

**Sabotage controls are real, not assumed.** Production and sabotage binaries
differ byte for byte for all seven families; production bindings read back
`sabotage=False` and sabotage bindings `sabotage=True`; and loading a sabotage
binding without `MOJOLEARN_HOST_ALLOW_SABOTAGE=1` is refused by name (the guard
was watched firing).

- `A_verify_before.log` — healthy CPU-only baseline, VERIFIED, 278 of 332.
- `B_prod_run.log`, `cpu-candidates.json` — the 27 candidates, 9 fixtures, 2 repeats.
- `C_diff_gpu.log`, `C_diff_gpu_owed.log`, `C_owed_cells.json` — candidates vs
  the three GPU columns, `--require-columns 4`: **0 DIVERGENT, train
  IDENTICAL=270, 225 OWED**, the only train shortfall being `svc-poly`.
- `D_sab_run.log`, `E_diff_sab.log` — **PENDING**: the same lanes under sabotage.
- `F_exposed_run.log`, `G_exposed_diff.log` — **PENDING**: kpss, bootstrap, permutation-test, monte-carlo vs the GPU columns.
- `H_exposed_sab.log`, `I_exposed_sabdiff.log` — **PENDING**: those four under sabotage.

All under `~/mojolearn-evidence/expose-inference/` (outside the repo).

### Gates

    docs_facts --check                      OK: 13 facts, 12 marked spans
    wheel_ci.py pins .                      OK: 56 build scripts
    wheel_ci.py inventory python/mojolearn  OK: 85 modules
    check_ext_lists.py --host               OK, 16 bindings, read from the manifest everywhere
    test_host_surface.py                    154 passed

`docs_facts --write` regenerated the `host_surface_table` spans in
`SUPPORT_MATRIX.md` and `docs/BYTE_LM_CPU_TRAINING.md`, because `resample` now
ships and `forecast` gained a class.

### Wheel size effect

**One binding added**, `_mojolearn_resample_host.so`, for the three analysis
functions. The other shipped host binaries run 250 KB to 750 KB on this Mac, so
expect the same order. Everything else in this lane costs **zero bytes**: the
false-VERIFIED fix is Python, and promoting the 30 candidates later adds no
binary and no table entry.

## 8. Not merged

The branch is pushed and **not merged**. The false-VERIFIED fix and the two
exposures are complete and standalone, but the promotion in section 4 is the
lane's headline and it is held on a file that has not been published. Merging
the settled parts alone is reasonable if Andrew wants them now.

`release/0.8.6` and `db9047b9f` were never touched. No Metal job was run.

## Resume

    cd /Users/andrewhendel/mojolearn-wt/lane-expose-inference
    git log --oneline -3                 # branch lane/expose-inference-surface
    python3 python/mojolearn/host_surface.py --public-reference-candidates
    python3 python/mojolearn/host_surface.py --saved-model-inference-owed
    python3 python/mojolearn/host_surface.py --wheel-notes
    ls ~/mojolearn-evidence/expose-inference/

To finish the promotion once `docs/lanes/FIXTURE_SHRINK_SCOPE.md` lands: move
the cleared lanes from `PUBLIC_REFERENCE_CANDIDATES` into
`public_reference_lanes()`, update `docs/VERIFY.md` line 38 (which spells the
nine lanes out) and its timing table line 98 ("8 lanes, 9 fixtures, 4 models"),
then re-run `verify --all --full` and the three gates.

## Done

- [x] three-population comparison, re-measured from code
- [x] `wheel_note` on all 32 families, test-enforced, no silent exclusion
- [x] **exposed** resample's three functions and `kpss_test`, no fit shipped
- [x] **fixed** the false VERIFIED, with the new test watched failing first
- [x] 30 promotable lanes measured, with two of my own criteria corrected
- [x] sabotage controls proven real before being relied on
- [ ] the exposure and candidate proofs (running; section 7 placeholders)
- [ ] **HELD**: promote once `FIXTURE_SHRINK_SCOPE.md` is published
- [ ] a GPU recording for dbscan, agglomerative and spectral predict
