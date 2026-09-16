# Lane status: lane/expose-inference-surface (2026-09-16)

Andrew's question: **do we have algorithms that are built but not exposed to
PyPI, and can we expose them?**

Short answer. Two were, and they are now exposed. Otherwise almost everything
a user would want to INFER with already shipped; the real gap was not what the
wheel carries but **what a user can check**. The wheel ships host families
serving 79 declared inference lanes and seven forest kinds, and
`python -m mojolearn verify --all` on a CPU-only install ran, **before this
lane, nine lanes and four portable models**. Everything else was taken on
faith. It now runs 39.

Three things came out of the lane: two genuinely unreachable functions are now
shipped, a verification that could not fail is fixed, and the checkable surface
went from **9 lanes to 39** at zero wheel cost. The promotion was held until the
fixture shrink published its scope and released once none of the thirty turned
out to be among the thirteen shrunk lanes.

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
| **lanes a CPU wheel user can verify** | **39** (was 9) | `public_reference_lanes()` |
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

**Evidence, measured 2026-09-16.** The functions were seen refusing before the
bindings existed and running after, and their lanes read IDENTICAL against the
three GPU columns.

BEFORE, on a CPU-only install whose host directory held neither binding (the
refusal has to be the BINDING's, so the probe treats anything but
ImportError/NotImplementedError as a wrong-reason failure):

    resample.bootstrap               refused (ImportError): no CPU implementation of
                                     _mojolearn_resample.resample_numeric_mode yet
    resample.permutation_test        refused (ImportError): same
    resample.monte_carlo_integrate   refused (ImportError): same
    kpss_test                        refused (ImportError): no CPU implementation of
                                     _mojolearn_tsa.kpss_test yet
    wrong-reason failures: 0

AFTER, the same four calls with the two bindings built:

    resample.bootstrap               RAN -> BootstrapResult
    resample.permutation_test        RAN -> PermutationTestResult
    resample.monte_carlo_integrate   RAN -> MonteCarloResult
    kpss_test                        RAN -> Array(shape=(1,), dtype='<u1')
    wrong-reason failures: 0

IDENTITY, the four lanes at 9 fixtures and 2 repeats, diffed against the three
GPU columns with `--require-columns 4`:

    summary (train):       IDENTICAL=36        (4 lanes x 9 fixtures, every one)
    summary (infer/model): N/A=72              (functions; no fitted model to save)
    summary (batch):       IDENTICAL=9, N/A=9, OWED=18
    require-columns 4 over [bootstrap, kpss, monte-carlo, permutation-test]: OK (18 OWED)

0 DIVERGENT and 0 REQUIRE FAIL. The `tsa` reference binding also compiled
cleanly into a throwaway directory, which is what proves the 76-line move of
`kpss_test_binding` into the shared module did not break the binding that keeps
the fit.

SABOTAGE, the same four lanes against a host set built with
`-D MOJOLEARN_HOST_SABOTAGE=1`:

    summary (train):       DIVERGENT=36        (every cell moved)
    summary (infer/model): N/A=72
    summary (batch):       DIVERGENT=27, N/A=9

**Not one cell stayed IDENTICAL.** The run was bound to the sabotage binaries
by SHA-256, including the newly built `_mojolearn_resample_host` and
`_mojolearn_forecast_host` that the exposure depends on, so the movement is the
sabotage arithmetic and not a mis-bound directory. Unlike the dbscan train
cells in section 4, these four have no inert corner.

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

## 4. Promoted: the checkable surface goes from 9 lanes to 39

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

### The sabotage, and where it is inert

The same 27 lanes were re-run against a host set built with
`-D MOJOLEARN_HOST_SABOTAGE=1` and diffed against the production column.

    summary (train):        DIVERGENT=228, IDENTICAL=15
    summary (infer/model):  DIVERGENT=372, IDENTICAL=51, N/A=63
    summary (batch):        DIVERGENT=225, N/A=18

**The control was proven real before it was relied on**, because a negative
control that fails for the wrong reason is worse than none. Production and
sabotage binaries differ byte for byte for all seven families; production reads
back `sabotage=False` and sabotage `True`; loading a sabotage binding without
`MOJOLEARN_HOST_ALLOW_SABOTAGE=1` is refused by name, watched firing; and the
sabotage run's own JSON records all seven bindings by SHA-256, every one
matching the sabotage set rather than production.

**228 of 243 train cells moved, and infer and batch moved on every lane.** The
15 that did not are worth stating plainly rather than rounding away:

- **`kmeans-cosine`, all nine fixtures.** Correct by construction, not a gap.
  It is the declared refusal lane: the fit refuses and the refusal sentence IS
  its cell (`_batch_decl("n/a:fit-refused", "kmeans-cosine")`), so no arithmetic
  runs and no sabotage could move it.
- **Six dbscan-family train cells**: `dbscan` on hashed, wide and negative,
  `dbscan-brute-l1` on wide, `dbscan-weighted` on hashed and wide. **On these
  the negative control does not fail, so it demonstrates nothing there.**

  The cause is the fixture, not a gap in the sabotage arm. The dbscan train
  cell hashes `labels_` alone, and labels are integers, so a value perturbation
  can only move the cell if it flips a label. Fitting the lane's own
  configuration (`eps=0.9, min_samples=5` over `X[:6000, :4]`) shows those
  fixtures are degenerate or nearly so:

      base       2 labels   noise 77, cluster 5923     -> moved under sabotage
      wide       1 label    all 6000 in one cluster    -> inert
      hashed     1 label    all 6000 in one cluster    -> inert
      negative   2 labels   noise 5, cluster 5995      -> inert

  `wide` and `hashed` put every row in one cluster, so there is no label left
  to flip; `negative` leaves five noise points out of 6000. That is why five of
  the six cells share one train hash across different lanes and fixtures.

  A correction to my own work, because the first attempt at this reached the
  wrong answer: I tried to confirm degeneracy by hashing candidate label
  vectors and comparing against `d17808b4b9261d5d`, and reported no match. That
  comparison was at the wrong level. `d17808b4b9261d5d` is the TRAIN CELL hash,
  which is a hash of the dict holding the labels hash; the labels hash is
  `1f6fbd85ac2efbda`, and my "all one cluster, int32" candidate produced exactly
  that. The hypothesis was right and my check was wrong.

  This does not sink the dbscan lanes: their `infer` cells moved 9/9 and their
  `batch` cells moved 9/9, so the sabotage does reach their arithmetic. What is
  insensitive is the train part on fixtures whose clustering is degenerate. If
  dbscan is promoted, that limit should be recorded with it, and a fixture that
  actually produces several clusters would be the way to make the train cell
  carry weight.

  This does not sink the dbscan lanes: their `infer` cells moved 9/9 and their
  `batch` cells moved 9/9, so the sabotage does reach their arithmetic. It is
  the train part on those fixtures that is insensitive. Before dbscan is
  promoted, either the cause should be pinned down or the train part's
  insensitivity recorded as a known limit of that cell.

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

### The hold, and its release

The promotion was held until the fixture shrink published its scope, and that
hold was the right call for a concrete reason: these references ship **in the
wheel's table**, so promoting a lane whose fixture then moved would have
shipped a reference a user's `verify` fails against, breaking the exact command
this lane exists to make trustworthy.

`docs/lanes/FIXTURE_SHRINK_SCOPE.md` landed at `e2bb9e541`, shrinking thirteen
lanes. **None of the thirty is among the thirteen** (checked, not eyeballed;
nor is any of the nine already-live lanes), so the evidence taken before the
shrink still stands. It was re-checked anyway on the merged harness, which had
gained 202 lines: `ols-no-intercept` and `kpss` both reproduce their recorded
hashes exactly. The thirty are now promoted.

One consequence worth flagging beyond this lane: the shrink did **not**
regenerate `python/mojolearn/verify_reference/table.json`, so the shipped table
still carries old-size references for the thirteen. No live or promoted lane is
among them, so nothing a user runs is affected today, but the table owes a
regeneration before any shrunk lane becomes publicly checkable.

This matters more than it sounds. These lanes' references ship **in the
wheel's table**. Promoting a lane whose fixture then changes would ship a
reference a user's `verify` fails against, breaking the exact command this lane
exists to make trustworthy. An unpromoted lane with a stale reference is
latent; a promoted one is a user-visible failure.

**`svc-poly` is the one lane NOT promoted**, and not for an arithmetic reason:
its cells rest on two columns (apple and cpu), so it cannot meet
`--require-columns 4`. It stays in `PUBLIC_REFERENCE_CANDIDATES` and joins the
public set the day a record carries its NVIDIA and AMD cells.

## 4b. A verifier a user can watch fail, and evidence instead of a verdict

A user who runs `verify` and reads VERIFIED is trusting two things they cannot
see: that we wrote an honest table, and that the comparison is real. That is
the same defect we spent the week finding in our own code, sitting in the
public command.

**`python -m mojolearn verify --self-test`** runs one lane twice through the
ordinary comparison, the same `run_cell` -> `judge_rows` -> `judge` path every
real lane takes. Untouched must read IDENTICAL; with every value of the input's
first column moved up one ULP it must read DIVERGENT. The perturbation is real
arithmetic at run time, so it needs no sabotage build and no second binding.
Measured on this Mac:

    untouched   3d1d7c30b12d9872  vs reference 3d1d7c30b12d9872  -> IDENTICAL
    perturbed   dd42c9b607526efe  vs reference 3d1d7c30b12d9872  -> DIVERGENT

**It is two-sided on purpose, and that caught my own mistake on the first run.**
The first perturbation moved a SINGLE value, `X[0, 0]`, by one ULP, and the
hash did not move: `ols` fits 20,000 x 16, and one last-bit change in one of
320,000 inputs never reaches the rounded coefficients. I had written a
self-test that could not fail. The clean/perturbed pair reported
`THE VERIFIER IS NOT TRUSTWORTHY ON THIS MACHINE` instead of passing quietly. A
one-sided version would have gone green and shipped a lie. The replacement size
is measured, not chosen: `X[0,0]` one ULP inert, `y[0]` one ULP inert,
`X[:, 0]` one ULP moves it, and a test pins it column-wide so nobody shrinks it
back by accident.

**Proven it would catch a broken comparator**, which is the property it exists
for. Stubbed to always return IDENTICAL, the perturbed arm fails; stubbed to
always return DIVERGENT, the clean arm fails; and the JSON records
`passed: false` carrying both differing hashes. Both directions are now a test
(`test_the_self_test_cannot_pass_with_a_broken_comparator`) rather than
something I did once by hand.

### The evidence document

`--json` and `--json-out PATH` emit the run as data, rendered from the SAME
object as the human report so the two cannot drift. Per cell: the hash computed
here, the hash expected, the verdict, the wall time. Plus what produced the
numbers, where each reference came from as a path under
`bench/results/identity_break/`, the self-test result in the same artifact, and
lane counts separate from cell-part counts.

**Two defects of my own, found by looking at the output rather than assuming:**

- **The binding provenance was empty on exactly the install it matters for.**
  `binding_artifacts()` scans `sys.modules` for `mojolearn._mojolearn*`, but
  host bindings load under `mojolearn._host.*`, so a CPU-only run reported
  **0 bindings** and answered nothing about which binary produced the numbers.
  Now 9, with sha256 and size.
- **Per-cell timings were dropped.** `judge_rows` rebuilds each row and lost the
  `seconds` I attached, so every cell read `None`. Now populated.
- And the lane counts first read **"6 checked of 2 requested"**, because the
  four portable models were folded in with the harness lanes. That is precisely
  the misreadable count the block exists to prevent; they are counted
  separately now.

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
- `D_sab_run.log`, `E_diff_sab.log`, `cpu-candidates-sab.json` — the same 27
  lanes under the sabotage host set: **228 of 243 train cells moved**, and
  infer and batch moved on every lane. See "the sabotage, and where it is
  inert" below.
- `F_exposed_run.log`, `G_exposed_diff.log`, `G_owed.json`, `cpu-exposed.json`
  — kpss, bootstrap, permutation-test and monte-carlo vs the three GPU
  columns: **train IDENTICAL=36 of 36, 0 DIVERGENT, `require-columns 4 ... OK
  (18 OWED)`**.
- `H_exposed_sab.log`, `I_exposed_sabdiff.log`, `cpu-exposed-sab.json` — those
  four under sabotage: **train DIVERGENT=36 of 36, batch DIVERGENT=27, and not
  one cell unchanged**. The sabotage run was bound to the sabotage binaries by
  SHA-256, the newly built resample and forecast ones included.

- `J_verify_after.log` — **the promoted surface end to end**, the run a user
  would make. `python -m mojolearn verify --all --full` on the CPU-only
  install, 39 lanes x 9 fixtures plus the 4 portable models:

      | all | 43 | 1065 | 0 | 158 | 0 | 189 |
      RESULT: VERIFIED (verified 1065 of 1412 cell parts
             (0 divergent, 158 owed, 0 refused, 189 n/a)). 523.7s. exit 0

  **1065 IDENTICAL, 0 DIVERGENT, 0 REFUSED**, against 278 of 332 cell parts
  before the promotion. It exercises both halves of this lane at once: the
  verdict line now says how much was checked, and because nothing refused the
  VERIFIED is an honest one rather than the old exit 0 over a run that mostly
  did not happen. 523.7s, which is why `docs/VERIFY.md` now says 8.7 minutes
  rather than the 20 I had guessed.

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

## 8. Merged, promotion included

Merged to main. Everything on the branch is run and verified. What lands is the
two exposures (proven refusing before, running after, IDENTICAL x4 on 36 of 36
train cells and DIVERGENT on 36 of 36 under sabotage), the false-VERIFIED fix
(its test watched failing against the old code first, and a healthy install
still VERIFIED), the reasons on all 32 families, and the promotion of thirty
lanes measured 0 DIVERGENT with their sabotage arm seen to move. The only lane
left unpromoted is `svc-poly`, and it stays a candidate because it rests on two
columns, not because of anything about its arithmetic.

Two behaviour changes a reader should know landed:

- **The wheels gain a sixteenth host binding**, `_mojolearn_resample_host.so`,
  so the bootstrap, the permutation test and Monte Carlo integration work on a
  CPU-only install. `kpss_test` needs no new binding; it is served from the
  already-shipped forecast binding.
- **`verify --all` exits 4 and prints INCOMPLETE** where it used to print
  VERIFIED and exit 0, whenever any judged part refused. That is the point of
  the change, but it will turn a previously green partial install red, which is
  the correct reading of a run that did not happen.

Marked for 0.8.7 in the CHANGELOG. `release/0.8.6` and `db9047b9f` were never
touched, no Metal job was run, and nothing was rented.

## Resume

    cd /Users/andrewhendel/mojolearn-wt/lane-expose-inference
    git log --oneline -3                 # branch lane/expose-inference-surface
    python3 python/mojolearn/host_surface.py --public-reference-candidates
    python3 python/mojolearn/host_surface.py --saved-model-inference-owed
    python3 python/mojolearn/host_surface.py --wheel-notes
    ls ~/mojolearn-evidence/expose-inference/

The promotion is done, so nothing here is owed to it. What is still owed, in
descending order of how much it would change:

1. **Regenerate `python/mojolearn/verify_reference/table.json`** for the
   thirteen lanes the fixture shrink moved. The shrink did not, so the shipped
   table still carries old-size references for them. No live or promoted lane
   is among the thirteen, so nothing a user runs is wrong today, but any of
   those lanes becomes publicly checkable only after the table is rebuilt.
2. **A GPU record carrying `svc-poly`'s NVIDIA and AMD cells**, which is all
   that stands between it and the public set.
3. **GPU recordings for `dbscan`, `agglomerative` and `spectral` predict**, the
   implemented saved-model inference no gate covers (section 5). `spectral`'s
   is now owed at the new 512-row size.
4. **A dbscan fixture that actually clusters**, if its train cell is ever to
   carry weight: on wide and hashed every row lands in one cluster, so the
   sabotage has no label to flip (section 4).

## Done

- [x] three-population comparison, re-measured from code
- [x] `wheel_note` on all 32 families, test-enforced, no silent exclusion
- [x] **exposed** resample's three functions and `kpss_test`, no fit shipped
- [x] **fixed** the false VERIFIED, with the new test watched failing first
- [x] 30 promotable lanes measured, with two of my own criteria corrected
- [x] sabotage controls proven real before being relied on
- [x] the promoted surface run end to end as a user would: VERIFIED, 1065 of
      1412 cell parts, 0 divergent, 0 refused, in 523.7 s
- [x] **`verify --self-test`**: a user can watch the comparison catch a wrong
      answer, two-sided so a comparator stuck on either verdict fails it,
      proven against a broken comparator in both directions and pinned by tests
- [x] **the evidence document** (`--json`, `--json-out`): per-cell computed and
      expected hashes with timings, binding sha256s, the committed column each
      reference came from as an openable path, the self-test in the same
      artifact, and counts that cannot be misread
- [x] three defects in my own evidence work found by reading the output:
      empty binding provenance on a CPU-only install, dropped per-cell
      timings, and "6 checked of 2 requested"
- [x] the exposure proof: refused before, ran after, IDENTICAL x4 on 36 of 36
      train cells, and DIVERGENT on 36 of 36 under sabotage
- [x] the candidate proof: 0 DIVERGENT, 26 of 27 IDENTICAL x4 on train
- [x] pinned down the six sabotage-insensitive dbscan train cells: degenerate
      clusterings (wide and hashed put all 6000 rows in one cluster), and a
      correction to my own first, wrong diagnosis of them
- [x] **promoted**: 30 lanes joined `public_reference_lanes()` once the shrink
      published its scope, taking the checkable CPU surface from 9 to 39 for
      zero wheel bytes; `svc-poly` stays a candidate on two columns
- [ ] regenerate `verify_reference/table.json` for the thirteen shrunk lanes
      before any of them becomes publicly checkable
- [ ] a GPU recording for dbscan, agglomerative and spectral predict
