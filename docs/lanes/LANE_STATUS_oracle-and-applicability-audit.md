# LANE STATUS: the oracle and applicability audit

lane/oracle-and-applicability-audit, 2026-09-16, rebased onto main 58e352b3f.
Every number below was re-derived after the rebase and none moved.

Andrew's framing, tested rather than assumed: "maybe our entire testing suite
is overbuilt, we built out the CPU suite to support more claims and now we run
that against everything and we should not, something is wrong with our oracle
and what we are testing against."

Half of that is right and it is the second half. The suite is large, but the
structural waste is already small and getting smaller, because two trims
landed today. What is weak is the oracle, and it is weak in a way that size
hides: **149 of the 162 lanes a release record runs assert nothing inside the
cell. Their hash is compared only against a previous hash of the same code, or
against another column of the same code.**

Everything below is derived by `tools/lane_applicability.py`, added in this
pass. Nothing here is hand-listed. Reproduce with

    python3 tools/lane_applicability.py --table
    python3 tools/lane_applicability.py --counts
    python3 tools/lane_applicability.py --oracle-counts
    python3 tools/lane_applicability.py --selfcheck

## SUPERSEDED IN PART, 2026-09-19: the oracle column below, and ten cells

Every number in this file is a snapshot of a 212-lane registry on 2026-09-16.
The registry is 254 lanes now, so re-derive before quoting any of it. Two
things were not merely stale, they were WRONG, and both were corrected on
2026-09-19; the reproduce lines above give the current answer.

1. **The classifier under-read its own evidence.** It looked for
   `_same_bytes` in the lane body only. Six `par-*` lanes hold two objects to
   each other one call away -- `par-byte-lm-model-pool` and
   `par-byte-lm-offload` inside `_byte_lm_replay`, `par-scaler`,
   `par-scaler-minmax`, `par-reference-knn` and `par-reference-knn-reg`
   through a `_mismatch_bytes` they RAISE -- and all six were reported here as
   `recorded-only`, cells that cannot fail. Five lanes read the opposite way:
   `embedding`, `gmm-sample`, `gmm-random-init-sample`, `gp-sample-y` and
   `gp-sample-y-normalize` ask ONE object twice and were reported
   `in-cell-independent`. Both errors are in the per-lane table at the bottom
   of this file.

2. **Four lanes really had no in-cell oracle and now do**: `par-mlp`,
   `par-samba`, `par-samba-clip` and `par-byte-lm`. The first three hold the
   sharded driver to the same ordered fold replayed on one object through the
   public single-device doors; the fourth holds the pooled byte-LM trainer to
   the replicated one the class documents for the purpose. Every `par-*` lane
   now carries an in-cell oracle: `--oracle-counts` reads 0 `recorded` among
   the 53, against 10 on 2026-09-16.

---

## 0. The two numbers

**HOW MANY CELLS CANNOT PROVE ANYTHING ON THE COLUMN THEY RUN ON.**

| scope | column | degenerate lanes | degenerate cells | share |
|---|---|---|---|---|
| registry, 212 lanes | any one-GPU column (apple-metal, nvidia-1gpu, amd-1gpu) | 53 | 477 of 1908 | 25.0% |
| registry, 212 lanes | cpu-host | 54 | 486 of 1908 | 25.5% |
| registry, 212 lanes | a two-GPU column | 3 | 27 of 1908 | 1.4% |
| record scope, 162 lanes | any GPU column | 3 | 27 of 1458 | 1.9% |
| record scope, 162 lanes | cpu-host | 4 | 36 of 1458 | 2.5% |

The registry row is what a caller who passes an explicit `--lanes` list can
still ask for. The record row is what a full column costs today, because
`tools/identity_break.py` already drops every `par-` lane from a run with no
`--lanes` (`RECORD_EXCLUDED_PREFIXES`, enforced at its lane selection). The
last full Apple and AMD columns actually recorded, on 2026-09-14, predate that
trim and carry the full cost: **279 of their 1494 cells are `par-*` cells run
on ONE device**, 18.7% of the column, and a further 1595 of the 5976 part
slots (train, infer, model, batch) are `n/a`, 26.7%.

**HOW MANY LANES ARE CHECKED AGAINST AN INDEPENDENT ORACLE.**

51 of 212, 24.1%, hold a differently built object against the lane's own and
raise when the bytes differ. **44 of those 51 are `par-*` driver lanes, which
a release record does not run.** Inside the record scope of 162 lanes the
count is **7**.

---

## 1. Method, and why it is derived

`identity_break.lane` is three lines that put a function in a dict. A lane
carries no column, no vendor, no device count and no tier. So "run every lane
on every column" was never a decision anyone made: it is the only thing the
structure can express, and the set is then trimmed by hand at each call site.
A hand-kept trim agrees with everything a few weeks after it is written, which
is the shape of six failures in this repository.

`tools/lane_applicability.py` therefore derives every fact from something
already enforced, the way `tools/lane_select.py` derives its edges. It answers
a different question from that file and does not edit or duplicate it:
`lane_select` answers "which lanes can a change move", this one answers "on
which columns is a lane's proposition expressible, and what is it compared
against".

| fact | derived from |
|---|---|
| the lane set | `identity_break.LANES`, read by IMPORT. A grep of `@lane(` finds 189; the registry holds 212, because 23 lanes register by call (the kde, knn, radius, gp and gmm families). |
| the device claim | `_par_devices` in the lane body's transitive CALL closure over the harness's own module-level `def`s. |
| the oracle class | every in-cell call that RAISES on disagreement (`_same_bytes`), with each side resolved to its root binding. |
| the CPU route | `host_surface.covered_lanes()`, that file's own answer, and the `_mojolearn_*` bindings each public class's door file names. |
| the record scope | `identity_break.record_lanes()` and `record_excluded_lanes()`. |

Two traps hit on the way, both worth recording:

* The first closure walked every `Name` and every attribute, and `rf-clf` came
  out as a two-device driver lane. `fit` and `predict` are the names of both
  harness helpers and estimator methods, and one collision hands every lane
  every helper. The fix is the call graph itself: a bare-name call is
  unambiguous, an attribute is not. `lane_select` pays for the same collision
  at file granularity with its `ENUMERATOR_MAX_BINDINGS` sink.
* The derived device claim is **checked against the lane name, never taken
  from it**. On this commit the two agree exactly: 50 lanes reach
  `_par_devices`, 50 lanes are named `par-*`, zero disagreements. That is the
  evidence that the harness's own `RECORD_EXCLUDED_PREFIXES = ("par-",)` trim
  is correct TODAY. It is a name prefix, so nothing keeps it correct; a
  disagreement is now an error the selfcheck fails on.

---

## 2. The truth table, by category

Five kinds fall out of the derivation. For each: what it proves, what would
have to break for it to fail, and where the proposition is expressible.

### multi-device-driver, 50 lanes

Every `par-*` lane. The lane hands a sharding driver `devices=_par_devices()`.

* **Claims** that splitting the work across devices changes no bit: in
  `_par_devices`'s own docstring, "a two-device column must hash equal, cell
  for cell, to the one-device column of the same commit; that equality is the
  drivers' whole claim".
* **Fails when** the shard split, the cross-device copy or the recombination
  order changes a bit. On AMD that is not theoretical: the MI300X SR-IOV
  stale-read finding was exactly a cross-device copy read before it was
  written.
* **Expressible on** two-device columns only. `_par_devices()` defaults to
  `"0"`, one device, so on every one-GPU column the equality is not false, it
  is not stated. 450 of the 1908 registry cells.
* **What survives on one device**: 44 of the 50 also hold the sharded fit
  against a plain fit in the cell (`par-forest` against a plain
  `RandomForestClassifier`). That comparison fires with one device, so the
  LOGICAL shard split is tested there; the DEVICE axis is not. The other 6
  (`par-byte-lm`, `par-byte-lm-model-pool`, `par-byte-lm-offload`, `par-mlp`,
  `par-samba`, `par-samba-clip`) assert nothing in the cell, so on a one-GPU
  column they are a hash of one shard on one device held only against a
  previous hash of themselves. The source-derived answer for
  `par-byte-lm-model-pool` and `par-byte-lm-offload` is the same one
  `docs/lanes/LANE_STATUS_lane-cpu-training-par-wave3.md` reached by hand:
  they compare host arithmetic with itself under a pooled label.

The harness already excludes all 50 from a full-column run and the two-device
legs are recorded separately (`amd-2xmi300x-gfx942.par-devices-0-1.json`,
`nvidia-2xh100-sm_90a.par-devices-0-1.json`, 279 cells each). **The trim is
right. The mechanism is a name prefix.**

### cpu-host-route-only, 3 lanes

`byte-lm-host-infer`, `byte-lm-host-infer-threaded`, `byte-lm-host-train`.

* **Claims** that the CPU forward and the CPU training step give fixed bits.
  `byte-lm-host-infer`'s own docstring: "its certificate is per CPU; this lane
  measures whatever CPU the box has."
* **Fails when** the host loop's arithmetic or accumulation order moves.
* **Expressible on** any column, but on a GPU column it measures that box's
  CPU and says nothing about Metal, CUDA or HIP. These are the 3 lanes still
  degenerate inside the record scope: 27 cells per GPU column. On Apple that
  is the scarcest hardware we own spent on an M4 CPU number a rented pod gives
  in parallel for $0.24/hour.

### gpu-only, 2 lanes

`gp-optimize`, `gp-optimize-restarts`: no CPU route in `host_surface`.

* **Expressible on** GPU columns. On the CPU column the cell REFUSES. A
  refusal and a pass read the same in a column total, which is how a
  wheel-installed CPU column once reported coverage for 747 cells it refused.

### vendor-independent-arithmetic, 17 lanes

The function lanes: `gemm-pinned`, `gemm-transposed`, `metrics*`, `cross-val`,
`bootstrap`, `permutation-test`, `monte-carlo`, `optim-sgd`, `optim-adam-clip`,
`cross-entropy-arms`, `kpss`, `bpe-trainer`, `rf-score-weighted`,
`gbdt-adapter-score-weighted`, `cross-val-folds`, `par-resample`,
`par-byte-lm*`. Detected from source, not from a list: every `_fit(...)` in the
body passes one argument, so `Fit.probe` stays at its class default
`n/a:function` and no infer, model or batch cell exists for them.

* **Claims** a pinned reduction order.
* **Expressible on** every column that has the route. Two of them
  (`bpe-trainer`, `cross-val-folds`) have no CPU route, so the CPU column
  refuses them.

### estimator-both-routes, 140 lanes

The bulk. A public estimator with both a device kernel and a CPU host
restatement.

* **Claims** that the same fit on every vendor gives the same bits.
* **Fails when** a vendor's kernel disagrees with another vendor's, which is
  visible only in `--diff` across columns.
* **Expressible on** every column. One column alone proves nothing about them
  beyond run-to-run stability. See section 3.

The full per-lane table is section 6.

---

## 3. The oracle, which is the real finding

For each lane, WHAT IS A PASSING CELL COMPARED AGAINST?

### The three classes asked for

| class | lanes | share |
|---|---|---|
| (i) independent, in the cell | 51 | 24.1% |
| (ii) self-comparison, in the cell | 6 | 2.8% |
| (iii) a recorded hash of the same code | 155 | 73.1% |

(i) is `_same_bytes` between two roots built by two different call chains:
`par-forest` holds `fit_forest(...)` against `ml.RandomForestClassifier(**kw).fit(...)`,
and raises naming the pair. It can fail on one box with no record at all.
44 of the 51 are `par-*`; the other 7 are `byte-lm-resident`, `embedding`,
`embedding-sort`, `gmm-sample`, `gmm-random-init-sample`, `gp-sample-y`,
`gp-sample-y-normalize`.

(ii) is `_same_bytes` between two doors of ONE fitted object: the four ARIMA
and two Holt-Winters lanes hold `forecast` against `predict` on the same fit,
`ivf-extend` holds an extended index against itself. It proves the doors agree
and cannot see a defect they share, which is every defect in the arithmetic
underneath them.

(iii) is everything else. The cell is a hash. It means something only when
`--diff` holds it against a previous hash, which is a regression test against
ourselves: **a defect that was present when the reference was recorded is
invisible forever.**

### The fourth class, which the tree actually leans on

Class (iii) is not as bad as 73% sounds, and the reason is the category the
audit asked for as "proves something only in combination with another column".
A lane with BOTH a device kernel and a CPU host route has two separate
implementations in this tree, so a GPU column diffed against the CPU column IS
an independent comparison, just not one that lives in the cell.

| anchor | all 212 | in the 162-lane record | has a live batch part |
|---|---|---|---|
| in-cell-independent | 51 | 7 | 46 |
| in-cell-self | 6 | 6 | 6 |
| cross-route (needs a second column) | 148 | 145 | 141 |
| recorded-only (nothing, anywhere) | 7 | 4 | 2 |

Three consequences, in order of how much they should change behavior.

1. **Running a cross-route lane twice on the same column adds zero evidence.**
   145 of the 162 record lanes are in that class. Their entire value is the
   diff against a column computed by different code. A second Apple column at
   the same commit, or an Apple column with no CPU column beside it, is not
   weak evidence, it is no evidence. This is the precise, defensible version
   of Andrew's "we run that against everything": the CPU suite is not the
   problem, it is the only thing making 145 of these lanes mean anything, and
   what should stop is asking a single column to carry them alone.

2. **Four record lanes have no oracle in any combination**: `bpe-trainer`,
   `cross-val-folds`, `gp-optimize`, `gp-optimize-restarts`. No in-cell
   assertion, no CPU route, so nothing but a recorded hash of themselves.
   `gp-optimize` and `gp-optimize-restarts` at least carry a live batch part.
   `bpe-trainer` and `cross-val-folds` carry nothing that can fail on one box.
   These four are where an oracle should be added, not where lanes should be
   cut.

3. **The suite's growth added oracle strength, not only bulk.** The
   2026-08-29 file carries 28 lanes, not 29: `bench/results/identity_break/apple-m4.identical.txt`
   has 29 table rows and one of them is the `| lane |` header. All 28 are
   still registered today, none was retired. Of them, 27
   are cross-route and 1 is in-cell-self, and **none has an in-cell
   independent oracle.** All 51 in-cell independent oracles arrived in the 184
   lanes added since. So "the suite is overbuilt" is not supported as stated:
   the additions are where the strong checks are. What is true is that 121 of
   those 184 additions are cross-route, so they inherited the weakness of the
   original 28 rather than fixing it.

### What else can fail in a run, to be fair to the harness

195 of the 212 lanes carry a live `batch` declaration, and the batch, rlpair,
ragged, batchscale, batchgrad and stepfull parts are real within-run
invariance checks that raise on one box. They are class (ii): the same
implementation asked in different shapes. They catch a shape-dependent defect
and cannot catch a wrong kernel. Counting them, most cells are not inert; they
are just not checked against anything that computes the answer a second way.

---

## 4. The proposed mechanism

`tools/lane_applicability.py`, added in this pass. Derived, and it REFUSES.

    LaneNotApplicable            raised by check(lanes, column, allow=())
    Scope.applicable(column)     -> (ok, reason)
    scopes()                     -> lane -> Scope, derived at call time
    degenerate(column)           -> lane -> reason

`allow=` exists so an exception is a written-down decision with a name
attached, and every allowed lane is still printed. There is no quiet skip: a
skipped cell and a passing cell look identical in a summary line, which is how
two shrunken cells went blind today and read as coverage.

### The refusal, demonstrated

```
$ python3 tools/lane_applicability.py --check --column apple-metal \
    --lanes par-forest par-samba byte-lm-host-infer rf-clf
REFUSING: 3 of 4 lane(s) cannot state their proposition on the apple-metal column.
Running them there produces cells that pass because nothing they assert can fail,
which reads in a summary exactly like coverage.
  par-forest: DEGENERATE (device axis): a multi-device driver lane on a column with
    1 device(s). Its claim, in `_par_devices`'s own docstring, is that a two-device
    column hashes equal cell for cell to the one-device column; with one shard that
    equality is not false, it is not expressible. the in-cell comparison against a
    plain fit still fires, so the LOGICAL SHARD split is tested here; the DEVICE
    axis is not
  par-samba: DEGENERATE (device axis): ... and this lane asserts nothing in the cell
    either, so on this column it is a hash of one shard on one device held only
    against a previous hash of itself
  byte-lm-host-infer: DEGENERATE: the lane's arithmetic is the CPU host route
    (LanguageModelInference), so on the apple-metal column it measures that box's
    CPU and says nothing about metal
  Drop them from this column, or name them in `allow` with a reason.
exit=2

$ python3 tools/lane_applicability.py --check --column nvidia-2gpu \
    --lanes par-forest par-kmeans rf-clf
ok: 3 lane(s) are all expressible on nvidia-2gpu
exit=0
```

The second command is the arm that matters as much as the first: a refusal
that fires on everything is a blanket and proves nothing.

### Watched failing first

`--selfcheck` has eight arms. Two negative controls were run with the fix
removed, and both fail, printing the matches rather than a count:

* `RAISING_COMPARE = ()`, blinding the oracle classifier:
  `SELFCHECK FAIL: par-forest ... must read independent`,
  `no lane classified self: the classifier collapsed`,
  `fewer than three oracle classes appear`,
  `holtwinters holds forecast to predict on ONE fitted object; it must read self`,
  `fewer than four anchor classes appear`, `selfcheck: FAILED`.
* The device clause replaced by `if False:`:
  `SELFCHECK FAIL: check() did NOT refuse par-forest on a one-device column`,
  `selfcheck: FAILED`.

Both restored from a byte copy, never `git checkout --`, and `selfcheck: ok`
after each.

Two further arms exist because the derivation must not be reading the lane
names: `la_scope_of_body` takes a function OBJECT with no name in hand and must
answer 2 for `par-kmeans` and 1 for `kmeans`.

### What to do with it, in order

1. **Wire the refusal into the harness.** At `tools/identity_break.py`'s lane
   selection, after the `--skip` filter and before the Apple guard, derive the
   column from `vendor`, `len(_par_devices())` and `host`, then call
   `lane_applicability.check(lanes, column)`. Not done in this pass, because
   it changes what an existing run does and that should be a decision, not a
   side effect of an audit. The patch is six lines and the column derivation
   is already in the harness.
2. **Replace the prefix trim with the derived test.** `RECORD_EXCLUDED_PREFIXES
   = ("par-",)` becomes "every lane whose body reaches `_par_devices`". Today
   the two sets are identical, proved above, so this is a no-op that stops
   being one the first time a driver lane is named something else.
3. **Drop the 3 host-route lanes from GPU columns**, or give them a reason in
   `allow`. 27 Apple cells, on the hardware that cannot be rented.
4. **Give the four unanchored record lanes an oracle**, or say in their
   docstrings that they are regression tests against a recorded hash and
   nothing more. `bpe-trainer` and `cross-val-folds` first, since they have no
   batch part either.
5. **Say in the record README which columns a diff needs.** A cross-route lane
   with one column is not weak evidence, it is none, and 145 of 162 record
   lanes are cross-route. A column recorded without the CPU column beside it
   should say so on its face.

### What this proposal needs from another session's files

Nothing in it changes the public surface: no estimator, parameter, default or
exported name moves, and nothing here touches `python/mojolearn/host_surface.py`,
`python/mojolearn/_verify_all.py`, `python/mojolearn/__main__.py` or
`docs/VERIFY.md`. Two items would need them and are written down rather than
done:

* **`host_surface.py`**: this audit reads `covered_lanes()` as "a CPU route
  exists". It is a training-coverage answer and is not quite the same
  question. A `route_lanes()` that says only "the CPU column has arithmetic
  for this lane" would make the CPU-column clause exact rather than close.
  Route it if you want that.
* **`_verify_all.py` / `__main__.py`**: nothing needed. The refusal belongs in
  `identity_break.py`, which is a dev tool and ships in no wheel.

---

## 5. What was NOT done, on purpose

No lane was deleted, shrunk or skipped. Deleting coverage is the mistake that
produced today's other finding, where two shrunken cells went blind and read
as passing. No fixture, `verify_reference/table.json` entry or record was
touched. No GPU was used: this is source analysis, and every number above
comes from parsing the tree and from JSON already on disk.

---

## 6. The truth table, all 212 lanes

`kind` is the applicability category, `oracle` is the in-cell class,
`anchor` is what actually holds the lane down, `claim_devices` is the device
count the lane's own claim needs, and each column says `yes` or `DEGENERATE`.
Regenerate with `python3 tools/lane_applicability.py --table`.

| lane | kind | oracle | anchor | claim_devices | cpu_route | in_record | amd-1gpu | amd-2gpu | apple-metal | cpu-host | nvidia-1gpu | nvidia-2gpu |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agglomerative | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| arima | estimator-both-routes | self | in-cell-self | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| arima-011 | estimator-both-routes | self | in-cell-self | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| arima-exog | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| arima-exog-seasonal | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| arima-seasonal-c | estimator-both-routes | self | in-cell-self | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| bootstrap | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| bpe-trainer | vendor-independent-arithmetic | recorded | recorded-only | 1 | no | yes | yes | yes | yes | DEGENERATE | yes | yes |
| byte-lm | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| byte-lm-host-infer | cpu-host-route-only | recorded | cross-route | 1 | yes | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes | DEGENERATE | DEGENERATE |
| byte-lm-host-infer-threaded | cpu-host-route-only | recorded | cross-route | 1 | yes | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes | DEGENERATE | DEGENERATE |
| byte-lm-host-train | cpu-host-route-only | recorded | cross-route | 1 | yes | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes | DEGENERATE | DEGENERATE |
| byte-lm-resident | estimator-both-routes | independent | in-cell-independent | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| cholesky | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| cross-entropy-arms | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| cross-val | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| cross-val-folds | vendor-independent-arithmetic | recorded | recorded-only | 1 | no | yes | yes | yes | yes | DEGENERATE | yes | yes |
| dbscan | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| dbscan-brute-l1 | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| dbscan-weighted | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| elasticnet | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| elasticnet-l2end-no-intercept | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| embedding | estimator-both-routes | independent | in-cell-independent | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| embedding-sort | estimator-both-routes | independent | in-cell-independent | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| et-clf | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| et-clf-entropy-bestfirst | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| et-reg | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| et-reg-bootstrap-parallel | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-adapter-clf | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-adapter-reg | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-adapter-score-weighted | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-categorical-ctr | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-categorical-ctr-tables | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-depthwise | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-exact-mae | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-feature-freq | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-lossguide | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-lossguide-newtoncosine | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-multiclass | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-nan-modes | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-onevsall | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-ordered-rmse | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-pair-logit | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-parametric-losses | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-pointwise-l2-bayesian-eval | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-query-rmse | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-rmse | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-symmetric | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-tensor-ctr-tables | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gbdt-yeti-rank | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gemm-pinned | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gemm-transposed | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gmm | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gmm-random-init | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gmm-random-init-sample | estimator-both-routes | independent | in-cell-independent | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gmm-sample | estimator-both-routes | independent | in-cell-independent | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gp | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gp-matern12 | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gp-matern32 | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gp-matern52-ard | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gp-normalize-y | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gp-optimize | gpu-only | recorded | recorded-only | 1 | no | yes | yes | yes | yes | DEGENERATE | yes | yes |
| gp-optimize-restarts | gpu-only | recorded | recorded-only | 1 | no | yes | yes | yes | yes | DEGENERATE | yes | yes |
| gp-sample-y | estimator-both-routes | independent | in-cell-independent | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gp-sample-y-normalize | estimator-both-routes | independent | in-cell-independent | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gpc | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| gpc-multiclass | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| hdbscan | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| hdbscan-leaf | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| holtwinters | estimator-both-routes | self | in-cell-self | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| holtwinters-multiplicative | estimator-both-routes | self | in-cell-self | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| iforest | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| iforest-tuned | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| ivf | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| ivf-euclidean | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| ivf-extend | estimator-both-routes | self | in-cell-self | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kde | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kde-cosine-minkowski | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kde-epanechnikov-l1 | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kde-exponential-chebyshev | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kde-linear-cosine | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kde-tophat-sqeuclidean | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kde-weighted | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kernel-ridge | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kmeans | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kmeans-array | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kmeans-classic-pp | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kmeans-cosine | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kmeans-random | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kmeans-sqrt | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kmeans-weighted | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn-chebyshev | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn-clf | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn-clf-distance | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn-cosine | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn-manhattan | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn-minkowski-p3 | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn-rbc | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn-reg | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn-reg-distance | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| knn-sqeuclidean | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| kpss | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| lasso | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| logistic | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| logistic-elasticnet | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| logistic-l1 | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| logistic-multiclass | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| logistic-unpenalized-no-intercept | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| mamba1 | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| mamba2 | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| mamba2-dtlimit | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| mamba3 | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| metrics | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| metrics-classification | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| metrics-fowlkes-mallows | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| minmax-scaler | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| minmax-scaler-clip | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| mlp | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| monte-carlo | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| nystroem | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| ols | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| ols-no-intercept | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| ols-weighted | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| optim-adam-clip | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| optim-sgd | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| par-arima | multi-device-driver | independent | in-cell-independent | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-boosting | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-boosting-clf | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-boosting-pointwise | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-boosting-reg | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-byte-lm | multi-device-driver | recorded | recorded-only | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-byte-lm-model-pool | multi-device-driver | recorded | recorded-only | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-byte-lm-offload | multi-device-driver | recorded | recorded-only | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-cd | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-cd-elasticnet | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-cholesky | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-dbscan | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-feature-freq | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-forest | multi-device-driver | independent | in-cell-independent | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-forest-et | multi-device-driver | independent | in-cell-independent | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-forest-et-clf | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-forest-pool | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-forest-reg | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-gmm | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-gp | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-gram | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-gram-ols | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-gram-pca | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-gram-tsvd | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-graph-agglomerative | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-graph-spectral | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-graph-umap | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-hdbscan | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-holtwinters | multi-device-driver | independent | in-cell-independent | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-iforest | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-kernel-ridge | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-kmeans | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-logistic | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-mlp | multi-device-driver | recorded | cross-route | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-nystroem | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-ordered-rmse | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-queries-kde | multi-device-driver | independent | in-cell-independent | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-queries-knn | multi-device-driver | independent | in-cell-independent | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-queries-nn | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-queries-radius | multi-device-driver | independent | in-cell-independent | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-rbf-sampler | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-reference-knn | multi-device-driver | independent | in-cell-independent | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-reference-knn-reg | multi-device-driver | independent | in-cell-independent | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-resample | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-samba | multi-device-driver | recorded | cross-route | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-samba-clip | multi-device-driver | recorded | cross-route | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-scaler | multi-device-driver | independent | in-cell-independent | 2 | yes | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-scaler-minmax | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-svm | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| par-svm-svr | multi-device-driver | independent | in-cell-independent | 2 | no | no | DEGENERATE | yes | DEGENERATE | DEGENERATE | DEGENERATE | yes |
| pca | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| pca-full-whiten | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| pca-whiten | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| permutation-test | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| radius | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| radius-chebyshev | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| radius-manhattan | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| radius-minkowski-p3 | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| rbf-sampler | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| rf-clf | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| rf-clf-balanced-parallel | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| rf-clf-entropy-log2-noboot | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| rf-reg | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| rf-reg-gamma-ig | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| rf-reg-poisson | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| rf-score-weighted | vendor-independent-arithmetic | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| ridge | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| ridge-no-intercept | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| samba | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| samba-untied-dropout-accum | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| spectral | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| spectral-precomputed | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| standard-scaler | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| standard-scaler-no-mean | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| standard-scaler-no-std | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| svc | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| svc-linear | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| svc-poly | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| svr | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| svr-linear | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| tokenizer | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| training-primitives | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| transformer | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| transformer-window | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| tsvd | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
| umap | estimator-both-routes | recorded | cross-route | 1 | yes | yes | yes | yes | yes | yes | yes | yes |
