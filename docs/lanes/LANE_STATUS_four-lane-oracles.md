# LANE STATUS: an independent oracle for the four recorded-only lanes

lane/four-lane-oracles, 2026-09-16, cut from main `bd87fd024`. 0.8.8 work.
Nothing that ships is touched. Three new files under `tools/`, six task lines
in `pixi.toml`, and this document, and no lane body, no fixture, no
`LANE_REVISIONS`, no `verify_reference/table.json`, no Mojo source.

`docs/lanes/LANE_STATUS_oracle-and-applicability-audit.md` (merged as
`c3d0cd422`) found that of the 162 lanes a release record runs, four are
anchored by nothing but a stored hash of the same code: `bpe-trainer`,
`cross-val-folds`, `gp-optimize` and `gp-optimize-restarts`. They can prove the
code has not changed. They cannot see a defect that was present when the
reference was recorded.

Each of the four now has a check that computes the answer a second way and can
disagree, and each of those checks has been SEEN TO FAIL against a real
implementation made wrong on purpose.

    pixi run check-bpe-trainer-oracle
    pixi run check-cross-val-folds-oracle
    pixi run check-gp-optimize-oracle
    pixi run check-gp-optimize-restarts-oracle

---

## 0. What the audit got right, and the two places it is wrong

The classification was re-derived from source before anything was built on it.

**`bpe-trainer` and `cross-val-folds`: the audit is right, and understated.**
Both lane bodies are pure Python integer work with no native call in the
selection. So they do not merely lack a CPU-versus-GPU pair, they compute THE
SAME BYTES ON EVERY COLUMN BY CONSTRUCTION. A cross-column diff of these two
cells is not weak evidence, it is arithmetically incapable of carrying any.

Two things the audit's derivation could not see, both real and neither
sufficient:

* `bpe-trainer` DOES have a second implementation in this tree.
  `pixi run check-bpe-trainer` holds `tokenizer/train/bpe_train.mojo` against
  `python/mojolearn/_bpe_trainer.py` file byte for file byte in both output
  formats. It is not in the cell, it needs a Mojo compile, and it is not what
  any column compares the lane against. It is also NOT INDEPENDENT in the sense
  that matters here, because the Mojo trainer was written against that Python file as
  its stated reference, so a wrong tie-break in the Python file is wrong in
  both and the byte-for-byte check passes.
* `cross-val-folds` DOES carry an in-cell property check. Its `partition` part
  is 1 when every fold is nonempty, train and test are disjoint, train is the
  complement, and the test blocks hold every row exactly once. The audit's
  `_same_bytes` derivation cannot see it because the lane HASHES it instead of
  raising. It is still not an oracle for what can go wrong. Measured below,
  under the lane's own `MOJOLEARN_FOLD_ORDER_SABOTAGE` every one of those
  invariants still passes, 136 of 136.

**`gp-optimize` and `gp-optimize-restarts`: the audit's `gpu-only` row is
wrong.** It says "no CPU route in `host_surface`. On the CPU column the cell
REFUSES." Measured on a CPU-only install with `_mojolearn_gp_host.so` present
and no GPU binding at all, on this worktree at `bd87fd024`:

    # vendor=cpu-apple-m4 host.column=cpu host.families=['_mojolearn_gp_host']
    | gp-optimize            | 736cdac8b4326966 |
    | gp-optimize infer      | d9df0d407450a397 |
    | gp-optimize model      | e00579045e0d743e |
    | gp-optimize batch      | f88dad7a538dcc0d |
    | gp-optimize-restarts   | 4a28559715cdc8ec |
    ...
    cells=4 stable=4 moved=0 refused=0

Both lanes produce STABLE cells on the CPU column. The gp host family
`routes="_mojolearn_gp"` wholesale, so the estimator resolves even though
neither lane name appears in that family's lane-description dict, which is what
`host_surface.covered_lanes()` reads and what the audit derived from. And the
two routes are genuinely different code. The GPU binding calls
`gaussian_process/estimator.mojo::gpr_lml_grad_host`, the host binding calls
`gaussian_process/host/gpr_grad_oracle.mojo::gpr_host_lml_grad`, whose own
header describes it as that function restated on the host.

So these two are CROSS-ROUTE, not recorded-only, and the audit's count of four
recorded-only record lanes should be two. `tools/lane_applicability.py` derives
the CPU route from `covered_lanes()`, which is a per-lane list; the route it
should derive from is the family's `routes` field, which is per binding. That
is a one-line difference in the derivation and the audit's own rule applies to
it, because a hand-kept per-lane list agrees with everything a few weeks after
it is written.

`gp-optimize` also already has an outside-the-cell oracle the audit did not
count. `python/mojolearn/tests/test_gp_optimizer.py`'s `arm_reference` compares
the optimized likelihood with scikit-learn's, and `arm_gradient` compares the
analytic gradient with central differences and with scikit-learn's. Three
limits, all of which the new check removes. It runs on its own fixture and not
the lane's, it checks the LIKELIHOOD and merely PRINTS the theta difference
(`rep.check("REFERENCE", True, ...)`, a check that cannot fail), and it
evaluates the gradient at the STARTING theta, never at the returned one.

None of this removes the gap. A cross-route diff needs two columns and says
nothing about whether the answer is the one the optimizer was asked for.

---

## 1. `bpe-trainer`

**Oracle shape: an independently written reference implementation** (shape (a)),
`tools/bpe_trainer_oracle_check.py`.

Byte-level BPE written from the algorithm's definition and held against
`mojolearn.tokenizer.BpeVocabularyTrainer` on the lane's own corpus, which is
the first 4,096 bytes of the `base` fixture viewed as bytes. It takes a
different route at every step where one was available:

| ours | the reference |
|---|---|
| pre-tokens grouped into `(piece, count)` pairs, sorted by bytes | one entry per pre-token OCCURRENCE, ungrouped and unsorted |
| selection scans `sorted(counts)` and keeps the first pair at the top count | selection takes the top count with `max`, SORTS the pairs at that count, and takes the first |
| a hand-decoded UTF-8 table and a literal White_Space range list | CPython's strict UTF-8 decoder over each 1..4 byte candidate, and White_Space derived from `unicodedata` as Zs, Zl, Zp plus U+0009..U+000D and U+0085 |
| `render_tokenizer_json` GENERATES the artifact text | the artifact is PARSED with `json.loads` and its vocab and merges are read back |

**Result on the lane's corpus: full agreement, and the tie-break is reached.**

    PRETOKEN: lane: the pre-token sequence agrees (3429 pre-tokens)
    TRAIN:    lane: the 302-token vocabulary agrees
    TRAIN:    lane: the 46 merges agree in order
    TRAIN:    lane: n_ties_broken agrees (42)
    TIES:     lane: the tie-break is REACHED (42 selections were ties)

Six corpora in all (the lane's, `ties_corpus()`, `synthetic_corpus()`, the same
at `min_frequency=5`, the `ties` fixture's bytes, and a two-document corpus).
Every one agrees, tokens, merges, tie counter and both artifacts.

**No disagreement found.** The independently derived White_Space set agrees
with the table in `_tokenizer_synthetic.py` and the independently decoded
UTF-8 agrees with the hand-rolled decoder, on all six corpora.

**What it can catch.** A tie-break that is not the stated total order, in
either direction; a merge applied overlapping instead of left to right
non-overlapping; a pair counted non-overlapping instead of overlapping; a
`min_frequency` off by one; a vocabulary that stops a merge early or late; a
pre-token boundary rule that differs from the pattern, including the
`\s+(?!\S)` backtrack and the ` ?\p{L}++` leading space; a `tokenizer.json`
whose vocab or merges disagree with the ranks file beside it.

**What it cannot catch.** A defect in the GPT-2 pattern string itself, which
both sides read from `_tokenizer_synthetic.PAT_STR`. Anything about the Mojo
trainer; this file never compiles Mojo. A corpus with properties none of the
six have. The byte-to-unicode spelling table, which both sides take from
`byte_to_char()`.

### Seen to fail, four ways

**1. The lane's own env sabotage, `MOJOLEARN_BPE_TRAINER_SABOTAGE=1`** (the
tie-break reversed to the largest `(left_id, right_id)`). The reference always
follows the STATED rule and never reads `sabotaged()`, which is what makes this
expressible at all.

    FAIL  TRAIN: lane: the 302-token vocabulary agrees
            len ours=301 ref=302
            first difference at rank 257
              ours = 3f7e  (b'?~')
              ref  = 3f3e  (b'?>')
              [258] ours=3f3e ref=3f7e
              [259] ours=cabe ref=053f
    FAIL  TRAIN: lane: the 46 merges agree in order
              ours = (63, 126, 257)
              ref  = (63, 62, 257)
    FAIL  TRAIN: lane: n_ties_broken agrees (42)
              ours = 41   ref = 42

**2. Scratch sabotage: `_apply` advances one instead of two** (an overlapping
rewrite). Restored from a byte copy; `_bpe_trainer.py` back to
`5b251e77fe3fb494...`.

    FAIL  TRAIN: lane: the 302-token vocabulary agrees
            len ours=320 ref=302
            first difference at rank 257
              ours = 3f1a1a  (b'?\x1a\x1a')
              ref  = 3f3e    (b'?>')
              [258] ours=3f1a1a1a ref=3f7e
    FAIL  TRAIN: lane: n_ties_broken agrees (42)
              ours = 0   ref = 42

**3. Scratch sabotage: `_select` iterates the dict instead of
`sorted(counts)`.** THE MOST INSTRUCTIVE OF THE THREE. The vocabulary stays 302
tokens, the merge count stays 46, and `n_ties_broken` stays 42, so every counter
the cell hashes is unmoved. Only the reference sees it:

    FAIL  TRAIN: lane: the 302-token vocabulary agrees
            len ours=302 ref=302
            first difference at rank 259
              ours = 083f  (b'\x08?')
              ref  = 053f  (b'\x05?')
              [260] ours=053f ref=083f
    FAIL  TRAIN: lane: the 46 merges agree in order
            first difference at merge 3
              ours = (8, 63, 259)
              ref  = (5, 63, 259)
    ok    TRAIN: lane: n_ties_broken agrees (42)

**4. Scratch sabotage: `_pair_counts` counts NON-overlapping.**

    FAIL  TRAIN: lane: the 302-token vocabulary agrees
            len ours=293 ref=302
            first difference at rank 256
              ours = 2b3f  (b'+?')
              ref  = 3f1a  (b'?\x1a')
    FAIL  TRAIN: lane: the 46 merges agree in order
              ours = (43, 63, 256)
              ref  = (63, 26, 256)
    FAIL  TRAIN: lane: n_ties_broken agrees (42)
              ours = 35

---

## 2. `cross-val-folds`

**Oracle shape: a combinatorial invariant AND an independent recomputation**
(shape (c), plus shape (a) for the assignment),
`tools/cross_val_folds_oracle_check.py`.

The invariant arm alone is not enough and the lane's own docstring says why, so
the file carries both and the sabotage below shows exactly what each one buys.

* **PARTITION**, which RAISES rather than hashing, and adds two checks the cell
  does not make, fold sizes differing by at most one and, on the stratified
  branch, every class spread within one fold.
* **REFERENCE**, the independent recomputation. Ours never builds the sorted
  encoded label vector; it counts each residue modulo `n_splits` in closed form,
  `first = (fold - offset) % n_splits`. The reference MATERIALIZES the sorted
  encoded vector, takes the round-robin stripes `y_order[i::n_splits]`, counts
  each class in each stripe and repeats the fold number that many times, which
  is the definition the closed form is a shortcut for.
* **SWITCH**, four switches that must flip, including the KFold hole held open
  on purpose, because it is a function of `len(y)` alone, so changing a label must leave
  it ALONE, and if it ever starts moving the lane's docstring is stale.
* **ORDER**, the lane's own finding held open: a within-class rotation must
  leave the stratified INDICES identical, must move the fold CONTENT, and must
  move `split_descriptor`'s `sha256` and `X_sha256` while leaving
  `fold_assignment_sha256` alone. The `sha256` one is what catches a descriptor
  that stopped reading X.
* **ENCODING**, a compatibility measurement rather than a verdict.

Five label sequences (the lane's classifier labels, the lane's regression
labels, an unbalanced three-class sequence whose first-seen order is not its
sorted order, a four-class interleave, and a two-class split at 1021/1027) at
four split counts each, 2, 3, 5 and 7.

**Result: everything holds, 148 checks.**

**No disagreement found, and one suspicion checked and withdrawn.** The
docstring of `_default_folds` says "Reference: sklearn 1.9.1 ... first-seen
class encoding", and scikit-learn was assumed here to encode classes in SORTED
order, which would have made the folds differ from theirs whenever the two
orders disagree. Read from source rather than memory,
`sklearn/model_selection/_split.py::StratifiedKFold._make_test_folds` in the
installed 1.9.0 does this:

    _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
    # y_inv encodes y according to lexicographic order. We invert y_idx to
    # map the classes so that they are encoded by order of appearance:
    _, class_perm = np.unique(y_idx, return_inverse=True)
    y_encoded = class_perm[y_inv]

scikit-learn encodes by order of appearance too. Measured on the constructed
cases, the two encodings give different folds in 6 of 16, and on every one of
those our assignment equals scikit-learn's exactly:

    ENCODING: unbalanced-3-class/3: ours matches ['first-seen']; the two encodings differ here
    ENCODING: unbalanced-3-class/3: the assignment equals scikit-learn's

(That arm runs only where scikit-learn is importable and says so when it is not.
The run above used `.pixi/envs/bench`.)

**What it can catch.** Any wrong fold ASSIGNMENT, including the one-row
rotation every invariant survives; an off-by-one in the residue arithmetic; an
`offset` that does not accumulate across classes; a KFold branch that
distributes the remainder to the wrong folds; folds that overlap, leave a row
out or are empty; a stratified branch that is not stratified; an assignment that
is not reproducible; a `split_descriptor` that stopped reading X or y.

**What it cannot catch.** A row order that is wrong for the caller's purpose.
`_default_folds` never reads X and cannot; the order is the caller's. The ORDER
arm records that hole rather than closing it. It also cannot decide the class
encoding question, only measure it.

### Seen to fail, four ways, and the first one is the point of the lane

**1. The lane's own env sabotage,
`MOJOLEARN_FOLD_ORDER_SABOTAGE=1 MOJOLEARN_HOST_ALLOW_SABOTAGE=1`** (the
row-to-fold assignment rotated by one).

| arm | ok | fail |
|---|---|---|
| PARTITION | 136 | **0** |
| REFERENCE | 0 | **20** |
| SWITCH | 4 | 0 |
| ORDER | 8 | 0 |
| ENCODING | 0 | 16 |

Every invariant the lane's cell tests passes. Every recomputation fails.

    FAIL  REFERENCE: lane-strat/2: the fold assignment agrees with the recomputation
            folds ours=2 ref=2
            fold 0: sizes ours=1024 ref=1024
              rows only in ours: [1021]
              rows only in ref : [0]
              first position 0: ours=1 ref=0

**2. Scratch sabotage: `first = (fold + offset) % n_splits`.** Caught by both
arms, PARTITION 2 failures and REFERENCE 11.

    FAIL  PARTITION: ragged-2-class/5: fold sizes differ by at most one (min 409, max 411)
    FAIL  REFERENCE: lane-strat/5: the fold assignment agrees with the recomputation
            fold 2: sizes ours=409 ref=410
              rows only in ref : [1237]

**3. Scratch sabotage: the class `offset` never accumulates
(`offset += 0`).** PARTITION 13 failures, REFERENCE 13.

    FAIL  PARTITION: lane-strat/5: fold sizes differ by at most one (min 409, max 411)
    FAIL  REFERENCE: lane-strat/5: the fold assignment agrees with the recomputation
            fold 0: sizes ours=411 ref=410
              rows only in ours: [433]

**4. Scratch sabotage: the KFold remainder goes to one fold too many
(`fold <= n % n_splits`).** PARTITION 4 failures, REFERENCE 4. This one leaves
the index range entirely:

    FAIL  PARTITION: lane-kfold/2: the test folds hold every row exactly once
            len ours=2049 ref=2048
            rows extra in ours (first 8): [2048]

`python/mojolearn/model_selection.py` restored from a byte copy after each,
back to `9c8052cbcdb8220b...`.

---

## 3. `gp-optimize` and `gp-optimize-restarts`

**Oracle shape: a property oracle with an independent likelihood underneath it**
(shape (b), with shape (a) for the objective rather than the answer),
`tools/gp_optimize_oracle_check.py`.

A value oracle is not available and should not be faked, because recomputing the answer
means writing a second bounded quasi-Newton optimizer, and two optimizers that
both converge need not converge to the same bits, so a disagreement would be
uninformative. What is checkable is OPTIMALITY, in four arms, on the lanes'
exact configurations and all nine fixtures.

* **OPTIMAL.** No probe within `LOCAL_RADIUS = 0.1` in log-hyperparameter space
  beats the answer's likelihood. The probes are per-coordinate steps, steps
  along the ASCENT DIRECTION given by the analytic gradient at the answer, and
  fixed-seed random directions. The gradient-informed probes are what make this
  arm able to fail, because wherever the projected gradient is not near zero, a small
  step along it raises the likelihood.
* **STATIONARY.** The projected gradient `max_i |clip(x_i - g_i) - x_i|` at the
  answer, stated twice, once from our analytic gradient and once from central
  differences of the float64 reference.
* **REFERENCE.** A float64 numpy restatement of
  `lml = -0.5 y^T (K + aI)^-1 y - 0.5 log|K + aI| - n/2 log 2pi`, with K built
  from the kernel formulas rather than from our Mojo. Our float32 likelihood
  and our analytic gradient are held to it AT THE RETURNED THETA.
* **CONTRACT.** One run per start, the reported likelihood is the best run's, a
  re-evaluation is bit for bit, the theta is inside the bounds and is exactly a
  float32, a second fit is identical, a different `random_state` moves the
  restarts, and the best-of-restarts selection is REACHED (below).

### The tolerances, and why each is derived rather than chosen

| name | value | where it comes from |
|---|---|---|
| the OPTIMAL sweep under our likelihood | `2 x` the float32 ULP of the likelihood | both sides are the SAME float32 function at two points, so there is no precision gap to absorb, only the granularity of the value itself. Two float32s one ULP apart are not a difference. |
| the OPTIMAL sweep under the reference | `2 x AGREE x (\|lml\| + 1)` | DERIVED: arm REFERENCE bounds the pointwise gap between the two objectives by `AGREE x (\|lml\| + 1)`, so a point optimal for one can be beaten under the other by at most twice that, once at the answer and once at the probe. |
| `AGREE = 1e-3` relative | reused, not restated | the bound `test_gp_optimizer.py` already uses for scikit-learn agreement, set once from the float32 precision argument. |
| `GRAD = 5e-3` relative | reused, not restated | the bound `test_gp_optimizer.py` already uses for its finite-difference arm, applied here at the RETURNED theta, which that file does not do. |
| `PGTOL = 1e-5` | the optimizer's own | applied only where the optimizer claims it, `stop == "pgtol"`. |
| `PG_DROP = 10` | stated | for any other stop reason. An optimizer that did not reduce the projected gradient by an order of magnitude did not optimize. |
| `FD_STEP = 1e-5` | stated | in float64 the difference quotient's noise floor is near 1e-9 at this step, so the step is not the limiting error. |

That the ULP bound is derived and not fitted is visible in the run, where the two
fixtures nearest the edge land at EXACTLY one ULP.

    OPTIMAL: gp-optimize/hashed: best probe 'ascent0.001' gains +1.525879e-05
             tolerance=3.052e-05 (2 x float32 ULP 1.526e-05), which is +1.00 ULP
    OPTIMAL: gp-optimize-restarts/odd: best probe 'coord5-0.1' gains +7.629395e-06
             tolerance=1.526e-05 (2 x float32 ULP 7.629e-06), which is +1.00 ULP

On both, the float64 reference says the probe is WORSE, so those two are float32
quantization and not a defect.

### Result: 268 checks pass, 2 are a known and recorded failure

**The float64 restatement agrees with ours closely, and this is the check
nothing in the tree was making at the returned point.**

    REFERENCE: gp-optimize/base: ours=-147.44848633 reference=-147.44852121
               relative=2.350e-07 bound=1e-03
    REFERENCE: gp-optimize/base: the analytic gradient at the RETURNED theta
               ours      =[ 5.321761e-04 -1.275342e-04  1.939113e-06]
               reference =[ 5.338492e-04 -1.275311e-04  1.965361e-06]
               worst relative=1.672e-06 bound=5e-03
    REFERENCE: gp-optimize-restarts/base: ours=-142.01058960 reference=-142.01062128
               relative=2.216e-07

### FINDING 1: on the `wide` fixture the answer is not a local maximum

`KNOWN_NON_STATIONARY` in the check file; `--strict` turns it into a failure.

On `wide` (columns scaled by `logspace(-4, 4)`) the optimizer stops on `ftol` at
a point whose gradient in the constant is still 1.8e-2, and a step of +0.1 there
RAISES the likelihood under both objectives:

    ---   gp-optimize/wide: theta=[4.150332e-03 6.833325e-04 1.151293e+01]
          lml=-13115.260742 runs=[(21, 24, 'ftol', -13115.2607421875)]
    REFERENCE: gp-optimize/wide: ours=-13115.26074219 reference=-13115.26050871
               gradient ours=[1.826879e-02 4.762721e-08 1.265601e+04]
                        ref =[1.826875e-02 0.000000e+00 1.265602e+04]
    known OPTIMAL: gp-optimize/wide: no probe within 0.1 of the answer beats our likelihood
            best probe 'random5x0.1' gains +4.882812e-03
            tolerance=1.953e-03 (2 x float32 ULP 9.766e-04), which is +5.00 ULP
    (and under the reference likelihood, 'coord0+0.1' gains +1.921217e-03)

Both likelihoods agree the probe is better, so this is not a float32 artifact.
`gp-optimize-restarts/wide` is the same, `coord0+0.1` gaining `+3.906250e-03`
ours and `+1.920790e-03` under the reference.

The far sweep, which is measured and not enforced, shows how far from the best
the point is. Re-optimizing from the best far probe:

    note OPTIMAL: gp-optimize/wide: the best FAR probe (33 of them) is 'bound0hi' at +26.466797
          re-optimizing from it reaches lml=-6793.433105, +6321.827637 against the
          lane's answer, theta=[ 11.512925 -11.512925  11.512925]

The recorded `gp-optimize/wide` cell is therefore a local optimum about 6,322
nats below what the same optimizer reaches from a different start inside the
same box, and `n_restarts_optimizer=2` on the restarts lane does not find the
better basin either. This is an optimizer-quality finding on a deliberately
badly scaled fixture, not a bitwise-identity finding, and it is recorded rather
than hidden by dropping the fixture. The check runs all nine fixtures by
default and prints this one as `known` with a pointer here.

### FINDING 2: the best-of-restarts selection is inert on every fixture

On all nine fixtures the `gp-optimize-restarts` lane runs, the FIRST start (the
kernel's own theta) wins, usually by tens of nats:

| fixture | run 0 | run 1 | run 2 |
|---|---|---|---|
| base | **-142.010590** | -147.449677 | -147.449677 |
| ties | **-161.331787** | -212.222504 | -212.237610 |
| hashed | **-149.994141** | -164.298401 | -164.298370 |
| wide | **-13115.259766** | -13115.278320 | -13115.282227 |
| denormal | **-139.643402** | -144.168304 | -144.168335 |
| denormal_ftz | **-139.643402** | -144.168304 | -144.168335 |
| dupes | **-134.807220** | -155.282623 | -155.282593 |
| odd | **-118.152588** | -138.940231 | -138.940247 |
| negative | **-116.762970** | -202.060974 | -202.060852 |

So the `best is None or f < best[1]` selection is never exercised by the lane,
and a defect in it would be invisible to every column of every record. Measured,
not inferred. A scratch sabotage that keeps the FIRST run instead of the best
changed nothing at all on `base`, `ties` and `negative` until this was fixed,
and the check reported `SABOTAGE NOT CAUGHT`.

The fix is in the check, not in the lane. Arm CONTRACT now also fits the same
kernel from a deliberately poor start (`1e-4` everywhere) with four restarts and
`random_state=11`, where a restart does win, and holds the selection to it:

    CONTRACT: gp-optimize-restarts/negative: from a poor start a RESTART wins,
              so the best-of selection is reached (winner is run 3)
        runs=[(23, 34, 'ftol', -202.06088256835938), (13, 40, 'ftol', -202.06085205078125),
              (35, 71, 'line-search', -116.76298522949219),
              (24, 85, 'line-search', -116.76295471191406), (10, 22, 'ftol', -202.06088256835938)]

### What these oracles can and cannot catch

**Can.** An optimizer that returns its starting point or any non-stationary
point; a descent direction with the wrong sign; a line search that accepts a
step that raises the objective; an active-set rule that clips to the wrong
bound; a best-of-restarts that reports a likelihood no run achieved or returns a
theta from a different run; a `random_state` that is ignored; a likelihood or an
analytic gradient that disagrees with the definition, including a wrong Matern
nu = 5/2 polynomial, a missing ridge, a missing `-n/2 log 2pi`, or a logdet with
the wrong factor of two; a fit that is not reproducible.

**Cannot.** A local maximum that is not the global one; the probes are local and
finite, which is exactly the hole FINDING 1 sits in. Anything about speed. THE
BITS, because optimality is a numerical property, not a bitwise one, and two columns
that both pass every arm here can still disagree bit for bit. This oracle and
the cross-column diff answer different questions and neither replaces the other.
Kernel structures outside the two the lanes use; the float64 reference refuses
anything else by name.

One thing worth recording could NOT be made expressible. The profile ridge
`alpha = 2^-20` cannot be moved to a value large enough for the REFERENCE arm to
see, because `gpr_fit_host` under `NUMERIC_IDENTICAL` refuses any unpinned alpha
and the only other pinned value is 0, which is numerically negligible beside a
white-noise term of 0.1. The ridge is fenced by the numeric profile rather than
by this oracle, and saying so is better than running an arm that cannot fail.

### Seen to fail, five ways

Every arm has been seen to fail on a sabotage aimed at it. All scratch edits
restored from byte copies; `_gp_optimizer.py` back to `a2299fdb5ae2cd51...`,
`_gp_impl.py` back to `712bdcd1c6869f62...`.

**1. `minimize` returns its starting point.** STATIONARY 12 failures, OPTIMAL
12.

    FAIL  STATIONARY: gp-optimize/base: the ours projected gradient fell by at least 10x
            projected gradient at the answer (ours) = 1.382e+01
            at the starting theta (ours)            = 1.382e+01
    FAIL  OPTIMAL: gp-optimize/base: no probe within 0.1 of the answer beats our likelihood
            best probe 'ascent0.1' gains +6.926007e+01
            which is +1134757.00 ULP
    FAIL  OPTIMAL: gp-optimize/base: no probe within 0.1 beats the REFERENCE likelihood
            best probe 'ascent0.1' gains +6.925998e+01

The gradient-informed probe is what catches it, which is why those probes are
there.

**2. The line search drops the Armijo condition.** STATIONARY 8, OPTIMAL 10,
CONTRACT 1.

    FAIL  OPTIMAL: gp-optimize/base: best probe 'coord2-0.01' gains +6.103516e-05
            ours=-147.44854736 tolerance=3.052e-05, which is +4.00 ULP

Four ULP, not one, so the ULP tolerance separates this from the quantization cases
above.

**3. The best-of-restarts keeps the FIRST run.** CONTRACT 6 failures, all of
them in the poor-start sub-arm added for FINDING 2. Before that sub-arm existed
this sabotage was NOT CAUGHT at all.

    FAIL  CONTRACT: gp-optimize-restarts/base: and the reported likelihood is that winning run's
            ours  = -147.44967651367188
            other = -145.5089111328125

**4. `random_state` is ignored by the restarts (`seed = 7`).** CONTRACT 3
failures.

    FAIL  CONTRACT: gp-optimize-restarts/base: a different random_state moves the restarts
            runs(state=7)         =[(27, 51, 'ftol', -142.01058959960938), (7, 24, 'ftol', -147.44967651367188), (21, 30, 'ftol', -147.44967651367188)]
            runs(state=2**40 + 7) =[(27, 51, 'ftol', -142.01058959960938), (7, 24, 'ftol', -147.44967651367188), (21, 30, 'ftol', -147.44967651367188)]

**5. The likelihood loses its `-n/2 log 2pi` term, in both producers.**
REFERENCE 6 failures, CONTRACT 12.

    FAIL  REFERENCE: gp-optimize/base: the likelihood agrees with the float64 definition
            ours=-88.63642020 reference=-147.44852121 relative=3.962e-01 bound=1e-03

`0.5 x 64 x log(2pi) = 58.81`, which is the gap exactly.

---

## 4. How each check was kept from being one that cannot fail

The audit of 2026-09-16 caught eight checks in one day that could only pass, so
each of these carries its own expressibility arm rather than trusting that a
sabotage reaches it.

| check | the arm that asks whether the defect is expressible here |
|---|---|
| bpe-trainer | `TIES` refuses when the lane's corpus produces zero ties, and separately measures that reversing the tie-break moves the reference's own vocabulary. The lane's corpus reaches 42 ties. |
| cross-val-folds | `SWITCH`, four switches that must flip, including one hole held open on purpose so that a branch which starts reading something new is caught. |
| gp-optimize | `OPTIMAL` requires the local sweep to reach points the answer beats, so a flat likelihood is reported rather than passed. The poor-start sub-arm exists because the lane's own configuration made the best-of selection inert. |
| all three | the reference NEVER reads the sabotage switch. The BPE reference always follows the stated tie-break, so running under `MOJOLEARN_BPE_TRAINER_SABOTAGE=1` produces a disagreement rather than two sides agreeing on the wrong answer. |

And each `--sabotage-expected` run REQUIRES a disagreement and exits nonzero on
agreement, which is how FINDING 2 was found: the sabotage reported
`SABOTAGE NOT CAUGHT` instead of passing quietly.

---

## 5. What is owed

1. `tools/lane_applicability.py` should derive the CPU route from the host
   family's `routes` field, not from `covered_lanes()`. That moves
   `gp-optimize` and `gp-optimize-restarts` out of `gpu-only` and out of
   `recorded-only`, and the audit's "four record lanes have no oracle in any
   combination" becomes two. Not done here; this lane touched nothing that
   ships or that the audit owns.
2. FINDING 1 is a decision for someone else: either the optimizer should not
   stop on `ftol` while a coordinate still has an order-of-magnitude larger
   gradient than `PGTOL`, or the `wide` fixture's recorded cell should be
   understood as a local optimum. `KNOWN_NON_STATIONARY` records it either way.
3. FINDING 2 suggests the `gp-optimize-restarts` lane would carry more if its
   starting kernel were one the restarts can beat. That is a lane and fixture
   change, which this lane is not allowed to make.
