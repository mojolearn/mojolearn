# lane/sabotage-sweep

Branch `lane/sabotage-sweep`, from main at 372609f8e. Worktree
`~/mojolearn-wt/sabotage-sweep`. Evidence
`bench/results/identity_break/2026-09-17_sabotage-sweep/`.

## The one-line answer

`docs/VERIFICATION_MATRIX.md` said 54 of 229 lanes had a sabotage SEEN to move
a build. The other 175 were not 175 broken arms; 135 of them were one rule and
one flag. Seven CPU pods and $0.478 later the count is **189 seen(build), 3
declared, 37 none**, and **180 lanes carry all four kinds** where 49 did.

| | before | after |
|---|---|---|
| seen(build) | 54 | 189 |
| declared | 136 | 3 |
| none | 39 | 37 |
| all four kinds | 49 | 180 |

## Why the 175 were not 175 broken arms

`tools/verification_matrix.py::negative_control_moves` asks
`stable_digest()` of the sabotage cell, and `stable_digest` refuses a part
recorded with fewer than two repeats ("refusal/N/A/one repeat is not
evidence"). **Every committed sabotage column in the tree was taken at
`--repeats 1`.** The CPU identity gate does it deliberately -- "negative
controls test detection, while production retains two stability repeats",
`.github/workflows/cpu-identity-gate.yml` -- and every manual lane copied the
gate. A sabotage cell that read STABLE with a hash plainly different from the
clean one therefore fell through to the `MOVED`/`DIVERGENT` fallback, which a
STABLE verdict does not satisfy, and the move was discarded in silence.

Measured against the largest such record, `2026-09-16_sabotage-audit`: **0 of
its 11 lanes count**, all eleven refused for the one repeat, with every
sabotage cell STABLE and different. About seventy committed sabotage columns
are in that state. They are one re-run away each.

The rule is not the problem and was not changed. Two repeats on the control
also proves something one repeat cannot: that the sabotaged build is itself
deterministic, so a moved cell is the arm and not noise.

## What was run

Seven RunPod CPU pods (`runpod/base:1.3.1-ubuntu2204`, x86-64 EPYC,
`MOJOLEARN_NUMERIC_MODE=identical`), each building its host families twice from
one tree -- production into `python/mojolearn/host/`, the control into
`python/mojolearn/host-sabotage/` -- then running the shard's lanes twice,
`--repeats 2` on both arms, `--fixtures base,ties` except where noted. Every
column records the loaded binding's sha256 and its own `<prefix>_sabotage()`
read back from INSIDE the process. `arm_bytes_differ.txt` shows no built pair
shares bytes. 135 of 138 lanes move.

## The three that did not, one finding each

1. **`kmeans-cosine` is structurally unsabotageable, and that is correct.**
   Its cell is the hash of a REFUSAL SENTENCE (`metric='cosine'` is refused by
   name in `cluster/impl/kmeans_params.mojo::validate`); no arithmetic runs, so
   no build can perturb it. A column that HASHES here means the refusal was
   lifted. No arm should be written.
2. **`par-scaler`'s arm fires and the cell cannot show it.** Under the
   preprocessing sabotage build the lane raises its own `ValueError:
   transform_scaler and plain transform differ: 2847 bytes of 16384`, so the
   cell reads REFUSED, and a refusal is not a catch. The repair is to HASH the
   disagreement instead of raising; that changes the lane's cell and every
   committed record of it, so it is OWED to a lane that owns `LANE_REVISIONS`,
   not taken here.
3. **`ordered-gradient-sum` had no negative control at all** -- INERT on ALL
   NINE fixtures, both repeats, same `gradients` hash in both arms. Fixed on
   this branch; see below.

## The define this lane had to write

`SAMBA_ACCUMULATE_HOST_SABOTAGE`, in `training/host/samba_ops_oracle.mojo`,
under the training family's own `MOJOLEARN_HOST_SABOTAGE` (the define
`host_surface.py` declares and `training_host_sabotage()` reads back, so a
build carrying it says so and `_backend` refuses it outside the gate). It steps
every returned cell of `host_samba_accumulate` by one unit in the last place
through `gemm_oracle_sabotage_value_flip`. Production compiles nothing below
the `comptime if`.

The file previously said the accumulate needed no arm because "a pairwise add
has no fold order to move (an addition commutes), so a sabotage arm there would
be inert, and the lanes that reach them also reach a GEMM ... which move". The
first half argues against an ORDER arm and is right -- it is the argument the
shared GEMM leaf arm lost on `ties` in `SABOTAGE_AUDIT_2026-09-16.md`, whose
remedy was a value flip. The second half is false: `ordered-gradient-sum`
reaches `ordered_sum_gradients` -> `accumulate_grads` -> `accumulate` ->
`host_samba_accumulate` and nothing else, with no GEMM anywhere on the path.

The unfixed side is recorded before the fix, in
`g-ordered-gradient-sum/` (IDENTICAL=9), so the arm is watched failing to fail.

## Two lanes that have no build, and what their control is

`bpe-trainer` and `cross-val-folds` read `none` in the matrix because they have
no host family, and they have no host family because NEITHER HAS A BUILD: both
run an independent Python implementation (`mojolearn/_bpe_trainer.py`,
`model_selection._default_folds`). Their negative control is an implementation
env switch inside the library -- `MOJOLEARN_BPE_TRAINER_SABOTAGE` and
`MOJOLEARN_FOLD_ORDER_SABOTAGE` -- which is not the harness and not a compiled
define. Both move, and `cpu-x86.replay.json` is the control for the control: a
second clean run with the switches off, IDENTICAL=4 against the first.

The matrix's vocabulary has two words, `build` and `harness`, and records these
as `build`. That is the right side of the distinction it is drawing (the
implementation was perturbed, not the probe) and the wrong word for a lane that
compiles nothing. Naming it here rather than widening the tool.

A measured claim was corrected while doing it. The `bpe-trainer` lane said its
tie-break arm "leaves `n_tokens` and `n_merges` alone -- the vocabulary still
fills to vocab_size". Read on the box: `base` 302 tokens / 46 merges / 42 ties
broken clean against 301 / 45 / 41 sabotaged, `ties` 309/53/33 against
308/52/31. The vocabulary never fills to 320 on that corpus, because
`min_frequency=2` runs out of pairs first.

## What is still owed, and by whom

**37 `par-*` lanes read `none`, and this lane deliberately did not chase
them.** They are the multi-GPU drivers; a driver lane's arm has to move on a
multi-device box, which is release work. Two things about them are worth
recording rather than leaving to the next reader:

- Several DO have a sabotage define in Mojo that the matrix cannot see, because
  the matrix reads defines only from `host_surface.FAMILIES`:
  `MOJOLEARN_CHOLESKY_PARALLEL_SABOTAGE` (`cholesky/multi_gpu.mojo`),
  `MOJOLEARN_GMM_PARALLEL_SABOTAGE` (`mixture/multi_gpu.mojo`),
  `MOJOLEARN_RESAMPLE_PARALLEL_SABOTAGE` (`resample/estimator.mojo`) and
  `MOJOLEARN_HIERARCHY_PARALLEL_SABOTAGE`
  (`hierarchy/impl/cluster/detail/multi_gpu.mojo`). So `none` overstates the
  gap for `par-cholesky`, `par-gmm`, `par-resample` and
  `par-graph-agglomerative`, which have an arm nobody has watched.
- The eleven `par-*` lanes that DO have a CPU verifier (`par-arima`,
  `par-forest`, `par-forest-et`, `par-holtwinters`, `par-mlp`,
  `par-queries-kde`, `par-queries-knn`, `par-queries-radius`,
  `par-reference-knn`, `par-reference-knn-reg`) all moved here; only
  `par-scaler` did not, for the reason in section 2 above.

**The CPU identity gate still throws away exactly this evidence on every run.**
It builds the whole host set a second time with the manifest's defines, runs
every covered lane under it, requires the four-column diff to fail, and uploads
the result as an artifact that expires. That was `lane/sabotage-evidence`'s
finding on 2026-09-16 and it still stands; what is new is that the artifact
would not have counted anyway, at one repeat.

**`gbdt-categorical-ctr-tables` and `gbdt-tensor-ctr-tables` cannot run from a
plain checkout of the tracked tree alone.** Their CTR fixtures live under
`bench/results/identity_break/2026-09-15_gbdt-ctr-tables/models`, which every
runner that excludes `bench/results` leaves at home. `f-recovery` passed
`--include` for them. Anything that ships the source without that path will see
both lanes REFUSE, and a REFUSED cell in a sabotage column reads as a catch.

---

# CLOSING SECTION, written after the session that ran this lane was killed

**Nothing was lost.** Every commit this lane made was already merged into
`origin/main` before the session ended: the seven evidence shards under
`bench/results/identity_break/2026-09-17_sabotage-sweep/`, the regenerated
`docs/VERIFICATION_MATRIX.md`, the `bpe-trainer` docstring correction and this
file. The branch `lane/sabotage-sweep` is a fast-forward of `origin/main` and
carries no unmerged work. A session resuming here should read the numbers below
rather than the ones above, which were current at 475b3dbd6.

## Where the count stands on main now

| | before this lane | after it | on main today |
|---|---|---|---|
| seen(build) | 54 | 189 | **190** |
| declared | 136 | 3 | 6 |
| none | 39 | 37 | 33 |
| all four kinds | 49 | 180 | see the matrix |

The 39 lanes still not seen are `kmeans-cosine` plus 38 `par-*` multi-GPU
drivers. `kmeans-cosine` is not a gap: its cell is the hash of a REFUSAL
SENTENCE (`metric='cosine'` is refused by name in
`cluster/impl/kmeans_params.mojo::validate`), no arithmetic runs, and a column
that HASHES there means the refusal was lifted. **No arm should be written for
it.** The `par-*` lanes need a multi-device box and are release work; four of
them (`par-cholesky`, `par-gmm`, `par-resample`, `par-graph-agglomerative`)
already have a Mojo define the matrix cannot see, because the matrix reads
defines only from `host_surface.FAMILIES`, so `none` overstates their gap.

## The one thing this lane was mid-way through, and how it ended

The last open question was `ordered-gradient-sum`. This lane measured it INERT
on all nine fixtures, wrote `SAMBA_ACCUMULATE_HOST_SABOTAGE` in
`training/host/samba_ops_oracle.mojo` to give it an arm, and then found that
the arm fires but the LANE converts the catch into a refusal: it compares
`ordered_sum_gradients` against an independent NumPy left fold and raised
`AssertionError`, so the cell read REFUSED on all nine fixtures. That mattered
twice -- a refusal is not a catch, and
`tools/cpu_identity_gate_check.py column` requires every covered lane to read
STABLE in the sabotage column, so shipping the arm on the family define would
have turned a green CPU identity gate red. The lane re-gated it onto its own
`MOJOLEARN_ACCUMULATE_SABOTAGE`, set by no build script and no gate, and named
the real repair as owed: hash the comparison instead of raising it.

**`lane/cpu-ordered-sum-completion` then did exactly that repair, and did it
better.** On main today the lane raises `NumericalMismatch(..., parts)`, an
exception that CARRIES the hashed parts, so the harness records a DIVERGENT
cell with stable hashes instead of a REFUSED one, and the arm is back on the
family define (`comptime if GEMM_ORACLE_HOST_SABOTAGE: cur[0] = 0.0`).
`bench/results/identity_break/2026-09-17_cpu-ordered-sum-completion/cpu-sabotage.json`
reads DIVERGENT with two stable repeats on all nine fixtures, and the matrix
counts it.

**Do not reapply `MOJOLEARN_ACCUMULATE_SABOTAGE`.** It is superseded. The pod
that was queued to prove the own define left the gate alone is moot and was
never needed.

## The remaining piece of that same shape

`par-scaler` is still `declared` for the identical reason
`ordered-gradient-sum` was: under the preprocessing family's sabotage build the
lane raises its own `ValueError: transform_scaler and plain transform differ:
2847 bytes of 16384`, so the cell reads REFUSED while the clean cell reads
STABLE. The arm works; the lane swallows it. **The remedy now exists in the
tree**: `NumericalMismatch` in `tools/identity_break.py`. Converting
`par-scaler`'s raise to it is a small, cheap change that does not touch the
clean cell, and it is the single highest-value item left that does not need a
multi-GPU box. `par-scaler-minmax` should be checked for the same shape.

## The other things a session with no context should not rediscover

- **Every committed sabotage column before 2026-09-17 was taken at
  `--repeats 1`, and the matrix discards all of them.**
  `verification_matrix.py::negative_control_moves` asks `stable_digest()` of
  the sabotage cell, which refuses a part with fewer than two repeats. The CPU
  identity gate does one repeat deliberately and every manual lane copied it.
  That single flag, not 175 broken arms, is what the "not seen" column was
  measuring. About seventy columns in the tree are one re-run away each.
- **The CPU identity gate still manufactures this evidence on every run and
  throws it away** (`lane/sabotage-evidence`'s finding, 2026-09-16). It builds
  the host set a second time with the manifest's defines, runs every covered
  lane under it, requires the four-column diff to fail, and uploads the result
  as an expiring artifact. At one repeat it would not have counted anyway.
- **FIVE FLIPS HAVE NO SWITCH TO FLIP** and cannot be sabotaged as they stand:
  DEVIATIONS 2620, 2621, 2622 (the OLS and min-norm rank guards), 2671 and 2672
  (the Jacobi phase merge and the k-means host staging), and the KDE ones 2625,
  2626 and 2660. Their old code was DELETED rather than gated, so there is zero
  `is_defined` to build against. Note them; do not fake an arm for them.
- **`gbdt-categorical-ctr-tables` and `gbdt-tensor-ctr-tables` cannot run from
  the tracked tree alone.** Their CTR fixtures live under
  `bench/results/identity_break/2026-09-15_gbdt-ctr-tables/models`, which every
  runner that excludes `bench/results` leaves at home. Pass `--include` for
  them, or both lanes REFUSE -- and a REFUSED cell in a sabotage column reads
  as a catch.
- **Shard your pods by host family, and build every family a lane routes to.**
  Five lanes in this sweep refused only because the shard's build lacked the
  family they route to (`metrics` for `cross-val`,
  `gbdt-adapter-score-weighted` and `rf-score-weighted`; `preprocessing` for
  `gp-normalize-y` and `gp-sample-y-normalize`). One recovery pod fixed all of
  them for $0.0225.

## Cost

Seven RunPod CPU pods, every DELETE verified, **$0.478 total** for 135 of 138
lanes. The eighth (`sabsweep-h`) was queued and is moot; it never ran.
