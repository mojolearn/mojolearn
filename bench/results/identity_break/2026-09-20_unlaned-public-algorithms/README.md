# The six public algorithms with no identity lane at all

lane/unlaned-public-algorithms, 2026-09-20, Apple M4, ONE CORE, `nice -n 19`,
CPU host bindings only (no Metal job ran; `vendor` is `cpu` in every column
here). Columns are `tools/identity_break.py --repeats 2`, every one recorded
at this lane's own commit (`commit` reads the same twelve characters in all
three files; `$(git log --oneline -1)` at the time of recording).

`--repeats 2` AND NOT 1, ON PURPOSE. `stable_digest()` refuses a part with
fewer than two repeats, so a sabotage cell reading STABLE beside a plainly
different hash falls through and the move is thrown away. Every column here
carries two fits per cell.

`tools/verification_matrix.py --json` reported 244 public algorithms and SIX
with no identity lane of any kind:

| public entry | lane written here | CPU column |
| --- | --- | --- |
| `parallel_gaussian_process.fit_gaussian_process_classifier` | `par-gpc-fit` | **yes** |
| `parallel_gaussian_process.predict_gaussian_process_classifier` | `par-gpc-predict` | **yes** |
| `mamba.Mamba1DecodeSession` | `mamba1-decode-session` | no, and none can exist |
| `transformer.TransformerDecodeSession` | `transformer-decode-session` | no, and none can exist |
| `models.ParallelCausalLM` | `par-causal-lm` | no, and not on Apple either |
| `parallel_model_selection.cross_val_score` | `par-cross-val` | no, and not on Apple either |

All six now have a lane, parts and a written-down owed column. TWO of the six
are closed on this column. The other four are not, and the reason is the same
in each case and is not a scheduling choice: their public surface REFUSES BY
NAME on a CPU-only install, so there is no arithmetic of theirs for a CPU
column to hash and no host define for a negative control to be wrong inside.

## The columns

| file | host set | what it is |
| --- | --- | --- |
| `clean.json` | clean rebuild of this branch's gp host source | 18 cells (2 lanes x 9 fixtures), every one STABLE; infer, model and batch STABLE on all 18; batchgrad, batchscale, ragged and stepfull declared `n/a` |
| `sabotage-gp.json` | the SAME source built `-D MOJOLEARN_HOST_SABOTAGE=1` | every cell moved: `--diff` reads DIVERGENT=18 train, 36 infer/model, 18 batch |
| `gpu-only-refusals-probe.json` | the clean host set | the four lanes with no CPU route, run anyway: four REFUSED cells, each naming its own class. Refused by `admit()` by name, and recorded so the refusal text is a committed fact rather than a claim. `.errors.txt` beside it holds the four messages verbatim |

`admit()`, printed rather than assumed:

    clean.json                    complete=True  cells=18  par_devices=0
      admit(default)  = None            <- ADMISSIBLE
      admit(par_axis) = None
    sabotage-gp.json              admit = "sabotage, partial, probe, unfixed or smoke run (by name)"
    gpu-only-refusals-probe.json  admit = "sabotage, partial, probe, unfixed or smoke run (by name)"

## What had to change for the two that closed

`fit_gaussian_process_classifier` and `predict_gaussian_process_classifier`
refused on this box with

    NotImplementedError: no CPU implementation of the parallel worker
    operation gpc_class_fit yet

and nothing else was wrong. Their partition is one-vs-rest CLASSES, cut in
the driver's own Python, sent one request per class and merged back in class
order by the driver's own Python (`_set_fitted`; the binary `[1 - v, v]`
pair, the row-wise normalization and the arg-max in
`predict_gaussian_process_classifier` itself). Each shard's own work is
`GaussianProcessClassifier._fit_binary` / `._latent`, which are the single
`gpc_fit` and `gpc_predict` entries the gp family's host binding exports, and
the host kernel matrix under them says in its own docstring that it is the
one-device path with `MOJOLEARN_GP_DEVICE_COUNT` unset. That is exactly the
bar `_parallel_pool.CPU_OPERATIONS` states, so the two operations were
admitted to it. The ROW-sharded GP driver (`par-gp`'s `gp_fit`) is still
deliberately absent from that set, for the reason written there.

### Admitting `gpc_class_fit` opened a hole, and the same change closes it

`CPU_OPERATIONS`'s docstring promises that outside `reference_training()`
"the worker's fit refuses exactly as a plain CPU fit does". Every other
admitted fit operation gets that for free because it runs its estimator's
PUBLIC `fit` and inherits `_mode._guard_cpu_training`. The GPC shard calls
`_fit_binary`, which sits BELOW that decorator. MEASURED on this CPU-only
install before the guard was added, in one process on one data set:

    plain fit     NotImplementedError: mojolearn: public CPU estimators support
                  inference from saved models; fit/training is reserved for the
                  internal bitwise verifier ...
    parallel fit  NO REFUSAL: the driver trained on a CPU-only install

`_parallel_worker.execute` now calls `require_training(state)` before the
shard fit, so both refuse with the same words.
`test_cpu_training_par_classical.py::test_the_gpc_class_fit_worker_refuses_outside_the_verifier`
asserts it, and was WATCHED TO FAIL with the guard removed (the assertion
message it printed is the one quoted above) before being put back.

## The sabotage was seen to move

The arm is the gp family's own define on `bindings/_mojolearn_gp_host.mojo`,
built from THIS branch's source, against a CLEAN build of the same source, so
the pair differs in one define and nothing else. A .so digest does not prove a
define, so both were interrogated: `gp_host_sabotage()` reads `False` on the
clean binding, and the loader refuses the other BY NAME
(`is a SABOTAGE build and computes wrong answers on purpose`).

    par-gpc-fit      base  53f3a09fc600d61d -> 59b4a6c2ea6454a4
                     L     cf8752e79a59c592 -> 07c415bd8bfe2ff3
                     pi    3e4e62ab648dd40a -> c2d8c272fe84b3a4
    par-gpc-predict  base  2f1217bcae26e6a9 -> 2e25f01e28255189
                     proba (moves on all nine fixtures)

`tools/identity_break.py --diff clean.json sabotage-gp.json`:

    summary:                DIVERGENT=18
    summary (infer/model):  DIVERGENT=36
    summary (batch):        DIVERGENT=18

## Parts that do NOT move, and why each is kept

| lane | part | why no build arm reaches it |
| --- | --- | --- |
| both | `flags` | type, method, length and class-count refusals; a predicate, not arithmetic. Hashed as a number so a LIFTED refusal reads DIVERGENT, which a raise would not |
| `par-gpc-fit` | `lml` on `hashed`, `odd`, `ties` | 3 of 9 fixtures hold while `L` and `pi` move on all nine. A lane whose only float part were the log marginal likelihood would have read clean on a third of this column |
| `par-gpc-predict` | `labels` 9/9, `three_labels` 8/9 | `predict` is a sign test and an arg-max; the arm perturbs the latent mean far below the gap between the top two classes. This is why `proba` is hashed beside them |

## What the two closed cells prove, and what they do not

Both lanes hold the sharded driver to the plain `GaussianProcessClassifier`
fit BYTE FOR BYTE inside the cell, so they fail on one box with no record.
That is AGREEMENT, not rightness: if `gpc_fit`'s Newton loop were wrong, the
driver and the plain fit would be wrong together and every column would read
IDENTICAL. The DEVICE axis is not tested here either -- at one device the
partition is one worker process per class, which is what
`identity_break._par_devices`'s docstring says of every `par-*` lane. The
two-device column is owed and is the only column on which the drivers' actual
claim is stateable.

## The four that did not close, and the exact refusals

Run anyway, recorded in `gpu-only-refusals-probe.json`:

    mamba1-decode-session       NotImplementedError: mojolearn Mamba1DecodeSession: the loaded
                                Mamba1Block binding exports no resident decode session (the CPU
                                host route and older GPU builds); the per-call step() is the path here
    transformer-decode-session  NotImplementedError: mojolearn TransformerDecodeSession: ... (the same,
                                naming its own class)
    par-causal-lm               NotImplementedError: ParallelCausalLM requires CUDA or HIP device isolation
    par-cross-val               NotImplementedError: parallel cross-validation requires CUDA or HIP GPU workers

`mamba1_session_create` and `transformer_decode_session_create` exist only in
`bindings/_mojolearn_mamba.mojo` and `bindings/_mojolearn_transformer.mojo`.
`ParallelCausalLM` and `parallel_model_selection.cross_val_score` check
`_backend.vendor() not in ('cuda', 'hip')` before they construct a layer or
prepare a fold, so an APPLE column cannot state their proposition either: the
M4 is not a smaller version of the right box, it is the wrong vendor.

Four REFUSED cells in a full column would read exactly like coverage, so
`identity_break.GPU_ONLY_LANES` drops these four from a full-column run on a
CPU-only install and prints the reason per lane. `--lanes` is never filtered,
which is how the probe above was taken, and a GPU column runs them like any
other lane. `lane_applicability` derives the same fact independently, from
`host_surface` and the lane bodies, and the probe column records it:

    degenerate_lanes = ['mamba1-decode-session', 'par-causal-lm',
                        'par-cross-val', 'transformer-decode-session']
    degenerate_column = 'cpu-host'

## Owed, exactly

No GPU work and no rental was done in this pass. The commands, to be taken in
ONE coordinated three-column record and not piecemeal:

    # Apple / NVIDIA / AMD, one device: the two decode sessions
    MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
      --lanes mamba1-decode-session,transformer-decode-session \
      --repeats 2 --step-full --batch-scale --ragged --batch-grad \
      --json <column>.json

    # Apple / NVIDIA / AMD, one device: the GPC drivers' GPU column
    MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \
      --lanes par-gpc-fit,par-gpc-predict \
      --repeats 2 --step-full --batch-scale --ragged --batch-grad \
      --json <column>.json

    # NVIDIA or AMD, TWO devices: the only column on which the four par-*
    # lanes' claim is stateable. par-causal-lm and par-cross-val have no
    # other column at all.
    MOJOLEARN_PAR_DEVICES=0,1 MOJOLEARN_NUMERIC_MODE=identical python3 \
      tools/identity_break.py \
      --lanes par-gpc-fit,par-gpc-predict,par-causal-lm,par-cross-val \
      --repeats 2 --step-full --batch-scale --ragged --batch-grad \
      --json <column>.2gpu.json

Read the two-device column through `admit(..., par_axis=True)` and credit only
`par-*` lanes from it.
