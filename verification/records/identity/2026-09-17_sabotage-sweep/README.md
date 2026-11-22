# The sabotage sweep: negative controls watched failing, at two repeats (2026-09-17)

Branch `lane/sabotage-sweep`, from main at 372609f8e.

## Why 175 lanes read "not seen", and why it was almost never the arm

`docs/VERIFICATION_MATRIX.md` counted 54 lanes of 229 whose sabotage had been
SEEN to move a build. The other 175 were not 175 broken arms. They were one
rule, applied to columns that could never satisfy it.

`tools/verification_matrix.py::negative_control_moves` asks
`stable_digest()` of the sabotage cell, and `stable_digest` refuses a part
with fewer than two repeats: "a repeated, stable digest; refusal/N/A/one
repeat is not evidence". Every committed sabotage column in the tree was
recorded at `--repeats 1`. The CPU identity gate does it on purpose
(`.github/workflows/cpu-identity-gate.yml`: "negative controls test
detection, while production retains two stability repeats"), and every manual
lane copied the gate. So a sabotage cell that read STABLE with a hash that
plainly differed from the clean one fell through `stable_digest` to the
`MOVED/DIVERGENT` fallback, which a STABLE verdict does not satisfy, and the
move was discarded.

Measured, against the largest such record
(`bench/results/identity_break/2026-09-16_sabotage-audit/`, 11 lanes, both
arms present, every sabotage cell STABLE and different): **0 of 11 lanes
counted**, all eleven for `sab-unstable(STABLE)`, which is the one-repeat
refusal. Seventy-odd committed sabotage columns are in that state.

This directory is the same arms, run again with `--repeats 2` on both sides.
Nothing about the arms changed. What changed is that the negative control is
now recorded at the standard the matrix asks for, and it also proves something
the one-repeat records could not: that the sabotaged build is itself
deterministic, so a moved cell is the arm and not noise.

## How each pair was taken

One RunPod CPU pod per shard, `runpod/base:1.3.1-ubuntu2204`, x86-64 EPYC,
`MOJOLEARN_NUMERIC_MODE=identical`, through `tools/runpod_cpu_leg.sh`. Per
shard the host families were built TWICE from the same tree in the same run:
production into `python/mojolearn/host/`, and the negative control into
`python/mojolearn/host-sabotage/` with
`-D MOJOLEARN_HOST_SABOTAGE=1 -D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1 -D MOJOLEARN_GBDT_CTR_HOST_SABOTAGE=1`
(`host_surface.sabotage_build_defines`'s set for the families here). Then:

    python3 tools/identity_break.py --lanes <shard> --fixtures base,ties --repeats 2 --json cpu-x86.json
    env MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
        python3 tools/identity_break.py --lanes <shard> --fixtures base,ties --repeats 2 --json cpu-x86.sabotage.json

The sabotage column names the binding it actually loaded and that binding's
own `<prefix>_sabotage()` read back from INSIDE the process
(`host.families[*].sha256` and `.sabotage` in each JSON), so neither column
rests on a file digest taken at queue time. `so_sha256.txt` and
`arm_bytes_differ.txt` record that the two sets are different binaries;
`missing_bindings.txt` is empty, because a binding absent from the sabotage
set would load nothing and read REFUSED, which a diff would miscount as a
catch.

`base` and `ties` are both run on purpose. `ties` is integer valued, and an
order-permutation arm cannot move an exact sum: that is how the shared GEMM
leaf arm was found inert in the 2026-09-16 audit. A lane that moves on `base`
and holds still on `ties` is reported here as exactly that.

## The shards

| shard | lanes | host families built (production and control) | pod | spend |
|---|---|---|---|---|
| `a-linear-neighbors` | 52 | core, estimators, preprocessing, solver, linalg, metrics | 16 vCPU | $0.0712 |
| `b-trees-gbdt` | 35 | core, estimators, linalg, gbdt, forest, rf, trees, svm | 16 vCPU | $0.1223 |
| `c-classical` | 25 | core, estimators, linalg, gp, gp_infer, metrics, arima, tsa, forecast, resample, mixture, mixture_infer, hdbscan, kernel_methods | 16 vCPU | $0.0991 |
| `d-neural` | 24 | core, estimators, linalg, training, mamba, transformer, byte_lm, neural, embedding | 16 vCPU | $0.1473 |
| `e-python-lanes` | 2 | core, gbdt | 8 vCPU | $0.0159 |
| `f-recovery` | 7 | the a, b and c families together, plus the CTR fixtures under `--include` | 16 vCPU | $0.0225 |
| `g-ordered-gradient-sum` | 1, all nine fixtures | (shipped with f) | | |

Seven pods, every DELETE verified (HTTP 204, then GET 404, then absent from
the listing), **$0.478 in total**.

`e-python-lanes` is the odd one and says so in its own command file.
`bpe-trainer` and `cross-val-folds` have no host family because NEITHER HAS A
BUILD: both run an independent Python implementation
(`mojolearn/_bpe_trainer.py`, `model_selection._default_folds`). Their
negative control is an IMPLEMENTATION env switch inside the library,
`MOJOLEARN_BPE_TRAINER_SABOTAGE` and `MOJOLEARN_FOLD_ORDER_SABOTAGE`, which is
not the harness and not a compiled define. The matrix's vocabulary has two
words, `build` and `harness`, and this is the first kind: the implementation
was perturbed, not the probe. `cpu-x86.replay.json` is the control for the
control -- a second clean run with the switches off, IDENTICAL=4 against the
first, so the move is the switch and not run-to-run variation.

## What moved

| lane | shard | fixtures | parts moved | fixtures where nothing moved |
|---|---|---|---|---|
| `arima` | c-classical | 2 | batch,infer,model,train | - |
| `arima-011` | c-classical | 2 | batch,infer,model,train | - |
| `arima-seasonal-c` | c-classical | 2 | batch,infer,model,train | - |
| `bootstrap` | c-classical | 2 | batch,train | - |
| `bpe-trainer` | e-python-lanes | 2 | train | - |
| `byte-lm` | d-neural | 2 | batch,infer,model,train | - |
| `byte-lm-host-infer` | d-neural | 2 | batch,infer,train | - |
| `byte-lm-host-infer-threaded` | d-neural | 2 | batch,infer,train | - |
| `byte-lm-host-train` | d-neural | 2 | infer,train | - |
| `byte-lm-resident` | d-neural | 2 | batch,infer,model,train | - |
| `cross-entropy-arms` | d-neural | 2 | batch,train | - |
| `cross-val` | f-recovery | 2 | batch,train | - |
| `cross-val-folds` | e-python-lanes | 2 | train | - |
| `dbscan` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `dbscan-brute-l1` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `dbscan-weighted` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `elasticnet` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `elasticnet-l2end-no-intercept` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `et-clf-entropy-bestfirst` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `et-reg-bootstrap-parallel` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-adapter-clf` | b-trees-gbdt | 2 | batch,infer,train | - |
| `gbdt-adapter-reg` | b-trees-gbdt | 2 | batch,infer,train | - |
| `gbdt-adapter-score-weighted` | f-recovery | 2 | train | - |
| `gbdt-categorical-ctr` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-categorical-ctr-tables` | f-recovery | 2 | batch,infer,train | - |
| `gbdt-depthwise` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-exact-mae` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-feature-freq` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-lossguide` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-lossguide-newtoncosine` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-multiclass` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-nan-modes` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-onevsall` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-ordered-rmse` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-pair-logit` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-parametric-losses` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-pointwise-l2-bayesian-eval` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-query-rmse` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-rmse` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-symmetric` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gbdt-tensor-ctr-tables` | f-recovery | 2 | batch,infer,train | - |
| `gbdt-yeti-rank` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `gemm-bf16` | c-classical | 2 | train | - |
| `gemm-int8` | c-classical | 2 | train | - |
| `gemm-pinned` | c-classical | 2 | batch,train | - |
| `gemm-transposed` | c-classical | 2 | batch,train | - |
| `gmm-random-init-sample` | c-classical | 2 | infer,model,train | - |
| `gmm-sample` | c-classical | 2 | infer,model,train | - |
| `gp-normalize-y` | f-recovery | 2 | batch,infer,model,train | - |
| `gp-optimize` | c-classical | 2 | batch,infer,model,train | - |
| `gp-optimize-restarts` | c-classical | 2 | batch,infer,model,train | - |
| `gp-sample-y` | c-classical | 2 | infer,model,train | - |
| `gp-sample-y-normalize` | f-recovery | 2 | infer,model,train | - |
| `gpc` | c-classical | 2 | batch,infer,model,train | - |
| `gpc-multiclass` | c-classical | 2 | batch,infer,model,train | - |
| `iforest-tuned` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `kde` | a-linear-neighbors | 2 | batch,infer,train | - |
| `kde-cosine-minkowski` | a-linear-neighbors | 2 | batch,infer,train | - |
| `kde-epanechnikov-l1` | a-linear-neighbors | 2 | batch,infer,train | - |
| `kde-exponential-chebyshev` | a-linear-neighbors | 2 | batch,infer,train | - |
| `kde-linear-cosine` | a-linear-neighbors | 2 | batch,infer,train | - |
| `kde-tophat-sqeuclidean` | a-linear-neighbors | 2 | batch,infer,train | - |
| `kde-weighted` | a-linear-neighbors | 2 | batch,infer,train | - |
| `kmeans-cosine` | a-linear-neighbors | 2 | **INERT** | base,ties |
| `knn` | a-linear-neighbors | 2 | batch,infer,train | - |
| `knn-chebyshev` | a-linear-neighbors | 2 | batch,infer,train | - |
| `knn-clf` | a-linear-neighbors | 2 | batch,infer,train | - |
| `knn-clf-distance` | a-linear-neighbors | 2 | batch,infer,train | - |
| `knn-cosine` | a-linear-neighbors | 2 | batch,infer,train | - |
| `knn-manhattan` | a-linear-neighbors | 2 | batch,infer,train | - |
| `knn-minkowski-p3` | a-linear-neighbors | 2 | batch,infer,train | - |
| `knn-rbc` | a-linear-neighbors | 2 | batch,infer,train | - |
| `knn-reg` | a-linear-neighbors | 2 | batch,infer,train | - |
| `knn-reg-distance` | a-linear-neighbors | 2 | batch,infer,train | - |
| `knn-sqeuclidean` | a-linear-neighbors | 2 | batch,infer,train | - |
| `lasso` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `logistic` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `logistic-elasticnet` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `logistic-l1` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `logistic-multiclass` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `logistic-unpenalized-no-intercept` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `mamba1-bf16w` | d-neural | 2 | batch,infer,train | - |
| `mamba1-int8w` | d-neural | 2 | batch,infer,train | - |
| `mamba2-bf16w` | d-neural | 2 | batch,infer,train | - |
| `mamba2-int8w` | d-neural | 2 | batch,infer,train | - |
| `mamba3-bf16w` | d-neural | 2 | batch,infer,train | - |
| `mamba3-int8w` | d-neural | 2 | batch,infer,train | - |
| `metrics` | c-classical | 2 | train | - |
| `metrics-classification` | c-classical | 2 | batch,train | - |
| `minmax-scaler` | a-linear-neighbors | 2 | batch,infer,model,train | ties |
| `minmax-scaler-clip` | a-linear-neighbors | 2 | batch,infer,model,train | ties |
| `mlp` | d-neural | 2 | batch,infer,model,train | - |
| `mlp-bf16w` | d-neural | 2 | infer,train | - |
| `mlp-int8w` | d-neural | 2 | infer,train | - |
| `monte-carlo` | c-classical | 2 | train | - |
| `ols-no-intercept` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `ols-weighted` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `optim-adam-clip` | d-neural | 2 | train | - |
| `optim-sgd` | d-neural | 2 | train | - |
| `ordered-gradient-sum` | g-ordered-gradient-sum | 9 | **INERT** | base,denormal,denormal_ftz,dupes,hashed,negative,odd,ties,wide |
| `par-arima` | c-classical | 2 | batch,infer,model,train | - |
| `par-forest` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `par-forest-et` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `par-holtwinters` | c-classical | 2 | batch,infer,model,train | - |
| `par-mlp` | d-neural | 2 | batch,infer,model,train | - |
| `par-queries-kde` | a-linear-neighbors | 2 | batch,infer,train | - |
| `par-queries-knn` | a-linear-neighbors | 2 | batch,infer,train | - |
| `par-queries-radius` | a-linear-neighbors | 2 | batch,infer,train | - |
| `par-reference-knn` | a-linear-neighbors | 2 | batch,infer,train | ties |
| `par-reference-knn-reg` | a-linear-neighbors | 2 | batch,infer,train | ties |
| `par-scaler` | a-linear-neighbors | 2 | **REFUSED under sabotage** (the lane's own assertion fires) | base,ties |
| `pca` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `pca-full-whiten` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `pca-whiten` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `permutation-test` | c-classical | 2 | batch,train | - |
| `radius` | a-linear-neighbors | 2 | batch,infer,train | - |
| `radius-chebyshev` | a-linear-neighbors | 2 | batch,infer,train | - |
| `radius-manhattan` | a-linear-neighbors | 2 | batch,infer,train | - |
| `radius-minkowski-p3` | a-linear-neighbors | 2 | batch,infer,train | - |
| `rf-clf` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `rf-clf-balanced-parallel` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `rf-clf-entropy-log2-noboot` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `rf-reg` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `rf-reg-gamma-ig` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `rf-reg-poisson` | b-trees-gbdt | 2 | batch,infer,model,train | - |
| `rf-score-weighted` | f-recovery | 2 | train | - |
| `ridge-no-intercept` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `samba-bf16w` | d-neural | 2 | infer,train | - |
| `samba-int8w` | d-neural | 2 | infer,train | - |
| `spectral-precomputed` | c-classical | 2 | batch,infer,model,train | - |
| `standard-scaler` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `standard-scaler-no-mean` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `standard-scaler-no-std` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `training-primitives` | d-neural | 2 | batch,infer,train | - |
| `transformer-bf16w` | d-neural | 2 | batch,infer,train | - |
| `transformer-int8w` | d-neural | 2 | batch,infer,train | - |
| `tsvd` | a-linear-neighbors | 2 | batch,infer,model,train | - |
| `umap` | c-classical | 2 | batch,infer,model,train | - |

135 of 138 lanes moved

## The three that did not move, and what each one is

**`kmeans-cosine` is INERT and should be.** Its cell is the hash of a REFUSAL
SENTENCE: `metric='cosine'` is refused by name in
`cluster/impl/kmeans_params.mojo::validate`, the lane catches the exception and
hashes its text, and no arithmetic runs at all. There is nothing for any build
to perturb, and the cell's job is the opposite one -- a column that HASHES a
result here means the refusal was lifted without a fused cosine arm. Reported
as a structural fact, not a gap; no arm should be written for it.

**`par-scaler`'s control fires and the cell cannot show it.** Under the
preprocessing family's sabotage build the lane raises its own
`ValueError: transform_scaler and plain transform differ: 2847 bytes of 16384`,
so the cell reads REFUSED while the clean cell reads STABLE. The arm worked;
the lane converted the catch into a refusal, and a refusal is not a catch
(`tools/cpu_identity_gate_check.py owed` says so, and the `cross-val-folds`
lane's own docstring gives the rule: "a raise reads REFUSED ... while a hash
that moves is a catch"). The repair is to HASH the disagreement instead of
raising, which changes the lane's cell and therefore every committed record of
it, so it is named here and left for a lane that owns `LANE_REVISIONS`.

**`ordered-gradient-sum` was INERT on all nine fixtures, and that is a defect
this branch fixes.** `g-ordered-gradient-sum/` is the unfixed measurement:
IDENTICAL=9 against the training family's sabotage build as main carries it.
The lane reaches `parallel_training.ordered_sum_gradients` ->
`_training_impl.accumulate_grads` -> the training binding's `accumulate` ->
`training/host/samba_ops_oracle.mojo::host_samba_accumulate`, and no GEMM is
anywhere on that path, so the family's only arms could not touch it. The file
said an arm there "would be inert" because an addition commutes, which is an
argument against an ORDER arm and the same one the shared GEMM leaf arm lost on
`ties` in the 2026-09-16 audit. `SAMBA_ACCUMULATE_HOST_SABOTAGE` is the value
flip that answers it; `g-accumulate-arm/` is the same nine fixtures with it
compiled in.

## What this record does not claim

These are CPU columns. x86-64 EPYC is bitwise identical to the M4 and to the
three GPU columns on every lane any record has compared, which is why a CPU
negative control is evidence about the arm at all, but a sabotage build seen
to move bytes on a CPU host binding says nothing about whether the same defect
would be caught on a GPU column: that is what the recorded GPU columns are
for, and those are release work.

Two fixtures, `base` and `ties`, except where a lane is named above as having
run all nine. A sabotage that moves ONE part on ONE fixture has been seen to
move a build; a wider fixture set would say more about WHERE it moves, not
whether it does.

Nothing here re-records a clean column for the reference table.
`python/mojolearn/verify_reference/table.json` belongs to lane/reference-regen
and this branch does not touch it.
