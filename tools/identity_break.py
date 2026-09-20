#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""identity_break: TRY TO BREAK cross-vendor bit-identity, on every estimator.

`tools/repeat_run_stability.py` asks one lane one question on one benign
fixture. This asks every public estimator EIGHT hostile questions, under the
`identical` tier, and writes a per-cell fingerprint so three vendors' runs
can be diffed cell by cell:

    ties      integer-grid data, every distance and split value tied many ways
    hashed    values derived from a hash, no structure, no ties, no symmetry
    wide      column magnitudes spanning 1e-4 .. 1e4 (accumulation order bites)
    denormal  a slice of the data sits below float32 normal (flush-to-zero)
    denormal_ftz  the same with subnormals flushed: the diagnostic twin
    dupes     duplicated rows, a constant column, an all-zero column
    odd       n = 12345, d = 17 -- nothing a block, warp or tile divides evenly
    negative  every value negative, shifted off the origin

    plus `base`, the stability harness's own fixture, as the control.

Every cell is fitted TWICE in one process so a run-to-run mover on this box
is separated from a cross-vendor divergence. A lane that raises reports
REFUSED with the message; nothing is swallowed.

THREE COLUMNS PER CELL, each supporting one public claim (2026-09-13), and a
fourth part, batch (2026-09-14). Every
column is computed on BOTH fits of the cell, so a MOVED column is this box
disagreeing with itself and a DIVERGENT column is two vendors disagreeing.

    train   the cell hash as it has always been (JSON keys `hashes`, `parts`,
            `verdict`), the just-fitted model read back on mostly its own
            training rows and, for the linear and decomposition lanes, its
            fitted state. It supports TRAINING identity, the same fit on
            every vendor. Its bytes and its keys are unchanged, so an old
            JSON diffs against a new one on this column.
    infer   sha256 of the fitted model's outputs on HELD-OUT rows the model
            never saw, the same fixture kind drawn from seed 1 (`odd` keeps
            its odd n and d), through the same public methods the train
            column hashes (predict, predict_proba, transform,
            decision_function, score_samples, kneighbors). It supports
            INFERENCE identity, the same predictions on unseen rows. A lane
            whose estimator has no out-of-sample method records
            `n/a:<reason>` and never a hash of training-row output.
            transductive means the labels belong to the fitted rows only
            (spectral answered transductive until 2026-09-15, when `predict`
            from a `prediction_data=True` fit arrived, DEVIATION 2860, and
            its two lanes' probe became `predict` on the held-out rows; DBSCAN and agglomerative answered transductive until
            2026-09-15, when `predict` from a `prediction_data=True` fit
            arrived, DEVIATION 2740, and their lanes' probe became
            `predict` on the held-out rows); KMeans answered no-predict until 2026-09-15, when
            `predict` arrived and its lanes' probe became `predict` on the
            held-out rows, with `predict` on the training rows held to
            `labels_`; `transform` joined the probe later that day, with
            `transform(X)` at `labels_` held to each row's minimum; function means the lane is not an estimator
            (linalg, metrics). THE FORECASTERS (holtwinters, arima; the
            infer probes of 2026-09-14) take no new rows, so their held-out
            axis is TIME: the infer column hashes the forecast beyond the
            fitted series at FORECAST_HORIZON steps, the fitted length, a
            horizon the train column (24 steps) does not hash, through
            every public out-of-sample entry (`forecast`, and ARIMA's
            `predict(n_obs, n_obs + h)`, which its docstring says is the
            same answer; the probe holds it to that). Until then those
            two lanes recorded `n/a:forecast`; a JSON that predates the
            probe reads ONE-COLUMN against a new one, never DIVERGENT.
    model   sha256 of the bytes `save(path)` wrote, for every estimator that
            has `save` and `load` (the random forests, the extra trees and
            every GradientBoosting lane). It supports ARTIFACT identity, the
            same model file on every vendor. The saved file is then loaded
            back and asked for the held-out rows again; if that hash differs
            from `infer` the column is RELOAD-MOVED, a file that does not
            predict what the model in memory predicts. Estimators without
            save/load record `n/a:no-save`. The trainers and SambaStack
            offer `save_checkpoint`/`from_checkpoint` instead; that pair
            feeds the same column (2026-09-13).
    batch   BATCH INVARIANCE (2026-09-14), JSON keys `batch`,
            `batch_verdict`, `batch_error`. Does a row's answer depend on
            which other rows were in the call? Serving systems answer
            differently at temperature 0 partly because batch size changes
            the arithmetic (Thinking Machines, "Defeating Nondeterminism in
            LLM Inference", September 2025); the transformer and Mamba
            contracts say batch and length are launch shape, not
            arithmetic, and this part holds the public surface to that.
            For every lane with an inference call, the fitted model is
            asked the lane's held-out input three ways through the lane's
            own methods: the WHOLE batch in one call, each of the first 16
            rows ALONE (a batch of 1; `--batch-alone N`), and a fixed uneven
            SPLIT, chunks of 1, 7 and the rest, concatenated. The causal
            sequence models (byte LM, Mamba 1/2/3, transformer, Samba) are
            also asked every prefix LENGTH 1, 7 and L-1 against the whole
            length, and the forecasters the horizons 1, 7 and H-1 against H
            (their batch of series is a different fit, so their part is
            length only). A sequence is compared alone against same-length
            peers here; right-padded ragged batches are the `ragged` part. The
            value is one hash of the whole-batch outputs when every
            comparison is the same bytes, else `BATCH_MOVED:<call>:<where>:
            <first output and element, both values in hex>` at the first
            difference. The verdict over repeats is STABLE, MOVED (the hash
            disagreed with itself run to run), BATCH_MOVED, N/A or REFUSED.
            BATCH_MOVED fails the run under IDENTICAL and is RECORDED, not
            failed, under FAST and DETERMINISTIC (neither promises it:
            checks/batch_invariance_check.mojo's tier table); `--diff` reads
            it the same way per column (BATCH_MOVED-RECORDED when no
            IDENTICAL column moved) and `--require-columns` counts batch
            hashes as it counts infer and model hashes. A JSON that predates
            the part carries no `batch_verdict` and reads NOT-COMPARED.
            The n/a reasons are transductive as for infer, function
            (metrics, resampling, cross-validation), no-batch-axis
            (the tokenizer in records before 2026-09-15, when
            BpeTokenizer.encode_batch gave it documents as rows), optimizer-step and training-step (the batch IS the
            arithmetic of a step), no-model (a trainer lane that returns
            no estimator). batch-dependent-by-contract was UMAP.transform's
            reason until lane/umap-batch-fix made the transform row
            separable and turned the exemption into a part. A method
            that refuses a batch of one BY NAME (the coordinate descent
            predict, mirroring cuML's cdPredict) is held to that refusal and
            its rows are asked in the smallest admitted batch instead
            (`_BatchRows` min_batch). `summary (infer/model):` and
            `summary (batch):` are separate lines of `--diff`. Cost is bounded: at most 1 + 16 + 3 extra calls per
            row call and 4 per length call, per cell per repeat; the
            declarations sit outside the lane bodies (`BATCH`), so no train,
            infer or model hash moves. IT TESTS the host call path's batch
            splitting through the bindings, on ONE box. IT DOES NOT TEST
            serving-scale B, padding or ragged batches, or the backward
            pass; since 2026-09-15 those are the opt-in parts `batchscale`,
            `ragged` and `batchgrad` below. No part here makes a
            cross-vendor statement; that comes from `--diff` over records.
            THE SABOTAGE turns every hashed batch cell BATCH_MOVED,
            and a run where it does not is a broken part

                MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1 MOJOLEARN_NUMERIC_MODE=identical \\
                  python3 tools/identity_break.py --lanes standard-scaler --fixtures base --repeats 1
            It flips the lowest bit of the first float element of every
            whole-batch answer (one ulp) and stamps `batch_sabotage: true`
            in the JSON.
    rlpair  THE RL PAIR (2026-09-15), JSON keys `rlpair`, `rlpair_verdict`,
            `rlpair_error`, `rlpair_sides`, on the lanes that declare it
            (`RLPAIR`: byte-lm, byte-lm-resident, byte-lm-host-infer,
            byte-lm-host-infer-threaded, mamba1, mamba2, mamba3,
            mamba2-dtlimit, transformer, transformer-window, samba,
            samba-untied-dropout-accum); every other lane carries no key.
            Reinforcement learning samples with one program and trains with
            another, and assumes the log-probability the sampler recorded
            for a token is the one the trainer recomputes (the true
            on-policy RL section of Thinking Machines, "Defeating
            Nondeterminism in LLM Inference", September 2025). SAMPLER:
            RLPAIR_SEQS = 5 prompts of RLPAIR_PROMPT = 4 held-out ids are
            prefilled and RLPAIR_DECODE = 12 tokens decoded one step at a
            time through the lane's public state API, greedy (argmax, lowest
            index on a tie, no RNG), recording each chosen id, its logits
            row and its NLL. TRAINER: the realized sequences, teacher
            forced, through the TRAINING forward, one cross-entropy call over
            every position. The NLL is `training.cross_entropy(reduction=
            'none')`, the library's own loss kernel (label smoothing 0.0),
            never numpy; the log-probability is its exact negation. Asserted,
            bytewise on ids, NLL and logits: (1) the sampler at B1 = 5 equals
            the trainer at B2 = 5; (2) the trainer at B2 = 1 per row and at
            the microbatch split 2, 3 equals it, the sampler at B1 = 1 per
            row equals it, and every further sampler (below) equals it;
            (3) CONTINUOUS BATCHING: row 4 is prefilled and decoded alone for
            3 tokens, then its state joins the batch of the other four at row
            2; row 1 leaves after token 8; every row's tokens equal the
            trainer's. The state rows are gathered and joined through the
            layouts the state classes document (`_rl_take`, `_rl_cat`); the
            Mamba-2/3 states and the KV cache hold ONE cursor per batch, so a
            row joins only at the batch's position. THE SIDES per lane: the
            blocks (no vocabulary of their own) are closed into the smallest
            language model SambaStack's forward is made of, a hashed (48, d)
            embedding, the block, a final RMSNorm and a tied head, all
            training primitives; sampler `allocate_state` + `forward(x,
            state)` + `step`, trainer the stateless `forward(x)` (the
            transformer's and Mamba-3's IDENTICAL stateless forward is a
            different native entry from their stateful one). samba: sampler
            `SambaStack.allocate_state` + `forward(ids, state)` + `step`
            (added 2026-09-15 for this part), trainer `forward(ids)`, the
            `_forward` its loss runs. byte-lm: the byte LM has NO decode
            state API, so its sampler recomputes the prefix (no KV cache)
            through `trainer.logits`, and a second sampler is
            LanguageModelInference on the CPU on the trainer's parameters
            when the host binding is present (`rlpair_sides` says whether it
            ran); trainer `trainer.logits`. byte-lm-host-infer: the CPU
            training step exposes no logits, so the trainer side is the
            reference forward `logits(threaded=False)` and the samplers are
            both arms. The value is one hash of the prompts and the trainer's
            ids, NLL and logits when every assertion holds, else
            `RLPAIR_MOVED:<pair>:seq <s> token <k>:<which differ, values in
            hex>` at the earliest token. Verdicts over repeats: STABLE,
            MOVED, RLPAIR_MOVED, N/A, REFUSED; RLPAIR_MOVED fails the run
            under IDENTICAL and is recorded under FAST and DETERMINISTIC.
            `summary (rlpair):` is its own `--diff` line, over the declaring
            lanes only; a JSON that predates the part reads NOT-COMPARED.
            Since the hash is the same bytes wherever the assertions hold,
            IDENTICAL xN on a cell says the sampler on any of those columns
            gives the trainer on any other the same log-probabilities (sample
            on the M4 CPU, train on the H100: the same bits). IT DOES NOT
            TEST sampling with an RNG, temperature, top-k or top-p, rows at
            different positions in one batch (not expressible), prompts of
            different lengths, the backward pass or the optimizer step, a
            KV cache for the byte LM, or any shape beyond these. THE
            SABOTAGE flips the lowest bit of the first sampler NLL and must
            turn every hashed rlpair cell RLPAIR_MOVED

                MOJOLEARN_IDENTITY_RLPAIR_SABOTAGE=1 MOJOLEARN_NUMERIC_MODE=identical \\
                  python3 tools/identity_break.py --lanes mamba1 --fixtures base --repeats 1

FOUR OPT-IN PARTS (2026-09-15, the fourth 2026-09-16), each its own JSON keys (`<part>`,
`<part>_verdict`, `<part>_error`, `<part>_notes`, `<part>_protocol`,
`<part>_sabotage`) and its own `summary (<part>):` line of `--diff`,
printed only where a column carries it. A run that does not pass the flag
writes none of the keys, so every record and gate grep before them reads
exactly as it did. The verdicts are the batch part's: STABLE, MOVED,
BATCH_MOVED (a failure under IDENTICAL only), N/A, REFUSED.

    batchgrad   `--batch-grad`. THE BACKWARD PASS. Two questions, and the
            contracts decide which one each gradient is asked.
            A PER-ROW gradient (the sequence blocks' `backward(x, g)["x"]`,
            `cross_entropy` dlogits under reduction='sum' with the divisor
            fixed at the whole batch's N, `linear_backward`'s da and
            `rms_norm_backward`'s dx) reads one row, so it is asked exactly
            as the batch part asks a forward: every row alone and the split
            1, 7, rest (loss contract 7.1; the block contracts' row
            independence). A gradient SUMMED over rows (every weight
            gradient) cannot equal a row alone by definition; its question
            is ACCUMULATION, and optimizer contract clause 9.2 says exactly
            when accumulation is bit exact: A equal row pieces of T tokens
            with `contract_leaf_size(T / A) == contract_leaf_size(T)`, T a
            multiple of the leaf, A dividing P and a power of two, combined
            by the balanced tree. So the part asks A in {2, 4, 8} at T = 512,
            asks `training.accumulation_is_aligned(T, A)`, combines aligned
            pieces with `training.accumulate_grads(pieces, tokens=T)` and
            holds every CLAIMED tensor to the whole batch byte for byte; A = 8
            (refused at T = 512) is the negative control, combined with no
            alignment claim and recorded as how many claimed tensors moved;
            the uneven split 1, 7, rest is recorded n/a with clause 9.2's
            reason (A = 3, not a power of two, pieces not subtrees). CLAIMED
            means the tensor's token contraction is the v1 GEMM at k' = T,
            read from each backward's source (`GRAD_CLAIMED`): all nine
            transformer weights; Mamba-1 all but A_log; Mamba-2 the two
            projections, both norms and D, not conv1d, dt_bias or A_log
            (serial folds in mamba2_ssd_backward.mojo); Mamba-3 all nine;
            `linear_backward` and `rms_norm_backward` dweight; for
            `SambaStack.loss_and_grads(num_items=T)` only the LM head and
            final norm, because clause 9.3 claims those two and reports the
            rest. A tensor that is not claimed is REPORTED: whether it moved
            at an aligned split is in the hash and in `batchgrad_notes`,
            never a failure. The embedding's CARRY (embedding contract 7.4,
            bit exact at EVERY split) is asked at 1, 7, rest and in pieces of
            16. n/a reasons: mean-reduction (SmallMLPTrainer), mean-reduction-
            fixed-batch (the byte LM trainers), optimizer-step, driver-lane
            (par-*), no-backward (every estimator without a gradient call;
            no classical estimator on the public surface has a gradient or a
            partial_fit, and the scalers' partial_fit raises by name).
            SABOTAGE `MOJOLEARN_IDENTITY_BATCHGRAD_SABOTAGE=1` flips one bit of
            every whole-batch gradient (every hashed cell must read
            BATCH_MOVED); `=serial` replaces condition 5's tree with a running
            pair sum, which the contract says is inert at A <= 2, so a cell
            that moves at A = 4 is condition 5 seen biting (recorded).
    batchscale  `--batch-scale`. SERVING-SCALE B. A whole call of 1024 rows
            and sub-batches of B in {1, 17, 64, 256} at its start, middle and
            end, every sub-batch row equal to its row of the whole; for the
            causal sequence models also B = 2 at L = 1024 against every
            prefix in {1, 7, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            512, 513, 1023} (the Mamba-3 chunk 64, the GEMM leaf 128, the
            Mamba-2 chunk 256). Lanes: the sequence blocks, the byte LM lanes
            (the long length loads the lane's parameters into the default
            shape with `length=1024`), Samba, and the classical serving set
            knn, rf-clf, rf-reg, gbdt-symmetric, gbdt-rmse, ols, ridge,
            logistic, pca. Inputs past a fixture's size wrap around it. The
            batch part's sabotage switch sabotages this part too.
    ragged  `--ragged`. PADDED AND RAGGED BATCHES, through the `lengths=`
            argument the causal sequence models take since 2026-09-15
            (python/mojolearn/_ragged.py: TransformerBlock, Mamba1/2/3Block
            and SambaStack `forward`, the byte LM `logits` and `next_bytes`).
            Rows of lengths (16, 1, 7, 9, 16, 3, 12, 15) at L = 16 and
            (300, 1, 65, 129, 257, 128) at L = 300, the padding filled with
            NaN (floats) or vocab + 7 (ids). Every row's real positions must
            equal the row run alone at its own length, and every padding
            output must be exactly +0.0; next_bytes must equal the row alone.
            The batch part's sabotage switch sabotages this part too.

    stepfull `--step-full`. THE DECODE IS THE PREFILL
            (lane/stateful-cpu-decoding, 2026-09-16). One fresh-state
            forward pass over a sequence of 16 positions, against the SAME
            sequence decoded one token at a time through `allocate_state`
            and `step` with the state carried, compared BITWISE POSITION BY
            POSITION; a mismatch reports the first differing position and
            both values, never a count. It asks the PUBLIC class, which on a
            CPU column is the `*Inference` wrapper over the shipped neural
            binding. It applies to the lanes with a decode state: the
            Transformer (both windows), Mamba-1/2/3 and the Samba stack.
            A model whose streaming path answers different bits from its
            prefill is a different model wearing the same name, which no
            cross-vendor statement about the prefill covers. The batch
            part's sabotage switch sabotages this part too;
            tools/step_vs_full_check.py carries the deeper arm, one ULP on
            one carried cache cell.

THE LANES, 178 (2026-09-14; 46 on 2026-09-13, pca-whiten the same night, 71
on 2026-09-14 from the claim-surface census, logistic-multiclass and
tokenizer the same day when those two got their doors, 16 `par-*` lanes that
evening for the ordered multi-GPU drivers run on ONE device, and 15 lanes for
the doors workstream D opened: Cholesky, the kernel methods, the Gaussian
mixture, HDBSCAN, resampling, the training primitives and the KMeans arms,
then 15 more `par-*` lanes that night for the multi-GPU drivers the first 16
missed, then `par-forest-pool`, `par-gmm`, `par-resample` and `par-hdbscan` for the drivers the multigpu lane added, and `ivf` and `embedding` when IVFIndex and Embedding left `_NOT_YET`, then `ivf-euclidean` when its metric stopped being refused, then `embedding-sort` for PLAN_SORT, and `par-cholesky`, `par-kernel-ridge`, `par-nystroem` and `par-rbf-sampler` on 2026-09-15, then `gmm-sample` and `gmm-random-init-sample` for GaussianMixture.sample the same day, then `gp-sample-y` and `gp-sample-y-normalize` for GaussianProcessRegressor.sample_y). One per public estimator
plus linalg and metrics, then one per public constructor VALUE that selects
a different numeric path and no earlier lane pins (a kernel, an objective, a
sampler, a solver, a metric, a reduction).

    trees      rf-clf rf-reg et-clf et-reg gbdt-symmetric gbdt-depthwise
               gbdt-lossguide gbdt-rmse gbdt-ordered-rmse gbdt-feature-freq
               iforest (last on purpose, see its comment)
    classical  kmeans knn knn-clf knn-reg radius dbscan pca tsvd ols ridge
               logistic lasso elasticnet svc svr kde gp agglomerative
               spectral holtwinters arima umap standard-scaler minmax-scaler
    neural     mlp byte-lm byte-lm-host-infer byte-lm-host-train mamba1
               mamba2 mamba3 transformer samba
    functions  gemm-pinned metrics
    2026-09-14 (docs/lanes/BRIEF_claim_surface_census_2026-09-14.md)
      trees    rf-clf-entropy-log2-noboot rf-clf-balanced-parallel rf-reg-poisson
               rf-reg-gamma-ig et-clf-entropy-bestfirst et-reg-bootstrap-parallel
               gbdt-multiclass gbdt-onevsall gbdt-parametric-losses
               gbdt-lossguide-newtoncosine gbdt-pointwise-l2-bayesian-eval
               gbdt-exact-mae gbdt-categorical-ctr gbdt-nan-modes gbdt-adapter-clf
               gbdt-adapter-reg gbdt-query-rmse gbdt-pair-logit iforest-tuned (last, with iforest)
      neural   mamba2-dtlimit transformer-window byte-lm-resident
               byte-lm-host-infer-threaded (the SHIPPED default arm; the
               2026-09-13 lane pins the reference arm) samba-untied-dropout-accum
               optim-sgd optim-adam-clip cross-entropy-arms
      classical kmeans-random kmeans-array kmeans-weighted dbscan-brute-l1
               dbscan-weighted kde-<kernel>-<metric> x5 kde-weighted pca-full-whiten
               ols-no-intercept ols-weighted ridge-no-intercept logistic-l1
               logistic-elasticnet logistic-unpenalized-no-intercept
               logistic-multiclass elasticnet-l2end-no-intercept svc-linear svr-linear knn-<metric> x5
               knn-rbc knn-clf-distance knn-reg-distance radius-<metric> x3
               standard-scaler-no-mean standard-scaler-no-std minmax-scaler-clip
               spectral-precomputed holtwinters-multiplicative kpss arima-011
               arima-seasonal-c gp-matern12 gp-matern32 gp-matern52-ard
      functions gemm-transposed metrics-classification tokenizer cross-val
    2026-09-15 (lane/arima-exog) arima-exog arima-exog-seasonal
    2026-09-14 evening, workstream D (docs/lanes/LANE_BODY_*.py)
      cholesky kernel-ridge nystroem rbf-sampler gmm gmm-random-init hdbscan
               hdbscan-leaf bootstrap permutation-test monte-carlo
               training-primitives kmeans-sqrt kmeans-classic-pp
      2026-09-18 (lane/kmeans-cosine-capability) kmeans-cosine REMOVED with
               the cosine metric itself; cuVS refuses cosine k-means too
    2026-09-14 evening (docs/multi_gpu/README.md, run with devices=(0,))
      par-forest par-forest-et par-boosting par-kmeans par-gram par-logistic
               par-cd par-svm par-gp par-dbscan par-scaler par-arima par-mlp
               par-samba par-byte-lm par-iforest (last, with iforest)
    2026-09-14 night (the drivers no lane reached; devices=_par_devices())
      par-queries-knn par-queries-radius par-queries-kde par-reference-knn
               par-reference-knn-reg par-graph-agglomerative par-graph-spectral
               par-graph-umap par-ordered-rmse par-feature-freq
               par-boosting-pointwise par-holtwinters par-byte-lm-model-pool
               par-byte-lm-offload par-samba-clip
    2026-09-14 night (drivers added by the multigpu lane; devices=_par_devices())
      par-forest-pool par-gmm par-resample par-hdbscan
    2026-09-15 (the Cholesky and kernel-method drivers; devices=_par_devices())
      par-cholesky par-kernel-ridge par-nystroem par-rbf-sampler
    2026-09-15 (GaussianMixture.sample, DEVIATIONS 2791 and 2792)
      gmm-sample gmm-random-init-sample
    2026-09-15 (GaussianProcessRegressor.sample_y, DEVIATION 2793)
      gp-sample-y gp-sample-y-normalize
    2026-09-14 night (lane/expose-ivf-embedding, the last two _NOT_YET doors)
      ivf embedding
    2026-09-14 night (fix/ivf-l2sqrt, metric='euclidean' no longer refused)
      ivf-euclidean
    2026-09-15 (lane/embedding-owed, PLAN_SORT through Embedding(plan="sort"))
      embedding-sort
    2026-09-15 (lane/cpu-training-small-gaps)
      metrics-fowlkes-mallows gbdt-adapter-score-weighted rf-score-weighted svc-poly
      gp-normalize-y
    2026-09-19 (lane/catboost-parity: CatBoost's GPU defaults, Ordered
               boosting, the seven border types, the quantile constant)
      gbdt-catboost-defaults gbdt-ordered gbdt-ordered-bayesian-noise
               gbdt-border-types gbdt-bfa-quantile par-ordered par-border-types

The 18 lanes added on 2026-09-13 (svr through samba above) are fed the SAME
fixture bytes in the shape their estimator wants; the derivation rules are
the helpers under "derived inputs". Where an estimator's own docstring says
its identity card is one vendor or two, the lane's docstring repeats that;
the tool measures, it does not claim.

    MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py --json apple.json
    python3 tools/identity_break.py --diff apple.json nvidia.json amd.json

A `--diff` prints, per lane, per fixture: IDENTICAL (all columns agree),
DIVERGENT (they do not), MOVED (a column disagreed with itself), or the
refusal, first for the train column and then in a second table for the
infer and model columns, where RELOAD-MOVED is also possible and a JSON
that predates the columns reads NOT-COMPARED, never DIVERGENT. It exits
non-zero on any DIVERGENT, MOVED or RELOAD-MOVED cell. FAST is refused on
purpose: a bitwise question to a FAST arm is a category error
(fast-is-not-identical).

A leg that splits one column across processes (two one-device processes on
a two-GPU box, or two sequential legs on one box type) joins the parts with

    python3 tools/identity_break.py --merge part_a.json part_b.json --json column.json

which refuses parts that differ in vendor, commit, mode, repeats, fixtures,
held-out bytes, batch protocol, par_devices, the host object or binding
digests (the loaded GPU bindings, and on a CPU column the host binding files
since 2026-09-15), and a cell two parts carry differently (2026-09-14 night).
The CPU identity gate splits its covered lanes across processes this way
(tools/cpu_identity_gate_check.py run-column). Parts from two legs
(two builds of the same commit) need `--allow-separate-builds`, and the
merged JSON then says `merged_separate_builds: true` and keeps each part's
binding digests.

THE FOURTH COLUMN, a CPU (the CPU training lane, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md section 3.3). On an install whose
`mojolearn.vendor()` is 'cpu' (no GPU set, a host binding under
mojolearn/host/ built), `--vendor` defaults to `cpu-<cpu model slug>` read
from the machine, the JSON gains a `host` object (cpu model, arch, the host
bindings built, each file's sha256, and what each reads back as its
kernel-matrix column), and a
lane whose family has no host fit records REFUSED with the by-name sentence
"no CPU implementation of <binding>.<function> yet", never a hash of
something else. Two provenance rules apply to EVERY column since the same
day. `--vendor` must be a box label matching ^[a-z0-9][a-z0-9_.-]*$ and not
a placeholder (the 2026-09-13 NVIDIA column recorded "box-arch", an
unexpanded env default in the leg body), and `commit` is REQUIRED, from
MOJOLEARN_COMMIT, else `git rev-parse HEAD` of this checkout, else a COMMIT
or commit.txt witness at the repository root; a JSON with an empty commit
is not written (all three 2026-09-13 GPU columns carry "commit": "").
`--diff ... --require-columns N --lanes a,b` exits non-zero when any named
lane has fewer than N real hashes on a compared cell, because `IDENTICAL x3`
on a lane the CPU column should cover is the CPU binding refusing, not a
pass.

A REFUSED PART NAMES ITS CAUSE (2026-09-19). Every `*_error` field carries
the stage, the exception TYPE, its message and the traceback of the raise,
capped at ERROR_TEXT_LIMIT characters FROM THE OUTER END, so what a cap
drops is the harness calling in and what it keeps is the frame that raised
(and, for a `GPU worker failed` RuntimeError, the worker's own traceback,
which travels at the very end). `<json>.errors.txt` beside the record holds
every refusal untruncated; with no `--json` the full text is printed after
the summary.

CPU WALL CLOCK (2026-09-19). `--jobs N` is an opt-in process supervisor for
independent lane shards. It requires `--json`, caps N to min(8, CPU count),
forces every child through `--require-cpu`, preserves each shard JSON as a
resume checkpoint, and merges in deterministic lane order. It never shares a
fitted estimator or native runtime between workers and never runs concurrent
jobs on a GPU.
"""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

from contextlib import contextmanager
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=1)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


# --------------------------------------------------------------------------
# what a refused part carries
# --------------------------------------------------------------------------
#
# A CAP THAT KEPT THE WRONG END (2026-09-19, lane/lm-attention-fallback).
# Every probe used to hand a cell `f"{stage}: {type(exc).__name__}: {exc}"`
# cut with `[:300]`, and `[:300]` keeps the OUTERMOST frames of a traceback
# and throws away the innermost -- the frame that actually raised. On
# 2026-09-19 a two-device MI300X run refused `par-queries-nn`'s batch part on
# all nine fixtures and every cell carried the same 300 characters:
#
#     batch: RuntimeError: GPU worker failed:
#     Traceback (most recent call last):
#       File ".../_parallel_worker.py", line 386, in main
#         response = (True, execute(request))
#                           ~~~~~~~^^^^^^^^^
#       File ".../_parallel_worker.py", li
#
# -- the worker's OWN dispatch frame, cut mid-word before the operation, the
# exception type or its message. Nine reproductions and no cause.
#
# Two things fix it. `_exc_text` appends the formatted traceback of the
# raise, and `traceback.format_exception` puts `Type: message` LAST, so a
# worker RuntimeError whose message carries a remote traceback
# (`_parallel_pool._call` raises 'GPU worker failed:\n<worker traceback>')
# ends this text with that remote traceback's innermost frame. `_clip_error`
# then keeps the first line AND THE TAIL, so what survives a cap is the
# innermost frames, and it records the untruncated text in `_FULL_ERRORS`,
# which `write_error_sidecar` writes beside the run's JSON.

#: The cap on the text a CELL carries. The full text is never capped: it is
#: written beside the JSON. Sized to hold a nested traceback (a worker
#: failure carries two) rather than to be small.
ERROR_TEXT_LIMIT = 8000

#: {"<lane>/<fixture> <part>": the untruncated text}. Written beside the
#: run's JSON by `write_error_sidecar`.
_FULL_ERRORS = {}


def _exc_text(stage, exc):
    """The text a refused part carries: `<stage>: <Type>: <first line of the
    message>`, then the FULL traceback of the raise.

    The traceback goes last because `traceback.format_exception` ends with
    `Type: message`, and a worker failure's message IS the worker's own
    traceback. `_clip_error` keeps the tail, so the innermost frame on both
    sides of the RPC survives a cap.
    """
    message = str(exc)
    head = f"{type(exc).__name__}: {message.split(chr(10), 1)[0]}"
    if stage:
        head = f"{stage}: {head}"
    tail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).rstrip("\n")
    return f"{head}\n{tail}"


def _clip_error(text, key=None, limit=None):
    """`text` capped KEEPING THE INNERMOST FRAMES, and the untruncated text
    remembered under `key` for the sidecar.

    A traceback's outer frames say where the harness called in, which the
    cell key already says; its inner frames say what raised. So the cap keeps
    the first line (the stage and the exception type, which the summary table
    reads) and then the END of the text, and names how much it dropped. The
    old `[:300]` kept exactly the half that carries no cause."""
    if text is None:
        return None
    if key is not None:
        _FULL_ERRORS[key] = text
    limit = ERROR_TEXT_LIMIT if limit is None else limit
    if len(text) <= limit:
        return text
    head = text.split("\n", 1)[0][:limit // 4]
    note = "\n... %d characters of outer frames elided (the full text is in the .errors.txt beside the JSON) ...\n"
    keep = limit - len(head) - len(note) - 8
    return head + (note % (len(text) - len(head) - keep)) + text[-keep:]


def write_error_sidecar(json_path):
    """The untruncated text of every refusal this run recorded, beside the
    run's JSON. Returns the path written, or None when there was nothing to
    write. A cap that drops evidence is only acceptable while the evidence
    is kept somewhere, and this is that somewhere."""
    if not json_path or not _FULL_ERRORS:
        return None
    path = str(json_path) + ".errors.txt"
    with open(path, "w", encoding="utf-8") as stream:
        for key in sorted(_FULL_ERRORS):
            stream.write(f"===== {key} =====\n{_FULL_ERRORS[key]}\n\n")
    return path


class CellTimer:
    def __init__(self, label):
        self.label = label
        self.started = time.perf_counter()
        self.stages = {}

    @contextmanager
    def stage(self, name):
        started = time.perf_counter()
        print(f"# START {self.label} {name}", flush=True)
        try:
            yield
        finally:
            seconds = time.perf_counter() - started
            self.stages[name] = self.stages.get(name, 0.0) + seconds
            print(f"# DONE {self.label} {name} {seconds:.3f}s", flush=True)

    def result(self):
        return dict(total_seconds=time.perf_counter() - self.started, stages=dict(self.stages))


def resume_signature(args, package_dir, harness, provenance):
    """Bind reuse to Python/binary bytes, execution settings and full protocol.

    Native modules load lazily, so hash all installed candidates before any fit,
    not just the subset imported at checkpoint time. Reject a changed installation
    or source harness even when its declared version/commit did not change.
    """
    files = [Path(harness)]
    root = Path(package_dir)
    files += sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in (".py", ".so", ".dylib"))
    # Some host/sabotage installations live outside the Python package.
    for key, value in os.environ.items():
        if key.startswith("MOJOLEARN_") and (key.endswith("HOST_DIR") or value.endswith((".so", ".dylib"))):
            path = Path(value)
            if path.is_file() and path.suffix in (".so", ".dylib"):
                files.append(path)
            elif key.endswith("HOST_DIR") and path.is_dir():
                files += sorted(path.rglob("*.so"))
    h = hashlib.sha256()
    for path in files:
        h.update(str(path.resolve()).encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
    options = {k: v for k, v in vars(args).items() if k not in ("json", "resume", "verbose")}
    env = {k: v for k, v in os.environ.items() if k.startswith(("MOJO", "MODULAR", "OMP_", "MKL_", "OPENBLAS_", "VECLIB_"))}
    # Environment values can contain credentials. Compare them without publishing them.
    env_hash = hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest()
    return dict(schema=1, source_sha256=h.hexdigest(), options=options,
                environment_sha256=env_hash, provenance=provenance)


def resume_cells(path, signature):
    with open(path) as stream:
        previous = json.load(stream)
    if previous.get("resume_signature") != signature:
        raise SystemExit("REFUSING TO RESUME: sources, bindings, environment or run protocol changed")
    return previous["cells"]


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: A box label: lowercase, digits, `_ . -`, never a placeholder.
VENDOR_LABEL = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
PLACEHOLDER_LABELS = frozenset({
    "box-arch", "box", "arch", "vendor", "unknown", "none", "cpu", "gpu",
    "label", "todo", "tbd", "placeholder", "x", "test",
})
COMMIT_HASH = re.compile(r"^[0-9a-f]{7,40}$")


def cpu_model():
    """This machine's CPU model string, or None. `sysctl -n
    machdep.cpu.brand_string` on macOS; `model name` from /proc/cpuinfo, then
    `lscpu` "Model name" on Linux (an ARM64 /proc/cpuinfo has no model name)."""
    try:
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            return out or None
        try:
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    key, _, value = line.partition(":")
                    if key.strip().lower() in ("model name", "cpu model") and value.strip():
                        return value.strip()
        except OSError:
            pass
        out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
        for line in out.splitlines():
            key, _, value = line.partition(":")
            if key.strip().lower() == "model name" and value.strip():
                return value.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def slug(text):
    """`Apple M4` -> `apple-m4`, `Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz`
    -> `intel-r-xeon-r-platinum-8272cl-cpu-2.60ghz`."""
    s = re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-.")
    return re.sub(r"-{2,}", "-", s)


def check_vendor_label(label):
    """Refuse a label that is not a box: the regex, the placeholders, and any
    unexpanded `$` or `{`."""
    bad = (not label or not VENDOR_LABEL.match(label) or label in PLACEHOLDER_LABELS
           or "$" in label or "{" in label)
    if bad:
        raise SystemExit(
            f"REFUSING: --vendor {label!r} is not a box label. It must match "
            "^[a-z0-9][a-z0-9_.-]*$ and must not be a placeholder "
            f"({', '.join(sorted(PLACEHOLDER_LABELS))}). Name the box: apple-m4, "
            "nvidia-h100-sm_90a, amd-mi325x-gfx942, cpu-<cpu model slug>. The "
            "2026-09-13 NVIDIA column recorded \"box-arch\" from an unexpanded env "
            "default in the leg body; this refusal is what stops the next one."
        )
    return label


def default_vendor_label(ml):
    """`cpu-<cpu model slug>` on a CPU-only install, derived from the machine
    and never typed; `platform.machine()` elsewhere, as before."""
    if ml.vendor() == "cpu":
        model = cpu_model()
        if not model:
            raise SystemExit(
                "REFUSING: on a CPU-only install --vendor defaults to cpu-<cpu "
                "model slug> and this machine's CPU model could not be read; "
                "pass --vendor cpu-<model> explicitly"
            )
        return "cpu-" + slug(model)
    return platform.machine().lower()


def commit_witness():
    """(commit, source). MOJOLEARN_COMMIT wins; else `git rev-parse HEAD` of
    the checkout this tool lives in; else a COMMIT or commit.txt witness at
    its root (what a leg archive carries). Refuses an empty or malformed
    value: a column with no commit cannot be tied to the source it ran."""
    candidates = [("MOJOLEARN_COMMIT", os.environ.get("MOJOLEARN_COMMIT", "").strip())]
    try:
        git = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git = ""
    candidates.append(("git rev-parse HEAD", git))
    for name in ("COMMIT", "commit.txt"):
        p = os.path.join(ROOT, name)
        if os.path.exists(p):
            with open(p) as fh:
                text = fh.read().strip()
            candidates.append((name, text.split()[0] if text else ""))
    for source, value in candidates:
        if not value:
            continue
        if not COMMIT_HASH.match(value):
            raise SystemExit(f"REFUSING: commit {value!r} from {source} is not a git hash")
        return value, source
    raise SystemExit(
        "REFUSING: no commit witness. Set MOJOLEARN_COMMIT, run from a git "
        "checkout, or place a COMMIT or commit.txt file at the repository root. "
        "A JSON with an empty commit is not written."
    )


def host_record(ml):
    """The `host` object of a CPU column: the machine, the host bindings
    built, and what each reads back (its kernel-matrix column, the column
    the accelerator predicates detected, its sabotage flag). A binding that
    refuses to load records the refusal, so a REFUSED cell is attributable
    to an unbuilt or refused family rather than a bug."""
    from mojolearn import _backend, host_surface
    families = {}
    for basename in _backend.host_families_built():
        prefix = basename[len("_mojolearn_"):]
        # The file's bytes (2026-09-15, lane/cpu-training-gate-budget). The
        # host bindings load by path, so `package.bindings` (the loaded
        # `mojolearn._mojolearn*` modules) never lists them, and a CPU column
        # was tied to no binding bytes at all; `--merge` reads these digests
        # to refuse parts that ran different builds.
        digest = {}
        try:
            path = _backend.host_module_path(basename)
            with open(path, "rb") as fh:
                digest = dict(sha256=hashlib.sha256(fh.read()).hexdigest(), size=os.path.getsize(path))
        except OSError as exc:
            digest = dict(sha256_error=f"{type(exc).__name__}: {exc}"[:200])
        try:
            m = _backend.load_host_module(basename)
            fam = dict(column=str(getattr(m, prefix + "_column")()), **digest)
            detected = getattr(m, prefix + "_detected_column", None)
            if detected is not None:
                fam["detected_column"] = str(detected())
            sab = getattr(m, prefix + "_sabotage", None)
            if sab is not None:
                fam["sabotage"] = bool(sab())
        except Exception as exc:
            fam = dict(error=f"{type(exc).__name__}: {exc}"[:300], **digest)
        families[basename] = fam
    columns = sorted(set(f.get("column", "unreadable") for f in families.values()))
    return dict(
        cpu_model=cpu_model(), arch=platform.machine(),
        target_cpu=os.environ.get("MOJOLEARN_TARGET_CPU", "unrecorded"),
        mojo_version=os.environ.get("MOJOLEARN_MOJO_VERSION", "unrecorded"),
        python=platform.python_version(),
        column=(columns[0] if len(columns) == 1 else ("none-built" if not columns else "MIXED:" + ",".join(columns))),
        routed=dict(_backend._HOST_MODULES),
        # The manifest the routing table and the gate lists are read from
        # (the host surface manifest lane, 2026-09-14), and every binding it
        # declares, so a family that is declared but absent from `families`
        # reads as NOT BUILT on this box rather than as a bug.
        surface=host_surface.SOURCE,
        declared=host_surface.bindings(),
        covered_lanes=host_surface.covered_lanes(),
        families=families,
    )


def _h(*arrays):
    """sha256 over dtype, shape and bytes of each array. REFUSES an object
    array: `np.asarray` of a dict, a list of unequal arrays or any Python
    object makes dtype=object, whose bytes are ADDRESSES, and the cell then
    reads MOVED on every box for no arithmetic reason (byte-lm-resident,
    2026-09-14, nine fixtures on two GPUs). A lane must hand this the
    arrays themselves."""
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        if a.dtype == object or a.dtype.kind in "OUSV":
            raise TypeError(f"identity_break._h: refusing dtype={a.dtype} (an object or text array "
                            "hashes addresses or encodings, not numbers); pass the numeric arrays")
        m.update(str(a.dtype).encode())
        m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def _train_hash(parts):
    """The train column's cell hash of one fit: `_h` over the lane's parts
    dict spelled `key=value`, sorted by key, joined by `|`. The run loop and
    `python -m mojolearn verify --all` (python/mojolearn/_verify_all.py, which
    imports this module) both call this, so the shipped verifier cannot hash
    a cell differently from the record."""
    return _h(np.frombuffer("|".join(f"{k}={v}" for k, v in sorted(parts.items())).encode(), dtype=np.uint8))


def _hfile(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:16]


# ---------------------------------------------------------------- fixtures
# Small on purpose: the Mac side of this runs under the no-heavy-local-compute
# rule and the point is the ARITHMETIC PATH, which n=20000 already exercises
# multi-block. Sizes are fixed so a hash means the same thing on every box.

N, D = 20000, 16

#: AUDIT ONLY. MOJOLEARN_IDENTITY_N scales the fixture row count (for example to
#: push the par-* lanes' device-to-device buffers above 1 MiB). A column run
#: with it records `package.fixture_n`, hashes a DIFFERENT fixture, and must
#: never be diffed against a record at the default; `--diff` refuses a mix.
FIXTURE_N_ENV = "MOJOLEARN_IDENTITY_N"
if os.environ.get(FIXTURE_N_ENV, "").strip():
    N = int(os.environ[FIXTURE_N_ENV])
    if N < 8192:
        raise SystemExit(f"{FIXTURE_N_ENV}={N}: the lanes slice up to 8192 fixture rows")

#: the seed of the held-out draw; the training draw is seed 0
HELDOUT_SEED = 1

#: the forecasters' held-out axis is time: steps beyond the fitted series
#: in the infer column, equal to the fitted length (the two lanes fit 512
#: observations per series), as the held-out row counts mirror the
#: training row counts elsewhere. The train column keeps its 24 steps.
FORECAST_HORIZON = 512


def _hashed_uniform(n, d, seed):
    """Values from sha256 of the index, so no RNG-family assumption and no
    repeated structure; per-cell distinct with overwhelming probability."""
    out = np.empty(n * d, dtype=np.float32)
    ctr = np.arange(n * d, dtype=np.uint64)
    # vectorised: hash 8-byte blocks in chunks
    raw = ctr.tobytes()
    dig = hashlib.sha256(raw + str(seed).encode()).digest()
    # expand deterministically with a counter-mode stream
    stream = bytearray()
    blk = 0
    while len(stream) < n * d * 4:
        stream += hashlib.sha256(dig + blk.to_bytes(8, "little")).digest()
        blk += 1
    u32 = np.frombuffer(bytes(stream[: n * d * 4]), dtype=np.uint32)
    out[:] = (u32.astype(np.float64) / 2**32).astype(np.float32)
    return out.reshape(n, d)


def fixture(kind, n=N, d=D, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "base":
        X = rng.standard_normal((n, d)).astype(np.float32)
    elif kind == "ties":
        X = rng.integers(0, 6, size=(n, d)).astype(np.float32)
    elif kind == "hashed":
        X = (_hashed_uniform(n, d, seed) * 4.0 - 2.0).astype(np.float32)
    elif kind == "wide":
        X = rng.standard_normal((n, d)).astype(np.float32)
        scale = np.logspace(-4, 4, d).astype(np.float32)
        X = (X * scale).astype(np.float32)
    elif kind == "denormal":
        X = rng.standard_normal((n, d)).astype(np.float32)
        # a quarter of the rows, three columns, pushed into the subnormal range
        X[: n // 4, :3] = (X[: n // 4, :3] * np.float32(1e-40)).astype(np.float32)
        assert np.any((X != 0) & (np.abs(X) < np.finfo(np.float32).tiny))
    elif kind == "denormal_ftz":
        # THE DIAGNOSTIC TWIN of `denormal`: the same bytes with every
        # subnormal replaced by a signed zero, which is what a flush-to-zero
        # backend sees. A vendor whose `denormal` cell equals another vendor's
        # `denormal_ftz` cell is flushing where the other is not.
        X, _, _ = fixture("denormal", n, d, seed)
        sub = (X != 0) & (np.abs(X) < np.finfo(np.float32).tiny)
        X = X.copy()
        X[sub] = np.copysign(np.float32(0.0), X[sub])
    elif kind == "dupes":
        X = rng.standard_normal((n, d)).astype(np.float32)
        X[n // 2:] = X[: n - n // 2]          # second half duplicates the first
        X[:, d - 2] = np.float32(3.5)         # a constant column
        X[:, d - 1] = np.float32(0.0)         # an all-zero column
    elif kind == "odd":
        n, d = 12345, 17
        X = rng.standard_normal((n, d)).astype(np.float32)
    elif kind == "negative":
        X = (-np.abs(rng.standard_normal((n, d))) - 2.0).astype(np.float32)
    else:
        raise ValueError(kind)
    return (X,) + labels_for(X, seed)


def heldout(kind):
    """Rows the model never saw: the same fixture kind from HELDOUT_SEED, so
    it has the same shape, scale and pathology as the training draw (`odd`
    keeps its odd n and d) and no row in common with it. Only X is handed to
    the lanes; the held-out labels are not part of any column."""
    X, _, _ = fixture(kind, seed=HELDOUT_SEED)
    return X


def labels_for(X, seed=0):
    """The fixture targets for any X (shared with
    `checks/gbdt_sub_byte_identity_check.py`, so its integer-grid fixtures
    carry labels by the same rule, byte for byte)."""
    # targets: a signed rule on two columns, and a linear regression target
    # Labels come from columns 3 and 4, which NO fixture perturbs (denormal
    # rewrites columns 0-2), so `denormal` and `denormal_ftz` hand every lane
    # the same labels and their twin comparison is about the features alone.
    # Until 2026-08-29 this read columns 0 and 1 and the twins carried 2493
    # different labels, which made the classifier twins uncomparable.
    s01 = X[:, 3] + 0.5 * X[:, 4]
    y_clf = (s01 > np.median(s01)).astype(np.int32)
    w = np.random.default_rng(seed + 1).standard_normal(X.shape[1]).astype(np.float32)
    # FIXED-ORDER, ELEMENTWISE, NO BLAS. `X @ w` in float32 goes through the
    # host BLAS (Accelerate on the Mac, OpenBLAS on a Linux box) and the
    # accumulation order differs between them: measured 2026-08-29, 13876 of
    # 20000 targets differed on ONE Mac between `X @ w` and this loop, and
    # every regression lane read DIVERGENT Apple-vs-AMD for that reason alone.
    # A cross-vendor probe must hand every vendor the same bytes.
    y_reg = np.zeros(X.shape[0], dtype=np.float32)
    for j in range(X.shape[1]):
        y_reg = (y_reg + X[:, j] * w[j]).astype(np.float32)
    return y_clf, y_reg


FIXTURES = ["base", "ties", "hashed", "wide", "denormal", "denormal_ftz", "dupes", "odd", "negative"]

LANES = {}

#: LANES WHOSE INPUT CHANGED AFTER RECORDS WERE TAKEN (2026-09-15). A column
#: records these as `lane_revisions`; `diff` and the verify reference table
#: (python/mojolearn/_verify_reference.py) treat a column whose revision of a
#: lane is not this one as carrying NO cell for that lane, so its cells read
#: OWED to the next record rather than DIVERGENT against hashes of different
#: input. Bump a lane's revision when its fixture or vocabulary changes.
#:   tokenizer  synthetic-vocab-1: the GPT-2 table left the tree; the lane
#:              loads the synthetic vocabulary (python/mojolearn/
#:              _tokenizer_synthetic.py). Host integer code, so one hash per
#:              cell on every device class; the GPU columns are owed at the
#:              next release record.
#:   hdbscan, hdbscan-leaf  rows-4000-1 / rows-2000-1: the fixture row count
#:              came down from 6000 (2026-09-16, lane/identity-fixtures-light).
#:              6000 was inherited from the dbscan lane, not chosen. Measured
#:              across 6000/4000/3000/2000/1500/1000/750, the Boruvka round
#:              count these lanes HASH holds at 5 down to 4000 (excess-of-mass)
#:              and to 2000 (leaf), and falls to 4 below, so each lane sits at
#:              the smallest size that still reaches the fifth merge round.
#:              Every committed column hashed the 6000-row fit, so those cells
#:              read OWED to the next record rather than DIVERGENT.
#: THE 2026-09-16 SHRINK (lane/identity-fixtures-light). Eleven more lanes had
#: their fixture cut so a bitwise check stops costing a workload's time. Every
#: committed column hashed the OLD input, so each entry here makes those cells
#: read OWED to the next record instead of DIVERGENT against different bytes.
#: The cut is per lane and the arithmetic knobs (tree counts, depths, losses,
#: optimizer settings, seasonal period) are untouched; what came down is INPUT
#: SIZE and STEP COUNT.
#: AN ARITHMETIC CHANGE STALES A REFERENCE THE SAME WAY, and this dict is
#: where it has to be said (lane/umap-batch-fix, 2026-09-16). Every entry
#: above is an INPUT that moved. `umap` and `par-graph-umap` are the first
#: entries where the input is unchanged and the TRANSFORM moved: its sigma
#: floor, edge schedule, negative-sample counter and refinement epoch count
#: are per row now rather than per request, so every committed umap and
#: par-graph-umap cell hashes arithmetic this harness no longer performs. The
#: mechanism is the one that is needed either way, a column whose revision is
#: not this one carrying NO cell for the lane, so those cells read OWED to the
#: next record instead of DIVERGENT on a user's machine. `umap` also joins
#: host_surface.PUBLIC_PENDING_LANES as `stale reference` until that record.
LANE_REVISIONS = {
    "tokenizer": "synthetic-vocab-1",
    # rows: these lanes fitted the full 20,000 x 16 fixture
    "gbdt-parametric-losses": "rows-1500-1",
    # NaN moved onto the columns the fitted trees split on; until 2026-09-16
    # (lane/dead-arms) the two nan_mode arms hashed the SAME bytes at every size
    "gbdt-nan-modes": "rows-1500-nan-in-split-columns-2",
    "gbdt-lossguide-newtoncosine": "rows-1500-1",
    "gbdt-pair-logit": "rows-1500-1",
    # rows: O(n^2) neighbourhood work
    "hdbscan": "rows-4000-1",
    "hdbscan-leaf": "rows-2000-1",
    "spectral": "rows-512-1",
    # observations: 128 still carries many periods of the seasonal 12
    "holtwinters": "obs-128-1",
    # training steps: 3 AdamW steps -> 1, and two of those cuts REVERSED
    # 2026-09-16 (lane/shrink-floors) because the one-step cell could no longer
    # FAIL: byte-lm back to 2 steps, samba-untied-dropout-accum back to 3. Each
    # reversal moves those lanes' bytes again, so their committed cells read
    # OWED to the next record rather than DIVERGENT. That costs nothing today:
    # release/0.8.7 already requires all four record columns to be retaken, and
    # deferring it would cost a whole re-record later. The reason each number is
    # where it is now lives on the lane, in @floor(...), not in a document.
    "byte-lm": "steps-2-1",
    "byte-lm-resident": "shape-l1-d16-ff32-1",
    "samba": "steps-1-1",
    "samba-untied-dropout-accum": "steps-3-1",
    # sequence length: the (2, 16, 32) slab -> (2, 8, 32)
    # dt_limit (0.01, 0.1) -> (0.5, 0.9): every dt this lane makes was ABOVE
    # the old upper bound, so S9 returned `hi` for every value, the lower
    # bound was inert and a clamp pinned to a constant read IDENTICAL
    # (2026-09-16, lane/dead-arms). The LENGTH did not move; 8 is still 8.
    "mamba2-dtlimit": "seqlen-8-dtlimit-straddle-2",
    # THE NORMS THAT WERE BITWISE EQUAL (2026-09-16, lane/dead-arms). Two
    # same-shape RMSNorm weights initialised to ones are the same tensor, so
    # exchanging them was the identity function, and these lanes never train,
    # so no step could separate them. `_block_weights(near_one=...)` gives
    # each its own elementwise vector within an eighth of unity.
    "mamba3": "norms-near-one-1",
    "transformer": "norms-near-one-1",
    "transformer-window": "norms-near-one-1",
    # arithmetic, not input: UMAP.transform is row separable now
    "umap": "transform-row-separable-1",
    "par-graph-umap": "transform-row-separable-1",
}


# ------------------------------------------------------------- fixture floors
# THE SMALLEST A LANE'S FIXTURE MAY GET, declared ON the lane, with the reason
# attached, and enforced by `tools/fixture_floors.py` in the push gate.
#
# WHY THIS IS CODE. On 2026-09-16 an audit found two shrunken cells that can no
# longer FAIL (docs/lanes/LANE_STATUS_shrink-blindness-audit.md). One of them,
# samba-untied-dropout-accum, had a floor WRITTEN DOWN before it was cut:
# docs/lanes/LANE_STATUS_lane-identity-fixtures-light.md section 1f is titled
# "LEFT BIG" and says three steps is the floor because step 3 is the first that
# evaluates the cosine arm of the schedule. It was cut to one step anyway,
# docs/lanes/FIXTURE_SHRINK_SCOPE.md carried forward only the ROWS half of that
# reasoning, and docs/lanes/LANE_STATUS_lane-neural-shape-shrink.md then
# recorded as FACT that the third step was kept. Nobody lied. The reason and
# the number lived in a different file from the fixture, so a change never had
# to walk past them.
#
# HOW IT READS. Under `@lane(...)`, and above the body it constrains:
#
#     @lane("samba-untied-dropout-accum")
#     @floor(steps=(3, "why, ending in the measurement that set the number"))
#     def _(ml, X, yc, yr, Xh=None):
#         steps, rows = 3, 32          # the floored locals
#         ...
#         for k in range(steps): ...   # the site, which must READ them
#
# WHAT THE GATE REFUSES (tools/fixture_floors.py has the four rules and a
# --self-test that watches each one refuse a real mutation of this file):
#   * a floored local below its floor, printing the reason;
#   * a shrunk lane (one with a LANE_REVISIONS entry) that declares no floor
#     for a dimension the checker can find a site for, which is DERIVED from
#     LANE_REVISIONS rather than hand-listed;
#   * a site that stops reading the floored local, so the number cannot be
#     bypassed while the floor still looks satisfied;
#   * a reason too short to be a reason, or one citing no date, `docs/` path
#     or `lane/` name to trace it back to a measurement.
#
# A FLOOR IS NOT A CLAIM THAT THE NUMBER IS RIGHT. Only a measurement is, which
# is why every reason has to cite one. To move a floor, measure, then change it
# here, where the next person reads why it was where it was.

def floor(**dims):
    """Declare a fixture floor on the lane below. Each keyword is a dimension
    (`steps`, `rows`, `observations`, `batch`, `seqlen`) and a
    `(minimum, why)` pair, written out in full so the number and the reason
    cannot be separated.
    This records; `tools/fixture_floors.py` enforces, from the source, so the
    check needs no numpy and no bindings and runs in the cheapest gate."""
    def deco(fn):
        fn.fixture_floors = dict(getattr(fn, "fixture_floors", {}), **dims)
        return fn
    return deco


def lane_floors():
    """Every declared floor, derived from LANES at call time: lane -> dim ->
    (minimum, why). Never hand-listed."""
    return {name: dict(getattr(fn, "fixture_floors", {}))
            for name, fn in LANES.items() if getattr(fn, "fixture_floors", None)}


#: REVISIONS THAT ARE NOT A SIZE, and what they changed instead. LANE_REVISIONS
#: records every lane whose input moved, which is broader than "was shrunk". A
#: key that changed a SIZE says so in its first token (`rows-`, `obs-`,
#: `steps-`, `seqlen-`, `batch-`) and MUST carry a matching @floor; every other
#: key lands here with a sentence saying what moved. An entry whose key does
#: name a size is refused by name, so this cannot become a way of opting out.
NON_SIZE_REVISIONS = {
    "tokenizer": (
        "what changed is the VOCABULARY, not a size: the lane swapped the GPT-2 table for a 512-rank "
        "synthetic one (2026-09-16, lane/identity-fixtures-light), and a vocabulary is a constructor "
        "choice with no integer in the body to floor. The thing worth watching here is not a count but "
        "how thin the coverage is: docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 5f measured "
        "4088 single-byte tokens and 4 two-byte tokens over the 4096 fixture bytes, so the BPE merge "
        "loop runs four times and the endoftext branch never matches"),
    "byte-lm-resident": (
        "what changed is the model SHAPE, one block at d_model 16 instead of two at 32 (2026-09-16, "
        "lane/neural-shape-shrink), which is a property of the trainer rather than a count in the "
        "fixture. The lane's claim survives it by construction: it asserts the resident export equals "
        "the stateless gradient BYTE FOR BYTE at whatever shape both are built at. Its step count is "
        "floored on the lane; see docs/lanes/LANE_STATUS_lane-neural-shape-shrink.md"),
    "umap": (
        "arithmetic, not input: UMAP.transform became row separable (2026-09-16, "
        "lane/umap-batch-determinism), so the cell moved without any fixture size moving. The lane "
        "still fits 1024 rows, exactly as it did before; see "
        "docs/lanes/LANE_STATUS_lane-umap-batch-fix.md"),
    "mamba3": (
        "what changed is the norm WEIGHTS, not a size (2026-09-16, lane/dead-arms). `B_norm.weight` and "
        "`C_norm.weight` are the same shape and were both a vector of ones, so they were the SAME TENSOR "
        "and exchanging them on the way in was the identity function: the cell read 63de4bf6b9f8262a with "
        "and without the swap. This lane never trains, so no step count could separate them and no floor "
        "can hold this; they now come from `_block_weights(near_one=...)`. See "
        "docs/lanes/LANE_STATUS_dead-arms.md section 2"),
    "transformer": (
        "the same defect as `mamba3` (2026-09-16, lane/dead-arms): `input_layernorm.weight` and "
        "`post_attention_layernorm.weight` are the same shape and were both ones, so the swap was the "
        "identity function and the cell read 295d4e62d4c78b14 either way. Not a size, and this lane does "
        "not train. See docs/lanes/LANE_STATUS_dead-arms.md section 2"),
    "transformer-window": (
        "the same pair as `transformer`, through the same helper (2026-09-16, lane/dead-arms); the cell "
        "read 49ffb2316f238e6d with and without the swap. The shrink audit did not name this lane, which "
        "is why the census in docs/lanes/LANE_STATUS_dead-arms.md section 2 was run over every sequence "
        "block rather than the two it listed. Not a size"),
    "par-graph-umap": (
        "the same row-separable UMAP.transform change as the `umap` lane (2026-09-16, "
        "lane/umap-batch-determinism); this driver's fixture size did not move either. See "
        "docs/lanes/LANE_STATUS_lane-umap-batch-fix.md"),
}


# ---------------------------------------------------------------- record scope
# WHAT A RELEASE RECORD COVERS, and how often the Apple column may be taken.
# Both were implicit before 2026-09-16 (lane/identity-fixtures-light): the
# record's scope was "every lane this file defines", and how often to run the
# Apple column was prose in a handoff. Both are stated here, in code, because
# a rule that is not code is not a check.

#: LANES A RELEASE RECORD DOES NOT COVER. Every `par-*` lane, for three
#: measured reasons:
#:   1. COST OUT OF PROPORTION. The `par-*` lanes were the whole remaining AMD
#:      gap for 0.8.6: 28 of the 30 lanes that column never reached. The owed
#:      lanes are a contiguous tail starting exactly at `par-arima`, and two
#:      further 60-minute Hot Aisle leases got past none of them.
#:   2. SEVERAL CANNOT BE COVERED HONESTLY AT ALL.
#:      docs/lanes/LANE_STATUS_lane-cpu-training-par-wave3.md: par-byte-lm-
#:      model-pool and par-byte-lm-offload compare host arithmetic with itself
#:      under a pooled label, and 21 more are cooperative families whose split
#:      lives inside the GPU binding with no host restatement.
#:   3. IN A RECORD THEY ARE NOT EVEN THE TWO-DEVICE CLAIM. `_par_devices()`
#:      defaults to device 0, so every `par-*` cell in a release record is a
#:      ONE-device run. The drivers' actual claim, that two devices hash equal
#:      to one, is made by the dedicated two-device legs, not by the record.
#: This excludes them from a FULL-COLUMN run only. `--lanes` names lanes
#: explicitly and is never filtered, so the two-device par legs and the CPU
#: identity gate are unaffected.
RECORD_EXCLUDED_PREFIXES = ("par-",)


def record_excluded_lanes():
    """The lanes a full-column release record does not run, sorted. Derived
    from LANES at call time, never hand-listed."""
    return sorted(n for n in LANES if n.startswith(RECORD_EXCLUDED_PREFIXES))


def record_lanes():
    """The lanes a full-column release record DOES run."""
    return [n for n in LANES if not n.startswith(RECORD_EXCLUDED_PREFIXES)]


# Apple qualification is the installed-wheel release gate. The full identity
# matrix is a diagnostic, not an additional per-release obligation.
APPLE_RELEASE_RECORD_ENV = "MOJOLEARN_APPLE_RELEASE_RECORD"
APPLE_FULL_DIAGNOSTIC_ENV = "MOJOLEARN_APPLE_FULL_DIAGNOSTIC"
APPLE_COLUMN_LANE_LIMIT = 1


def _is_apple_gpu(host):
    """True on a Mac running the GPU (Metal) build. `host` is truthy only on
    a CPU-only install, where this rule does not apply at all."""
    return sys.platform == "darwin" and not host


def refuse_routine_apple_column(lanes, host, env=None):
    """Refuse broad Metal matrices unless explicitly requested as diagnostics."""
    env = os.environ if env is None else env
    if not _is_apple_gpu(host) or len(lanes) <= APPLE_COLUMN_LANE_LIMIT:
        return ""
    if env.get(APPLE_FULL_DIAGNOSTIC_ENV) == "1":
        return ""
    return (
        f"REFUSING: {len(lanes)} Apple lanes request a broad identity matrix. "
        "This is not required for a PyPI update: use the installed-wheel Metal "
        "checks in release-provenance.yml. Routine checks use CPU. "
        "For an Apple-specific failure, run one bounded diagnostic lane. "
        f"Only an intentional full investigation should set {APPLE_FULL_DIAGNOSTIC_ENV}=1; "
        f"{APPLE_RELEASE_RECORD_ENV} alone does not enable the full matrix."
    )


def stale_revision_lanes(record):
    """The LANE_REVISIONS lanes whose revision in `record` (a column JSON) is
    not this harness's, sorted."""
    have = record.get("lane_revisions") or {}
    return sorted(name for name, rev in LANE_REVISIONS.items() if have.get(name) != rev)


def lane(name):
    def deco(fn):
        LANES[name] = fn
        return fn
    return deco


class Fit(dict):
    """A lane's answer. It IS the training-row parts dict, hashed exactly as
    it has always been (the train column), and it carries the fitted
    estimator and the held-out probe as attributes so the SAME fit answers
    the infer and model columns. `probe` is either a callable taking an
    estimator and returning the arrays to hash on the held-out rows, or an
    `n/a:<reason>` string for an estimator with no out-of-sample method.
    `checks/gbdt_sub_byte_identity_check.py` calls a lane with four
    arguments and reads only the dict; that contract is unchanged."""
    est = None
    probe = "n/a:function"
    #: An `n/a:<reason>` for the model column of a lane whose saved file
    #: holds only what the lane passed in (lane/inference-embedding-ivf-
    #: cholesky, 2026-09-15): the Embedding table is the caller's weight, so
    #: its file hash is not arithmetic and no host sabotage can move it. The
    #: saved-table lookup is tools/classical_host_gate.py's embedding lane.
    model_na = None


class NumericalMismatch(AssertionError):
    """A completed numerical result disagrees with an independent oracle.

    Keep the measured parts for negative-control evidence while still failing
    ordinary callers and reference admission. This is not an exception waiver.
    """
    def __init__(self, message, parts):
        super().__init__(message)
        self.parts = dict(parts)


#: lane/catboost-parity (2026-09-19): GradientBoosting's SymmetricTree
#: defaults became CatBoost's GPU learner's (1000 iterations, the auto
#: learning rate, random_strength 1.0, the Bayesian bootstrap and the
#: small-iteration leaf count), and the adapters inherit them. Every GBDT lane
#: below was recorded under the EARLIER defaults, so each one now passes that
#: configuration explicitly through `_gbdt`: a default flip must never move a
#: recorded cell. What a lane already passes is kept. The leaf count is
#: pinned only for the losses whose own default is not 1 (the only ones the
#: small-iteration rule changes, catboost_options.cpp:120-240).
_GBDT_LEGACY_LEAF_ITERATIONS = {"Logloss": 10, "CrossEntropy": 10, "Poisson": 10,
                                "Tweedie": 20, "Expectile": 5, "PairLogit": 10}


def _gbdt_legacy_kw(cls, kw):
    """`kw` with the pre-2026-09-19 GBDT defaults filled in where absent."""
    out = dict(kw)
    name = getattr(cls, "__name__", "")
    loss = out.get("loss", {"GradientBoostingClassifier": "Logloss",
                            "GradientBoostingRegressor": "RMSE"}.get(name, "RMSE"))
    out.setdefault("learning_rate", 0.03)
    out.setdefault("random_strength", 0.0)
    out.setdefault("bootstrap_type", "No")
    if name in ("GradientBoostingClassifier", "GradientBoostingRegressor"):
        out.setdefault("l2_leaf_reg", 3.0)
    if (out.get("grow_policy", "SymmetricTree") == "SymmetricTree"
            and "leaf_estimation_iterations" not in out
            and "leaf_estimation_method" not in out
            and loss in _GBDT_LEGACY_LEAF_ITERATIONS):
        out["leaf_estimation_iterations"] = _GBDT_LEGACY_LEAF_ITERATIONS[loss]
    if loss in ("MAE", "Quantile", "MAPE"):
        # unset resolved False for these three until 2026-09-19, when their
        # CalcSampleQuantile constant landed and unset became their True
        out.setdefault("boost_from_average", False)
    return out


def _gbdt(cls, **kw):
    """`cls(**kw)` under the configuration the GBDT lanes were recorded at."""
    return cls(**_gbdt_legacy_kw(cls, kw))


def _fit(parts, est=None, probe="n/a:function", model_na=None):
    f = Fit(parts)
    f.est = est
    f.probe = probe
    f.model_na = model_na
    return f


# ---------------------------------------------------------------- lanes
# One per PUBLIC estimator, plus linalg and metrics. Each returns a hash of
# the things a user would read back, plus the fitted estimator and what to
# ask it on the held-out rows `Xh`. The held-out slice sizes mirror the
# training-row slice sizes the train column already uses, so the infer
# column costs what the train column costs.

@lane("rf-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, yc)
    predict, proba = m._predict_with_proba(X)
    return _fit(dict(predict=_h(predict), proba=_h(proba)),
                m, lambda e: e._predict_with_proba(Xh))


@lane("rf-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("et-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, yc)
    predict, proba = m._predict_with_proba(X)
    return _fit(dict(predict=_h(predict), proba=_h(proba)),
                m, lambda e: e._predict_with_proba(Xh))


@lane("et-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.ExtraTreesRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-symmetric")
def _(ml, X, yc, yr, Xh=None):
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="Logloss").fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-symmetric-eval")
def _(ml, X, yc, yr, Xh=None):
    """The symmetric Logloss fit WITH A HELD-OUT SET: their test cursor, the
    held-out curve, the Iter detector and `use_best_model`
    (lane/close-no-cpu-path-gbdt, 2026-09-20; the CPU route is
    gbdt/host/gbdt_oracle_eval.mojo, which closed the last Plain arm that
    refused an eval set by name). The eval set is the lane's own held-out
    slice with its labels.

    THREE FITS, AND THE SECOND TWO HAVE A SHAPE OF THEIR OWN FOR A REASON.

    `m` is gbdt-symmetric's fit exactly (20 depth-6 trees at `_gbdt`'s
    recorded options) plus an eval set, with the detector off and
    `use_best_model` off. Its `predict` and `proba` must therefore be
    gbdt-symmetric's, bit for bit, on every column: an eval set does not
    reach the learn cursor, the borders, the splits or the leaves on the
    Plain doc-parallel path, and if it ever does, the two lanes disagree.
    What is new in it is `test_loss_curve` alone.

    `od` and `sh` run 30 depth-7 trees at learning_rate 1.8 because AT THE
    RECORDED SHAPE THE DETECTOR NEVER FIRES AND THE SHRINK NEVER CUTS. That
    was measured, not assumed: at 20 depth-6 trees and rate 0.03 the
    held-out curve is still falling at the last tree on all nine fixtures,
    so `od` and `sh` came out byte-identical to `m` and the whole
    detector-and-shrink half of this lane was REACHED BUT INERT -- it hashed
    something, and it would have hashed the same something with the
    detector deleted. The sabotage arm proved it: under
    -D MOJOLEARN_GBDT_EVAL_SABOTAGE=1 the two curve parts moved on all nine
    fixtures and `od_stopped`, `od_best_iteration`, `shrunk_predict` and
    `shrunk_proba` did not move at all. The shape below overfits on purpose,
    and the two asserts are what keep it overfitting: a change that makes
    the curve monotone again reads REFUSED on this lane rather than passing
    with nothing measured."""
    ych = labels_for(Xh, HELDOUT_SEED)[0]
    base = dict(n_estimators=20, max_depth=6, loss="Logloss")
    m = _gbdt(ml.GradientBoosting, use_best_model=False, **base).fit(
        X, yc, eval_set=(Xh, ych))
    # the overfitting shape (measured: fires and cuts on all nine fixtures)
    stop = dict(n_estimators=30, max_depth=7, loss="Logloss",
                learning_rate=1.8, l2_leaf_reg=3.0)
    od = _gbdt(ml.GradientBoosting, od_type="Iter", od_wait=2,
               use_best_model=False, **stop).fit(X, yc, eval_set=(Xh, ych))
    sh = _gbdt(ml.GradientBoosting, use_best_model=True,
               best_model_min_trees=3, **stop).fit(X, yc, eval_set=(Xh, ych))
    if not od.stopped_early_ or len(od.test_loss_curve_) >= stop["n_estimators"]:
        raise RuntimeError(
            "gbdt-symmetric-eval: the Iter detector did not fire (trees="
            + str(len(od.test_loss_curve_)) + " of " + str(stop["n_estimators"])
            + ", stopped_early_=" + str(od.stopped_early_) + "). The detector"
            " half of this lane would hash the same bytes as the fit without"
            " one, which is not a measurement; retune the overfitting shape"
            " rather than record this.")
    sh_curve = np.asarray(sh.test_loss_curve_, dtype=np.float64)
    sh_cut = int(np.argmin(sh_curve)) + 1
    if not 0 < sh_cut < len(sh_curve):
        raise RuntimeError(
            "gbdt-symmetric-eval: ShrinkToBestIteration did not cut (argmin+1="
            + str(sh_cut) + " of " + str(len(sh_curve)) + " trees). Same"
            " problem as above, on the shrink instead of the detector.")
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)),
                     test_loss_curve=_h(np.asarray(m.test_loss_curve_, dtype=np.float64)),
                     learn_loss_curve=_h(np.asarray(m.loss_curve_, dtype=np.float64)),
                     best_iteration=_h(np.asarray([m.best_iteration_], dtype=np.int64)),
                     od_predict=_h(od.predict(X)),
                     od_test_loss_curve=_h(np.asarray(od.test_loss_curve_, dtype=np.float64)),
                     od_learn_loss_curve=_h(np.asarray(od.loss_curve_, dtype=np.float64)),
                     od_stopped=_h(np.asarray([od.stopped_early_, len(od.test_loss_curve_)], dtype=np.int64)),
                     od_best_iteration=_h(np.asarray([od.best_iteration_, sh.best_iteration_], dtype=np.int64)),
                     shrunk_predict=_h(sh.predict(X)),
                     shrunk_proba=_h(sh.predict_proba(X)),
                     shrunk_test_loss_curve=_h(sh_curve)),
                sh, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-depthwise")
def _(ml, X, yc, yr, Xh=None):
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, grow_policy="Depthwise",
                            loss="Logloss").fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-lossguide")
def _(ml, X, yc, yr, Xh=None):
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_leaves=32, grow_policy="Lossguide",
                            loss="Logloss").fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-rmse")
def _(ml, X, yc, yr, Xh=None):
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="RMSE").fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


#: The six non-default `feature_border_type`s (lane/catboost-parity); the
#: default, GreedyLogSum, is every other GBDT lane's border search.
_GBDT_BORDER_TYPES = ("Median", "Uniform", "UniformAndQuantiles", "MaxLogSum",
                      "MinEntropy", "GreedyMinEntropy")


@lane("gbdt-ordered")
def _(ml, X, yc, yr, Xh=None):
    """GradientBoosting(boosting_type='Ordered') at the recorded GBDT
    configuration of `_gbdt` (no bootstrap, no score noise): a Logloss fit
    (their four permutations, ten Newton steps per leaf) with an eval set,
    the Iter detector at wait 5 and use_best_model (their test cursor), and
    an RMSE fit (boost_from_average on, as their unset default), 20 trees of
    depth 6 each. The fold structure search, every (permutation, fold)
    cursor and the estimation permutation reach the train column through both
    fits' predictions and learn curves and the Logloss fit's held-out curve."""
    ych = labels_for(Xh, HELDOUT_SEED)[0]
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="Logloss",
              boosting_type="Ordered", od_type="Iter", od_wait=5
              ).fit(X, yc, eval_set=(Xh, ych))
    r = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="RMSE",
              boosting_type="Ordered").fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)),
                     test_loss_curve=_h(np.asarray(m.test_loss_curve_, dtype=np.float64)),
                     best_iteration=_h(np.asarray([m.best_iteration_], dtype=np.int64)),
                     rmse=_h(r.predict(X)),
                     rmse_loss_curve=_h(np.asarray(r.loss_curve_, dtype=np.float64))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-ordered-bayesian-noise")
def _(ml, X, yc, yr, Xh=None):
    """Ordered boosting at CatBoost's GPU defaults for the parts `_gbdt`
    pins elsewhere: the Bayesian bootstrap at temperature 1 on the quality
    slices and random_strength 1.0 with its model-length decay, one Logloss
    fit of 20 depth-6 trees with a NewtonCosine twin (Newton weights in the
    search planes). The learning rate is pinned to 0.03 and the leaf count to
    Logloss's ten, so the lane moves only with the bootstrap and the noise."""
    kw = dict(n_estimators=20, max_depth=6, loss="Logloss", boosting_type="Ordered",
              learning_rate=0.03, leaf_estimation_iterations=10)
    m = ml.GradientBoosting(**kw).fit(X, yc)
    nc = ml.GradientBoosting(score_function="NewtonCosine", **kw).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), newton_cosine=_h(nc.predict(X))),
                m, lambda e: (e.predict(Xh),))


@lane("gbdt-catboost-defaults")
def _(ml, X, yc, yr, Xh=None):
    """The SymmetricTree defaults this implementation takes from CatBoost's
    GPU learner (catboost 1.2.10), everything unset but the tree count and
    the loss: the auto learning rate off the GPU coefficient rows (read back
    as `learning_rate_`), the Bayesian bootstrap at temperature 1,
    random_strength 1 with its model-length decay, and the small-iteration
    leaf rule (one Newton step below 200 trees and 20 features). 20 trees,
    Plain (below 500 trees), Logloss, and the same fit through
    GradientBoostingClassifier, which defers every default. At 20 trees the
    auto rate is capped at 0.5 on every fixture, so a third fit, 50 depth-2
    trees on the first 2000 rows, reads the formula below its cap (0.475 on
    the GPU rows; 0.216 on the CPU rows the sabotage arm swaps in)."""
    m = ml.GradientBoosting(n_estimators=20, loss="Logloss").fit(X, yc)
    c = ml.GradientBoostingClassifier(n_estimators=20).fit(X, yc)
    a = ml.GradientBoosting(n_estimators=50, max_depth=2, loss="Logloss").fit(X[:2000], yc[:2000])
    lr = np.asarray([m.learning_rate_, a.learning_rate_], dtype=np.float64)
    return _fit(dict(predict=_h(m.predict(X)), learning_rate=_h(lr),
                     classifier_proba=_h(c.predict_proba(X)),
                     auto_rate_fit=_h(a.predict(X[:2000]))),
                m, lambda e: (e.predict(Xh),))


@lane("gbdt-bfa-quantile")
def _(ml, X, yc, yr, Xh=None):
    """boost_from_average at CatBoost's default for MAE, Quantile and MAPE
    (unset is True for them since 2026-09-19): the starting point is their
    CalcSampleQuantile constant with the delta adjust
    (`gbdt/metrics/sample_quantile.mojo`, host code the device fit and the
    CPU host path share). Plain MAE, Quantile at alpha 0.25, MAPE, and one
    Ordered MAE fit, 8 trees of depth 4 each."""
    y = _pos(yr)
    parts, first = {}, None
    for name, kw in (("MAE", dict(loss="MAE")),
                     ("Quantile", dict(loss="Quantile", loss_alpha=0.25)),
                     ("MAPE", dict(loss="MAPE")),
                     ("Ordered-MAE", dict(loss="MAE", boosting_type="Ordered"))):
        m = ml.GradientBoosting(n_estimators=8, max_depth=4, learning_rate=0.03,
                                random_strength=0.0, bootstrap_type="No", **kw).fit(X, y)
        parts[name] = _h(m.predict(X))
        first = first or m
    return _fit(parts, first, lambda e: (e.predict(Xh),))


@lane("gbdt-border-types")
def _(ml, X, yc, yr, Xh=None):
    """One symmetric Logloss fit per non-default feature_border_type (8
    trees of depth 4 at 32 borders, the recorded GBDT configuration of
    `_gbdt`), each hashed on the training AND held-out rows, so every
    type's borders reach the train column; the infer, model and batch
    columns probe the Median fit. Border selection is host code shared by
    the device fit and the CPU host path (`select_borders`)."""
    parts, first = {}, None
    for bt in _GBDT_BORDER_TYPES:
        m = _gbdt(ml.GradientBoosting, n_estimators=8, max_depth=4, border_count=32,
                  loss="Logloss", feature_border_type=bt).fit(X, yc)
        parts[bt] = _h(m.predict(X))
        parts[bt + "-heldout"] = _h(m.predict(Xh))
        first = first or m
    return _fit(parts, first, lambda e: (e.predict(Xh),))


def _km_probe(X, Xh):
    """The k-means lanes' infer probe (2026-09-15). `predict` on the TRAINING
    rows must be `labels_` bit for bit, because it is the fit's own final
    assignment (cluster/estimator.mojo::kmeans_predict and
    cluster/host/kmeans_oracle.mojo::host_kmeans_predict); where it is not,
    the probe raises naming the pair and only the infer cell reads REFUSED.
    `transform` (2026-09-15, cluster/estimator.mojo::kmeans_transform) is
    asked too: each cell is the fused kernel's cell, so `transform(X)` at
    `labels_` must be every training row's minimum bit for bit, and the
    held-out distances are hashed beside the held-out labels.
    Only the held-out rows are hashed, never the training-row output, and
    the train column hashes the fit's own attributes, so no train cell
    moves."""
    def probe(e):
        _same_bytes("predict(X)", e.predict(X), "labels_", e.labels_)
        t = np.asarray(e.transform(X))
        at_label = t[np.arange(t.shape[0]), np.asarray(e.labels_, dtype=np.int64)]
        _same_bytes("transform(X)[i, labels_[i]]", at_label, "transform(X).min(axis=1)", t.min(axis=1))
        return (e.predict(Xh), e.transform(Xh))
    return probe


@lane("kmeans")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KMeans(n_clusters=8, random_state=3).fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_)), m, _km_probe(X, Xh))


@lane("knn")
def _(ml, X, yc, yr, Xh=None):
    m = ml.NearestNeighbors(n_neighbors=8).fit(X[:4096])
    d, i = m.kneighbors(X[4096:4160])
    return _fit(dict(dist=_h(d), idx=_h(i)), m, lambda e: e.kneighbors(Xh[:64]))


@lane("knn-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KNeighborsClassifier(n_neighbors=8).fit(X[:4096], yc[:4096])
    return _fit(dict(predict=_h(m.predict(X[4096:4160])), proba=_h(m.predict_proba(X[4096:4160]))),
                m, lambda e: (e.predict(Xh[:64]), e.predict_proba(Xh[:64])))


@lane("knn-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KNeighborsRegressor(n_neighbors=8).fit(X[:4096], yr[:4096])
    return _fit(dict(predict=_h(m.predict(X[4096:4160]))), m, lambda e: (e.predict(Xh[:64]),))


@lane("dbscan")
def _(ml, X, yc, yr, Xh=None):
    # prediction_data=True (2026-09-15) copies the fit's core mask out and
    # moves no train byte; infer is DBSCAN.predict on 256 held-out rows, the
    # nearest core sample within eps (DEVIATION 2740).
    m = ml.DBSCAN(eps=0.9, min_samples=5, prediction_data=True).fit(X[:6000, :4])
    return _fit(dict(labels=_h(m.labels_)), m, lambda e: (e.predict(Xh[:256, :4]),))


@lane("pca")
def _(ml, X, yc, yr, Xh=None):
    m = ml.PCA(n_components=4).fit(X)
    return _fit(dict(components=_h(m.components_), variance=_h(m.explained_variance_), transform=_h(m.transform(X[:256]))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("pca-whiten")
def _(ml, X, yc, yr, Xh=None):
    # The whitened transform (the kde svc host lane, 2026-09-14): the same
    # fit as `pca` with `whiten=True`, so the whiten scale kernel and its
    # host restatement have a cell of their own on every column.
    m = ml.PCA(n_components=4, whiten=True).fit(X)
    return _fit(dict(components=_h(m.components_), variance=_h(m.explained_variance_), transform=_h(m.transform(X[:256]))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("tsvd")
def _(ml, X, yc, yr, Xh=None):
    m = ml.TruncatedSVD(n_components=4).fit(X)
    return _fit(dict(components=_h(m.components_), transform=_h(m.transform(X[:256]))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("ols")
def _(ml, X, yc, yr, Xh=None):
    m = ml.LinearRegression().fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("ridge")
def _(ml, X, yc, yr, Xh=None):
    m = ml.Ridge(alpha=1.0).fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("logistic")
def _(ml, X, yc, yr, Xh=None):
    m = ml.LogisticRegression(max_iter=50).fit(X, yc)
    return _fit(dict(coef=_h(m.coef_), proba=_h(m.predict_proba(X[:256]))), m, lambda e: (e.predict_proba(Xh[:256]),))


@lane("lasso")
def _(ml, X, yc, yr, Xh=None):
    m = ml.Lasso(alpha=0.01, max_iter=200).fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("elasticnet")
def _(ml, X, yc, yr, Xh=None):
    m = ml.ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=200).fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("svc")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SVC(C=1.0, kernel="rbf", max_iter=200).fit(X[:2000], yc[:2000])
    return _fit(dict(decision=_h(m.decision_function(X[2000:2256])), predict=_h(m.predict(X[2000:2256]))),
                m, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("kde")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KernelDensity(bandwidth=0.7).fit(X[:4096, :4])
    return _fit(dict(scores=_h(m.score_samples(X[4096:4352, :4]))), m, lambda e: (e.score_samples(Xh[:256, :4]),))


@lane("agglomerative")
def _(ml, X, yc, yr, Xh=None):
    # prediction_data=True (2026-09-15) keeps the training rows and moves no
    # train byte; infer is predict on 256 held-out rows, the single-linkage
    # rule (DEVIATION 2740).
    m = ml.AgglomerativeClustering(n_clusters=4, prediction_data=True).fit(X[:2000, :4])
    return _fit(dict(labels=_h(m.labels_)), m, lambda e: (e.predict(Xh[:256, :4]),))


@lane("spectral")
@floor(rows=(512, "512 rows is the SHARPER fixture, not merely the cheaper one: it resolves a smaller "
                  "relative move of column 0 than the 2000 it replaced, 1e-4 against 1e-3, measured "
                  "2026-09-16 (docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 4). Below 512 "
                  "nothing has been measured, and the eigenpair problem this lane hashes gets easier "
                  "as it gets smaller."))
def _(ml, X, yc, yr, Xh=None):
    # prediction_data=True (lane/spectral-predict, 2026-09-15) copies the
    # eigenpairs, degree scaling and centroids out of the fit and moves no
    # train byte; infer is predict on 256 held-out rows, the Nystrom
    # extension and the fit's k-means assignment (DEVIATION 2860).
    rows = 512                                   # FLOORED, see @floor above
    m = ml.SpectralClustering(n_clusters=4, random_state=3, prediction_data=True).fit(X[:rows, :4])
    return _fit(dict(labels=_h(m.labels_)), m, lambda e: (e.predict(Xh[:256, :4]),))


@lane("holtwinters")
@floor(observations=(128, "128 observations still carries ten periods of the seasonal 12, and resolves a "
                          "SMALLER move than the 512 it replaced, 1e-7 against 1e-5 (2026-09-16, "
                          "docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 4, where all 128 "
                          "observations were also shown live one at a time). Below two seasonal periods "
                          "the initial seasonal estimate has nothing to average over."))
def _(ml, X, yc, yr, Xh=None):
    observations = 128                           # FLOORED, see @floor above
    series = (np.cumsum(X[:observations, 0]) + 50.0).astype(np.float32)
    series = series - series.min() + 1.0     # positive, for the multiplicative path
    m = ml.ExponentialSmoothing(series, seasonal="additive", seasonal_periods=12).fit()
    # ExponentialSmoothing takes endog in the constructor and has no
    # predict(X): the only out-of-sample output is the forecast, and its
    # held-out axis is the horizon. The train column keeps forecast(24) as
    # it has always been; the infer column (2026-09-14) asks for
    # FORECAST_HORIZON steps through both return paths, the flat
    # single-series buffer and the `index=0` strided read.
    return _fit(dict(forecast=_h(m.forecast(24))), m,
                lambda e: _same_bytes("forecast(h)", e.forecast(FORECAST_HORIZON),
                                      "forecast(h, index=0)", e.forecast(FORECAST_HORIZON, index=0)))


def _mismatch_bytes(name_a, a, name_b, b):
    """`_same_bytes`'s comparison WITHOUT the raise: the message when the two
    disagree, None when they agree.

    A lane that has parts to hand should raise `NumericalMismatch` with them
    rather than a bare exception, or a negative control that WORKS is recorded
    as a refusal and a refusal is not a catch (lane/sabotage-sweep, 2026-09-17:
    `par-scaler`'s sabotage arm fired and the cell read REFUSED on both
    fixtures, so the lane stayed `declared` in the matrix while its arm was
    doing its job)."""
    ba, bb = np.asarray(a).ravel().tobytes(), np.asarray(b).ravel().tobytes()
    if ba == bb:
        return None
    n = sum(x != y for x, y in zip(ba, bb)) + abs(len(ba) - len(bb))
    return f"{name_a} and {name_b} differ: {n} bytes of {max(len(ba), len(bb))}"


def _same_bytes(name_a, a, name_b, b):
    """The forecasters' infer probe: two public entries the estimator
    documents as the same answer. Returns both for hashing when their bytes
    agree; raises, naming the pair and the byte count, when they do not, so
    the infer column reads REFUSED with the message instead of a hash that
    hides which of the two moved."""
    message = _mismatch_bytes(name_a, a, name_b, b)
    if message is not None:
        raise ValueError(message)
    return a, b


@lane("gemm-pinned")
def _(ml, X, yc, yr, Xh=None):
    a = X[:256].T.copy()                     # 16 x 256 or 17 x 256
    b = X[256:256 + 128].copy()              # 128 x d
    wide = np.ascontiguousarray(X[:4096, :].reshape(-1)[: 256 * 4096].reshape(256, 4096)) \
        if X.size >= 256 * 4096 else X[:256, :]
    c1 = ml.linalg.matmul(a.astype(np.float32), X[:256].astype(np.float32), identical=True)
    c2 = ml.linalg.matmul(wide.astype(np.float32),
                          np.ascontiguousarray(wide[:128].T).astype(np.float32),
                          identical=True) if wide.shape[1] == 4096 else c1
    return _fit(dict(small=_h(c1), wide=_h(c2)))


@lane("metrics")
def _(ml, X, yc, yr, Xh=None):
    # labels_ is a mojolearn Array since the NumPy-free change; the modulo
    # below and the metrics take ndarrays, so view it once.
    labels = np.asarray(ml.KMeans(n_clusters=4, random_state=3).fit(X[:3000, :4]).labels_)
    mt = ml.metrics
    return _fit(dict(
        accuracy=_h(np.float64(mt.accuracy_score(yc[:3000], (labels % 2).astype(np.int32)))),
        ari=_h(np.float64(mt.adjusted_rand_score(yc[:3000], labels))),
        vmeasure=_h(np.float64(mt.v_measure_score(yc[:3000], labels))),
        r2=_h(np.float64(mt.r2_score(yr[:3000], yr[:3000] * np.float32(0.9)))),
        silhouette=_h(np.float64(mt.silhouette_score(X[:3000, :4], labels))),
    ))


# ---------------------------------------------------------------- derived inputs
# The lanes below (2026-09-13) cover the estimators that were not in the
# probe, the ones whose natural input is not a feature matrix (a batch of
# series, a token stream, a (batch, length, d_model) activation slab) get it
# DERIVED FROM THE FIXTURE BYTES by the rules here, so every vendor is handed
# the same hostile values in a different shape. Anything that has to be the
# same on every box and is NOT part of the answer (weights, permutations, a
# radius) comes from the hashed stream or from fixed-order host arithmetic,
# never from a host BLAS or a host transcendental (labels_for's lesson).

def _hw(shape, seed, lo, hi):
    """A float32 tensor of `shape` from the hashed stream, uniform on
    [lo, hi); the seed is a string naming the lane and the tensor."""
    n = int(np.prod(shape))
    u = _hashed_uniform(n, 1, seed).reshape(-1)
    return np.ascontiguousarray((u * np.float32(hi - lo) + np.float32(lo)).astype(np.float32).reshape(shape))


def _seq(X, b, l, dm, skip=0):
    """A (b, l, dm) float32 activation slab, the fixture's values in
    row-major order, `skip` values in, the next b*l*dm of them. The
    `odd` fixture has 12345*17 values, enough for every slab asked for."""
    flat = np.ascontiguousarray(X).reshape(-1)
    return np.ascontiguousarray(flat[skip: skip + b * l * dm].reshape(b, l, dm)).astype(np.float32)


def _ids(X, b, l, vocab=256):
    """A (b, l) int32 token stream, the first b*l BYTES of the fixture
    (its float32 values viewed as bytes), so the `ties` fixture hands the
    language models a stream heavy in repeated bytes and `hashed` a flat
    one."""
    raw = np.frombuffer(np.ascontiguousarray(X).tobytes()[: b * l], dtype=np.uint8)
    return np.ascontiguousarray((raw.astype(np.int32) % vocab).reshape(b, l))


def _three_class(X):
    """Targets in 0..2 for the fixed 8-16-3 MLP, from the two columns
    no fixture perturbs (the same ones labels_for reads)."""
    return ((X[:, 3] > 0).astype(np.int32) + (X[:, 4] > 0).astype(np.int32)).astype(np.int32)


def _coded(X):
    """The FeatureFreq estimator wants dense categorical codes in its
    source columns, two code columns (a 2-way and a 4-way split on
    columns 4, 5 and 6 at their medians) in front of the first eight
    numeric columns, which vary in every fixture (`dupes` holds its
    constant and its zero column at the end)."""
    c0 = (X[:, 4] > np.median(X[:, 4])).astype(np.float32)
    c1 = (2 * (X[:, 5] > np.median(X[:, 5])).astype(np.int32)
          + (X[:, 6] > np.median(X[:, 6])).astype(np.int32)).astype(np.float32)
    return np.ascontiguousarray(np.column_stack([c0, c1, X[:, :8]]).astype(np.float32))


def _radius_for(index, queries, metric="euclidean"):
    """A radius scaled to the fixture, six tenths of the median distance
    from the query rows to the first index row, in fixed-order float64
    host arithmetic (elementwise, no BLAS), then rounded to float32. A
    fixed radius would return nothing on `ties` and everything on
    `denormal`. The distance is the lane's own metric (euclidean, manhattan
    or chebyshev; the Lp lane uses the euclidean radius, which admits at
    least as many neighbors at p = 3), because an L1 ball of the euclidean
    radius held no neighbor at all on `base` (2026-09-14)."""
    ref = index[0].astype(np.float64)
    acc = np.zeros(queries.shape[0], dtype=np.float64)
    for j in range(index.shape[1]):
        diff = np.abs(queries[:, j].astype(np.float64) - ref[j])
        if metric == "euclidean":
            acc = acc + diff ** 2
        elif metric == "manhattan":
            acc = acc + diff
        elif metric == "chebyshev":
            acc = np.maximum(acc, diff)
        else:
            raise ValueError(metric)
    d = np.sqrt(acc) if metric == "euclidean" else acc
    return float(np.float32(0.6 * np.median(d)))


def _ragged(result):
    """The arrays to hash for a ragged radius query. Per-row counts, then
    every distance and every index in row order (sorted within a row by
    the estimator, `sort_results=True`)."""
    dists, idx = result
    # Rows are read by index and length, the ragged container's documented
    # contract (`radius_neighbors`: "[i] and len() read the same;
    # np.asarray(container) does not"), never by iterating or converting
    # the container: an outsider's numpy 2.4.4 with the 0.8.5 wheel raised
    # "setting an array element with a sequence" on six fixtures under the
    # iterating spelling (2026-09-14, bench/results/verify_external/).
    n = len(idx)
    rows_i = [np.asarray(idx[i]).reshape(-1).astype(np.int64) for i in range(n)]
    rows_d = [np.asarray(dists[i], dtype=np.float32).reshape(-1) for i in range(n)]
    lens = np.asarray([r.size for r in rows_i], dtype=np.int64)
    dd = np.concatenate(rows_d) if rows_d else np.zeros(0, dtype=np.float32)
    ii = np.concatenate(rows_i) if rows_i else np.zeros(0, dtype=np.int64)
    return lens, dd, ii


def _byte_lm_params(shape):
    """Named tensors for the byte LM from the hashed stream, norms at one
    and everything else uniform on [-1/8, 1/8) (the fan-in scale of a
    32-wide model as a dyadic rational, so no host sqrt). The same on
    every fixture, because the fixture is the TOKEN stream, not the weights.
    Returns (named dict, flat registry-order vector)."""
    named = {}
    for name, shp in zip(shape.parameter_names, shape.parameter_shapes):
        if name.endswith("norm1_w") or name.endswith("norm2_w"):
            named[name] = np.ones(shp, dtype=np.float32)
        else:
            named[name] = _hw(shp, "byte-lm:" + name, -0.125, 0.125)
    flat = np.ascontiguousarray(np.concatenate([named[n].reshape(-1) for n in shape.parameter_names]))
    return named, flat


def _byte_lm_shape(ml, e):
    """The byte LM shape a fit ACTUALLY runs, never the default.

    Until 2026-09-16 every byte LM lane ran the shipped profile, so the
    parts below could construct `ByteLanguageModelConfig()` and be right.
    `byte-lm-resident` is now one block at d_model 16
    (lane/neural-shape-shrink), and a part that assumes the default reads
    REFUSED with `parameters must be float32 [34944]` instead of checking
    anything. `LanguageModelInference` and the host trainer expose `shape`;
    the GPU trainer carries it in `state_dict()['model_shape']`, which it
    writes only when the shape is not the shipped profile."""
    shape = getattr(e, "shape", None)
    if isinstance(shape, ml.ByteLanguageModelConfig):
        return shape
    try:
        moved = e.state_dict().get("model_shape")
    except Exception:
        moved = None
    return ml.ByteLanguageModelConfig(**moved) if moved else ml.ByteLanguageModelConfig()


def _block_weights(lane, shapes, ones=(), near_one=()):
    """A weight dict for a sequence block. `ones` names get a vector of ones,
    `near_one` names get 1 + hashed uniform on [-1/8, 1/8), and the rest are
    hashed uniform on [-1/8, 1/8).

    WHY `near_one` EXISTS (2026-09-16, lane/dead-arms). A norm weight of ones
    is the natural initialisation and it was what every norm here got. Where a
    block carries TWO norms OF THE SAME SHAPE, that made the two tensors
    BITWISE EQUAL, so exchanging them on the way in was the identity function
    and no lane could see it. Unlike the byte LM, these lanes never train, so
    nothing separates the two afterwards either and no number of steps is a
    remedy. Measured on the base fixture, one core:

        mamba3             B_norm.weight <-> C_norm.weight
                           63de4bf6b9f8262a -> 63de4bf6b9f8262a   BLIND
        transformer        input_layernorm <-> post_attention_layernorm
                           295d4e62d4c78b14 -> 295d4e62d4c78b14   BLIND
        transformer-window the same pair
                           49ffb2316f238e6d -> 49ffb2316f238e6d   BLIND

    with each tensor READ (a +0.25 on either member moves the cell), so this
    is a permutation the lane cannot see, not a tensor it never touches. Under
    `near_one` the same exchange is DETECTED in all three.

    The vector is elementwise, not a second constant, so it also separates a
    norm that scales per channel from one that scales by a single value, and
    it stays within an eighth of unity so no activation is crushed and no
    denormal is manufactured."""
    return {name: (np.ones(shp, dtype=np.float32) if name in ones
                   else (np.float32(1.0) + _hw(shp, f"{lane}:near_one:{name}", -0.125, 0.125)
                         ).astype(np.float32) if name in near_one
                   else _hw(shp, f"{lane}:{name}", -0.125, 0.125))
            for name, shp in shapes.items()}


#: DEVIATION 2712 instrumentation: `run()` sets this to "<lane>/<fixture>/<repeat>"
#: before every fit; with MOJOLEARN_IDENTITY_DUMP_DIR set, `_block_fit` saves
#: every array it hashes (and its inputs, and the carried state) as
#: <dir>/<lane>-<fixture>-<repeat>.npz under the keys tools/mamba2_step_probe.py
#: writes for its `lane` order, so `mamba2_step_probe.py diff` compares a
#: harness fit against a probe run or against the same fit on another box
#: element by element. Off (unset), nothing here runs.
_DUMP_TAG = ""


def _dump(arrays):
    d = os.environ.get("MOJOLEARN_IDENTITY_DUMP_DIR", "")
    if not d or not _DUMP_TAG:
        return
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, _DUMP_TAG.replace("/", "-") + ".npz")
    np.savez_compressed(path, **{"lane__" + k: np.ascontiguousarray(np.asarray(v)) for k, v in arrays.items()},
                        __info=np.asarray([f"tag={_DUMP_TAG}", "source=identity_break._block_fit"]))


def _block_fit(blk, x, g, state_kw):
    """The train column shared by the four sequence blocks. A stateless
    prefill, the same prefill into a fresh state followed by one decode
    step, and the backward pass (IDENTICAL only, by the blocks' own
    contract) with a fixture-derived cotangent."""
    y = np.asarray(blk.forward(x))
    st = blk.allocate_state(x.shape[0], **state_kw)
    y_state = np.asarray(blk.forward(x, st))
    state_after_prefill = {k: np.asarray(getattr(st, k)).copy() for k in ("h", "conv_window", "buffer_xbc", "buffer_dtraw")
                           if hasattr(st, k)}
    y_step = np.asarray(blk.step(np.ascontiguousarray(x[:, :1, :]), st))
    state_after_step = {k: np.asarray(getattr(st, k)).copy() for k in ("h", "conv_window", "buffer_xbc", "buffer_dtraw")
                        if hasattr(st, k)}
    grads = blk.backward(x, g)
    parts = dict(forward=_h(y), prefill=_h(y_state), step=_h(y_step),
                 backward=_h(*[np.asarray(grads[k]) for k in sorted(grads)]))
    _dump(dict(x=x, g=g, forward=y, prefill=y_state, step=y_step,
               **{f"state_after_prefill.{k}": v for k, v in state_after_prefill.items()},
               **{f"state_after_step.{k}": v for k, v in state_after_step.items()},
               **{f"backward.{k}": np.asarray(grads[k]) for k in sorted(grads)}))
    return parts


# ---------------------------------------------------------------- lanes (2026-09-13)
# The estimators that were not in the probe. Cross-vendor standing is per
# estimator and is NOT this tool's claim; where the estimator's own
# docstring says its identity card is one vendor or two, the lane says so
# and measures anyway.

@lane("svr")
def _(ml, X, yc, yr, Xh=None):
    """SVR's docstring says it makes no cross-vendor claim and a three-vendor
    card is owed. Sized like the svc lane."""
    m = ml.SVR(C=1.0, kernel="rbf", epsilon=0.1, max_iter=200).fit(X[:2000], yr[:2000])
    return _fit(dict(predict=_h(m.predict(X[2000:2256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("arima")
def _(ml, X, yc, yr, Xh=None):
    """Four series of 512 observations, the first four columns of the
    fixture, transposed. The fit through this lane is IDENTICAL x3 since the
    136-lane record. Like holtwinters there
    are no new rows to feed: the held-out axis is the horizon. The train
    column keeps forecast(24); the infer column (2026-09-14) asks for
    FORECAST_HORIZON steps through both public out-of-sample entries,
    `forecast(h)` and `predict(n_obs, n_obs + h)`, which `_arima_impl.py`
    says are the same answer; `_same_bytes` holds it to that, so a byte
    between them reads REFUSED with its name, never a quiet hash."""
    series = np.ascontiguousarray(X[:512, :4].T)
    m = ml.ARIMA(order=(1, 0, 0)).fit(series)
    return _fit(dict(ar=_h(m.ar_), mu=_h(m.mu_), sigma2=_h(m.sigma2_), forecast=_h(m.forecast(24))),
                m, lambda e: _same_bytes("forecast(h)", e.forecast(FORECAST_HORIZON),
                                         "predict(n_obs, n_obs + h)",
                                         e.predict(e.n_obs_, e.n_obs_ + FORECAST_HORIZON)))


@lane("gp")
def _(ml, X, yc, yr, Xh=None):
    """The dense Cholesky is n^2 memory, so 256 rows of four columns, with
    a white-noise term so duplicate rows (`ties`) still factor. Three
    columns since the 136-lane record (IDENTICAL x3 on every cell)."""
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
    m = ml.GaussianProcessRegressor(kernel=k).fit(X[:256, :4], yr[:256])
    mean, std = m.predict(X[256:320, :4], return_std=True)
    return _fit(dict(alpha=_h(m.alpha_), L=_h(m.L_), lml=_h(np.float64(m.log_marginal_likelihood_value_)),
                     mean=_h(mean), std=_h(std)),
                m, lambda e: e.predict(Xh[:64, :4], return_std=True))


def _gpc_three_classes(X):
    """Three balanced classes for the first 256 rows: the stable rank of the
    labels' own score (columns 3 and 4, `labels_for`'s rule) cut in thirds,
    so every fixture, `negative` and `ties` included, hands the one-vs-rest
    lane three classes."""
    s = (X[:256, 3] + np.float32(0.5) * X[:256, 4]).astype(np.float32)
    y3 = np.empty(256, dtype=np.int64)
    y3[np.argsort(s, kind="stable")] = (np.arange(256) * 3) // 256
    return y3


def _gpc_parts(m, q):
    fits = m.estimators_
    return dict(L=_h(*[e.L_ for e in fits]), pi=_h(*[e.pi_ for e in fits]), W_sr=_h(*[e.W_sr_ for e in fits]),
                lml=_h(np.array([e.log_marginal_likelihood_value_ for e in fits] +
                                [m.log_marginal_likelihood_value_], dtype=np.float64)),
                n_iter=_h(np.array([e.n_iter_ for e in fits], dtype=np.int64)),
                predict=_h(m.predict(q)), proba=_h(m.predict_proba(q)))


@lane("gpc")
def _(ml, X, yc, yr, Xh=None):
    """GaussianProcessClassifier, binary, optimizer=None (DEVIATION 1761):
    the Laplace fit's Newton loop, whose iteration count is on the card
    (DEVIATION 2830), sized like the gp lane. The model column is the saved
    classifier and its reload."""
    m = ml.GaussianProcessClassifier(kernel=ml.ConstantKernel(1.0) * ml.RBF(1.0)).fit(X[:256, :4], yc[:256])
    return _fit(_gpc_parts(m, X[256:320, :4]), m, lambda e: (e.predict(Xh[:64, :4]), e.predict_proba(Xh[:64, :4])))


@lane("gpc-multiclass")
def _(ml, X, yc, yr, Xh=None):
    """GaussianProcessClassifier one-vs-rest over three classes
    (DEVIATION 2833) with a Matern nu=1.5 kernel scaled by a constant."""
    k = ml.ConstantKernel(2.0) * ml.Matern(1.0, nu=1.5)
    m = ml.GaussianProcessClassifier(kernel=k).fit(X[:256, :4], _gpc_three_classes(X))
    return _fit(_gpc_parts(m, X[256:320, :4]), m, lambda e: (e.predict(Xh[:64, :4]), e.predict_proba(Xh[:64, :4])))


@lane("umap")
def _(ml, X, yc, yr, Xh=None):
    """Exact neighbor search is quadratic, so 1024 rows of eight columns
    and eight epochs. UMAP's docstring says fit certificates do not certify
    transform, which is what the infer column asks."""
    m = ml.UMAP(n_neighbors=8, n_components=2, n_epochs=8, random_state=3).fit(X[:1024, :8])
    return _fit(dict(embedding=_h(m.embedding_)), m, lambda e: (e.transform(Xh[:64, :8]),))


@lane("radius")
def _(ml, X, yc, yr, Xh=None):
    """Sized like the knn lane; the radius is scaled to the fixture by
    _radius_for and is part of the train column. Results sorted within a
    row so the hash asks about membership and order, not device order."""
    index, q = X[:4096], X[4096:4160]
    r = _radius_for(index, q)
    m = ml.RadiusNeighbors(radius=r).fit(index)
    lens, dd, ii = _ragged(m.radius_neighbors(q, sort_results=True))
    return _fit(dict(radius=_h(np.float32(r)), counts=_h(lens), dist=_h(dd), idx=_h(ii)),
                m, lambda e: _ragged(e.radius_neighbors(Xh[:64], sort_results=True)))


@lane("standard-scaler")
def _(ml, X, yc, yr, Xh=None):
    m = ml.StandardScaler().fit(X)
    t = m.transform(X[:256])
    return _fit(dict(mean=_h(m.mean_), var=_h(m.var_), scale=_h(m.scale_), transform=_h(t),
                     inverse=_h(m.inverse_transform(t))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("minmax-scaler")
def _(ml, X, yc, yr, Xh=None):
    m = ml.MinMaxScaler().fit(X)
    t = m.transform(X[:256])
    return _fit(dict(data_min=_h(m.data_min_), data_max=_h(m.data_max_), scale=_h(m.scale_), min=_h(m.min_),
                     transform=_h(t), inverse=_h(m.inverse_transform(t))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("gbdt-ordered-rmse")
def _(ml, X, yc, yr, Xh=None):
    """OrderedRMSE takes one explicit row permutation; it is the stable
    argsort of a hashed stream, the same on every box."""
    perm = np.argsort(_hashed_uniform(X.shape[0], 1, "ordered-permutation").reshape(-1), kind="stable")
    m = ml.OrderedRMSE(n_estimators=20, max_depth=6).fit(X, yr, permutation=perm)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-feature-freq")
def _(ml, X, yc, yr, Xh=None):
    """ExperimentalTwoLevelFeatureFreq, one depth-two tree over two coded
    source columns (_coded) and eight numeric columns."""
    m = ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7).fit(_coded(X), yr)
    return _fit(dict(predict=_h(m.predict(_coded(X)))), m, lambda e: (e.predict(_coded(Xh)),))


@lane("mlp")
def _(ml, X, yc, yr, Xh=None):
    """SmallMLPTrainer, the fixed 8-16-3 network. Three AdamW steps on
    three 64-row batches of the first eight columns, targets from
    _three_class, starting weights the surface test's arange rule. Train
    column is the three losses, the weights after the steps and the
    logits on the 256 training rows; the checkpoint is the model column."""
    w = [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
         for s in ((16, 8), (16,), (3, 16), (3,))]
    m = ml.SmallMLPTrainer(*w, data_schedule={"dataset": "identity_break", "order": "sequential"})
    Xm = np.ascontiguousarray(X[:256, :8])
    t = _three_class(X[:256])
    losses = [np.float64(m.train_step(Xm[64 * k:64 * (k + 1)], t[64 * k:64 * (k + 1)])["loss"]) for k in range(3)]
    return _fit(dict(loss=_h(np.asarray(losses)),
                     weights=_h(*[np.asarray(m.weights_[k]) for k in sorted(m.weights_)]),
                     logits=_h(np.asarray(m.predict_logits(Xm)))),
                m, lambda e: (np.asarray(_neural_inference(ml, "mlp", e).predict_logits(np.ascontiguousarray(Xh[:256, :8]))),))


@lane("byte-lm")
@floor(steps=(2, "TWO steps, not one, and the second one is the whole point. _byte_lm_params sets every "
                 "norm1_w and norm2_w to a vector of ONES, so at the input of the FIRST step they are "
                 "bitwise equal and a read-side exchange of them is the identity function: at one step "
                 "this cell CANNOT FAIL under that defect. They separate by 2.000e-03 after one step, so "
                 "step 2 is the first that can see it, measured blind at one step and detected at two "
                 "(2026-09-16, docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 2, reversed by "
                 "lane/shrink-floors)."))
def _(ml, X, yc, yr, Xh=None):
    """SmallByteLanguageModelTrainer (LanguageModelTrainer is the same
    class) at the default b2-l32-d32 profile, 34944 parameters. Two
    AdamW steps on two (2, 33) windows of the fixture bytes (_ids),
    weights from _byte_lm_params. The identity claim is per shape by the
    trainer's own contract; this is one shape."""
    shape = ml.ByteLanguageModelConfig()
    named, _ = _byte_lm_params(shape)
    m = ml.SmallByteLanguageModelTrainer(named, data_schedule={"dataset": "identity_break", "order": "sequential"},
                                         shape=shape)
    steps = 2                                    # FLOORED, see @floor above
    ids = _ids(X, steps * shape.batch, shape.length + 1)
    losses = [np.float64(m.train_step(ids[2 * k:2 * k + 2])["loss"]) for k in range(steps)]
    return _fit(dict(loss=_h(np.asarray(losses)), params=_h(np.asarray(m.parameters_)),
                     logits=_h(np.asarray(m.logits(ids[:2, :-1])))),
                m, lambda e: (np.asarray(e.logits(_ids(Xh, shape.batch, shape.length))),))


@lane("byte-lm-host-infer")
def _(ml, X, yc, yr, Xh=None):
    """LanguageModelInference, the CPU forward path, on the reference
    (unthreaded) arm, from the same starting weights as byte-lm. Its
    certificate is per CPU (docs/BYTE_LM_CPU_INFERENCE.md); this lane
    measures whatever CPU the box has."""
    shape = ml.ByteLanguageModelConfig()
    _, flat = _byte_lm_params(shape)
    m = ml.LanguageModelInference(flat, shape=shape, threaded=False)
    ids = _ids(X, shape.batch, shape.length + 1)
    return _fit(dict(loss_bits=_h(np.uint32(m.loss_bits(ids))), logits=_h(np.asarray(m.logits(ids[:, :-1]))),
                     next=_h(np.asarray(m.next_bytes(ids[:, :-1])))),
                m, lambda e: (np.asarray(e.logits(_ids(Xh, shape.batch, shape.length))),))


@lane("byte-lm-host-train")
def _(ml, X, yc, yr, Xh=None):
    """LanguageModelHostTrainer, one CPU training step (its docstring says not
    certified yet). It has no evaluation-only entry (`loss` IS a step), so
    the held-out probe is the loss bits of a second step on the held-out
    window."""
    shape = ml.ByteLanguageModelConfig()
    _, flat = _byte_lm_params(shape)
    m = ml.LanguageModelHostTrainer(flat, shape=shape)
    bits = m.train_step(_ids(X, shape.batch, shape.length + 1))
    return _fit(dict(loss_bits=_h(np.uint32(bits)), params=_h(np.asarray(m.parameters_)), m=_h(np.asarray(m.m_))),
                m, lambda e: (np.uint32(e.train_step(_ids(Xh, shape.batch, shape.length + 1))),))


# The sequence blocks: (2, 16, 32) slabs from the fixture (_seq), hashed
# weights at the smallest legal d_model (32, the Mamba-2/3 rule). Their
# docstrings say the IDENTICAL card is one vendor (transformer) or that
# broader backward qualification is open (Mamba); measured here on all.

#: THE SHAPES THE `language-model-config` LANE ASKS FOR, none of them the
#: shipped profile: an ODD length with one block and a small vocabulary, and
#: a three-block model with n_heads == n_kv (no grouped-query sharing) over a
#: vocabulary that is not a power of two. Chosen against what the other byte
#: LM lanes already run -- `byte-lm`, `byte-lm-host-infer` and
#: `byte-lm-host-train` all run the DEFAULT b2-l32-d32-h4-kv2-ff64-v256 and
#: `byte-lm-resident` one block at d_model 16 -- so every number the config
#: derives (the offsets, n_total, n_tensors and the profile's `.v3` suffix)
#: takes a value no other cell in any record has taken.
LM_CONFIG_SHAPES = (
    dict(batch=1, length=5, d_model=16, n_heads=2, n_kv=1, head_dim=8,
         intermediate=24, n_layers=1, vocab_size=61),
    dict(batch=3, length=7, d_model=24, n_heads=3, n_kv=3, head_dim=8,
         intermediate=40, n_layers=3, vocab_size=37),
)

#: Shapes the config must REFUSE and the words it must refuse them with, one
#: per clause of `ByteLanguageModelConfig.__post_init__`. Each is the DEFAULT
#: shape with the named fields replaced, so the only reason any of them is
#: refused is the clause it names.
LM_CONFIG_REFUSALS = (
    (dict(batch=0), "Byte-LM dimensions must be in [1, 2**20]"),
    (dict(batch=True), "requires integer dimensions"),
    (dict(length=1.5), "requires integer dimensions"),
    (dict(length=8193), "L <= 8192"),
    (dict(d_model=33), "DM = H*HD"),
    (dict(n_kv=3), "divisible by KV"),
    (dict(head_dim=9, d_model=36), "even HD"),
    (dict(length=8192, d_model=64, n_heads=32, n_kv=1, head_dim=2, intermediate=2),
     "signed 32-bit indexing"),
)


@lane("language-model-config")
def _(ml, X, yc, yr, Xh=None):
    """LanguageModelConfig (`python/mojolearn/language_model.py`, the public
    name of `ByteLanguageModelConfig`), the object that SELECTS a byte LM
    shape, at two shapes no other lane runs (LM_CONFIG_SHAPES).

    WHY THIS IS A LANE AND NOT ONLY A TEST. The class is a frozen dataclass
    and its own work is integer bookkeeping, so on its own it would belong in
    python/mojolearn/tests/. What makes it a cell is that its numbers are the
    only thing between a caller and the native code: `parameter_shapes` and
    `offsets` cut one flat float32 vector into the tensors a forward pass
    reads, `n_total` is the length that vector must have, and `profile` is
    the string `LanguageModelInference.__init__` holds the compiled binding's
    own answer to before it will run at all. An off-by-one in any of them is
    not a Python bug, it is a different model, and on a shape no lane asks
    for nothing would catch it. Until 2026-09-19 every byte LM cell in every
    record ran the shipped profile or `byte-lm-resident`'s one block.

    THE PARTS. `profiles` is what the COMPILED side calls each shape, read
    back through the binding's `byte_lm_config_profile` (the GPU binding
    computes it natively; on a CPU column `_byte_lm_trainer_host.py` routes
    it to the host binding's `byte_lm_host_profile`), so the cell records the
    native name and not Python's. `derived` is the config's own integer
    table -- the native shape, every offset, n_total, n_tensors and every
    parameter shape -- which is what a checkpoint reader and a weight packer
    both index with. `logits`, `loss` and `next` are a real forward at each
    shape through `LanguageModelInference` on the reference (unthreaded) arm,
    from `_byte_lm_params` weights at that shape and ids cut from the fixture
    bytes; they are what moves under a sabotage build. `flags` carries the
    LM_CONFIG_REFUSALS, the `to_dict` round trip,
    the shape refusals of `LanguageModelInference`, and the ALIAS itself:
    `LanguageModelConfig` must BE `ByteLanguageModelConfig` or the public
    name documents a class that is not the one that runs.

    SABOTAGE, AND WHAT IT CANNOT REACH. -D MOJOLEARN_HOST_SABOTAGE=1 on the
    byte LM host family (gemm_oracle's descending leaf, the arm the other
    byte LM host lanes read DIVERGENT under) moves the cell on both
    fixtures: MEASURED on the M4, one core, --repeats 2, 2026-09-19,
    `base` edb3d7f33a909fb1 -> 2b0060261bcb25ff and `ties`
    48d4577b6abf0a41 -> 6dadf23bf0853110, through `logits`
    (089290b669328bf1 -> 11fedffc40c2d6eb) and `loss` (342dcda55c6727c4 ->
    0b83c7ae1ebdf243).

    THREE PARTS DO NOT MOVE UNDER IT AND ONE MORE DID NOT. `profiles`,
    `derived` and `flags` are integers and a string with no float fold
    anywhere near them, so no build define can reach them; what guards those
    is the recorded hash itself, which reads DIVERGENT on every column at
    once if an offset or the profile's spelling changes. `next` did not move
    either, and that is a measurement and not a design: the arm perturbs the
    fold ORDER, the resulting logit changes are far below the gap between
    the top two candidates on these fixtures, and an argmax survives it. It
    is kept because it is the part a user reads, and because a real defect in
    the shape arithmetic would move it; it is not evidence of anything on its
    own."""
    B = ml._backend.binding("_mojolearn_byte_lm", "identical")
    profiles, derived, logits, losses, nexts, flags = [], [], [], [], [], []
    for kw in LM_CONFIG_SHAPES:
        cfg = ml.LanguageModelConfig(**kw)
        profiles.append(str(B.byte_lm_config_profile(list(cfg.native_shape))))
        derived.extend(cfg.native_shape)
        derived.extend(cfg.offsets)
        derived.extend((cfg.n_total, cfg.n_tensors, len(cfg.parameter_names)))
        for shp in cfg.parameter_shapes:
            derived.extend((len(shp),) + tuple(shp))
        _, flat = _byte_lm_params(cfg)
        m = ml.LanguageModelInference(flat, shape=cfg, threaded=False)
        ids = _ids(X, cfg.batch, cfg.length + 1, vocab=cfg.vocab_size)
        logits.append(np.asarray(m.logits(ids[:, :-1])).reshape(-1))
        losses.append(np.uint32(m.loss_bits(ids)))
        nexts.append(np.asarray(m.next_bytes(ids[:, :-1])).reshape(-1))
        # The constructor refuses unless the compiled profile IS the config's,
        # so reaching here is itself the agreement; the flags record it as a
        # number because a raise would read REFUSED and not DIVERGENT.
        flags.extend((m.profile == cfg.profile, str(m.shape.profile) == profiles[-1],
                      ml.LanguageModelConfig(**cfg.to_dict()) == cfg,
                      len(flat) == cfg.n_total, cfg.offsets[-1] == cfg.n_total,
                      len(cfg.parameter_shapes) == cfg.n_tensors))
    for kw, needle in LM_CONFIG_REFUSALS:
        flags.append(_refused(lambda kw=kw: ml.LanguageModelConfig(**kw), needle))
    default = ml.LanguageModelConfig()
    flags.extend((
        # THE ALIAS. `language_model.py` spells it
        # `LanguageModelConfig = ByteLanguageModelConfig`; if that ever
        # became a subclass or a wrapper, every checkpoint written under the
        # public name would carry a different class and nothing else here
        # would notice. python/mojolearn/tests/test_public_aliases.py holds
        # the same identity at import time, and the private helpers
        # `require_shape` and `state_shape` are held there too: they are not
        # on the public surface, so they are a test and not a cell.
        ml.LanguageModelConfig is ml.ByteLanguageModelConfig,
        default.profile == str(B.byte_lm_config_profile(list(default.native_shape))),
        _refused(lambda: ml.LanguageModelInference(np.zeros(3, dtype=np.float32),
                                                   shape=ml.LanguageModelConfig(**LM_CONFIG_SHAPES[0])),
                 "parameters must be float32"),
        _refused(lambda: ml.LanguageModelInference(np.zeros(8, dtype=np.float32),
                                                   shape=default.native_shape),
                 "shape must be a ByteLanguageModelConfig"),
    ))
    cfg0 = ml.LanguageModelConfig(**LM_CONFIG_SHAPES[0])
    return _fit(dict(profiles=_h(np.frombuffer("|".join(profiles).encode("ascii"), dtype=np.uint8)),
                     derived=_h(np.asarray(derived, dtype=np.int64)),
                     logits=_h(*logits), loss=_h(np.asarray(losses, dtype=np.uint32)),
                     next=_h(*nexts), flags=_h(np.asarray(flags, dtype=np.int64))),
                ml.LanguageModelInference(_byte_lm_params(cfg0)[1], shape=cfg0, threaded=False),
                lambda e: (np.asarray(e.logits(_ids(Xh, cfg0.batch, cfg0.length, vocab=cfg0.vocab_size))),))


@lane("mamba1")
def _(ml, X, yc, yr, Xh=None):
    dm, di, r = 32, 64, 2
    w = _block_weights("mamba1", {
        "norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4),
        "conv1d.bias": (di,), "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
        "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (dm, di)},
        ones=("norm.weight",))
    blk = ml.Mamba1Block(w)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), {})
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


@lane("mamba2")
def _(ml, X, yc, yr, Xh=None):
    dm, di, nh = 32, 64, 1
    cd, dip = di + 256, 2 * di + 256 + nh
    w = _block_weights("mamba2", {
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
        "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
        "out_proj.weight": (dm, di)},
        ones=("block_norm.weight", "norm.weight"))
    blk = ml.Mamba2Block(w)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), {})
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


@lane("mamba3")
def _(ml, X, yc, yr, Xh=None):
    dm, di, nh = 32, 64, 1
    dip = 2 * di + 256 + 3 * nh + 32
    w = _block_weights("mamba3", {
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "dt_bias": (nh,),
        "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128), "C_bias": (nh, 128),
        "D": (nh,), "out_proj.weight": (dm, di)},
        ones=("block_norm.weight",),
        # B_norm and C_norm are the same shape; at ones they were bitwise
        # equal and a swap of them was invisible (2026-09-16, lane/dead-arms)
        near_one=("B_norm.weight", "C_norm.weight"))
    blk = ml.Mamba3Block(w)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), {})
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


@lane("transformer")
def _(ml, X, yc, yr, Xh=None):
    dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
    w = _block_weights("transformer", {
        "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
        "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm), "v_proj.weight": (nkv * hd, dm),
        "o_proj.weight": (dm, nh * hd), "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
        "down_proj.weight": (dm, it)},
        # the two layernorms are the same shape; at ones they were bitwise
        # equal and a swap of them was invisible (2026-09-16, lane/dead-arms)
        near_one=("input_layernorm.weight", "post_attention_layernorm.weight"))
    blk = ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), dict(max_tokens=32))
    return _fit(parts, blk, lambda e: (np.asarray(_neural_inference(ml, "transformer", e).forward(_seq(Xh, 2, 16, dm))),))


# ---------------------------------------------------------------- low-bit weight lanes (2026-09-17)
# lane/identical-lowbit-inference. Each lane is its base lane with the 2-D
# projection weights PACKED (mojolearn.lowbit: bf16 bits, or int8 codes with
# a power-of-two exponent per row), materialized exactly on this column and
# run through the block's fp32 path; gemm/IDENTICAL_LOWBIT_CONTRACT.md. The
# cells therefore hash the packed model's answer, and the equality of that
# answer with the fp32 block on the materialized weights is asserted by
# python/mojolearn/tests/test_lowbit_weights.py on one box; across columns
# it is this harness's diff. The gemm lanes hash the two profiles directly.


def _lowbit_block_lane(ml, X, Xh, base, fmt):
    if base == "transformer":
        dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
        w = _block_weights(base, {
            "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
            "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm), "v_proj.weight": (nkv * hd, dm),
            "o_proj.weight": (dm, nh * hd), "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
            "down_proj.weight": (dm, it)},
            near_one=("input_layernorm.weight", "post_attention_layernorm.weight"))
        blk = ml.TransformerBlock(ml.lowbit.pack(w, fmt), n_heads=nh, n_kv_heads=nkv)
        state_kw = dict(max_tokens=32)
    elif base == "mamba1":
        dm, di, r = 32, 64, 2
        w = _block_weights(base, {
            "norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4),
            "conv1d.bias": (di,), "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
            "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (dm, di)},
            ones=("norm.weight",))
        blk = ml.Mamba1Block(ml.lowbit.pack(w, fmt))
        state_kw = {}
    elif base == "mamba2":
        dm, di, nh = 32, 64, 1                      # the mamba2 lane's dims
        cd, dip = di + 256, 2 * di + 256 + nh
        w = _block_weights(base, {
            "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
            "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
            "out_proj.weight": (dm, di)},
            ones=("block_norm.weight", "norm.weight"))
        blk = ml.Mamba2Block(ml.lowbit.pack(w, fmt))
        state_kw = {}
    else:
        dm = 32
        di = 2 * dm
        w = _block_weights(base, _mamba3_shapes(dm, di), ones=("block_norm.weight",),
                           near_one=("B_norm.weight", "C_norm.weight"))
        blk = ml.Mamba3Block(ml.lowbit.pack(w, fmt))
        state_kw = {}
    assert blk.weight_format == fmt, (base, fmt, blk.weight_format)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), state_kw)
    return _fit(parts, blk, lambda e: (np.asarray(_neural_inference(ml, base, e).forward(_seq(Xh, 2, 16, dm))),))


def _mamba3_shapes(dm, di):
    """The Mamba-3 block's weight shapes, the mamba3 lane's dict at nh = 1."""
    nh = 1
    dip = 2 * di + 256 + 3 * nh + 32
    return {
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "dt_bias": (nh,),
        "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128), "C_bias": (nh, 128),
        "D": (nh,), "out_proj.weight": (dm, di)}


@lane("transformer-bf16w")
def _(ml, X, yc, yr, Xh=None):
    """The transformer lane with bf16-stored projection weights."""
    return _lowbit_block_lane(ml, X, Xh, "transformer", "bfloat16")


@lane("transformer-int8w")
def _(ml, X, yc, yr, Xh=None):
    """The transformer lane with int8-stored projection weights."""
    return _lowbit_block_lane(ml, X, Xh, "transformer", "int8")


@lane("mamba1-bf16w")
def _(ml, X, yc, yr, Xh=None):
    return _lowbit_block_lane(ml, X, Xh, "mamba1", "bfloat16")


@lane("mamba1-int8w")
def _(ml, X, yc, yr, Xh=None):
    return _lowbit_block_lane(ml, X, Xh, "mamba1", "int8")


@lane("mamba2-bf16w")
def _(ml, X, yc, yr, Xh=None):
    return _lowbit_block_lane(ml, X, Xh, "mamba2", "bfloat16")


@lane("mamba2-int8w")
def _(ml, X, yc, yr, Xh=None):
    return _lowbit_block_lane(ml, X, Xh, "mamba2", "int8")


@lane("mamba3-bf16w")
def _(ml, X, yc, yr, Xh=None):
    return _lowbit_block_lane(ml, X, Xh, "mamba3", "bfloat16")


@lane("mamba3-int8w")
def _(ml, X, yc, yr, Xh=None):
    return _lowbit_block_lane(ml, X, Xh, "mamba3", "int8")


def _lowbit_mlp_lane(ml, X, Xh, fmt):
    """The mlp lane's network with its two matrices stored packed: the
    trainer is built on the materialized weights and asked for logits only
    (no step), so the cell is the packed model's answer."""
    w = [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
         for s in ((16, 8), (16,), (3, 16), (3,))]
    packed = ml.lowbit.pack({"weight1": w[0], "weight2": w[2]}, fmt)
    f32, got = ml.lowbit.unpack(packed)
    assert got == fmt
    m = ml.SmallMLPTrainer(f32["weight1"], w[1], f32["weight2"], w[3],
                           data_schedule={"dataset": "identity_break", "order": "sequential"})
    Xm = np.ascontiguousarray(X[:256, :8])
    return _fit(dict(logits=_h(np.asarray(m.predict_logits(Xm))),
                     weights=_h(*[np.asarray(f32[k]) for k in ("weight1", "weight2")])),
                m, lambda e: (np.asarray(_neural_inference(ml, "mlp", e).predict_logits(np.ascontiguousarray(Xh[:256, :8]))),))


@lane("mlp-bf16w")
def _(ml, X, yc, yr, Xh=None):
    return _lowbit_mlp_lane(ml, X, Xh, "bfloat16")


@lane("mlp-int8w")
def _(ml, X, yc, yr, Xh=None):
    return _lowbit_mlp_lane(ml, X, Xh, "int8")


def _lowbit_samba_lane(ml, X, Xh, fmt):
    """The samba lane's stack with every 2-D registry tensor stored packed,
    forward logits only (no step)."""
    cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)
    seed = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3)
    w = {n: np.asarray(v).copy() for n, v in seed.parameters().items()} if hasattr(seed, "parameters") \
        else {n: np.asarray(seed.arrays[n]).copy() for n in seed.names}
    packed = ml.lowbit.pack(w, fmt)
    f32, got = ml.lowbit.unpack(packed)
    assert got == fmt
    m = ml.SambaStack(cfg, weights=f32, lr=1e-3)
    ids = _ids(X, 2, 17)
    logits = np.asarray(m.forward(ids[:, :-1]))
    return _fit(dict(logits=_h(logits)), m,
                lambda e: (np.asarray(_neural_inference(ml, "samba", e).forward(_ids(Xh, 2, 17)[:, :-1])),))


@lane("samba-bf16w")
def _(ml, X, yc, yr, Xh=None):
    return _lowbit_samba_lane(ml, X, Xh, "bfloat16")


@lane("samba-int8w")
def _(ml, X, yc, yr, Xh=None):
    return _lowbit_samba_lane(ml, X, Xh, "int8")


@lane("gemm-bf16")
def _(ml, X, yc, yr, Xh=None):
    """`mojolearn.identical.gemm.bf16f32.v1` on the gemm-pinned lane's
    slices: a float32 left operand times a bf16 right operand, and both
    operands bf16, NN and NT."""
    a = np.ascontiguousarray(X[:256]).astype(np.float32)          # 256 x d
    b = np.ascontiguousarray(X[256:256 + 128]).astype(np.float32)  # 128 x d
    bb = np.asarray(ml.linalg.to_bf16(b))
    c1 = ml.linalg.matmul_bf16(a, bb, transpose_b=True)            # 256 x 128
    c2 = ml.linalg.matmul_bf16(np.asarray(ml.linalg.to_bf16(a)), bb, transpose_b=True)
    c3 = ml.linalg.matmul_bf16(np.ascontiguousarray(a[:64].T), np.asarray(ml.linalg.to_bf16(a[64:128])))  # d x d, NN
    return _fit(dict(nt=_h(c1), both=_h(c2), nn=_h(c3), bits=_h(bb)))


@lane("gemm-int8")
def _(ml, X, yc, yr, Xh=None):
    """`mojolearn.identical.gemm.int8i32.v1` on the same slices: the codes
    and exponents the quantizer produces and the dequantized product."""
    a = np.ascontiguousarray(X[:256]).astype(np.float32)
    b = np.ascontiguousarray(X[256:256 + 128]).astype(np.float32)
    qa, ea = ml.linalg.quantize_int8(a)
    qb, eb = ml.linalg.quantize_int8(b)
    c = ml.linalg.matmul_int8((qa, ea), (qb, eb))
    return _fit(dict(codes=_h(np.asarray(qa), np.asarray(qb)), exps=_h(np.asarray(ea), np.asarray(eb)),
                     product=_h(c), dequant=_h(ml.linalg.dequantize_int8(qb, eb))))


def _lane_refuses(fn, text):
    """True when `fn()` raises with `text` in the message. A lane records a
    refusal as a NUMBER and never as an assertion: a raise reads REFUSED,
    which no column total can tell from a pass (the tokenizer lane's note,
    2026-09-15)."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return text in str(exc)
    return False


@lane("saved-model-host-infer")
def _(ml, X, yc, yr, Xh=None):
    """THE PUBLIC SAVED-MODEL CPU INFERENCE SURFACE by its own names:
    `mojolearn.HostForest`, `mojolearn.HostGBDT`, `mojolearn.host_predict`
    and `mojolearn.host_predict_proba`. Lane lane/laneless-public-classes,
    2026-09-19.

    WHAT WAS ALREADY TRUE, AND WHAT WAS NOT. `tools/forest_host_gate.py`
    compares `host_model(<file>).predict` and `.predict_proba` against a
    SHA-256 a GPU box recorded, over 24 committed fixtures (three vendors x
    eight kinds under `bench/results/forest_host/`); that gate is real and it
    passes -- re-run on the M4 at this commit, 8 Apple fixtures IDENTICAL.
    docs/COVERAGE_AUDIT_2026-09-18.md is right that HostForest and HostGBDT
    are not two unimplemented algorithms. TWO THINGS IT DOES NOT COVER.
    First, `host_predict` and `host_predict_proba`, the one-call entries the
    package exports, are reached by no gate and no lane; the gate calls
    `host_model`. Second, and this is the one that costs coverage: the gate
    REFUSES a `parallel_groves` archive by name (`tools/forest_host_gate.py`,
    "the host engine is sequential"), a sentence that stopped being true on
    lane/forest-groves-cpu-and-speed (2026-09-17) when
    `core/forest_host_groves.mojo` and `forest_host_groves_prepare/predict/
    release` landed. So the groves HOST engine -- the 32 fixed tree groups
    and the 16/8/4/2/1 fold that a groves-saved model infers with on a CPU --
    had no recorded evidence of any kind. This lane runs all three.

    THE MODELS. The `rf-clf` lane's RandomForestClassifier, the
    `gbdt-symmetric` lane's GradientBoosting and a RandomForestRegressor
    saved with `inference_engine="parallel_groves"`, each written with
    `save` and read back with `HostForest.from_file` / `HostGBDT.from_file`.
    Same fits, same fixture, so a cell here and the cell in those lanes are
    the same arithmetic reached two ways.

    THE PARTS. `rf`, `gbdt` and `groves` are the host predictions (and the
    two probability columns) on training rows. `sha` is each host model's
    `model_sha256`, the bytes it will predict with, which moves if the
    ARCHIVE moved even where the predictions round to the same answer.
    `flags` holds what the classes must agree about: that each host model
    answers exactly what the estimator that saved it answers, that
    `host_predict`/`host_predict_proba` on the same FILE answer exactly what
    the loaded object answers, that the engine each archive names is the
    engine the model reports, and the tree counts and class lists.

    THE INFER COLUMN is the held-out probe of the same three host models, so
    the CPU-inference claim is made on rows no forest saw. The MODEL column
    is `n/a:no-save`: a host model is a reader, it writes no file.

    SABOTAGE. The forest family's define is `MOJOLEARN_FOREST_HOST_SABOTAGE`,
    NOT the generic `MOJOLEARN_HOST_SABOTAGE`, which reaches nothing in
    `core/forest_host_predict.mojo`, `core/gbdt_host_predict.mojo` or
    `core/forest_host_groves.mojo`; a column built with the generic one alone
    reads this cell UNMOVED, which a column total cannot tell from a pass.
    Under the family's define all three engines move (the groves engine
    carries a second arm of its own, `MOJOLEARN_FOREST_GROVES_SABOTAGE`).
    MEASURED on the M4, one core, `--repeats 2`, ALL NINE FIXTURES
    (`bench/results/identity_break/2026-09-19_laneless-public-classes/`):
    the clean prebuilt host set against `-D MOJOLEARN_FOREST_HOST_SABOTAGE=1`
    moves 9 of 9 cells (`base` 3918f891959b852a -> 4f407d8f1968fd09) through
    `rf`, `gbdt`, `groves` and `flags`; the same set against
    `-D MOJOLEARN_HOST_SABOTAGE=1` ALONE moves 0 of 9 and every part reads
    the clean value. `sha` does not move under either, and that is correct:
    it hashes the ARCHIVE the estimator wrote, which the host binding only
    reads."""
    rows = np.ascontiguousarray(X[:4096])
    held = np.ascontiguousarray(Xh[:512])
    rf = ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, yc)
    gb = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss").fit(X, yc)
    gr = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7,
                                  inference_engine="parallel_groves").fit(X, yr)
    with tempfile.TemporaryDirectory(prefix="ib-saved-model-host-") as d:
        paths = {}
        for tag, est in (("rf", rf), ("gbdt", gb), ("groves", gr)):
            paths[tag] = os.path.join(d, f"{tag}.npz")
            est.save(paths[tag])
        hrf = ml.HostForest.from_file(paths["rf"])
        hgb = ml.HostGBDT.from_file(paths["gbdt"])
        hgr = ml.HostForest.from_file(paths["groves"])
        # THE ONE-CALL ENTRIES, on the same files: `host_predict(path, X)` is
        # `host_model(path).predict(X)` and must be its bytes, not a second
        # arithmetic. A fresh load per call is the point -- a user calls these
        # without holding a model.
        one = dict(rf=np.asarray(ml.host_predict(paths["rf"], held)),
                   rf_proba=np.asarray(ml.host_predict_proba(paths["rf"], held)),
                   gbdt=np.asarray(ml.host_predict(paths["gbdt"], held)),
                   gbdt_proba=np.asarray(ml.host_predict_proba(paths["gbdt"], held)),
                   groves=np.asarray(ml.host_predict(paths["groves"], held)))
    flags = np.array([
        # the host model answers what the estimator that saved it answers
        np.asarray(hrf.predict(held)).tobytes() == np.asarray(rf.predict(held)).tobytes(),
        np.asarray(hrf.predict_proba(held)).tobytes() == np.asarray(rf.predict_proba(held)).tobytes(),
        np.asarray(hgb.predict(held)).tobytes() == np.asarray(gb.predict(held)).tobytes(),
        np.asarray(hgb.predict_proba(held)).tobytes() == np.asarray(gb.predict_proba(held)).tobytes(),
        np.asarray(hgr.predict(held)).tobytes() == np.asarray(gr.predict(held)).tobytes(),
        # the one-call entries answer what the loaded object answers
        one["rf"].tobytes() == np.asarray(hrf.predict(held)).tobytes(),
        one["rf_proba"].tobytes() == np.asarray(hrf.predict_proba(held)).tobytes(),
        one["gbdt"].tobytes() == np.asarray(hgb.predict(held)).tobytes(),
        one["gbdt_proba"].tobytes() == np.asarray(hgb.predict_proba(held)).tobytes(),
        one["groves"].tobytes() == np.asarray(hgr.predict(held)).tobytes(),
        # the engine the archive names is the engine the model reports
        hrf.inference_engine == "sequential", hgb.inference_engine == "sequential",
        hgr.inference_engine == "parallel_groves",
        hrf.estimator == "RandomForestClassifier", hgb.estimator == "GradientBoosting",
        hgr.estimator == "RandomForestRegressor",
        hrf.n_trees == 16, hgr.n_trees == 16, hgb.n_trees == gb.n_estimators,
        list(hrf.classes_) == list(rf.classes_),
        hrf.n_features_in_ == rf.n_features_in_,
        # a regressor archive has no predict_proba, and it refuses BY NAME
        _lane_refuses(lambda: hgr.predict_proba(held), "has no predict_proba"),
    ], dtype=np.int64)
    shas = np.frombuffer(b"".join(bytes.fromhex(m.model_sha256())
                                  for m in (hrf, hgb, hgr)), dtype=np.uint8)
    return _fit(dict(rf=_h(np.asarray(hrf.predict(rows)), np.asarray(hrf.predict_proba(rows))),
                     gbdt=_h(np.asarray(hgb.predict(rows)), np.asarray(hgb.predict_proba(rows))),
                     groves=_h(np.asarray(hgr.predict(rows))),
                     sha=_h(shas), flags=_h(flags)),
                hrf, lambda e: (np.asarray(e.predict(held)), np.asarray(e.predict_proba(held)),
                                np.asarray(hgb.predict(held)), np.asarray(hgb.predict_proba(held)),
                                np.asarray(hgr.predict(held))))


#: THE OPERAND `lowbit-conversions` CONVERTS: a 2-D block of the fixture,
#: the only rank `pack_one` takes. 64 x 16 is 1,024 elements, which reshape
#: into the 1-D and 3-D bit buffers `widen_bf16` is asked for -- the ranks
#: the model loader hands it for a norm weight and an embedding table and
#: that `pack_one` refuses by name.
LOWBIT_ROWS, LOWBIT_COLS = 64, 16


def _lowbit_operands(A):
    """(2-D float32 block, the bf16 bit buffer reshaped 1-D, and 3-D) from a
    fixture slab. The reshapes are of the SAME bits, so a shape that moved
    is visible in `widened` without a second conversion."""
    return np.ascontiguousarray(A[:LOWBIT_ROWS, :LOWBIT_COLS]).astype(np.float32)


@lane("lowbit-conversions")
def _(ml, X, yc, yr, Xh=None):
    """THE LOW-BIT CONVERSION SEAMS THEMSELVES (`python/mojolearn/lowbit.py`,
    contract `gemm/IDENTICAL_LOWBIT_CONTRACT.md` clauses L-1 through L-6), as
    a user reaches them: `lowbit.pack_one`, `lowbit.materialize_one`,
    `lowbit.widen_bf16`, `lowbit.format_of`, `lowbit.is_packed` and
    `linalg.from_bf16`. Lane lane/laneless-public-classes, 2026-09-19.

    WHY A LANE AND NOT ONLY THE `*-bf16w` LANES. Ten lanes already run blocks
    whose weights were packed, but every one of them hashes a BLOCK OUTPUT:
    a logit, a hidden state, a GEMM product. A conversion defect that the
    following fp32 GEMM happens to absorb -- a bf16 rounded down instead of
    to nearest even, an int8 row exponent off by one -- is a DIFFERENT MODEL
    on disk and those cells can still agree. What is hashed here is the
    STORED BYTES: the bf16 bit buffer, the int8 codes and their per-row
    exponents, and the float32 each materializes back to.

    THE PARTS. `bf16_bits` and `int8` are what `pack_one` writes, which is
    what a low-bit checkpoint holds. `materialized` is `materialize_one` of
    each, the exact float32 an inference class actually computes with.
    `widened` is `widen_bf16` on the SAME bits reshaped 1-D and 3-D, the
    ranks `pack_one` refuses and the model loader needs. `f32_copy` is
    `pack_one(x, "float32")`, the identity arm, which must be x's bytes.
    `linalg` is `linalg.from_bf16`, the public linalg spelling of the same
    widening. `flags` carries the two predicates (`is_packed`, `format_of`,
    including the refusal of a dict that mixes formats), the shape and
    format each packed object reports, the refusals `pack_one` owes, and
    the four AGREEMENTS BETWEEN SPELLINGS below.

    THE THREE SPELLINGS, AND THE SILENT FALLBACK. `lowbit.py` materializes
    through the linalg extension's kernels on a GPU column, through
    `_mojolearn_linalg_host` on a CPU column, and through a pure-Python
    integer construction when neither loads -- and `_conversion_backend()`
    SWALLOWS the exception that chooses the third. So a column whose linalg
    binding refused to load (a sabotage build outside the gate, a missing
    .so) computes every conversion in Python and reads exactly like a
    column that ran the compiled seam. `flags` therefore carries
    `_conversion_backend() == "python"` as a number: it is 0 on a GPU column
    and 0 on a CPU column with the binding, so it is vendor-independent, and
    a column that fell back reads DIVERGENT against every other. The four
    agreement flags hold the spelling that RAN to the pure-Python one, which
    is the equality `python/mojolearn/tests/test_lowbit_weights.py` asserts
    on one box and this cell asserts on every column.

    SABOTAGE, AND THE ARM THAT DID NOT EXIST UNTIL THIS LANE.
    `-D MOJOLEARN_HOST_SABOTAGE=1` on the linalg family reaches
    `gemm_oracle`'s leaf and `gemm_int8_oracle`'s dequantized cell and
    NOTHING ELSE: `widen_bf16`, `narrow_bf16`, `quantize_rows_int8` and
    `dequantize_rows_int8` in `gemm/host/gemm_lowbit_oracle.mojo` -- the
    whole of L-1 through L-6, and every byte this lane hashes -- carried no
    arm at all. Measured: with the family define alone this cell is
    UNMOVED. So the four seams were given one, `MOJOLEARN_LOWBIT_CONVERT_SABOTAGE`
    (a low-bit flip on a bf16 pattern and an int8 code, the fp32 value flip
    on a widened or dequantized float32, so no input can make any of them a
    no-op), named for the linalg family in
    `host_surface.GATE_SABOTAGE_OWN_DEFINES` so the gate's sabotage set
    carries it beside the family one. MEASURED on the M4, one core,
    `--repeats 2`, ALL NINE FIXTURES
    (`bench/results/identity_break/2026-09-19_laneless-public-classes/`):
    the clean prebuilt host set against a linalg binding built with BOTH
    defines moves 9 of 9 cells (`base` 0ff2da2cf430c48a -> d58be6e94de78728)
    through `bf16_bits`, `int8`, `materialized`, `widened`, `linalg` and
    `flags`; against `-D MOJOLEARN_HOST_SABOTAGE=1` ALONE it moves 0 of 9
    and every part reads the clean value, which is the measurement the
    paragraph above reports. `f32_copy` does not move under either and
    cannot: `pack_one(x, "float32")` is a copy that touches no binding.

    A clean build of the edited oracle changes no shipping bit. The 15 other
    linalg and low-bit cells (`gemm-bf16`, `gemm-int8`, `gemm-pinned`,
    `gemm-transposed`, `cholesky` and the ten `*-bf16w`/`*-int8w` lanes) hash
    exactly what the pre-edit prebuilt binding hashed, base fixture, measured
    in the same directory."""
    from mojolearn import lowbit as _lb
    a2 = _lowbit_operands(X)
    bf = ml.lowbit.pack_one(a2, "bfloat16")
    q8 = ml.lowbit.pack_one(a2, "int8")
    f32 = ml.lowbit.pack_one(a2, "float32")
    bits = np.asarray(bf.bits)
    codes, exps = np.asarray(q8.codes), np.asarray(q8.exponents)
    mbf = np.asarray(ml.lowbit.materialize_one(bf))
    mq8 = np.asarray(ml.lowbit.materialize_one(q8))
    w1 = np.asarray(ml.lowbit.widen_bf16(bf.bits.reshape((LOWBIT_ROWS * LOWBIT_COLS,))))
    w3 = np.asarray(ml.lowbit.widen_bf16(bf.bits.reshape((4, LOWBIT_ROWS // 4, LOWBIT_COLS))))
    lin = np.asarray(ml.linalg.from_bf16(bf.bits))
    # The pure-Python spelling of the same four seams, from the module's own
    # private functions; it is `checks/numerics.mojo` written over Python
    # ints and never a NumPy vectorization, so agreeing with it is a claim
    # about the compiled seam and not about NumPy.
    py_bits = np.asarray(_lb._to_bf16_py(a2))
    py_wide = np.asarray(_lb._from_bf16_py(bits))
    py_codes, py_exps = _lb._quantize_int8_py(a2)
    py_deq = np.asarray(_lb._dequantize_int8_py(q8.codes, q8.exponents))

    def _refuses(fn, text):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            return text in str(exc)
        return False

    flags = np.array([
        ml.lowbit.is_packed(bf), ml.lowbit.is_packed(q8),
        ml.lowbit.is_packed(a2), ml.lowbit.is_packed(f32),
        ml.lowbit.format_of({"w": bf}) == "bfloat16",
        ml.lowbit.format_of({"w": q8}) == "int8",
        ml.lowbit.format_of({"w": a2}) == "float32",
        ml.lowbit.format_of([bf, a2]) == "bfloat16",
        _refuses(lambda: ml.lowbit.format_of({"a": bf, "b": q8}), "store a model one way"),
        bf.format == "bfloat16" and tuple(bf.shape) == (LOWBIT_ROWS, LOWBIT_COLS),
        q8.format == "int8" and tuple(q8.shape) == (LOWBIT_ROWS, LOWBIT_COLS),
        _refuses(lambda: ml.lowbit.pack_one(a2[:, 0], "bfloat16"), "must be 2-D to pack"),
        _refuses(lambda: ml.lowbit.pack_one(a2, "float16"), "unknown format"),
        # the same bits three ways: materialize_one, widen_bf16 at two other
        # ranks, and the public linalg entry
        lin.tobytes() == mbf.tobytes(),
        w1.tobytes() == mbf.tobytes(),
        w3.tobytes() == mbf.tobytes(),
        np.asarray(f32).tobytes() == a2.tobytes(),
        # the spelling that RAN against the pure-Python construction
        py_bits.tobytes() == bits.tobytes(),
        py_wide.tobytes() == mbf.tobytes(),
        np.asarray(py_codes).tobytes() == codes.tobytes(),
        np.asarray(py_exps).tobytes() == exps.tobytes(),
        py_deq.tobytes() == mq8.tobytes(),
        # THE SILENT-FALLBACK TRIPWIRE: 0 on every column that has a binding
        _lb._conversion_backend() == "python",
    ], dtype=np.int64)
    return _fit(dict(bf16_bits=_h(bits), int8=_h(codes, exps),
                     materialized=_h(mbf, mq8), widened=_h(w1, w3),
                     f32_copy=_h(np.asarray(f32)), linalg=_h(lin), flags=_h(flags)))


@lane("samba")
@floor(steps=(1, "one step is measured to be ENOUGH for this lane's arms, which is the only reason a "
                 "floor of one is honest here: its 20 parameter tensors contain no bitwise-equal "
                 "same-shape pair, so no permutation of two of them is invisible the way byte-lm's norm "
                 "weights were, and the layer-order swap and a change of the AdamW betas both move the "
                 "ONE-step cell (2026-09-16, docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 4). "
                 "Adding an arm first evaluated at a later step means raising this number first, which is "
                 "exactly what samba-untied-dropout-accum needed and did not get."),
       batch=(2, "two sequences per step is the shipped batch and the size every arm above was measured "
                 "live at (2026-09-16, docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 4). One "
                 "sequence takes the batch axis out of every kernel this stack launches."))
def _(ml, X, yc, yr, Xh=None):
    """SambaStack, one Mamba-3 layer and one attention layer at d_model 32
    over a 256-byte vocabulary, weights from the stack's own seeded
    generator, ONE AdamW step on one (2, 17) window of the fixture bytes
    (three before 2026-09-16, floored at one above). SUPPORT_MATRIX says cross-vendor qualification of the training
    surface is open. The checkpoint is the model column."""
    cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)
    m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3)
    steps, batch = 1, 2                          # FLOORED, see @floor above
    ids = _ids(X, steps * batch, 17)
    losses = [np.float64(m.train_step(ids[batch * k:batch * k + batch, :-1],
                                      ids[batch * k:batch * k + batch, 1:])["loss"])
              for k in range(steps)]
    params = m.parameters()
    return _fit(dict(loss=_h(np.asarray(losses)), logits=_h(np.asarray(m.forward(ids[:2, :-1]))),
                     params=_h(*[np.asarray(params[k]) for k in sorted(params)])),
                m, lambda e: (np.asarray(e.forward(_ids(Xh, 2, 16))),))


# ---------------------------------------------------------------- lanes (2026-09-14, the claim-surface census)
# docs/lanes/BRIEF_claim_surface_census_2026-09-14.md: every public constructor
# value that selects a DIFFERENT NUMERIC PATH (a kernel, an objective, a
# sampler, a solver, a metric, a reduction) and that no lane above pins.
# Each lane below bundles the values that share a fit so the cell count stays
# near one per uncovered kernel. Sizes match the lanes above. Where a value
# needs a target the fixture does not have (positive, three classes, a NaN),
# the lane derives it from the fixture bytes by a rule here, never from a
# host BLAS or a host transcendental.

def _three_class_centered(X):
    """Targets in 0..2 from the two unperturbed columns split at their
    MEDIANS, so every fixture carries all three classes; _three_class
    splits at zero and the `negative` fixture then has one class, which
    the multiclass losses refuse by name. The mlp lane keeps _three_class
    so its recorded cells do not move."""
    return ((X[:, 3] > np.median(X[:, 3])).astype(np.int32)
            + (X[:, 4] > np.median(X[:, 4])).astype(np.int32)).astype(np.int32)


def _pos(y):
    """A strictly positive regression target: |y| + 1, elementwise."""
    return (np.abs(y) + np.float32(1.0)).astype(np.float32)


def _with_nan(X):
    """The fixture with NaN in every eighth row, so a quantizer's nan_mode has
    something to place. No fixture carries a NaN (the other lanes would refuse
    it), so the GBDT nan lanes make their own.

    THE NaN MUST REACH A SPLIT (2026-09-16, lane/dead-arms). Until today this
    wrote NaN into columns 5, 6 and 7 only. `labels_for` builds `y_clf` from
    columns 3 and 4 alone, so the fitted ensemble split on columns 3 and 4
    alone and no NaN ever reached a split decision. `nan_mode="Min"` and
    `nan_mode="Max"` therefore hashed the SAME bytes at 1500, 6000 and 20000
    rows and the `gbdt-nan-modes` lane could not fail. Measured:

        shipped 5:8 nan cols [5, 6, 7]       min=a21e31ec9cc7597d max=a21e31ec9cc7597d
        this fixture, nan cols [3, 4, 5, 6, 7] min=5d7ce7cc59e261b0 max=e57525a0fb5169d2

    So column 3 and column 4 carry a NaN too, STAGGERED (3 on every eighth
    row, 4 four rows later) so that no row loses both label columns at once
    and both split columns are placed independently. Columns 5, 6 and 7 keep
    their NaN, which is what makes the fixture place a NaN on a column the
    trees do NOT split on as well as on the two they do. `labels_for` is
    applied to the CLEAN fixture, so no label moves."""
    Xn = X.copy()
    Xn[::8, 5:8] = np.float32(np.nan)
    Xn[::8, 3] = np.float32(np.nan)
    Xn[4::8, 4] = np.float32(np.nan)
    return Xn


def _affinity(P):
    """A dense affinity for the precomputed spectral lane, from fixed-order
    float64 host arithmetic (elementwise, no BLAS, no transcendental):
    1 / (1 + d2) where d2 is under the matrix median, else exactly zero
    (COO conversion drops the zeros, as cuML's does)."""
    Q = P.astype(np.float64)
    d2 = np.zeros((Q.shape[0], Q.shape[0]), dtype=np.float64)
    for j in range(Q.shape[1]):
        col = Q[:, j]
        d2 = d2 + (col[:, None] - col[None, :]) ** 2
    t = float(np.median(d2))
    A = np.where(d2 < t, 1.0 / (1.0 + d2), 0.0)
    return np.ascontiguousarray(A.astype(np.float32))


def _cross_affinity(Q, P):
    """The affinity of rows Q to the training rows P under `_affinity`'s
    rule: 1 / (1 + d2) where d2 is under the TRAINING matrix's median (the
    threshold `_affinity(P)` used), else exactly zero. Fixed-order float64
    host arithmetic, elementwise."""
    Pd = P.astype(np.float64)
    Qd = Q.astype(np.float64)
    d2 = np.zeros((Pd.shape[0], Pd.shape[0]), dtype=np.float64)
    c2 = np.zeros((Qd.shape[0], Pd.shape[0]), dtype=np.float64)
    for j in range(Pd.shape[1]):
        col = Pd[:, j]
        d2 = d2 + (col[:, None] - col[None, :]) ** 2
        c2 = c2 + (Qd[:, j][:, None] - col[None, :]) ** 2
    t = float(np.median(d2))
    return np.ascontiguousarray(np.where(c2 < t, 1.0 / (1.0 + c2), 0.0).astype(np.float32))


def _exact_prob(n, seed):
    """A probability vector with no host fold: ranks (k+1)/S with S the
    integer n(n+1)/2, permuted by the hashed stream."""
    order = np.argsort(_hashed_uniform(n, 1, seed).reshape(-1), kind="stable")
    ranks = (order.astype(np.float64) + 1.0) / float(n * (n + 1) // 2)
    return ranks.astype(np.float32)


# -- trees

@lane("rf-clf-entropy-log2-noboot")
def _(ml, X, yc, yr, Xh=None):
    """Entropy splits, log2 features, no bootstrap, a level-order leaf cap."""
    m = ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7, criterion="entropy",
                                  max_features="log2", bootstrap=False, max_leaves=64).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("rf-clf-balanced-parallel")
def _(ml, X, yc, yr, Xh=None):
    """class_weight selects the weighted fit entry; parallel_groves the
    other predict kernel and a resident device model."""
    m = ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7, class_weight="balanced",
                                  inference_engine="parallel_groves").fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("rf-reg-poisson")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7, criterion="poisson",
                                 max_features="sqrt", max_samples=0.6).fit(X, _pos(yr))
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("rf-reg-gamma-ig")
def _(ml, X, yc, yr, Xh=None):
    """Two fits, gamma and inverse gaussian, on a strictly positive target."""
    y = _pos(yr)
    g = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7, criterion="gamma").fit(X, y)
    i = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7,
                                 criterion="inverse_gaussian", max_features=None).fit(X, y)
    return _fit(dict(gamma=_h(g.predict(X)), inverse_gaussian=_h(i.predict(X))),
                g, lambda e: (e.predict(Xh),))


@lane("et-clf-entropy-bestfirst")
def _(ml, X, yc, yr, Xh=None):
    """max_leaf_nodes selects the best-first frontier grower, a different
    builder from the level-wise one every other forest lane runs."""
    m = ml.ExtraTreesClassifier(n_estimators=16, random_state=7, criterion="entropy", max_leaf_nodes=32,
                                max_features=4).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("et-reg-bootstrap-parallel")
def _(ml, X, yc, yr, Xh=None):
    m = ml.ExtraTreesRegressor(n_estimators=16, max_depth=8, random_state=7, max_features="log2",
                               bootstrap=True, max_samples=0.5, inference_engine="parallel_groves").fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-multiclass")
def _(ml, X, yc, yr, Xh=None):
    """MultiClass with class weights on the three-class target."""
    t = _three_class_centered(X)
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="MultiClass",
                            class_weights=[1.0, 2.0, 0.5]).fit(X, t)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-onevsall")
def _(ml, X, yc, yr, Xh=None):
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="MultiClassOneVsAll").fit(X, _three_class_centered(X))
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-parametric-losses")
@floor(rows=(1500, "1500 rows was measured to cost no detection against the 20000 this lane used to fit "
                   "(2026-09-16, docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 4): the eleven loss parts "
                   "still separate into ten distinct hashes, and the union of columns the fitted "
                   "ensembles actually split on is at least 13 of 16. Nothing below 1500 is measured."))
def _(ml, X, yc, yr, Xh=None):
    """The ten losses no lane above fits, eight trees of depth four each:
    four take a mandatory parameter; CrossEntropy takes a probability
    target; the rest a positive target."""
    rows = 1500                                  # FLOORED, see @floor above
    X, yc, yr = X[:rows], yc[:rows], yr[:rows]   # rows are a fixture size, not a claim (2026-09-16)
    y = _pos(yr)
    fits = {
        "Quantile": dict(), "MAE": dict(), "LogLinQuantile": dict(), "MAPE": dict(), "Poisson": dict(),
        "Lq": dict(loss_q=3.0), "Expectile": dict(loss_alpha=0.3),
        "Tweedie": dict(loss_variance_power=1.5), "Huber": dict(loss_delta=1.0),
    }
    parts, first = {}, None
    for loss, kw in fits.items():
        m = _gbdt(ml.GradientBoosting, n_estimators=8, max_depth=4, loss=loss, **kw).fit(X, y)
        parts[loss] = _h(m.predict(X))
        first = first or m
    ce = _gbdt(ml.GradientBoosting, n_estimators=8, max_depth=4, loss="CrossEntropy").fit(X, yc.astype(np.float32))
    parts["CrossEntropy"] = _h(ce.predict(X))
    return _fit(parts, first, lambda e: (e.predict(Xh),))


@lane("gbdt-lossguide-newtoncosine")
@floor(rows=(1500, "at 1500 rows the fitted ensemble still splits on ALL 16 columns despite "
                   "feature_fraction=0.5, and 13 of its 20 trees still reach max_leaves=32 "
                   "(2026-09-16, docs/lanes/LANE_STATUS_shrink-blindness-audit.md sections 4 and 5e). Fewer rows starve "
                   "the lossguide split budget this lane exists to exercise."))
def _(ml, X, yc, yr, Xh=None):
    """NewtonCosine, the child-hessian and split-gain and leaf-count
    thresholds, column subsampling, random strength, Bernoulli bootstrap,
    gradient leaves with three iterations: one Lossguide fit."""
    rows = 1500                                  # FLOORED, see @floor above
    X, yc, yr = X[:rows], yc[:rows], yr[:rows]   # rows are a fixture size, not a claim (2026-09-16)
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_leaves=32, grow_policy="Lossguide", loss="Logloss",
                            score_function="NewtonCosine", min_child_hessian=1.0, min_split_gain=0.01,
                            min_data_in_leaf=8, feature_fraction=0.5, random_strength=1.0,
                            bootstrap_type="Bernoulli", subsample=0.7,
                            leaf_estimation_method="Gradient", leaf_estimation_iterations=3).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-pointwise-l2-bayesian-eval")
def _(ml, X, yc, yr, Xh=None):
    """The pointwise searcher, L2 scores, Bayesian bootstrap with a
    temperature, boost from average on Logloss, an eval set with
    iteration-based early stopping and best-model truncation, and row
    weights: one symmetric fit."""
    w = _hw((X.shape[0],), "gbdt-eval:sample_weight", 0.5, 1.5)
    ych = labels_for(Xh, HELDOUT_SEED)[0]
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="Logloss", score_function="L2",
                            use_pointwise_searcher=True, bootstrap_type="Bayesian", bagging_temperature=0.5,
                            boost_from_average=True, od_type="Iter", od_wait=5, use_best_model=True
                            ).fit(X, yc, sample_weight=w, eval_set=(Xh, ych))
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-exact-mae")
def _(ml, X, yc, yr, Xh=None):
    """The Exact (weighted quantile) leaf estimator and the Poisson bootstrap."""
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="MAE", leaf_estimation_method="Exact",
                            bootstrap_type="Poisson", subsample=0.6).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-categorical-ctr")
def _(ml, X, yc, yr, Xh=None):
    """A categorical feature and a one-hot feature (disjoint: cat_features
    makes its own one-hot decision) with permutation_count=2, on the coded
    columns of _coded. NO CTR IS BUILT HERE: _coded's column 0 holds two
    categories, at or below the GPU one_hot_max_size of 2, so train makes it
    a one-hot column, no permutation-dependent feature exists and the fit
    runs one permutation (the Apple column's model texts on all nine
    fixtures carry no ctr record, dumped on the M4 2026-09-15). This lane
    measures the one-hot categorical arm; a CTR column needs a source with
    more than two categories."""
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="Logloss", cat_features=[0],
                            one_hot_features=[1], permutation_count=2,
                            ctr_estimation_permutation_id=0).fit(_coded(X), yc)
    return _fit(dict(predict=_h(m.predict(_coded(X)))), m, lambda e: (e.predict(_coded(Xh)),))


#: lane/inference-gbdt-ctr-tables (2026-09-15). The saved-model directory
#: of the two CTR table lanes: a GPU column FITS and writes
#: `<dir>/<lane>.<fixture>.npz` (repeat 0); a CPU column, whose training
#: path refuses CTR tables by name, LOADS that file through
#: `mojolearn.host_model` instead of fitting, so its train, infer and batch
#: cells are HostGBDT's predictions from the GPU's own saved model and its
#: model cell is that file's hash.
GBDT_CTR_MODELS_ENV = "MOJOLEARN_IDENTITY_GBDT_CTR_MODELS"


def _rank_codes(col, cuts):
    """Dense codes from the RANK of a column (a stable argsort, ties by row
    order), cut at the fractions `cuts`: host arithmetic only, the same
    bytes on every box, every code present, uneven group sizes."""
    n = col.shape[0]
    ranks = np.empty(n, dtype=np.int64)
    ranks[np.argsort(col, kind="stable")] = np.arange(n, dtype=np.int64)
    return np.searchsorted(np.asarray(cuts, dtype=np.float64), ranks.astype(np.float64) / n,
                           side="right").astype(np.int64)


def _ctr_tables_x(X, heldout=False):
    """gbdt-categorical-ctr-tables' input: column 0 has eight categories
    (seven rank groups of column 3 and an eighth held by ONE training row),
    column 1 five (rank groups of column 4), both above the GPU
    one_hot_max_size of 2, so each becomes four CTR columns (Borders at three
    priors and FeatureFreq); column 2 is a two-category one-hot column; then
    the first eight numeric columns. On the held-out rows every 97th row
    carries category 8 and every 101st category 12 in column 0 (never seen
    in training), every 89th category 7 (seen once), every 103rd category 9
    in column 1 (never seen)."""
    c0 = _rank_codes(X[:, 3], (0.30, 0.50, 0.65, 0.77, 0.86, 0.93))
    c1 = _rank_codes(X[:, 4], (0.20, 0.45, 0.70, 0.90))
    c2 = (X[:, 5] > np.median(X[:, 5])).astype(np.int64)
    rows = np.arange(X.shape[0])
    if heldout:
        c0[rows % 89 == 0] = 7
        c0[rows % 97 == 0] = 8
        c0[rows % 101 == 0] = 12
        c1[rows % 103 == 0] = 9
    else:
        c0[int(np.argsort(X[:, 6], kind="stable")[-1])] = 7
    return np.ascontiguousarray(np.column_stack(
        [c0.astype(np.float32), c1.astype(np.float32), c2.astype(np.float32), X[:, :8]]).astype(np.float32))


def _ctr_tables_xh(Xh):
    return _ctr_tables_x(Xh, heldout=True)


def _tensor_ctr_x(X, heldout=False):
    """gbdt-tensor-ctr-tables' input: two categorical source columns (four
    rank groups of column 3 plus a fifth category held by ONE training row,
    four rank groups of column 4), then the first four numeric columns. On
    the held-out rows every 89th row carries source 0 category 4 (the
    combinations seen once, or never), every 97th category 6 (past its
    cardinality, the empty-value key)."""
    c0 = _rank_codes(X[:, 3], (0.40, 0.70, 0.90))
    c1 = _rank_codes(X[:, 4], (0.25, 0.55, 0.80))
    rows = np.arange(X.shape[0])
    if heldout:
        c0[rows % 89 == 0] = 4
        c0[rows % 97 == 0] = 6
    else:
        c0[int(np.argsort(X[:, 6], kind="stable")[-1])] = 4
    return np.ascontiguousarray(np.column_stack(
        [c0.astype(np.float32), c1.astype(np.float32), X[:, :4]]).astype(np.float32))


def _tensor_ctr_xh(Xh):
    return _tensor_ctr_x(Xh, heldout=True)


def _tensor_ctr_y(Xt):
    """The combination's learn frequency, float32, plus a small numeric
    term: a target the tensor FeatureFreq column predicts and neither source
    alone does."""
    key = Xt[:, 0].astype(np.int64) * 16 + Xt[:, 1].astype(np.int64)
    _, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    freq = counts[inverse].astype(np.float64) / Xt.shape[0]
    return (freq * 10.0 + 0.01 * Xt[:, 2].astype(np.float64)).astype(np.float32)


def _ctr_saved_or_fit(ml, fit):
    """The estimator a CTR table lane answers with: fitted (and saved,
    under GBDT_CTR_MODELS_ENV, on repeat 0) on a GPU column; on a CPU
    column the HostGBDT of the saved file, tagged with its path."""
    lane_name, fx, repeat = (_DUMP_TAG.split("/") + ["", "", ""])[:3]
    d = os.environ.get(GBDT_CTR_MODELS_ENV, "").strip()
    path = os.path.join(d, f"{lane_name}.{fx}.npz") if d and lane_name else None
    if ml.vendor() == "cpu":
        if not d:
            from mojolearn._verification_ctr_models import resolve_model
            path = resolve_model(lane_name, fx)
        if not path or not os.path.isfile(path):
            raise RuntimeError(f"CPU training refuses CTR tables by name; set {GBDT_CTR_MODELS_ENV} to the "
                               f"directory of this lane's GPU-saved models (no {path})")
        from mojolearn._forest_host import binary_path, host_model
        est = host_model(path)
        # the train and batch cells predict through this object, so it takes
        # _probe_fit_host's guard: a forest binding loaded earlier under the
        # shared module name (host_record reads MOJOLEARN_HOST_DIR) is not the
        # MOJOLEARN_FOREST_HOST_BINARY asked for
        bound = getattr(getattr(est, "_binding", None), "__file__", None)
        if bound is not None and os.path.realpath(bound) != os.path.realpath(binary_path()):
            raise RuntimeError(f"the forest binding in this process is {bound}, "
                               f"not MOJOLEARN_FOREST_HOST_BINARY {binary_path()}")
        est.identity_saved_path = path
        return est
    est = fit()
    if path and repeat == "0":
        os.makedirs(d, exist_ok=True)
        est.save(path)
    return est


@lane("gbdt-categorical-ctr-tables")
def _(ml, X, yc, yr, Xh=None):
    """Gradient boosting whose categorical columns are above
    one_hot_max_size, so the saved model carries real CTR tables (Borders at
    three priors and FeatureFreq per column, ctr_table and ctr_entry records)
    beside a one-hot column: 20 depth-6 Logloss trees. The held-out rows
    carry unseen and seen-once categories."""
    Xc = _ctr_tables_x(X)
    m = _ctr_saved_or_fit(ml, lambda: _gbdt(ml.GradientBoosting, 
        n_estimators=20, max_depth=6, loss="Logloss", cat_features=[0, 1, 2]).fit(Xc, yc))
    return _fit(dict(predict=_h(m.predict(Xc)), proba=_h(m.predict_proba(Xc))),
                m, lambda e: (e.predict(_ctr_tables_xh(Xh)), e.predict_proba(_ctr_tables_xh(Xh))))


@lane("gbdt-tensor-ctr-tables")
def _(ml, X, yc, yr, Xh=None):
    """ExperimentalTwoLevelFeatureFreq on two categorical sources with a
    target the combination's frequency predicts, so the tree splits on the
    tensor FeatureFreq column and the saved model carries a
    tensor_ctr_registry with feature_freq_tensor records. The held-out rows
    carry combinations seen once, never seen, and a code past the source's
    cardinality."""
    Xt = _tensor_ctr_x(X)
    m = _ctr_saved_or_fit(ml, lambda: ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7).fit(
        Xt, _tensor_ctr_y(Xt)))
    return _fit(dict(predict=_h(m.predict(Xt))), m, lambda e: (e.predict(_tensor_ctr_xh(Xh)),))


@lane("gbdt-nan-modes")
@floor(rows=(1500, "the nan_mode arm is LIVE at 1500 as of 2026-09-16 (lane/dead-arms): Min and Max hash "
                   "da54f1f980861f61 against d8e2a3e5305a859b here, and they are DISTINCT on all nine "
                   "fixtures including `ties`. Until that day they hashed the SAME bytes at 1500, 6000 "
                   "and 20000, because _with_nan wrote NaN into columns 5, 6 and 7 while the fit splits "
                   "only on 3 and 4, so no row count meant anything. The floor stays at 1500 because that "
                   "is the size the live arm was measured at; see docs/lanes/LANE_STATUS_dead-arms.md "
                   "section 1."))
def _(ml, X, yc, yr, Xh=None):
    """nan_mode Min and Max on a fixture that actually carries NaN
    (_with_nan); on a NaN-free column the quantizer collapses both to
    Forbidden, which is why no lane above could reach them."""
    rows = 1500                                  # FLOORED, see @floor above
    X, yc, yr = X[:rows], yc[:rows], yr[:rows]   # rows are a fixture size, not a claim (2026-09-16)
    Xn = _with_nan(X)
    lo = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="Logloss", nan_mode="Min").fit(Xn, yc)
    hi = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="Logloss", nan_mode="Max").fit(Xn, yc)
    return _fit(dict(min=_h(lo.predict(Xn)), max=_h(hi.predict(Xn))),
                lo, lambda e: (e.predict(_with_nan(Xh)),))


@lane("gbdt-adapter-clf")
def _(ml, X, yc, yr, Xh=None):
    """The sklearn-style classifier adapter, the only caller of the binary
    probability and class entries. It refuses save, so no model column."""
    m = _gbdt(ml.GradientBoostingClassifier, n_estimators=20, max_depth=6).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)),
                     decision=_h(m.decision_function(X[:512]))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh), e.decision_function(Xh[:512])))


@lane("gbdt-adapter-reg")
def _(ml, X, yc, yr, Xh=None):
    m = _gbdt(ml.GradientBoostingRegressor, n_estimators=20, max_depth=6).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


#: the gbdt-query-rmse lane's query sizes, cycled in row order: sizes of one
#: and two rows (a query of one has zero QueryRMSE derivative on every row)
#: beside larger ones, uneven on purpose so the 32-lane query reduce sees
#: queries shorter than, near and longer than one lane stride.
RANK_QUERY_SIZES = (1, 2, 7, 3, 16, 1, 5, 40, 2, 9)

#: the relevance cut points: comparisons only, no float arithmetic, so every
#: box hands the lane the same grades
RANK_CUTS = (-1.0, 0.0, 1.0, 2.0)


def _rank_groups(n):
    """The gbdt-query-rmse group ids, one per row, consecutive by
    construction (`RANK_QUERY_SIZES` cycled; the last query truncated at
    `n`). Shared with the CatBoost CPU reference script, so both sides read
    the same grouping."""
    ids = []
    q = 0
    while len(ids) < n:
        ids.extend([q] * RANK_QUERY_SIZES[q % len(RANK_QUERY_SIZES)])
        q += 1
    return np.asarray(ids[:n], dtype=np.int64)


def _relevance(yr):
    """Tie-heavy graded relevance 0..4 from the fixture's regression target,
    cut at `RANK_CUTS` (`np.digitize`, comparisons only), as float32."""
    return np.digitize(yr, np.asarray(RANK_CUTS, dtype=np.float32)).astype(np.float32)


@lane("gbdt-query-rmse")
def _(ml, X, yc, yr, Xh=None):
    """QueryRMSE (learning to rank, the querywise target) on uneven queries
    with tie-heavy grades: 20 depth-6 symmetric trees, the predictions and
    the learn loss curve hashed. Predict is row-wise, so the held-out probe
    and the batch part apply."""
    g = _rank_groups(X.shape[0])
    rel = _relevance(yr)
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="QueryRMSE").fit(X, rel, group_id=g)
    return _fit(dict(predict=_h(m.predict(X)), loss_curve=_h(np.asarray(m.loss_curve_, dtype=np.float64))),
                m, lambda e: (e.predict(Xh),))


@lane("gbdt-pair-logit")
@floor(rows=(1500, "1500 rows still carries 178 query groups, 994 explicit pairs and every grade 0..4 "
                   "(2026-09-16, docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 4). A ranking loss needs "
                   "groups holding MIXED grades, and that is what fewer rows take away first."))
def _(ml, X, yc, yr, Xh=None):
    """PairLogit (learning to rank, pairwise derivatives on query groups):
    20 depth-6 symmetric trees on the gbdt-query-rmse queries and grades with
    the pairs generated from them, and a second fit of 8 trees given explicit
    `pairs` and `pairs_weight` (every third generated-style pair of the first
    40 queries, hashed weights), so both input paths are hashed. Predict is
    row-wise, so the held-out probe and the batch part apply."""
    rows = 1500                                  # FLOORED, see @floor above
    X, yc, yr = X[:rows], yc[:rows], yr[:rows]   # a fixture size, not a claim; 178 query groups remain (2026-09-16)
    g = _rank_groups(X.shape[0])
    rel = _relevance(yr)
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="PairLogit").fit(X, rel, group_id=g)
    pairs = []
    begin = 0
    for q in range(40):
        size = int(np.count_nonzero(g == q))
        for a in range(begin, begin + size):
            for b in range(a + 1, begin + size):
                if rel[a] != rel[b] and (a + b) % 3 == 0:
                    pairs.append((a, b) if rel[a] > rel[b] else (b, a))
        begin += size
    pw = _hw((len(pairs),), "gbdt-pair-logit:pairs_weight", 0.5, 2.0)
    e = _gbdt(ml.GradientBoosting, n_estimators=8, max_depth=4, loss="PairLogit").fit(
        X, rel, group_id=g, pairs=pairs, pairs_weight=pw)
    return _fit(dict(predict=_h(m.predict(X)), loss_curve=_h(np.asarray(m.loss_curve_, dtype=np.float64)),
                     explicit_predict=_h(e.predict(X)),
                     explicit_loss_curve=_h(np.asarray(e.loss_curve_, dtype=np.float64))),
                m, lambda est: (est.predict(Xh),))


@lane("gbdt-yeti-rank")
def _(ml, X, yc, yr, Xh=None):
    """YetiRank (learning to rank, sampled permutations on query groups): 20
    depth-6 symmetric trees on the gbdt-query-rmse queries and grades, and a
    second fit of 8 depth-4 trees at random_state=7, so the hash covers the
    derivative draws' stream as well as the task tables. Unweighted: the host
    column refuses sample_weight on every loss. YetiRank has no loss value
    (the target writes 0), so `loss_curve_` is hashed only to pin that.
    Predict is row-wise, so the held-out probe and the batch part apply."""
    g = _rank_groups(X.shape[0])
    rel = _relevance(yr)
    m = _gbdt(ml.GradientBoosting, n_estimators=20, max_depth=6, loss="YetiRank").fit(X, rel, group_id=g)
    e = _gbdt(ml.GradientBoosting, n_estimators=8, max_depth=4, loss="YetiRank", random_state=7).fit(
        X, rel, group_id=g)
    return _fit(dict(predict=_h(m.predict(X)), loss_curve=_h(np.asarray(m.loss_curve_, dtype=np.float64)),
                     seeded_predict=_h(e.predict(X))),
                m, lambda est: (est.predict(Xh),))


def _weighted_score_parts(clf, reg, X, yc, yr, tag):
    """score(X, y, sample_weight) on 1024 rows the fit did not see: hashed
    weights on [0.25, 4), the same weights with every seventh zeroed, unit
    weights, and the unweighted score beside them. Weights come from the
    hashed stream, never from the fixture's own floats."""
    lo, hi = 2000, 3024
    Xs = np.ascontiguousarray(X[lo:hi])
    ycs = np.ascontiguousarray(yc[lo:hi])
    yrs = np.ascontiguousarray(yr[lo:hi])
    w = _hw((hi - lo,), f"{tag}:w", 0.25, 4.0)
    wz = w.copy()
    wz[::7] = np.float32(0.0)
    ones = np.ones(hi - lo, dtype=np.float32)
    parts = {}
    for kind, m, y in (("clf", clf, ycs), ("reg", reg, yrs)):
        parts[f"{kind}_weighted"] = _h(np.float64(m.score(Xs, y, sample_weight=w)))
        parts[f"{kind}_zeroed"] = _h(np.float64(m.score(Xs, y, sample_weight=wz)))
        parts[f"{kind}_unit"] = _h(np.float64(m.score(Xs, y, sample_weight=ones)))
        parts[f"{kind}_unweighted"] = _h(np.float64(m.score(Xs, y)))
    return parts


@lane("gbdt-adapter-score-weighted")
def _(ml, X, yc, yr, Xh=None):
    """GradientBoostingClassifier.score and GradientBoostingRegressor.score
    with sample_weight (lane/cpu-training-small-gaps, 2026-09-15): scikit-
    learn's weighted accuracy and R2 on the pinned-sum path. A lane of its
    own so the adapter lanes' committed hashes do not move; 2000 training
    rows."""
    clf = _gbdt(ml.GradientBoostingClassifier, n_estimators=20, max_depth=6).fit(X[:2000], yc[:2000])
    reg = _gbdt(ml.GradientBoostingRegressor, n_estimators=20, max_depth=6).fit(X[:2000], yr[:2000])
    return _fit(_weighted_score_parts(clf, reg, X, yc, yr, "gbdt-adapter-score-weighted"))


@lane("rf-score-weighted")
def _(ml, X, yc, yr, Xh=None):
    """RandomForestClassifier.score and RandomForestRegressor.score with
    sample_weight (lane/cpu-training-small-gaps, 2026-09-15), through the
    forest protocol's score that the Extra Trees share; 2000 training rows."""
    clf = ml.RandomForestClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X[:2000], yc[:2000])
    reg = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X[:2000], yr[:2000])
    return _fit(_weighted_score_parts(clf, reg, X, yc, yr, "rf-score-weighted"))


# -- neural

@lane("mamba2-dtlimit")
@floor(seqlen=(8, "L=8 keeps everything this lane claims live: all three dt_limit changes are DETECTED "
                  "and all 8 sequence positions move the cell (2026-09-16, "
                  "docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 4). Length was never the "
                  "lever on the dead path here, which was the CLAMP: under the old (0.01, 0.1) every dt "
                  "saturated at the upper bound at L=16 as well as at L=8, so dt_bias was read "
                  "one-sidedly and a clamp pinned to a constant read IDENTICAL. Fixed by moving the "
                  "clamp, not the length (2026-09-16, lane/dead-arms, "
                  "docs/lanes/LANE_STATUS_dead-arms.md section 3)."))
def _(ml, X, yc, yr, Xh=None):
    """The active dt clamp (seam S9); at the default (0, inf) it cannot
    move a bit, which is what the mamba2 lane measures.

    THE CLAMP MUST BITE ON BOTH SIDES AND ALSO NOT BITE (2026-09-16,
    lane/dead-arms). Until today this lane ran `dt_limit=(0.01, 0.1)`, and
    the dt it makes lies in [0.283, 1.110] on `base` (measured off the
    library by bisecting each bound until the cell moves; the nine ranges
    are in docs/lanes/LANE_STATUS_dead-arms.md). EVERY dt was therefore
    above the upper bound, so S9 returned `hi` for every value and the lane
    read one branch of a three-branch clamp:

      - `dt_bias` shifted by +0.01, +0.1, +0.25, +1, +4, +16 and by -1 left
        the cell UNMOVED at L=8 and at L=16; only -4 and -16 moved it;
      - the LOWER bound was inert: walking `lo` from 0 to 0.0999 with `hi`
        at 0.1 never moved the cell, on any of the nine fixtures;
      - `dt_limit=(0.1, 0.1)`, a clamp that returns a CONSTANT for every
        input, read IDENTICAL to the shipped cell. The lane exists for the
        clamp and could not tell it from a constant.

    (0.5, 0.9) sits inside that dt range, so some values clamp low, some
    clamp high and some pass through untouched. Measured over the nine
    fixtures: the lower bound is live on eight (not `ties`, whose dt starts
    at 0.645), the upper bound on eight (not `negative`, whose dt stops at
    0.589), the constant-clamp probe is DETECTED on nine, and a `dt_bias`
    shift of +/-0.01 is DETECTED on nine. No single pair can do better:
    `ties` starts above where `negative` ends."""
    dm, di, nh = 32, 64, 1
    cd, dip = di + 256, 2 * di + 256 + nh
    w = _block_weights("mamba2", {
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
        "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
        "out_proj.weight": (dm, di)},
        ones=("block_norm.weight", "norm.weight"))
    seqlen = 8                                   # FLOORED, see @floor above
    blk = ml.Mamba2Block(w, dt_limit=(0.5, 0.9))
    parts = _block_fit(blk, _seq(X, 2, seqlen, dm), _seq(X, 2, seqlen, dm, skip=1024), {})
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, seqlen, dm))),))


@lane("transformer-window")
def _(ml, X, yc, yr, Xh=None):
    """Sliding-window attention: a different mask, a ring KV cache and a
    shorter key span than the full causal block the transformer lane runs."""
    dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
    w = _block_weights("transformer", {
        "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
        "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm), "v_proj.weight": (nkv * hd, dm),
        "o_proj.weight": (dm, nh * hd), "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
        "down_proj.weight": (dm, it)},
        # the two layernorms are the same shape; at ones they were bitwise
        # equal and a swap of them was invisible (2026-09-16, lane/dead-arms)
        near_one=("input_layernorm.weight", "post_attention_layernorm.weight"))
    blk = ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv, window=8)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), dict(max_tokens=32))
    return _fit(parts, blk, lambda e: (np.asarray(_neural_inference(ml, "transformer-window", e).forward(_seq(Xh, 2, 16, dm))),))


@lane("byte-lm-resident")
@floor(steps=(1, "ONE is the weakest floor there is, and this lane is the case where that is the truthful "
                 "number. Its claim is the SESSION, not a shape or a schedule: the resident export must "
                 "equal the stateless path's gradient byte for byte, which `_same_bytes` below asserts at "
                 "whatever step count both run, and its per-tensor `grads` part catches a norm-weight "
                 "exchange on the WRITE side at one step. It is blind to the READ-side exchange by the "
                 "same mechanism byte-lm was (2026-09-16, "
                 "docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 2), and byte-lm, the shipped "
                 "profile of the same trainer, is floored at two steps to carry that class for both."))
def _(ml, X, yc, yr, Xh=None):
    """The device-owned session (resident=True, lean step results), the
    path the byte-lm lane's docstring says is bit-equal to the stateless
    one and measured on one H100 only.

    SHAPE (lane/neural-shape-shrink, 2026-09-16): one block at d_model 16,
    not the shipped two-block d_model-32 profile. THIS LANE'S CLAIM IS THE
    SESSION, not a shape: it asserts the resident export equals the
    stateless path's gradient BYTE FOR BYTE at whatever shape both are
    built at (`_same_bytes` below), so the claim is shape-independent and
    survives the cut. The shipped profile keeps its device column from the
    `byte-lm` lane, which is floored at it on purpose. Depth is the lever
    that removes Metal work here: cost on Metal is per-launch overhead, and
    halving the block count halves the launches (measured 1.380 s -> 0.849 s
    per resident step, and 0.644 s with d_model 16 and intermediate 32)."""
    shape = ml.ByteLanguageModelConfig(n_layers=1, d_model=16, n_heads=2, n_kv=1,
                                       head_dim=8, intermediate=32)
    named, _ = _byte_lm_params(shape)
    m = ml.SmallByteLanguageModelTrainer(named, data_schedule={"dataset": "identity_break", "order": "sequential"},
                                         shape=shape, resident=True, step_result="lean")
    steps = 1                                    # FLOORED, see @floor above
    ids = _ids(X, steps * shape.batch, shape.length + 1)
    losses = [np.float64(m.train_step(ids[2 * k:2 * k + 2])["loss"]) for k in range(steps)]
    # The exported gradient of the last step, flat and per tensor. Hash the
    # ARRAYS: `export_gradients()` returns a dict holding a nested dict, and
    # `np.asarray` of a dict is a zero-dimensional object array whose bytes
    # are the dict's address, which read MOVED on every fixture on the H100
    # and the MI300X on 2026-09-14 (the 118-lane record) and on this Mac,
    # with loss, params and logits equal to the byte-lm lane bit for bit.
    # That was this lane's hashing, not the session (fixed the same day).
    grads = m.export_gradients()
    flat = np.asarray(grads["flat_gradients"])
    named = {k: np.asarray(v) for k, v in grads["gradients"].items()}
    # Gate G1 of the device-owned step design (RUN OWED on the H100 since
    # 2026-09-11, docs/lanes/DESIGN_lm_device_owned_step_2026-09-11.md):
    # the resident export equals the stateless path's returned gradient for
    # the same three steps; a byte between them reads REFUSED with the pair
    # named, never a quiet hash.
    s = ml.SmallByteLanguageModelTrainer(_byte_lm_params(shape)[0],
                                         data_schedule={"dataset": "identity_break", "order": "sequential"},
                                         shape=shape)
    for k in range(steps):
        stateless = s.train_step(ids[2 * k:2 * k + 2])
    _same_bytes("resident export_gradients()['flat_gradients']", flat,
                "stateless train_step()['flat_gradients']", np.asarray(stateless["flat_gradients"]))
    return _fit(dict(loss=_h(np.asarray(losses)), params=_h(np.asarray(m.parameters_)),
                     grads=_h(flat, *[named[k] for k in sorted(named)]),
                     logits=_h(np.asarray(m.logits(ids[:2, :-1])))),
                m, lambda e: (np.asarray(e.logits(_ids(Xh, shape.batch, shape.length))),))


@lane("byte-lm-host-infer-threaded")
def _(ml, X, yc, yr, Xh=None):
    """LanguageModelInference on its SHIPPED DEFAULT arm, threaded=True, a
    different host kernel (SIMD, a split fold) from the reference arm the
    byte-lm-host-infer lane pins. Two threads, the smallest split, so the
    fold's combine is exercised on every box the same way."""
    shape = ml.ByteLanguageModelConfig()
    _, flat = _byte_lm_params(shape)
    m = ml.LanguageModelInference(flat, shape=shape, threaded=True, threads=2)
    ids = _ids(X, shape.batch, shape.length + 1)
    return _fit(dict(loss_bits=_h(np.uint32(m.loss_bits(ids))), logits=_h(np.asarray(m.logits(ids[:, :-1]))),
                     next=_h(np.asarray(m.next_bytes(ids[:, :-1])))),
                m, lambda e: (np.asarray(e.logits(_ids(Xh, shape.batch, shape.length))),))


@lane("samba-untied-dropout-accum")
@floor(steps=(3, "step 3 is the FIRST step that evaluates the cosine arm at all. `_Schedule._progress` "
                 "returns the linear warmup value while t <= warmup_steps and this lane runs "
                 "WarmupCosineLR(warmup_steps=2), so at one or two steps the cosine decay, and the exact "
                 "rational `_cos_pi_interval` / `_decide_f32` path under it, are never reached: the cell "
                 "could not tell its own schedule from a linear one or from a constant one. Measured "
                 "blind at one step and detected at three (2026-09-16, "
                 "docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 3; the same floor was written "
                 "in prose in docs/lanes/LANE_STATUS_lane-identity-fixtures-light.md section 1f BEFORE "
                 "the cut and did not stop it, which is why it is code now)."),
       batch=(32, "32 rows per step is T = 512 tokens, the smallest size at which "
                  "`training.accumulation_is_aligned` admits the A = 4 accumulation split this lane "
                  "exists to claim; at 16 rows A = 4 is refused BY NAME, which deletes the claim rather "
                  "than shrinking it (measured, docs/lanes/LANE_STATUS_lane-identity-fixtures-light.md "
                  "section 1f, 2026-09-16)."))
def _(ml, X, yc, yr, Xh=None):
    """Untied embeddings, dropout on, four accumulation microbatches, a
    global-norm clip and a warmup-cosine schedule: the training knobs the
    samba lane leaves at their defaults. Batches of 32 rows of 17, so each
    step is T = 512 tokens, a split clause 9.2 admits at A = 4 (64 at
    A = 4 is refused by name; the admitted (T, A) pairs come from
    `training.accumulation_is_aligned`)."""
    cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64,
                         tie_embeddings=False, dropout=0.1)
    m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3, max_norm=1.0, accumulation_steps=4,
                      lr_schedule=ml.training.WarmupCosineLR(1e-3, warmup_steps=2, total_steps=8, min_lr=1e-5))
    steps, batch = 3, 32                         # FLOORED, see @floor above
    ids = _ids(X, steps * batch, 17)
    losses = [np.float64(m.train_step(ids[batch * k:batch * k + batch, :-1],
                                      ids[batch * k:batch * k + batch, 1:])["loss"])
              for k in range(steps)]
    params = m.parameters()
    return _fit(dict(loss=_h(np.asarray(losses)), logits=_h(np.asarray(m.forward(ids[:2, :-1]))),
                     params=_h(*[np.asarray(params[k]) for k in sorted(params)])),
                m, lambda e: (np.asarray(e.forward(_ids(Xh, 2, 16))),))


def _optim_tensors(lane):
    """Three float32 parameter tensors and two gradient lists from the
    hashed stream, for the bare optimizer lanes."""
    shapes = ((16, 8), (16,), (3, 16))
    params = [_hw(s, f"{lane}:p{k}", -0.5, 0.5) for k, s in enumerate(shapes)]
    grads = [[_hw(s, f"{lane}:g{j}{k}", -1.0, 1.0) for k, s in enumerate(shapes)] for j in range(2)]
    return params, grads


@lane("optim-sgd")
def _(ml, X, yc, yr, Xh=None):
    """SGD with momentum, Nesterov, coupled weight decay, a clip on the
    step and a constant schedule with warmup; and a second optimizer with
    dampening. No lane above constructs SGD."""
    T = ml.training
    p1, g1 = _optim_tensors("sgd-nesterov")
    o1 = T.SGD(p1, lr=1e-2, momentum=0.9, dampening=0.0, nesterov=True, weight_decay=0.01,
               lr_schedule=T.ConstantLR(1e-2, warmup_steps=2))
    n1 = [np.float64(o1.step(g, max_norm=1.0)) for g in g1]
    p2, g2 = _optim_tensors("sgd-dampening")
    o2 = T.SGD(p2, lr=1e-2, momentum=0.9, dampening=0.5, nesterov=False)
    n2 = [np.float64(o2.step(g)) for g in g2]
    return _fit(dict(nesterov=_h(*p1), nesterov_norms=_h(np.asarray(n1)),
                     dampening=_h(*p2), dampening_norms=_h(np.asarray(n2))))


@lane("optim-adam-clip")
def _(ml, X, yc, yr, Xh=None):
    """Adam (coupled decay) under a warmup-linear schedule, AdamW at zero
    decay through the two-microbatch accumulated step, and a bare
    clip_grad_norm_ whose pre-clip norm and scaled gradients are hashed."""
    T = ml.training
    p1, g1 = _optim_tensors("adam")
    o1 = T.Adam(p1, lr=1e-3, weight_decay=0.01,
                lr_schedule=T.WarmupLinearLR(1e-3, warmup_steps=2, total_steps=8, min_lr=0.0))
    n1 = [np.float64(o1.step(g, max_norm=1.0)) for g in g1]
    p2, g2 = _optim_tensors("adamw0")
    o2 = T.AdamW(p2, lr=1e-3, weight_decay=0.0, accumulation_steps=2)
    n2 = np.float64(o2.step_accumulated(g2, 256))     # T = 256 at A = 2 is a split clause 9.2 admits
    p3, g3 = _optim_tensors("clip")
    total = np.float64(T.clip_grad_norm_(g3[0], 1.0))
    return _fit(dict(adam=_h(*p1), adam_norms=_h(np.asarray(n1)), adamw0=_h(*p2), adamw0_norm=_h(n2),
                     clip_norm=_h(total), clipped=_h(*g3[0])))


@lane("cross-entropy-arms")
def _(ml, X, yc, yr, Xh=None):
    """reduction='none', 'sum' with an explicit divisor, label smoothing
    above zero (a different kernel by the loss contract), and an
    ignore_index that masks rows. Logits from the hashed stream, targets
    from the fixture bytes."""
    T = ml.training
    logits = _hw((64, 16), "ce:logits", -2.0, 2.0)
    t = (_ids(X, 1, 64).reshape(64) % 16).astype(np.int32)
    tm = t.copy(); tm[::4] = 0
    none = T.cross_entropy(logits, t, reduction="none")
    ssum = T.cross_entropy(logits, t, reduction="sum", num_items=17)
    smooth, dsmooth = T.cross_entropy(logits, t, label_smoothing=0.1, return_grad=True)
    masked, dmasked = T.cross_entropy(logits, tm, ignore_index=0, return_grad=True)
    return _fit(dict(none=_h(np.asarray(none)), sum=_h(np.float64(ssum)),
                     smooth=_h(np.float64(smooth)), dsmooth=_h(np.asarray(dsmooth)),
                     masked=_h(np.float64(masked)), dmasked=_h(np.asarray(dmasked))))


# -- classical

@lane("kmeans-random")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KMeans(n_clusters=8, init="random", n_init=3, random_state=3).fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_)), m, _km_probe(X, Xh))


@lane("kmeans-array")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KMeans(n_clusters=8, init="array", init_centroids=np.ascontiguousarray(X[:8]), random_state=3).fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_)), m, _km_probe(X, Xh))


@lane("kmeans-weighted")
def _(ml, X, yc, yr, Xh=None):
    w = _hw((X.shape[0],), "kmeans:sample_weight", 0.5, 1.5)
    m = ml.KMeans(n_clusters=8, random_state=3).fit(X, sample_weight=w)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_)), m, _km_probe(X, Xh))


@lane("dbscan-brute-l1")
def _(ml, X, yc, yr, Xh=None):
    m = ml.DBSCAN(eps=0.9, min_samples=5, metric="manhattan", algorithm="brute",
                  prediction_data=True).fit(X[:6000, :4])
    return _fit(dict(labels=_h(m.labels_)), m, lambda e: (e.predict(Xh[:256, :4]),))


@lane("dbscan-weighted")
def _(ml, X, yc, yr, Xh=None):
    w = _hw((6000,), "dbscan:sample_weight", 0.5, 1.5)
    m = ml.DBSCAN(eps=0.9, min_samples=5, prediction_data=True).fit(X[:6000, :4], sample_weight=w)
    return _fit(dict(labels=_h(m.labels_)), m, lambda e: (e.predict(Xh[:256, :4]),))


def _kde_lane(kernel, metric):
    # cosine distance is undefined at the origin and the estimator refuses
    # an all-zero row by name (DEVIATION 553); `ties` has one in four
    # columns, so the cosine lane shifts its rows by a constant, which
    # moves every row off the origin and leaves the pathology (ties) intact
    shift = np.float32(8.0) if metric == "cosine" else np.float32(0.0)

    def body(ml, X, yc, yr, Xh=None):
        m = ml.KernelDensity(bandwidth=0.7, kernel=kernel, metric=metric).fit(X[:4096, :4] + shift)
        return _fit(dict(scores=_h(m.score_samples(X[4096:4352, :4] + shift))), m,
                    lambda e: (e.score_samples(Xh[:256, :4] + shift),))
    body.__doc__ = f"KernelDensity kernel={kernel!r} metric={metric!r}: leaving the gaussian-euclidean pair leaves the fused kernel."
    return body


# five kernels crossed with five metrics, one pair per lane; every kernel
# and every metric other than the kde lane's gaussian-euclidean appears once
for _kernel, _metric in (("tophat", "sqeuclidean"), ("epanechnikov", "l1"), ("exponential", "chebyshev"),
                         ("linear", "cosine"), ("cosine", "minkowski")):
    lane(f"kde-{_kernel}-{_metric}")(_kde_lane(_kernel, _metric))


@lane("kde-weighted")
def _(ml, X, yc, yr, Xh=None):
    w = _hw((4096,), "kde:sample_weight", 0.5, 1.5)
    m = ml.KernelDensity(bandwidth=0.7).fit(X[:4096, :4], sample_weight=w)
    return _fit(dict(scores=_h(m.score_samples(X[4096:4352, :4]))), m, lambda e: (e.score_samples(Xh[:256, :4]),))


@lane("pca-full-whiten")
def _(ml, X, yc, yr, Xh=None):
    """The dense SVD fit (svd_solver='full'), a different algorithm from
    the covariance eigensolve, with the whitened transform."""
    m = ml.PCA(n_components=4, whiten=True, svd_solver="full").fit(X)
    return _fit(dict(components=_h(m.components_), variance=_h(m.explained_variance_), transform=_h(m.transform(X[:256]))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("ols-no-intercept")
def _(ml, X, yc, yr, Xh=None):
    m = ml.LinearRegression(fit_intercept=False).fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("ols-weighted")
def _(ml, X, yc, yr, Xh=None):
    w = _hw((X.shape[0],), "ols:sample_weight", 0.5, 1.5)
    m = ml.LinearRegression().fit(X, yr, sample_weight=w)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("ridge-no-intercept")
def _(ml, X, yc, yr, Xh=None):
    m = ml.Ridge(alpha=1.0, fit_intercept=False).fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("logistic-l1")
def _(ml, X, yc, yr, Xh=None):
    """An L1 penalty selects OWL-QN, a different solver from L-BFGS."""
    m = ml.LogisticRegression(penalty="l1", C=1.0, max_iter=50).fit(X, yc)
    return _fit(dict(coef=_h(m.coef_), proba=_h(m.predict_proba(X[:256]))), m, lambda e: (e.predict_proba(Xh[:256]),))


@lane("logistic-multiclass")
def _(ml, X, yc, yr, Xh=None):
    """More than two classes route through the softmax loss (the
    logistic-multiclass lane, docs/lanes/BRIEF_logistic_multiclass_2026-09-14.md
    section 5). Three classes from the fixture's own labels: the binary
    rule plus one for rows whose column 5 is above its median. Column 5 is
    one no fixture perturbs (denormal rewrites 0-2, dupes 14-15, the labels
    read 3-4), and the median split keeps all three classes present on
    every fixture (a sign split leaves `negative` with two)."""
    y3 = (yc + (X[:, 5] > np.median(X[:, 5]))).astype(np.int32)
    m = ml.LogisticRegression(max_iter=50).fit(X, y3)
    return _fit(dict(coef=_h(m.coef_), proba=_h(m.predict_proba(X[:256]))), m,
                lambda e: (e.predict_proba(Xh[:256]), e.predict(Xh[:256])))


@lane("logistic-elasticnet")
def _(ml, X, yc, yr, Xh=None):
    m = ml.LogisticRegression(penalty="elasticnet", l1_ratio=0.5, max_iter=50).fit(X, yc)
    return _fit(dict(coef=_h(m.coef_), proba=_h(m.predict_proba(X[:256]))), m, lambda e: (e.predict_proba(Xh[:256]),))


@lane("logistic-unpenalized-no-intercept")
def _(ml, X, yc, yr, Xh=None):
    m = ml.LogisticRegression(penalty=None, fit_intercept=False, max_iter=50).fit(X, yc)
    return _fit(dict(coef=_h(m.coef_), proba=_h(m.predict_proba(X[:256]))), m, lambda e: (e.predict_proba(Xh[:256]),))


@lane("elasticnet-l2end-no-intercept")
def _(ml, X, yc, yr, Xh=None):
    """l1_ratio=0 is the end where the soft threshold never fires; no
    intercept skips the pre and post processing passes."""
    m = ml.ElasticNet(alpha=0.01, l1_ratio=0.0, fit_intercept=False, max_iter=200).fit(X, yr)
    return _fit(dict(coef=_h(m.coef_), predict=_h(m.predict(X[:256]))), m, lambda e: (e.predict(Xh[:256]),))


@lane("svc-linear")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SVC(C=1.0, kernel="linear", max_iter=200).fit(X[:2000], yc[:2000])
    return _fit(dict(decision=_h(m.decision_function(X[2000:2256])), predict=_h(m.predict(X[2000:2256]))),
                m, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("svc-poly")
def _(ml, X, yc, yr, Xh=None):
    """SVC(kernel='poly') (lane/cpu-training-small-gaps, 2026-09-15): the
    identical linear Gram, then kernel_methods' polynomial epilogue (one fused
    multiply-add, an ascending repeated product; DEVIATION 1663) at degree 3,
    gamma 0.1 and coef0 1.0. Sized like the svc lane."""
    m = ml.SVC(C=1.0, kernel="poly", degree=3, gamma=0.1, coef0=1.0, max_iter=200).fit(X[:2000], yc[:2000])
    return _fit(dict(decision=_h(m.decision_function(X[2000:2256])), predict=_h(m.predict(X[2000:2256]))),
                m, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("svr-linear")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SVR(C=1.0, kernel="linear", epsilon=0.1, max_iter=200).fit(X[:2000], yr[:2000])
    return _fit(dict(predict=_h(m.predict(X[2000:2256]))), m, lambda e: (e.predict(Xh[:256]),))


def _knn_lane(**kw):
    def body(ml, X, yc, yr, Xh=None):
        m = ml.NearestNeighbors(n_neighbors=8, **kw).fit(X[:4096])
        d, i = m.kneighbors(X[4096:4160])
        return _fit(dict(dist=_h(d), idx=_h(i)), m, lambda e: e.kneighbors(Xh[:64]))
    body.__doc__ = f"NearestNeighbors {kw}: a different distance op, or (rbc) a different search entry."
    return body


for _name, _kw in (("sqeuclidean", dict(metric="sqeuclidean")), ("manhattan", dict(metric="manhattan")),
                   ("chebyshev", dict(metric="chebyshev")), ("cosine", dict(metric="cosine")),
                   ("minkowski-p3", dict(metric="minkowski", p=3)), ("rbc", dict(algorithm="rbc"))):
    lane(f"knn-{_name}")(_knn_lane(**_kw))


@lane("knn-clf-distance")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KNeighborsClassifier(n_neighbors=8, weights="distance").fit(X[:4096], yc[:4096])
    return _fit(dict(predict=_h(m.predict(X[4096:4160])), proba=_h(m.predict_proba(X[4096:4160]))),
                m, lambda e: (e.predict(Xh[:64]), e.predict_proba(Xh[:64])))


@lane("knn-reg-distance")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KNeighborsRegressor(n_neighbors=8, weights="distance").fit(X[:4096], yr[:4096])
    return _fit(dict(predict=_h(m.predict(X[4096:4160]))), m, lambda e: (e.predict(Xh[:64]),))


def _radius_lane(**kw):
    def body(ml, X, yc, yr, Xh=None):
        index, q = X[:4096], X[4096:4160]
        r = _radius_for(index, q, kw["metric"] if kw["metric"] in ("manhattan", "chebyshev") else "euclidean")
        m = ml.RadiusNeighbors(radius=r, **kw).fit(index)
        lens, dd, ii = _ragged(m.radius_neighbors(q, sort_results=True))
        return _fit(dict(radius=_h(np.float32(r)), counts=_h(lens), dist=_h(dd), idx=_h(ii)),
                    m, lambda e: _ragged(e.radius_neighbors(Xh[:64], sort_results=True)))
    body.__doc__ = f"RadiusNeighbors {kw}: the ball-cover L1, Linf and Lp arms; the radius is _radius_for's, scaled to the fixture."
    return body


for _name, _kw in (("manhattan", dict(metric="manhattan")), ("chebyshev", dict(metric="chebyshev")),
                   ("minkowski-p3", dict(metric="minkowski", p=3))):
    lane(f"radius-{_name}")(_radius_lane(**_kw))


@lane("standard-scaler-no-mean")
def _(ml, X, yc, yr, Xh=None):
    m = ml.StandardScaler(with_mean=False).fit(X)
    t = m.transform(X[:256])
    return _fit(dict(var=_h(m.var_), scale=_h(m.scale_), transform=_h(t), inverse=_h(m.inverse_transform(t))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("standard-scaler-no-std")
def _(ml, X, yc, yr, Xh=None):
    m = ml.StandardScaler(with_std=False).fit(X)
    t = m.transform(X[:256])
    return _fit(dict(mean=_h(m.mean_), transform=_h(t), inverse=_h(m.inverse_transform(t))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("minmax-scaler-clip")
def _(ml, X, yc, yr, Xh=None):
    """A non-default range and the clamp, which only clip=True runs; the
    held-out rows reach outside the fitted range, so the clamp fires."""
    m = ml.MinMaxScaler(feature_range=(-1.0, 1.0), clip=True).fit(X[:4096])
    t = m.transform(X[4096:4352])
    return _fit(dict(scale=_h(m.scale_), min=_h(m.min_), transform=_h(t), inverse=_h(m.inverse_transform(t))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("spectral-precomputed")
def _(ml, X, yc, yr, Xh=None):
    """The precomputed affinity entry, a different binding from the
    nearest-neighbors graph; the affinity is _affinity's, host-derived in
    fixed order so the input is the same bytes on every box."""
    A = _affinity(X[:1000, :4])
    m = ml.SpectralClustering(n_clusters=4, affinity="precomputed", random_state=3,
                              prediction_data=True).fit(A)
    # infer (lane/spectral-predict, 2026-09-15, DEVIATION 2860): predict on
    # the affinity of 256 held-out rows to the 1000 training rows, under
    # _affinity's rule with the training matrix's threshold. The batch part
    # reads the same matrix from the estimator.
    Ah = _cross_affinity(Xh[:256, :4], X[:1000, :4])
    m._identity_heldout_affinity = Ah
    return _fit(dict(affinity=_h(A), labels=_h(m.labels_)), m, lambda e: (e.predict(Ah),))


@lane("holtwinters-multiplicative")
def _(ml, X, yc, yr, Xh=None):
    """The multiplicative decomposition: divide where additive subtracts,
    throughout the fit and the forecast."""
    series = (np.cumsum(X[:512, 0]) + 50.0).astype(np.float32)
    series = series - series.min() + 1.0
    m = ml.ExponentialSmoothing(series, seasonal="multiplicative", seasonal_periods=12).fit()
    return _fit(dict(forecast=_h(m.forecast(24))), m,
                lambda e: _same_bytes("forecast(h)", e.forecast(FORECAST_HORIZON),
                                      "forecast(h, index=0)", e.forecast(FORECAST_HORIZON, index=0)))


@lane("kpss")
def _(ml, X, yc, yr, Xh=None):
    """kpss_test, which had no lane: the undifferenced series and a
    seasonally differenced one with the statistic returned."""
    series = np.ascontiguousarray(X[:512, :4])      # (n_obs, n_series), a series per column
    flat = ml.kpss_test(series, d=0)
    diffed, stat = ml.kpss_test(series, d=1, D=1, s=12, return_statistic=True)
    return _fit(dict(flags=_h(np.asarray(flat)), diffed=_h(np.asarray(diffed)), stat=_h(np.asarray(stat))))


def _select_d_series(values):
    series = np.array(values, dtype=np.float32, order="C", copy=True)
    # Mixed stationary, integrated and twice-integrated series exercise
    # first-round success, a later success and the d_max fallback.
    series[:, 1] = np.cumsum(series[:, 1], dtype=np.float32)
    series[:, 2] = np.cumsum(np.cumsum(series[:, 2], dtype=np.float32), dtype=np.float32)
    return series


@lane("select-d")
def _(ml, X, yc, yr, Xh=None):
    """Public differencing-order selection, including caller-supplied seasonal D.

    This chooses ordinary d, not automatic seasonal D (which is not exposed).
    Hash the per-series results with both seasonal and nonseasonal inputs.
    """
    series = _select_d_series(X[:512, :4])
    return _fit({f"D{D}": _h(np.asarray(ml.select_d(series, D=D, s=12 if D else 0,
                                                   d_max=2-D))) for D in (0, 1)})


@lane("arima-011")
def _(ml, X, yc, yr, Xh=None):
    """Differencing and a moving-average term, and no intercept (trend
    None on a differenced series is k=0)."""
    series = np.ascontiguousarray(X[:512, :4].T)
    m = ml.ARIMA(order=(0, 1, 1)).fit(series)
    return _fit(dict(ma=_h(m.ma_), sigma2=_h(m.sigma2_), forecast=_h(m.forecast(24))),
                m, lambda e: _same_bytes("forecast(h)", e.forecast(FORECAST_HORIZON),
                                         "predict(n_obs, n_obs + h)",
                                         e.predict(e.n_obs_, e.n_obs_ + FORECAST_HORIZON)))


@lane("arima-seasonal-c")
def _(ml, X, yc, yr, Xh=None):
    """A seasonal AR block (period 4, so rd = 5 stays inside the
    implemented Kalman kernel; period 12 is refused by name at rd = 13)
    and an explicit intercept."""
    series = np.ascontiguousarray(X[:512, :4].T)
    m = ml.ARIMA(order=(1, 0, 0), seasonal_order=(1, 0, 0, 4), trend="c").fit(series)
    return _fit(dict(ar=_h(m.ar_), sar=_h(m.sar_), mu=_h(m.mu_), sigma2=_h(m.sigma2_), forecast=_h(m.forecast(24))),
                m, lambda e: _same_bytes("forecast(h)", e.forecast(FORECAST_HORIZON),
                                         "predict(n_obs, n_obs + h)",
                                         e.predict(e.n_obs_, e.n_obs_ + FORECAST_HORIZON)))


#: The exogenous regressor columns of series `b` in the arima-exog lanes:
#: two fixture columns per series, `4 + 2b` and `5 + 2b` (columns 4 to 11,
#: which no fixture perturbs: denormal rewrites 0 to 2, dupes the last two).
_ARIMA_EXOG_PER_SERIES = 2


def _arima_exog_block(X, rows):
    """`(4, rows, 2)` float32: the regressors of the four series over the
    first `rows` rows of `X`, series major then time then regressor, which is
    `ARIMA`'s exog layout (DEVIATION 996)."""
    k = _ARIMA_EXOG_PER_SERIES
    return np.ascontiguousarray(
        np.stack([X[:rows, 4 + k * b: 4 + k * b + k] for b in range(4)]).astype(np.float32))


def _arima_exog_inputs(X):
    """The arima-exog lanes' training inputs: the arima lane's four series of
    512 observations, their regressors over the same 512 rows, and the
    regressors' next 24 rows (X[512:536]) for the train column's forecast(24),
    so no train cell reads the held-out draw."""
    series = np.ascontiguousarray(X[:512, :4].T)
    exog = _arima_exog_block(X, 512)
    fut24 = _arima_exog_block(X[512:], 24)
    return series, exog, fut24


def _arima_exog_probe(e, Xh):
    """The infer probe: FORECAST_HORIZON steps with the regressors' future
    values from the held-out rows, through `forecast(h, exog)` and
    `predict(n_obs, n_obs + h, exog)`, held to the same bytes."""
    F = _arima_exog_block(Xh, FORECAST_HORIZON)
    return _same_bytes("forecast(h, exog)", e.forecast(FORECAST_HORIZON, exog=F),
                       "predict(n_obs, n_obs + h, exog)",
                       e.predict(e.n_obs_, e.n_obs_ + FORECAST_HORIZON, exog=F))


@lane("arima-exog")
def _(ml, X, yc, yr, Xh=None):
    """Regression with ARIMA(1, 0, 0) errors and an intercept (lane/arima-exog,
    2026-09-15): two regressors per series, beta estimated by least squares
    before the ARMA start and then jointly with it by the L-BFGS, and added
    to every prediction as the observation intercept."""
    series, exog, fut24 = _arima_exog_inputs(X)
    m = ml.ARIMA(order=(1, 0, 0), trend="c").fit(series, exog)
    return _fit(dict(beta=_h(m.beta_), ar=_h(m.ar_), mu=_h(m.mu_), sigma2=_h(m.sigma2_),
                     forecast=_h(m.forecast(24, exog=fut24))),
                m, lambda e: _arima_exog_probe(e, Xh))


@lane("arima-exog-seasonal")
def _(ml, X, yc, yr, Xh=None):
    """Differenced, seasonal, with two regressors (lane/arima-exog,
    2026-09-15): ARIMA(1, 1, 0)(1, 0, 0)_4, so the regressors are differenced
    beside y, their future values are differenced against their past
    (prepare_future_data) and the forecast is undifferenced after the
    observation intercept; rd = 6, r = 5, the implemented Kalman arms."""
    series, exog, fut24 = _arima_exog_inputs(X)
    m = ml.ARIMA(order=(1, 1, 0), seasonal_order=(1, 0, 0, 4)).fit(series, exog)
    return _fit(dict(beta=_h(m.beta_), ar=_h(m.ar_), sar=_h(m.sar_), sigma2=_h(m.sigma2_),
                     forecast=_h(m.forecast(24, exog=fut24))),
                m, lambda e: _arima_exog_probe(e, Xh))


def _gp_lane(nu, length_scale):
    def body(ml, X, yc, yr, Xh=None):
        k = ml.ConstantKernel(1.0) * ml.Matern(length_scale, nu=nu) + ml.WhiteKernel(0.1)
        m = ml.GaussianProcessRegressor(kernel=k).fit(X[:256, :4], yr[:256])
        mean, std = m.predict(X[256:320, :4], return_std=True)
        return _fit(dict(alpha=_h(m.alpha_), L=_h(m.L_), lml=_h(np.float64(m.log_marginal_likelihood_value_)),
                         mean=_h(mean), std=_h(std)),
                    m, lambda e: e.predict(Xh[:64, :4], return_std=True))
    body.__doc__ = f"Matern nu={nu} length_scale={length_scale}: a kernel node kind the gp lane never launches; the vector length scale is the ARD divide loop."
    return body


@lane("gp-normalize-y")
def _(ml, X, yc, yr, Xh=None):
    """GaussianProcessRegressor(normalize_y=True) (lane/cpu-training-small-gaps,
    2026-09-15): the gp lane's kernel and slices with y centered and scaled by
    StandardScaler's pinned folds before the fit, and the predictive mean and
    std scaled back on the host. The target is shifted by 50 so the mean is
    not near zero."""
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
    y = np.ascontiguousarray(yr[:256] + np.float32(50.0)).astype(np.float32)
    m = ml.GaussianProcessRegressor(kernel=k, normalize_y=True).fit(X[:256, :4], y)
    mean, std = m.predict(X[256:320, :4], return_std=True)
    return _fit(dict(alpha=_h(m.alpha_), L=_h(m.L_), lml=_h(np.float64(m.log_marginal_likelihood_value_)),
                     y_stats=_h(np.float32([m._y_train_mean, m._y_train_std])),
                     mean=_h(mean), std=_h(std)),
                m, lambda e: e.predict(Xh[:64, :4], return_std=True))


def _gp_sample_y_lane(normalize_y):
    def body(ml, X, yc, yr, Xh=None):
        k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
        y = yr[:256]
        if normalize_y:
            y = np.ascontiguousarray(yr[:256] + np.float32(50.0)).astype(np.float32)
        m = ml.GaussianProcessRegressor(kernel=k, normalize_y=normalize_y).fit(X[:256, :4], y)
        s = m.sample_y(X[256:320, :4], n_samples=3, random_state=11)
        _same_bytes("sample_y(n_samples=3)", s, "sample_y(n_samples=3) again",
                    m.sample_y(X[256:320, :4], n_samples=3, random_state=11))
        return _fit(dict(sample=_h(s)), m,
                    lambda e: (e.sample_y(Xh[:64, :4], n_samples=4, random_state=2 ** 40 + 5),))
    body.__doc__ = (f"GaussianProcessRegressor.sample_y (2026-09-15) on the gp{'-normalize-y' if normalize_y else ''} "
                    "lane's fit: train hashes sample_y over 64 rows, 3 draws, held to a second call bit for bit; "
                    "infer hashes 4 draws over 64 held-out rows with a key whose high word is set. The posterior "
                    "covariance is factored by the identical Cholesky at 2^-20 and the normals are position-mapped "
                    "Philox (DEVIATION 2793). Separate lanes so the gp lanes' recorded cells do not move.")
    return body


lane("gp-sample-y")(_gp_sample_y_lane(False))
lane("gp-sample-y-normalize")(_gp_sample_y_lane(True))


def _gp_optimize_lane(restarts):
    def body(ml, X, yc, yr, Xh=None):
        if restarts:
            k = ml.ConstantKernel(1.0) * ml.Matern([1.0, 1.0, 1.0, 1.0], nu=2.5) + ml.WhiteKernel(0.1)
            m = ml.GaussianProcessRegressor(kernel=k, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=2,
                                            random_state=7)
        else:
            k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
            m = ml.GaussianProcessRegressor(kernel=k, optimizer="fmin_l_bfgs_b")
        m.fit(X[:64, :4], yr[:64])
        mean, std = m.predict(X[64:128, :4], return_std=True)
        runs = m._optimizer_runs
        return _fit(dict(theta=_h(np.array(m.kernel_.theta, dtype=np.float64)),
                         params=_h(np.array(m.kernel_._free_values(), dtype=np.float64)),
                         runs=_h(np.array([[r[0], r[1]] for r in runs], dtype=np.int64),
                                 np.array([r[3] for r in runs], dtype=np.float64)),
                         lml=_h(np.float64(m.log_marginal_likelihood_value_)),
                         alpha=_h(m.alpha_), L=_h(m.L_), mean=_h(mean), std=_h(std)),
                    m, lambda e: e.predict(Xh[:64, :4], return_std=True))
    body.__doc__ = ("GaussianProcessRegressor(optimizer='fmin_l_bfgs_b') (lane/gp-optimizer, 2026-09-15): the kernel "
                    "hyperparameters maximize the log marginal likelihood through the identical gradient "
                    "(DEVIATION 2880) and the projected L-BFGS (DEVIATION 2881)"
                    + (", from the kernel's theta and two Philox restarts keyed by random_state=7, over an ARD Matern "
                       "nu=2.5 scaled by a constant plus white noise" if restarts else
                       ", over the gp lane's kernel") +
                    ". 64 training rows of four columns; train hashes the optimized theta, the float32 "
                    "hyperparameters, every run's iteration and evaluation counts and likelihood, the fit and the "
                    "prediction at 64 rows. Separate lanes so the recorded gp cells do not move.")
    return body


lane("gp-optimize")(_gp_optimize_lane(False))
lane("gp-optimize-restarts")(_gp_optimize_lane(True))


for _name, _nu, _ls in (("matern12", 0.5, 1.0), ("matern32", 1.5, 1.0), ("matern52-ard", 2.5, [1.0, 2.0, 0.5, 4.0])):
    lane(f"gp-{_name}")(_gp_lane(_nu, _ls))


@lane("gemm-transposed")
def _(ml, X, yc, yr, Xh=None):
    """OP_NT and OP_TN, the two GEMM ops the gemm-pinned lane (OP_NN) does
    not run."""
    a = np.ascontiguousarray(X[:256]).astype(np.float32)               # 256 x d
    b = np.ascontiguousarray(X[256:384]).astype(np.float32)            # 128 x d
    nt = ml.linalg.matmul(a, b, transpose_b=True, identical=True)      # 256 x 128
    at = np.ascontiguousarray(X[:256].T).astype(np.float32)            # d x 256
    c = np.ascontiguousarray(X[384:384 + at.shape[0]]).astype(np.float32)   # d x d
    tn = ml.linalg.matmul(at, c, transpose_a=True, identical=True)     # 256 x d
    return _fit(dict(nt=_h(nt), tn=_h(tn)))


@lane("linalg-qr")
def _(ml, X, yc, yr, Xh=None):
    """`linalg.qr` (python/mojolearn/_linalg_impl.py) through
    `_mojolearn_linalg_host::qr_r`: the Householder QR's R on BOTH slice
    arms. `host_qr_slice_count` halves from QR_MAX_SLICES while a slice
    would hold fewer than QR_SLICE_ROWS_PER_COL rows per column, so at
    n_cols=5 a 48-row matrix takes the TWO-slice arm (the scratch pass and
    the second reduction over it) and at n_cols=4 a 16-row matrix takes the
    ONE-slice arm. A lane that ran only one of them would read IDENTICAL
    while the other drifted. The third part is rank deficient by
    construction -- a duplicated column -- which puts an exact zero on R's
    diagonal and makes the reflector's sign choice observable."""
    tall = np.ascontiguousarray(X[:48, :5]).astype(np.float32)
    small = np.ascontiguousarray(X[:16, :4]).astype(np.float32)
    dup = np.ascontiguousarray(X[:48, [0, 1, 1, 2]]).astype(np.float32)
    return _fit(dict(two_slice=_h(ml.linalg.qr(tall)),
                     one_slice=_h(ml.linalg.qr(small)),
                     rank_deficient=_h(ml.linalg.qr(dup))))


@lane("linalg-eigh")
def _(ml, X, yc, yr, Xh=None):
    """`linalg.eigh` through `_mojolearn_linalg_host::eigh`: the symmetric
    Jacobi eigensolver under numpy's name and numpy's ASCENDING order, which
    is the reverse of the one `PCA` reads from the same sweep. Three parts.
    A generic Gram matrix. A matrix with a REPEATED eigenvalue (a scaled
    identity block), where the eigenvectors within the degenerate subspace
    are fixed only by the sweep order and the sign flip, so it is the part
    that moves if either changes. And a DIAGONAL matrix, where the sweep
    performs no rotation at all and the answer is the input -- the arm that
    catches an ordering permutation applied unconditionally."""
    g = np.ascontiguousarray(X[:64, :5]).astype(np.float32)
    gram = np.ascontiguousarray(g.T @ g)
    gram = np.ascontiguousarray(((gram + gram.T) * np.float32(0.5)).astype(np.float32))
    rep = np.ascontiguousarray(np.diag(np.array([4.0, 4.0, 4.0, 1.0], np.float32)))
    diag = np.ascontiguousarray(np.diag(np.array([3.0, 0.5, 2.0, 9.0], np.float32)))
    out = {}
    for name, m in (("gram", gram), ("repeated", rep), ("diagonal", diag)):
        w, v = ml.linalg.eigh(m)
        out[name + "_w"] = _h(w)
        out[name + "_v"] = _h(v)
    return _fit(out)


@lane("linalg-svdvals")
def _(ml, X, yc, yr, Xh=None):
    """`linalg.svdvals` through `_mojolearn_linalg_host::svdvals`: the QR
    then the one-sided Jacobi, read DESCENDING. The second part is rank
    deficient (a duplicated column), so its smallest singular value is an
    exact zero and the sort has a tie at the bottom; the third is scaled by
    a power of two, which changes every exponent and no mantissa, so the
    values must scale exactly and the ORDER must not move at all."""
    a = np.ascontiguousarray(X[:48, :5]).astype(np.float32)
    dup = np.ascontiguousarray(X[:48, [0, 1, 1, 2]]).astype(np.float32)
    scaled = np.ascontiguousarray(a * np.float32(0.125))
    return _fit(dict(generic=_h(ml.linalg.svdvals(a)),
                     rank_deficient=_h(ml.linalg.svdvals(dup)),
                     scaled=_h(ml.linalg.svdvals(scaled))))


@lane("metrics-classification")
def _(ml, X, yc, yr, Xh=None):
    """The nineteen metric functions the metrics lane does not call, and
    the averaging, zero-division, normalize, base and beta arms. Scores are
    a fixture column; an all-zero prediction makes the zero-division
    substitute observable."""
    mt = ml.metrics
    n = 3000
    yt = yc[:n]
    s = np.ascontiguousarray(X[:n, 3]).astype(np.float32)
    yp = (s > np.float32(0.0)).astype(np.int32)
    zeros = np.zeros(n, dtype=np.int32)
    lo, hi = float(s.min()), float(s.max())
    prob = np.clip((s - np.float32(lo)) / np.float32(hi - lo), np.float32(1e-3), np.float32(1 - 1e-3)).astype(np.float32)
    labels = np.asarray(ml.KMeans(n_clusters=4, random_state=3).fit(X[:n, :4]).labels_)
    parts = {}
    for avg in (None, "binary", "micro", "macro", "weighted"):
        for fn in (mt.precision_score, mt.recall_score, mt.f1_score):
            parts[f"{fn.__name__}:{avg}"] = _h(np.asarray(fn(yt, yp, average=avg), dtype=np.float64))
    for zd in (0, 1):
        parts[f"precision_zero_division:{zd}"] = _h(np.float64(mt.precision_score(yt, zeros, zero_division=zd)))
    parts["log_loss:mean"] = _h(np.float64(mt.log_loss(yt, prob)))
    parts["log_loss:sum"] = _h(np.float64(mt.log_loss(yt, prob, normalize=False)))
    parts["entropy:nats"] = _h(np.float64(mt.entropy(labels)))
    parts["entropy:base2"] = _h(np.float64(mt.entropy(labels, base=2)))
    parts["v_measure:beta2"] = _h(np.float64(mt.v_measure_score(yt, labels, beta=2.0)))
    parts["roc_auc"] = _h(np.float64(mt.roc_auc_score(yt, s)))
    parts["confusion"] = _h(np.asarray(mt.confusion_matrix(yt, yp)))
    parts["pr_curve"] = _h(*[np.asarray(a) for a in mt.precision_recall_curve(yt, s)])
    parts["mse"] = _h(np.float64(mt.mean_squared_error(yr[:n], yr[:n] * np.float32(0.9))))
    parts["mae"] = _h(np.float64(mt.mean_absolute_error(yr[:n], yr[:n] * np.float32(0.9))))
    parts["rmse"] = _h(np.float64(mt.root_mean_squared_error(yr[:n], yr[:n] * np.float32(0.9))))
    parts["rand"] = _h(np.float64(mt.rand_score(yt, labels)))
    parts["mutual_info"] = _h(np.float64(mt.mutual_info_score(yt, labels)))
    parts["homogeneity"] = _h(np.float64(mt.homogeneity_score(yt, labels)))
    parts["completeness"] = _h(np.float64(mt.completeness_score(yt, labels)))
    parts["kl"] = _h(np.float64(mt.kl_divergence(_exact_prob(64, "kl:P"), _exact_prob(64, "kl:Q"))))
    parts["trustworthiness"] = _h(np.float64(mt.trustworthiness(X[:512, :4], X[:512, :2], n_neighbors=5)))
    parts["silhouette_samples"] = _h(np.asarray(mt.silhouette_samples(X[:n, :4], labels)))
    return _fit(parts)


@lane("metrics-homogeneity-completeness")
def _(ml, X, yc, yr, Xh=None):
    """The public three-score convenience API, including nondefault beta.

    Its result order and beta forwarding were absent from the harness even
    though the individual metrics had lanes. Use all fixture labels and a
    distinct eight-way clustering; global contingency reductions have no
    row-wise batch-invariance promise.
    """
    mt = ml.metrics
    yt = np.ascontiguousarray(yc).astype(np.int32)
    split = ((X[:, 3] > 0).astype(np.int32) + 2 * (X[:, 4] > 0).astype(np.int32)
             + 4 * (X[:, 5] > 0).astype(np.int32)).astype(np.int32)
    parts = {}
    for beta in (0.5, 1.0, 2.0):
        got = np.asarray(mt.homogeneity_completeness_v_measure(yt, split, beta=beta), dtype=np.float64)
        expected = np.asarray((mt.homogeneity_score(yt, split), mt.completeness_score(yt, split),
                               mt.v_measure_score(yt, split, beta=beta)), dtype=np.float64)
        if got.shape != (3,) or got.tobytes() != expected.tobytes():
            raise AssertionError("homogeneity_completeness_v_measure order/beta differs from scalar APIs")
        parts[f"hcv:beta{beta}"] = _h(got)
    return _fit(parts)


@lane("metrics-fowlkes-mallows")
def _(ml, X, yc, yr, Xh=None):
    """fowlkes_mallows_score (lane/cpu-training-small-gaps, 2026-09-15),
    scikit-learn's definition over the integer contingency matrix. A lane
    of its own so metrics-classification's committed train hashes do not
    move. The clusterings are fixed host comparisons of fixture columns,
    no fit: an eight-way split on the signs of columns 3, 4 and 5 against
    the class labels, a relabeled copy (score 1.0), the all-singletons
    split (tk == 0, score 0.0), and a two-row call."""
    mt = ml.metrics
    n = 3000
    yt = np.ascontiguousarray(yc[:n]).astype(np.int32)
    split = ((X[:n, 3] > 0).astype(np.int32) + 2 * (X[:n, 4] > 0).astype(np.int32)
             + 4 * (X[:n, 5] > 0).astype(np.int32)).astype(np.int32)
    parts = dict(
        split=_h(np.float64(mt.fowlkes_mallows_score(yt, split))),
        reversed=_h(np.float64(mt.fowlkes_mallows_score(split, yt))),
        relabeled=_h(np.float64(mt.fowlkes_mallows_score(split, (np.int32(7) - split) * np.int32(3)))),
        singletons=_h(np.float64(mt.fowlkes_mallows_score(yt, np.arange(n, dtype=np.int32)))),
        two_rows=_h(np.float64(mt.fowlkes_mallows_score(yt[:2], split[:2]))),
    )
    return _fit(parts)


@lane("tokenizer")
def _(ml, X, yc, yr, Xh=None):
    """BpeTokenizer (python/mojolearn/tokenizer.py), host integers and
    tables through _mojolearn_tokenizer_host; no float arithmetic, so
    cross-vendor identity is by construction and what this measures is
    that the SAME binary bytes were built on every box. mojolearn ships no
    vocabulary, so the lane loads the synthetic one mojolearn trains itself
    (python/mojolearn/_tokenizer_synthetic.py, 512 ranks) at revision
    LANE_REVISIONS["tokenizer"]. The fixture bytes are the first 4,096 bytes
    of X viewed as bytes (the byte-lm lane's derivation without _ids'
    modulus), encoded with <|endoftext|> allowed, then decoded back; the
    held-out probe encodes Xh's first 4,096 bytes
    (docs/lanes/BRIEF_expose_tokenizer_2026-09-14.md section 3)."""
    tok = ml.tokenizer.BpeTokenizer._synthetic()
    raw = np.ascontiguousarray(X).tobytes()[:4096]
    ids = np.asarray(tok.encode_bytes(raw, allow_endoftext=True), dtype=np.int32)
    back = np.frombuffer(tok.decode_bytes(ids.tolist()), dtype=np.uint8)
    # The round trip is a hashed part, not an assertion (2026-09-15): a
    # sabotaged binary must read DIVERGENT, and a raise would read REFUSED,
    # which the owed check (tools/cpu_identity_gate_check.py owed) does not
    # count as a catch. A production column hashes roundtrip=1 everywhere.
    roundtrip = np.int64(1 if back.tobytes() == raw else 0)
    return _fit(dict(ids=_h(ids), decoded=_h(back), roundtrip=_h(roundtrip), n_vocab=_h(np.int64(tok.n_vocab))),
                tok, lambda e: (np.asarray(e.encode_bytes(np.ascontiguousarray(Xh).tobytes()[:4096],
                                                          allow_endoftext=True), dtype=np.int32),))


@lane("bpe-trainer")
def _(ml, X, yc, yr, Xh=None):
    """BpeVocabularyTrainer (python/mojolearn/tokenizer.py): TRAINING a
    byte-level BPE vocabulary, where the tokenizer lane only applies one.

    Host integers and tables -- counts, ids and one comparison -- with no
    float anywhere in the selection, so this is NOT a cross-vendor claim and
    there is no GPU column to owe: vocabulary training has no GPU path in any
    library. What the cell says is that the same corpus and config produce
    the same vocabulary BYTES on this box as on every other.

    The corpus is the first 4,096 bytes of X viewed as bytes, the same
    derivation the tokenizer lane uses, as ONE document. Both emitted formats
    are hashed, because both are what a user ships beside a model: ours
    (rank<TAB>hex) and the ecosystem's (tokenizer.json). `n_ties_broken` is
    hashed too, and it is the part that matters most -- it is how a reader
    can see the tie-break rule was REACHED on this fixture, without which the
    sabotage below would be inert.

    SABOTAGE: MOJOLEARN_BPE_TRAINER_SABOTAGE=1 reverses ONLY the tie-break
    (largest (left_id, right_id) among the pairs at the top count instead of
    smallest). It moves ALL FIVE parts. MEASURED on x86-64 EPYC, 2026-09-17,
    both fixtures (`bench/results/identity_break/2026-09-17_sabotage-sweep/
    e-python-lanes/`): `base` 302 tokens, 46 merges, 42 ties broken clean
    against 301, 45 and 41 sabotaged; `ties` 309/53/33 against 308/52/31.

    THIS PARAGRAPH USED TO SAY the arm "leaves `n_tokens` and `n_merges`
    alone -- the vocabulary still fills to vocab_size". It does not, and it
    never did on this fixture: `min_frequency=2` exhausts the pairs worth
    merging long before `vocab_size=320`, so the vocabulary's SIZE is an
    output of the merge sequence and not a constant, and one different
    tie-break costs one merge. Read as written, the old sentence invited a
    reader to drop the counters from the cell; the counters move too.

    The two artifact hashes are still the load-bearing parts, for the reason
    the old sentence was reaching for: they are what a user ships beside a
    model, and a counter that happened to agree would not make the
    vocabularies the same. The Mojo trainer carries the same arm as a build
    define, and `pixi run check-bpe-trainer-sabotage` is where THAT one is
    watched failing; this Python door's arm is watched failing in the
    directory above.

    THE DOOR NOW RUNS THE MOJO TRAINER (lane/bpe-builder-native,
    2026-09-18): `BpeVocabularyTrainer` defaults to `bpe_train` in the
    tokenizer host binding and falls back to the Python reference only when
    the binding lacks it. Before (main, Python) vs after (Mojo): IDENTICAL on
    all nine fixtures, `base` still 6ed8b49585df3d85. The binding built with
    -D MOJOLEARN_BPE_TRAINER_SABOTAGE=1 moves every cell (`base` ->
    f5172d25e6499662, the env arm's value) and the env arm now reaches the
    Mojo trainer as `break_ties_high`
    (bench/results/identity_break/2026-09-18_bpe-builder-native/)."""
    raw = np.ascontiguousarray(X).tobytes()[:4096]
    v = ml.tokenizer.BpeVocabularyTrainer(vocab_size=320, min_frequency=2).train([raw])
    ranks = np.frombuffer(v.render_ranks().encode("ascii"), dtype=np.uint8)
    tj = np.frombuffer(v.render_tokenizer_json().encode("ascii"), dtype=np.uint8)
    return _fit(dict(ranks=_h(ranks), tokenizer_json=_h(tj),
                     n_tokens=_h(np.int64(v.n_tokens)),
                     n_merges=_h(np.int64(len(v.merges))),
                     n_ties_broken=_h(np.int64(v.n_ties_broken))))


@lane("bpe-vocabulary")
def _(ml, X, yc, yr, Xh=None):
    """TrainedBpeVocabulary (python/mojolearn/tokenizer.py), which had ZERO
    lanes while the trainer that returns it and the tokenizer it builds each
    had one (lane/tokenized-corpus, 2026-09-18). What a user does with a
    trained vocabulary is WRITE it and USE it, so that is what is hashed:
    the two files as written to disk (`write_ranks`, `write_tokenizer_json`),
    whether each is byte-equal to its `render_*` text, the vocabulary's
    identity (`identity`, the sha256 a model carries), and the ids and bytes
    of Xh's first 4,096 bytes through `tokenizer()` and through the WRITTEN
    rank file loaded back, which must agree.

    The vocabulary is re-made through the public constructor from the
    trainer's tokens, merges and stats, so the object hashed is one the lane
    built itself and not only one the trainer returned.

    SABOTAGE, two arms, both must move it. MEASURED on the M4, one core,
    `--repeats 2`, all nine fixtures DIVERGENT under each
    (bench/results/identity_break/2026-09-18_tokenized-corpus/):
    MOJOLEARN_BPE_TRAINER_SABOTAGE=1 (the trainer's reversed tie-break) moves
    `ranks`, `tokenizer_json`, `identity` and `ids` and leaves `decoded`,
    `roundtrip` and `flags` (a different vocabulary still round trips); the
    tokenizer host build with -D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1 (ids
    written in reverse) moves `ids`, `decoded` and `roundtrip` and leaves the
    files and the identity alone."""
    import tempfile
    TV = ml.tokenizer.TrainedBpeVocabulary
    raw = np.ascontiguousarray(X).tobytes()[:4096]
    held = np.ascontiguousarray(Xh).tobytes()[:4096]
    t = ml.tokenizer.BpeVocabularyTrainer(vocab_size=320, min_frequency=2).train([raw])
    v = TV(list(t.tokens), list(t.merges), dict(t.stats))
    with tempfile.TemporaryDirectory(prefix="ib-bpe-vocabulary-") as d:
        rp, jp = os.path.join(d, "v.ranks.tsv"), os.path.join(d, "v.tokenizer.json")
        v.write_ranks(rp)
        v.write_tokenizer_json(jp)
        with open(rp, "rb") as fh:
            ranks = fh.read()
        with open(jp, "rb") as fh:
            tj = fh.read()
        from_file = ml.tokenizer.BpeTokenizer.from_ranks_file(rp)
        ids_file = np.asarray(from_file.encode_bytes(held), dtype=np.int32)
    tok = v.tokenizer()
    ids = np.asarray(tok.encode_bytes(held), dtype=np.int32)
    back = tok.decode_bytes(ids.tolist())
    ident = json.dumps(v.identity, sort_keys=True).encode("ascii")
    flags = np.asarray([ranks == v.render_ranks().encode("ascii"), tj == v.render_tokenizer_json().encode("ascii"),
                        tok.identity == v.identity, from_file.identity == v.identity,
                        ids.tobytes() == ids_file.tobytes()], dtype=np.int64)
    return _fit(dict(ranks=_h(np.frombuffer(ranks, dtype=np.uint8)), tokenizer_json=_h(np.frombuffer(tj, dtype=np.uint8)),
                     identity=_h(np.frombuffer(ident, dtype=np.uint8)), ids=_h(ids),
                     decoded=_h(np.frombuffer(back, dtype=np.uint8)),
                     roundtrip=_h(np.int64(1 if back == held else 0)), flags=_h(flags)))


@lane("tokenized-corpus")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.lm_corpus (lane/tokenized-corpus, 2026-09-18): a corpus
    tokenized ONCE with a vocabulary trained on it, cached, and read back as
    training batches. The corpus is X's first 16,384 bytes as a file with no
    manifest (one train range); `prepare` trains a 300-rank vocabulary on it
    with `BpeVocabularyTrainer`, cuts 4,096-byte documents at the last 0x0A,
    and writes the pinned id array. Hashed: the id array's bytes, the
    vocabulary identity, the ids of TokenBatches(2, 64) at steps 0, 1 and 7,
    the data schedule the trainer would carry, and flags: a second `prepare`
    reads the SAME cache entry, `tokenizer_for` accepts the model's own
    vocabulary and refuses the synthetic one by name, and the user-vocabulary
    arm (the trained rank file passed back as `vocab=`) lands on the same
    ids.

    SABOTAGE, MEASURED like bpe-vocabulary's (same directory), all nine
    fixtures DIVERGENT under each: MOJOLEARN_BPE_TRAINER_SABOTAGE=1 moves
    `tokens`, `identity`, `batches` and `schedule`; the tokenizer host
    sabotage build moves `tokens`, `batches` and `schedule` (the schedule
    carries the id array's sha256) and leaves `identity`."""
    import tempfile
    LC = ml.lm_corpus
    raw = np.ascontiguousarray(X).tobytes()[:16384]
    with tempfile.TemporaryDirectory(prefix="ib-tokenized-corpus-") as d:
        path = os.path.join(d, "corpus.txt")
        with open(path, "wb") as fh:
            fh.write(raw)
        cache = os.path.join(d, "cache")
        c = LC.prepare(path, cache_dir=cache, vocab_size=300, vocab_sample_bytes=16384, document_bytes=4096)
        again = LC.prepare(path, cache_dir=cache, vocab_size=300, vocab_sample_bytes=16384, document_bytes=4096)
        user = LC.prepare(path, cache_dir=cache, vocab=str(c.vocabulary_path), document_bytes=4096)
        with open(os.path.join(c.tokens_dir, "tokens.i32"), "rb") as fh:
            tokens = np.frombuffer(fh.read(), dtype="<i4")
        b = c.batches(2, 64)
        steps = np.concatenate([b.ids(k) for k in (0, 1, 7)])
        schedule = json.dumps(b.data_schedule(), sort_keys=True).encode("ascii")
        ok = LC.tokenizer_for(b.data_schedule(), c.vocabulary_path)
        try:
            LC.require_vocabulary(b.data_schedule(), ml.tokenizer.BpeTokenizer._synthetic())
            refused = 0
        except ValueError as exc:
            refused = int("vocabulary mismatch" in str(exc))
        flags = np.asarray([isinstance(c, LC.TokenizedCorpus), again.tokens_dir == c.tokens_dir,
                            user.manifest["sha256"] == c.manifest["sha256"],
                            ok.identity == c.vocabulary, refused, isinstance(b, LC.TokenBatches)], dtype=np.int64)
        ident = json.dumps(c.vocabulary, sort_keys=True).encode("ascii")
    return _fit(dict(tokens=_h(tokens), identity=_h(np.frombuffer(ident, dtype=np.uint8)), batches=_h(steps),
                     schedule=_h(np.frombuffer(schedule, dtype=np.uint8)), flags=_h(flags)))


# --------------------------------------------- the Hugging Face loading path
# THE NAMESPACE THAT HAD NO LANE (lane/models-namespace-lanes, 2026-09-19).
# `tools/verification_matrix.py --json` enumerates the public surface and
# names the lanes that reach each entry. On 2026-09-19 it reported FOURTEEN
# entries with no lane of any kind and every one of them was
# `mojolearn.models`: CausalLM, Checkpoint, SafetensorsFile, Tokenizer,
# ParallelCausalLM, plan_for, pretokenize and pattern_name, each counted
# twice where the package re-exports a submodule's name.
#
# WHAT WAS ALREADY THERE, AND WHY IT IS NOT A LANE. Two things, and neither
# produces a cell a record carries. `python/mojolearn/tests/
# test_models_loader.py` is 53 assertions over synthetic checkpoints, which
# say the loader agrees WITH ITSELF on one box and nothing about two boxes
# agreeing on a byte. `python -m mojolearn._verify_causal_lm` (lane/causal-
# lm-distributed-proof) is better than that -- it captures a loaded model's
# logits, prefill, decode and greedy ids for eight architectures and
# compares a CPU capture against a GPU one -- but it is a separate tool with
# its own file format: `--diff` cannot read it, the verification matrix
# cannot see it, `verify --all` does not run it, it has no fixture axis, and
# its own header says its fault controls "alter composition, not native
# arithmetic", so it carries no arithmetic negative control at all. The
# lanes below put the same claim where every other estimator's lives, with
# nine hostile fixtures under it and a build sabotage measured against each.
# They reuse that tool's own synthetic checkpoints
# (`python/mojolearn/_causal_lm_fixtures.py`) so the two are comparable.
#
# THREE LANES, CUT WHERE THE CODE PATHS CUT and not one per symbol:
#
#   hf-checkpoint  THE FILE, AND WHAT ITS SHAPE IS TAKEN TO BE.
#                  `SafetensorsFile` and `Checkpoint` -- the header, the
#                  mmap, the five admitted dtypes, the sharded index and the
#                  refusals -- and `plan_for`, the option matrix that decides
#                  which tensor name carries which weight. They are the two
#                  halves of `CausalLM.load` that run before one weight
#                  reaches a block and they read the same thing: a directory.
#   hf-tokenizer   THE TEXT. `Tokenizer.from_pretrained` over a
#                  `tokenizer.json`, the three pre-tokenization patterns
#                  (`pretokenize`, `pattern_name`), encode, decode, the
#                  added tokens and the SentencePiece refusals.
#   hf-causal-lm   THE ARITHMETIC. The assembled model's logits, one decode
#                  step held against the prefill, and greedy `generate`, on
#                  the column's own device (`device="auto"`, the public
#                  default: a GPU column runs the GPU blocks, a CPU column
#                  the `neural_inference` classes over
#                  `_mojolearn_neural_host`).
#
# `ParallelCausalLM` GETS NO LANE, deliberately. It raises
# NotImplementedError unless `_backend.vendor()` is 'cuda' or 'hip' -- in
# `load` before a file is opened and again in `__init__` -- so on the CPU
# column and on Apple there is no arithmetic to hash and its cell would be a
# REFUSED, which a column total cannot tell from a build that did not run.
# What pins it is python/mojolearn/tests/test_parallel_causal_lm.py, which
# is a TEST and not a lane and says so in its header.

#: The Llama-shaped checkpoint these lanes write: two layers, grouped-query
#: attention (4 heads over 2 KV heads, so q and k/v are not the same shape),
#: head_dim 8, a vocabulary that is not a power of two and an UNTIED head, so
#: every name the llama row of the option matrix lists exists in the file and
#: no two of them are the same shape by accident.
#: THE CHECKPOINTS THESE LANES WRITE ARE THE PACKAGE'S OWN, not a fourth
#: spelling of them. `python/mojolearn/_causal_lm_fixtures.py` already builds
#: a tiny config and its tensors for every architecture the option matrix
#: admits, and `python -m mojolearn._verify_causal_lm` (lane/causal-lm-
#: distributed-proof) captures a loaded model from exactly those; so do the
#: tests. Restating them here would be a second statement of the synthetic
#: checkpoint to drift from, and would make this lane's cells incomparable
#: with that tool's captures.
#:
#: `(architecture, tie_word_embeddings)`, the eight rows `family_fixture`
#: builds and `_verify_causal_lm.ARCHITECTURES` captures. The tied rows are
#: the ones whose `head_name` is None, where the head IS the embedding table
#: and `_read_weights` reads no `lm_head.weight`. `hf-checkpoint`'s `flags`
#: holds this tuple to that module's, so the two cannot drift apart quietly.
HF_ARCHITECTURES = (("llama", False), ("llama", True), ("mistral", False),
                    ("qwen2", False), ("qwen3", False), ("phi3", False),
                    ("mamba", True), ("mamba2", True))

#: Configs the option matrix must REFUSE and the words it must refuse them
#: with, one per clause a caller can trip without writing a checkpoint. Each
#: is the fixtures' Llama config with the named fields replaced, so the only
#: reason any of them is refused is the clause it names; the odd `head_dim`
#: replaces the whole shape, because 9 cannot also satisfy
#: `n_heads*head_dim == hidden_size`.
HF_PLAN_REFUSALS = (
    (dict(model_type="gpt2"), "is not in the option matrix"),
    (dict(model_type="gemma"), "model_type 'gemma'"),
    (dict(model_type="gemma2"), "model_type 'gemma2'"),
    (dict(num_local_experts=8), "mixture of experts"),
    (dict(rope_scaling={"rope_type": "linear", "factor": 2.0}), "rope_scaling"),
    (dict(hidden_act="gelu"), "SwiGLU"),
    (dict(use_sliding_window=True, max_window_layers=1), "sliding-window-alternating"),
    (dict(layer_types=["full_attention", "sliding_attention"]), "alternating attention kinds"),
    (dict(attention_dropout=0.1), "inference has no dropout"),
    (dict(num_key_value_heads=3), "is not a multiple of it"),
    (dict(num_attention_heads=5), "must equal hidden_size"),
    (dict(hidden_size=18, num_attention_heads=2, num_key_value_heads=2, head_dim=9,
          intermediate_size=36), "head_dim must be even"),
    (dict(num_hidden_layers="two"), "an integer is required"),
    (dict(vocab_size=None), "the field is required"),
    (dict(sliding_window=-1), "an integer width or null"),
)


def _hf_bytes(a):
    """A tensor's raw bytes as a uint8 array, whatever its dtype and rank.
    Hashing BYTES and not values is what lets one part carry five dtypes at
    once, and what makes a NaN payload a hashed fact rather than a value
    that compares unequal to itself."""
    return np.frombuffer(np.ascontiguousarray(np.asarray(a)).tobytes(), dtype=np.uint8)


def _hf_refused(fn, want):
    """1 when `fn()` raised with `want` in the message, else 0.

    A HASHED FLAG AND NOT AN ASSERTION, for the tokenizer lane's reason: a
    raise reads REFUSED, which the owed check does not count as a catch,
    while a hash that moves is a catch. The `want` strings are the
    load-bearing words of each refusal, so a refusal that stops naming its
    reason moves the cell. The MESSAGE is not hashed here because every
    safetensors and loader refusal carries the temporary directory's path,
    which differs run to run and would read MOVED on one box for no
    arithmetic reason."""
    try:
        fn()
    except Exception as exc:  # noqa: BLE001
        return 1 if want in str(exc) else 0
    return 0


def _hf_plan_row(plan):
    """A plan as the table a checkpoint reader and a weight packer both index
    with: the kind, the sizes, every interface option with its value, the
    three top-level names and, per layer, `(block key, checkpoint name, row
    slice)`. Objects that are identities and not values (the `HFConfig`, the
    family's plan function) are left out; an address is not a fact about a
    model, and `_h` would hash the address."""
    return {
        "model_type": plan.model_type, "kind": plan.kind, "n_layers": plan.n_layers,
        "d_model": plan.d_model, "vocab_size": plan.vocab_size,
        "tie_embeddings": plan.tie_embeddings, "norm_eps": repr(plan.norm_eps),
        "max_position_embeddings": plan.max_position_embeddings,
        "n_heads": getattr(plan, "n_heads", None), "n_kv_heads": getattr(plan, "n_kv_heads", None),
        "head_dim": getattr(plan, "head_dim", None),
        "intermediate": getattr(plan, "intermediate", None),
        "block_options": {k: repr(v) for k, v in sorted(plan.block_options.items())},
        "block_kwargs": {k: repr(v) for k, v in sorted(plan.block_kwargs.items())},
        "embed_name": plan.embed_name, "norm_name": plan.norm_name, "head_name": plan.head_name,
        "checkpoint_names": plan.checkpoint_names(),
        "layer_weights": [[[k, n, list(rows) if rows is not None else None]
                           for k, n, rows in plan.layer_weights(i)] for i in range(plan.n_layers)],
    }


def _hf_text(s):
    """Text as a uint8 array: `_h` refuses a text dtype, and rightly."""
    return np.frombuffer(s.encode("utf-8"), dtype=np.uint8)


@lane("hf-checkpoint")
def _(ml, X, yc, yr, Xh=None):
    """`mojolearn.models.SafetensorsFile`, `Checkpoint` and `plan_for`: what
    a checkpoint directory HOLDS and what its shape is taken to be, the two
    halves of `CausalLM.load` that run before one weight reaches a block.

    WHY ONE LANE AND NOT TWO. They have one input (a directory), they fail
    together (a name the plan asks for and the file does not hold is one
    error, raised once by `_read_weights` across both), and neither has a
    GPU path to owe: the reader is a header, an mmap and one copy per
    tensor, and the option matrix is integers and strings. Two cells that
    always move together are one cell with extra bookkeeping.

    THE PARTS.

    `dtypes` is every dtype the reader admits, read back from a file the
    lane wrote, concatenated in NAME ORDER: F32, I32 and I64 as stored, F16
    widened by `safetensors.widen_f16`'s own bit construction, and BF16
    widened through `mojolearn.lowbit.widen_bf16` (the linalg host binding
    on a CPU column, the linalg extension on a GPU one). The F16 and BF16
    payloads are the FIXTURE'S BYTES read as bit patterns rather than as
    values, so the widening is asked about subnormals, infinities and NaN
    payloads on every fixture instead of about whatever nine well-behaved
    numbers a hand-written case would have carried. A NaN hashes as its
    bits, which is the point: a widening that dropped a payload moves it.

    `bits` is that same BF16 tensor read with `bf16="bits"` -- the form a
    block accepts directly, a `lowbit.BF16Weight` at rank 2 -- whose uint16
    must be the stored bits unchanged.

    `sharded` is the same tensors written as THREE shards with an index and
    read back through `Checkpoint`. `flags` carries what makes it a part and
    not a duplicate: that the sharded read is byte-equal to the single-file
    read and that `names()` agrees across the two layouts.

    `plan` is `plan_for` over all EIGHT rows of HF_ARCHITECTURES, hashed as
    each one's whole derived table (see `_hf_plan_row`). An off-by-one in a
    row slice is not a Python bug, it is a different model, and `phi3`'s
    fused `qkv_proj`/`gate_up_proj` is the one row that has a row slice at
    all. `flags` holds HF_ARCHITECTURES to `_verify_causal_lm.ARCHITECTURES`
    and the architecture set to `models.FAMILIES` minus the two SentencePiece
    families and the `smollm` alias, so a family added to the matrix and not
    to this lane is a flag that drops to 0 rather than a silent omission.

    `refusals` is the MESSAGES of HF_PLAN_REFUSALS joined, not merely the
    flags: `plan_for`'s refusals name the model_type, the field and the
    value and carry no path, so the words are hashable and a refusal that
    stops naming its reason moves the cell. `caught` is the same refusals as
    flags. The READER's refusals do carry a path, so they appear in `flags`
    only.

    `flags` also carries the RE-EXPORTS: `models.SafetensorsFile` must BE
    `models.safetensors.SafetensorsFile` and `models.plan_for` must BE
    `models.config.plan_for`, or the public names document objects that are
    not the ones that run (`language-model-config`'s alias part, for the
    same reason).

    SABOTAGE, AND THE ONE BINDING THIS LANE STANDS ON. The only native call
    in the whole lane is `lowbit.widen_bf16`'s `from_bf16` in the linalg
    host binding; everything else is a header parse, a memoryview cast and
    integer bookkeeping in Python. So the family's GENERIC arm is the wrong
    one and its OWN arm is the right one, and both were measured on the M4,
    one core, `--repeats 2`, nine fixtures, 2026-09-19
    (bench/results/identity_break/2026-09-19_models-namespace):

      -D MOJOLEARN_HOST_SABOTAGE=1 alone: THE CELL DOES NOT MOVE, on any
      fixture, in any of the eight parts (`base` stays 2e4e083eaea46d80).
      That arm is gemm_oracle's descending leaf and the factorizations; a
      bit shift has no accumulation order to perturb, so there is nothing
      in it for this lane to catch. A set built with it alone is not a
      negative control for this lane, exactly as it is not one for
      `bpe-trainer` or `lowbit-conversions`.

      -D MOJOLEARN_LOWBIT_CONVERT_SABOTAGE=1 (the four conversion seams'
      own arm, in GATE_SABOTAGE_OWN_DEFINES["linalg"]): the cell MOVES on
      every fixture, `base` 2e4e083eaea46d80 -> fb18c7b40e708e5b, through
      `dtypes` and `sharded` -- the two parts that carry the widened BF16
      tensor.

    THE OTHER SIX PARTS DO NOT MOVE UNDER EITHER ARM and cannot. `bits` is
    the STORED uint16, which no conversion touches by construction; `infos`,
    `plan`, `refusals`, `caught` and `flags` are integers, names and
    sentences with no float anywhere near them. What guards those is the
    recorded hash itself, which reads DIVERGENT on every column at once if
    an offset, a dtype table entry, a tensor name or a refusal sentence
    changes -- the guard `language-model-config`'s `derived` and `flags`
    parts already rest on."""
    from mojolearn import _causal_lm_fixtures as fx
    from mojolearn import _verify_causal_lm as capture
    S, C, plan_for = ml.models.SafetensorsFile, ml.models.Checkpoint, ml.models.plan_for
    st = ml.models.safetensors
    cfg0 = fx._llama_config()
    raw = np.ascontiguousarray(X).tobytes()
    bits = np.frombuffer(raw[:512], dtype="<u2")
    f32 = np.frombuffer(raw[512:512 + 256 * 4], dtype="<f4")
    i32 = np.frombuffer(raw[2048:2048 + 64 * 4], dtype="<i4")
    i64 = np.frombuffer(raw[4096:4096 + 32 * 8], dtype="<i8")
    tensors = {"t.f32": ("F32", (16, 16), f32.tobytes()), "t.f16": ("F16", (16, 16), bits.tobytes()),
               "t.bf16": ("BF16", (16, 16), bits.tobytes()), "t.i32": ("I32", (8, 8), i32.tobytes()),
               "t.i64": ("I64", (8, 4), i64.tobytes()), "t.f32s": ("F32", (), f32[:1].tobytes())}
    with tempfile.TemporaryDirectory(prefix="ib-hf-checkpoint-") as d:
        one = fx._write_checkpoint(os.path.join(d, "one"), cfg0, tensors, shards=1)
        many = fx._write_checkpoint(os.path.join(d, "many"), cfg0, tensors, shards=3)
        sf = S(os.path.join(one, "model.safetensors"))
        single = C.open(one)
        sharded = ml.models.safetensors.Checkpoint.open(many)
        try:
            names = sorted(sf.names())
            dtypes = np.concatenate([_hf_bytes(single.read(n)) for n in names])
            sharded_bytes = np.concatenate([_hf_bytes(sharded.read(n)) for n in sorted(sharded.names())])
            packed = single.read("t.bf16", bf16="bits")
            bitsback = _hf_bytes(getattr(packed, "bits", packed))
            infos = _hf_text("|".join(f"{t.name},{t.dtype},{t.shape},{t.nbytes},{t.size}"
                                      for t in (sf.info(n) for n in names)))
            empty = os.path.join(d, "empty")
            os.makedirs(empty, exist_ok=True)
            loose = fx._write_checkpoint(os.path.join(d, "loose"), cfg0, tensors, shards=2)
            os.remove(os.path.join(loose, st.INDEX_NAME))
            bad = os.path.join(d, "bad")
            os.makedirs(bad, exist_ok=True)
            badfile = st.write_safetensors(os.path.join(bad, "x.safetensors"),
                                           {"t.f64": ("F64", (4,), b"\0" * 32)})
            flags = np.asarray([
                dtypes.tobytes() == sharded_bytes.tobytes(),
                names == sorted(sharded.names()),
                bitsback.tobytes() == bits.tobytes(),
                "t.f32" in single and "t.nope" not in single,
                _hf_refused(lambda: single.read("t.nope"), "no tensor"),
                _hf_refused(lambda: single.read("t.f32", bf16="raw"), "bf16 must be"),
                _hf_refused(lambda: st.SafetensorsFile(badfile).read("t.f64"), "refuses by name"),
                _hf_refused(lambda: C.open(empty), "holds neither"),
                _hf_refused(lambda: C.open(loose), "not an index"),
                _hf_refused(lambda: C.open(os.path.join(d, "nope")), "does not exist"),
                ml.models.SafetensorsFile is st.SafetensorsFile,
                ml.models.Checkpoint is st.Checkpoint,
                ml.models.plan_for is ml.models.config.plan_for,
                tuple(HF_ARCHITECTURES) == tuple(capture.ARCHITECTURES),
                sorted(set(a for a, _ in HF_ARCHITECTURES) | {"gemma", "gemma2", "smollm"})
                == sorted(ml.models.FAMILIES),
            ], dtype=np.int64)
        finally:
            sf.close()
            single.close()
            sharded.close()
    plans = [_hf_plan_row(plan_for(fx.family_fixture(arch, tied)[0]))
             for arch, tied in HF_ARCHITECTURES]
    messages, caught = [], []
    for over, want in HF_PLAN_REFUSALS:
        cfg = fx._llama_config()
        cfg.update(over)
        try:
            ml.models.config.plan_for(cfg)
            messages.append("<no refusal>")
            caught.append(0)
        except Exception as exc:  # noqa: BLE001
            messages.append(f"{type(exc).__name__}: {exc}")
            caught.append(1 if (want in str(exc)
                                and isinstance(exc, ml.models.UnsupportedModel)) else 0)
    return _fit(dict(dtypes=_h(dtypes), bits=_h(bitsback), sharded=_h(sharded_bytes),
                     infos=_h(infos), plan=_h(_hf_text(json.dumps(plans, sort_keys=True))),
                     refusals=_h(_hf_text("\n".join(messages))),
                     caught=_h(np.asarray(caught, dtype=np.int64)), flags=_h(flags)))


@lane("hf-tokenizer")
def _(ml, X, yc, yr, Xh=None):
    """`mojolearn.models.Tokenizer` over a `tokenizer.json`, with
    `pretokenize` and `pattern_name`: the THREE pre-tokenization patterns a
    byte-level BPE family can carry, which is the one thing
    `mojolearn.tokenizer.BpeTokenizer` does not parameterize (DEVIATION
    2960).

    THE VOCABULARY IS OURS. mojolearn ships none, so the lane trains one on
    the FIXTURE'S OWN BYTES with `BpeVocabularyTrainer` (the `bpe-trainer`
    lane's trainer, the Mojo one since lane/bpe-builder-native) and hands
    its tokens and merge pairs to `Tokenizer`. The vocabulary therefore
    moves with the fixture and so do the ids: `ties` trains on repeated
    bytes and `hashed` on a flat stream, and the two cut differently.

    WHAT RUNS WHERE, which is why this lane can move at all. For
    `pattern="gpt2"`, `_encode_ordinary` goes through
    `BpeTokenizer.from_token_bytes`, the compiled door in the tokenizer host
    binding. For `llama3` and `qwen2` the cut is Python over the byte codes
    (`_pretoken_end_llama`) and the merge is `_tokenizer_synthetic._bpe`,
    the Python reference the Mojo merge is held to. So `gpt2_ids` and
    `json_ids` are the parts with a build behind them, and `llama3_ids` and
    `qwen2_ids` pin the Python cut.

    THE PARTS. `bounds` is `pretokenize` on one hostile text under all three
    patterns -- contractions in both cases, a digit run (three at a time for
    llama3, one for qwen2, uncut for gpt2), a non-ASCII word, a run of
    spaces, CRLF and a trailing tab -- so the digit rule and the
    case-insensitive contraction are REACHED and not merely available.
    `names` is `pattern_name` over every spelling in `PATTERNS` plus one
    that is not a pattern, as indices into the sorted name list, which makes
    the two GPT-2 spellings' collapse onto one name a hashed fact.

    `gpt2_ids`, `llama3_ids` and `qwen2_ids` encode the FIXTURE'S OWN first
    4,096 bytes, and `text_ids` the hostile text; see the comment at `doc`
    for why the ids cannot be the hostile text's. `json_ids` encodes the
    same document through a tokenizer loaded from a `tokenizer.json` the
    lane wrote, held in `flags` to the directly-constructed one byte for
    byte, which is the claim `from_pretrained` makes. `specials` encodes a
    document carrying an added token BOTH WAYS (`allow_special` False, then
    True), and `decoded` is the round trip back to bytes.

    `flags` also carries the SENTENCEPIECE REFUSALS by name (a Unigram
    model, `byte_fallback`, a Metaspace pre-tokenizer, an unknown Split
    regex, a directory holding only `tokenizer.model`), the id outside the
    vocabulary, and the re-export: `models.Tokenizer` must BE
    `models.tokenizer.Tokenizer`.

    SABOTAGE: TWO ARMS OF THE TOKENIZER FAMILY, AND NEITHER ALONE REACHES
    EVERY PART. Measured on the M4, one core, `--repeats 2`, nine fixtures,
    2026-09-19 (bench/results/identity_break/2026-09-19_models-namespace);
    the cell moves under each, and both are in this family's
    GATE_SABOTAGE_OWN_DEFINES, so the gate's set carries both:

      -D MOJOLEARN_TOKENIZER_HOST_SABOTAGE=1, the ENCODER arm (it reverses
      the ids of every encoded document), moves `gpt2_ids`, `text_ids`,
      `json_ids`, `specials`, `decoded` and `flags` -- everything that goes
      through the compiled door, `flags` included, because a reversed
      encoder breaks the round trip the flags hold. `base`
      a66a961b530381ae -> 4bb26860b4f41795.

      -D MOJOLEARN_BPE_TRAINER_SABOTAGE=1, the TRAINER's arm (a reversed
      merge tie-break), moves `gpt2_ids`, `llama3_ids`, `qwen2_ids`,
      `json_ids` and `specials` -- every id part, because it moves the
      VOCABULARY the ids are drawn from, the Python patterns' ids with the
      compiled one's. `base` a66a961b530381ae -> dc2af787603953e6.

    `bounds` and `names` move under NEITHER and cannot: the pre-tokenizer
    boundaries and the pattern-name lookup are Python over byte codes, with
    no binding of any kind behind them, and `names` does not even read the
    vocabulary. `text_ids` does not move under the trainer arm, and that is
    a measurement: the hostile text is plain ASCII, and a vocabulary trained
    on float32 bytes has no merge that applies to it, so a different
    vocabulary gives the same ids. `decoded` does not move under the trainer
    arm either, because a differently-encoded document still round trips.
    What guards those four is the recorded hash.

    The generic -D MOJOLEARN_HOST_SABOTAGE=1 reaches NOTHING in the
    tokenizer binding, which is why this family has its own defines at all;
    a set built with it alone leaves this cell where it found it."""
    T = ml.models.Tokenizer
    tk = ml.models.tokenizer
    raw = np.ascontiguousarray(X).tobytes()[:8192]
    v = ml.tokenizer.BpeVocabularyTrainer(vocab_size=320, min_frequency=2).train([raw])
    tokens = list(v.tokens)
    merges = [(tokens[a], tokens[b]) for a, b, _ in v.merges]
    text = b"IT'S 12345 caf\xc3\xa9 hello  world\r\n\n  x-y \t"
    bounds = np.concatenate([np.asarray(ml.models.tokenizer.pretokenize(text, p), dtype=np.int64)
                             for p in ("gpt2", "llama3", "qwen2")])
    known = sorted(tk.PATTERNS)
    names = np.asarray([known.index(ml.models.tokenizer.pattern_name(s))
                        for _, spellings in sorted(tk.PATTERNS.items()) for s in spellings]
                       + [-1 if tk.pattern_name("not a pattern") is None else 0], dtype=np.int64)
    # THE DOCUMENT IS THE FIXTURE'S OWN BYTES and the hostile text above is
    # not, and the difference cost this lane its fixture axis once. Encoding
    # only `text` (plain ASCII) through a vocabulary trained on float32 bytes
    # reaches NO merge -- every merge in that vocabulary is over high byte
    # values ASCII never carries -- so the ids were the raw bytes and `base`
    # and `ties` hashed the SAME cell, a983a295bf677e0b on both, measured
    # 2026-09-19 before this line existed. Ids that cannot move with the
    # fixture cannot move under the trainer's arm either, which is the
    # negative control this lane rests on. So the three id parts encode the
    # first 4,096 bytes of X (the `tokenizer` lane's own derivation), where
    # the merges apply, and `text` keeps the pre-tokenizer parts, where the
    # patterns differ.
    doc = np.ascontiguousarray(X).tobytes()[:4096]
    ids = {}
    for p in ("gpt2", "llama3", "qwen2"):
        ids[p] = np.asarray(T(tokens, merges, pattern=p).encode_bytes(doc), dtype=np.int32)
    gpt2 = T(tokens, merges, pattern="gpt2")
    text_ids = np.asarray(gpt2.encode_bytes(text), dtype=np.int32)
    decoded = np.frombuffer(gpt2.decode_bytes(ids["gpt2"].tolist()), dtype=np.uint8)
    cut = tk.pretokenize(text, "gpt2")
    pieces = [text[a:b] for a, b in zip(cut, cut[1:])]
    spelled = ml.tokenizer._byte_to_char()
    eot, eot_id = "<|endoftext|>", len(tokens)

    def spell(bs):
        return "".join(spelled[b] for b in bs)

    def tokenizer_json(regex, model_over=None, **over):
        vocab = {spell(t): i for i, t in enumerate(tokens)}
        vocab[eot] = eot_id
        model = dict(type="BPE", dropout=None, unk_token=None, continuing_subword_prefix=None,
                     end_of_word_suffix=None, fuse_unk=False, byte_fallback=False,
                     ignore_merges=False, vocab=vocab,
                     merges=[[spell(a), spell(b)] for a, b in merges])
        model.update(model_over or {})
        pre = ({"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                "use_regex": True} if regex is None else
               {"type": "Sequence", "pretokenizers": [
                   {"type": "Split", "pattern": {"Regex": regex}, "behavior": "Isolated",
                    "invert": False},
                   {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                    "use_regex": False}]})
        out = dict(version="1.0", truncation=None, padding=None,
                   added_tokens=[dict(id=eot_id, content=eot, single_word=False, lstrip=False,
                                      rstrip=False, normalized=False, special=True)],
                   normalizer=None, pre_tokenizer=pre, post_processor=None,
                   decoder={"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                            "use_regex": False},
                   model=model)
        out.update(over)
        return out

    def write(root, data):
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "tokenizer.json"), "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        return root

    with tempfile.TemporaryDirectory(prefix="ib-hf-tokenizer-") as d:
        plain = write(os.path.join(d, "gpt2"), tokenizer_json(None))
        llama = write(os.path.join(d, "llama3"), tokenizer_json(tk.PATTERNS["llama3"][0]))
        loaded = T.from_pretrained(plain)
        loaded_llama = ml.models.tokenizer.Tokenizer.from_pretrained(llama)
        json_ids = np.asarray(loaded.encode_bytes(doc), dtype=np.int32)
        str_ids = np.asarray(loaded.encode(text.decode("utf-8", "replace"),
                                           add_special_tokens=False), dtype=np.int32)
        marked = doc + eot.encode() + doc[:8]
        plain_ids = np.asarray(loaded.encode_bytes(marked, allow_special=False), dtype=np.int32)
        special_ids = np.asarray(loaded.encode_bytes(marked, allow_special=True), dtype=np.int32)
        specials = np.concatenate([plain_ids, special_ids])
        sp = write(os.path.join(d, "unigram"), tokenizer_json(None, {"type": "Unigram"}))
        bf = write(os.path.join(d, "bytefallback"), tokenizer_json(None, {"byte_fallback": True}))
        ms = write(os.path.join(d, "metaspace"),
                   tokenizer_json(None, pre_tokenizer={"type": "Metaspace",
                                                       "replacement": "▁"}))
        odd = write(os.path.join(d, "oddregex"), tokenizer_json("(?i:'zz)|\\p{L}+"))
        onlymodel = os.path.join(d, "onlymodel")
        os.makedirs(onlymodel, exist_ok=True)
        with open(os.path.join(onlymodel, "tokenizer.model"), "wb") as fh:
            fh.write(b"\0")
        flags = np.asarray([
            json_ids.tobytes() == ids["gpt2"].tobytes(),
            str_ids.tobytes() == text_ids.tobytes(),
            np.asarray(loaded_llama.encode_bytes(doc),
                       dtype=np.int32).tobytes() == ids["llama3"].tobytes(),
            loaded.pattern == "gpt2" and loaded_llama.pattern == "llama3",
            loaded.n_ranks == len(tokens) and loaded.n_vocab == len(tokens) + 1,
            [bytes(p) for p in loaded.pretokenize(text)] == pieces,
            loaded.decode_bytes(special_ids.tolist()) == marked,
            eot_id in special_ids.tolist() and eot_id not in plain_ids.tolist(),
            _hf_refused(lambda: T.from_pretrained(sp), "Unigram"),
            _hf_refused(lambda: T.from_pretrained(bf), "byte_fallback"),
            _hf_refused(lambda: T.from_pretrained(ms), "Metaspace"),
            _hf_refused(lambda: T.from_pretrained(odd), "not one of the known patterns"),
            _hf_refused(lambda: T.from_pretrained(onlymodel), "SentencePiece family"),
            _hf_refused(lambda: T(tokens, merges, pattern="gpt3"), "unknown pattern"),
            _hf_refused(lambda: tk.pretokenize(text, "gpt3"), "unknown pattern"),
            _hf_refused(lambda: gpt2.decode_bytes([eot_id + 99]), "outside"),
            ml.models.Tokenizer is tk.Tokenizer,
        ], dtype=np.int64)
    return _fit(dict(bounds=_h(bounds), names=_h(names), gpt2_ids=_h(ids["gpt2"]),
                     llama3_ids=_h(ids["llama3"]), qwen2_ids=_h(ids["qwen2"]),
                     text_ids=_h(text_ids), json_ids=_h(json_ids), specials=_h(specials),
                     decoded=_h(decoded), flags=_h(flags)),
                gpt2, lambda e: (np.asarray(e.encode_bytes(np.ascontiguousarray(Xh).tobytes()[:4096]),
                                            dtype=np.int32),))


#: The ids `hf-causal-lm` runs: 4 rows of 12 tokens from the fixture bytes,
#: modulo the fixtures' vocabulary, and 6 tokens decoded past them. Four
#: rows so the batch part has rows to split, and a length the prefix part
#: can cut at 1, 7 and 11.
HF_LM_BATCH, HF_LM_LEN, HF_LM_NEW = 4, 12, 6

#: The fixtures' vocabulary, the one every architecture in HF_ARCHITECTURES
#: is built at (`_causal_lm_fixtures._llama_config`); the ids are taken
#: modulo it so one id stream feeds all eight models.
HF_LM_VOCAB = 64


@lane("hf-causal-lm")
def _(ml, X, yc, yr, Xh=None):
    """`mojolearn.models.CausalLM`: a checkpoint directory loaded and RUN, on
    the column's own device.

    `device="auto"` is the public default and is what this lane passes, so
    the cell means what a user means by it: a GPU column assembles
    `TransformerBlock`/`Mamba1Block` and the training extension's embedding,
    RMS norm and head, and a CPU column assembles
    `TransformerBlockInference`/`Mamba1BlockInference` and `_CpuPrimitives`
    over `_mojolearn_neural_host`. The claim is the one the package exists
    to make: the same checkpoint gives the same logits on every vendor.

    EIGHT MODELS, the rows of HF_ARCHITECTURES, each assembled from the
    package's own synthetic checkpoint (`_causal_lm_fixtures.family_fixture`,
    the same one `python -m mojolearn._verify_causal_lm` captures) written
    as two shards. They are not eight spellings of one thing: `llama` tied
    and untied differ in whether `head_name` is None and a `lm_head.weight`
    is read at all, `mistral` carries a sliding window, `qwen2` the three
    attention biases, `qwen3` the q/k norms, `phi3` the fused projections
    split by rows, and the two Mamba rows take the mamba1 and mamba2 block
    classes instead of the transformer.

    THE PARTS. `every` is all eight forwards over the same ids, concatenated
    in HF_ARCHITECTURES order: the part that says the assembly of each row
    is right. `logits` is the Llama row's forward alone, so a moved `every`
    can be attributed. `generate` is greedy decoding, the prompt as one
    prefill and every later token as one `step`; it is the only public entry
    where argmax ties are visible (`_argmax_last` breaks them to the lowest
    index, a scan with a strict `>`), and on the `ties` fixture that rule is
    REACHED. `step` is the decode logits alone. `params` hashes
    `parameters()` in CHECKPOINT-NAME order, which is the name mapping read
    back: a weight that landed under the wrong block key comes back under
    the wrong name and moves it.

    `flags` carries the STEP-AGAINST-PREFILL equality -- a fresh state, the
    first L-1 tokens prefilled, then one `step`, held bitwise against
    position L-1 of the one-shot forward -- which is the property a serving
    system rests on and which `--step-full` measures for the blocks. It is a
    flag and not an assertion so a build where it fails reads DIVERGENT
    rather than REFUSED. It also holds the refusals (`device="tpu"`, an
    unknown `weight_format`, `max_positions` past the config's own, a state
    allocated on another model, float ids, a `greedy=False` decode, a
    `max_tokens` past the capacity) and the re-export: `models.CausalLM`
    must BE `models.causal_lm.CausalLM`.

    THE WEIGHTS ARE NOT THE FIXTURE'S. They come from the fixtures module's
    LCG, the same bytes on every machine, because `wide` puts 1e4 in every
    cell and `denormal` puts subnormals there, and either one inside a
    softmax makes `CausalLM.__init__`'s finiteness check raise -- the cell
    would read REFUSED on every fixture but `base` and the lane would be
    measuring the fixture's magnitude rather than the arithmetic. WHAT THE
    FIXTURE MOVES here is the TOKEN IDS, so every fixture decodes a
    different sequence through the same weights.

    SABOTAGE. On a CPU column, `-D MOJOLEARN_HOST_SABOTAGE=1` on the neural
    host family (gemm_oracle's descending leaf, the arm `mlp`,
    `transformer` and the Mamba lanes read DIVERGENT under) moves the cell
    on every fixture: MEASURED on the M4, one core, `--repeats 2`, nine
    fixtures, 2026-09-19 (bench/results/identity_break/
    2026-09-19_models-namespace), `base` 13d9b4ba89461093 ->
    9b87625cbfe914f6, through `every`, `logits` and `step`, and
    the infer and batch columns move with it.

    THREE PARTS DO NOT MOVE UNDER IT. `params` cannot: those are the bytes
    the reader handed over, which no arithmetic touches. `flags` cannot
    either: the arm perturbs the fold on BOTH sides of the
    step-against-prefill comparison at once, so the equality survives it,
    and every other flag is a refusal no build define reaches.

    `generate` NOT MOVING IS A MEASUREMENT AND NOT A DESIGN, and it is the
    same finding `language-model-config`'s `next` part carries: the arm
    changes the fold ORDER, the resulting logit changes are far below the
    gap between the top two logits at every position of this fixture, and
    greedy decoding is an argmax, so the ids come out the same. A lane
    whose ONLY part were `generate` would be a lane that cannot fail under
    this arm; that is why the logits are hashed beside it."""
    from mojolearn import _causal_lm_fixtures as fx
    M = ml.models
    ids = _ids(X, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)
    probe_ids = _ids(Xh, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)
    A = ml.Array.from_buffer
    with tempfile.TemporaryDirectory(prefix="ib-hf-causal-lm-") as d:
        roots, models = [], []
        for arch, tied in HF_ARCHITECTURES:
            cfg, tensors = fx.family_fixture(arch, tied)
            root = fx._write_checkpoint(os.path.join(d, f"{arch}-{int(tied)}"), cfg, tensors)
            roots.append(root)
            models.append(ml.models.causal_lm.CausalLM.load(root))
        every = np.concatenate([_hf_bytes(np.asarray(one.forward(A(ids)))) for one in models])
        llama, m = roots[0], models[0]
        mamba = models[-2]  # ("mamba", True): the tied head, no lm_head.weight
        logits = np.asarray(m.forward(A(ids)))
        gen = np.asarray(m.generate(A(ids), HF_LM_NEW), dtype=np.int32)
        state = m.allocate_state(HF_LM_BATCH, HF_LM_LEN)
        m.forward(A(np.ascontiguousarray(ids[:, :HF_LM_LEN - 1])), state)
        step = np.asarray(m.step(A(np.ascontiguousarray(ids[:, HF_LM_LEN - 1:])), state))
        params = np.concatenate([_hf_bytes(w) for _, w in sorted(m.parameters().items())])
        other = mamba.allocate_state(HF_LM_BATCH, HF_LM_LEN)
        flags = np.asarray([
            step.tobytes() == np.ascontiguousarray(logits[:, HF_LM_LEN - 1, :]).tobytes(),
            gen[:, :HF_LM_LEN].tobytes() == ids.tobytes(),
            all(one.unused_names == [] for one in models),
            m.plan.head_name is not None and mamba.plan.head_name is None,
            [one.device for one in models] == [m.device] * len(models),
            all(one.weight_format == "float32" for one in models),
            sorted(m.parameters()) == sorted(m.plan.checkpoint_names()),
            [one.plan.kind for one in models]
            == ["transformer"] * 6 + ["mamba1", "mamba2"],
            _hf_refused(lambda: M.CausalLM.load(llama, device="tpu"), "device must be"),
            _hf_refused(lambda: M.CausalLM.load(llama, weight_format="fp8"), "weight_format must be"),
            _hf_refused(lambda: M.CausalLM.load(llama, max_positions=4096),
                        "max_position_embeddings"),
            _hf_refused(lambda: m.step(A(np.ascontiguousarray(ids[:, :1])), other),
                        "belongs to another model"),
            _hf_refused(lambda: m.forward(A(np.ascontiguousarray(ids.astype(np.float32)))),
                        "ids must be integer"),
            _hf_refused(lambda: m.generate(A(ids), 1, greedy=False), "greedy=True"),
            _hf_refused(lambda: m.allocate_state(HF_LM_BATCH, 10 ** 6), "max_tokens"),
            ml.models.CausalLM is ml.models.causal_lm.CausalLM,
        ], dtype=np.int64)
    return _fit(dict(every=_h(every), logits=_h(logits), generate=_h(gen), step=_h(step),
                     params=_h(params), flags=_h(flags)),
                m, lambda e: (np.asarray(e.forward(A(probe_ids))),))


@lane("cross-val")
def _(ml, X, yc, yr, Xh=None):
    """cross_val_score, which had no lane: three unshuffled folds of the
    sklearn-style boosting regressor (cross-validation clones through
    get_params, which the classical estimators do not implement), scored
    by the adapter's own score."""
    scores = ml.model_selection.cross_val_score(_gbdt(ml.GradientBoostingRegressor, n_estimators=8, max_depth=4), X, yr, cv=3)
    return _fit(dict(scores=_h(np.asarray(scores, dtype=np.float64))))


class _FoldClf:
    """A classifier stand-in for `split_descriptor`, which refuses to guess.
    The descriptor reads only `_estimator_type`; nothing is fitted."""
    _estimator_type = "classifier"

    def get_params(self, deep=False):
        return {}


@lane("cross-val-folds")
def _(ml, X, yc, yr, Xh=None):
    """THE FOLD PARTITION ITSELF, AND WHAT IT FAILS TO PIN
    (lane/data-ordering-determinism, 2026-09-16).

    Data ordering is link 3 of the reproducible-pipeline chain
    (docs/lanes/PLAN_cross_vendor_llm.md). For the neural path it is
    deliberately outside our boundary: the byte-LM trainer does not fetch or
    reorder a corpus and takes a caller-supplied `data_schedule` descriptor
    instead. Fold assignment looked like the one piece of data ordering that
    IS ours end to end, and no lane hashed it.

    It turned out to be ours only halfway, which is the finding this lane
    carries. `model_selection._default_folds` NEVER READS X. The stratified
    branch is a function of the label sequence; the KFold branch is a function
    of `len(y)` alone, its folds being contiguous blocks of positions. So the
    four fold-index parts below CANNOT MOVE under a row permutation that
    preserves the label sequence, and the KFold two cannot move under any
    permutation at all, while the rows the estimator is fitted on change
    completely. Measured on 2048 rows, both classes 1024, 1010 label runs: a
    within-class rotation of all 2048 rows moved 0 of 4 index hashes and 4 of
    4 content hashes. That is why `content` is a part and not a comment.

    `descriptor` is the repair: `model_selection.split_descriptor` hashes X and
    y IN ARRIVAL ORDER beside the fold assignment, so a reproduction recipe has
    something that moves when the order does. It is the cross-validation
    analogue of `data_schedule`, and like `data_schedule` it records what is
    outside the boundary rather than pretending to control it.

    The `cross-val` lane above hashes the SCORES of a fitted boosting
    regressor. It would move if the folds moved, so it covers the fold
    assignment BY ARGUMENT; what it cannot do is say so anywhere a reader can
    check, because it needs a GPU or a host binding to produce a cell at all
    and reads REFUSED on a CPU-only install (`gather_rows_bytes`, measured
    2026-09-16). This lane is pure Python over the labels and the split count:
    no RNG, no seed, no native call, nothing to key. It produces a cell on
    EVERY column, including a CPU-only wheel.

    `partition` is a HASHED part and not an assertion, for the tokenizer lane's
    reason: a raise reads REFUSED, which the owed check does not count as a
    catch, while a hash that moves is a catch. It is 1 when every fold is
    nonempty, train and test are disjoint, train is exactly the complement, and
    the test blocks hold every row exactly once.

    NEGATIVE CONTROL: MOJOLEARN_FOLD_ORDER_SABOTAGE=1 with
    MOJOLEARN_HOST_ALLOW_SABOTAGE=1 rotates the row-to-fold assignment by one.
    Every invariant `partition` tests still passes and every fold keeps its
    size; only the assignment differs. So `partition` stays 1 and the four fold
    hashes MOVE. MEASURED both ways on 2026-09-16: with the switch defined and
    never called (the state this lane was found in) the arm is INERT and all
    nine parts hold still, which is a check that cannot fail; with it wired
    into `_default_folds` all eight order parts move and `partition` holds."""
    ms = ml.model_selection
    n = 2048
    Xn = np.ascontiguousarray(X[:n])
    parts, ok = {}, 1
    for tag, labels, clf in (("strat", np.ascontiguousarray(yc[:n]).tolist(), True),
                             ("kfold", np.ascontiguousarray(yr[:n]).tolist(), False)):
        for splits in (3, 5):
            folds = list(ms._default_folds(labels, splits, clf))
            train = [np.asarray(a, dtype=np.int64) for a, _ in folds]
            test = [np.asarray(b, dtype=np.int64) for _, b in folds]
            parts[f"{tag}{splits}"] = _h(*(train + test))
            # WHAT THE INDEX HASH MISSES: the rows themselves, in the order the
            # estimator is handed them.
            parts[f"{tag}{splits}content"] = _h(*[Xn[b] for b in test])
            if (len(folds) != splits
                    or any(a.size == 0 or b.size == 0 for a, b in zip(train, test))
                    or any(np.intersect1d(a, b).size for a, b in zip(train, test))
                    or any(a.size + b.size != n for a, b in zip(train, test))
                    or not np.array_equal(np.sort(np.concatenate(test)),
                                          np.arange(n, dtype=np.int64))):
                ok = 0
    parts["partition"] = _h(np.int64(ok))
    d = ms.split_descriptor(Xn, np.ascontiguousarray(yc[:n]), estimator=_FoldClf(), cv=5)
    parts["descriptor"] = _h(np.frombuffer(
        "|".join(f"{k}={d[k]}" for k in sorted(d)).encode("ascii"), dtype=np.uint8))
    return _fit(parts)


# ---------------------------------------------------------------- lanes (2026-09-14 evening, workstream D, the doors)
# The eight families that had oracles and no door (the claim-surface census,
# section 4) plus the six training primitives and the KMeans arms, given
# bindings and classes on lane/expose-d; the lane bodies were written there
# as docs/lanes/LANE_BODY_<family>.py and merged here by the harness's owner
# without changing their arithmetic. Every derived input follows the
# fixed-order host rule (`_affinity`, `_hw`, `_ids`, `_seq`).

def _cauchy_spd(P):
    Q = P.astype(np.float64)
    d2 = np.zeros((Q.shape[0], Q.shape[0]), dtype=np.float64)
    for j in range(Q.shape[1]):
        c = Q[:, j]
        d2 += (c[:, None] - c[None, :]) ** 2
    A = 1.0 / (1.0 + d2)
    A[np.arange(Q.shape[0]), np.arange(Q.shape[0])] += 1.0
    return np.ascontiguousarray(A.astype(np.float32))


@lane("cholesky")
def _(ml, X, yc, yr, Xh=None):
    """Cholesky (python/mojolearn/_cholesky_impl.py) through _mojolearn_gp:
    potrf, logdet and potrs on a 256 x 256 Cauchy-kernel SPD matrix at the
    profile's pinned ridge. The GP lane already covers the same kernels
    behind a kernel matrix; this lane covers the door and the bare solve."""
    A = _cauchy_spd(X[:256, :4])
    c = ml.Cholesky().fit(A)
    assert c.info_ == 0, "cholesky lane: the Cauchy matrix did not factor (info=%d)" % c.info_
    B = np.ascontiguousarray(np.stack([yr[:256], yr[256:512]], 1).astype(np.float32))
    return _fit(dict(L=_h(c.L_), logdet=_h(np.float64(c.logdet_)), info=_h(np.int64(c.info_)),
                     nb=_h(np.int64(c.nb_)), jitter=_h(np.float32(c.jitter_)), solve=_h(c.solve(B))),
                c, lambda e: (e.solve(np.ascontiguousarray(Xh[:256, :2])),))


@lane("kernel-ridge")
def _(ml, X, yc, yr, Xh=None):
    """KernelRidge (python/mojolearn/kernel_methods.py) at the rbf kernel:
    the kernel matrix, the ridge, potrf and potrs, and the identical GEMM
    at OP_NN in predict. Train hashes dual_coef_ and the predictions on
    the training rows; infer predicts 64 held-out rows; the model column
    is n/a:no-save."""
    m = ml.KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5).fit(X[:256, :4], yr[:256])
    return _fit(dict(dual=_h(m.dual_coef_), info=_h(np.int64(m.info_)), predict=_h(m.predict(X[256:320, :4]))),
                m, lambda e: (e.predict(Xh[:64, :4]),))


@lane("nystroem")
def _(ml, X, yc, yr, Xh=None):
    """Nystroem at the rbf kernel with 32 components of 256 rows: the
    device permutation (km_basis_indices), the basis kernel, the Jacobi
    eigensolver with its sweep count, the sign flip, the clip and the
    transposed normalization (DEVIATION 1674). Train hashes every model
    array the estimator carries; infer transforms 64 held-out rows."""
    m = ml.Nystroem(kernel="rbf", gamma=0.5, n_components=32, random_state=7).fit(X[:256, :4])
    return _fit(dict(components=_h(m.components_), indices=_h(m.component_indices_), normalization=_h(m.normalization_),
                     eigenvalues=_h(m.eigenvalues_), eigenvectors=_h(m.eigenvectors_), sweeps=_h(np.int64(m.sweeps_)),
                     transform=_h(m.transform(X[256:320, :4]))),
                m, lambda e: (e.transform(Xh[:64, :4]),))


def _kernel_variant(ml, X, yr, Xh, family, kernel):
    # Fixed power-of-two input scaling keeps the polynomial matrix bounded
    # while preserving fixture ties and denormal/FTZ distinctions.
    x = np.ascontiguousarray(X[:48, :4] * np.float32(0.125))
    test = np.ascontiguousarray(X[48:64, :4] * np.float32(0.125))
    held = np.ascontiguousarray(Xh[:64, :4] * np.float32(0.125))
    kw = dict(kernel=kernel, gamma=0.5, coef0=0.25, degree=3)
    if family == "kernel-ridge":
        m = ml.KernelRidge(alpha=64.0, **kw).fit(x, yr[:48])
        return _fit(dict(dual=_h(m.dual_coef_), info=_h(np.int64(m.info_)),
                         predict=_h(m.predict(test))), m, lambda e: (e.predict(held),))
    m = ml.Nystroem(n_components=8, random_state=7, **kw).fit(x)
    return _fit(dict(components=_h(m.components_), indices=_h(m.component_indices_),
                     normalization=_h(m.normalization_), eigenvalues=_h(m.eigenvalues_),
                     eigenvectors=_h(m.eigenvectors_), sweeps=_h(np.int64(m.sweeps_)),
                     transform=_h(m.transform(test))), m, lambda e: (e.transform(held),))


@lane("kernel-ridge-poly")
def _(ml, X, yc, yr, Xh=None):
    return _kernel_variant(ml, X, yr, Xh, "kernel-ridge", "poly")


@lane("kernel-ridge-sigmoid")
def _(ml, X, yc, yr, Xh=None):
    return _kernel_variant(ml, X, yr, Xh, "kernel-ridge", "sigmoid")


@lane("kernel-ridge-laplacian")
def _(ml, X, yc, yr, Xh=None):
    return _kernel_variant(ml, X, yr, Xh, "kernel-ridge", "laplacian")


@lane("nystroem-poly")
def _(ml, X, yc, yr, Xh=None):
    return _kernel_variant(ml, X, yr, Xh, "nystroem", "poly")


@lane("nystroem-sigmoid")
def _(ml, X, yc, yr, Xh=None):
    return _kernel_variant(ml, X, yr, Xh, "nystroem", "sigmoid")


@lane("nystroem-laplacian")
def _(ml, X, yc, yr, Xh=None):
    return _kernel_variant(ml, X, yr, Xh, "nystroem", "laplacian")


@lane("rbf-sampler")
def _(ml, X, yc, yr, Xh=None):
    """RBFSampler with 64 random Fourier features over the 16 fixture
    columns: the Philox draws (weights and offsets), sigma and scale, and
    the identical GEMM at OP_NN plus cos in transform. The fit reads only
    n_features, so the train column's draws are the same on every fixture
    and only the transform moves with the data."""
    m = ml.RBFSampler(gamma=0.5, n_components=64, random_state=1).fit(X)
    return _fit(dict(weights=_h(m.random_weights_), offset=_h(m.random_offset_),
                     sigma=_h(np.float32(m.sigma_)), scale=_h(np.float32(m.scale_)), transform=_h(m.transform(X[:256]))),
                m, lambda e: (e.transform(Xh[:256]),))


@lane("gmm")
def _(ml, X, yc, yr, Xh=None):
    """GaussianMixture (python/mojolearn/mixture.py), full covariance,
    four components, init through the identity-certified k-means. Train
    hashes every model array plus n_iter_, converged_ and lower_bound_
    (the estimator's header: part of the card); infer scores and labels
    64 held-out rows; the model column is n/a:no-save."""
    m = ml.GaussianMixture(n_components=4, max_iter=30, random_state=3).fit(X[:6000, :4])
    return _fit(dict(weights=_h(m.weights_), means=_h(m.means_), covariances=_h(m.covariances_),
                     precisions=_h(m.precisions_cholesky_), logdet=_h(m.log_det_chol_),
                     n_iter=_h(np.int64(m.n_iter_)), converged=_h(np.int64(int(m.converged_))),
                     lower_bound=_h(np.float32(m.lower_bound_)),
                     labels=_h(m.predict(X[:6000, :4])), proba=_h(m.predict_proba(X[:256, :4]))),
                m, lambda e: (e.score_samples(Xh[:64, :4]), e.predict(Xh[:64, :4]), e.predict_proba(Xh[:64, :4])))


@lane("gmm-random-init")
def _(ml, X, yc, yr, Xh=None):
    """The same fit under init_params='random' (position-mapped Philox
    responsibilities, DEVIATION 1733), the other implemented init."""
    m = ml.GaussianMixture(n_components=4, max_iter=30, random_state=3, init_params="random").fit(X[:6000, :4])
    return _fit(dict(weights=_h(m.weights_), means=_h(m.means_), covariances=_h(m.covariances_),
                     n_iter=_h(np.int64(m.n_iter_)), lower_bound=_h(np.float32(m.lower_bound_)),
                     labels=_h(m.predict(X[:6000, :4]))),
                m, lambda e: (e.score_samples(Xh[:64, :4]),))


def _gmm_sample_lane(init_params):
    def body(ml, X, yc, yr, Xh=None):
        m = ml.GaussianMixture(n_components=4, max_iter=30, random_state=3, init_params=init_params).fit(X[:6000, :4])
        xs, ys = m.sample(256)
        _same_bytes("sample(256) X", xs, "sample(256) X again", m.sample(256)[0])
        return _fit(dict(sample_X=_h(xs), sample_y=_h(ys)), m, lambda e: tuple(e.sample(1024)))
    body.__doc__ = (f"GaussianMixture.sample (2026-09-15) on the gmm{'' if init_params == 'kmeans' else '-random-init'} "
                    "lane's fit: train hashes sample(256), held to a second call bit for bit; infer hashes "
                    "sample(1024). Position-mapped Philox draws (DEVIATION 2791) through precisions_cholesky_ "
                    "(DEVIATION 2792). A separate lane so the gmm lanes' recorded cells do not move.")
    return body


lane("gmm-sample")(_gmm_sample_lane("kmeans"))
lane("gmm-random-init-sample")(_gmm_sample_lane("random"))


@lane("hdbscan")
@floor(rows=(4000, "the Boruvka round count this lane HASHES holds at 5 down to 4000 rows and falls to 4 "
                   "below it, measured across 6000/4000/3000/2000/1500/1000/750 (2026-09-16, "
                   "lane/identity-fixtures-light), and the fifth merge round is where a divergence first "
                   "shows. Detection is unchanged at 4000: the resolution ladder reads 1e-7 at 4000 and "
                   "at 6000 (docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 4)."))
def _(ml, X, yc, yr, Xh=None):
    """HDBSCAN (python/mojolearn/hdbscan.py), cuML's runner path at its
    defaults with excess-of-mass selection: the core distances, the
    Boruvka MST, single linkage, the condensed tree and the labels.
    Train hashes the labels, the core distances and the four integers the
    fit reports (cluster, outlier, Boruvka round and condensed cluster
    counts); the integer stages are where a divergence first shows.
    The fit keeps prediction_data (2026-09-15), which moves no train byte;
    infer is approximate_predict's labels and probabilities on 256
    held-out rows (hdbscan.pyx:1264, predict.cuh:220-262), then
    membership_vector on the same rows and all_points_membership_vectors
    (hdbscan.pyx:1180, :1114, soft_clustering.cuh:385-627, DEVIATION 1616)."""
    # 4000 rows, not 6000 (2026-09-16, lane/identity-fixtures-light). The row
    # count is a fixture size, not a claim, and 6000 was inherited from the
    # dbscan lane rather than chosen. MEASURED across 6000/4000/3000/2000/
    # 1500/1000/750: the Boruvka round count, which this lane HASHES, holds at
    # 5 down to 4000 and falls to 4 at 3000, so 4000 is the smallest size that
    # still reaches the fifth merge round. Below it the lane would hash a
    # structurally different fit. Costs 2.9 s instead of 6.4 s to build.
    rows = 4000                                  # FLOORED, see @floor above
    m = ml.HDBSCAN(min_cluster_size=5, prediction_data=True).fit(X[:rows, :4])
    return _fit(dict(labels=_h(m.labels_), core=_h(m.core_distances_),
                     counts=_h(np.asarray([m.n_clusters_, m.n_outliers_, m.n_boruvka_rounds_, m.n_condensed_clusters_], dtype=np.int64))),
                m, lambda e: tuple(ml.hdbscan.approximate_predict(e, Xh[:256, :4]))
                + (ml.hdbscan.membership_vector(e, Xh[:256, :4]), ml.hdbscan.all_points_membership_vectors(e)))


@lane("hdbscan-leaf")
@floor(rows=(2000, "leaf selection holds its fifth Boruvka round down to 2000 rows and loses it below, "
                   "measured across 6000/4000/3000/2000/1500/1000/750 (2026-09-16, "
                   "lane/identity-fixtures-light); the resolution ladder reads 1e-7 at 2000 and at 6000 "
                   "(docs/lanes/LANE_STATUS_shrink-blindness-audit.md section 4), so the cut cost no "
                   "detection and the round count is what sets the floor."))
def _(ml, X, yc, yr, Xh=None):
    """The same fit under cluster_selection_method='leaf' with
    min_samples below min_cluster_size and allow_single_cluster, the
    other selection arm and the two knobs that change which condensed
    clusters become labels."""
    # 2000 rows, not 6000 (2026-09-16, lane/identity-fixtures-light), by the
    # same measurement as the hdbscan lane above. The leaf arm holds its
    # Boruvka round count at 5 all the way down to 2000 and falls to 4 at
    # 1500, so 2000 is its floor; it keeps 19 clusters and 37 condensed
    # clusters there, a structure with plenty left to disagree about. Costs
    # 0.7 s instead of 6.2 s to build.
    rows = 2000                                  # FLOORED, see @floor above
    m = ml.HDBSCAN(min_cluster_size=8, min_samples=3, cluster_selection_method="leaf", allow_single_cluster=True,
                   prediction_data=True).fit(X[:rows, :4])
    return _fit(dict(labels=_h(m.labels_), core=_h(m.core_distances_),
                     counts=_h(np.asarray([m.n_clusters_, m.n_outliers_, m.n_boruvka_rounds_, m.n_condensed_clusters_], dtype=np.int64))),
                m, lambda e: tuple(ml.hdbscan.approximate_predict(e, Xh[:256, :4]))
                + (ml.hdbscan.membership_vector(e, Xh[:256, :4]), ml.hdbscan.all_points_membership_vectors(e)))


@lane("bootstrap")
def _(ml, X, yc, yr, Xh=None):
    """resample.bootstrap over the first 4096 values of yr, 2048
    replicates: the Philox index map, the per-replicate folds (mean,
    std), the segmented sort (quantile), the paired two-column statistics
    (pearson, diff_means), the percentile and basic intervals and the
    standard error. Every distribution and every scalar is hashed; the
    r_first slice equality is asserted, as the surface promises it.

    The two SORTED statistics (quantile, trimmed_mean) take 1024
    replicates, 1024 * 4096 = RESAMPLE_MAX_SORT_CELLS (1 << 22,
    resample/checks/statistics.mojo), the largest run the sort path admits;
    at 2048 the lane read REFUSED by name on the M4 on every fixture
    (2026-09-14 night), and the limit is a comptime constant, so it
    refused on every vendor and the lane had no cell anywhere."""
    rs = ml.resample
    x = np.ascontiguousarray(yr[:4096])
    two = np.ascontiguousarray(np.stack([yr[:4096], X[:4096, 3]], 1).astype(np.float32))
    parts = {}
    for name, kw in (("mean", dict(data=x, statistic="mean")),
                     ("std", dict(data=x, statistic="std", method="basic")),
                     ("quantile", dict(data=x, statistic="quantile", q_or_prop=0.25, alternative="less")),
                     ("trimmed", dict(data=x, statistic="trimmed_mean", q_or_prop=0.1, alternative="greater")),
                     ("pearson", dict(data=two, statistic="pearson")),
                     ("diff", dict(data=two, statistic="diff_means", method="basic"))):
        n_resamples = 1024 if kw["statistic"] in ("quantile", "trimmed_mean") else 2048
        b = rs.bootstrap(n_resamples=n_resamples, random_state=3, **kw)
        parts[name] = _h(b.distribution, b.sorted_distribution,
                         np.asarray([b.point_estimate, b.standard_error, b.confidence_interval[0], b.confidence_interval[1]], dtype=np.float64),
                         np.asarray([b.order_low, b.order_high], dtype=np.int64))
    whole = np.asarray(rs.bootstrap(x, n_resamples=1024, random_state=3).distribution)
    part = np.asarray(rs.bootstrap(x, n_resamples=512, random_state=3, r_first=256).distribution)
    assert np.array_equal(whole[256:768].view(np.uint32), part.view(np.uint32)), "bootstrap lane: r_first slice is not bit-identical"
    parts["r_first"] = _h(part)
    return _fit(parts)


@lane("permutation-test")
def _(ml, X, yc, yr, Xh=None):
    """resample.permutation_test between the fixture's two label groups
    of yr (the first 512 rows of each), 2048 permutations, three
    alternatives: the pooled Philox permutation map, the between-group
    fold and the conservative p-value (DEVIATION 1702). The pooled sample
    is at most 1024 = PERM_MAX_POOLED (resample/checks/index_map.mojo, a
    comptime limit, refused by name above it); with 2048 per group the
    lane read REFUSED on the M4 on every fixture (2026-09-14 night) and so
    on every vendor."""
    rs = ml.resample
    a = np.ascontiguousarray(yr[:4096][yc[:4096] == 0][:512])
    b = np.ascontiguousarray(yr[:4096][yc[:4096] == 1][:512])
    if a.size < 8 or b.size < 8:
        a, b = np.ascontiguousarray(yr[:512]), np.ascontiguousarray(yr[512:1024])
    parts = {}
    for alt in ("two-sided", "less", "greater"):
        p = rs.permutation_test(a, b, statistic="diff_means", n_resamples=2048, random_state=3, alternative=alt)
        parts[alt] = _h(p.null_distribution, np.asarray([p.statistic, p.pvalue], dtype=np.float64),
                        np.asarray([p.count_less, p.count_greater], dtype=np.int64))
    return _fit(parts)


@lane("monte-carlo")
def _(ml, X, yc, yr, Xh=None):
    """resample.monte_carlo_integrate, the three compiled integrands over
    a box whose corners come from the fixture's first two column minima
    and maxima (host min and max, exact), 65536 draws: the position-mapped
    Philox points and the chunked pinned fold. The closed forms are
    hashed beside the estimates."""
    rs = ml.resample
    lo = [float(np.min(X[:, 0])), float(np.min(X[:, 1]))]
    hi = [float(np.max(X[:, 0])), float(np.max(X[:, 1]))]
    if not (hi[0] > lo[0] and hi[1] > lo[1]):
        lo, hi = [0.0, 0.0], [1.0, 2.0]
    parts = {}
    for f in ("const", "sum", "product"):
        r = rs.monte_carlo_integrate(f, lo, hi, 65536, random_state=1)
        parts[f] = _h(np.asarray([r.integral, r.mean, r.volume, r.closed_form], dtype=np.float64))
    return _fit(parts)


@lane("ordered-gradient-sum")
def _(ml, X, yc, yr, Xh=None):
    """Public ordered shard reduction: preserve shard order and input bytes.

    Normal finite Float32 values keep the independent NumPy left-fold oracle
    outside FTZ subtleties. The first element cancels 2**100, so a balanced
    reduction returns zero where the required left fold returns three.
    """
    from mojolearn.parallel_training import ordered_sum_gradients
    seed = np.where(X[:4, :8] >= 0, np.float32(1), np.float32(-1))
    shards = [[np.ascontiguousarray(seed * np.float32(i + 1)),
               np.array([i, -i, i + 1], dtype=np.float32)] for i in range(4)]
    for shard, value in zip(shards, (2.0**100, 1.0, -(2.0**100), 3.0)):
        shard[0][0, 0] = np.float32(value)
    before = [[a.tobytes() for a in shard] for shard in shards]
    expected = [a.copy() for a in shards[0]]
    for shard in shards[1:]:
        expected = [np.add(a, b, dtype=np.float32) for a, b in zip(expected, shard)]
    got = ordered_sum_gradients(iter(shards))
    if len(got) != len(expected):
        raise AssertionError("ordered gradient sum lost a tensor")
    mismatch = False
    for actual, wanted in zip(got, expected):
        actual = np.asarray(actual)
        if actual.shape != wanted.shape or actual.dtype != wanted.dtype:
            raise AssertionError("ordered gradient sum changed tensor shape or dtype")
        mismatch |= actual.tobytes() != wanted.tobytes()
    if before != [[a.tobytes() for a in shard] for shard in shards]:
        raise AssertionError("ordered gradient sum mutated its inputs")
    parts = dict(gradients=_h(*(np.asarray(a) for a in got)))
    if mismatch:
        raise NumericalMismatch("ordered gradient sum differs from Float32 left fold", parts)
    return _fit(parts)


#: The (T, A) pairs `grad-accumulation` asks `accumulation_is_aligned`, one
#: per side of clause 9.2: a divisible split, an indivisible one, A = 1 (no
#: split at all), A = T (one token per microbatch) and a split with more
#: microbatches than tokens. The answers are the binding's, not Python's.
ACCUM_SPLITS = ((1024, 4), (1024, 3), (1024, 1), (1024, 1024), (12, 5), (7, 7), (100, 8))


@lane("grad-accumulation")
def _(ml, X, yc, yr, Xh=None):
    """GRADIENT ACCUMULATION by its public names: `training.accumulate_grads`
    and `training.accumulation_is_aligned` (optimizer contract clause 9.2).
    Lane lane/laneless-public-classes, 2026-09-19.

    WHAT IT IS. A step too large for one device is run as A microbatches and
    their weight gradients combined. Clause 9.2 says the combination is a
    BALANCED TREE in ascending microbatch index -- `ftz(ftz(x) + ftz(y))` at
    every node, never a running sum -- so that the combined bits are the
    unsplit step's bits, and `accumulation_is_aligned(T, A)` is the
    predicate that says for which (T, A) that holds. Both are native:
    `accumulate` and `accumulation_is_aligned` in the training binding,
    `training/host/samba_ops_oracle.mojo::host_samba_accumulate` on the CPU
    column. Until this lane the pair was reached only by the opt-in
    `--batch-grad` part, which no default record carries, so no column in
    any record held either of them.

    THE PARTS. `combined` is `accumulate_grads` over A = 4 microbatches of
    three tensors of different shapes, values cut from the fixture; the
    first element of the first tensor is DELIBERATELY NONZERO (see the
    sabotage note). `single` is the same call at A = 1, the no-split arm,
    which the contract says is `ftz` of the input. `unaligned` is the same
    microbatches with `tokens=None`, the explicit no-claim form, which must
    still be the same tree. `pairs` is a second shape set passed as bare
    arrays rather than lists, because `accumulate_grads` returns an array
    for an array input and a list for a list input and a caller can tell
    them apart. `aligned` is `accumulation_is_aligned` over ACCUM_SPLITS,
    the BINDING's answers as integers. `flags` holds the refusals the
    function owes by name: an empty microbatch list, a ragged tensor count,
    a shape that differs between microbatches and tokens < 1.

    SABOTAGE. `-D MOJOLEARN_HOST_SABOTAGE=1` on the training family reaches
    `host_samba_accumulate`, whose arm sets the FIRST combined element to
    0.0 (`training/host/samba_ops_oracle.mojo`: pairwise addition has no
    fold-order fault, so the arm corrupts a value). That arm is INERT on any
    fixture whose first combined element is already zero, which is why this
    lane builds its microbatches so that element is not, and the `+ 1.0`
    below is the whole of that defence. MEASURED, which is why it is there:
    `X[0, 0]` is exactly 0.0 on `denormal_ftz` and a subnormal that flushes
    to 0.0 on `denormal`, so microbatches cut straight from the fixture
    would have combined to 0.0 in that element and the arm -- which WRITES
    0.0 there -- would have left the cell exactly where it found it on two
    of the nine fixtures. A negative control that cannot fail on a fixture
    is not a negative control on that fixture.

    MEASURED on the M4, one core, `--repeats 2`, ALL NINE FIXTURES
    (`bench/results/identity_break/2026-09-19_laneless-public-classes/`):
    9 of 9 cells move (`base` 7914bf0b244d9142 -> c18d7abf3e2a978e), through
    `combined`, `unaligned` and `pairs`.

    THREE PARTS DO NOT MOVE UNDER IT, each for its own reason and none of
    them a defect. `single` is the A = 1 call, which `host_samba_accumulate`
    answers from an EARLY RETURN above the arm (`if a == 1`, the flushed
    input), so no define placed at the end of the fold can reach it; it is
    kept because A = 1 is the no-split boundary a caller passes when
    accumulation is configured off, and a defect in that branch would move
    it. `aligned` and `flags` are integers out of a predicate and a set of
    refusals with no float fold near them, so no arithmetic arm can reach
    them either; what guards those is the recorded hash itself, which reads
    DIVERGENT on every column at once if the predicate's answer changes."""
    T = ml.training
    base = np.ascontiguousarray(X[:16, :8]).astype(np.float32)
    # Three tensors per microbatch, four microbatches. `+ 1.0` keeps the
    # FIRST element of the first tensor away from zero on every fixture,
    # which is the only thing that makes the family's arm reach this cell.
    micro = [[np.ascontiguousarray(base * np.float32(k + 1) + np.float32(1.0)),
              np.ascontiguousarray(base[:4, :3] - np.float32(k)),
              np.array([k + 1, -(k + 1), 2 * k + 1], dtype=np.float32)]
             for k in range(4)]
    combined = T.accumulate_grads(micro, 1024)
    single = T.accumulate_grads([micro[0]], 1024)
    unaligned = T.accumulate_grads(micro, None)
    # arrays in, one array out (the shape a caller can tell from a list)
    flat = [np.ascontiguousarray(base * np.float32(k + 1) + np.float32(1.0)) for k in range(4)]
    pairs = T.accumulate_grads(flat, 1024)
    aligned = np.array([T.accumulation_is_aligned(t, a) for t, a in ACCUM_SPLITS], dtype=np.int64)

    def _refuses(fn, text):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            return text in str(exc)
        return False

    flags = np.array([
        _refuses(lambda: T.accumulate_grads([], 1024), "no microbatches"),
        _refuses(lambda: T.accumulate_grads([micro[0], micro[1][:2]], 1024), "tensors"),
        _refuses(lambda: T.accumulate_grads([micro[0], [micro[1][0][:4]] + micro[1][1:]], 1024), "shape"),
        _refuses(lambda: T.accumulate_grads(micro, 0), "tokens must be >= 1"),
        len(combined) == 3, len(single) == 3,
        # an array input returns one array, a list input a list of arrays
        not isinstance(pairs, list),
        isinstance(combined, list),
    ], dtype=np.int64)
    return _fit(dict(combined=_h(*(np.asarray(a) for a in combined)),
                     single=_h(*(np.asarray(a) for a in single)),
                     unaligned=_h(*(np.asarray(a) for a in unaligned)),
                     pairs=_h(np.asarray(pairs)),
                     aligned=_h(aligned), flags=_h(flags)))


@lane("training-primitives")
def _(ml, X, yc, yr, Xh=None):
    """embedding_forward/backward (the gather and the run-sorted fold),
    rms_norm_forward/backward, linear_forward/backward (the identical
    GEMM at OP_NT and the two backward products). Every output hashed;
    the probe runs the three forwards on a held-out slab."""
    T = ml.training
    V, D, N, K = 64, 32, 64, 16
    w_emb = _hw((V, D), "prim:emb", -0.25, 0.25)
    ids = (_ids(X, 1, N).reshape(N) % V).astype(np.int32)
    x = np.ascontiguousarray(_seq(X, 1, N, D).reshape(N, D))
    g = np.ascontiguousarray(np.abs(_hw((D,), "prim:rms", 0.5, 1.5)))
    w_lin = _hw((K, D), "prim:lin", -0.25, 0.25)
    dy = _hw((N, D), "prim:dy", -1.0, 1.0)
    dc = _hw((N, K), "prim:dc", -1.0, 1.0)
    emb = T.embedding_forward(w_emb, ids)
    demb = T.embedding_backward(dy, ids, V)
    rms = T.rms_norm_forward(x, g, 1e-5)
    dx, dg = T.rms_norm_backward(dy, x, g, 1e-5)
    lin = T.linear_forward(x, w_lin)
    da, dw = T.linear_backward(dc, x, w_lin)
    parts = dict(emb=_h(np.asarray(emb)), demb=_h(np.asarray(demb)), rms=_h(np.asarray(rms)),
                 drms=_h(np.asarray(dx), np.asarray(dg)), lin=_h(np.asarray(lin)), dlin=_h(np.asarray(da), np.asarray(dw)))
    xh = np.ascontiguousarray(_seq(Xh, 1, N, D).reshape(N, D))
    idh = (_ids(Xh, 1, N).reshape(N) % V).astype(np.int32)
    return _fit(parts, T, lambda e: (np.asarray(e.embedding_forward(w_emb, idh)), np.asarray(e.rms_norm_forward(xh, g, 1e-5)),
                                     np.asarray(e.linear_forward(xh, w_lin))))


@lane("ivf")
def _(ml, X, yc, yr, Xh=None):
    """IVFIndex (python/mojolearn/_ivf_impl.py), cuVS ivf_flat build plus
    search under one card: the coarse k-means quantizer, the CSR list
    layout, the probe selection and the per-list distance tiles. 16 lists
    and 4 probes over an index of 4096 rows, so the probe set is a real
    subset; 64 queries. Train hashes the distances, the original row ids
    (the tie class: ids diverging while distances agree) and the
    per-query candidate counts. The index does not cross, so the model
    column is n/a:no-save; the probe searches 64 held-out rows."""
    m = ml.IVFIndex(n_lists=16, n_probes=4, n_neighbors=8, random_state=3).fit(X[:4096])
    d, i = m.search(X[4096:4160])
    return _fit(dict(dist=_h(d), idx=_h(i), cand=_h(m.n_candidates_)),
                m, lambda e: e.search(Xh[:64]) + (e.n_candidates_,))


@lane("ivf-euclidean")
def _(ml, X, yc, yr, Xh=None):
    """IVFIndex with metric='euclidean' (cuVS L2SqrtExpanded), the `ivf`
    lane's shape otherwise. A different quantizer (k-means under the rooted
    reduction, so a different inertia and possibly a different iteration
    count) and the root of the k selected squared distances
    (ivf_common.mojo::postprocess_distances). Refused at the door until
    2026-09-14: every norm was rooted and the search returned zeros
    (fix/ivf-l2sqrt, ivf_check's check_l2_sqrt_is_the_root_of_l2)."""
    m = ml.IVFIndex(n_lists=16, n_probes=4, n_neighbors=8, metric="euclidean", random_state=3).fit(X[:4096])
    d, i = m.search(X[4096:4160])
    return _fit(dict(dist=_h(d), idx=_h(i), cand=_h(m.n_candidates_)),
                m, lambda e: e.search(Xh[:64]) + (e.n_candidates_,))


@lane("ivf-extend")
def _(ml, X, yc, yr, Xh=None):
    """IVFIndex.extend (python/mojolearn/_ivf_impl.py; lane/inference-embedding-
    ivf-cholesky stage 2, 2026-09-15), cuVS ivf_flat::extend with fixed
    centres. The `ivf` lane's index (16 lists, random_state 3) built on rows
    0..3072, extended by rows 3072..4096 in ONE call and, from a clone of the
    same built index, in TWO calls split at 3584: the two extended indexes
    must be the same bytes, and the new rows' lists the same, or the train
    cell reads REFUSED naming the array. Then 64 queries with 4 probes. Train
    hashes the extended CSR arrays, the new rows' lists and the search; the
    model column saves the extended index; the probe searches 64 held-out
    rows."""
    base = ml.IVFIndex(n_lists=16, n_probes=4, n_neighbors=8, random_state=3).fit(X[:3072])
    one = base._clone().extend(X[3072:4096])
    two = base._clone().extend(X[3072:3584])
    first = np.asarray(two.extend_labels_).copy()
    two.extend(X[3584:4096])
    for name in ("centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_"):
        _same_bytes(f"one-call extend {name}", getattr(one, name), f"two-call extend {name}", getattr(two, name))
    _same_bytes("one-call new-row lists", one.extend_labels_, "two-call new-row lists",
                np.concatenate([first, np.asarray(two.extend_labels_)]))
    d, i = one.search(X[4096:4160])
    return _fit(dict(offsets=_h(one.list_offsets_), carry=_h(one.list_indices_), data=_h(one.list_data_),
                     lists=_h(one.extend_labels_), dist=_h(d), idx=_h(i), cand=_h(one.n_candidates_)),
                one, lambda e: e.search(Xh[:64]) + (e.n_candidates_,))


@lane("embedding")
def _(ml, X, yc, yr, Xh=None):
    """Embedding (python/mojolearn/embedding.py), profile
    mojolearn.identical.embedding.fp32.v1: the gather, the ascending fold
    with padding_idx=3, the same fold without a padding row, and the
    microbatch carry. V=128, d=16, T=512 ids from the fixture's bytes (runs
    of several positions per row, so the fold order is exercised) and dY
    from the fixture's values (the denormal fixture puts subnormals in the
    fold, where the E-seam flush pins are the answer). The carry, split at
    t0=171, must equal the unsplit gradient byte for byte (contract 7.4) or
    the train cell reads REFUSED naming the pair. No model crosses; the
    probe gathers 512 held-out ids."""
    V, Dm, T = 128, 16, 512
    w = _hw((V, Dm), "emb:w", -0.25, 0.25)
    ids = (_ids(X, 1, T).reshape(T) % V).astype(np.int32)
    dy = np.ascontiguousarray(_seq(X, 1, T, Dm).reshape(T, Dm))
    e = ml.Embedding(V, Dm, padding_idx=3, weight=w)
    y = np.asarray(e.forward(ids))
    dw = np.asarray(e.backward(ids, dy))
    t0 = 171
    g1 = e.backward(ids[:t0], dy[:t0])
    g2 = np.asarray(e.backward(ids[t0:], dy[t0:], grad=g1))
    _same_bytes("carried dW (t0=171)", g2, "unsplit dW", dw)
    dw_nopad = np.asarray(ml.Embedding(V, Dm, weight=w).backward(ids, dy))
    idh = (_ids(Xh, 1, T).reshape(T) % V).astype(np.int32)
    return _fit(dict(fwd=_h(y), dw=_h(dw), dw_nopad=_h(dw_nopad)),
                e, lambda m: (np.asarray(m.forward(idh)),), model_na="n/a:input-table")


@lane("embedding-sort")
def _(ml, X, yc, yr, Xh=None):
    """Embedding(plan="sort"), contract section 6.2's PLAN_SORT (the device
    total-key bitonic sort) instead of PLAN_SCAN, on the `embedding` lane's
    exact inputs: V=128, d=16, T=512, padding_idx=3, the same fold without a
    padding row, and the carry split at t0=171. The plan is an execution plan
    and not the specification, so the lane also runs plan="scan" in the same
    process and the train cell reads REFUSED naming the pair unless every
    sort gradient equals its scan twin byte for byte (contract 11(d)); the
    recorded hashes are therefore the `embedding` lane's hashes too. The
    probe gathers 512 held-out ids."""
    V, Dm, T = 128, 16, 512
    w = _hw((V, Dm), "emb:w", -0.25, 0.25)
    ids = (_ids(X, 1, T).reshape(T) % V).astype(np.int32)
    dy = np.ascontiguousarray(_seq(X, 1, T, Dm).reshape(T, Dm))
    e = ml.Embedding(V, Dm, padding_idx=3, weight=w, plan="sort")
    y = np.asarray(e.forward(ids))
    dw = np.asarray(e.backward(ids, dy))
    t0 = 171
    g1 = e.backward(ids[:t0], dy[:t0])
    g2 = np.asarray(e.backward(ids[t0:], dy[t0:], grad=g1))
    _same_bytes("sort carried dW (t0=171)", g2, "sort unsplit dW", dw)
    dw_nopad = np.asarray(ml.Embedding(V, Dm, weight=w, plan="sort").backward(ids, dy))
    _same_bytes("sort dW", dw, "scan dW", np.asarray(ml.Embedding(V, Dm, padding_idx=3, weight=w).backward(ids, dy)))
    _same_bytes("sort dW without padding", dw_nopad, "scan dW without padding",
                np.asarray(ml.Embedding(V, Dm, weight=w).backward(ids, dy)))
    idh = (_ids(Xh, 1, T).reshape(T) % V).astype(np.int32)
    return _fit(dict(fwd=_h(y), dw=_h(dw), dw_nopad=_h(dw_nopad)),
                e, lambda m: (np.asarray(m.forward(idh)),), model_na="n/a:input-table")


@lane("kmeans-sqrt")
def _(ml, X, yc, yr, Xh=None):
    """metric='l2_sqrt_expanded': cuVS's L2SqrtExpanded, the root taken on
    the assignment's reduced distance (`metric_is_sqrt`, `identical_sqrt`,
    DEVIATION 2715) and so in the inertia, the row norms squared as under
    L2Expanded (DEVIATION 2716); a different inertia_ from the squared arm."""
    m = ml.KMeans(n_clusters=8, random_state=3, metric="l2_sqrt_expanded").fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_),
                     inertia=_h(np.float64(m.inertia_)), scales=_h(np.asarray([m.sum_scale_, m.weight_scale_], dtype=np.float64))),
                m, _km_probe(X, Xh))


@lane("kmeans-classic-pp")
def _(ml, X, yc, yr, Xh=None):
    """oversampling_factor=0.0: the classic sequential k-means++ seeding
    (detail/kmeans.cuh:910-915's `== 0` arm), a different algorithm from
    the scalable k-means|| the default selects."""
    m = ml.KMeans(n_clusters=8, random_state=3, oversampling_factor=0.0).fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_), inertia=_h(np.float64(m.inertia_))),
                m, _km_probe(X, Xh))


# ---------------------------------------------------------------- lanes (2026-09-14 evening, the multi-GPU drivers on ONE device)
# The ordered multi-GPU drivers (docs/multi_gpu/README.md) promise that a fit
# split into K logical shards or tree ranges reduces, in a fixed order, to
# the bits of the same fit on one device, and that the physical GPU count
# never selects the reduction order. So every driver has a lane here that
# runs with devices=_par_devices() and the smallest sharding that exercises the split,
# on every vendor including this Mac; where the driver's contract is
# equality to the plain fit, the lane holds the two to the same bytes and
# reads REFUSED with the pair named if they differ. A two-device run on the
# same commit must then hash equal to these cells, which is the claim the
# drivers' own two-H100 gates make and no three-vendor record has held yet.

#: AUDIT ONLY. MOJOLEARN_IDENTITY_WIDE=1 widens the neural par-* lanes'
#: models (byte LM d_model 256, head_dim 64, intermediate 1024; Samba d_model
#: 256, intermediate 1280) so their pooled gradient, optimizer and clipping
#: buffers pass 1 MiB. Recorded as `package.wide`; `--diff` refuses a mix.
WIDE_ENV = "MOJOLEARN_IDENTITY_WIDE"
PAR_WIDE = os.environ.get(WIDE_ENV, "0").strip() == "1"


def _par_byte_lm_config(ml):
    if PAR_WIDE:
        return ml.ByteLanguageModelConfig(d_model=256, n_heads=4, n_kv=2, head_dim=64, intermediate=1024)
    return ml.ByteLanguageModelConfig()


def _par_samba_config(ml):
    if PAR_WIDE:
        return ml.SambaConfig(vocab=256, d_model=256, layers=("mamba3", "attention"), n_heads=2, intermediate=1280)
    return ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)


def _par_devices():
    """The device tuple the par-* lanes hand the drivers: MOJOLEARN_PAR_DEVICES,
    comma-separated device indices, default "0". A two-device column names
    itself through the `package.par_devices` field the run records, and must
    hash equal, cell for cell, to the one-device column of the same commit;
    that equality is the drivers' whole claim."""
    raw = os.environ.get("MOJOLEARN_PAR_DEVICES", "0").strip() or "0"
    try:
        devs = tuple(int(x) for x in raw.split(",") if x.strip() != "")
    except ValueError:
        devs = ()
    if not devs or len(set(devs)) != len(devs) or any(d < 0 for d in devs):
        raise SystemExit(f"REFUSING: MOJOLEARN_PAR_DEVICES={raw!r} is not a list of distinct nonnegative device indices")
    return devs


@lane("par-forest")
def _(ml, X, yc, yr, Xh=None):
    """fit_forest, 16 trees in four ranges of four, held to the plain
    RandomForestClassifier fit of the rf-clf lane."""
    from mojolearn.parallel_ensemble import fit_forest
    kw = dict(n_estimators=16, max_depth=8, random_state=7)
    par = fit_forest(ml.RandomForestClassifier(**kw), X, yc, devices=_par_devices(), trees_per_shard=4)
    plain = ml.RandomForestClassifier(**kw).fit(X, yc)
    _same_bytes("fit_forest predict_proba", par.predict_proba(X[:2048]), "plain predict_proba", plain.predict_proba(X[:2048]))
    return _fit(dict(predict=_h(par.predict(X)), proba=_h(par.predict_proba(X))),
                par, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("par-forest-et")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn.parallel_ensemble import fit_forest
    kw = dict(n_estimators=16, max_depth=8, random_state=7)
    par = fit_forest(ml.ExtraTreesRegressor(**kw), X, yr, devices=_par_devices(), trees_per_shard=4)
    plain = ml.ExtraTreesRegressor(**kw).fit(X, yr)
    _same_bytes("fit_forest predict", par.predict(X[:2048]), "plain predict", plain.predict(X[:2048]))
    return _fit(dict(predict=_h(par.predict(X))), par, lambda e: (e.predict(Xh),))


@lane("par-boosting")
def _(ml, X, yc, yr, Xh=None):
    """fit_boosting on the gbdt-symmetric configuration, feature histograms
    partitioned, held to the plain fit."""
    from mojolearn.parallel_ensemble import fit_boosting
    kw = dict(n_estimators=20, max_depth=6, loss="Logloss")
    par = fit_boosting(_gbdt(ml.GradientBoosting, **kw), X, yc, devices=_par_devices())
    plain = _gbdt(ml.GradientBoosting, **kw).fit(X, yc)
    _same_bytes("fit_boosting predict", par.predict(X[:2048]), "plain predict", plain.predict(X[:2048]))
    return _fit(dict(predict=_h(par.predict(X)), proba=_h(par.predict_proba(X))),
                par, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("par-kmeans")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn.parallel_classical import fit_kmeans
    par = fit_kmeans(ml.KMeans(n_clusters=8, random_state=3), X, devices=_par_devices())
    plain = ml.KMeans(n_clusters=8, random_state=3).fit(X)
    _same_bytes("fit_kmeans centers", par.cluster_centers_, "plain centers", plain.cluster_centers_)
    return _fit(dict(centers=_h(par.cluster_centers_), labels=_h(par.labels_)), par, _km_probe(X, Xh))


@lane("par-gram")
def _(ml, X, yc, yr, Xh=None):
    """fit_gram_estimator on Ridge: the 128 pinned Gram chunks distributed."""
    from mojolearn.parallel_classical import fit_gram_estimator
    par = fit_gram_estimator(ml.Ridge(alpha=1.0), X, yr, devices=_par_devices())
    plain = ml.Ridge(alpha=1.0).fit(X, yr)
    _same_bytes("fit_gram_estimator coef", par.coef_, "plain coef", plain.coef_)
    return _fit(dict(coef=_h(par.coef_), predict=_h(par.predict(X[:256]))), par, lambda e: (e.predict(Xh[:256]),))


@lane("par-logistic")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn.parallel_classical import fit_logistic
    par = fit_logistic(ml.LogisticRegression(max_iter=50), X, yc, devices=_par_devices())
    plain = ml.LogisticRegression(max_iter=50).fit(X, yc)
    _same_bytes("fit_logistic coef", par.coef_, "plain coef", plain.coef_)
    return _fit(dict(coef=_h(par.coef_), proba=_h(par.predict_proba(X[:256]))), par, lambda e: (e.predict_proba(Xh[:256]),))


@lane("par-cd")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn.parallel_classical import fit_coordinate_descent
    par = fit_coordinate_descent(ml.Lasso(alpha=0.01, max_iter=200), X, yr, devices=_par_devices())
    plain = ml.Lasso(alpha=0.01, max_iter=200).fit(X, yr)
    _same_bytes("fit_coordinate_descent coef", par.coef_, "plain coef", plain.coef_)
    return _fit(dict(coef=_h(par.coef_), predict=_h(par.predict(X[:256]))), par, lambda e: (e.predict(Xh[:256]),))


@lane("par-svm")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn.parallel_classical import fit_svm, predict_svm
    kw = dict(C=1.0, kernel="rbf", max_iter=200)
    par = fit_svm(ml.SVC(**kw), X[:2000], yc[:2000], devices=_par_devices())
    plain = ml.SVC(**kw).fit(X[:2000], yc[:2000])
    dec = predict_svm(par, X[2000:2256], devices=_par_devices(), method="decision_function")
    _same_bytes("predict_svm decision", dec, "plain decision", plain.decision_function(X[2000:2256]))
    return _fit(dict(decision=_h(dec), predict=_h(predict_svm(par, X[2000:2256], devices=_par_devices(), method="predict"))),
                par, lambda e: (e.decision_function(Xh[:256]), e.predict(Xh[:256])))


@lane("par-gp")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn.parallel_classical import fit_gaussian_process, predict_gaussian_process
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
    par = fit_gaussian_process(ml.GaussianProcessRegressor(kernel=k), X[:256, :4], yr[:256], devices=_par_devices())
    plain = ml.GaussianProcessRegressor(kernel=k).fit(X[:256, :4], yr[:256])
    mean, std = predict_gaussian_process(par, X[256:320, :4], devices=_par_devices(), return_std=True)
    pm, ps = plain.predict(X[256:320, :4], return_std=True)
    _same_bytes("predict_gaussian_process mean", mean, "plain mean", pm)
    return _fit(dict(alpha=_h(par.alpha_), L=_h(par.L_), mean=_h(mean), std=_h(std)),
                par, lambda e: e.predict(Xh[:64, :4], return_std=True))


@lane("par-dbscan")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn.parallel_classical import fit_dbscan
    par = fit_dbscan(ml.DBSCAN(eps=0.9, min_samples=5, prediction_data=True), X[:6000, :4], devices=_par_devices())
    plain = ml.DBSCAN(eps=0.9, min_samples=5).fit(X[:6000, :4])
    _same_bytes("fit_dbscan labels", par.labels_, "plain labels", plain.labels_)
    return _fit(dict(labels=_h(par.labels_)), par, lambda e: (e.predict(Xh[:256, :4]),))


@lane("par-scaler")
def _(ml, X, yc, yr, Xh=None):
    """fit_scaler and transform_scaler with four columns per shard, so the
    16-column fixture is four shards, held to the plain scaler.

    The disagreement is raised as `NumericalMismatch` CARRYING the parts, not
    as a bare ValueError. Measured 2026-09-17 (lane/sabotage-sweep,
    `bench/results/identity_break/2026-09-17_sabotage-sweep/a-linear-neighbors/`):
    under the preprocessing family's sabotage build this lane raised
    `transform_scaler and plain transform differ: 2847 bytes of 16384` and the
    cell read REFUSED, so a negative control that was WORKING was recorded as
    a refusal, and a refusal is not a catch. The clean cell is unchanged: the
    same parts in the same order whenever the two agree."""
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    par = fit_scaler(ml.StandardScaler(), X, devices=_par_devices(), columns_per_shard=4)
    plain = ml.StandardScaler().fit(X)
    t = transform_scaler(par, X[:256], devices=_par_devices(), columns_per_shard=4)
    mismatch = _mismatch_bytes("transform_scaler", t, "plain transform", plain.transform(X[:256]))
    parts = dict(mean=_h(par.mean_), var=_h(par.var_), transform=_h(t),
                 inverse=_h(transform_scaler(par, t, devices=_par_devices(), columns_per_shard=4, inverse=True)))
    if mismatch:
        raise NumericalMismatch(mismatch, parts)
    return _fit(parts, par, lambda e: (e.transform(Xh[:256]),))


@lane("par-arima")
def _(ml, X, yc, yr, Xh=None):
    """fit_arima with two series per shard over the arima lane's four series."""
    from mojolearn.parallel_classical import fit_arima
    series = np.ascontiguousarray(X[:512, :4].T)
    par = fit_arima(ml.ARIMA(order=(1, 0, 0)), series, devices=_par_devices(), series_per_shard=2)
    plain = ml.ARIMA(order=(1, 0, 0)).fit(series)
    _same_bytes("fit_arima ar", par.ar_, "plain ar", plain.ar_)
    return _fit(dict(ar=_h(par.ar_), mu=_h(par.mu_), sigma2=_h(par.sigma2_), forecast=_h(par.forecast(24))),
                par, lambda e: _same_bytes("forecast(h)", e.forecast(FORECAST_HORIZON),
                                           "predict(n_obs, n_obs + h)", e.predict(e.n_obs_, e.n_obs_ + FORECAST_HORIZON)))


@lane("par-forecast-arima")
def _(ml, X, yc, yr, Xh=None):
    """predict_arima and forecast_arima -- the PREDICTION drivers, which
    par-arima does not reach (it shards a fit) -- over the same four series,
    split THREE to a shard: the uneven 3 + 1 partition, so the last shard's
    width differs from the first's and a one-series shard is exercised.

    IN-CELL ORACLE (2026-09-19, lane/lm-attention-fallback). The drivers'
    entire claim is that a SERIES partition moves no bit, so the unsharded
    call is the oracle: the sharded prediction and forecast are held byte
    for byte, in this cell, to `ARIMA.predict` and `ARIMA.forecast` of the
    same fitted estimator. Unlike par-mlp's fold (which is not a plain
    step), nothing here needs a replay. The two sides are two call chains --
    a free driver function against the estimator's own door -- so a defect
    in `_ranges`, in the per-shard `__dict__` slice, in the SHARD ORDER or
    in the merge offsets separates them on this box with no record to lean
    on. The infer probe re-runs the driver on the reloaded model.

    Reachable on a CPU column since `forecast_predict` joined
    `_parallel_pool.CPU_OPERATIONS`; before that all four
    `parallel_forecasting` entries refused by name on every CPU install."""
    from mojolearn.parallel_forecasting import predict_arima, forecast_arima
    series = np.ascontiguousarray(X[:512, :4].T)
    d = _par_devices()
    m = ml.ARIMA(order=(1, 0, 0)).fit(series)
    sharded = predict_arima(m, 0, m.n_obs_, devices=d, series_per_shard=3)
    _same_bytes("predict_arima(0, n_obs)", sharded, "plain ARIMA.predict(0, n_obs)", m.predict(0, m.n_obs_))
    ahead = forecast_arima(m, 24, devices=d, series_per_shard=3)
    _same_bytes("forecast_arima(24)", ahead, "plain ARIMA.forecast(24)", m.forecast(24))
    return _fit(dict(predict=_h(sharded), forecast=_h(ahead)), m,
                lambda e: _same_bytes(
                    f"forecast_arima({FORECAST_HORIZON})",
                    forecast_arima(e, FORECAST_HORIZON, devices=d, series_per_shard=3),
                    f"plain ARIMA.forecast({FORECAST_HORIZON})", e.forecast(FORECAST_HORIZON)))


# ---------------------------------------------------------------- the neural drivers' in-cell oracle (2026-09-19)
# THE DEGENERATE CELL THESE TWO HELPERS CLOSE. par-mlp, par-samba and
# par-samba-clip hashed the state after a sharded run and held it to NOTHING:
# `tools/lane_applicability.py --oracle-counts` read them `recorded`, so the
# cell passed by comparing today's bytes to a previous record of the same
# code, and on a box with no record it could not fail at all. Their own
# docstrings said why nobody had written one: the driver's contract is NOT a
# plain step over the concatenated rows, because each shard's loss is its own
# mean and the shard gradients are SUMMED, not averaged
# (parallel_training.ordered_sum_gradients: "Neither this fold nor the
# byte-LM fold divides by shard count"). A plain fit is therefore the wrong
# oracle here, and the right one is the fold itself, rebuilt from the public
# single-device doors: it is what `_parallel_worker` does on the far side of
# the pool, minus the pool.
#
# WHAT THIS CAN AND CANNOT CATCH. The reference shares the gradient and
# optimizer KERNELS with the driver, exactly as `par-forest`'s plain
# RandomForestClassifier shares the tree kernels with `fit_forest`. What it
# does not share is everything the driver exists to do: the DevicePool, the
# frozen-snapshot serialization, the host transport, the per-shard worker's
# state round trip, the shard ORDER of the fold and the cooperative update.
# A defect in any of those moves the two apart inside one cell, on one box,
# with no record. A defect in a kernel both call is invisible to it, and the
# cross-vendor record remains the only thing that sees that.


def _ordered_replay_mlp(plain, steps):
    """ParallelNeuralTrainer's MLP contract rebuilt on one object through the
    public doors: every shard's gradient from the SAME weights (loss_and_grads
    advances no optimizer), `ordered_sum_gradients` over them in shard order,
    one `apply_gradients`. `_parallel_worker.mlp_gradient` / `mlp_update`."""
    from mojolearn.parallel_training import ordered_sum_gradients
    for shards in steps:
        parts = [plain.loss_and_grads(x, y)[1] for x, y in shards]
        plain.apply_gradients(ordered_sum_gradients(parts))
    return plain


def _ordered_replay_samba(plain, steps):
    """The same for a SambaStack, with the driver's token offsets (it advances
    `offset` by each shard's token count within a step) and its clip: the
    worker's `samba_gradient` / `samba_update`. Returns the LAST step's
    reduced gradient, COPIED BEFORE the optimizer step exactly as the worker's
    `retained` is, because `optimizer.step` clips in place and
    `export_gradients()` carries the PRE-clip bytes. Measured 2026-09-19: with
    the copy left out this returned the clipped gradient and the first arm
    fired, `export_gradients[0] ... differ: 32711 bytes of 32768`.

    dropout is 0 in every par-samba configuration, so the driver passes
    `dropout_stream=None` and never draws from the generator; a configuration
    with dropout would need the driver's own stream and is refused here rather
    than silently compared against a different draw."""
    from mojolearn.parallel_training import ordered_sum_gradients
    if plain.config.dropout > 0.0:
        raise ValueError("_ordered_replay_samba cannot replay a dropout stream")
    reduced = None
    for shards in steps:
        parts, offset = [], 0
        for x, y in shards:
            parts.append(plain.loss_and_grads(x, y, dropout_stream=None, token_offset=offset)[1])
            offset += int(np.asarray(x).size)
        total = ordered_sum_gradients(parts)
        reduced = [np.asarray(g).copy() for g in total]
        plain.optimizer.step(total, max_norm=plain.max_norm)
    return reduced


@lane("par-mlp")
def _(ml, X, yc, yr, Xh=None):
    """ParallelNeuralTrainer over the mlp lane's trainer: three logical
    shards of 64 rows, one ordered update per step, three steps; its
    contract is equality to its own one-device replay, not to a plain
    step over 192 rows, so the parts are hashed and not held to mlp.

    IN-CELL ORACLE (2026-09-19): the same three steps replayed on a second
    SmallMLPTrainer built from the same weights through `_ordered_replay_mlp`,
    every weight held to the driver's bit for bit. The parts are unchanged."""
    from mojolearn.parallel_training import ParallelNeuralTrainer
    w = [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
         for s in ((16, 8), (16,), (3, 16), (3,))]
    schedule = {"dataset": "identity_break", "order": "sequential"}
    m = ml.SmallMLPTrainer(*w, data_schedule=schedule)
    plain = ml.SmallMLPTrainer(*w, data_schedule=schedule)
    Xm = np.ascontiguousarray(X[:576, :8])
    t = _three_class(X[:576])
    steps = [[(Xm[192 * s + 64 * k: 192 * s + 64 * (k + 1)], t[192 * s + 64 * k: 192 * s + 64 * (k + 1)])
              for k in range(3)] for s in range(3)]
    with ParallelNeuralTrainer(m, devices=_par_devices(), logical_shards=3) as tr:
        for shards in steps:
            tr.train_step(shards)
        ck = tr.checkpoint()
    _ordered_replay_mlp(plain, steps)
    for k in sorted(m.weights_):
        _same_bytes(f"ParallelNeuralTrainer weights_[{k}]", np.asarray(m.weights_[k]),
                    "ordered single-device replay", np.asarray(plain.weights_[k]))
    _same_bytes("ParallelNeuralTrainer predict_logits", np.asarray(m.predict_logits(Xm[:256])),
                "ordered single-device replay predict_logits", np.asarray(plain.predict_logits(Xm[:256])))
    return _fit(dict(weights=_h(*[np.asarray(m.weights_[k]) for k in sorted(m.weights_)]),
                     logits=_h(np.asarray(m.predict_logits(Xm[:256]))),
                     shards=_h(np.int64(ck["logical_shards"]))),
                m, lambda e: (np.asarray(e.predict_logits(np.ascontiguousarray(Xh[:256, :8]))),))


@lane("par-samba")
def _(ml, X, yc, yr, Xh=None):
    """ParallelNeuralTrainer over the samba lane's stack, two logical shards
    of (2, 17) windows per step, two steps.

    IN-CELL ORACLE (2026-09-19): the same two steps replayed on a second
    SambaStack loaded from the same starting state through
    `_ordered_replay_samba`, every parameter and the logits held to the
    driver's bit for bit. The parts are unchanged."""
    from mojolearn.parallel_training import ParallelNeuralTrainer
    cfg = _par_samba_config(ml)
    m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3)
    plain = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3)
    plain.load_state_dict(m.state_dict())
    ids = _ids(X, 8, 17)
    steps = [[(ids[4 * s + 2 * k: 4 * s + 2 * k + 2, :-1], ids[4 * s + 2 * k: 4 * s + 2 * k + 2, 1:])
              for k in range(2)] for s in range(2)]
    with ParallelNeuralTrainer(m, devices=_par_devices(), logical_shards=2) as tr:
        for shards in steps:
            tr.train_step(shards)
    _ordered_replay_samba(plain, steps)
    params, ref = m.parameters(), plain.parameters()
    for k in sorted(params):
        _same_bytes(f"ParallelNeuralTrainer parameters()[{k}]", np.asarray(params[k]),
                    "ordered single-device replay", np.asarray(ref[k]))
    _same_bytes("ParallelNeuralTrainer forward", np.asarray(m.forward(ids[:2, :-1])),
                "ordered single-device replay forward", np.asarray(plain.forward(ids[:2, :-1])))
    return _fit(dict(logits=_h(np.asarray(m.forward(ids[:2, :-1]))),
                     params=_h(*[np.asarray(params[k]) for k in sorted(params)])),
                m, lambda e: (np.asarray(e.forward(_ids(Xh, 2, 16))),))


@lane("par-byte-lm")
def _(ml, X, yc, yr, Xh=None):
    """ParallelByteLanguageModelTrainer from the byte-lm lane's starting
    state, two logical shards on one device, three steps; the state after
    and the per-shard losses are the parts.

    IN-CELL ORACLE (2026-09-19): the trainer's own documented comparison,
    `pool_optimizer=False`, which "retains complete optimizer replicas for
    comparison" (parallel_training.ParallelByteLanguageModelTrainer). The
    pooled run under test is held, through `_byte_lm_replay`, to a replicated
    one on the first device step by step: the per-shard losses, all four state
    arrays and the exported gradient. It catches a defect in the pooled
    optimizer ranges, the rollback copies, the reduction buffers and the
    cross-device ownership; a defect in the shard fold itself, which both
    sides share, still needs the record.

    THE PARTS DO NOT MOVE, and this is not an argument: `_byte_lm_seed` is
    this lane's own seed construction verbatim and `_byte_lm_replay`'s loop is
    this lane's own loop verbatim, and the committed 166-lane columns
    (2026-09-14, apple-m4 / nvidia-h100 / nvidia-2xh100 / amd-mi325x) record
    this lane's `loss`, `params` and `m` parts BIT-IDENTICAL, on all nine
    fixtures, to the two sibling lanes that already hold their driver to this
    same replicated reference through this same helper.

    NOT RUN HERE. `bindings/_mojolearn_byte_lm_host.mojo` registers no
    `byte_lm_parallel_*` entry, so this lane REFUSES on every CPU column (it
    does on the committed one, `2026-09-16_cpu-par-samba`) and the new arm's
    first binding-level run is owed on a GPU column."""
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer
    shape, state0 = _byte_lm_seed(ml)
    ids = _ids(X, 6 * shape.batch, shape.length + 1)
    with ParallelByteLanguageModelTrainer(state0, devices=_par_devices(), logical_shards=2) as tr, \
         ParallelByteLanguageModelTrainer(state0, devices=_par_devices()[:1], logical_shards=2,
                                          pool_optimizer=False) as ref:
        losses, state, _ = _byte_lm_replay(tr, ref, ids)
    return _fit(dict(loss=_h(np.asarray(losses)), params=_h(np.asarray(state["parameters"])),
                     m=_h(np.asarray(state["m"]))))


# ---------------------------------------------------------------- lanes (2026-09-14 night, the multi-GPU drivers the par-* lanes above missed)
# A repo-wide grep of this file on main e22374acd found no lane for the query
# drivers (parallel_neighbors.ParallelQueries,
# parallel_neighbors_reference.ReferenceShardedNeighbors), the graph drivers
# (parallel_graph.fit_graph, transform_umap), the ordered and two-level GBDT
# drivers and the pointwise searcher under fit_boosting,
# fit_exponential_smoothing, the layer-owned and host-offloaded byte-LM
# trainers, or the pooled clipping path of ParallelNeuralTrainer (reached
# only with a clip). Same rules as the lanes above: devices=_par_devices(),
# the smallest sharding that splits the work, and `_same_bytes` against the
# plain fit or the replica trainer wherever the driver's contract is
# equality. The pooled neural gradient buffers and the resident isolation
# forest model pool ride inside par-mlp, par-samba and par-iforest; those
# lanes' code paths are the pooled ones since the drivers changed.

#: query rows per ParallelQueries shard: 64 query rows are four shards
PAR_QUERY_ROWS = 16


def _pq(e, R, method, **kw):
    """One ParallelQueries call on a fitted estimator, the pool opened and
    closed around it."""
    from mojolearn.parallel_neighbors import ParallelQueries
    with ParallelQueries(e, devices=_par_devices(), rows_per_shard=PAR_QUERY_ROWS) as pq:
        return pq.query(np.ascontiguousarray(R), method=method, **kw)


@lane("par-queries-knn")
def _(ml, X, yc, yr, Xh=None):
    """ParallelQueries over the knn-clf lane's KNeighborsClassifier, 64 query
    rows in shards of 16: kneighbors, predict and predict_proba, each held to
    the plain call."""
    from mojolearn.parallel_neighbors import ParallelQueries
    m = ml.KNeighborsClassifier(n_neighbors=8).fit(X[:4096], yc[:4096])
    q = np.ascontiguousarray(X[4096:4160])
    with ParallelQueries(m, devices=_par_devices(), rows_per_shard=PAR_QUERY_ROWS) as pq:
        d, i = pq.query(q, method="kneighbors")
        pr = pq.query(q, method="predict")
        pp = pq.query(q, method="predict_proba")
    d0, i0 = m.kneighbors(q)
    _same_bytes("ParallelQueries kneighbors distances", d, "plain distances", d0)
    _same_bytes("ParallelQueries kneighbors indices", i, "plain indices", i0)
    _same_bytes("ParallelQueries predict", pr, "plain predict", m.predict(q))
    _same_bytes("ParallelQueries predict_proba", pp, "plain predict_proba", m.predict_proba(q))
    return _fit(dict(dist=_h(d), idx=_h(i), predict=_h(pr), proba=_h(pp)),
                m, lambda e: (_pq(e, Xh[:64], "predict"), _pq(e, Xh[:64], "predict_proba")))


@lane("par-queries-radius")
def _(ml, X, yc, yr, Xh=None):
    """ParallelQueries over the radius lane's RadiusNeighbors, ragged rows
    joined in input order, held to the plain call."""
    index, q = X[:4096], np.ascontiguousarray(X[4096:4160])
    r = _radius_for(index, q)
    m = ml.RadiusNeighbors(radius=r).fit(index)
    par = _ragged(_pq(m, q, "radius_neighbors", sort_results=True))
    plain = _ragged(m.radius_neighbors(q, sort_results=True))
    for k, name in enumerate(("counts", "distances", "indices")):
        _same_bytes(f"ParallelQueries radius_neighbors {name}", par[k], f"plain {name}", plain[k])
    lens, dd, ii = par
    return _fit(dict(radius=_h(np.float32(r)), counts=_h(lens), dist=_h(dd), idx=_h(ii)),
                m, lambda e: _ragged(_pq(e, Xh[:64], "radius_neighbors", sort_results=True)))


@lane("par-queries-kde")
def _(ml, X, yc, yr, Xh=None):
    """ParallelQueries over the kde lane's KernelDensity, 256 query rows in
    shards of 16, held to the plain score_samples."""
    m = ml.KernelDensity(bandwidth=0.7).fit(X[:4096, :4])
    q = np.ascontiguousarray(X[4096:4352, :4])
    s = _pq(m, q, "score_samples")
    _same_bytes("ParallelQueries score_samples", s, "plain score_samples", m.score_samples(q))
    return _fit(dict(scores=_h(s)), m, lambda e: (_pq(e, Xh[:256, :4], "score_samples"),))


#: reference rows per ReferenceShardedNeighbors shard: the 4096-row index is four shards
PAR_REFERENCE_ROWS = 1024


def _rsn(e):
    from mojolearn.parallel_neighbors_reference import ReferenceShardedNeighbors
    return ReferenceShardedNeighbors(e, devices=_par_devices(), reference_rows_per_shard=PAR_REFERENCE_ROWS,
                                     query_rows_per_shard=PAR_QUERY_ROWS)


@lane("par-reference-knn")
def _(ml, X, yc, yr, Xh=None):
    """ReferenceShardedNeighbors over the knn-clf lane's classifier: the
    4096-row reference in four shards of 1024, 64 query rows in shards of
    16, the merged distance-bit keys and the original vote half, held to the
    plain kneighbors, predict and predict_proba."""
    m = ml.KNeighborsClassifier(n_neighbors=8).fit(X[:4096], yc[:4096])
    q = np.ascontiguousarray(X[4096:4160])
    with _rsn(m) as rs:
        d, i = rs.kneighbors(q)
        pr, pp = rs.predict(q), rs.predict_proba(q)
    d0, i0 = m.kneighbors(q)
    parts = dict(dist=_h(d), idx=_h(i), predict=_h(pr), proba=_h(pp))
    # Retain measured bytes when the independent comparison detects a fault;
    # a generic refusal would discard the evidence and cannot qualify a control.
    for name, actual, expected in (("distances", d, d0), ("indices", i, i0),
                                   ("predict", pr, m.predict(q)),
                                   ("predict_proba", pp, m.predict_proba(q))):
        mismatch = _mismatch_bytes("ReferenceShardedNeighbors " + name, actual,
                                   "plain " + name, expected)
        if mismatch:
            raise NumericalMismatch(mismatch, parts)

    def probe(e):
        with _rsn(e) as rs:
            return rs.predict(Xh[:64]), rs.predict_proba(Xh[:64])
    return _fit(parts, m, probe)


@lane("par-reference-knn-reg")
def _(ml, X, yc, yr, Xh=None):
    """ReferenceShardedNeighbors over the knn-reg lane's regressor, the
    merged neighbors' regression vote, held to the plain predict."""
    m = ml.KNeighborsRegressor(n_neighbors=8).fit(X[:4096], yr[:4096])
    q = np.ascontiguousarray(X[4096:4160])
    with _rsn(m) as rs:
        pr = rs.predict(q)
    parts = dict(predict=_h(pr))
    mismatch = _mismatch_bytes("ReferenceShardedNeighbors predict", pr, "plain predict", m.predict(q))
    if mismatch:
        raise NumericalMismatch(mismatch, parts)

    def probe(e):
        with _rsn(e) as rs:
            return (rs.predict(Xh[:64]),)
    return _fit(parts, m, probe)


@lane("par-graph-agglomerative")
def _(ml, X, yc, yr, Xh=None):
    """fit_graph over the agglomerative lane's fit, held to the plain labels
    and merge tree."""
    from mojolearn.parallel_graph import fit_graph
    A = np.ascontiguousarray(X[:2000, :4])
    par = fit_graph(ml.AgglomerativeClustering(n_clusters=4, prediction_data=True), A, devices=_par_devices())
    plain = ml.AgglomerativeClustering(n_clusters=4).fit(A)
    _same_bytes("fit_graph labels_", par.labels_, "plain labels_", plain.labels_)
    _same_bytes("fit_graph children_", par.children_, "plain children_", plain.children_)
    return _fit(dict(labels=_h(par.labels_), children=_h(par.children_)), par,
                lambda e: (e.predict(Xh[:256, :4]),))


@lane("par-graph-spectral")
def _(ml, X, yc, yr, Xh=None):
    """fit_graph over the spectral lane's fit (the eigensolver on the root,
    the neighbor rows and the KMeans assignment distributed), held to the
    plain labels and embedding."""
    from mojolearn.parallel_graph import fit_graph
    A = np.ascontiguousarray(X[:2000, :4])
    par = fit_graph(ml.SpectralClustering(n_clusters=4, random_state=3), A, devices=_par_devices())
    plain = ml.SpectralClustering(n_clusters=4, random_state=3).fit(A)
    _same_bytes("fit_graph labels_", par.labels_, "plain labels_", plain.labels_)
    _same_bytes("fit_graph embedding_", par.embedding_, "plain embedding_", plain.embedding_)
    return _fit(dict(labels=_h(par.labels_), embedding=_h(par.embedding_)), par, "n/a:transductive")


@lane("par-graph-umap")
def _(ml, X, yc, yr, Xh=None):
    """fit_graph and transform_umap over the umap lane's fit, held to the
    plain embedding and the plain transform of 64 further rows."""
    from mojolearn.parallel_graph import fit_graph, transform_umap
    kw = dict(n_neighbors=8, n_components=2, n_epochs=8, random_state=3)
    A = np.ascontiguousarray(X[:1024, :8])
    par = fit_graph(ml.UMAP(**kw), A, devices=_par_devices())
    plain = ml.UMAP(**kw).fit(A)
    _same_bytes("fit_graph embedding_", par.embedding_, "plain embedding_", plain.embedding_)
    q = np.ascontiguousarray(X[1024:1088, :8])
    t = transform_umap(par, q, devices=_par_devices())
    _same_bytes("transform_umap", t, "plain transform", plain.transform(q))
    return _fit(dict(embedding=_h(par.embedding_), transform=_h(t)),
                par, lambda e: (transform_umap(e, np.ascontiguousarray(Xh[:64, :8]), devices=_par_devices()),))


@lane("par-ordered-rmse")
def _(ml, X, yc, yr, Xh=None):
    """fit_ordered_rmse on the gbdt-ordered-rmse lane's fit and permutation,
    held to the plain fit."""
    from mojolearn.parallel_ensemble import fit_ordered_rmse
    perm = np.argsort(_hashed_uniform(X.shape[0], 1, "ordered-permutation").reshape(-1), kind="stable")
    par = fit_ordered_rmse(ml.OrderedRMSE(n_estimators=20, max_depth=6), X, yr, permutation=perm,
                           devices=_par_devices())
    plain = ml.OrderedRMSE(n_estimators=20, max_depth=6).fit(X, yr, permutation=perm)
    p = par.predict(X)
    _same_bytes("fit_ordered_rmse predict", p, "plain predict", plain.predict(X))
    return _fit(dict(predict=_h(p)), par, lambda e: (e.predict(Xh),))


@lane("par-ordered")
def _(ml, X, yc, yr, Xh=None):
    """fit_boosting on the gbdt-ordered lane's Logloss fit (feature groups of
    the fold histograms partitioned; permutations, folds, cursors and leaves
    on the root), held to the plain one-device fit."""
    from mojolearn.parallel_ensemble import fit_boosting
    kw = dict(n_estimators=20, max_depth=6, loss="Logloss", boosting_type="Ordered")
    par = fit_boosting(_gbdt(ml.GradientBoosting, **kw), X, yc, devices=_par_devices())
    plain = _gbdt(ml.GradientBoosting, **kw).fit(X, yc)
    p = par.predict(X)
    _same_bytes("fit_boosting ordered predict", p, "plain predict", plain.predict(X))
    return _fit(dict(predict=_h(p)), par, lambda e: (e.predict(Xh),))


@lane("par-border-types")
def _(ml, X, yc, yr, Xh=None):
    """fit_boosting on the gbdt-border-types lane's six fits (feature
    histograms partitioned by whole packed groups; border selection is host
    code on the root), each held to the plain one-device fit."""
    from mojolearn.parallel_ensemble import fit_boosting
    parts, first = {}, None
    for bt in _GBDT_BORDER_TYPES:
        kw = dict(n_estimators=8, max_depth=4, border_count=32, loss="Logloss",
                  feature_border_type=bt)
        par = fit_boosting(_gbdt(ml.GradientBoosting, **kw), X, yc, devices=_par_devices())
        plain = _gbdt(ml.GradientBoosting, **kw).fit(X, yc)
        p = par.predict(X)
        _same_bytes(f"fit_boosting {bt} predict", p, "plain predict", plain.predict(X))
        parts[bt] = _h(p)
        first = first or par
    return _fit(parts, first, lambda e: (e.predict(Xh),))


@lane("par-feature-freq")
def _(ml, X, yc, yr, Xh=None):
    """fit_feature_freq on the gbdt-feature-freq lane's fit, held to the plain fit."""
    from mojolearn.parallel_ensemble import fit_feature_freq
    Xc = _coded(X)
    par = fit_feature_freq(ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7), Xc, yr,
                           devices=_par_devices())
    plain = ml.ExperimentalTwoLevelFeatureFreq(sources=[0, 1], random_state=7).fit(Xc, yr)
    p = par.predict(Xc)
    _same_bytes("fit_feature_freq predict", p, "plain predict", plain.predict(Xc))
    return _fit(dict(predict=_h(p)), par, lambda e: (e.predict(_coded(Xh)),))


@lane("par-boosting-pointwise")
def _(ml, X, yc, yr, Xh=None):
    """fit_boosting with use_pointwise_searcher=True (whole packed feature
    groups per device, the root gathering the interleaved weight and target
    pairs), held to the plain pointwise fit."""
    from mojolearn.parallel_ensemble import fit_boosting
    kw = dict(n_estimators=20, max_depth=6, loss="Logloss", use_pointwise_searcher=True)
    par = fit_boosting(_gbdt(ml.GradientBoosting, **kw), X, yc, devices=_par_devices())
    plain = _gbdt(ml.GradientBoosting, **kw).fit(X, yc)
    _same_bytes("fit_boosting pointwise predict_proba", par.predict_proba(X[:2048]),
                "plain predict_proba", plain.predict_proba(X[:2048]))
    return _fit(dict(predict=_h(par.predict(X)), proba=_h(par.predict_proba(X))),
                par, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


def _hw_series(X):
    """Four positive series of 512 observations, one per ROW (the
    `(ts_num, n)` layout), from the fixture's first four columns by the
    holtwinters lane's rule applied per series."""
    S = np.cumsum(X[:512, :4], axis=0).T.astype(np.float32) + np.float32(50.0)
    return np.ascontiguousarray((S - S.min(axis=1, keepdims=True) + np.float32(1.0)).astype(np.float32))


@lane("par-holtwinters")
def _(ml, X, yc, yr, Xh=None):
    """fit_exponential_smoothing, four series in shards of two, held to the
    plain four-series fit on every fitted array and the forecast."""
    from mojolearn.parallel_classical import fit_exponential_smoothing
    S = _hw_series(X)
    kw = dict(seasonal="additive", seasonal_periods=12, ts_num=4)
    par = fit_exponential_smoothing(ml.ExponentialSmoothing(S, **kw), devices=_par_devices(), series_per_shard=2)
    plain = ml.ExponentialSmoothing(S, **kw).fit()
    names = ("level_", "trend_", "season_", "sse_", "alpha_", "beta_", "gamma_", "n_iter_", "criterion_")
    for name in names:
        _same_bytes(f"fit_exponential_smoothing {name}", getattr(par, name), f"plain {name}", getattr(plain, name))
    f = par.forecast(24)
    _same_bytes("fit_exponential_smoothing forecast(24)", f, "plain forecast(24)", plain.forecast(24))
    return _fit(dict(**{n.strip("_"): _h(getattr(par, n)) for n in names}, forecast=_h(f)),
                par, lambda e: (e.forecast(FORECAST_HORIZON),))


@lane("par-forecast-holtwinters")
def _(ml, X, yc, yr, Xh=None):
    """predict_exponential_smoothing and forecast_exponential_smoothing --
    the PREDICTION drivers, which par-holtwinters does not reach -- over the
    same four series, three to a shard. That 3 + 1 split exercises the
    one-series shard, whose worker returns cuML's 1-D shape instead of a
    `(steps, ts_num)` block and whose components slice is a single column;
    the `index=` arm asks for the third return shape on top of it.
    In-sample prediction is refused by the driver by name, so `start = n`.

    IN-CELL ORACLE (2026-09-19, lane/lm-attention-fallback): par-forecast-
    arima's, read on this driver -- the sharded answer held byte for byte,
    in the cell, to `ExponentialSmoothing.predict` and `.forecast` of the
    same fitted estimator, which is what a series partition is claimed to
    leave alone. Reachable on a CPU column only since `forecast_predict`
    joined `_parallel_pool.CPU_OPERATIONS`."""
    from mojolearn.parallel_forecasting import (forecast_exponential_smoothing,
                                                predict_exponential_smoothing)
    S = _hw_series(X)
    d = _par_devices()
    m = ml.ExponentialSmoothing(S, seasonal="additive", seasonal_periods=12, ts_num=4).fit()
    sharded = predict_exponential_smoothing(m, m.n, m.n + 24, devices=d, series_per_shard=3)
    _same_bytes("predict_exponential_smoothing(n, n + 24)", sharded,
                "plain ExponentialSmoothing.predict(n, n + 24)", m.predict(m.n, m.n + 24))
    ahead = forecast_exponential_smoothing(m, 24, devices=d, series_per_shard=3)
    _same_bytes("forecast_exponential_smoothing(24)", ahead,
                "plain ExponentialSmoothing.forecast(24)", m.forecast(24))
    one = forecast_exponential_smoothing(m, 24, index=2, devices=d, series_per_shard=3)
    _same_bytes("forecast_exponential_smoothing(24, index=2)", one,
                "plain ExponentialSmoothing.forecast(24, index=2)", m.forecast(24, index=2))
    return _fit(dict(predict=_h(sharded), forecast=_h(ahead), indexed=_h(one)), m,
                lambda e: _same_bytes(
                    f"forecast_exponential_smoothing({FORECAST_HORIZON})",
                    forecast_exponential_smoothing(e, FORECAST_HORIZON, devices=d, series_per_shard=3),
                    f"plain ExponentialSmoothing.forecast({FORECAST_HORIZON})", e.forecast(FORECAST_HORIZON)))


def _byte_lm_seed(ml):
    shape = _par_byte_lm_config(ml)
    named, _ = _byte_lm_params(shape)
    seed = ml.SmallByteLanguageModelTrainer(named, data_schedule={"dataset": "identity_break", "order": "sequential"},
                                            shape=shape)
    return shape, seed.state_dict()


def _same_state(name_a, a, name_b, b):
    for key in ("parameters", "m", "v", "flags"):
        _same_bytes(f"{name_a} {key}", np.asarray(a[key]), f"{name_b} {key}", np.asarray(b[key]))


def _byte_lm_replay(trainer, reference, ids):
    """Three steps of two logical shards on both trainers from the same
    state, losses held equal step by step; returns (losses, state, grads)."""
    losses = []
    for step in range(3):
        shards = [ids[4 * step: 4 * step + 2], ids[4 * step + 2: 4 * step + 4]]
        a, b = trainer.train_step(shards), reference.train_step(shards)
        _same_bytes(f"step {step} losses", np.asarray(a["losses"], dtype=np.float32),
                    "replica trainer losses", np.asarray(b["losses"], dtype=np.float32))
        losses.append([np.float64(v) for v in a["losses"]])
    state, ref = trainer.state_dict(), reference.state_dict()
    _same_state("trainer", state, "replica trainer", ref)
    grads = trainer.export_gradients()
    _same_bytes("export_gradients", grads, "replica trainer export_gradients", reference.export_gradients())
    return losses, state, grads


@lane("par-byte-lm-model-pool")
def _(ml, X, yc, yr, Xh=None):
    """PooledByteLanguageModelTrainer (decoder layers owned by devices, the
    embedding and head on the first) from the byte-lm lane's starting
    state, two logical shards, three steps, held to the replicated
    ParallelByteLanguageModelTrainer (pool_optimizer=False) on the first
    device: losses, parameters, moments, flags and gradients, the
    comparison tools/byte_lm_model_pool_check.py makes."""
    from mojolearn.model_pool_training import PooledByteLanguageModelTrainer
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer
    shape, state0 = _byte_lm_seed(ml)
    ids = _ids(X, 6 * shape.batch, shape.length + 1)
    with PooledByteLanguageModelTrainer(state0, devices=_par_devices(), logical_shards=2) as tr, \
         ParallelByteLanguageModelTrainer(state0, devices=_par_devices()[:1], logical_shards=2,
                                          pool_optimizer=False) as ref:
        losses, state, grads = _byte_lm_replay(tr, ref, ids)
    return _fit(dict(loss=_h(np.asarray(losses)), params=_h(np.asarray(state["parameters"])),
                     m=_h(np.asarray(state["m"])), v=_h(np.asarray(state["v"])), grads=_h(np.asarray(grads))))


@lane("par-byte-lm-offload")
def _(ml, X, yc, yr, Xh=None):
    """OffloadedByteLanguageModelTrainer, which admits exactly one device
    (the first of _par_devices(), so a two-device column runs it on one),
    held to the replicated ParallelByteLanguageModelTrainer the same way,
    the comparison tools/byte_lm_offload_check.py makes."""
    from mojolearn.offload_training import OffloadedByteLanguageModelTrainer
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer
    shape, state0 = _byte_lm_seed(ml)
    ids = _ids(X, 6 * shape.batch, shape.length + 1)
    with OffloadedByteLanguageModelTrainer(state0, devices=_par_devices()[:1], logical_shards=2) as tr, \
         ParallelByteLanguageModelTrainer(state0, devices=_par_devices()[:1], logical_shards=2,
                                          pool_optimizer=False) as ref:
        losses, state, grads = _byte_lm_replay(tr, ref, ids)
    return _fit(dict(loss=_h(np.asarray(losses)), params=_h(np.asarray(state["parameters"])),
                     m=_h(np.asarray(state["m"])), v=_h(np.asarray(state["v"])), grads=_h(np.asarray(grads))))


@lane("par-samba-clip")
def _(ml, X, yc, yr, Xh=None):
    """ParallelNeuralTrainer over the par-samba lane's stack with a global
    norm clip (max_norm=0.5), the only way into the pooled clipping tensors
    (whole tensors on owners, the cross-tensor norm on the first device);
    two logical shards of (2, 17) windows, two steps.

    IN-CELL ORACLE (2026-09-19): the same two steps replayed on a second
    SambaStack loaded from the same starting state through
    `_ordered_replay_samba`, which passes the SAME `max_norm` to the
    single-device optimizer, so the clip is in the comparison and not beside
    it. The retained pre-update gradient `export_gradients()` carries, every
    parameter and the logits are held to the replay's bit for bit. The parts
    are unchanged."""
    from mojolearn.parallel_training import ParallelNeuralTrainer
    cfg = _par_samba_config(ml)
    m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3, max_norm=0.5)
    plain = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3, max_norm=0.5)
    plain.load_state_dict(m.state_dict())
    ids = _ids(X, 8, 17)
    steps = [[(ids[4 * s + 2 * k: 4 * s + 2 * k + 2, :-1], ids[4 * s + 2 * k: 4 * s + 2 * k + 2, 1:])
              for k in range(2)] for s in range(2)]
    with ParallelNeuralTrainer(m, devices=_par_devices(), logical_shards=2) as tr:
        losses = []
        for shards in steps:
            res = tr.train_step(shards)
            losses.append([np.float64(v) for v in res["losses"]])
        grads = tr.export_gradients()
    reduced = _ordered_replay_samba(plain, steps)
    params, ref = m.parameters(), plain.parameters()
    # zip() truncates, so a short list would quietly shrink the comparison.
    _same_bytes("export_gradients tensor count", np.int64(len(grads)),
                "ordered single-device replay tensor count", np.int64(len(reduced)))
    for k, (g, r) in enumerate(zip(grads, reduced)):
        _same_bytes(f"ParallelNeuralTrainer export_gradients[{k}]", np.asarray(g),
                    "ordered single-device replay gradient", np.asarray(r))
    for k in sorted(params):
        _same_bytes(f"ParallelNeuralTrainer parameters()[{k}]", np.asarray(params[k]),
                    "ordered single-device replay", np.asarray(ref[k]))
    _same_bytes("ParallelNeuralTrainer forward", np.asarray(m.forward(ids[:2, :-1])),
                "ordered single-device replay forward", np.asarray(plain.forward(ids[:2, :-1])))
    return _fit(dict(loss=_h(np.asarray(losses)), logits=_h(np.asarray(m.forward(ids[:2, :-1]))),
                     grads=_h(*[np.asarray(g) for g in grads]),
                     params=_h(*[np.asarray(params[k]) for k in sorted(params)])),
                m, lambda e: (np.asarray(e.forward(_ids(Xh, 2, 16))),))



# LAST ON PURPOSE (2026-08-29): on a RunPod RTX 4090 the isolation forest
# binding hung at its first fit in every tier (a second DeviceContext beside
# the caller's deadlocked at teardown on sm_89; fixed by DEVIATION 1944, the
# estimator now uses the caller's context), and a SECOND 4090 defect made the
# next GPU call after one RandomForest.fit hang -- DEVIATION 1946, the same
# class one call later: the binding's context was destroyed at its last use
# and its own buffers were freed behind it. 1946 is FIXED IN SOURCE AND UNRUN
# ON A 4090 (see bench/results/identity_break/RESULTS.md, RUN OWED). A hang
# cannot be caught from inside this process, so the JSON is written after
# every lane, the caller wraps the run in `timeout`, and the historically
# hanging lane runs last; a killed run still leaves every earlier lane on
# disk.
@lane("iforest")
def _(ml, X, yc, yr, Xh=None):
    m = ml.IsolationForest(n_estimators=16, random_state=5).fit(X)
    return _fit(dict(scores=_h(m.score_samples(X)), predict=_h(m.predict(X[:512]))),
                m, lambda e: (e.score_samples(Xh), e.predict(Xh[:512])))

@lane("iforest-tuned")
def _(ml, X, yc, yr, Xh=None):
    """An integer subsample, a column fraction, the with-replacement draw,
    a contamination percentile offset and an explicit depth cap: every
    isolation-forest knob the iforest lane leaves at its default. After
    iforest for the same reason it runs last."""
    m = ml.IsolationForest(n_estimators=16, random_state=5, max_samples=512, max_features=0.5,
                           bootstrap=True, contamination=0.1, max_depth=6).fit(X)
    return _fit(dict(scores=_h(m.score_samples(X)), predict=_h(m.predict(X[:512]))),
                m, lambda e: (e.score_samples(Xh), e.predict(Xh[:512])))

@lane("par-iforest")
def _(ml, X, yc, yr, Xh=None):
    """fit_isolation_forest and score_isolation_forest on the iforest lane's
    configuration, tree ranges on one device, held to the plain fit; after
    iforest-tuned for the reason iforest runs last."""
    from mojolearn.parallel_ensemble import fit_isolation_forest, score_isolation_forest
    par = fit_isolation_forest(ml.IsolationForest(n_estimators=16, random_state=5), X, devices=_par_devices())
    plain = ml.IsolationForest(n_estimators=16, random_state=5).fit(X)
    scores = score_isolation_forest(par, X, devices=_par_devices(), method="score_samples")
    _same_bytes("score_isolation_forest", scores, "plain score_samples", plain.score_samples(X))
    return _fit(dict(scores=_h(scores), predict=_h(par.predict(X[:512]))),
                par, lambda e: (e.score_samples(Xh), e.predict(Xh[:512])))


# ---------------------------------------------------------------- lanes (2026-09-14 night, drivers added by the multigpu lane)
# ParallelForestPredictor (resident RF/ET groves, 32 fixed logical groves) and
# fit_gaussian_mixture / predict_gaussian_mixture (row-sharded E-steps). Same
# rules as the par-* lanes above: devices=_par_devices() and `_same_bytes`
# against the one-device public path the driver promises to equal.

@lane("par-forest-pool")
def _(ml, X, yc, yr, Xh=None):
    """ParallelForestPredictor over a 40-tree RandomForestClassifier with
    inference_engine='parallel_groves' (40 trees put two trees in groves 0..7,
    so the within-grove order is reached), held to the estimator's own
    parallel_groves predictions."""
    from mojolearn.parallel_ensemble import ParallelForestPredictor
    m = ml.RandomForestClassifier(n_estimators=40, max_depth=8, random_state=7,
                                  inference_engine="parallel_groves").fit(X, yc)

    def pooled(e, R):
        # A reloaded model is asked the same way; its engine is set, not assumed.
        e.inference_engine = "parallel_groves"
        with ParallelForestPredictor(e, devices=_par_devices()) as pool:
            return pool.predict(R), pool.predict_proba(R)

    pred, proba = pooled(m, X)
    _same_bytes("ParallelForestPredictor predict_proba", proba, "parallel_groves predict_proba", m.predict_proba(X))
    return _fit(dict(predict=_h(pred), proba=_h(proba)), m, lambda e: pooled(e, Xh))


@lane("par-gmm")
def _(ml, X, yc, yr, Xh=None):
    """fit_gaussian_mixture and predict_gaussian_mixture on the gmm lane's
    configuration, held to the plain fit's covariances and labels."""
    from mojolearn.parallel_classical import fit_gaussian_mixture, predict_gaussian_mixture
    kw = dict(n_components=4, max_iter=30, random_state=3)
    par = fit_gaussian_mixture(ml.GaussianMixture(**kw), X[:6000, :4], devices=_par_devices())
    plain = ml.GaussianMixture(**kw).fit(X[:6000, :4])
    _same_bytes("fit_gaussian_mixture covariances_", par.covariances_, "plain covariances_", plain.covariances_)
    labels = predict_gaussian_mixture(par, X[:6000, :4], devices=_par_devices(), method="predict")
    _same_bytes("predict_gaussian_mixture", labels, "plain predict", plain.predict(X[:6000, :4]))
    return _fit(dict(weights=_h(par.weights_), means=_h(par.means_), covariances=_h(par.covariances_),
                     precisions=_h(par.precisions_cholesky_), n_iter=_h(np.int64(par.n_iter_)),
                     lower_bound=_h(np.float32(par.lower_bound_)), labels=_h(labels)),
                par, lambda e: (predict_gaussian_mixture(e, Xh[:64, :4], devices=_par_devices(), method="score_samples"),
                                predict_gaussian_mixture(e, Xh[:64, :4], devices=_par_devices(), method="predict")))


@lane("par-resample")
def _(ml, X, yc, yr, Xh=None):
    """parallel_classical.bootstrap, permutation_test and
    monte_carlo_integrate (global replicate, permutation and 256-sample
    chunk ranges on _par_devices()) on the bootstrap, permutation-test and
    monte-carlo lanes' inputs, each held to the one-device public function."""
    from mojolearn import parallel_classical as pc
    rs = ml.resample
    dev = _par_devices()
    x = np.ascontiguousarray(yr[:4096])
    parts = {}
    for name, kw in (("mean", dict(statistic="mean", n_resamples=2048)),
                     ("quantile", dict(statistic="quantile", q_or_prop=0.25, n_resamples=1024, alternative="less"))):
        b = pc.bootstrap(x, devices=dev, random_state=3, **kw)
        plain = rs.bootstrap(x, random_state=3, **kw)
        _same_bytes("parallel bootstrap distribution", b.distribution, "bootstrap distribution", plain.distribution)
        parts["bootstrap-" + name] = _h(b.distribution, b.sorted_distribution,
                                        np.asarray([b.point_estimate, b.standard_error, b.confidence_interval[0],
                                                    b.confidence_interval[1]], dtype=np.float64))
    a = np.ascontiguousarray(yr[:512])
    c = np.ascontiguousarray(yr[512:1024])
    p = pc.permutation_test(a, c, devices=dev, statistic="diff_means", n_resamples=2048, random_state=3)
    plain = rs.permutation_test(a, c, statistic="diff_means", n_resamples=2048, random_state=3)
    _same_bytes("parallel permutation null", p.null_distribution, "permutation null", plain.null_distribution)
    parts["permutation"] = _h(p.null_distribution, np.asarray([p.statistic, p.pvalue], dtype=np.float64),
                              np.asarray([p.count_less, p.count_greater], dtype=np.int64))
    lo, hi = [0.0, 0.0], [1.0, 2.0]
    r = pc.monte_carlo_integrate("product", lo, hi, 65536 + 300, devices=dev, random_state=1)
    plain = rs.monte_carlo_integrate("product", lo, hi, 65536 + 300, random_state=1)
    _same_bytes("parallel monte carlo", np.asarray([r.integral, r.mean]), "monte carlo", np.asarray([plain.integral, plain.mean]))
    parts["monte-carlo"] = _h(np.asarray([r.integral, r.mean, r.volume, r.closed_form], dtype=np.float64))
    return _fit(parts)


@lane("par-hdbscan")
def _(ml, X, yc, yr, Xh=None):
    """fit_hdbscan on the hdbscan lane's configuration (the neighbors and
    hierarchy row drivers under _par_devices()), held to the plain fit's
    labels and core distances.

    ROWS: this lane keeps 6000 while the plain hdbscan lane came down to 4000
    (2026-09-16, lane/identity-fixtures-light). Deliberate, and it breaks
    nothing: this lane builds BOTH sides of its own comparison at the same
    size, so `_same_bytes` below still compares like with like. It was left
    alone because `par-*` is out of the release record's scope
    (RECORD_EXCLUDED_PREFIXES), so shrinking it would buy no column time while
    costing another hash revision. "The hdbscan lane's configuration" means
    its estimator settings, not its row count."""
    from mojolearn.parallel_classical import fit_hdbscan
    par = fit_hdbscan(ml.HDBSCAN(min_cluster_size=5, prediction_data=True), X[:6000, :4], devices=_par_devices())
    plain = ml.HDBSCAN(min_cluster_size=5).fit(X[:6000, :4])
    _same_bytes("fit_hdbscan labels_", par.labels_, "plain labels_", plain.labels_)
    _same_bytes("fit_hdbscan core_distances_", par.core_distances_, "plain core_distances_", plain.core_distances_)
    return _fit(dict(labels=_h(par.labels_), core=_h(par.core_distances_),
                     counts=_h(np.asarray([par.n_clusters_, par.n_outliers_, par.n_boruvka_rounds_,
                                           par.n_condensed_clusters_], dtype=np.int64))),
                par, lambda e: tuple(ml.hdbscan.approximate_predict(e, Xh[:256, :4])))


# ---------------------------------------------------------------- lanes (2026-09-15, the kernel-method and Cholesky drivers)
# fit_cholesky / solve_cholesky, fit_kernel_method / apply_kernel_method and
# transform_rbf_sampler. The shapes sit ABOVE 1 MiB on purpose: a 600 x 600
# float32 factor is 1.44 MiB, the size at which the device-to-device column
# solve read wrong columns on two MI300X
# (bench/results/multi_gpu/2026-09-14/cholesky-mi300x-diag/). Same rules as
# the par-* lanes above.

@lane("par-cholesky")
def _(ml, X, yc, yr, Xh=None):
    """fit_cholesky and solve_cholesky on a 600 x 600 Cauchy-kernel SPD
    matrix with three right-hand sides (whole trailing-update rows and whole
    right-hand-side columns on _par_devices()), held to the plain fit's
    factor and solve."""
    from mojolearn.parallel_classical import fit_cholesky, solve_cholesky
    dev = _par_devices()
    A = _cauchy_spd(X[:600, :4])
    par = fit_cholesky(ml.Cholesky(), A, devices=dev)
    plain = ml.Cholesky().fit(A)
    assert par.info_ == 0, "par-cholesky lane: the Cauchy matrix did not factor (info=%d)" % par.info_
    _same_bytes("fit_cholesky L_", par.L_, "plain L_", plain.L_)
    B = np.ascontiguousarray(np.stack([yr[:600], yr[600:1200], yr[1200:1800]], 1).astype(np.float32))
    x = solve_cholesky(par, B, devices=dev)
    _same_bytes("solve_cholesky", x, "plain solve", plain.solve(B))
    return _fit(dict(L=_h(par.L_), logdet=_h(np.float64(par.logdet_)), info=_h(np.int64(par.info_)),
                     nb=_h(np.int64(par.nb_)), jitter=_h(np.float32(par.jitter_)), solve=_h(x)),
                par, lambda e: (solve_cholesky(e, np.ascontiguousarray(Xh[:600, :3]), devices=_par_devices()),))


@lane("par-kernel-ridge")
def _(ml, X, yc, yr, Xh=None):
    """fit_kernel_method and apply_kernel_method on a KernelRidge at the rbf
    kernel over 600 rows with two targets (kernel rows through the SVM seam,
    factor rows and target columns through the Cholesky driver), held to the
    plain fit's dual coefficients and predictions."""
    from mojolearn.parallel_classical import fit_kernel_method, apply_kernel_method
    dev = _par_devices()
    kw = dict(alpha=0.1, kernel="rbf", gamma=0.5)
    Y = np.ascontiguousarray(np.stack([yr[:600], yr[600:1200]], 1).astype(np.float32))
    par = fit_kernel_method(ml.KernelRidge(**kw), X[:600, :4], Y, devices=dev)
    plain = ml.KernelRidge(**kw).fit(X[:600, :4], Y)
    _same_bytes("fit_kernel_method dual_coef_", par.dual_coef_, "plain dual_coef_", plain.dual_coef_)
    pred = apply_kernel_method(par, X[600:856, :4], devices=dev)
    _same_bytes("apply_kernel_method predict", pred, "plain predict", plain.predict(X[600:856, :4]))
    return _fit(dict(dual=_h(par.dual_coef_), info=_h(np.int64(par.info_)), predict=_h(pred)),
                par, lambda e: (apply_kernel_method(e, Xh[:64, :4], devices=_par_devices()),))


@lane("par-nystroem")
def _(ml, X, yc, yr, Xh=None):
    """fit_kernel_method and apply_kernel_method on a Nystroem at the rbf
    kernel with 32 components of 600 rows (kernel rows on _par_devices(),
    the Jacobi eigensolver on the root), held to the plain fit's model
    arrays and transform."""
    from mojolearn.parallel_classical import fit_kernel_method, apply_kernel_method
    dev = _par_devices()
    kw = dict(kernel="rbf", gamma=0.5, n_components=32, random_state=7)
    par = fit_kernel_method(ml.Nystroem(**kw), X[:600, :4], devices=dev)
    plain = ml.Nystroem(**kw).fit(X[:600, :4])
    _same_bytes("fit_kernel_method components_", par.components_, "plain components_", plain.components_)
    _same_bytes("fit_kernel_method normalization_", par.normalization_, "plain normalization_", plain.normalization_)
    out = apply_kernel_method(par, X[600:1624, :4], devices=dev)
    _same_bytes("apply_kernel_method transform", out, "plain transform", plain.transform(X[600:1624, :4]))
    return _fit(dict(components=_h(par.components_), indices=_h(par.component_indices_),
                     normalization=_h(par.normalization_), eigenvalues=_h(par.eigenvalues_),
                     eigenvectors=_h(par.eigenvectors_), sweeps=_h(np.int64(par.sweeps_)), transform=_h(out)),
                par, lambda e: (apply_kernel_method(e, Xh[:64, :4], devices=_par_devices()),))


@lane("par-rbf-sampler")
def _(ml, X, yc, yr, Xh=None):
    """transform_rbf_sampler with 1024 random Fourier features over 1000
    rows in shards of 300 rows (each shard's output 1.17 MiB) on
    _par_devices(), held to the one-call transform."""
    from mojolearn.parallel_classical import transform_rbf_sampler
    m = ml.RBFSampler(gamma=0.5, n_components=1024, random_state=1).fit(X)
    out = transform_rbf_sampler(m, X[:1000], devices=_par_devices(), rows_per_shard=300)
    _same_bytes("transform_rbf_sampler", out, "plain transform", m.transform(X[:1000]))
    return _fit(dict(weights=_h(m.random_weights_), offset=_h(m.random_offset_), transform=_h(out)),
                m, lambda e: (transform_rbf_sampler(e, Xh[:256], devices=_par_devices(), rows_per_shard=100),))


# ---------------------------------------------------------------- lanes (2026-09-16, lane/multigpu-audit)
# The ELEVEN public estimators a multi-GPU driver ALREADY ADMITS BY TYPE and
# that no par-* lane ever asked. Read from the drivers' admission checks, not
# from prose: `fit_forest` takes all four forest classes but only the RF
# classifier and the ET regressor had lanes; `fit_boosting` takes the two
# sklearn-style adapters beside GradientBoosting; `fit_gram_estimator` takes
# LinearRegression, PCA and TruncatedSVD beside Ridge; `fit_coordinate_descent`
# takes ElasticNet beside Lasso; `fit_svm`/`predict_svm` take SVR beside SVC;
# `fit_scaler` takes MinMaxScaler beside StandardScaler; and `ParallelQueries`
# takes NearestNeighbors beside the three estimators that had lanes. So the
# multi-GPU PATH existed and the identity record simply did not carry it.
# Nothing here is new capability: no driver, no binding flag, no Mojo.
#
# These eleven are GPU-ONLY ON PURPOSE. They are NOT added to any family's
# `training_lanes` in python/mojolearn/host_surface.py, so they never enter
# `covered_lanes()` and therefore never enter `record_covered_lanes()`. That
# is deliberate: the day `TRAINING_GPU_COLUMNS` is repointed at a record taken
# under a narrower scope, the 11 CPU-covered par-* lanes that ARE in the
# covered set lose their GPU columns unless dropped or admitted, and these
# eleven cannot join that problem because they were never claimed. Their cells
# come from an explicit two-device `par` leg (`--lanes`), not from a release
# record, and until such a leg runs they are simply absent from every column.
# They are NOT eligible for `--owed-json`: `_owed_status` admits a part as
# OWED only when a CPU column hashes it STABLE, and these have no CPU column
# by construction, so a diff naming them reports them short, never OWED.
# Same rules as every par lane above: devices=_par_devices(), the smallest
# sharding that splits the work, and `_same_bytes` against the plain fit.

@lane("par-forest-reg")
def _(ml, X, yc, yr, Xh=None):
    """fit_forest on the rf-reg lane's RandomForestRegressor, 16 trees in four
    ranges of four, held to the plain fit. par-forest carries the classifier;
    the driver admits this class on the same global tree ID ranges."""
    from mojolearn.parallel_ensemble import fit_forest
    kw = dict(n_estimators=16, max_depth=8, random_state=7)
    par = fit_forest(ml.RandomForestRegressor(**kw), X, yr, devices=_par_devices(), trees_per_shard=4)
    plain = ml.RandomForestRegressor(**kw).fit(X, yr)
    _same_bytes("fit_forest predict", par.predict(X[:2048]), "plain predict", plain.predict(X[:2048]))
    return _fit(dict(predict=_h(par.predict(X))), par, lambda e: (e.predict(Xh),))


@lane("par-forest-et-clf")
def _(ml, X, yc, yr, Xh=None):
    """fit_forest on the et-clf lane's ExtraTreesClassifier. par-forest-et
    carries the regressor; this is the fourth forest class the driver takes."""
    from mojolearn.parallel_ensemble import fit_forest
    kw = dict(n_estimators=16, max_depth=8, random_state=7)
    par = fit_forest(ml.ExtraTreesClassifier(**kw), X, yc, devices=_par_devices(), trees_per_shard=4)
    plain = ml.ExtraTreesClassifier(**kw).fit(X, yc)
    _same_bytes("fit_forest predict_proba", par.predict_proba(X[:2048]),
                "plain predict_proba", plain.predict_proba(X[:2048]))
    return _fit(dict(predict=_h(par.predict(X)), proba=_h(par.predict_proba(X))),
                par, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("par-boosting-clf")
def _(ml, X, yc, yr, Xh=None):
    """fit_boosting on the gbdt-adapter-clf lane's GradientBoostingClassifier,
    the sklearn-style adapter `fit_boosting` admits beside GradientBoosting;
    the same packed feature groups par-boosting distributes."""
    from mojolearn.parallel_ensemble import fit_boosting
    kw = dict(n_estimators=20, max_depth=6)
    par = fit_boosting(_gbdt(ml.GradientBoostingClassifier, **kw), X, yc, devices=_par_devices())
    plain = _gbdt(ml.GradientBoostingClassifier, **kw).fit(X, yc)
    _same_bytes("fit_boosting predict_proba", par.predict_proba(X[:2048]),
                "plain predict_proba", plain.predict_proba(X[:2048]))
    return _fit(dict(predict=_h(par.predict(X)), proba=_h(par.predict_proba(X)),
                     decision=_h(par.decision_function(X[:512]))),
                par, lambda e: (e.predict(Xh), e.predict_proba(Xh), e.decision_function(Xh[:512])))


@lane("par-boosting-reg")
def _(ml, X, yc, yr, Xh=None):
    """fit_boosting on the gbdt-adapter-reg lane's GradientBoostingRegressor."""
    from mojolearn.parallel_ensemble import fit_boosting
    kw = dict(n_estimators=20, max_depth=6)
    par = fit_boosting(_gbdt(ml.GradientBoostingRegressor, **kw), X, yr, devices=_par_devices())
    plain = _gbdt(ml.GradientBoostingRegressor, **kw).fit(X, yr)
    _same_bytes("fit_boosting predict", par.predict(X[:2048]), "plain predict", plain.predict(X[:2048]))
    return _fit(dict(predict=_h(par.predict(X))), par, lambda e: (e.predict(Xh),))


@lane("par-gram-ols")
def _(ml, X, yc, yr, Xh=None):
    """fit_gram_estimator on the ols lane's LinearRegression: the pinned Gram
    chunks par-gram distributes for Ridge, on the minimum-norm OLS path."""
    from mojolearn.parallel_classical import fit_gram_estimator
    par = fit_gram_estimator(ml.LinearRegression(), X, yr, devices=_par_devices())
    plain = ml.LinearRegression().fit(X, yr)
    _same_bytes("fit_gram_estimator coef", par.coef_, "plain coef", plain.coef_)
    return _fit(dict(coef=_h(par.coef_), predict=_h(par.predict(X[:256]))),
                par, lambda e: (e.predict(Xh[:256]),))


@lane("par-gram-pca")
def _(ml, X, yc, yr, Xh=None):
    """fit_gram_estimator on the pca lane's PCA. The fixture is tall, so the
    covariance route runs; wide full PCA is refused by the driver by name."""
    from mojolearn.parallel_classical import fit_gram_estimator
    par = fit_gram_estimator(ml.PCA(n_components=4), X, devices=_par_devices())
    plain = ml.PCA(n_components=4).fit(X)
    _same_bytes("fit_gram_estimator components", par.components_, "plain components", plain.components_)
    return _fit(dict(components=_h(par.components_), variance=_h(par.explained_variance_),
                     transform=_h(par.transform(X[:256]))),
                par, lambda e: (e.transform(Xh[:256]),))


@lane("par-gram-tsvd")
def _(ml, X, yc, yr, Xh=None):
    """fit_gram_estimator on the tsvd lane's TruncatedSVD, the fourth class
    the Gram driver admits."""
    from mojolearn.parallel_classical import fit_gram_estimator
    par = fit_gram_estimator(ml.TruncatedSVD(n_components=4), X, devices=_par_devices())
    plain = ml.TruncatedSVD(n_components=4).fit(X)
    _same_bytes("fit_gram_estimator components", par.components_, "plain components", plain.components_)
    return _fit(dict(components=_h(par.components_), transform=_h(par.transform(X[:256]))),
                par, lambda e: (e.transform(Xh[:256]),))


@lane("par-cd-elasticnet")
def _(ml, X, yc, yr, Xh=None):
    """fit_coordinate_descent on the elasticnet lane's ElasticNet, the second
    class the dot-leaf driver admits beside Lasso."""
    from mojolearn.parallel_classical import fit_coordinate_descent
    kw = dict(alpha=0.01, l1_ratio=0.5, max_iter=200)
    par = fit_coordinate_descent(ml.ElasticNet(**kw), X, yr, devices=_par_devices())
    plain = ml.ElasticNet(**kw).fit(X, yr)
    _same_bytes("fit_coordinate_descent coef", par.coef_, "plain coef", plain.coef_)
    return _fit(dict(coef=_h(par.coef_), predict=_h(par.predict(X[:256]))),
                par, lambda e: (e.predict(Xh[:256]),))


@lane("par-svm-svr")
def _(ml, X, yc, yr, Xh=None):
    """fit_svm and predict_svm on the svr lane's SVR, the second class the
    kernel-row driver admits. predict_svm refuses decision_function for SVR
    by name, so only predict is asked."""
    from mojolearn.parallel_classical import fit_svm, predict_svm
    kw = dict(C=1.0, kernel="rbf", epsilon=0.1, max_iter=200)
    par = fit_svm(ml.SVR(**kw), X[:2000], yr[:2000], devices=_par_devices())
    plain = ml.SVR(**kw).fit(X[:2000], yr[:2000])
    pred = predict_svm(par, X[2000:2256], devices=_par_devices(), method="predict")
    _same_bytes("predict_svm", pred, "plain predict", plain.predict(X[2000:2256]))
    return _fit(dict(predict=_h(pred)), par, lambda e: (e.predict(Xh[:256]),))


@lane("par-scaler-minmax")
def _(ml, X, yc, yr, Xh=None):
    """fit_scaler and transform_scaler on the minmax-scaler lane's
    MinMaxScaler with four columns per shard, the second class the column
    driver admits. Its five fitted statistics are the plain lane's.

    `NumericalMismatch` rather than a bare raise, for `par-scaler`'s reason
    above; the clean cell is unchanged."""
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    par = fit_scaler(ml.MinMaxScaler(), X, devices=_par_devices(), columns_per_shard=4)
    plain = ml.MinMaxScaler().fit(X)
    t = transform_scaler(par, X[:256], devices=_par_devices(), columns_per_shard=4)
    mismatch = _mismatch_bytes("transform_scaler", t, "plain transform", plain.transform(X[:256]))
    parts = dict(data_min=_h(par.data_min_), data_max=_h(par.data_max_), scale=_h(par.scale_),
                 min=_h(par.min_), transform=_h(t),
                 inverse=_h(transform_scaler(par, t, devices=_par_devices(), columns_per_shard=4,
                                             inverse=True)))
    if mismatch:
        raise NumericalMismatch(mismatch, parts)
    return _fit(parts, par, lambda e: (e.transform(Xh[:256]),))


@lane("par-queries-nn")
def _(ml, X, yc, yr, Xh=None):
    """ParallelQueries over the knn lane's NearestNeighbors, 64 query rows in
    shards of 16, held to the plain kneighbors. par-queries-knn carries the
    classifier; this is the bare index the driver also admits."""
    from mojolearn.parallel_neighbors import ParallelQueries
    m = ml.NearestNeighbors(n_neighbors=8).fit(X[:4096])
    q = np.ascontiguousarray(X[4096:4160])
    with ParallelQueries(m, devices=_par_devices(), rows_per_shard=PAR_QUERY_ROWS) as pq:
        d, i = pq.query(q, method="kneighbors")
    d0, i0 = m.kneighbors(q)
    _same_bytes("ParallelQueries kneighbors distances", d, "plain distances", d0)
    _same_bytes("ParallelQueries kneighbors indices", i, "plain indices", i0)
    return _fit(dict(dist=_h(d), idx=_h(i)), m, lambda e: _pq(e, Xh[:64], "kneighbors"))


# ---------------------------------------------------------------- lanes (2026-09-19, lane/laneless-public-classes)
# THE LAST PUBLIC CLASS WITH NO LANE. A census of `mojolearn.__all__` against
# this file on 2026-09-19 returned three names it never mentions:
# `DistributedIVFIndex`, `LanguageModelConfig` and `GPT2Tokenizer`. Two of
# the three are ALIASES that `tools/verification_matrix.py::package_index`
# already resolves -- `LanguageModelConfig = ByteLanguageModelConfig` in
# language_model.py and `GPT2Tokenizer -> BpeTokenizer` in
# tokenizer.py::_DEPRECATED_ALIASES -- so the lanes of the classes they name
# are their lanes, and what was missing for them was a check that the ALIAS
# still resolves and still warns, which is a test and not arithmetic
# (python/mojolearn/tests/test_public_aliases.py). `DistributedIVFIndex` was
# the real gap: a class of its own, in `__all__`, with no cell anywhere.
#
# ITS CPU ROUTE HAD TO BE BUILT FIRST, and that is the load-bearing part of
# this lane. The driver's workers call `ivf_flat_partial_search` and
# `ivf_finalize_distances`, two names the ivf HOST bindings did not register
# and `_parallel_pool.CPU_OPERATIONS` did not admit, so before this lane the
# class could not run on a CPU at all and a lane for it would have read
# REFUSED on every CPU column. `ivf/host/ivf_host.mojo::host_ivf_search`
# now carries the device search's `partial_storage` arms and both host
# bindings register both names; see those files for what each arm is.

def _refused(call, needle):
    """1 when `call()` raises with `needle` in the message, 0 when it raises
    something else, 0 when it does not raise. A REFUSAL HASHED AS A NUMBER,
    never asserted: a lane that raises reads REFUSED, and a REFUSED cell and a
    passing cell are the same thing in a column total (the `tokenizer` lane's
    `roundtrip` part, 2026-09-15). Recorded this way a lifted refusal reads
    DIVERGENT against the reference, which is a catch."""
    try:
        call()
    except Exception as exc:
        return 1 if needle in str(exc) else 0
    return 0


def _dist_search(cls, index, queries, devices):
    """One `DistributedIVFIndex` search over `index`, the pool opened and
    closed around it (the `_pq` helper's shape). Returns the distances, the
    global ids and the merged per-query candidate counts."""
    with cls.from_index(index, devices=devices) as shards:
        d, i = shards.search(queries)
        return d, i, np.asarray(shards.n_candidates_).copy()


@lane("par-ivf")
def _(ml, X, yc, yr, Xh=None):
    """DistributedIVFIndex (python/mojolearn/parallel_ivf.py): a BUILT
    IVF-Flat index partitioned into disjoint stored shards with the coarse
    quantizer REPLICATED, searched shard by shard and merged back to global
    row ids. The `ivf` lane's index exactly (16 lists, 4 probes, k=8, 4096
    rows, random_state 3) and its 64 queries, so a reader can hold this
    lane's cells against that one.

    WHAT THE CELL SAYS. The driver's whole contract is equality to the plain
    search, so the squared-metric arm is held to `IVFIndex.search` BYTE FOR
    BYTE and the train cell reads REFUSED naming the pair if it is not: the
    local-id maps, the per-shard candidate counts and the global merge by
    `(distance, original id)` either reproduce the one-index answer or they
    do not. `cand` is the merged candidate total per query, which is what
    says the shards were probed at all rather than a k-sized answer arriving
    from one of them.

    THE EUCLIDEAN ARM IS A FLAG, NOT A RAISE, and the reason is measured.
    Under `metric='euclidean'` the shards return SQUARED distances and the
    root is taken once, afterwards, by `ivf_finalize_distances` on the
    merged row -- that seam is the one thing the squared arm cannot reach.
    Its agreement with the plain search is hashed as a flag rather than
    asserted because the sabotage build's value flip is applied BEFORE that
    root on this route and AFTER it on the plain one, so a raise would make
    the sabotaged cell read REFUSED, which the owed check
    (tools/cpu_identity_gate_check.py owed) does not count as a catch -- the
    `tokenizer` lane's `roundtrip` part carries the same reasoning. MEASURED
    on the M4, one core, --repeats 2, base fixture, 2026-09-19: clean
    `root_dist` cdbf46f042cf7c81 and `flags` b1e0e3b7b8fee07b against
    6381524ce593f295 and 9ce81bd2eece53a4 under the
    -D MOJOLEARN_HOST_SABOTAGE=1 ivf binding, so the arm moves TWICE over --
    the distances and the flag that says they still agree.

    WHAT THE SABOTAGE DOES NOT MOVE, said rather than left to be assumed.
    The same pair leaves `idx`, `cand`, `root_idx` (on `base`) and
    `thin_idx` where it found them, because that build's value arm steps a
    float32 mantissa by ONE UNIT and is therefore monotone on the positive
    distances, so nothing is reordered; on the `ties` fixture, where
    distances collide, `root_idx` does move (433afb8e970006c6 ->
    d85c1e527d64badd). Every distance part moves on both fixtures, and so
    does the cell (base 82ac27da71d34ea9 -> 1a0e8a596286d0c3, infer
    fd6eed35b1338d17 -> c70054c87bc14b51).

    THE DEVICE AXIS IS `_par_devices()`, as every par-* lane's is, and at one
    device the partition is ONE shard: the merge still runs, the id map is
    still applied and the equality above still fires, but no shard can be
    short of candidates. The arms that only a multi-shard column reaches --
    `min(k, n_cand)` selection and the all-zero padding of a shard that owns
    NO candidate for a query -- are what the `thin_*` parts are for: 128
    rows over 16 lists probed twice, k=6, so a shard holds few candidates or
    none. At MOJOLEARN_PAR_DEVICES=0,1,2,3,4,5,6,7 on the M4, 54 of 64 shard
    answers were short and 42 were EMPTY, and the merged answer was still
    byte-equal to the plain search (2026-09-19,
    lane/laneless-public-classes). Those three parts read the same at one
    shard and at eight, as every part here does; that equality, cell for
    cell across shard counts, is the driver's claim and the reason this lane
    is worth running on a column with no device at all.

    THE REFUSALS ARE PART OF THE CLASS. `from_index` is the only door (the
    constructor refuses by name), it refuses an index that is not ours and
    one that was never fitted, a closed index refuses to search, and a query
    of the wrong width refuses. None of them is arithmetic and none moves
    under a sabotage build; they are in `flags` because a driver that stops
    refusing has changed its contract and nothing else here would see it."""
    from mojolearn.parallel_ivf import DistributedIVFIndex
    dev = _par_devices()
    kw = dict(n_lists=16, n_probes=4, n_neighbors=8, random_state=3)
    q = np.ascontiguousarray(X[4096:4160])
    m = ml.IVFIndex(**kw).fit(X[:4096])
    d0, i0 = m.search(q)
    with DistributedIVFIndex.from_index(m, devices=dev) as shards:
        d, i = shards.search(q)
        cand = np.asarray(shards.n_candidates_).copy()
        closed = shards
    _same_bytes("DistributedIVFIndex distances", d, "plain IVFIndex distances", d0)
    _same_bytes("DistributedIVFIndex ids", i, "plain IVFIndex ids", i0)
    e = ml.IVFIndex(metric="euclidean", **kw).fit(X[:4096])
    ed0, ei0 = e.search(q)
    with DistributedIVFIndex.from_index(e, devices=dev) as rooted:
        ed, ei = rooted.search(q)
    # THE SHORT-FILL ARM (see the docstring): 128 rows over 16 lists probed
    # twice, so a shard of this index owns few candidates or none for a
    # query. Its answer does not depend on the shard count, so the cell is
    # the same at one device and at eight; what changes is which arms of
    # `host_ivf_search` the column walks through to produce it.
    thin = ml.IVFIndex(n_lists=16, n_probes=2, n_neighbors=6, random_state=5).fit(X[:128])
    tq = np.ascontiguousarray(X[4096:4104])
    td0, ti0 = thin.search(tq)
    td, ti, tcand = _dist_search(DistributedIVFIndex, thin, tq, dev)
    _same_bytes("DistributedIVFIndex short-fill distances", td, "plain IVFIndex distances", td0)
    _same_bytes("DistributedIVFIndex short-fill ids", ti, "plain IVFIndex ids", ti0)
    flags = np.asarray([
        np.asarray(ed).tobytes() == np.asarray(ed0).tobytes(),
        np.asarray(ei).tobytes() == np.asarray(ei0).tobytes(),
        _refused(lambda: DistributedIVFIndex(), "from_index"),
        _refused(lambda: DistributedIVFIndex.from_index(ml.IVFIndex(**kw), devices=dev), "fit or load"),
        _refused(lambda: DistributedIVFIndex.from_index(m.list_data_, devices=dev), "mojolearn.IVFIndex"),
        _refused(lambda: closed.search(q), "is closed"),
        _refused(lambda: _dist_search(DistributedIVFIndex, m, np.ascontiguousarray(q[:, :2]), dev), "query features"),
    ], dtype=np.int64)
    return _fit(dict(dist=_h(d), idx=_h(i), cand=_h(cand), root_dist=_h(ed), root_idx=_h(ei),
                     thin_dist=_h(td), thin_idx=_h(ti), thin_cand=_h(tcand), flags=_h(flags)),
                m, lambda est: _dist_search(DistributedIVFIndex, est, np.ascontiguousarray(Xh[:64]), _par_devices()))


# ------------------------------------------------- the six unlaned public algorithms
# lane/unlaned-public-algorithms (2026-09-20). `tools/verification_matrix.py
# --json` reported SIX public entries with no identity lane at all: the two
# resident decode sessions, `models.ParallelCausalLM`, the two
# `parallel_gaussian_process` classifier drivers and
# `parallel_model_selection.cross_val_score`. An algorithm with no lane cannot
# be missing a cell, so a lane census hides it entirely; these six now have
# names, parts and a written-down owed column.
#
# THEY DO NOT ALL COST THE SAME. Measured on this branch, on the M4's CPU
# host route, before a line of lane body was written:
#
#   `fit_gaussian_process_classifier` / `predict_gaussian_process_classifier`
#       refused with `no CPU implementation of the parallel worker operation
#       gpc_class_fit yet` -- a CPU_OPERATIONS gap and nothing more. The
#       class-level partition is the driver's own Python and each shard is the
#       gp family's host `gpc_fit`/`gpc_predict`, which the covered `gpc` and
#       `gpc-multiclass` lanes already hash, so the two operations were
#       admitted to `_parallel_pool.CPU_OPERATIONS` against that docstring's
#       bar (the reasoning is there, not here). Both lanes take a CPU column.
#
#   `Mamba1DecodeSession` / `TransformerDecodeSession` refuse BY NAME on the
#       host route: `mamba1_session_create` and
#       `transformer_decode_session_create` exist only in
#       `bindings/_mojolearn_mamba.mojo` and
#       `bindings/_mojolearn_transformer.mojo`. There is no CPU arithmetic to
#       hash, so these two lanes are in `GPU_ONLY_LANES` and a CPU-only
#       install drops them BY NAME instead of recording nine REFUSED cells,
#       which in a column total reads exactly like coverage.
#
#   `ParallelCausalLM` and `parallel_model_selection.cross_val_score` refuse
#       on anything that is not CUDA or HIP -- not only on the CPU column but
#       on Apple's too (`_backend.vendor() not in ('cuda', 'hip')`, raised
#       before either one touches a fold or a layer). They are `par-*` lanes
#       by nature AND `GPU_ONLY_LANES` by route: their cells exist only on a
#       two-device NVIDIA or AMD column.
#
# WHAT THE CPU CELLS PROVE AND WHAT THEY DO NOT. The two GPC lanes hold the
# sharded driver to the plain fit byte for byte inside the cell, so they can
# fail on this box with no record at all. That is AGREEMENT, not rightness: if
# `gpc_fit`'s Newton loop were wrong, the driver and the plain fit would be
# wrong together and every column would say IDENTICAL. The device axis is not
# tested here either; at one device the partition is one worker process per
# class, which is what `_par_devices`'s docstring says of every `par-*` lane.
#
# FOUR OF THE SIX HAVE NO SABOTAGE ARM ON THIS PASS, AND NOT BECAUSE ONE WAS
# SKIPPED. A negative control is a BUILD that computes a wrong answer through
# the same door the lane opens; the four lanes in `GPU_ONLY_LANES` open no
# door on a CPU-only install at all, so there is nothing for a host define to
# be wrong inside. Their arms are the ordinary ones of the families that serve
# them on a GPU column -- the mamba and transformer bindings for the two decode
# sessions, the neural set for `par-causal-lm`, the gbdt set for
# `par-cross-val`'s folds -- and each is owed with the column, not before it.
# This is written down rather than left blank because a lane with no arm and
# no reason is indistinguishable from a lane whose arm was never run.

#: Lanes whose PUBLIC SURFACE refuses by name on a CPU-only install, so a CPU
#: column has no arithmetic of theirs to run (lane/unlaned-public-algorithms,
#: 2026-09-20). A full-column run on a CPU-only install drops them by name and
#: says so; `--lanes` names lanes explicitly and is never filtered, and a GPU
#: column runs them exactly as it runs every other lane, so no record loses a
#: cell. This is NOT `RECORD_EXCLUDED_PREFIXES`, which is about a release
#: record's scope on every column; this is one column's inability to state the
#: proposition at all, the same fact `lane_applicability` derives for cpu-host.
GPU_ONLY_LANES = {
    "mamba1-decode-session":
        "Mamba1DecodeSession: mamba1_session_create exists only in "
        "bindings/_mojolearn_mamba.mojo, and the constructor refuses by name on the host route",
    "transformer-decode-session":
        "TransformerDecodeSession: transformer_decode_session_create exists only in "
        "bindings/_mojolearn_transformer.mojo, and the constructor refuses by name on the host route",
    "par-causal-lm":
        "models.ParallelCausalLM raises NotImplementedError unless _backend.vendor() is cuda or hip, "
        "before it builds a single layer",
    "par-cross-val":
        "parallel_model_selection.cross_val_score raises NotImplementedError unless "
        "_backend.vendor() is cuda or hip, before it prepares a single fold",
}


def _decode_session_probe(blk, state_fn, rows, length, dm, Xh):
    """A resident decode session's outputs against the PER-CALL step, which
    is the session's whole documented contract ("Every output is BYTE FOR
    BYTE the per-call `step` on the same block and state").

    Returns `(parts, session_outputs)`. The session is opened on one fresh
    state and stepped `length` times; a second fresh state is stepped the
    same `length` times through `blk.step(x, state)` with no session at all,
    and the two are held to each other IN THE CELL. `sync_state` is then
    asked for the resident state and held to the per-call state's buffers,
    because a session whose outputs agree but whose state does not is a
    session that cannot be handed back."""
    x = _seq(Xh, rows, length, dm)
    session_out, per_call_out = [], []
    st = state_fn()
    with blk.decode_session(st) as sess:
        for t in range(length):
            session_out.append(np.ascontiguousarray(np.asarray(sess.step(np.ascontiguousarray(x[:, t:t + 1])))))
        sess.sync_state()
        resident = _decode_state_arrays(st)
    plain = state_fn()
    for t in range(length):
        per_call_out.append(np.ascontiguousarray(np.asarray(blk.step(np.ascontiguousarray(x[:, t:t + 1]), plain))))
    _same_bytes("decode_session step", np.concatenate(session_out, axis=1),
                "per-call block.step", np.concatenate(per_call_out, axis=1))
    _same_bytes("decode_session sync_state", np.concatenate(resident),
                "per-call state", np.concatenate(_decode_state_arrays(plain)))
    return session_out, resident


def _decode_state_arrays(state):
    """Every float buffer a decode state carries, in a fixed name order, so
    two states are comparable byte for byte."""
    out = []
    for name in ("conv_window", "h", "k_cache", "v_cache"):
        buf = getattr(state, name, None)
        if buf is not None:
            out.append(np.ascontiguousarray(np.asarray(buf)).reshape(-1))
    return out


@lane("mamba1-decode-session")
def _(ml, X, yc, yr, Xh=None):
    """`Mamba1Block.decode_session(state)` -> Mamba1DecodeSession
    (DEVIATION 2941), the RESIDENT decode path: one device context and one
    set of device structs built once and stepped, instead of per token.

    THE PARTS. `step` is the 16 decoded positions of two rows, the same
    weights and the same `_seq` slab the `mamba1` lane's held-out probe uses,
    so a reader can hold this lane's cells against that one. `state` is the
    resident `conv_window` and `h` after `sync_state()`. `flags` is the
    session's refusals, which are part of its contract and which no sabotage
    build can move: a closed session refuses to step, a state that already
    belongs to a session refuses to open a second one, and a row count that
    does not match the state's refuses by name.

    THE ORACLE IS IN THE CELL. Both hashed parts are held BYTE FOR BYTE to
    the per-call `block.step` on a second fresh state, which is the session
    docstring's own claim, so this cell can fail on one box with no record.

    `batch` IS NOT A BATCH AXIS HERE, and this is the part of the declaration
    that needed thinking about rather than defaulting. The batch part asks
    "does a row's answer depend on which other rows were in the call"; a
    decode session is opened FOR a fixed batch size (`state.batch_size`, the
    resident device buffers are allocated for it) and `step` refuses a row
    count that is not the session's by name. Asking a batch of one of a
    session opened for two is not a smaller batch of the same call, it is a
    different session, and the answer it gives belongs to `mamba1`'s batch
    part, which already asks the per-call `step` the session is held to.
    `batchscale` is the same fact at serving scale. Declared n/a, with that
    reason, in the declarations below.

    NO CPU COLUMN EXISTS, and none can be faked (GPU_ONLY_LANES above).
    OWED, to be taken in the coordinated three-column record:
      MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \\
        --lanes mamba1-decode-session --repeats 2 --step-full --batch-scale --ragged --batch-grad \\
        --json <column>.json"""
    dm, di, r = 32, 64, 2
    w = _block_weights("mamba1", {
        "norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4),
        "conv1d.bias": (di,), "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
        "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (dm, di)},
        ones=("norm.weight",))
    blk = ml.Mamba1Block(w)
    from mojolearn.mamba import Mamba1DecodeSession
    steps, resident = _decode_session_probe(blk, lambda: blk.allocate_state(2), 2, 16, dm, X)
    held = blk.allocate_state(2)
    sess = blk.decode_session(held)
    sess.close()
    busy = blk.allocate_state(2)
    open_session = blk.decode_session(busy)
    flags = np.asarray([
        # THE DOOR RETURNS THE DOCUMENTED CLASS. `decode_session` is the only
        # public door onto it, so the name is checked here rather than assumed;
        # it is also what attributes this lane to `mamba.Mamba1DecodeSession`
        # in tools/verification_matrix.py, which reads the lane body's
        # references and not a hand-kept map.
        type(open_session) is Mamba1DecodeSession,
        int(open_session.is_open) and not sess.is_open,
        _refused(lambda: sess.step(np.zeros((2, 1, dm), dtype=np.float32)), "session is closed"),
        _refused(lambda: blk.decode_session(busy), "resident"),
        _refused(lambda: open_session.step(np.zeros((3, 1, dm), dtype=np.float32)), "rows"),
        _refused(lambda: blk.decode_session(None), "state is required"),
    ], dtype=np.int64)
    open_session.close()
    return _fit(dict(step=_h(*steps), state=_h(*resident), flags=_h(flags)),
                blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


@lane("transformer-decode-session")
def _(ml, X, yc, yr, Xh=None):
    """`TransformerBlock.decode_session(state)` -> TransformerDecodeSession
    (DEVIATION 2940), the RESIDENT decode path over one KV cache.

    THE PARTS. `step` is the 16 decoded positions of two rows on the
    `transformer` lane's weights and slab; `prefill` is the session's
    `forward(x)`, the chunked continuation on the resident cache, which the
    Mamba-1 session has no counterpart for; `state` is the resident k and v
    caches after `sync_state()`; `flags` is the refusals (a closed session, a
    state already owned by a session, a window that does not match the
    block's, a missing state). Both arithmetic parts are held BYTE FOR BYTE
    in the cell to the per-call `block.step` and `block.forward` on a second
    fresh state, which is the session docstring's own claim.

    `batch` and `batchscale` are n/a for the Mamba-1 session's reason, which
    is written out on that lane: the session is opened for a fixed
    `state.batch_size` and refuses any other row count by name, so there is
    no batch axis to vary without opening a different session.

    NO CPU COLUMN EXISTS (GPU_ONLY_LANES above). OWED:
      MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \\
        --lanes transformer-decode-session --repeats 2 --step-full --batch-scale --ragged --batch-grad \\
        --json <column>.json"""
    dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
    w = _block_weights("transformer", {
        "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
        "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm), "v_proj.weight": (nkv * hd, dm),
        "o_proj.weight": (dm, nh * hd), "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
        "down_proj.weight": (dm, it)},
        near_one=("input_layernorm.weight", "post_attention_layernorm.weight"))
    blk = ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv)

    from mojolearn.transformer import TransformerDecodeSession

    def fresh():
        return blk.allocate_state(2, max_tokens=32)

    steps, resident = _decode_session_probe(blk, fresh, 2, 16, dm, X)
    # THE PREFILL HALF, which only the transformer session has: L tokens on
    # the resident cache in one call, held to the per-call forward on a
    # second fresh state.
    chunk = _seq(X, 2, 16, dm, skip=1024)
    st = fresh()
    with blk.decode_session(st) as sess:
        prefill = np.ascontiguousarray(np.asarray(sess.forward(chunk)))
    plain = fresh()
    _same_bytes("decode_session forward", prefill,
                "per-call block.forward", np.asarray(blk.forward(chunk, plain)))
    held = fresh()
    sess = blk.decode_session(held)
    sess.close()
    busy = fresh()
    open_session = blk.decode_session(busy)
    flags = np.asarray([
        # the door returns the documented class; see the mamba1 lane
        type(open_session) is TransformerDecodeSession,
        int(open_session.is_open) and not sess.is_open,
        _refused(lambda: sess.step(np.zeros((2, 1, dm), dtype=np.float32)), "session is closed"),
        _refused(lambda: blk.decode_session(busy), "resident"),
        _refused(lambda: blk.decode_session(None), "state is required"),
        _refused(lambda: blk.decode_session(blk.allocate_state(2, max_tokens=32, window=8)), "window"),
    ], dtype=np.int64)
    open_session.close()
    return _fit(dict(step=_h(*steps), prefill=_h(prefill), state=_h(*resident), flags=_h(flags)),
                blk, lambda e: (np.asarray(_neural_inference(ml, "transformer", e).forward(_seq(Xh, 2, 16, dm))),))


def _par_gpc_kernel(ml):
    """The `gpc` lane's kernel, so this driver's cells are readable against
    that lane's."""
    return ml.ConstantKernel(1.0) * ml.RBF(1.0)


@lane("par-gpc-fit")
def _(ml, X, yc, yr, Xh=None):
    """`parallel_gaussian_process.fit_gaussian_process_classifier`: the
    GaussianProcessClassifier's ONE-VS-REST class problems scheduled one to a
    worker, held to the plain `GaussianProcessClassifier.fit`.

    THE SHAPE IS THE `gpc` AND `gpc-multiclass` LANES', deliberately: 256
    rows of four columns, the same kernel, and both arms in one cell -- the
    BINARY fit, whose `columns = [1]` means the driver builds exactly ONE
    task and the merge is trivial, and the THREE-CLASS fit over
    `_gpc_three_classes`, where the driver builds three tasks and
    `_set_fitted` merges them in class order. A cell with only the binary arm
    could not see a class-order defect at all.

    THE ORACLE IS IN THE CELL. `_same_bytes` holds the driver's per-class
    Cholesky factor and the three-class posterior mean to the plain fit's, so
    this cell fails on one box with no record. What it does NOT say is that
    the Laplace fit is RIGHT: a wrong Newton loop is wrong identically on both
    sides of that comparison and on every column.

    THE DEVICE AXIS IS `_par_devices()`, as every `par-*` lane's is. At one
    device the partition is one worker PROCESS per class, which exercises the
    Python split, the pickling of each class's target vector and the ordered
    merge, and says nothing about two devices. The two-device column is owed.

    `parts`: `L` and `pi` are the per-class Laplace state the driver
    published, `lml` the log marginal likelihoods, `three_L` and `three_lml`
    the three-class arm, `flags` the driver's refusals (a foreign estimator,
    a FAST numeric mode, a label vector of the wrong length, a single-class
    target). None of the refusals is arithmetic; they are hashed because a
    driver that stops refusing has changed its contract and nothing else here
    would see it.

    SABOTAGE, SEEN TO MOVE. The arm is the gp family's own define,
    `-D MOJOLEARN_HOST_SABOTAGE=1` on `bindings/_mojolearn_gp_host.mojo`,
    against a CLEAN build of the SAME source (a .so digest does not prove a
    define; `gp_host_sabotage()` reads False on one and the loader refuses the
    other BY NAME). MEASURED on the M4, one core, `nice -n 19`, `--repeats 2`,
    all nine fixtures, 2026-09-20
    (bench/results/identity_break/2026-09-20_unlaned-public-algorithms/):
    every cell, infer, model and batch hash of BOTH GPC lanes moved, 18 of 18
    cells and 36 of 36 infer/model parts DIVERGENT. `base`
    53f3a09fc600d61d -> 59b4a6c2ea6454a4 here, `L`
    cf8752e79a59c592 -> 07c415bd8bfe2ff3, `pi`
    3e4e62ab648dd40a -> c2d8c272fe84b3a4.

    WHAT THE ARM DOES NOT MOVE, said rather than left to be assumed. `flags`
    never moves on any fixture (d529dd94544bb118 both sides, 9 of 9): a
    refusal is a type check and a length check, not arithmetic, and no build
    define can reach it -- which is the whole reason it is hashed as a number.
    `lml` holds on `hashed`, `odd` and `ties` while `L` and `pi` move on all
    nine; a lane whose only float part were the log marginal likelihood would
    have read clean on a third of this column.

    CPU COLUMN: yes, since `gpc_class_fit` joined
    `_parallel_pool.CPU_OPERATIONS` on this branch. OWED GPU COLUMNS:
      MOJOLEARN_NUMERIC_MODE=identical python3 tools/identity_break.py \\
        --lanes par-gpc-fit,par-gpc-predict --repeats 2 --step-full --batch-scale --ragged --batch-grad \\
        --json <column>.json
    and the two-device leg that is the actual claim:
      MOJOLEARN_PAR_DEVICES=0,1 MOJOLEARN_NUMERIC_MODE=identical python3 \\
        tools/identity_break.py --lanes par-gpc-fit,par-gpc-predict \\
        --repeats 2 --step-full --batch-scale --ragged --batch-grad --json <column>.2gpu.json"""
    from mojolearn.parallel_gaussian_process import fit_gaussian_process_classifier
    dev = _par_devices()
    k = _par_gpc_kernel(ml)
    xs, ys = X[:256, :4], yc[:256]
    par = fit_gaussian_process_classifier(ml.GaussianProcessClassifier(kernel=k), xs, ys, devices=dev)
    plain = ml.GaussianProcessClassifier(kernel=k).fit(xs, ys)
    _same_bytes("fit_gaussian_process_classifier L", par.estimators_[0].L_,
                "plain L", plain.estimators_[0].L_)
    k3 = ml.ConstantKernel(2.0) * ml.Matern(1.0, nu=1.5)
    y3 = _gpc_three_classes(X)
    par3 = fit_gaussian_process_classifier(ml.GaussianProcessClassifier(kernel=k3), xs, y3, devices=dev)
    plain3 = ml.GaussianProcessClassifier(kernel=k3).fit(xs, y3)
    _same_bytes("fit_gaussian_process_classifier three-class proba", par3.predict_proba(X[256:320, :4]),
                "plain three-class proba", plain3.predict_proba(X[256:320, :4]))
    flags = np.asarray([
        _refused(lambda: fit_gaussian_process_classifier(ml.GaussianProcessRegressor(kernel=k), xs, ys, devices=dev),
                 "requires mojolearn.GaussianProcessClassifier"),
        _refused(lambda: fit_gaussian_process_classifier(
            ml.GaussianProcessClassifier(kernel=k, numeric_mode="fast"), xs, ys, devices=dev), "IDENTICAL"),
        _refused(lambda: fit_gaussian_process_classifier(
            ml.GaussianProcessClassifier(kernel=k), xs, ys[:128], devices=dev), "y length"),
        _refused(lambda: fit_gaussian_process_classifier(
            ml.GaussianProcessClassifier(kernel=k), xs, np.zeros(256, dtype=np.int64), devices=dev),
            "at least two classes"),
    ], dtype=np.int64)
    return _fit(dict(L=_h(*[e.L_ for e in par.estimators_]),
                     pi=_h(*[e.pi_ for e in par.estimators_]),
                     lml=_h(np.array([e.log_marginal_likelihood_value_ for e in par.estimators_],
                                     dtype=np.float64)),
                     three_L=_h(*[e.L_ for e in par3.estimators_]),
                     three_lml=_h(np.array([e.log_marginal_likelihood_value_ for e in par3.estimators_],
                                           dtype=np.float64)),
                     flags=_h(flags)),
                par, lambda e: (e.predict(Xh[:64, :4]), e.predict_proba(Xh[:64, :4])))


@lane("par-gpc-predict")
def _(ml, X, yc, yr, Xh=None):
    """`parallel_gaussian_process.predict_gaussian_process_classifier`: the
    PREDICTION driver, which `par-gpc-fit` does not reach (it schedules a
    fit), over a fit taken the ordinary way.

    Each worker is handed ONE class's covariance state plus the shared
    training and query matrices, and the merge -- the binary `[1 - v, v]`
    pair, the multiclass row-wise normalization and the arg-max that picks a
    code -- is the driver's own Python in
    `predict_gaussian_process_classifier` itself. Both `method` arms are
    asked, on the binary and the three-class fit, and each is held BYTE FOR
    BYTE in the cell to the plain estimator's `predict`/`predict_proba`,
    which is the driver's whole documented contract.

    `predict` returns the DECODED labels, so it is hashed as the integer
    codes `classes_.index` gives back rather than as an object array (`_h`
    refuses one, and would be hashing addresses if it did not).

    SABOTAGE, SEEN TO MOVE, the same gp-family pair `par-gpc-fit` describes.
    MEASURED, all nine fixtures, `--repeats 2`, 2026-09-20: `base`
    2f1217bcae26e6a9 -> 2e25f01e28255189, `proba`
    on every fixture, `three_proba` on every fixture, and the cell, infer,
    model and batch hashes on all nine.

    THE LABELS MOSTLY DO NOT MOVE, AND THAT IS WHY `proba` IS HASHED BESIDE
    THEM. `labels` reads the same bytes under the sabotage on 9 of 9 fixtures
    and `three_labels` on 8 of 9 (`wide` is the exception): the arm perturbs
    the latent mean far below the gap between the top two classes, and a
    `predict` is a sign test and an arg-max. A lane whose only part were the
    predicted labels would have read CLEAN under a build that computes the
    posterior wrongly everywhere. `flags` does not move either, for
    `par-gpc-fit`'s reason.

    Device axis, oracle and the limits of what agreement proves: as
    `par-gpc-fit`. CPU column: yes. OWED GPU columns: the commands on
    `par-gpc-fit`."""
    from mojolearn.parallel_gaussian_process import predict_gaussian_process_classifier
    dev = _par_devices()
    k = _par_gpc_kernel(ml)
    xs, q = X[:256, :4], X[256:320, :4]
    m = ml.GaussianProcessClassifier(kernel=k).fit(xs, yc[:256])
    proba = predict_gaussian_process_classifier(m, q, devices=dev, method="predict_proba")
    labels = predict_gaussian_process_classifier(m, q, devices=dev, method="predict")
    _same_bytes("predict_gaussian_process_classifier proba", proba, "plain predict_proba", m.predict_proba(q))
    _same_bytes("predict_gaussian_process_classifier predict", _label_codes(m, labels),
                "plain predict", _label_codes(m, m.predict(q)))
    k3 = ml.ConstantKernel(2.0) * ml.Matern(1.0, nu=1.5)
    m3 = ml.GaussianProcessClassifier(kernel=k3).fit(xs, _gpc_three_classes(X))
    proba3 = predict_gaussian_process_classifier(m3, q, devices=dev, method="predict_proba")
    labels3 = predict_gaussian_process_classifier(m3, q, devices=dev, method="predict")
    _same_bytes("predict_gaussian_process_classifier three-class proba", proba3,
                "plain three-class predict_proba", m3.predict_proba(q))
    flags = np.asarray([
        _refused(lambda: predict_gaussian_process_classifier(m, q, devices=dev, method="decision_function"),
                 "method must be predict or predict_proba"),
        _refused(lambda: predict_gaussian_process_classifier(
            ml.GaussianProcessRegressor(kernel=k), q, devices=dev),
            "requires mojolearn.GaussianProcessClassifier"),
        _refused(lambda: predict_gaussian_process_classifier(m, np.ascontiguousarray(q[:, :2]), devices=dev),
                 "features"),
    ], dtype=np.int64)
    return _fit(dict(proba=_h(proba), labels=_h(_label_codes(m, labels)),
                     three_proba=_h(proba3), three_labels=_h(_label_codes(m3, labels3)),
                     flags=_h(flags)),
                m, lambda e: (e.predict_proba(Xh[:64, :4]),))


def _label_codes(estimator, labels):
    """The integer position of each predicted label in `classes_`. `predict`
    hands back the caller's own label objects, and `_h` refuses an object
    array because its bytes are addresses."""
    order = {v: i for i, v in enumerate(list(estimator.classes_))}
    return np.asarray([order[v] for v in list(labels)], dtype=np.int64)


@lane("par-causal-lm")
def _(ml, X, yc, yr, Xh=None):
    """`models.ParallelCausalLM`: EXPLICIT LAYER OWNERSHIP, one device index
    per checkpoint layer, hidden activations crossing layer boundaries
    through host memory. Sequential model parallelism, not tensor
    parallelism and not a throughput claim; the class says so itself.

    THE CLAIM IS THE ORDINARY `CausalLM` PATH'S BITS. `_RemoteBlock` and
    `_RemotePrimitives` replace the in-process block and the embedding, norm
    and head calls with RPCs to the layer's owner, and the class docstring
    says "Output and state mathematics are the ordinary CausalLM path". So
    the cell holds the layer-owned model's logits to a plain `CausalLM` on
    the same weights, byte for byte, exactly as the other `par-*` drivers are
    held to their plain fit.

    THERE IS NO COLUMN ON THIS MACHINE, AND NOT ONLY BECAUSE IT IS A CPU.
    `__init__` and `load` both raise
    `NotImplementedError('ParallelCausalLM requires CUDA or HIP device
    isolation')` before a single layer is constructed, so an APPLE column
    cannot state this lane's proposition either -- the M4 is not a smaller
    version of the right box, it is the wrong vendor. It is in
    `GPU_ONLY_LANES` for that reason and in `par-*` for the ordinary one.

    OWED, on a two-device CUDA or HIP box and nowhere else:
      MOJOLEARN_PAR_DEVICES=0,1 MOJOLEARN_NUMERIC_MODE=identical python3 \\
        tools/identity_break.py --lanes par-causal-lm --repeats 2 \\
        --step-full --batch-scale --ragged --batch-grad --json <column>.2gpu.json"""
    from mojolearn import _causal_lm_fixtures as fx
    from mojolearn.models import ParallelCausalLM
    dev = _par_devices()
    A = ml.Array.from_buffer
    ids = _ids(X, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)
    cfg, tensors = fx.family_fixture("llama", False)
    with tempfile.TemporaryDirectory(prefix="ib-par-causal-lm-") as d:
        root = fx._write_checkpoint(os.path.join(d, "llama"), cfg, tensors)
        plain = ml.models.causal_lm.CausalLM.load(root)
        want = np.ascontiguousarray(np.asarray(plain.forward(A(ids))))
        n_layers = plain.plan.n_layers
        # ONE DEVICE INDEX PER LAYER, cycling the pool so a two-device column
        # puts consecutive layers on different owners and every hidden state
        # crosses a real boundary; at one device the tuple repeats an index,
        # which the class documents as allowed and which is exactly the
        # degenerate case `_par_devices` describes.
        owners = tuple(dev[i % len(dev)] for i in range(n_layers))
        with ParallelCausalLM.load(root, layer_devices=owners) as par:
            got = np.ascontiguousarray(np.asarray(par.forward(A(ids))))
            params = np.concatenate([_hf_bytes(w) for _, w in sorted(par.parameters().items())])
            names_match = sorted(par.parameters()) == sorted(plain.parameters())
        _same_bytes("ParallelCausalLM forward", got, "plain CausalLM forward", want)
        flags = np.asarray([
            names_match,
            _refused(lambda: ParallelCausalLM.load(root, layer_devices=owners[:-1]), "layer_devices"),
            _refused(lambda: ParallelCausalLM.load(root, layer_devices=(-1,) * n_layers), "layer_devices"),
            _refused(lambda: ParallelCausalLM.load(root, layer_devices=owners, weight_format="fp8"),
                     "weight_format"),
            ml.models.ParallelCausalLM is ParallelCausalLM,
        ], dtype=np.int64)
    return _fit(dict(logits=_h(got), params=_h(params), flags=_h(flags)))


@lane("par-cross-val")
def _(ml, X, yc, yr, Xh=None):
    """`parallel_model_selection.cross_val_score`: ONE FRESH ESTIMATOR PER
    FOLD on the requested devices, dispatched in bounded waves of
    `len(pool.devices)` including a final partial wave.

    THE CLAIM IS THE SERIAL API'S BITS. Its own docstring says "Fold
    validation, cloning, scoring and output order match the serial API", so
    the cell holds the parallel scores to `model_selection.cross_val_score`
    on the same estimator and folds, byte for byte. The `cross-val` lane
    hashes the serial side of that comparison and the `cross-val-folds` lane
    hashes the fold partition itself; this lane adds only the dispatch, which
    is the one thing neither of those can see.

    LIKE `par-causal-lm`, THIS IS NOT A CPU GAP THAT A CPU RUN COULD CLOSE.
    `cross_val_score` raises `NotImplementedError('parallel cross-validation
    requires CUDA or HIP GPU workers')` immediately after constructing the
    pool and before `_prepare_folds`, so on a CPU or Apple column there is no
    fold, no clone and no score to hash -- only the refusal, and a lane whose
    only cell is a refusal sentence is the `kmeans-cosine` shape, which is why
    this one is in `GPU_ONLY_LANES` instead of pretending to a column.

    It also calls `require_distinct_workers` on a device inventory, so a
    ONE-device run of it is not merely degenerate on the device axis the way
    the other `par-*` lanes are; the width-1 wave is a different dispatch
    from the width-2 one. The owed column is therefore TWO devices, and a
    one-device column would be evidence about a code path the driver does not
    take in service.

    OWED, on a two-device CUDA or HIP box and nowhere else:
      MOJOLEARN_PAR_DEVICES=0,1 MOJOLEARN_NUMERIC_MODE=identical python3 \\
        tools/identity_break.py --lanes par-cross-val --repeats 2 \\
        --step-full --batch-scale --ragged --batch-grad --json <column>.2gpu.json"""
    from mojolearn.parallel_model_selection import cross_val_score
    dev = _par_devices()
    est = _gbdt(ml.GradientBoostingRegressor, n_estimators=8, max_depth=4)
    scores = cross_val_score(est, X, yr, devices=dev, cv=3)
    serial = ml.model_selection.cross_val_score(
        _gbdt(ml.GradientBoostingRegressor, n_estimators=8, max_depth=4), X, yr, cv=3)
    _same_bytes("parallel cross_val_score", np.asarray(scores, dtype=np.float64),
                "serial cross_val_score", np.asarray(serial, dtype=np.float64))
    flags = np.asarray([
        _refused(lambda: cross_val_score(est, X, yr, devices=()), "devices"),
        _refused(lambda: cross_val_score(
            _gbdt(ml.GradientBoostingRegressor, n_estimators=8, max_depth=4, numeric_mode="fast"),
            X, yr, devices=dev, cv=3), "IDENTICAL"),
        _refused(lambda: cross_val_score(est, X, yr, devices=dev, cv=3, error_score=0.0), "error_score"),
    ], dtype=np.int64)
    return _fit(dict(scores=_h(np.asarray(scores, dtype=np.float64)), flags=_h(flags)))


def _neural_inference(ml, lane_name, est):
    """The estimator a neural lane's held-out and batch cells ask
    (lane/inference-tokenizer-neural, 2026-09-15). On a CPU column
    (`ml.vendor() == "cpu"`) it is the PUBLIC inference class built from the
    fitted model: `MLPInference.from_checkpoint` on the trainer's saved
    checkpoint, or `TransformerBlockInference` on the block's nine weights,
    both over the shipped `_mojolearn_neural_host`. On a GPU column it is the
    fitted estimator itself, as before, so no GPU cell changes meaning. The
    train rows never go through here."""
    if ml.vendor() != "cpu":
        return est
    lane_name = _lowbit_base(lane_name)
    if lane_name == "mlp":
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mlp.json")
            est.save_checkpoint(path)
            return ml.MLPInference.from_checkpoint(path)
    # lane/inference-neural-forward (2026-09-15): the Mamba blocks from their
    # named weights, the Samba stack from its saved checkpoint and the byte LM
    # from its exported checkpoint, each through its public inference class.
    if lane_name in ("mamba1", "mamba2", "mamba3", "mamba2-dtlimit"):
        cls = dict(mamba1=ml.Mamba1BlockInference, mamba3=ml.Mamba3BlockInference).get(
            lane_name, ml.Mamba2BlockInference)
        kw = dict(dt_limit=est.dt_limit) if cls is ml.Mamba2BlockInference else {}
        return cls(dict(zip(est._W_NAMES, est._w)), **kw)
    if lane_name in ("samba", "samba-untied-dropout-accum"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "samba.ckpt")
            est.save_checkpoint(path)
            return ml.SambaInference.from_checkpoint(path)
    if lane_name in ("byte-lm", "byte-lm-resident"):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "byte_lm.json")
            est.export_checkpoint(path)
            return ml.LanguageModelInference.from_checkpoint(path)
    return ml.TransformerBlockInference(dict(zip(est._W_NAMES, est._w)), n_heads=est.n_heads,
                                        n_kv_heads=est.n_kv_heads, head_dim=est.head_dim, window=est.window)


#: The lanes whose infer, reload and batch cells ask the public CPU inference
#: class on a CPU column through `_public_est` (lane/inference-neural-forward).
#: mlp, transformer and transformer-window wrap theirs in the lane bodies and
#: batch declarations above, so they are named only for the opt-in parts.
NEURAL_PUBLIC_LANES = ("mamba1", "mamba2", "mamba3", "mamba2-dtlimit", "samba",
                       "samba-untied-dropout-accum", "byte-lm", "byte-lm-resident",
                       # lane/identical-lowbit-inference (2026-09-17)
                       "mamba1-bf16w", "mamba1-int8w", "mamba2-bf16w", "mamba2-int8w",
                       "mamba3-bf16w", "mamba3-int8w")


#: lane/identical-lowbit-inference (2026-09-17): the `-bf16w` and `-int8w`
#: lanes are their base lane with the projection weights stored packed
#: (mojolearn.lowbit) and materialized exactly on the column; every helper
#: keyed by lane name reads the base name.
LOWBIT_SUFFIXES = ("-bf16w", "-int8w")


def _lowbit_base(name):
    for suf in LOWBIT_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name
NEURAL_PUBLIC_PART_LANES = NEURAL_PUBLIC_LANES + ("transformer", "transformer-window")


def _public_est(name, est, lanes=NEURAL_PUBLIC_LANES):
    """`est`, or on a CPU column the public inference class built from it
    for a lane in `lanes`. GPU columns are unchanged."""
    if name not in lanes or est is None:
        return est
    return _neural_inference(sys.modules["mojolearn"], name, est)


# ---------------------------------------------------------------- the batch part (2026-09-14)
# See the `batch` part in the module docstring. A declaration per lane, kept
# OUT of the lane bodies so no train, infer or model hash can move because
# of it. Each declaration is either an `n/a:<reason>` string or a function
# `(ml, est, Xh) -> [call, ...]` whose calls are _BatchRows or _BatchPrefix.

#: rows evaluated ALONE (batch of 1) per rows call; `--batch-alone` overrides
BATCH_ALONE = 16

#: the uneven split: chunks [0, 1), [1, 8), [8, n)
BATCH_SPLIT = (1, 8)

#: the sabotage switch: perturbs the WHOLE-batch evaluation of every call so
#: the part is SEEN TO FAIL; recorded as `batch_sabotage` in the JSON
BATCH_SABOTAGE_ENV = "MOJOLEARN_IDENTITY_BATCH_SABOTAGE"

BATCH = {}


def _batch_decl(spec, *names):
    for n in names:
        if n in BATCH:
            raise RuntimeError(f"identity_break: lane {n!r} has two batch declarations")
        BATCH[n] = spec


class _PerRow(list):
    """A ragged output: one array per row (a radius query's neighbors)."""


class _BatchRows:
    """One row-wise inference call. `R` is the held-out input whose axis 0
    is the batch; `fn(R[a:b])` returns a tuple of outputs whose axis 0 (or
    _PerRow index) is the same rows. The harness evaluates the whole batch,
    each of the first `BATCH_ALONE` rows alone, and the split 1, 7, rest.

    `min_batch` and `refusal` declare a method that REFUSES a smaller batch
    BY NAME. The coordinate descent predict matches cuML's `cdPredict`
    (`cd.cuh:341`, `ASSERT(n_rows > 1, "Parameter n_rows: number of rows
    cannot be less than two")`, restated in solver/impl/cd.mojo::cd_predict
    and bindings/_mojolearn_solver_host.mojo), so a row cannot be asked
    alone; until 2026-09-14 evening the part asked anyway and lasso and
    elasticnet read REFUSED on every fixture. With min_batch = m the harness
    first holds the call to its refusal (a batch of m - 1 must raise with
    `refusal` in the message, or the part reads REFUSED saying the refusal
    was lifted), then asks each of the first `BATCH_ALONE` rows inside the
    smallest admitted window, m rows starting at it, and splits into chunks
    of at least m rows. min_batch = 1 is the protocol above, byte for byte."""

    def __init__(self, label, R, fn, min_batch=1, refusal=None):
        self.label, self.R, self.fn = label, np.ascontiguousarray(R), fn
        if min_batch < 1 or (min_batch > 1 and not refusal):
            raise ValueError(f"{label}: min_batch {min_batch} needs the refusal sentence it mirrors")
        self.min_batch, self.refusal = int(min_batch), refusal


class _BatchPrefix:
    """One length call for a causal or sequential output. `fn(p)` returns a
    tuple of outputs whose `axis` is the length; the answer at length `p`
    must equal the first `p` positions of the answer at `full`."""

    def __init__(self, label, full, fn, axis):
        self.label, self.full, self.fn, self.axis = label, int(full), fn, axis


def _prefix_lengths(full):
    return sorted(p for p in {1, 7, full - 1} if 1 <= p < full)


def _as_rows(outputs, n, what):
    """Per-row form of a call's outputs: a list of n rows, each a tuple of
    (dtype, shape, bytes) per output. Refuses an output whose leading axis
    is not the n rows, and an object or text array (as `_h` does)."""
    rows = [[] for _ in range(n)]
    for k, o in enumerate(outputs):
        if isinstance(o, _PerRow):
            if len(o) != n:
                raise ValueError(f"{what}: output {k} has {len(o)} rows, expected {n}")
            items = [np.ascontiguousarray(np.asarray(v)) for v in o]
        else:
            a = np.asarray(o)
            if a.ndim == 0 or a.shape[0] != n:
                raise ValueError(f"{what}: output {k} has shape {a.shape}, expected leading axis {n}")
            items = [np.ascontiguousarray(a[i]) for i in range(n)]
        for i, v in enumerate(items):
            if v.dtype == object or v.dtype.kind in "OUSV":
                raise TypeError(f"{what}: output {k} is dtype={v.dtype}; pass the numeric arrays")
            rows[i].append((v.dtype.str, v.shape, v.tobytes()))
    return rows


def _elem_hex(b, itemsize, e):
    return "0x" + b[e * itemsize:(e + 1) * itemsize][::-1].hex()


def _row_mismatch(whole, other, name_other):
    """None when the two rows are the same bytes, else where they first
    differ, with both values in hex (little-endian element bytes)."""
    for k, ((dw, sw, bw), (do, so, bo)) in enumerate(zip(whole, other)):
        if dw != do or sw != so:
            return f"output {k}: whole {dw}{list(sw)} vs {name_other} {do}{list(so)}"
        if bw != bo:
            size = np.dtype(dw).itemsize
            first = next(j for j in range(len(bw)) if bw[j] != bo[j])
            e = first // size
            return (f"output {k} element {e}: whole {_elem_hex(bw, size, e)} "
                    f"vs {name_other} {_elem_hex(bo, size, e)}")
    if len(whole) != len(other):
        return f"whole has {len(whole)} outputs, {name_other} has {len(other)}"
    return None


def _sabotage_rows(rows):
    """Flip the lowest bit of the first element of the first floating output
    of the whole-batch rows (one ulp for a float; an integer output is used
    when none floats), in place. A byte flip and not `np.nextafter`, which
    on NumPy 1.x promotes a float32 scalar against a Python float to float64
    and can round back to the same bits, and which leaves a NaN a NaN."""
    for kinds in ("f", "iub"):
        for r in rows:
            for k, (dt, shape, b) in enumerate(r):
                if np.dtype(dt).kind not in kinds or not b:
                    continue
                flipped = bytearray(b)
                flipped[0] ^= 1
                r[k] = (dt, shape, bytes(flipped))
                return


def _split_bounds(n, min_batch=1):
    """[0, 1, 8, n] for min_batch 1 (the protocol since the part began);
    every chunk at least min_batch rows otherwise ([0, 2, 8, n] at 2)."""
    bounds = [0]
    for b in BATCH_SPLIT:
        b = max(b, bounds[-1] + min_batch)
        if b <= n - min_batch:
            bounds.append(b)
    return bounds + [n]


def _eval_batch_rows(call, alone, sabotage, digest):
    n, mb = call.R.shape[0], call.min_batch
    if n < mb:
        raise ValueError(f"{call.label}: {n} rows is below the declared min_batch {mb}")
    if mb > 1:
        # the declared refusal is part of the claim: a smaller batch must
        # still refuse with the mirrored sentence
        try:
            call.fn(np.ascontiguousarray(call.R[:mb - 1]))
        except Exception as exc:
            if call.refusal not in str(exc):
                raise RuntimeError(f"{call.label}: a batch of {mb - 1} raised something other than the declared "
                                   f"refusal {call.refusal!r}: {type(exc).__name__}: {exc}") from None
        else:
            raise RuntimeError(f"{call.label}: a batch of {mb - 1} no longer refuses ({call.refusal!r} was "
                               f"declared); the refusal was lifted, so drop min_batch and ask rows alone")
    whole = _as_rows(call.fn(np.ascontiguousarray(call.R)), n, f"{call.label} whole")
    if sabotage:
        _sabotage_rows(whole)
    for i in range(min(alone, n)):
        a = min(i, n - mb)
        one = _as_rows(call.fn(np.ascontiguousarray(call.R[a:a + mb])), mb,
                       f"{call.label} row {i} alone" if mb == 1 else f"{call.label} rows [{a},{a + mb})")
        where = "alone" if mb == 1 else f"in a batch of {mb}"
        m = _row_mismatch(whole[i], one[i - a], where)
        if m:
            return f"BATCH_MOVED:{call.label}:row {i} of {n} {where}:{m}"
    bounds = _split_bounds(n, mb)
    got = []
    for a, b in zip(bounds, bounds[1:]):
        got.extend(_as_rows(call.fn(np.ascontiguousarray(call.R[a:b])), b - a, f"{call.label} chunk [{a},{b})"))
    split_name = "split " + ",".join(str(b - a) for a, b in zip(bounds, bounds[1:]))
    for i in range(n):
        m = _row_mismatch(whole[i], got[i], split_name)
        if m:
            return f"BATCH_MOVED:{call.label}:row {i} of {n} in {split_name}:{m}"
    digest.update((f"rows:{call.label}:{n}" + (f":min_batch={mb}" if mb > 1 else "")).encode())
    for r in whole:
        for dt, shape, b in r:
            digest.update(f"{dt}{shape}".encode())
            digest.update(b)
    return None


def _eval_batch_prefix(call, sabotage, digest):
    full = [np.array(np.asarray(o), copy=True) for o in call.fn(call.full)]
    for o in full:
        if o.dtype == object or o.dtype.kind in "OUSV":
            raise TypeError(f"{call.label}: output is dtype={o.dtype}; pass the numeric arrays")
    if sabotage:
        # the same one-bit flip as _sabotage_rows, on the whole-length answer
        for kinds in ("f", "iub"):
            hit = next((i for i, o in enumerate(full) if o.dtype.kind in kinds and o.size), None)
            if hit is not None:
                o = np.ascontiguousarray(full[hit])
                flipped = bytearray(o.tobytes())
                flipped[0] ^= 1
                full[hit] = np.frombuffer(bytes(flipped), dtype=o.dtype).reshape(o.shape)
                break
    for p in (getattr(call, "lengths", None) or _prefix_lengths(call.full)):
        part = [np.asarray(o) for o in call.fn(p)]
        for k, (F, P) in enumerate(zip(full, part)):
            want = np.ascontiguousarray(np.take(F, np.arange(p), axis=call.axis))
            P = np.ascontiguousarray(P)
            where = f"BATCH_MOVED:{call.label}:length {p} of {call.full}"
            if want.dtype != P.dtype or want.shape != P.shape:
                return f"{where}:output {k}: whole {want.dtype.str}{list(want.shape)} vs prefix {P.dtype.str}{list(P.shape)}"
            wb, pb = want.tobytes(), P.tobytes()
            if wb != pb:
                size = want.dtype.itemsize
                e = next(j for j in range(len(wb)) if wb[j] != pb[j]) // size
                idx = np.unravel_index(e, want.shape)
                pos = int(idx[call.axis])
                return (f"{where}:position {pos} output {k} element {e}: whole {_elem_hex(wb, size, e)} "
                        f"vs prefix {_elem_hex(pb, size, e)}")
        if len(part) != len(full):
            return f"BATCH_MOVED:{call.label}:length {p}: {len(part)} outputs vs {len(full)}"
    digest.update(f"prefix:{call.label}:{call.full}:{call.axis}".encode())
    for o in full:
        o = np.ascontiguousarray(o)
        digest.update(f"{o.dtype.str}{o.shape}".encode())
        digest.update(o.tobytes())
    return None


def _probe_batch(fit, name, ml, Xh, alone, sabotage):
    """The batch part of ONE fit: (value, error). The value is a 16-hex hash
    of every call's whole-batch outputs when every alone, split and prefix
    evaluation is the same bytes, a `BATCH_MOVED:<call>:<where>:<values>`
    verdict at the first one that is not, or an `n/a:<reason>` string. A
    raise leaves the value None (REFUSED) with the stage in the error."""
    spec = BATCH.get(name, "n/a:UNDECLARED")
    if isinstance(spec, str):
        return spec, None
    try:
        calls = spec(ml, _public_est(name, fit.est), Xh)
        digest = hashlib.sha256()
        for call in calls:
            if isinstance(call, _BatchRows):
                moved = _eval_batch_rows(call, alone, sabotage, digest)
            else:
                moved = _eval_batch_prefix(call, sabotage, digest)
            if moved:
                return moved[:400], None
        return digest.hexdigest()[:16], None
    except Exception as exc:
        return None, _exc_text("batch", exc)
    finally:
        # drivers a declaration opened for the part (the query drivers)
        while _BATCH_CLOSE:
            _BATCH_CLOSE.pop()()


def _batch_column_verdict(values):
    """STABLE, MOVED (the hash disagreed across repeats), BATCH_MOVED (a
    repeat's whole batch disagreed with its rows alone, a split or a
    prefix), N/A or REFUSED."""
    if any(v is None for v in values):
        return "REFUSED"
    if all(v.startswith("n/a") for v in values):
        return "N/A"
    if any(v.startswith("BATCH_MOVED") for v in values):
        return "BATCH_MOVED"
    return "STABLE" if len(set(values)) == 1 else "MOVED"


# The declarations, every lane. Inputs are the lane's held-out slice and
# the lane's own method; a derived input (_coded, _with_nan, the kde cosine
# shift) is derived ONCE from the whole held-out slice, because a median or a
# stride recomputed on a chunk would hand the chunk different bytes.

def _rows_calls(*methods, sl=slice(None), prep=None, min_batch=1, refusal=None):
    def spec(ml, e, Xh):
        R = Xh[sl] if prep is None else prep(Xh)[sl]
        return [_BatchRows(m, R, (lambda m: lambda r: (getattr(e, m)(r),))(m), min_batch, refusal) for m in methods]
    return spec


_batch_decl(_rows_calls("predict", "predict_proba"),
            "gbdt-ordered",
            "rf-clf", "et-clf", "gbdt-symmetric", "gbdt-symmetric-eval",
            "rf-clf-entropy-log2-noboot", "rf-clf-balanced-parallel",
            "et-clf-entropy-bestfirst", "gbdt-multiclass", "gbdt-onevsall", "gbdt-pointwise-l2-bayesian-eval",
            "par-forest", "par-boosting", "par-forest-et-clf")
_batch_decl(_rows_calls("predict"),
            "rf-reg", "et-reg", "gbdt-depthwise", "gbdt-lossguide", "gbdt-rmse", "gbdt-ordered-rmse",
            "rf-reg-poisson", "rf-reg-gamma-ig", "et-reg-bootstrap-parallel", "gbdt-parametric-losses",
            "gbdt-lossguide-newtoncosine", "gbdt-exact-mae", "gbdt-adapter-reg", "par-forest-et",
            "gbdt-query-rmse", "gbdt-pair-logit", "gbdt-yeti-rank",
            "par-forest-reg", "par-boosting-reg", "gbdt-border-types",
            "gbdt-ordered-bayesian-noise", "par-ordered", "gbdt-bfa-quantile",
            "gbdt-catboost-defaults", "par-border-types")
_batch_decl(_rows_calls("predict", prep=_coded), "gbdt-feature-freq", "gbdt-categorical-ctr")
_batch_decl(_rows_calls("predict", prep=_with_nan), "gbdt-nan-modes")
_batch_decl(_rows_calls("predict", "predict_proba", prep=_ctr_tables_xh), "gbdt-categorical-ctr-tables")
_batch_decl(_rows_calls("predict", prep=_tensor_ctr_xh), "gbdt-tensor-ctr-tables")


def _batch_gbdt_adapter_clf(ml, e, Xh):
    return (_rows_calls("predict", "predict_proba")(ml, e, Xh)
            + _rows_calls("decision_function", sl=slice(0, 512))(ml, e, Xh))


_batch_decl(_batch_gbdt_adapter_clf, "gbdt-adapter-clf", "par-boosting-clf")


def _batch_iforest(ml, e, Xh):
    return (_rows_calls("score_samples")(ml, e, Xh)
            + _rows_calls("predict", sl=slice(0, 512))(ml, e, Xh))


_batch_decl(_batch_iforest, "iforest", "iforest-tuned", "par-iforest")

_batch_decl(_rows_calls("predict", "transform"), "kmeans", "kmeans-random", "kmeans-array", "kmeans-weighted",
            "kmeans-sqrt", "kmeans-classic-pp", "par-kmeans")
# The transductive clustering lanes (read 2026-09-15 against the Python
# estimator, the GPU binding, the CPU host binding and cuML v26.08.00).
#   DBSCAN and AgglomerativeClustering are NOT transductive since
#     lane/inference-transductive-predict (2026-09-15): with
#     prediction_data=True both have `predict` on the GPU and CPU host
#     estimators bindings (labeled_reference_predict, DEVIATION 2740, NEW
#     capability: cuML dbscan.pyx has fit (:301) and fit_predict (:478) only,
#     agglomerative.pyx fit (:139) and fit_predict (:216) only), so the part
#     asks predict on 64 held-out rows.
#   SpectralClustering is NOT transductive since lane/spectral-predict
#     (2026-09-15): with prediction_data=True it has `predict` on the GPU and
#     CPU host metrics bindings (spectral_predict, the Nystrom extension,
#     DEVIATION 2860, NEW capability: cuML spectral_clustering.pyx has fit
#     (:263) and fit_predict (:239) only), so the part asks predict on 64
#     held-out rows, or on their affinity rows for the precomputed lane.
#   HDBSCAN is NOT transductive since 2026-09-15: HDBSCAN(prediction_data=True)
#     and mojolearn.hdbscan.approximate_predict (cuML hdbscan.pyx:1264,
#     predict.cuh:220-262) on both bindings, below, and since the same day
#     membership_vector (:1180, soft_clustering.cuh:501-627, DEVIATION 1616),
#     which the part asks too. all_points_membership_vectors (:1114) has no
#     held-out rows; the lanes' infer probe hashes it.
_batch_decl(_rows_calls("predict", sl=np.s_[:64, :4]),
            "dbscan", "dbscan-brute-l1", "dbscan-weighted", "par-dbscan", "agglomerative")
_batch_decl(_rows_calls("predict", sl=np.s_[:64, :4]), "spectral")
_batch_decl(lambda ml, e, Xh: [_BatchRows("predict", e._identity_heldout_affinity[:64], lambda r: (e.predict(r),))],
            "spectral-precomputed")


def _batch_hdbscan(ml, e, Xh):
    """approximate_predict's labels and probabilities and membership_vector's
    rows, 64 held-out rows."""
    return [_BatchRows("approximate_predict", Xh[:64, :4], lambda r: tuple(ml.hdbscan.approximate_predict(e, r))),
            _BatchRows("membership_vector", Xh[:64, :4], lambda r: (ml.hdbscan.membership_vector(e, r),))]


_batch_decl(_batch_hdbscan, "hdbscan", "hdbscan-leaf")


_batch_decl(_rows_calls("predict", "predict_proba", sl=(slice(0, 64), slice(0, 4))), "gpc", "gpc-multiclass")


def _batch_kneighbors(ml, e, Xh):
    return [_BatchRows("kneighbors", Xh[:64], lambda r: tuple(e.kneighbors(r)))]


_batch_decl(_batch_kneighbors, "knn", "knn-sqeuclidean", "knn-manhattan", "knn-chebyshev", "knn-cosine",
            "knn-minkowski-p3", "knn-rbc")
_batch_decl(_rows_calls("predict", "predict_proba", sl=slice(0, 64)), "knn-clf", "knn-clf-distance")
_batch_decl(_rows_calls("predict", sl=slice(0, 64)), "knn-reg", "knn-reg-distance")


def _batch_radius(ml, e, Xh):
    def fn(r):
        dists, idx = e.radius_neighbors(r, sort_results=True)
        n = len(idx)
        return (_PerRow(np.asarray(dists[i], dtype=np.float32).reshape(-1) for i in range(n)),
                _PerRow(np.asarray(idx[i]).reshape(-1).astype(np.int64) for i in range(n)))
    return [_BatchRows("radius_neighbors", Xh[:64], fn)]


_batch_decl(_batch_radius, "radius", "radius-manhattan", "radius-chebyshev", "radius-minkowski-p3")
_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "pca", "pca-whiten", "pca-full-whiten", "tsvd",
            "standard-scaler", "minmax-scaler", "standard-scaler-no-mean", "standard-scaler-no-std",
            "minmax-scaler-clip", "par-scaler", "rbf-sampler",
            "par-gram-pca", "par-gram-tsvd", "par-scaler-minmax")
_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "ols", "ridge", "ols-no-intercept",
            "ols-weighted", "ridge-no-intercept", "par-gram", "svr", "svr-linear",
            "par-gram-ols", "par-svm-svr")
# The coordinate descent predict refuses one row BY NAME, mirroring cuML's
# cdPredict (cd.cuh:341); see _BatchRows. Its rows are asked in windows of two.
CD_PREDICT_REFUSAL = "Parameter n_rows: number of rows cannot be less than two"
_batch_decl(_rows_calls("predict", sl=slice(0, 256), min_batch=2, refusal=CD_PREDICT_REFUSAL),
            "lasso", "elasticnet", "elasticnet-l2end-no-intercept", "par-cd", "par-cd-elasticnet")
_batch_decl(_rows_calls("predict_proba", sl=slice(0, 256)), "logistic", "logistic-l1", "logistic-elasticnet",
            "logistic-unpenalized-no-intercept", "par-logistic")
_batch_decl(_rows_calls("predict_proba", "predict", sl=slice(0, 256)), "logistic-multiclass")
_batch_decl(_rows_calls("decision_function", "predict", sl=slice(0, 256)), "svc", "svc-linear", "par-svm", "svc-poly")
_batch_decl(_rows_calls("score_samples", sl=(slice(0, 256), slice(0, 4))), "kde", "kde-weighted",
            "kde-tophat-sqeuclidean", "kde-epanechnikov-l1", "kde-exponential-chebyshev",
            "kde-cosine-minkowski")
# the cosine metric lane shifts every row by 8 (see _kde_lane); the shift is
# per element, so it is applied to the whole slice before any split
_batch_decl(_rows_calls("score_samples", sl=(slice(0, 256), slice(0, 4)),
                        prep=lambda Xh: (Xh + np.float32(8.0)).astype(np.float32)),
            "kde-linear-cosine")


def _batch_gp(ml, e, Xh):
    return [_BatchRows("predict(return_std=True)", Xh[:64, :4], lambda r: tuple(e.predict(r, return_std=True)))]


_batch_decl(_batch_gp, "gp", "gp-matern12", "gp-matern32", "gp-matern52-ard", "par-gp", "gp-normalize-y",
            "gp-optimize", "gp-optimize-restarts")
# GaussianProcessRegressor.sample_y factors the posterior covariance over ALL
# the rows of one call, so a row's draw depends on every other row asked with
# it: splitting the rows changes the covariance, by the reference's contract
# (gaussian_process/checks/sample_y.mojo, DEVIATION 2793)
_batch_decl("n/a:jointly-correlated (sample_y draws all rows of a call from one joint posterior)",
            "gp-sample-y", "gp-sample-y-normalize")
# UMAP.transform WAS batch-dependent by its own contract, and until
# lane/umap-batch-fix (2026-09-16) these two lanes carried a batch EXEMPTION
# saying so. It named four couplings, all of which are now per row
# (umap/transform.mojo and the CPU host restatement
# umap/host/umap_oracle.mojo, which stays character identical to it):
#   1. the sigma floor 0.001 * mean was over EVERY query's neighbor
#      distances and is now over the row's own k;
#   2. the edge schedule scaled each weight by the maximum over the whole
#      batch and now uses the row's own maximum;
#   3. the negative-sample counter hashed the batch-local edge ordinal
#      row * k + j, so a row's draws depended on its POSITION in the request,
#      and is now keyed on a hash of the row's own k neighbor indices;
#   4. with n_epochs == 0 the epoch count was 100 or 30 by n_queries and is
#      now 100 at every size. The lanes pass n_epochs=8, so this one was
#      never reached in any recorded column.
# So the exemption becomes a PART. A BATCH_MOVED here is now a defect, and it
# is the arm that would catch any of the four coming back. The claim it
# encodes was measured bitwise on the CPU host route and on Metal, both routes
# agreeing to the last bit, by umap/checks/batch_determinism_check.mojo,
# umap/checks/batch_determinism_device_check.mojo and
# umap/checks/batch_epoch_cliff_check.mojo.
# This is a deliberate divergence from cuML, whose transform couples a batch
# the same four ways (runner.cuh:549-554 epochs by inputs.n,
# fuzzy_simpl_set/naive.cuh:168 the mean_dist sigma floor,
# optimize_batch_kernel.cuh:267 Philox seeded by the batch COO row) and whose
# docstring says "the transform() function is stochastic" (umap.pyx:1475).
# The standing rule is not to reproduce a reference library's bug.


def _batch_umap(ml, e, Xh):
    """UMAP.transform, each row alone against the whole request."""
    return [_BatchRows("transform", np.ascontiguousarray(Xh[:64, :8]),
                       lambda r: (e.transform(r),))]


def _batch_par_graph_umap(ml, e, Xh):
    """The same arm through the parallel graph surface, whose transform is a
    free function over the fitted graph rather than a method."""
    from mojolearn.parallel_graph import transform_umap
    return [_BatchRows("transform_umap", np.ascontiguousarray(Xh[:64, :8]),
                       lambda r: (transform_umap(e, np.ascontiguousarray(r),
                                                 devices=_par_devices()),))]


_batch_decl(_batch_umap, "umap")
_batch_decl(_batch_par_graph_umap, "par-graph-umap")
def _kernel_variant_rows(X):
    return np.ascontiguousarray(X[:64, :4] * np.float32(0.125))


_batch_decl(_rows_calls("predict", prep=_kernel_variant_rows),
            "kernel-ridge-poly", "kernel-ridge-sigmoid", "kernel-ridge-laplacian")
_batch_decl(_rows_calls("transform", prep=_kernel_variant_rows),
            "nystroem-poly", "nystroem-sigmoid", "nystroem-laplacian")
_batch_decl(_rows_calls("predict", sl=(slice(0, 64), slice(0, 4))), "kernel-ridge")
_batch_decl(_rows_calls("transform", sl=(slice(0, 64), slice(0, 4))), "nystroem")
_batch_decl(_rows_calls("score_samples", "predict", "predict_proba", sl=(slice(0, 64), slice(0, 4))), "gmm")
_batch_decl(_rows_calls("score_samples", sl=(slice(0, 64), slice(0, 4))), "gmm-random-init")
# GaussianMixture.sample draws n_samples rows from the fitted model and reads no
# input rows, so there is no row axis to split (mixture/checks/sample.mojo)
_batch_decl("n/a:no-batch-axis (GaussianMixture.sample takes no input rows)", "gmm-sample", "gmm-random-init-sample")


def _batch_cholesky(ml, e, Xh):
    """The batch of a triangular solve is its right-hand sides, so each
    column of B is solved alone and against the whole B. Two columns, so the split is
    one and one."""
    B = np.ascontiguousarray(Xh[:256, :2])
    return [_BatchRows("solve (right-hand sides)", B.T,
                       lambda r: (np.asarray(e.solve(np.ascontiguousarray(r.T))).reshape(B.shape[0], -1).T,))]


_batch_decl(_batch_cholesky, "cholesky")


def _batch_forecast(ml, e, Xh):
    """No rows to split (a series batch would be a different fit); the
    held-out axis is the horizon, so the batch part asks for LENGTH
    invariance, forecast(p) equal to the first p steps of forecast(H)."""
    return [_BatchPrefix("forecast", FORECAST_HORIZON, lambda h: (e.forecast(h),), axis=-1)]


_batch_decl(_batch_forecast, "holtwinters", "holtwinters-multiplicative", "arima", "arima-011",
            "arima-seasonal-c", "par-arima")


def _batch_par_forecast(driver, axis, **kw):
    """The forecasting DRIVERS' batch part (2026-09-19): `_batch_forecast`'s
    LENGTH invariance asked through the sharded entry instead of the
    estimator's own, so the part exercises the partition at every horizon
    the prefix protocol asks for (1, 7 and H - 1), not only at H. There are
    still no input rows to split: a series batch would be a different fit."""
    def spec(ml, e, Xh):
        import mojolearn.parallel_forecasting as pf
        d = _par_devices()
        fn = getattr(pf, driver)
        return [_BatchPrefix(driver, FORECAST_HORIZON,
                             lambda h: (fn(e, h, devices=d, series_per_shard=3, **kw),), axis=axis)]
    return spec


_batch_decl(_batch_par_forecast("forecast_arima", -1), "par-forecast-arima")
_batch_decl(_batch_par_forecast("forecast_exponential_smoothing", 0), "par-forecast-holtwinters")


def _batch_forecast_exog(ml, e, Xh):
    """The forecasters' LENGTH invariance with regressors (lane/arima-exog):
    forecast(p, exog=F[:, :p]) equal to the first p steps of forecast(H, F).
    A forecast step reads its own row of the future regressors and, when
    differenced, the rows `period` before it, never a later one, so a prefix
    of the future is enough for a prefix of the forecast."""
    F = _arima_exog_block(Xh, FORECAST_HORIZON)
    return [_BatchPrefix("forecast", FORECAST_HORIZON,
                         lambda h: (e.forecast(h, exog=np.ascontiguousarray(F[:, :h])),), axis=-1)]


_batch_decl(_batch_forecast_exog, "arima-exog", "arima-exog-seasonal")


def _batch_kpss(ml, e, Xh):
    """kpss_test takes a batch of series, one per column; each series tested
    alone against the four together."""
    S = np.ascontiguousarray(Xh[:512, :4])

    def fn(r):
        y = np.ascontiguousarray(r.T)
        k = r.shape[0]
        flat = ml.kpss_test(y, d=0)
        diffed, stat = ml.kpss_test(y, d=1, D=1, s=12, return_statistic=True)
        return tuple(np.asarray(a).reshape(k, -1) for a in (flat, diffed, stat))
    return [_BatchRows("kpss_test", S.T, fn)]


_batch_decl(_batch_kpss, "kpss")


def _batch_gemm_pinned(ml, e, Xh):
    """GEMM's batch is its row count m; contract 6.1 says the leaf size may
    not read it. Rows of A alone against the whole A, OP_NN."""
    B = np.ascontiguousarray(Xh[256:320].T).astype(np.float32)          # d x 64
    return [_BatchRows("matmul OP_NN", Xh[:256].astype(np.float32),
                       lambda r: (np.asarray(ml.linalg.matmul(r, B, identical=True)),))]


def _batch_gemm_transposed(ml, e, Xh):
    b = np.ascontiguousarray(Xh[256:384]).astype(np.float32)             # 128 x d
    c = np.ascontiguousarray(Xh[384:384 + Xh.shape[1]]).astype(np.float32)   # d x d
    a = Xh[:256].astype(np.float32)
    return [_BatchRows("matmul OP_NT", a, lambda r: (np.asarray(ml.linalg.matmul(r, b, transpose_b=True, identical=True)),)),
            _BatchRows("matmul OP_TN", a, lambda r: (np.asarray(ml.linalg.matmul(np.ascontiguousarray(r.T), c,
                                                                                 transpose_a=True, identical=True)),))]


_batch_decl(_batch_gemm_pinned, "gemm-pinned")
_batch_decl("n/a:profile lane; the products are hashed whole", "gemm-bf16", "gemm-int8")
# lane/linalg-public (2026-09-19). A factorization is a GLOBAL REDUCTION over
# every row of its input: the Householder QR folds each column's norm across
# all n_rows, and the Jacobi sweeps run to convergence on the whole matrix.
# Splitting the rows does not give a batch of the same call, it gives a
# DIFFERENT MATRIX with a different factorization, and there is no per-row
# output to hold against a whole-batch one. `eigh` has no row axis at all --
# its input is n x n and both axes are the same object.
_batch_decl("n/a:whole-matrix-decomposition (a factorization reduces over every row; "
            "splitting the rows gives a different matrix, not a batch of the same call, "
            "and there is no per-row output)",
            "linalg-qr", "linalg-eigh", "linalg-svdvals")
_batch_decl("n/a:weight-format lane; the batch part is measured on its base lane", "mlp-bf16w", "mlp-int8w", "samba-bf16w", "samba-int8w")
_batch_decl(_batch_gemm_transposed, "gemm-transposed")
# The function, tokenizer, optimizer and byte LM trainer lanes
# (lane/cpu-training-batch-fill-func, 2026-09-15). Each lane was read against
# its public API for an axis whose elements may not read their neighbors; the
# ones that have one declare it below, the rest name the missing method.

#: the batch of global replicate, fold or parameter-row indices these parts ask
BATCH_RANGE_ROWS = 64


def _range_rows(label, n, fn, min_batch=1, refusal=None):
    """A _BatchRows over the global indices [0, n) of a call whose `first`
    handle addresses a contiguous range (resample's r_first). The harness
    only ever hands contiguous windows R[a:b], so a window is the call
    `fn(first=a, count=b - a)`; anything else is refused, not re-indexed."""
    idx = np.arange(n, dtype=np.int64).reshape(n, 1)

    def call(r):
        first, count = int(r[0, 0]), int(r.shape[0])
        if not np.array_equal(r[:, 0], np.arange(first, first + count, dtype=np.int64)):
            raise ValueError(f"{label}: a range call needs a contiguous window, got {r[:, 0].tolist()}")
        return fn(first, count)
    return _BatchRows(label, idx, call, min_batch, refusal)


#: bootstrap refuses ONE replicate by name, because its standard error is the
#: ddof=1 deviation of the distribution (resample/checks/intervals.mojo); its
#: replicates are asked in windows of two, as the coordinate descent predict is
BOOTSTRAP_ONE_REFUSAL = "the standard error needs at least 2 resamples"


def _batch_resample_inputs(Xh):
    """The bootstrap and permutation-test lanes' inputs, derived from the
    held-out rows by the same rules (labels_for's targets): yr's first 4096
    values, yr paired with column 3, and the two label groups' first 512
    rows (the lane's fallback when a group is short). Column 3 alone is
    constant over the first 1024 denormal rows, where pearson refuses by
    name; the lane's pairing is not."""
    ych, yrh = labels_for(Xh, HELDOUT_SEED)
    x = np.ascontiguousarray(yrh[:4096]).astype(np.float32)
    two = np.ascontiguousarray(np.stack([yrh[:4096], Xh[:4096, 3]], 1).astype(np.float32))
    a = np.ascontiguousarray(yrh[:4096][ych[:4096] == 0][:512])
    b = np.ascontiguousarray(yrh[:4096][ych[:4096] == 1][:512])
    if a.size < 8 or b.size < 8:
        a, b = np.ascontiguousarray(yrh[:512]), np.ascontiguousarray(yrh[512:1024])
    return x, two, a, b


#: bootstrap statistics asked by the batch part: every STATISTICS arm, the
#: two sorted ones (quantile, trimmed_mean) and the two paired ones included
BATCH_BOOTSTRAP_ARMS = (("mean", dict(statistic="mean")), ("std", dict(statistic="std")),
                        ("quantile", dict(statistic="quantile", q_or_prop=0.25)),
                        ("trimmed_mean", dict(statistic="trimmed_mean", q_or_prop=0.1)),
                        ("pearson", dict(statistic="pearson")), ("diff_means", dict(statistic="diff_means")))


def _batch_resample(bootstrap, permutation_test, prefix=""):
    """resample's own batch-invariance handle (python/mojolearn/resample.py
    module docstring: "replicates [r_first, r_first + n_resamples) of one run
    are bit-identical to the corresponding slice of a whole run"): replicate
    r of `distribution` and permutation r of `null_distribution` alone
    (n_resamples=1, r_first=r) against 64 of them and the split 1, 7, 56.
    The point estimate, interval, standard error and p-value are reductions
    over the run and are not asked."""
    def spec(ml, e, Xh):
        x, two, a, b = _batch_resample_inputs(Xh)
        calls = []
        for name, kw in BATCH_BOOTSTRAP_ARMS:
            data = two if kw["statistic"] in ("pearson", "diff_means") else x
            calls.append(_range_rows(
                f"{prefix}bootstrap {name} distribution", BATCH_RANGE_ROWS,
                lambda first, count, data=data, kw=kw: (np.asarray(bootstrap(
                    data, n_resamples=count, r_first=first, random_state=3, **kw).distribution),),
                min_batch=2, refusal=BOOTSTRAP_ONE_REFUSAL))
        calls.append(_range_rows(
            f"{prefix}permutation_test diff_means null_distribution", BATCH_RANGE_ROWS,
            lambda first, count: (np.asarray(permutation_test(
                a, b, statistic="diff_means", n_resamples=count, r_first=first, random_state=3).null_distribution),)))
        return calls
    return spec


def _batch_bootstrap_lane(ml, e, Xh):
    return _batch_resample(ml.resample.bootstrap, ml.resample.permutation_test)(ml, e, Xh)[:-1]


def _batch_permutation_lane(ml, e, Xh):
    return _batch_resample(ml.resample.bootstrap, ml.resample.permutation_test)(ml, e, Xh)[-1:]


def _batch_par_resample(ml, e, Xh):
    """The parallel driver through the same r_first handle on _par_devices()
    (parallel_classical.bootstrap: "Replicate r is a pure function of (seed,
    r, data) (the r_first handle)"); the mean and quantile arms the lane
    fits, and the permutation null."""
    from mojolearn import parallel_classical as pc
    dev = _par_devices()
    calls = _batch_resample(lambda d, **kw: pc.bootstrap(d, devices=dev, **kw),
                            lambda a, b, **kw: pc.permutation_test(a, b, devices=dev, **kw),
                            prefix="parallel ")(ml, e, Xh)
    return [c for c in calls if " mean " in c.label or " quantile " in c.label or "permutation" in c.label]


def _batch_silhouette_chunks(ml, e, Xh):
    """silhouette_samples' batch is cuML's chunk, not a row subset: a
    sample's coefficient reads every other row, so rows alone would be a
    different input. The docstring (_metrics_impl.py::silhouette_score,
    chunksize) says chunk sizes are "gated to one byte pattern", and both the
    GPU binding and metrics/host/metrics_oracle.mojo take the chunk. A window
    [a, b) is therefore the whole sample scored at chunksize = b - a with rows
    [a, b) read back: row i at chunksize 1 and at 1, 7 and 248 against 256."""
    X4 = np.ascontiguousarray(Xh[:256, :4]).astype(np.float32)
    labels = (np.arange(256) % 4).astype(np.int32)
    idx = np.arange(256, dtype=np.int64).reshape(256, 1)

    def fn(r):
        a, b = int(r[0, 0]), int(r[-1, 0]) + 1
        s = np.asarray(ml.metrics.silhouette_samples(X4, labels, chunksize=b - a))
        return (np.ascontiguousarray(s[a:b]),)
    return [_BatchRows("silhouette_samples (chunksize = window rows)", idx, fn)]


def _batch_cross_val(ml, e, Xh):
    """cross_val_score's batch is its folds: a fresh clone per fold
    (model_selection.py, "fitting a fresh clone serially"), so fold k's
    score may not read the other folds. Three explicit unshuffled folds of
    768 held-out rows (column 0 the target, columns 1.. the features), passed
    as the index pairs `cv` accepts; each fold alone against the three."""
    Xf = np.ascontiguousarray(Xh[:768, 1:]).astype(np.float32)
    y = np.ascontiguousarray(Xh[:768, 0]).astype(np.float32)
    rows = np.arange(768)
    folds = [(np.ascontiguousarray(np.concatenate([rows[:256 * k], rows[256 * (k + 1):]])),
              np.ascontiguousarray(rows[256 * k:256 * (k + 1)])) for k in range(3)]
    idx = np.arange(3, dtype=np.int64).reshape(3, 1)
    return [_BatchRows("cross_val_score (folds)", idx, lambda r: (np.asarray(ml.model_selection.cross_val_score(
        _gbdt(ml.GradientBoostingRegressor, n_estimators=8, max_depth=4), Xf, y,
        cv=[folds[int(k)] for k in r[:, 0]]), dtype=np.float64),))]


def _batch_optim_rows(label, seed, make, steps):
    """An optimizer over ONE (64, 8) parameter tensor: rows of the tensor
    are the batch (the optimizer contract's parameter-count invariance clause
    is stated for max_norm=None, _training_impl.py::_Optimizer.step, so no
    call here clips; clip_grad_norm_ and max_norm are a function of every
    gradient in the registry by contract 3.5 and have no such axis). A
    window of rows is a fresh optimizer over those rows' parameters, stepped
    with those rows' gradients; the parameters and both moment buffers after
    the steps are compared row by row."""
    P = _hw((BATCH_RANGE_ROWS, 8), f"batch:{seed}:p", -0.5, 0.5)
    G = [_hw((BATCH_RANGE_ROWS, 8), f"batch:{seed}:g{j}", -1.0, 1.0) for j in range(4)]
    idx = np.arange(BATCH_RANGE_ROWS, dtype=np.int64).reshape(BATCH_RANGE_ROWS, 1)

    def fn(r):
        k = r[:, 0]
        p = np.ascontiguousarray(P[k])
        o = make(p)
        steps(o, [np.ascontiguousarray(g[k]) for g in G])
        n = p.shape[0]
        return (p, np.asarray(o.exp_avg).reshape(n, 8), np.asarray(o.exp_avg_sq).reshape(n, 8))
    return _BatchRows(label, idx, fn)


def _two_steps(o, g):
    o.step([g[0]])
    o.step([g[1]])


def _batch_optim_sgd(ml, e, Xh):
    """The optim-sgd lane's two SGD arms without the clip."""
    T = ml.training
    return [_batch_optim_rows("SGD nesterov weight_decay ConstantLR warmup", "sgd-nesterov",
                              lambda p: T.SGD([p], lr=1e-2, momentum=0.9, dampening=0.0, nesterov=True,
                                              weight_decay=0.01, lr_schedule=T.ConstantLR(1e-2, warmup_steps=2)),
                              _two_steps),
            _batch_optim_rows("SGD dampening", "sgd-dampening",
                              lambda p: T.SGD([p], lr=1e-2, momentum=0.9, dampening=0.5, nesterov=False),
                              _two_steps)]


def _batch_optim_adam(ml, e, Xh):
    """The optim-adam-clip lane's Adam and accumulated AdamW arms without
    the clip (T = 256 at A = 2, the lane's split)."""
    T = ml.training
    return [_batch_optim_rows("Adam weight_decay WarmupLinearLR", "adam",
                              lambda p: T.Adam([p], lr=1e-3, weight_decay=0.01,
                                               lr_schedule=T.WarmupLinearLR(1e-3, warmup_steps=2, total_steps=8,
                                                                            min_lr=0.0)),
                              _two_steps),
            _batch_optim_rows("AdamW step_accumulated A=2", "adamw0",
                              lambda p: T.AdamW([p], lr=1e-3, weight_decay=0.0, accumulation_steps=2),
                              lambda o, g: (o.step_accumulated([[g[0]], [g[1]]], 256),
                                            o.step_accumulated([[g[2]], [g[3]]], 256)))]


def _batch_saved_model_host(ml, e, Xh):
    """The saved-model host inference part: the LOADED HostForest's two
    prediction entries over held-out rows. A row's answer must not depend on
    how many other rows were in the call -- the forest host binding walks
    trees row by row and writes one output slab, so a row-block bug in that
    walk shows up here and nowhere else in this lane. Only the `est` model
    (the RandomForestClassifier archive) is reachable from a declaration;
    the GBDT and groves models are the lane body's."""
    R = np.ascontiguousarray(Xh[:64])
    return [_BatchRows("HostForest.predict", R, lambda r: (np.asarray(e.predict(r)),)),
            _BatchRows("HostForest.predict_proba", R, lambda r: (np.asarray(e.predict_proba(r)),))]


def _batch_lowbit_conversions(ml, e, Xh):
    """The low-bit part: `pack_one` and `materialize_one` over held-out
    rows. bf16 is elementwise, so its rows are trivially separable; INT8 IS
    NOT ELEMENTWISE -- the exponent is per row (contract L-3) -- and this is
    the part that says the row's exponent is a function of that row and not
    of the block it was packed with. The outputs' leading axis is the rows:
    the bf16 bits, the int8 codes, the per-row exponents and the two
    materialized float32 blocks."""
    R = np.ascontiguousarray(Xh[:64, :LOWBIT_COLS]).astype(np.float32)

    def fn(r):
        bf = ml.lowbit.pack_one(r, "bfloat16")
        q8 = ml.lowbit.pack_one(r, "int8")
        return (np.asarray(bf.bits), np.asarray(q8.codes), np.asarray(q8.exponents),
                np.asarray(ml.lowbit.materialize_one(bf)), np.asarray(ml.lowbit.materialize_one(q8)))

    return [_BatchRows("pack_one + materialize_one", R, fn)]


_batch_decl(_batch_saved_model_host, "saved-model-host-infer")
_batch_decl(_batch_lowbit_conversions, "lowbit-conversions")
# `grad-accumulation` HAS no batch axis to vary: the A microbatches ARE the
# split whose arithmetic the lane hashes, exactly as the optimizer lanes'
# step is (the `optimizer-step` and `training-step` reasons above). Asking
# the same call with fewer microbatches is a DIFFERENT accumulation, not the
# same answer in a smaller batch.
_batch_decl("n/a:accumulation-step (the A microbatches are the arithmetic the lane hashes; a call with "
            "fewer of them is a different accumulation, not a smaller batch of the same one)",
            "grad-accumulation")
_batch_decl("n/a:scalar-reduction (accuracy_score, adjusted_rand_score, v_measure_score, r2_score and "
            "silhouette_score each return one float over every row, python/mojolearn/_metrics_impl.py; the "
            "per-sample silhouette_samples is asked on metrics-classification)", "metrics")
_batch_decl(_batch_silhouette_chunks, "metrics-classification")
_batch_decl("n/a:global-contingency-reduction (three scalar scores over all labels; no per-row output)",
            "metrics-homogeneity-completeness")
_batch_decl("n/a:corpus-global-vocabulary-training (pair counts depend on the complete corpus; no per-row output)",
            "bpe-trainer", "bpe-vocabulary", "tokenized-corpus")
_batch_decl(_batch_cross_val, "cross-val")
# The fold PARTITION is metadata, not an inference call: the lane fits nothing
# and asks nothing for a row's answer, so there is no batch axis to vary
# (lane/data-ordering-determinism, 2026-09-16).
_batch_decl("n/a:function", "cross-val-folds")
# ---- lane/unlaned-public-algorithms (2026-09-20), the six laneless entries.
#
# THE DECODE SESSIONS HAVE NO BATCH AXIS TO VARY, and this needed thinking
# about rather than defaulting to the `predict`-shaped declaration every
# estimator gets. The batch part's question is "does a row's answer depend on
# which other rows were in the call". A resident decode session is opened FOR
# a fixed batch: `state.batch_size` sizes the device buffers the session
# builds once, and `step` refuses any other row count BY NAME
# (`the session holds N rows, x has B = M`). Asking two of the session's rows
# alone is therefore not the same call on a smaller batch, it is a DIFFERENT
# SESSION on a different state, whose answer belongs to the per-call `step`
# that `mamba1` and `transformer` already declare and that this lane's own
# cell holds the session to byte for byte. `batchscale` is the same fact at
# serving scale and is left at its default for the same reason.
_batch_decl("n/a:fixed-session-batch (a decode session is opened for state.batch_size and its step refuses any "
            "other row count by name, so a row alone is a different session and not a smaller batch of this "
            "call; the per-call step the session is held to carries the batch part on the mamba1 and "
            "transformer lanes)",
            "mamba1-decode-session", "transformer-decode-session")
# The GPC drivers DO have a row axis: `predict_gaussian_process_classifier`
# takes query rows and the merge is per row, so the part asks it exactly as
# the `gpc` lanes ask the plain estimator. The FIT driver has none -- its
# shards are classes, not rows -- so it is declared on the fitted estimator's
# prediction, which is what its infer column hashes.
_batch_decl(_rows_calls("predict", "predict_proba", sl=(slice(0, 64), slice(0, 4))),
            "par-gpc-fit", "par-gpc-predict")
# `par-cross-val`'s answer is one score per FOLD, not per row: a fold is a
# partition of the whole data set and a smaller call is a different
# cross-validation, exactly as `cross-val-folds` says of the partition.
_batch_decl("n/a:fold-reduction (one score per fold over that fold's whole test block; a call with fewer rows "
            "is a different cross-validation, not the same one in a smaller batch)", "par-cross-val")
# `language-model-config` (lane/laneless-public-classes, 2026-09-19) is a
# FROZEN DATACLASS. `parameter_shapes`, `offsets`, `n_total` and `profile`
# are functions of the CONFIG, not of any input rows -- the lane passes no
# data at all -- so there is no batch axis to hold a whole-batch answer
# against, and no smaller batch to ask for.
_batch_decl("n/a:no-input-rows (a frozen config object; its shapes and offsets are functions of the config, and the lane passes no data)", "language-model-config")
_batch_decl(_batch_bootstrap_lane, "bootstrap")
_batch_decl(_batch_permutation_lane, "permutation-test")
_batch_decl("n/a:scalar-fold (resample.monte_carlo_integrate returns only integral, mean, volume and closed_form "
            "folded over [i_first, i_first + n_samples) by the pinned chunk tree, python/mojolearn/resample.py:196; "
            "no per-sample output exists to compare an i_first range against)", "monte-carlo")
def _batch_tokenizer(ml, e, Xh):
    """BpeTokenizer.encode_batch and decode_bytes_batch
    (lane/inference-tokenizer-neural, 2026-09-15). A row of the held-out
    slice is one document: its 8 * d float64 bytes, so a batch is 64
    documents of binary text with every byte value in reach. Each
    document's ids in the whole batch must be its ids alone and in any
    split; the decode call reads the whole batch's ids back.
    MOJOLEARN_TOKENIZER_BATCH_SABOTAGE=1 in the binding swaps ids across
    document boundaries and must read BATCH_MOVED."""
    R = np.ascontiguousarray(Xh[:64])
    ids_rows = [np.asarray(i, dtype=np.int32)
                for i in e.encode_batch([r.tobytes() for r in R], allow_endoftext=True)]

    def enc(r):
        return (_PerRow(np.asarray(i, dtype=np.int32)
                        for i in e.encode_batch([row.tobytes() for row in r], allow_endoftext=True)),)

    def dec(r):
        return (_PerRow(np.frombuffer(b, dtype=np.uint8)
                        for b in e.decode_bytes_batch([ids_rows[int(k)].tolist() for k in r[:, 0]])),)

    return [_BatchRows("encode_batch", R, enc),
            _BatchRows("decode_bytes_batch", np.arange(len(R), dtype=np.int64).reshape(-1, 1), dec)]


_batch_decl(_batch_tokenizer, "tokenizer")


# ------------------------------------------- the Hugging Face loading path
# `hf-checkpoint` reads a file and plans a shape: no fitted model, no rows
# in and no per-row output, so there is no batch axis to vary.
_batch_decl("n/a:function (a header parse, a memoryview cast and the option matrix; the lane fits nothing and "
            "asks nothing for a row's answer)", "hf-checkpoint")
# `models.Tokenizer` has ONE document per call: `encode`, `encode_bytes`,
# `decode` and `decode_bytes` all take a single text (python/mojolearn/
# models/tokenizer.py). `BpeTokenizer.encode_batch` -- the entry the
# `tokenizer` lane's part uses -- has no counterpart here, and the GPT-2
# door this class opens is built per instance from the token bytes, so a
# batch call would be the harness's own loop and not a surface the class
# offers. The day `Tokenizer` grows a batch entry this becomes a part.
_batch_decl("n/a:no-batch-axis (mojolearn.models.Tokenizer encodes and decodes one document per call; it has no "
            "encode_batch, and a loop written here would test the harness, not the class)", "hf-tokenizer")


def _batch_hf_causal_lm(ml, e, Xh):
    """CausalLM.forward from a zero state: each sequence alone against the
    whole batch of 4, and every prefix length against L = 12. The API takes
    no padding and no ragged batch, so the batch is same-length sequences,
    as the Samba and byte LM parts' is. `e` is the lane's Llama model, the
    first of HF_ARCHITECTURES; the other seven are hashed in `every` and
    are not asked again here, because a batch part per architecture would
    cost eight times the calls to test one splitting rule."""
    ids = _ids(Xh, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)

    def fwd(rows):
        return (np.asarray(e.forward(ml.Array.from_buffer(np.ascontiguousarray(rows)))),)

    return [_BatchRows("forward", ids, fwd),
            _BatchPrefix("forward", HF_LM_LEN, lambda p: fwd(ids[:, :p]), axis=1)]


_batch_decl(_batch_hf_causal_lm, "hf-causal-lm")
# `par-causal-lm` is `hf-causal-lm`'s Llama row with the layers owned by
# separate worker processes, so its batch axis is the same one: the layer
# boundary must not make a row's answer depend on its neighbours. The lane's
# `est` is None (it returns no estimator), so the declaration rebuilds the
# model the lane built; it runs only on the CUDA/HIP columns where the lane
# can run at all (GPU_ONLY_LANES).
def _batch_par_causal_lm(ml, e, Xh):
    from mojolearn import _causal_lm_fixtures as fx
    from mojolearn.models import ParallelCausalLM
    ids = _ids(Xh, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)
    dev = _par_devices()
    cfg, tensors = fx.family_fixture("llama", False)
    with tempfile.TemporaryDirectory(prefix="ib-par-causal-lm-batch-") as d:
        root = fx._write_checkpoint(os.path.join(d, "llama"), cfg, tensors)
        n_layers = ml.models.causal_lm.CausalLM.load(root).plan.n_layers
        with ParallelCausalLM.load(
                root, layer_devices=tuple(dev[i % len(dev)] for i in range(n_layers))) as par:

            def fwd(rows):
                return (np.asarray(par.forward(ml.Array.from_buffer(np.ascontiguousarray(rows)))),)

            return [_BatchRows("forward", ids, fwd),
                    _BatchPrefix("forward", HF_LM_LEN, lambda p: fwd(ids[:, :p]), axis=1)]


_batch_decl(_batch_par_causal_lm, "par-causal-lm")
_batch_decl(_batch_optim_sgd, "optim-sgd")
_batch_decl(_batch_optim_adam, "optim-adam-clip")
_batch_decl("n/a:mean-reduction-fixed-batch (LanguageModelHostTrainer has train_step and loss only, and loss IS a "
            "step, python/mojolearn/_byte_lm_host.py:392-429; one loss and one update from mean cross-entropy "
            "over ids of the profile's fixed (batch, length + 1), so no output belongs to one sequence; the "
            "per-sequence logits are asked by byte-lm-host-infer; lane/batch-invariance-2's batchgrad records the "
            "same reason for this lane and its batchscale and ragged parts do not ask it)", "byte-lm-host-train")
#: the three multi-GPU byte LM trainers: the same reason, one string
PAR_BYTE_LM_BATCH_NA = ("n/a:driver-step (ParallelByteLanguageModelTrainer and its Pooled and Offloaded subclasses "
                        "have train_step, state_dict, export_gradients and checkpoint only, "
                        "python/mojolearn/parallel_training.py:89-172; train_step returns per-shard mean losses "
                        "and one update from the shard-mean gradient, so no output belongs to one sequence, and "
                        "the shard split is held to the replica trainer by the train column)")
_batch_decl(PAR_BYTE_LM_BATCH_NA, "par-byte-lm")
# metrics.fowlkes_mallows_score (lane/cpu-training-small-gaps) keeps the reason main gave it
_batch_decl("n/a:function", "metrics-fowlkes-mallows")
_batch_decl("n/a:scalar-reduction (score(X, y, sample_weight) returns one float over every row it is "
            "handed, python/mojolearn/_gbdt_adapters.py and _forest_protocol.py; the predict and "
            "predict_proba it wraps keep their batch parts on gbdt-adapter-clf, gbdt-adapter-reg, "
            "rf-clf and rf-reg)", "gbdt-adapter-score-weighted", "rf-score-weighted")


def _batch_cross_entropy(ml, e, Xh):
    """reduction='none' is per row (the loss contract says N is a launch
    shape for everything except L12). Rows are indices into one logits slab and
    one target vector, so a chunk is handed exactly its rows' bytes."""
    T = ml.training
    logits = _hw((64, 16), "ce:logits", -2.0, 2.0)
    t = (_ids(Xh, 1, 64).reshape(64) % 16).astype(np.int32)
    idx = np.arange(64, dtype=np.int64).reshape(64, 1)
    return [_BatchRows("cross_entropy(reduction='none')", idx,
                       lambda r: (np.asarray(T.cross_entropy(np.ascontiguousarray(logits[r[:, 0]]),
                                                             np.ascontiguousarray(t[r[:, 0]]), reduction="none")),))]


def _batch_training_primitives(ml, e, Xh):
    T = ml.training
    V, D, N, K = 64, 32, 64, 16
    w_emb = _hw((V, D), "prim:emb", -0.25, 0.25)
    g = np.ascontiguousarray(np.abs(_hw((D,), "prim:rms", 0.5, 1.5)))
    w_lin = _hw((K, D), "prim:lin", -0.25, 0.25)
    xh = np.ascontiguousarray(_seq(Xh, 1, N, D).reshape(N, D))
    idh = (_ids(Xh, 1, N).reshape(N) % V).astype(np.int32)
    idx = np.arange(N, dtype=np.int64).reshape(N, 1)
    take = lambda a, r: np.ascontiguousarray(a[r[:, 0]])
    return [_BatchRows("embedding_forward", idx, lambda r: (np.asarray(T.embedding_forward(w_emb, take(idh, r))),)),
            _BatchRows("rms_norm_forward", idx, lambda r: (np.asarray(T.rms_norm_forward(take(xh, r), g, 1e-5)),)),
            _BatchRows("linear_forward", idx, lambda r: (np.asarray(T.linear_forward(take(xh, r), w_lin)),))]


def _batch_ivf(ml, e, Xh):
    """A search is row-wise in its queries (policy 3 rebuilds the same index
    from the same fit rows on every call), so each query alone against 64."""
    return [_BatchRows("search", Xh[:64], lambda r: e.search(r) + (e.n_candidates_,))]


def _batch_embedding(ml, e, Xh):
    """The gather is row-wise in the ids; each id alone against 64."""
    idh = (_ids(Xh, 1, 64).reshape(64) % e.num_embeddings).astype(np.int32)
    idx = np.arange(64, dtype=np.int64).reshape(64, 1)
    return [_BatchRows("forward", idx, lambda r: (np.asarray(e.forward(np.ascontiguousarray(idh[r[:, 0]]))),))]


# `par-ivf` (lane/laneless-public-classes, 2026-09-19) runs the SAME query
# through the distributed driver, so it takes the SAME batch part as its
# base lane -- the pattern every other `par-*` lane follows (par-iforest
# with iforest, par-svm with svc, par-gp with gp).
_batch_decl(_batch_ivf, "ivf", "ivf-euclidean", "par-ivf")


def _batch_ivf_extend(ml, e, Xh):
    """extend is row-wise in the new rows' LISTS: a row's list comes from the
    fixed centres alone, so each held-out row extended alone, and in pieces,
    must name the list it names inside the whole batch of 64. Every call
    extends a fresh clone of the fitted index; the ids are not compared,
    because a row's id is its position in the extension by contract."""
    return [_BatchRows("extend (new-row lists)", Xh[:64],
                       lambda r: (np.asarray(e._clone().extend(r).extend_labels_),))]


_batch_decl(_batch_ivf_extend, "ivf-extend")
_batch_decl(_batch_embedding, "embedding", "embedding-sort")
_batch_decl(_batch_cross_entropy, "cross-entropy-arms")
def _batch_select_d(ml, e, Xh):
    # A sample here is a time series: split series columns, never time steps.
    series_rows = np.ascontiguousarray(_select_d_series(Xh[:256, :8]).T)
    return [_BatchRows(f"select_d D={D}", series_rows,
                      lambda rows, D=D: (np.asarray(ml.select_d(np.ascontiguousarray(rows.T),
                          D=D, s=12 if D else 0, d_max=2-D)),)) for D in (0, 1)]


_batch_decl(_batch_select_d, "select-d")

_batch_decl(_batch_training_primitives, "training-primitives")
_batch_decl("n/a:ordered-shard-reduction (the specified shard order defines the sum; no per-row prediction)",
            "ordered-gradient-sum")
_batch_decl(lambda ml, e, Xh: _rows_calls("predict_logits", sl=(slice(0, 256), slice(0, 8)))(
    ml, _neural_inference(ml, "mlp", e), Xh), "mlp")
_batch_decl(_rows_calls("predict_logits", sl=(slice(0, 256), slice(0, 8))), "par-mlp")

#: the sequence models' held-out batch: 8 sequences of 16, so eight rows alone
SEQ_BATCH, SEQ_LEN = 8, 16


def _batch_byte_lm(ml, e, Xh):
    """logits `[B, L, vocab]`, each sequence alone against 16 of them, and
    every prefix length against the whole length (the model is causal and
    prefills from position 0, so position t may not read the length)."""
    shape = _byte_lm_shape(ml, e)
    ids = _ids(Xh, 16, shape.length)
    return [_BatchRows("logits", ids, lambda r: (np.asarray(e.logits(r)),)),
            _BatchPrefix("logits", shape.length, lambda p: (np.asarray(e.logits(np.ascontiguousarray(ids[:, :p]))),),
                         axis=1)]


_batch_decl(_batch_byte_lm, "byte-lm", "byte-lm-resident", "byte-lm-host-infer", "byte-lm-host-infer-threaded")


def _batch_block(ml, e, Xh):
    """forward `(B, L, d_model)` from a zero state, each sequence alone
    against 8, and every prefix length against L = 16. The API has no
    padding or ragged batch, so the batch is same-length sequences."""
    x = _seq(Xh, SEQ_BATCH, SEQ_LEN, 32)
    return [_BatchRows("forward", x, lambda r: (np.asarray(e.forward(r)),)),
            _BatchPrefix("forward", SEQ_LEN, lambda p: (np.asarray(e.forward(np.ascontiguousarray(x[:, :p, :]))),),
                         axis=1)]


_batch_decl(_batch_block, "mamba1", "mamba2", "mamba3", "mamba2-dtlimit",
            "mamba1-bf16w", "mamba1-int8w", "mamba2-bf16w", "mamba2-int8w", "mamba3-bf16w", "mamba3-int8w")
_batch_decl(lambda ml, e, Xh: _batch_block(ml, _neural_inference(ml, "transformer", e), Xh),
            "transformer", "transformer-window", "transformer-bf16w", "transformer-int8w")


def _batch_samba(ml, e, Xh):
    ids = _ids(Xh, SEQ_BATCH, SEQ_LEN)
    return [_BatchRows("forward", ids, lambda r: (np.asarray(e.forward(r)),)),
            _BatchPrefix("forward", SEQ_LEN, lambda p: (np.asarray(e.forward(np.ascontiguousarray(ids[:, :p]))),),
                         axis=1)]


_batch_decl(_batch_samba, "samba", "samba-untied-dropout-accum", "par-samba", "par-samba-clip")

# The 2026-09-14 night par-* lanes. The query drivers ARE inference, so their
# batch part goes through the driver: one driver is opened per part and
# reused by every alone, split and whole call (a pool per call would start
# twenty worker processes per call), then closed by _probe_batch.
_BATCH_CLOSE = []


def _batch_pq(*methods, sl, ragged=False, **kw):
    def spec(ml, e, Xh):
        from mojolearn.parallel_neighbors import ParallelQueries
        pq = ParallelQueries(e, devices=_par_devices(), rows_per_shard=PAR_QUERY_ROWS)
        _BATCH_CLOSE.append(pq.close)
        R = Xh[sl]
        calls = []
        for meth in methods:
            if ragged:
                def fn(r, meth=meth):
                    dists, idx = pq.query(r, method=meth, **kw)
                    n = len(idx)
                    return (_PerRow(np.asarray(dists[i], dtype=np.float32).reshape(-1) for i in range(n)),
                            _PerRow(np.asarray(idx[i]).reshape(-1).astype(np.int64) for i in range(n)))
            else:
                def fn(r, meth=meth):
                    out = pq.query(r, method=meth, **kw)
                    return tuple(out) if isinstance(out, tuple) else (out,)
            calls.append(_BatchRows(f"ParallelQueries {meth}", R, fn))
        return calls
    return spec


def _batch_rsn(*methods):
    def spec(ml, e, Xh):
        rs = _rsn(e)
        _BATCH_CLOSE.append(rs.close)
        calls = []
        for meth in methods:
            def fn(r, meth=meth):
                out = getattr(rs, meth)(r)
                return tuple(out) if isinstance(out, tuple) else (out,)
            calls.append(_BatchRows(f"ReferenceShardedNeighbors {meth}", Xh[:64], fn))
        return calls
    return spec


_batch_decl(_batch_pq("kneighbors", "predict", "predict_proba", sl=slice(0, 64)), "par-queries-knn")
_batch_decl(_batch_pq("kneighbors", sl=slice(0, 64)), "par-queries-nn")
_batch_decl(_batch_pq("radius_neighbors", sl=slice(0, 64), ragged=True, sort_results=True), "par-queries-radius")
_batch_decl(_batch_pq("score_samples", sl=(slice(0, 256), slice(0, 4))), "par-queries-kde")
_batch_decl(_batch_rsn("kneighbors", "predict", "predict_proba"), "par-reference-knn")
_batch_decl(_batch_rsn("predict"), "par-reference-knn-reg")
# parallel_graph.fit_graph fits; the fitted AgglomerativeClustering(prediction_data=True)
# predicts like the plain lane's (DEVIATION 2740)
_batch_decl(_rows_calls("predict", sl=np.s_[:64, :4]), "par-graph-agglomerative")
# par-graph-spectral fits without prediction_data; its train cells hold the
# fit_graph labels and embedding to the plain fit's, and it asks no predict
_batch_decl("n/a:transductive (the par-graph-spectral lane fits without prediction_data; "
            "SpectralClustering.predict is the spectral lane's)", "par-graph-spectral")
_batch_decl(_rows_calls("predict"), "par-ordered-rmse")
_batch_decl(_rows_calls("predict", prep=_coded), "par-feature-freq")
_batch_decl(_rows_calls("predict", "predict_proba"), "par-boosting-pointwise")
_batch_decl(lambda ml, e, Xh: [_BatchPrefix("forecast", FORECAST_HORIZON, lambda h: (e.forecast(h),), axis=0)],
            "par-holtwinters")
_batch_decl(PAR_BYTE_LM_BATCH_NA, "par-byte-lm-model-pool", "par-byte-lm-offload")
_batch_decl(_rows_calls("predict", "predict_proba"), "par-forest-pool")
_batch_decl(_rows_calls("score_samples", "predict", sl=(slice(0, 64), slice(0, 4))), "par-gmm")
_batch_decl(_batch_par_resample, "par-resample")
# parallel_classical.fit_hdbscan copies the worker's fitted estimator back, so a
# prediction_data=True fit carries the prediction data and the root answers
# approximate_predict; its two-device cells are owed to the release record.
_batch_decl(_batch_hdbscan, "par-hdbscan")
_batch_decl(_rows_calls("predict", sl=(slice(0, 64), slice(0, 4))), "par-kernel-ridge")
_batch_decl(_rows_calls("transform", sl=(slice(0, 64), slice(0, 4))), "par-nystroem")
_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "par-rbf-sampler")
def _batch_par_cholesky(ml, e, Xh):
    """_batch_cholesky at the par-cholesky lane's 600-row factor."""
    B = np.ascontiguousarray(Xh[:600, :2])
    return [_BatchRows("solve (right-hand sides)", B.T,
                       lambda r: (np.asarray(e.solve(np.ascontiguousarray(r.T))).reshape(B.shape[0], -1).T,))]


_batch_decl(_batch_par_cholesky, "par-cholesky")


# ---------------------------------------------------------------- the batchgrad part (2026-09-15)
# See `batchgrad` in the module docstring. The backward twin of the batch
# part, opt-in (`--batch-grad`), declared OUTSIDE the lane bodies like BATCH
# so no train, infer, model or batch hash moves because of it. A declaration
# is an `n/a:<reason>` string or a function `(ml, est, Xh) -> [call, ...]`
# whose calls are _BatchRows (a per-row gradient: every split is claimed),
# _GradAccum (a gradient summed over rows: only clause 9.2's aligned splits
# are claimed) or _GradCarry (the embedding carry: every split is claimed).

#: the batchgrad sabotage: `1` (or `bit`) flips one bit of every whole-batch
#: gradient, so every hashed cell MUST read BATCH_MOVED; `serial` replaces
#: clause 9.2 condition 5's balanced tree with a running pair sum, which the
#: contract says is inert at A <= 2 and fixture-dependent at A = 4, so its
#: result is recorded and not required
BATCHGRAD_SABOTAGE_ENV = "MOJOLEARN_IDENTITY_BATCHGRAD_SABOTAGE"

#: the microbatch counts asked of every accumulated gradient; the ones clause
#: 9.2 refuses at the call's token count are recorded n/a with that reason
GRAD_ACCUM_SPLITS = (2, 4, 8)

#: the reason the batch part's uneven split (1, 7, rest) is never asked of an
#: accumulated gradient, from training/IDENTICAL_OPTIMIZER_CONTRACT.md
GRAD_UNEVEN_SPLIT_NA = ("n/a:split 1,7,rest (optimizer contract clause 9.2: three unequal pieces, "
                        "A = 3 is not a power of two (condition 4) and the pieces are not complete "
                        "subtrees (condition 3))")

BATCHGRAD = {}


def _batchgrad_decl(spec, *names):
    for n in names:
        if n in BATCHGRAD:
            raise RuntimeError(f"identity_break: lane {n!r} has two batchgrad declarations")
        BATCHGRAD[n] = spec


class _GradAccum:
    """One gradient that is a SUM over the rows of a call (a weight
    gradient). `fn(a, b)` returns `{name: array}` for rows [a, b) of the
    call's input; `n` rows of `tokens_per_row` tokens make the call's token
    count T. The harness computes the whole-batch gradient, then for every A
    in GRAD_ACCUM_SPLITS asks `training.accumulation_is_aligned(T, A)`; an
    aligned split is computed as A equal row pieces combined by
    `training.accumulate_grads(pieces, tokens=T)` (the balanced tree,
    condition 5) and every tensor in `claimed` must equal the whole batch
    byte for byte. A tensor NOT in `claimed` is REPORTED: its contraction is
    not the v1 GEMM over the token count (a serial fold, the run-sorted
    embedding fold), so clause 9.2 does not reach it; whether it moved is
    part of the hash (the same arithmetic on every vendor gives the same
    pattern) and never a failure. A split the clause refuses is the negative
    control: it is combined with `tokens=None` (the tree with no alignment
    claim) and the number of claimed tensors that MOVED is recorded, not
    required. `why` names the source that sorts the tensors."""

    def __init__(self, label, n, tokens_per_row, fn, claimed, why):
        self.label, self.n, self.tokens_per_row, self.fn = label, int(n), int(tokens_per_row), fn
        self.claimed, self.why = frozenset(claimed), why


class _GradCarry:
    """One gradient accumulated by CARRY (embedding contract 7.4: the second
    microbatch's fold continues from the first's stored accumulator, bit
    exact at EVERY split, no alignment condition). `fresh(a, b)` is the
    gradient of positions [a, b) from +0.0 and `carry(a, b, g)` continues
    `g`. Asked at the batch part's uneven split and at a split every
    `BATCH_ALONE` positions."""

    def __init__(self, label, n, fresh, carry):
        self.label, self.n, self.fresh, self.carry = label, int(n), fresh, carry


def _as_named(out):
    return {k: np.ascontiguousarray(np.asarray(v)) for k, v in out.items()}


def _flip_first_float(a):
    a = np.ascontiguousarray(a)
    raw = bytearray(a.tobytes())
    if raw:
        raw[0] ^= 1
    return np.frombuffer(bytes(raw), dtype=a.dtype).reshape(a.shape)


def _first_diff(a, b):
    ab, bb = a.tobytes(), b.tobytes()
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"whole {a.dtype.str}{list(a.shape)} vs {b.dtype.str}{list(b.shape)}"
    size = a.dtype.itemsize
    e = next(j for j in range(len(ab)) if ab[j] != bb[j]) // size
    return f"element {e}: whole {_elem_hex(ab, size, e)} vs {_elem_hex(bb, size, e)}"


def _eval_grad_accum(ml, call, sabotage, digest, notes):
    T = ml.training
    whole = _as_named(call.fn(0, call.n))
    names = sorted(whole)
    unknown = sorted(call.claimed - set(names))
    if unknown:
        raise ValueError(f"{call.label}: claimed tensors {unknown} are not in the gradient {names}")
    if sabotage in ("1", "bit"):
        hit = next((k for k in names if k in call.claimed), names[0])
        whole[hit] = _flip_first_float(whole[hit])
    tokens = call.n * call.tokens_per_row
    reported, asked = [], []
    notes.append(f"{call.label}: {GRAD_UNEVEN_SPLIT_NA}")
    for a in GRAD_ACCUM_SPLITS:
        if call.n % a:
            notes.append(f"{call.label}: A={a} n/a:{call.n} rows are not divisible into {a} equal pieces")
            continue
        aligned = bool(T.accumulation_is_aligned(tokens, a))
        rows = call.n // a
        pieces = [_as_named(call.fn(j * rows, (j + 1) * rows)) for j in range(a)]
        for j, p in enumerate(pieces):
            if sorted(p) != names:
                raise ValueError(f"{call.label}: piece {j} of A={a} has tensors {sorted(p)}, whole has {names}")
        if sabotage == "serial" and aligned:
            acc = [pieces[0][k] for k in names]
            for p in pieces[1:]:
                acc = T.accumulate_grads([acc, [p[k] for k in names]], tokens=None)
            comb = acc
        else:
            comb = T.accumulate_grads([[p[k] for k in names] for p in pieces], tokens=tokens if aligned else None)
        comb = dict(zip(names, (np.ascontiguousarray(np.asarray(c)) for c in comb)))
        moved = [k for k in names if comb[k].tobytes() != whole[k].tobytes()]
        if not aligned:
            n_claimed = sum(k in call.claimed for k in moved)
            notes.append(f"{call.label}: A={a} n/a:clause 9.2 refuses T={tokens} at A={a}; negative control "
                         f"(tokens=None): {n_claimed} of {len(call.claimed)} claimed tensors moved")
            asked.append(f"A{a}:refused:{n_claimed}")
            continue
        asked.append(f"A{a}:aligned")
        for k in moved:
            if k in call.claimed:
                return (f"BATCH_MOVED:{call.label}:A={a} of T={tokens} (clause 9.2 aligned) tensor {k} "
                        f"{_first_diff(whole[k], comb[k])}")
            reported.append(f"A{a}:{k}")
    if reported:
        notes.append(f"{call.label}: REPORTED (not claimed; {call.why}) moved at aligned splits: {reported}")
    digest.update(f"accum:{call.label}:{call.n}x{call.tokens_per_row}:{','.join(asked)}:"
                  f"claimed={','.join(sorted(call.claimed))}:reported-moved={','.join(reported)}".encode())
    for k in names:
        digest.update(f"{k}{whole[k].dtype.str}{whole[k].shape}".encode())
        digest.update(whole[k].tobytes())
    return None


def _eval_grad_carry(ml, call, sabotage, digest, notes):
    whole = np.ascontiguousarray(np.asarray(call.fresh(0, call.n)))
    if sabotage in ("1", "bit"):
        whole = _flip_first_float(whole)
    splits = [_split_bounds(call.n), list(range(0, call.n, BATCH_ALONE)) + [call.n]]
    for bounds in splits:
        g = None
        for a, b in zip(bounds, bounds[1:]):
            g = call.fresh(a, b) if g is None else call.carry(a, b, g)
        g = np.ascontiguousarray(np.asarray(g))
        if g.tobytes() != whole.tobytes():
            name = ",".join(str(b - a) for a, b in zip(bounds, bounds[1:]))
            if len(bounds) > 5:
                name = f"{len(bounds) - 1} pieces of {BATCH_ALONE}"
            return f"BATCH_MOVED:{call.label}:carry over split {name} {_first_diff(whole, g)}"
    digest.update(f"carry:{call.label}:{call.n}".encode())
    digest.update(f"{whole.dtype.str}{whole.shape}".encode())
    digest.update(whole.tobytes())
    return None


def _grad_rows(label, x, g, fn):
    """A per-row gradient over rows of (x, g): the rows are index rows, so a
    chunk is handed exactly its rows' bytes of both operands."""
    idx = np.arange(x.shape[0], dtype=np.int64).reshape(-1, 1)
    take = lambda a, r: np.ascontiguousarray(a[r[:, 0]])
    return _BatchRows(label, idx, lambda r: (np.asarray(fn(take(x, r), take(g, r))),))


#: the batchgrad slab for the sequence blocks: 8 sequences of 64 tokens, so
#: T = 512 and clause 9.2 admits A = 2 and A = 4 and refuses A = 8
GRAD_SEQ_BATCH, GRAD_SEQ_LEN = 8, 64

#: the tensors whose token contraction IS the v1 GEMM at k' = the token
#: count, per block, read from the public backward's source on 2026-09-15.
#: Transformer: every weight gradient, the seven projections through
#: `_route_b` and both RMSNorm weights as `ones . dprod` (DEVIATION 1410),
#: transformer/checks/transformer_backward.mojo. Mamba-1: the four
#: projections' dB and the RED_* reductions (`mamba_backward_reduce_into`,
#: OP_NN at (1, W, M)), NOT `A_log` ("T4 dA over t: t DESCENDING ... NOT
#: routed", "T5 parameter fold over b", mamba/checks/mamba_backward.mojo),
#: modeling_mamba_prefill_backward.mojo. Mamba-2: the two projections and
#: the block norm, gated norm and D reductions, NOT conv1d.weight,
#: conv1d.bias, dt_bias or A_log, which mamba/impl/ops/mamba2_ssd_backward.mojo
#: folds serially over every (row, position) in one thread per channel
#: (`mamba2_conv_backward_kernel`, `mamba2_dt_backward_kernel`,
#: `mamba2_da_product_backward_kernel`). Mamba-3: every weight gradient (the
#: two projections and the seven RED3_* reductions),
#: mamba/impl/modules/mamba3_prefill_backward.mojo.
GRAD_CLAIMED = {
    "transformer": ("input_layernorm.weight", "post_attention_layernorm.weight", "q_proj.weight", "k_proj.weight",
                    "v_proj.weight", "o_proj.weight", "gate_proj.weight", "up_proj.weight", "down_proj.weight"),
    "mamba1": ("norm.weight", "in_proj.weight", "conv1d.weight", "conv1d.bias", "x_proj.weight", "dt_proj.weight",
               "dt_proj.bias", "D", "out_proj.weight"),
    "mamba2": ("block_norm.weight", "in_proj.weight", "D", "norm.weight", "out_proj.weight"),
    "mamba3": ("block_norm.weight", "in_proj.weight", "dt_bias", "B_norm.weight", "C_norm.weight", "B_bias",
               "C_bias", "D", "out_proj.weight"),
}


def _batchgrad_block(kind):
    def spec(ml, blk, Xh):
        dm = blk.d_model
        x = _seq(Xh, GRAD_SEQ_BATCH, GRAD_SEQ_LEN, dm)
        g = _seq(Xh, GRAD_SEQ_BATCH, GRAD_SEQ_LEN, dm, skip=GRAD_SEQ_BATCH * GRAD_SEQ_LEN * dm)
        # The row-invariance proof and the accumulation proof deliberately
        # ask several of the same exact slices (the whole batch and every
        # one-row piece).  A backward call returns both x and weight
        # gradients, so running it again merely to select the other half is
        # duplicate production work, not independent evidence.  Key by the
        # complete input bytes: distinct slices still execute independently,
        # while an exactly repeated claim reuses the one native result it is
        # supposed to compare.
        cache = {}

        def backward_cached(xr, gr):
            xr, gr = np.ascontiguousarray(xr), np.ascontiguousarray(gr)
            key = (xr.dtype.str, xr.shape, xr.tobytes(), gr.dtype.str, gr.shape, gr.tobytes())
            if key not in cache:
                cache[key] = blk.backward(xr, gr)
            return cache[key]

        piece = lambda a, b: {k: v for k, v in backward_cached(x[a:b], g[a:b]).items() if k != "x"}
        return [_grad_rows("backward x (per row)", x, g, lambda xr, gr: backward_cached(xr, gr)["x"]),
                _GradAccum("backward weights", GRAD_SEQ_BATCH, GRAD_SEQ_LEN, piece, GRAD_CLAIMED[kind],
                           "a serial fold over tokens, not the v1 GEMM at k' = T; see GRAD_CLAIMED")]
    return spec


_batchgrad_decl(_batchgrad_block("transformer"), "transformer", "transformer-window", "transformer-bf16w", "transformer-int8w")
_batchgrad_decl(_batchgrad_block("mamba1"), "mamba1", "mamba1-bf16w", "mamba1-int8w")
_batchgrad_decl(_batchgrad_block("mamba2"), "mamba2", "mamba2-dtlimit")
_batchgrad_decl(_batchgrad_block("mamba3"), "mamba3")


def _batchgrad_samba(ml, st, Xh):
    """`SambaStack.loss_and_grads(..., num_items=count)`, the call
    `train_step` accumulates: T = 8 x 64 = 512 target tokens, each piece
    divided by the whole step's count, combined by the clause 9.2 tree.
    Clause 9.3 CLAIMS the LM head and final-norm weight gradients and says
    the embedding's run-sorted fold and the Mamba-3 block's per-head
    reductions are reported, not claimed; this part follows that text
    exactly (test_samba_surface.py arm STACK-ACCUM asserts the same two). A
    dropout stack draws its mask from a fixed stream at each piece's token
    offset, as `train_step` does."""
    ids = _ids(Xh, GRAD_SEQ_BATCH, GRAD_SEQ_LEN + 1, vocab=st.config.vocab)
    inputs, targets = np.ascontiguousarray(ids[:, :-1]), np.ascontiguousarray(ids[:, 1:])
    count = int(targets.size)
    stream = 7 if st.config.dropout > 0.0 else None

    def piece(a, b):
        _, grads = st.loss_and_grads(np.ascontiguousarray(inputs[a:b]), np.ascontiguousarray(targets[a:b]),
                                     num_items=count, dropout_stream=stream, token_offset=a * GRAD_SEQ_LEN)
        return dict(zip(st.names, grads))
    claimed = [n for n in ("lm_head.weight", "norm_f.weight") if n in st.names]
    return [_GradAccum("loss_and_grads(num_items=T)", GRAD_SEQ_BATCH, GRAD_SEQ_LEN, piece, claimed,
                       "optimizer contract clause 9.3 claims only the LM head and final-norm weights")]


_batchgrad_decl(_batchgrad_samba, "samba", "samba-untied-dropout-accum")


def _batchgrad_cross_entropy(ml, e, Xh):
    """dlogits under reduction='sum' with the divisor fixed at the whole
    batch's N (`num_items`), so a row's gradient is its slice of the whole
    one: loss contract 7.1, L14-L16 read one row and never N (only L12, the
    scalar loss, reads N, and it is not asked). Three kernels: plain, label
    smoothing above zero (a different kernel, 6.2(c)) and an ignore_index
    that masks rows (7.3)."""
    T = ml.training
    logits = _hw((64, 16), "ce:logits", -2.0, 2.0)
    t = (_ids(Xh, 1, 64).reshape(64) % 16).astype(np.int32)
    tm = t.copy(); tm[::4] = 0
    grad = lambda lg, tg, **kw: T.cross_entropy(lg, tg[:, 0].astype(np.int32), reduction="sum", num_items=64,
                                                 return_grad=True, **kw)[1]
    return [_grad_rows("dlogits sum/num_items=64", logits, t.reshape(-1, 1), grad),
            _grad_rows("dlogits label_smoothing=0.1", logits, t.reshape(-1, 1),
                       lambda lg, tg: grad(lg, tg, label_smoothing=0.1)),
            _grad_rows("dlogits ignore_index=0", logits, tm.reshape(-1, 1),
                       lambda lg, tg: grad(lg, tg, ignore_index=0))]


_batchgrad_decl(_batchgrad_cross_entropy, "cross-entropy-arms")


def _batchgrad_training_primitives(ml, e, Xh):
    """linear_backward and rms_norm_backward over M = 512 rows: the input
    gradients per row, the weight gradients accumulated. Both weight
    gradients are the v1 GEMM at k' = M (`linear_backward`'s docstring names
    the clause 9.2 contraction; `rms_norm_backward` runs
    training/samba_ops.mojo::samba_rms_norm_backward_host, the transformer's
    `bwd_rms_norm`, `ones . dprod`), so both are claimed. embedding_backward
    is the run-sorted fold, which clause 9.3 reports and does not claim."""
    T = ml.training
    V, D, M, K = 64, 32, 512, 16
    x = np.ascontiguousarray(_seq(Xh, 1, M, D).reshape(M, D))
    dy = np.ascontiguousarray(_seq(Xh, 1, M, D, skip=M * D).reshape(M, D))
    dc = _hw((M, K), "prim:grad:dc", -1.0, 1.0)
    w_lin = _hw((K, D), "prim:lin", -0.25, 0.25)
    gam = np.ascontiguousarray(np.abs(_hw((D,), "prim:rms", 0.5, 1.5)))
    ids = (_ids(Xh, 1, M).reshape(M) % V).astype(np.int32)
    lin = lambda a, b: dict(zip(("da", "dweight"), T.linear_backward(dc[a:b], x[a:b], w_lin)))
    rms = lambda a, b: dict(zip(("dx", "dweight"), T.rms_norm_backward(dy[a:b], x[a:b], gam, 1e-5)))
    return [_grad_rows("linear_backward da (per row)", dc, x, lambda d_, xr: T.linear_backward(d_, xr, w_lin)[0]),
            _GradAccum("linear_backward dweight", M, 1, lambda a, b: {"dweight": lin(a, b)["dweight"]},
                       ("dweight",), "-"),
            _grad_rows("rms_norm_backward dx (per row)", dy, x, lambda d_, xr: T.rms_norm_backward(d_, xr, gam, 1e-5)[0]),
            _GradAccum("rms_norm_backward dweight", M, 1, lambda a, b: {"dweight": rms(a, b)["dweight"]},
                       ("dweight",), "-"),
            _GradAccum("embedding_backward", M, 1,
                       lambda a, b: {"dweight": T.embedding_backward(dy[a:b], ids[a:b], V)}, (),
                       "the run-sorted ascending fold, reported by clause 9.3")]


_batchgrad_decl(_batchgrad_training_primitives, "training-primitives")


def _batchgrad_embedding(ml, e, Xh):
    """Embedding.backward's carry (contract 7.4, bit exact at every split),
    T = 512 held-out ids with the lane's padding_idx and plan."""
    n, d = 512, e.embedding_dim
    ids = (_ids(Xh, 1, n).reshape(n) % e.num_embeddings).astype(np.int32)
    dy = np.ascontiguousarray(_seq(Xh, 1, n, d).reshape(n, d))
    return [_GradCarry("backward carry", n, lambda a, b: e.backward(ids[a:b], dy[a:b]),
                       lambda a, b, g: e.backward(ids[a:b], dy[a:b], grad=g))]


_batchgrad_decl(_batchgrad_embedding, "embedding", "embedding-sort")

# The lanes that train and are NOT asked, each with the reason from its own
# source. Every other lane has no gradient call on its public surface.
_batchgrad_decl("n/a:mean-reduction (SmallMLPTrainer.loss_and_grads is mean cross-entropy over the call's own "
                "rows, _mlp_impl.py::_gradient; a row alone or a piece carries a different divisor by definition)",
                "mlp", "par-mlp")
_batchgrad_decl("n/a:mean-reduction-fixed-batch (the byte LM trainers step mean cross-entropy over ids of the "
                "profile's fixed (batch, length + 1); ByteLanguageModelConfig.batch is part of the profile)",
                "byte-lm", "byte-lm-resident", "byte-lm-host-train")
_batchgrad_decl("n/a:optimizer-step (no gradient is computed; accumulate_grads is asked by the block, samba "
                "and primitive lanes)", "optim-sgd", "optim-adam-clip")
_batchgrad_decl("n/a:driver-lane (a multi-GPU driver held to its single-device twin by its train column; the "
                "twin lane carries the batchgrad part)", "par-samba", "par-samba-clip", "par-byte-lm",
                "par-byte-lm-model-pool", "par-byte-lm-offload")

#: every undeclared lane: its public surface has no backward, gradient or
#: partial_fit (the scalers' partial_fit raises NotImplementedError by name)
BATCHGRAD_DEFAULT = "n/a:no-backward"


# ---------------------------------------------------------------- the batchscale part (2026-09-15)
# See `batchscale` in the module docstring. Opt-in (`--batch-scale`). The
# batch part asks at most 16 rows alone and a split of 1, 7 and the rest;
# checks/batch_invariance_check.mojo asks the blocks at B in {1, 17, 64}.
# This part asks the PUBLIC surface at serving sizes: a whole call of
# SCALE_WHOLE rows, and sub-batches of SCALE_BATCHES rows at the start, the
# middle and the end of it, every row of every sub-batch equal to its row
# of the whole call; and the causal sequence models at a long length,
# every prefix in SCALE_PREFIXES equal to the first positions of the whole.

SCALE_BATCHES = (1, 17, 64, 256)
SCALE_WHOLE = 1024
SCALE_LONG_L = 1024
#: 64 is the Mamba-3 chunk, 128 the v1 GEMM leaf, 256 the Mamba-2 chunk
SCALE_PREFIXES = (1, 7, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023)

BATCHSCALE = {}


def _batchscale_decl(spec, *names):
    for n in names:
        if n in BATCHSCALE:
            raise RuntimeError(f"identity_break: lane {n!r} has two batchscale declarations")
        BATCHSCALE[n] = spec


class _ScaleRows:
    """One row-wise call asked at serving sizes. `fn(R[a:b])` returns a
    tuple of outputs whose axis 0 is the rows, as for _BatchRows."""

    def __init__(self, label, R, fn):
        self.label, self.R, self.fn = label, np.ascontiguousarray(R), fn


def _eval_scale_rows(call, sabotage, digest):
    n = call.R.shape[0]
    whole = _as_rows(call.fn(call.R), n, f"{call.label} whole B={n}")
    if sabotage:
        _sabotage_rows(whole)
    for b in SCALE_BATCHES:
        if b > n:
            continue
        for a in sorted({0, (n - b) // 2, n - b}):
            got = _as_rows(call.fn(np.ascontiguousarray(call.R[a:a + b])), b, f"{call.label} B={b} at {a}")
            for i in range(b):
                m = _row_mismatch(whole[a + i], got[i], f"B={b} starting at row {a}")
                if m:
                    return f"BATCH_MOVED:{call.label}:row {a + i} of {n} in B={b} starting at row {a}:{m}"
    digest.update(f"scale:{call.label}:{n}:{','.join(map(str, SCALE_BATCHES))}".encode())
    for r in whole:
        for dt, shape, raw in r:
            digest.update(f"{dt}{shape}".encode())
            digest.update(raw)
    return None


def _tiled(X, shape):
    """`shape` float32 values from the fixture, row-major and wrapped (the
    serving sizes ask for more values than some fixtures hold; the fixture
    lengths do not divide the row sizes, so wrapped rows are not copies)."""
    return np.ascontiguousarray(np.resize(np.ascontiguousarray(X).reshape(-1), int(np.prod(shape))).reshape(shape))


def _tiled_ids(X, b, l, vocab=256):
    raw = np.frombuffer(np.ascontiguousarray(X).tobytes(), dtype=np.uint8)
    return np.ascontiguousarray((np.resize(raw, b * l).astype(np.int32) % vocab).reshape(b, l))


def _scale_prefix(label, full, fn):
    call = _BatchPrefix(label, full, fn, axis=1)
    call.lengths = [p for p in SCALE_PREFIXES if p < full]
    return call


def _batchscale_block(ml, blk, Xh):
    dm = blk.d_model
    x = _tiled(Xh, (SCALE_WHOLE, SEQ_LEN, dm))
    xl = _tiled(Xh[::-1], (2, SCALE_LONG_L, dm))
    return [_ScaleRows(f"forward L={SEQ_LEN}", x, lambda r: (np.asarray(blk.forward(r)),)),
            _scale_prefix(f"forward B=2 L={SCALE_LONG_L}", SCALE_LONG_L,
                          lambda p: (np.asarray(blk.forward(np.ascontiguousarray(xl[:, :p, :]))),))]


_batchscale_decl(_batchscale_block, "mamba1", "mamba2", "mamba3", "mamba2-dtlimit", "transformer",
                 "transformer-window", "mamba1-bf16w", "mamba1-int8w", "mamba2-bf16w", "mamba2-int8w",
                 "mamba3-bf16w", "mamba3-int8w", "transformer-bf16w", "transformer-int8w")


def _long_byte_lm(ml, e):
    """The lane's model at length SCALE_LONG_L. A byte LM registry does not
    depend on the length (RoPE, no position table), so the same flat
    parameters load into THE LANE'S OWN shape with only `length` changed.
    Read the shape from the fit rather than assuming the shipped profile:
    since 2026-09-16 byte-lm-resident runs one block at d_model 16."""
    shape = ml.ByteLanguageModelConfig(**dict(_byte_lm_shape(ml, e).to_dict(), length=SCALE_LONG_L))
    if isinstance(e, ml.LanguageModelInference):
        return ml.LanguageModelInference(e._parameters, shape=shape, threaded=e._threaded, threads=e._threads)
    return ml.SmallByteLanguageModelTrainer(np.asarray(e.parameters_),
                                            data_schedule={"dataset": "identity_break", "order": "sequential"},
                                            shape=shape)


def _batchscale_byte_lm(ml, e, Xh):
    shape = _byte_lm_shape(ml, e)
    ids = _tiled_ids(Xh, SCALE_WHOLE, shape.length)
    long_m = _long_byte_lm(ml, e)
    idl = _tiled_ids(Xh[::-1], 2, SCALE_LONG_L)
    return [_ScaleRows(f"logits L={shape.length}", ids, lambda r: (np.asarray(e.logits(r)),)),
            _scale_prefix(f"logits B=2 L={SCALE_LONG_L}", SCALE_LONG_L,
                          lambda p: (np.asarray(long_m.logits(np.ascontiguousarray(idl[:, :p]))),))]


_batchscale_decl(_batchscale_byte_lm, "byte-lm", "byte-lm-resident", "byte-lm-host-infer",
                 "byte-lm-host-infer-threaded")


def _batchscale_samba(ml, st, Xh):
    ids = _tiled_ids(Xh, SCALE_WHOLE, SEQ_LEN, st.config.vocab)
    idl = _tiled_ids(Xh[::-1], 2, SCALE_LONG_L, st.config.vocab)
    return [_ScaleRows(f"forward L={SEQ_LEN}", ids, lambda r: (np.asarray(st.forward(r)),)),
            _scale_prefix(f"forward B=2 L={SCALE_LONG_L}", SCALE_LONG_L,
                          lambda p: (np.asarray(st.forward(np.ascontiguousarray(idl[:, :p]))),))]


_batchscale_decl(_batchscale_samba, "samba", "samba-untied-dropout-accum")


def _scale_calls(*methods, sl=slice(None), prep=None):
    def spec(ml, e, Xh):
        R = (Xh if prep is None else prep(Xh))[sl][:SCALE_WHOLE]
        return [_ScaleRows(m, R, (lambda m: lambda r: (getattr(e, m)(r),))(m)) for m in methods]
    return spec


def _batchscale_kneighbors(ml, e, Xh):
    return [_ScaleRows("kneighbors", Xh[:SCALE_WHOLE], lambda r: tuple(e.kneighbors(r)))]


# The representative classical set the gap named: neighbors, forests, GBDT
# predict, linear models and a decomposition.
_batchscale_decl(_batchscale_kneighbors, "knn")
_batchscale_decl(_scale_calls("predict", "predict_proba"), "rf-clf", "gbdt-symmetric")
_batchscale_decl(_scale_calls("predict"), "rf-reg", "gbdt-rmse", "ols", "ridge")
_batchscale_decl(_scale_calls("predict_proba"), "logistic")
_batchscale_decl(_scale_calls("transform"), "pca")

BATCHSCALE_DEFAULT = "n/a:not-in-the-serving-set"


# ---------------------------------------------------------------- the ragged part (2026-09-15)
# See `ragged` in the module docstring. Opt-in (`--ragged`). The causal
# sequence models take `lengths=` since 2026-09-15 (python/mojolearn/_ragged.py):
# a right-padded batch whose real positions must equal each row alone at its
# own length and whose padding outputs must be exactly +0.0. The padding
# INPUT is filled with junk (NaN for the float blocks, out-of-vocabulary ids
# for the token models) so a padding value that reached any real position,
# or any refusal, is seen.

#: row lengths of the short ragged batch (L = SEQ_LEN) and of the long one;
#: the long lengths straddle the Mamba-3 chunk (64), the v1 GEMM leaf (128)
#: and the Mamba-2 chunk (256)
RAGGED_LENGTHS = (16, 1, 7, 9, 16, 3, 12, 15)
RAGGED_LONG_LENGTHS = (300, 1, 65, 129, 257, 128)

RAGGED = {}


def _ragged_decl(spec, *names):
    for n in names:
        if n in RAGGED:
            raise RuntimeError(f"identity_break: lane {n!r} has two ragged declarations")
        RAGGED[n] = spec


class _RaggedCall:
    """One ragged call. `x` is (B, L, ...) with junk at the padding
    positions, `padded(x, lengths)` returns a tuple of outputs whose axis 0
    is the rows, and `alone(x_row)` the same outputs for one row cut to its
    length. `axes[k]` is output k's length axis (1), or None for a per-row
    output (next_bytes) compared whole."""

    def __init__(self, label, x, lengths, padded, alone, axes=(1,)):
        self.label, self.x, self.lengths = label, np.ascontiguousarray(x), tuple(lengths)
        self.padded, self.alone, self.axes = padded, alone, tuple(axes)


def _eval_ragged(call, sabotage, digest):
    outs = [np.array(np.asarray(o), copy=True) for o in call.padded(call.x, call.lengths)]
    if len(outs) != len(call.axes):
        raise ValueError(f"{call.label}: {len(outs)} outputs, {len(call.axes)} axes declared")
    if sabotage:
        outs[0] = _flip_first_float(outs[0])
    for i, n in enumerate(call.lengths):
        alone = [np.asarray(o) for o in call.alone(np.ascontiguousarray(call.x[i:i + 1, :n]))]
        for k, (O, A, ax) in enumerate(zip(outs, alone, call.axes)):
            A = np.ascontiguousarray(A)
            want = np.ascontiguousarray(O[i:i + 1] if ax is None else O[i:i + 1, :n])
            where = f"BATCH_MOVED:{call.label}:row {i} (length {n} of {call.x.shape[1]}) output {k}"
            if want.shape != A.shape or want.dtype != A.dtype:
                return f"{where}: padded {want.dtype.str}{list(want.shape)} vs alone {A.dtype.str}{list(A.shape)}"
            if want.tobytes() != A.tobytes():
                return f"{where} real positions: {_first_diff(want, A).replace('whole', 'padded', 1).replace(' vs ', ' vs alone ', 1)}"
            if ax is not None:
                pad = np.ascontiguousarray(O[i, n:])
                if pad.tobytes() != bytes(pad.nbytes):
                    first = next(j for j, c in enumerate(pad.tobytes()) if c) // pad.dtype.itemsize
                    return f"{where}: padding output element {first} is not +0.0"
    digest.update(f"ragged:{call.label}:{list(call.x.shape)}:{call.lengths}".encode())
    for o in outs:
        o = np.ascontiguousarray(o)
        digest.update(f"{o.dtype.str}{o.shape}".encode())
        digest.update(o.tobytes())
    return None


def _junk_float(x, lengths):
    x = np.array(x, dtype=np.float32, copy=True)
    for i, n in enumerate(lengths):
        x[i, n:] = np.float32("nan")
    return x


def _junk_ids(ids, lengths, vocab):
    ids = np.array(ids, dtype=np.int32, copy=True)
    for i, n in enumerate(lengths):
        ids[i, n:] = vocab + 7
    return ids


def _ragged_block(ml, blk, Xh):
    dm = blk.d_model
    L2 = max(RAGGED_LONG_LENGTHS)
    x = _junk_float(_seq(Xh, len(RAGGED_LENGTHS), SEQ_LEN, dm), RAGGED_LENGTHS)
    xl = _junk_float(_tiled(Xh, (len(RAGGED_LONG_LENGTHS), L2, dm)), RAGGED_LONG_LENGTHS)
    fwd = lambda a, lens: (np.asarray(blk.forward(a, lengths=lens)),)
    one = lambda r: (np.asarray(blk.forward(r)),)
    return [_RaggedCall(f"forward(lengths) L={SEQ_LEN}", x, RAGGED_LENGTHS, fwd, one),
            _RaggedCall(f"forward(lengths) L={L2}", xl, RAGGED_LONG_LENGTHS, fwd, one)]


_ragged_decl(_ragged_block, "mamba1", "mamba2", "mamba3", "mamba2-dtlimit", "transformer", "transformer-window",
             "mamba1-bf16w", "mamba1-int8w", "mamba2-bf16w", "mamba2-int8w", "mamba3-bf16w", "mamba3-int8w",
             "transformer-bf16w", "transformer-int8w")


def _ragged_byte_lm(ml, e, Xh):
    shape = _byte_lm_shape(ml, e)
    lens = tuple(min(n * 2, shape.length) for n in RAGGED_LENGTHS)
    ids = _junk_ids(_ids(Xh, len(lens), shape.length), lens, shape.vocab_size)
    both = lambda a, ls: (np.asarray(e.logits(a, lengths=ls)), np.asarray(e.next_bytes(a, lengths=ls), dtype=np.int64))
    one = lambda r: (np.asarray(e.logits(r)), np.asarray(e.next_bytes(r), dtype=np.int64))
    calls = [_RaggedCall(f"logits, next_bytes(lengths) L={shape.length}", ids, lens, both, one, axes=(1, None))]
    long_m = _long_byte_lm(ml, e)
    L2 = max(RAGGED_LONG_LENGTHS)
    idl = _junk_ids(_tiled_ids(Xh, len(RAGGED_LONG_LENGTHS), L2), RAGGED_LONG_LENGTHS, shape.vocab_size)
    calls.append(_RaggedCall(f"logits(lengths) L={L2}", idl, RAGGED_LONG_LENGTHS,
                             lambda a, ls: (np.asarray(long_m.logits(a, lengths=ls)),),
                             lambda r: (np.asarray(long_m.logits(r)),)))
    return calls


_ragged_decl(_ragged_byte_lm, "byte-lm", "byte-lm-resident", "byte-lm-host-infer", "byte-lm-host-infer-threaded")


def _ragged_samba(ml, st, Xh):
    v = st.config.vocab
    ids = _junk_ids(_ids(Xh, len(RAGGED_LENGTHS), SEQ_LEN, v), RAGGED_LENGTHS, v)
    L2 = max(RAGGED_LONG_LENGTHS)
    idl = _junk_ids(_tiled_ids(Xh, len(RAGGED_LONG_LENGTHS), L2, v), RAGGED_LONG_LENGTHS, v)
    fwd = lambda a, ls: (np.asarray(st.forward(a, lengths=ls)),)
    one = lambda r: (np.asarray(st.forward(r)),)
    return [_RaggedCall(f"forward(lengths) L={SEQ_LEN}", ids, RAGGED_LENGTHS, fwd, one),
            _RaggedCall(f"forward(lengths) L={L2}", idl, RAGGED_LONG_LENGTHS, fwd, one)]


_ragged_decl(_ragged_samba, "samba", "samba-untied-dropout-accum")
_ragged_decl("n/a:driver-lane (the multi-GPU drivers take no lengths; the single-device twin lane is asked)",
             "par-samba", "par-samba-clip", "par-byte-lm", "par-byte-lm-model-pool", "par-byte-lm-offload")

RAGGED_DEFAULT = "n/a:no-sequence-axis"


# ---------------------------------------------------------------- stepfull (lane/stateful-cpu-decoding, 2026-09-16)
# THE DECODE IS THE PREFILL. A model that decodes one token at a time with a
# carried state must answer, at every position, the bits the same model
# answers when the whole sequence is run as one fresh-state forward pass. If
# it does not, the streaming path is a different model wearing the same
# name, and no cross-vendor statement about the prefill says anything about
# what a user who streams actually gets.
#
# This part runs both, POSITION BY POSITION, through the PUBLIC class (the
# `*Inference` wrappers on a CPU column, `_public_est` over
# NEURAL_PUBLIC_PART_LANES). A mismatch is reported as the FIRST differing
# position with both values, never a count. The hash is the full pass's
# bytes, so it is the same bytes on every column where the equality holds
# and `--diff` carries the cross-vendor statement.
#
# `tools/step_vs_full_check.py` is the same comparison outside the harness,
# with a deeper fail-first arm: it moves ONE carried cache cell by ONE ULP
# and requires the per-position check to fire. This part's own sabotage
# (BATCH_SABOTAGE_ENV, as ragged's) perturbs the full pass so every hashed
# cell MUST read BATCH_MOVED.

#: rows and positions decoded one at a time
STEPFULL_ROWS, STEPFULL_LENGTH = 2, 16

STEPFULL = {}


def _stepfull_decl(spec, *names):
    for n in names:
        if n in STEPFULL:
            raise RuntimeError(f"identity_break: lane {n!r} has two stepfull declarations")
        STEPFULL[n] = spec


class _StepFullCall:
    """One model's two runs of the same sequence. `full(x)` is ONE
    fresh-state forward pass over all of it; `alloc()` makes the zero state
    and `step(x_t, state)` decodes position t carrying it. Both outputs
    have the positions on axis 1."""

    def __init__(self, label, x, full, alloc, step):
        self.label, self.x = label, np.ascontiguousarray(x)
        self.full, self.alloc, self.step = full, alloc, step


def _eval_stepfull(call, sabotage, digest):
    whole = np.ascontiguousarray(np.asarray(call.full(call.x)))
    if sabotage:
        whole = _flip_first_float(whole)
    state = call.alloc()
    got = np.empty_like(whole)
    for t in range(call.x.shape[1]):
        one = np.asarray(call.step(np.ascontiguousarray(call.x[:, t:t + 1]), state))
        one = np.ascontiguousarray(one).reshape(whole.shape[0], 1, -1)
        if one.shape != whole[:, t:t + 1].shape or one.dtype != whole.dtype:
            return (f"BATCH_MOVED:{call.label}: position {t} step "
                    f"{one.dtype.str}{list(one.shape)} vs full "
                    f"{whole.dtype.str}{list(whole[:, t:t + 1].shape)}")
        got[:, t:t + 1] = one
    for t in range(whole.shape[1]):
        want = np.ascontiguousarray(whole[:, t:t + 1])
        have = np.ascontiguousarray(got[:, t:t + 1])
        if want.tobytes() != have.tobytes():
            return (f"BATCH_MOVED:{call.label}: FIRST DIFFERING POSITION {t} of "
                    f"{whole.shape[1]}: {_first_diff(want, have).replace('whole', 'full', 1)}")
    digest.update(f"stepfull:{call.label}:{list(call.x.shape)}".encode())
    whole = np.ascontiguousarray(whole)
    digest.update(f"{whole.dtype.str}{whole.shape}".encode())
    digest.update(whole.tobytes())
    return None


def _stepfull_block_spec(state_kw=None):
    """A bare block: `forward(x)` against `allocate_state` + `step`."""
    def spec(ml, blk, Xh, _kw=state_kw):
        kw = dict(max_tokens=STEPFULL_LENGTH) if _kw == "kv" else {}
        x = _seq(Xh, STEPFULL_ROWS, STEPFULL_LENGTH, blk.d_model)
        return [_StepFullCall(
            f"forward(x) vs allocate_state + step, L={STEPFULL_LENGTH}", x,
            lambda a: blk.forward(np.ascontiguousarray(a)),
            lambda: blk.allocate_state(STEPFULL_ROWS, **kw),
            lambda a, st: blk.step(a, st))]
    return spec


def _stepfull_samba(ml, e, Xh):
    """The stack's logits: `forward(ids)` against `allocate_state` + `step`.
    `step` returns (B, vocab); the evaluator reshapes it to one position."""
    ids = _ids(Xh, STEPFULL_ROWS, STEPFULL_LENGTH, vocab=e.config.vocab)
    return [_StepFullCall(
        f"forward(ids) vs allocate_state + step, L={STEPFULL_LENGTH}", ids,
        lambda a: e.forward(np.ascontiguousarray(a)),
        lambda: e.allocate_state(STEPFULL_ROWS, STEPFULL_LENGTH),
        lambda a, st: e.step(a, st))]


_stepfull_decl(_stepfull_block_spec(), "mamba1", "mamba2", "mamba3", "mamba2-dtlimit",
               "mamba1-bf16w", "mamba1-int8w", "mamba2-bf16w", "mamba2-int8w", "mamba3-bf16w", "mamba3-int8w")
_stepfull_decl(_stepfull_block_spec("kv"), "transformer", "transformer-window", "transformer-bf16w", "transformer-int8w")
_stepfull_decl(_stepfull_samba, "samba", "samba-untied-dropout-accum")
_stepfull_decl("n/a:driver-lane (the multi-GPU drivers carry no decode state; the single-device twin lane is asked)",
               "par-samba", "par-samba-clip", "par-byte-lm", "par-byte-lm-model-pool", "par-byte-lm-offload")


def _stepfull_session_spec(state_kw=None):
    """lane/unlaned-public-algorithms (2026-09-20). THE RESIDENT DECODE IS
    THE PREFILL. The ordinary stepfull spec asks the block's per-call
    `step`; these two lanes' subject is the SESSION, whose whole claim is
    that its output is the per-call step's, so the part asks the block's
    fresh-state `forward` against the SESSION's streaming `step`. That is a
    strictly stronger statement than either half alone: the lane's own cell
    holds the session to the per-call step, and this holds the same session
    to the prefill."""
    def spec(ml, blk, Xh, _kw=state_kw):
        kw = dict(max_tokens=STEPFULL_LENGTH) if _kw == "kv" else {}
        x = _seq(Xh, STEPFULL_ROWS, STEPFULL_LENGTH, blk.d_model)
        sessions = []

        def alloc():
            st = blk.allocate_state(STEPFULL_ROWS, **kw)
            sessions.append(blk.decode_session(st))
            return sessions[-1]

        return [_StepFullCall(
            f"forward(x) vs decode_session step, L={STEPFULL_LENGTH}", x,
            lambda a: blk.forward(np.ascontiguousarray(a)),
            alloc,
            lambda a, sess: sess.step(a))]
    return spec


_stepfull_decl(_stepfull_session_spec(), "mamba1-decode-session")
_stepfull_decl(_stepfull_session_spec("kv"), "transformer-decode-session")
_stepfull_decl("n/a:driver-lane (the layer owners hold the state in their own worker processes; the "
               "single-device CausalLM twin, hf-causal-lm, carries the prefill-against-decode flag)",
               "par-causal-lm")
_stepfull_decl("n/a:no-decode-state (a cross-validation score and a Gaussian process classifier posterior "
               "have no sequence axis and no carried state)",
               "par-cross-val", "par-gpc-fit", "par-gpc-predict")

STEPFULL_DEFAULT = "n/a:no-decode-state"


# ---------------------------------------------------------------- the opt-in parts, one runner

#: part name -> (declarations, default, CLI flag, sabotage env)
EXTRA_PARTS = {
    "batchgrad": (BATCHGRAD, BATCHGRAD_DEFAULT, "batch_grad", BATCHGRAD_SABOTAGE_ENV),
    "batchscale": (BATCHSCALE, BATCHSCALE_DEFAULT, "batch_scale", BATCH_SABOTAGE_ENV),
    "ragged": (RAGGED, RAGGED_DEFAULT, "ragged", BATCH_SABOTAGE_ENV),
    "stepfull": (STEPFULL, STEPFULL_DEFAULT, "step_full", BATCH_SABOTAGE_ENV),
}


def _part_protocol(part, alone):
    if part == "batchgrad":
        return dict(rows=dict(alone=alone, split=list(BATCH_SPLIT) + ["n"]), accum_splits=list(GRAD_ACCUM_SPLITS),
                    uneven_split=GRAD_UNEVEN_SPLIT_NA, carry_splits=[list(BATCH_SPLIT) + ["n"], f"every {alone}"],
                    seq=[GRAD_SEQ_BATCH, GRAD_SEQ_LEN])
    if part == "batchscale":
        return dict(batches=list(SCALE_BATCHES), whole=SCALE_WHOLE, long_l=SCALE_LONG_L,
                    prefixes=list(SCALE_PREFIXES))
    if part == "stepfull":
        return dict(rows=STEPFULL_ROWS, length=STEPFULL_LENGTH,
                    full="forward(x) from a zero state, one pass",
                    stepped="allocate_state then step(x_t, state) at every position",
                    compared="bitwise, position by position")
    return dict(lengths=list(RAGGED_LENGTHS), long_lengths=list(RAGGED_LONG_LENGTHS), padding="nan/vocab+7")


def _probe_part(part, fit, name, ml, Xh, alone, sabotage):
    """One opt-in part of ONE fit: (value, error, notes), the value as in
    _probe_batch. `sabotage` is the part's env value, '' when off."""
    table, default = EXTRA_PARTS[part][0], EXTRA_PARTS[part][1]
    spec = table.get(name, default)
    notes = []
    if isinstance(spec, str):
        return spec, None, notes
    flip = sabotage not in ("", "0", "serial")
    try:
        # batchgrad differentiates the fitted block, which no inference class can
        calls = spec(ml, fit.est if part == "batchgrad" else _public_est(name, fit.est, NEURAL_PUBLIC_PART_LANES), Xh)
        digest = hashlib.sha256()
        for call in calls:
            if isinstance(call, _BatchRows):
                moved = _eval_batch_rows(call, alone, flip, digest)
            elif isinstance(call, _BatchPrefix):
                moved = _eval_batch_prefix(call, flip, digest)
            elif isinstance(call, _GradAccum):
                moved = _eval_grad_accum(ml, call, sabotage, digest, notes)
            elif isinstance(call, _GradCarry):
                moved = _eval_grad_carry(ml, call, sabotage, digest, notes)
            elif isinstance(call, _ScaleRows):
                moved = _eval_scale_rows(call, flip, digest)
            elif isinstance(call, _RaggedCall):
                moved = _eval_ragged(call, flip, digest)
            elif isinstance(call, _StepFullCall):
                moved = _eval_stepfull(call, flip, digest)
            else:
                raise TypeError(f"{part}: unknown call type {type(call).__name__}")
            if moved:
                return moved[:400], None, notes
        return digest.hexdigest()[:16], None, notes
    except Exception as exc:
        return None, _exc_text(part, exc), notes
    finally:
        while _BATCH_CLOSE:
            _BATCH_CLOSE.pop()()


# ---------------------------------------------------------------- rlpair (2026-09-15)
# THE RL PAIR. Reinforcement learning on a language model samples with one
# program and trains with another, and the policy-gradient arithmetic
# assumes the log-probability the sampler recorded for a token is the one
# the trainer recomputes for it. This part holds the public surface to that,
# bit for bit, per lane; the cross-vendor and CPU statements come from
# `--diff` over the records, because the hash is the same bytes wherever
# every assertion below holds. The module docstring's `rlpair` entry is the
# protocol; IDENTITY_PATHS.md says what is proven and what is not.

#: the sabotage switch: flips the lowest bit of the first sampler
#: log-probability, so every hashed rlpair cell MUST read RLPAIR_MOVED
RLPAIR_SABOTAGE_ENV = "MOJOLEARN_IDENTITY_RLPAIR_SABOTAGE"

#: sequences, prompt length and decoded tokens
RLPAIR_SEQS, RLPAIR_PROMPT, RLPAIR_DECODE = 5, 4, 12

#: the trainer's uneven microbatch split of the RLPAIR_SEQS rows: [0, 2), [2, 5)
RLPAIR_SPLIT = (2,)

#: continuous batching: sequence 4 is prefilled and decoded ALONE for its
#: first RLPAIR_JOIN_AT tokens, then its state joins the running batch of the
#: other four at row RLPAIR_JOIN_ROW; sequence 1 leaves once its token
#: RLPAIR_LEAVE_AFTER is chosen
RLPAIR_JOINER, RLPAIR_JOIN_AT, RLPAIR_JOIN_ROW = 4, 3, 2
RLPAIR_LEAVER, RLPAIR_LEAVE_AFTER = 1, 8

#: the vocabulary of the head the harness puts on a bare sequence block
RLPAIR_BLOCK_VOCAB = 48

RLPAIR = {}


def _rlpair_protocol(enabled=True):
    return dict(seqs=RLPAIR_SEQS, prompt=RLPAIR_PROMPT, decode=RLPAIR_DECODE,
                trainer_split=list(RLPAIR_SPLIT) + ["n"], joiner=RLPAIR_JOINER,
                join_at=RLPAIR_JOIN_AT, join_row=RLPAIR_JOIN_ROW, leaver=RLPAIR_LEAVER,
                leave_after=RLPAIR_LEAVE_AFTER, block_vocab=RLPAIR_BLOCK_VOCAB,
                logprob="training.cross_entropy(reduction='none')", enabled=enabled)


def _rlpair_decl(spec, *names):
    for n in names:
        if n in RLPAIR:
            raise RuntimeError(f"identity_break: lane {n!r} has two rlpair declarations")
        RLPAIR[n] = spec


class _RLSampler:
    """One sampler: `prefill(prompts (B, P) int32) -> (state, logits (B, V))`,
    `step(tokens (B,) int32, state) -> (state, logits (B, V))`, and the row
    operations continuous batching needs, `take(state, rows)` and
    `cat(states)`, on the caller-owned state the lane's API documents."""

    def __init__(self, name, prefill, step, take=None, cat=None):
        self.name, self.prefill, self.step = name, prefill, step
        self.take = take or _rl_take
        self.cat = cat or _rl_cat


def _rl_logits(a, what, lead):
    """A float32 logits array whose leading axes are `lead`; refuses any
    other dtype rather than casting (a cast would be bits no side made)."""
    a = np.asarray(a)
    if a.dtype != np.float32:
        raise TypeError(f"{what}: logits are {a.dtype}, the part compares float32 bits")
    if a.ndim != len(lead) + 1 or tuple(a.shape[:len(lead)]) != tuple(lead):
        raise ValueError(f"{what}: logits have shape {a.shape}, expected leading axes {lead}")
    return np.array(a, copy=True)


def _rl_nll(ml, logits, targets):
    """-log softmax(logits)[target] per row, through the LIBRARY's own
    cross-entropy: `mojolearn.training.cross_entropy(..., reduction='none')`,
    label smoothing 0.0 (the loss profile `mojolearn.identical.loss.ce.fp32.v1`,
    the kernel SambaStack's loss calls), never numpy. The log-probability is
    its negation, which is exact, so the part stores and compares the NLL."""
    y = np.ascontiguousarray(np.asarray(targets).reshape(-1).astype(np.int32))
    out = np.asarray(ml.training.cross_entropy(np.ascontiguousarray(logits), y, reduction="none"))
    if out.dtype != np.float32 or out.shape != y.shape:
        raise TypeError(f"cross_entropy(reduction='none') returned {out.dtype}{out.shape}")
    return np.array(out, copy=True)


def _rl_decode(ml, side, prompts):
    """Greedy decode of every prompt row together: token k is the argmax
    (lowest index on a tie) of the logits the prefill (k = 0) or the step
    that consumed token k - 1 returned. Returns (ids (B, T) int32, nll
    (B, T) float32, logits (B, T, V) float32), each nll computed from that
    step's (B, V) logits in one cross-entropy call."""
    b = prompts.shape[0]
    st, lg = side.prefill(np.ascontiguousarray(prompts))
    lg = _rl_logits(lg, f"{side.name} prefill", (b,))
    ids = np.zeros((b, RLPAIR_DECODE), dtype=np.int32)
    nll = np.zeros((b, RLPAIR_DECODE), dtype=np.float32)
    logits = np.zeros((b, RLPAIR_DECODE, lg.shape[1]), dtype=np.float32)
    for k in range(RLPAIR_DECODE):
        if k:
            st, lg = side.step(np.ascontiguousarray(ids[:, k - 1]), st)
            lg = _rl_logits(lg, f"{side.name} step {k}", (b,))
        tok = np.argmax(lg, axis=1).astype(np.int32)
        ids[:, k], logits[:, k], nll[:, k] = tok, lg, _rl_nll(ml, lg, tok)
    return ids, nll, logits


def _rl_continuous(ml, side, prompts):
    """The same greedy decode with the batch changing under it: the joiner
    runs alone, then joins; the leaver leaves. Returns per sequence the
    (ids, nll, logits) it produced, the leaver's cut at its last token."""
    b, J = prompts.shape[0], RLPAIR_JOINER
    got = {s: ([], [], []) for s in range(b)}
    last = {}

    def record(rows, lg, what):
        lg = _rl_logits(lg, what, (len(rows),))
        tok = np.argmax(lg, axis=1).astype(np.int32)
        nll = _rl_nll(ml, lg, tok)
        for r, s in enumerate(rows):
            got[s][0].append(tok[r]); got[s][1].append(nll[r]); got[s][2].append(lg[r])
            last[s] = tok[r]

    active = [s for s in range(b) if s != J]
    st, lg = side.prefill(np.ascontiguousarray(prompts[active]))
    record(active, lg, f"{side.name} continuous prefill")
    jst, jlg = side.prefill(np.ascontiguousarray(prompts[J:J + 1]))
    record([J], jlg, f"{side.name} continuous joiner prefill")
    for k in range(1, RLPAIR_DECODE):
        if k < RLPAIR_JOIN_AT:
            jst, jlg = side.step(np.asarray([last[J]], dtype=np.int32), jst)
            record([J], jlg, f"{side.name} continuous joiner step {k}")
        elif k == RLPAIR_JOIN_AT:
            st = side.cat([side.take(st, range(0, RLPAIR_JOIN_ROW)), jst,
                           side.take(st, range(RLPAIR_JOIN_ROW, len(active)))])
            active = active[:RLPAIR_JOIN_ROW] + [J] + active[RLPAIR_JOIN_ROW:]
        if k == RLPAIR_LEAVE_AFTER + 1:
            keep = [r for r, s in enumerate(active) if s != RLPAIR_LEAVER]
            st = side.take(st, keep)
            active = [active[r] for r in keep]
        st, lg = side.step(np.asarray([last[s] for s in active], dtype=np.int32), st)
        record(active, lg, f"{side.name} continuous step {k}")
    return {s: (np.asarray(v[0], dtype=np.int32), np.asarray(v[1], dtype=np.float32),
                np.asarray(v[2], dtype=np.float32)) for s, v in got.items()}


def _rl_train(ml, trainer, prompts, ids, bounds):
    """Teacher-forced: the realized sequences (prompt then the sampled
    tokens) through the TRAINING forward in the chunks `bounds` names, one
    cross-entropy call per chunk over every position of the chunk (targets
    shifted by one, as a loss is), then the decoded positions kept.
    Returns (ids, nll, logits) shaped as `_rl_decode`'s."""
    S = np.ascontiguousarray(np.concatenate([prompts, ids], axis=1).astype(np.int32))
    inp, tgt = np.ascontiguousarray(S[:, :-1]), S[:, 1:]
    nlls, lgs = [], []
    for a, c in zip(bounds, bounds[1:]):
        lg = _rl_logits(trainer(np.ascontiguousarray(inp[a:c])), f"trainer rows [{a},{c})", (c - a, inp.shape[1]))
        nll = _rl_nll(ml, lg.reshape(-1, lg.shape[2]), tgt[a:c]).reshape(c - a, inp.shape[1])
        nlls.append(nll); lgs.append(lg)
    p = RLPAIR_PROMPT - 1
    return (ids.copy(), np.ascontiguousarray(np.concatenate(nlls)[:, p:]),
            np.ascontiguousarray(np.concatenate(lgs)[:, p:, :]))


def _rl_mismatch(pair, ref, got, seqs=None):
    """None when (ids, nll, logits) are the same bytes, else the earliest
    decoded token (then the lowest sequence) where any of the three differ,
    naming which differ and the first differing value of each in hex."""
    names = ("ids", "nll", "logits")
    if seqs is None:
        seqs = list(range(ref[0].shape[0]))
    for name, r, g in zip(names, ref, got):
        if r.shape != g.shape or r.dtype != g.dtype:
            return f"{pair}:{name} {r.dtype.str}{list(r.shape)} vs {g.dtype.str}{list(g.shape)}"
    masks = []
    ref = tuple(np.ascontiguousarray(a) for a in ref)
    got = tuple(np.ascontiguousarray(a) for a in got)
    for r, g in zip(ref, got):
        w = r.view(np.uint32 if r.dtype.itemsize == 4 else np.uint8)
        v = g.view(np.uint32 if g.dtype.itemsize == 4 else np.uint8)
        m = w != v
        masks.append(m.reshape(m.shape[0], m.shape[1], -1).any(axis=2))
    any_m = masks[0] | masks[1] | masks[2]
    if not any_m.any():
        return None
    tok = int(np.nonzero(any_m.any(axis=0))[0][0])
    row = int(np.nonzero(any_m[:, tok])[0][0])
    parts = []
    for name, r, g, m in zip(names, ref, got, masks):
        if m[row, tok]:
            rr, gg = np.ascontiguousarray(r[row, tok]).reshape(-1), np.ascontiguousarray(g[row, tok]).reshape(-1)
            e = int(np.nonzero(rr.view(np.uint32 if rr.dtype.itemsize == 4 else np.uint8)
                               != gg.view(np.uint32 if gg.dtype.itemsize == 4 else np.uint8))[0][0])
            size = rr.dtype.itemsize
            parts.append(f"{name}" + (f" element {e}" if rr.size > 1 else "")
                         + f" {_elem_hex(rr.tobytes(), size, e)} vs {_elem_hex(gg.tobytes(), size, e)}")
    return f"RLPAIR_MOVED:{pair}:seq {seqs[row]} token {tok}:" + "; ".join(parts)


def _rl_flip(a):
    """The sabotage: the lowest bit of the first element, in place (on the
    array's own bytes; `reshape` of a non-contiguous view would flip a copy)."""
    if not a.flags.c_contiguous:
        raise ValueError("rlpair sabotage needs a C-contiguous array")
    a.view(np.uint8).reshape(-1)[0] ^= 1


def _rl_bounds(n, cuts):
    return [0] + [c for c in cuts if 0 < c < n] + [n]


def _probe_rlpair(fit, name, ml, Xh, sabotage):
    """The rlpair part of ONE fit: (value, error, sides). The value is a
    16-hex hash of the prompts, the sampled ids, the log-probabilities (as
    NLL) and the logits when every assertion holds, `RLPAIR_MOVED:<pair>:
    <where>:<values>` at the first that does not, or an `n/a:<reason>`
    string. A raise leaves the value None (REFUSED) with the error."""
    spec = RLPAIR[name]
    if isinstance(spec, str):
        return spec, None, []
    sides = []
    try:
        model = spec(ml, fit.est, Xh)
        prompts = np.ascontiguousarray(model["prompts"].astype(np.int32))
        samplers, (tname, trainer) = model["samplers"], model["trainer"]
        sides = [f"sampler:{s.name}" for s in samplers] + [f"trainer:{tname}"] + list(model.get("notes", []))
        n = prompts.shape[0]
        # 1. the sampler at B1 = n against the trainer at B2 = n
        ref = _rl_decode(ml, samplers[0], prompts)
        if sabotage:
            _rl_flip(ref[1])
        whole = _rl_train(ml, trainer, prompts, ref[0], [0, n])
        moved = _rl_mismatch(f"sampler B={n} vs trainer B={n}", whole, ref)
        if moved:
            return moved[:500], None, sides
        # 2. B2 = 1 and the microbatch split, against the trainer at B2 = n
        alone = [_rl_train(ml, trainer, prompts[i:i + 1], ref[0][i:i + 1], [0, 1]) for i in range(n)]
        alone = tuple(np.ascontiguousarray(np.concatenate([a[j] for a in alone])) for j in range(3))
        moved = _rl_mismatch("trainer B=1 vs trainer B=%d" % n, whole, alone)
        if moved:
            return moved[:500], None, sides
        bounds = _rl_bounds(n, RLPAIR_SPLIT)
        split = _rl_train(ml, trainer, prompts, ref[0], bounds)
        sname = ",".join(str(c - a) for a, c in zip(bounds, bounds[1:]))
        moved = _rl_mismatch(f"trainer split {sname} vs trainer B={n}", whole, split)
        if moved:
            return moved[:500], None, sides
        # 2. B1 = 1 on the first sampler, and every further sampler at B1 = n
        for i in range(n):
            one = _rl_decode(ml, samplers[0], prompts[i:i + 1])
            moved = _rl_mismatch(f"sampler B=1 vs trainer B={n}", tuple(x[i:i + 1] for x in whole), one, seqs=[i])
            if moved:
                return moved[:500], None, sides
        for s in samplers[1:]:
            other = _rl_decode(ml, s, prompts)
            moved = _rl_mismatch(f"sampler {s.name} B={n} vs trainer B={n}", whole, other)
            if moved:
                return moved[:500], None, sides
        # 3. continuous batching on the first sampler
        cont = _rl_continuous(ml, samplers[0], prompts)
        for s in range(n):
            k = cont[s][0].shape[0]
            moved = _rl_mismatch(f"continuous (join row {RLPAIR_JOINER} at token {RLPAIR_JOIN_AT}, leave row "
                                 f"{RLPAIR_LEAVER} after token {RLPAIR_LEAVE_AFTER}) vs trainer B={n}",
                                 tuple(x[s:s + 1, :k] for x in whole), tuple(c[None] for c in cont[s]), seqs=[s])
            if moved:
                return moved[:500], None, sides
        digest = hashlib.sha256()
        for a in (prompts,) + whole:
            a = np.ascontiguousarray(a)
            digest.update(f"{a.dtype.str}{a.shape}".encode())
            digest.update(a.tobytes())
        return digest.hexdigest()[:16], None, sides
    except Exception as exc:
        return None, _exc_text("rlpair", exc), sides


def _rlpair_column_verdict(values):
    """STABLE, MOVED (the hash disagreed across repeats), RLPAIR_MOVED (an
    assertion failed in a repeat), N/A or REFUSED."""
    if any(v is None for v in values):
        return "REFUSED"
    if all(v.startswith("n/a") for v in values):
        return "N/A"
    if any(v.startswith("RLPAIR_MOVED") for v in values):
        return "RLPAIR_MOVED"
    return "STABLE" if len(set(values)) == 1 else "MOVED"


# the caller-owned states, row by row. A state class's pieces whose axis 0 is
# the batch, and the cursors one batch shares (a batch cannot hold two).
_RL_ROW_PIECES = {
    "Mamba1State": (("conv_window", "h"), ()),
    "Mamba2State": (("conv_window", "h", "buffer_xbc", "buffer_dtraw"), ("buffered_tokens",)),
    "Mamba3State": (("theta", "h", "buffer_qrot", "buffer_krot", "buffer_v", "buffer_dt", "buffer_sig",
                     "buffer_adt", "pending_k", "pending_v"), ("buffered_tokens", "pending")),
}


def _rl_kv_block(st):
    """Floats per batch row in a TransformerState buffer: the linear cache is
    packed at stride cached_tokens, the ring at stride window (its class
    docstring), so row r is the flat slice [r * block, (r + 1) * block)."""
    stride = st.window if st.window > 0 else st.cached_tokens
    return st.n_kv_heads * stride * st.head_dim, st.n_kv_heads * st.capacity * st.head_dim


def _rl_take(st, rows):
    import copy
    rows = list(rows)
    kind = type(st).__name__
    if isinstance(st, np.ndarray):
        return np.ascontiguousarray(st[rows])
    new = copy.copy(st)
    if kind in _RL_ROW_PIECES:
        for p in _RL_ROW_PIECES[kind][0]:
            setattr(new, p, np.ascontiguousarray(np.asarray(getattr(st, p))[rows]))
    elif kind == "TransformerState":
        blk, cap = _rl_kv_block(st)
        for p in ("k_cache", "v_cache"):
            src = np.asarray(getattr(st, p)).reshape(-1)
            dst = np.zeros(len(rows) * cap, dtype=np.float32)
            for j, r in enumerate(rows):
                dst[j * blk:(j + 1) * blk] = src[r * blk:(r + 1) * blk]
            setattr(new, p, dst)
        new.batch_size = len(rows)
    elif kind == "SambaState":
        new.layers = [_rl_take(s, rows) for s in st.layers]
        new.batch_size = len(rows)
    else:
        raise TypeError(f"rlpair: no row layout for a {kind}")
    return new


def _rl_cat(states):
    import copy
    first = states[0]
    kind = type(first).__name__
    if isinstance(first, np.ndarray):
        return np.ascontiguousarray(np.concatenate(states, axis=0))
    new = copy.copy(first)
    if kind in _RL_ROW_PIECES:
        pieces, cursors = _RL_ROW_PIECES[kind]
        for c in cursors:
            if len(set(getattr(s, c) for s in states)) != 1:
                raise ValueError(f"rlpair: {kind}.{c} differs across the joined rows; the state holds one per batch")
        for p in pieces:
            setattr(new, p, np.ascontiguousarray(np.concatenate([np.asarray(getattr(s, p)) for s in states])))
    elif kind == "TransformerState":
        for c in ("cached_tokens", "window", "max_tokens", "n_kv_heads", "head_dim"):
            if len(set(getattr(s, c) for s in states)) != 1:
                raise ValueError(f"rlpair: TransformerState.{c} differs across the joined rows; the cache holds one per batch")
        blk, cap = _rl_kv_block(first)
        total = sum(s.batch_size for s in states)
        for p in ("k_cache", "v_cache"):
            dst = np.zeros(total * cap, dtype=np.float32)
            at = 0
            for s in states:
                src = np.asarray(getattr(s, p)).reshape(-1)
                dst[at * blk:(at + s.batch_size) * blk] = src[:s.batch_size * blk]
                at += s.batch_size
            setattr(new, p, dst)
        new.batch_size = total
    elif kind == "SambaState":
        new.layers = [_rl_cat([s.layers[i] for s in states]) for i in range(len(first.layers))]
        new.batch_size = sum(s.batch_size for s in states)
    else:
        raise TypeError(f"rlpair: no row layout for a {kind}")
    return new


def _rl_prefix_sampler(name, logits_fn):
    """A sampler for a model with NO decode state API (the byte LM): the
    state is the id prefix and every step recomputes the forward over it,
    reading the last position. That is a real sampler (no KV cache), and it
    is the only one the byte LM surface can express."""
    def prefill(prompts):
        return prompts.copy(), np.asarray(logits_fn(prompts))[:, -1, :]

    def step(tokens, st):
        st = np.ascontiguousarray(np.concatenate([st, tokens.reshape(-1, 1)], axis=1).astype(np.int32))
        return st, np.asarray(logits_fn(st))[:, -1, :]
    return _RLSampler(name, prefill, step)


def _rlpair_byte_lm(ml, e, Xh):
    """The GPU trainer (`SmallByteLanguageModelTrainer.logits`, the device
    forward its loss uses, DEVIATION 2658). Sampler 1 recomputes the prefix
    through that same entry; sampler 2 is the CPU inference class on the
    trainer's parameters (threaded, two threads), when this install has the
    host binding: sample on the CPU, train on the GPU, in one process."""
    shape = _byte_lm_shape(ml, e)
    samplers = [_rl_prefix_sampler("trainer.logits prefix recompute", e.logits)]
    notes = []
    try:
        cpu = ml.LanguageModelInference(np.asarray(e.parameters_), shape=shape, threaded=True, threads=2)
        samplers.append(_rl_prefix_sampler("LanguageModelInference threaded=True threads=2 (CPU)",
                                           lambda ids: cpu.logits(ids)))
    except ImportError as exc:
        notes.append(f"absent:CPU sampler ({str(exc)[:120]})")
    return dict(prompts=_ids(Xh, RLPAIR_SEQS, RLPAIR_PROMPT, vocab=shape.vocab_size), samplers=samplers,
                trainer=("trainer.logits", e.logits), notes=notes)


def _rlpair_byte_lm_host(ml, e, Xh):
    """LanguageModelInference on the CPU. The CPU training step
    (LanguageModelHostTrainer) exposes no logits, so the trainer side is the
    REFERENCE forward, `logits(threaded=False)`, the oracles as written; the
    samplers are the lane's own arm and the other arm."""
    shape = e.shape
    other = not e._threaded
    samplers = [_rl_prefix_sampler("lane arm prefix recompute", lambda ids: e.logits(ids)),
                _rl_prefix_sampler(f"threaded={other} prefix recompute",
                                   lambda ids: e.logits(ids, threaded=other, threads=2 if other else None))]
    return dict(prompts=_ids(Xh, RLPAIR_SEQS, RLPAIR_PROMPT, vocab=shape.vocab_size), samplers=samplers,
                trainer=("logits(threaded=False)", lambda ids: e.logits(ids, threaded=False)))


def _rlpair_block(ml, blk, table_seed, state_kw):
    """A bare block has no vocabulary, so the harness closes it into the
    smallest language model SambaStack's training forward is made of:
    `embedding_forward` (a hashed (48, d) table on [-1/8, 1/8)) -> the
    block -> the final `rms_norm_forward` (ones, eps 1e-5) -> an UNTIED
    `linear_forward` head (hashed on [-1/2, 1/2)). A head tied to the
    table decoded the prompt's last token twelve times on every block (the
    residual carries the input embedding straight to the head), which asks
    the decode path one question over and over. The sampler carries the
    block's own state (`allocate_state`, `forward(x, state)`, `step`); the
    trainer is the block's stateless `forward(x)`, the forward its
    zero-state backward recomputes and the one SambaStack's training
    forward calls."""
    T_ = ml.training
    V, d = RLPAIR_BLOCK_VOCAB, blk.d_model
    E = _hw((V, d), f"rlpair:{table_seed}:embed", -0.125, 0.125)
    W = _hw((V, d), f"rlpair:{table_seed}:head", -0.5, 0.5)
    g = np.ones(d, dtype=np.float32)

    def emb(ids):
        b, l = ids.shape
        return np.asarray(T_.embedding_forward(E, np.ascontiguousarray(ids.reshape(-1)))).reshape(b, l, d)

    def head(y):
        b, l = y.shape[0], y.shape[1]
        hn = np.asarray(T_.rms_norm_forward(np.ascontiguousarray(y), g, 1e-5))
        return np.asarray(T_.linear_forward(np.ascontiguousarray(hn.reshape(b * l, d)), W)).reshape(b, l, V)

    def prefill(prompts):
        st = blk.allocate_state(prompts.shape[0], **state_kw)
        return st, head(np.asarray(blk.forward(emb(prompts), st)))[:, -1, :]

    def step(tokens, st):
        return st, head(np.asarray(blk.step(emb(tokens.reshape(-1, 1)), st)))[:, 0, :]

    return dict(prompts=None,
                samplers=[_RLSampler("allocate_state + forward(x, state) + step", prefill, step)],
                trainer=("forward(x) stateless", lambda ids: head(np.asarray(blk.forward(emb(ids))))))


def _rlpair_block_spec(state_kw=None):
    def spec(ml, e, Xh, _kw=state_kw):
        kw = dict(max_tokens=RLPAIR_PROMPT + RLPAIR_DECODE) if _kw == "kv" else {}
        m = _rlpair_block(ml, e, type(e).__name__, kw)
        m["prompts"] = _ids(Xh, RLPAIR_SEQS, RLPAIR_PROMPT, vocab=RLPAIR_BLOCK_VOCAB)
        return m
    return spec


def _rlpair_samba(ml, e, Xh):
    """SambaStack's decode (`allocate_state`, `forward(ids, state)`, `step`,
    2026-09-15) against its training forward, `forward(ids)` with no state,
    the `_forward` its loss and gradients run. No dropout on either side."""
    total = RLPAIR_PROMPT + RLPAIR_DECODE

    def prefill(prompts):
        st = e.allocate_state(prompts.shape[0], total)
        return st, np.asarray(e.forward(prompts, st))[:, -1, :]

    def step(tokens, st):
        return st, np.asarray(e.step(tokens, st))

    return dict(prompts=_ids(Xh, RLPAIR_SEQS, RLPAIR_PROMPT, vocab=e.config.vocab),
                samplers=[_RLSampler("allocate_state + forward(ids, state) + step", prefill, step)],
                trainer=("forward(ids) stateless", lambda ids: e.forward(ids)))


_rlpair_decl(_rlpair_byte_lm, "byte-lm", "byte-lm-resident")
_rlpair_decl(_rlpair_byte_lm_host, "byte-lm-host-infer", "byte-lm-host-infer-threaded")
_rlpair_decl(_rlpair_block_spec(), "mamba1", "mamba2", "mamba3", "mamba2-dtlimit",
             "mamba1-bf16w", "mamba1-int8w", "mamba2-bf16w", "mamba2-int8w", "mamba3-bf16w", "mamba3-int8w")
_rlpair_decl(_rlpair_block_spec("kv"), "transformer", "transformer-window", "transformer-bf16w", "transformer-int8w")
_rlpair_decl(_rlpair_samba, "samba", "samba-untied-dropout-accum")


# ---------------------------------------------------------------- run / diff

def _save_load(est):
    """The (save method, load classmethod, suffix) an estimator offers.
    `save`/`load` on the forests, the boosting lanes and, since the
    classical host inference lane (2026-09-13 evening), ols, ridge, tsvd,
    logistic and pca, plus kde and svc since the kde svc host lane
    (2026-09-14), written to `.npz` as always, or
    `save_checkpoint`/`from_checkpoint` on the trainers and SambaStack
    (2026-09-13), a JSON envelope. None where it has neither."""
    if est is None:
        return None
    for s, l, suffix in (("save", "load", ".npz"), ("save_checkpoint", "from_checkpoint", ".json")):
        if callable(getattr(est, s, None)) and callable(getattr(type(est), l, None)):
            return s, l, suffix
    return None


def _has_save_load(est):
    return _save_load(est) is not None


#: PUBLIC CPU INFERENCE (lane/inference-gbdt-modes, 2026-09-15). `1` for
#: every lane, or a comma list of lanes: the infer cell is the held-out probe
#: asked of `mojolearn.host_model(<the saved file>)` (HostForest, HostGBDT or
#: a classical host model, the shipped wheel families) instead of the fitted
#: estimator, and the model cell hashes that file. The reload cell stays the
#: estimator class's own `load`, so a host answer that differs from the
#: class's reads RELOAD-MOVED here as well as DIVERGENT against the GPU
#: columns. A file host_model refuses reads REFUSED. The JSON records the
#: lanes under `host_infer`.
HOST_INFER_ENV = "MOJOLEARN_IDENTITY_HOST_INFER"


def _host_infer_lanes():
    raw = os.environ.get(HOST_INFER_ENV, "").strip()
    if not raw or raw == "0":
        return None
    return True if raw == "1" else set(p.strip() for p in raw.split(",") if p.strip())


def _host_infer_on(name):
    lanes = _host_infer_lanes()
    return lanes is True or (lanes is not None and name in lanes)


def _probe_fit_host(fit, name):
    """`_probe_fit` under HOST_INFER_ENV: save, then predict through
    `host_model`. Same return shape."""
    if not _has_save_load(fit.est):
        return None, None, None, "infer: host infer asked of an estimator with no save"
    save, load, suffix = _save_load(fit.est)
    from mojolearn._forest_host import binary_path, host_model
    try:
        with tempfile.TemporaryDirectory(prefix="identity_break_host_") as tmp:
            path = os.path.join(tmp, f"{name}{suffix}")
            getattr(fit.est, save)(path)
            model = _hfile(path)
            try:
                host = host_model(path)
                # THE BINARY THAT PREDICTED IS THE ONE NAMED (2026-09-15): host_record
                # loads MOJOLEARN_HOST_DIR's forest binding under the module name
                # _forest_host reuses from sys.modules, so a MOJOLEARN_FOREST_HOST_BINARY
                # pointing elsewhere (a sabotage build) was silently not the one asked,
                # and a forest sabotage column read IDENTICAL on a RunPod x86 pod.
                bound = getattr(getattr(host, "_binding", None), "__file__", None)
                if bound is not None and type(host).__name__ in ("HostGBDT", "HostForest") and (
                        os.path.realpath(bound) != os.path.realpath(binary_path())):
                    raise RuntimeError(f"the forest binding in this process is {bound}, "
                                       f"not MOJOLEARN_FOREST_HOST_BINARY {binary_path()}")
                infer = _h(*fit.probe(host_model(path)))
            except Exception as exc:
                return None, None, None, _exc_text("infer (host_model)", exc)
            reload = _h(*fit.probe(getattr(type(fit.est), load)(path)))
    except Exception as exc:
        return None, None, None, _exc_text("model", exc)
    if fit.model_na:
        # The same n/a as `_probe_fit` (Fit.model_na): the host answer is
        # still the infer cell, the file of caller-given bytes is not a cell.
        return infer, fit.model_na, None, None
    return infer, model, reload, None


def _probe_saved_host(fit, name):
    """The infer, model and reload columns of a CPU fit that LOADED a GPU
    column's saved file (GBDT_CTR_MODELS_ENV): the held-out probe of a fresh
    `host_model(<file>)`, the model part n/a, and the probe of a second fresh
    load. The same binary guard as `_probe_fit_host`.

    The model part is n/a on a CPU column since lane/cpu-verifier-gaps-7
    (2026-09-15): the file is the GPU column's bytes, which the CPU column
    did not write, so its hash is not a CPU cell, and the CPU gate's
    sabotage arm could never move it (every OWED part must move). The GPU
    columns still hash the file they wrote. A second load that predicts
    differently from the first is refused (the model part reads REFUSED),
    because an n/a model part skips the RELOAD-MOVED comparison."""
    from mojolearn._forest_host import binary_path, host_model
    path = fit.est.identity_saved_path
    try:
        host = host_model(path)
        bound = getattr(getattr(host, "_binding", None), "__file__", None)
        if bound is not None and os.path.realpath(bound) != os.path.realpath(binary_path()):
            raise RuntimeError(f"the forest binding in this process is {bound}, "
                               f"not MOJOLEARN_FOREST_HOST_BINARY {binary_path()}")
        infer = _h(*fit.probe(host))
    except Exception as exc:
        return None, None, None, _exc_text("infer (saved model)", exc)
    try:
        reload = _h(*fit.probe(host_model(path)))
    except Exception as exc:
        return infer, None, None, _exc_text("model (saved model)", exc)
    if reload != infer:
        return infer, None, None, (f"model (saved model): a second host_model load of {path} predicts "
                                   f"{reload}, the first {infer}")
    return infer, "n/a:gpu-saved-file (a CPU column loads the GPU column's model and writes none)", None, None


def _probe_fit(fit, name):
    """The infer and model columns of ONE fit. Returns (infer, model, reload,
    error) where infer and model are a hash or an `n/a:<reason>` string,
    reload is the held-out hash of the loaded-back file or None, and error is
    the exception text of whichever stage raised (with the stage name). A
    stage that raised leaves its column None, which reads REFUSED."""
    if not callable(fit.probe):
        return fit.probe, "n/a:no-save", None, None
    if getattr(fit.est, "identity_saved_path", None):
        return _probe_saved_host(fit, name)
    if _host_infer_on(name):
        return _probe_fit_host(fit, name)
    try:
        infer = _h(*fit.probe(_public_est(name, fit.est)))
    except Exception as exc:
        return None, None, None, _exc_text("infer", exc)
    if fit.model_na:
        return infer, fit.model_na, None, None
    if not _has_save_load(fit.est):
        return infer, "n/a:no-save", None, None
    save, load, suffix = _save_load(fit.est)
    try:
        with tempfile.TemporaryDirectory(prefix="identity_break_") as tmp:
            path = os.path.join(tmp, f"{name}{suffix}")
            try:
                getattr(fit.est, save)(path)
            except NotImplementedError:
                # the sklearn-style GBDT adapters carry save/load that refuse
                # by name (archives are not implemented; pickle keeps the
                # state), which is an absence, not a broken file
                return infer, "n/a:save-not-implemented", None, None
            model = _hfile(path)
            back = getattr(type(fit.est), load)(path)
            reload = _h(*fit.probe(_public_est(name, back)))
    except Exception as exc:
        return infer, None, None, _exc_text("model", exc)
    return infer, model, reload, None


def _column_verdict(values, reloads=None, infers=None):
    """STABLE, MOVED, RELOAD-MOVED, N/A or REFUSED for one column over the
    repeats of a cell. `values` holds one entry per repeat, a hash or an
    `n/a:` string, or None where that repeat's stage raised. `reloads` and
    `infers` are passed for the model column only; a repeat whose loaded
    file predicts differently from the model in memory is RELOAD-MOVED."""
    if any(v is None for v in values):
        return "REFUSED"
    if all(isinstance(v, str) and v.startswith("n/a") for v in values):
        return "N/A"
    if reloads is not None and any(r != i for r, i in zip(reloads, infers)):
        return "RELOAD-MOVED"
    return "STABLE" if len(set(values)) == 1 else "MOVED"


def _shown(verdict, values):
    if verdict in ("STABLE", "N/A"):
        return values[0]
    if verdict == "MOVED":
        return "MOVED " + values[0][:8]
    return verdict


def run(args):
    from mojolearn._cpu_reference import reference_training
    with reference_training():
        return _run_reference(args)


def _run_reference(args):
    import mojolearn as ml
    if getattr(args, "require_cpu", False) and ml.vendor() != "cpu":
        raise SystemExit("REFUSING: --require-cpu loaded a GPU backend")
    required = getattr(args, "require_backend", None)
    if required and ml.vendor() != required:
        raise SystemExit(f"REFUSING: requested {required} backend, loaded {ml.vendor()}")
    mode = ml.numeric_mode()
    want = os.environ.get("MOJOLEARN_NUMERIC_MODE", "fast").strip().lower() or "fast"
    if mode != want:
        raise SystemExit(f"REFUSING TO MEASURE: asked for {want!r}, loaded {mode!r}")
    if mode == "fast" and not args.allow_fast:
        raise SystemExit("REFUSING: this is a bitwise question and FAST makes no "
                         "bitwise promise; use MOJOLEARN_NUMERIC_MODE=identical")
    # PROVENANCE BEFORE THE FIRST FIT. The label and the commit are refused
    # here, in seconds, not after nine fixtures of fits.
    vendor = check_vendor_label(args.vendor if args.vendor is not None else default_vendor_label(ml))
    commit, commit_source = commit_witness()
    host = host_record(ml) if ml.vendor() == "cpu" else None
    package = dict(version=getattr(ml, "__version__", "unknown"), package_dir=os.path.dirname(ml.__file__),
                   numpy=np.__version__, python=platform.python_version(),
                   par_devices=",".join(str(d) for d in _par_devices()))
    if N != 20000:
        package["fixture_n"] = N
    if PAR_WIDE:
        package["wide"] = True
    print(f"# vendor={vendor} commit={commit} ({commit_source})"
          + (f" host.cpu_model={host['cpu_model']!r} host.column={host['column']} "
             f"host.families={sorted(host['families'])}" if host else ""))

    if args.lanes:
        unknown = [n for n in args.lanes.split(",") if n and n not in LANES]
        if unknown:
            raise SystemExit(f"REFUSING: --lanes names no lane: {unknown}; lanes are {sorted(LANES)}")
    lanes = [n for n in LANES if not args.lanes or n in args.lanes.split(",")]
    skip = set(x for x in args.skip.split(",") if x)
    lanes = [n for n in lanes if n not in skip]
    if skip:
        print(f"# SKIPPED on request: {sorted(skip)}")
    # THE RELEASE RECORD'S SCOPE (RECORD_EXCLUDED_PREFIXES, above). A
    # FULL-COLUMN run leaves these out. `--lanes` names lanes explicitly and is
    # never filtered here, so the two-device par legs and the CPU identity
    # gate, which both pass --lanes, are unaffected.
    if not args.lanes:
        out_of_scope = [n for n in lanes if n.startswith(RECORD_EXCLUDED_PREFIXES)]
        if out_of_scope:
            lanes = [n for n in lanes if n not in out_of_scope]
            print(f"# OUT OF RECORD SCOPE ({len(out_of_scope)} lanes): {sorted(out_of_scope)}")
    # NO CPU ARITHMETIC EXISTS FOR THESE (GPU_ONLY_LANES, above). Their public
    # surface refuses BY NAME on a CPU-only install, so a full-column run here
    # would record REFUSED cells, and a REFUSED cell and a passing cell are the
    # same thing in a column total. Dropped by name with the reason printed;
    # `--lanes` is never filtered, and a GPU column runs them as it runs
    # everything else, so no record loses a cell.
    if not args.lanes and host:
        no_route = [n for n in lanes if n in GPU_ONLY_LANES]
        if no_route:
            lanes = [n for n in lanes if n not in no_route]
            print(f"# NO CPU ROUTE ({len(no_route)} lanes), owed on a GPU column:")
            for n in sorted(no_route):
                print(f"#   {n}: {GPU_ONLY_LANES[n]}")
    # Broad Apple matrices are explicit diagnostics, not release requirements.
    _apple_refusal = refuse_routine_apple_column(lanes, host)
    if _apple_refusal:
        raise SystemExit(_apple_refusal)
    unknown_fixtures = set(filter(None, args.fixtures.split(","))) - set(FIXTURES)
    if unknown_fixtures:
        raise SystemExit(f"REFUSING: unknown fixtures: {sorted(unknown_fixtures)}")
    if args.repeats < 1:
        raise SystemExit("REFUSING: --repeats must be positive")
    if getattr(args, "resume", False) and not args.json:
        raise SystemExit("REFUSING: --resume requires --json")
    fixtures = [f for f in FIXTURES if not args.fixtures or f in args.fixtures.split(",")]
    data = {f: fixture(f) for f in fixtures}
    held = {f: heldout(f) for f in fixtures}
    # THE FIXTURE IS PART OF THE RESULT. Every vendor must be handed the same
    # bytes, and --diff refuses to compare columns whose fixtures differ. The
    # held-out draw is recorded under its own key so a JSON that predates it
    # still matches on `fixtures`.
    fixture_hashes = {f: dict(X=_h(X), y_clf=_h(yc), y_reg=_h(yr))
                      for f, (X, yc, yr) in data.items()}
    heldout_hashes = {f: dict(X=_h(X)) for f, X in held.items()}
    cells = {}
    batch_sabotage = os.environ.get(BATCH_SABOTAGE_ENV, "").strip() not in ("", "0")
    if args.batch_alone < 1:
        raise SystemExit("REFUSING: --batch-alone must be at least 1")
    batch_protocol = dict(alone=args.batch_alone, split=list(BATCH_SPLIT) + ["n"], prefix="1,7,full-1",
                          enabled=not args.no_batch)
    if batch_sabotage:
        print(f"# {BATCH_SABOTAGE_ENV} is ON: every whole-batch evaluation is perturbed by one "
              "low-bit flip; every batch cell with a hash MUST read BATCH_MOVED. This JSON is not evidence.")
    rlpair_sabotage = os.environ.get(RLPAIR_SABOTAGE_ENV, "").strip() not in ("", "0")
    rlpair_protocol = _rlpair_protocol(enabled=not args.no_rlpair)
    if rlpair_sabotage:
        print(f"# {RLPAIR_SABOTAGE_ENV} is ON: the first sampler log-probability of every rlpair cell is "
              "perturbed by one low-bit flip; every rlpair cell with a hash MUST read RLPAIR_MOVED. "
              "This JSON is not evidence.")
    undeclared = [n for n in lanes if n not in BATCH]
    if undeclared and not args.no_batch:
        print(f"# WARNING: no batch declaration for {undeclared}; their batch part reads n/a:UNDECLARED")
    # the opt-in parts (2026-09-15): absent from the JSON unless asked, so a
    # record that never asked diffs as it always did
    extra = [part for part, (_, _, flag, _) in EXTRA_PARTS.items() if getattr(args, flag)]
    extra_sabotage = {part: os.environ.get(EXTRA_PARTS[part][3], "").strip().lower() for part in extra}
    extra_sabotage = {k: ("" if v == "0" else v) for k, v in extra_sabotage.items()}
    for part in extra:
        if extra_sabotage[part]:
            print(f"# {EXTRA_PARTS[part][3]}={extra_sabotage[part]} is ON for the {part} part: "
                  + ("the tree is replaced by a running pair sum (recorded, not required to move)"
                     if extra_sabotage[part] == "serial" else
                     "every hashed cell MUST read BATCH_MOVED") + ". This JSON is not evidence.")

    signature = None
    if args.json:
        signature = resume_signature(args, package["package_dir"], __file__,
                                     dict(commit=commit, mode=mode, vendor=vendor,
                                          platform=platform.platform(), python=platform.python_version(),
                                          numpy=np.__version__, fixtures=fixture_hashes, heldout=heldout_hashes))
    if getattr(args, "resume", False):
        cells.update(resume_cells(args.json, signature))
        with open(args.json) as stream:
            package["bindings"] = json.load(stream).get("package", {}).get("bindings", [])
        print(f"# RESUME {len(cells)} completed cells", flush=True)

    def dump(complete):
        record = dict(mode=mode, repeats=args.repeats, platform=platform.platform(),
                      vendor=vendor, commit=commit, commit_source=commit_source,
                      heldout_seed=HELDOUT_SEED, fixtures=fixture_hashes,
                      heldout=heldout_hashes, cells=cells, complete=complete,
                      skipped=sorted(skip), batch_protocol=batch_protocol,
                      batch_sabotage=batch_sabotage, rlpair_protocol=rlpair_protocol,
                      rlpair_sabotage=rlpair_sabotage, lane_revisions=dict(LANE_REVISIONS))
        # A DEGENERATE CELL READS AS COVERAGE (2026-09-19). `verify_lanes.py`
        # asks `lane_applicability.check()` whether a lane can state its
        # proposition on the column being run, and REFUSES the ones that
        # cannot. Running THIS FILE directly bypasses that: on 2026-09-19 the
        # three host-routed lanes were run against an Apple column and each
        # produced NINE STABLE CELLS, which in any summary is indistinguishable
        # from coverage. The column now says so about itself, so a reader and
        # `_verify_reference.admit` can both see it without re-deriving the
        # rule. Recorded, not refused: a record that names its own weakness is
        # worth more than one that was never written.
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import lane_applicability as _la
            _la.use_harness(sys.modules[__name__])   # never exec this file twice
            # FROM THE LOADED BACKEND, NOT A FLAG. `identity_break.py` has no
            # `--backend`; the first draft of this check read one and was
            # therefore INERT -- it could never fire, which is the same defect
            # it exists to catch. `ml.vendor()` is what actually ran.
            _col = dict(cpu="cpu-host", metal="apple-metal",
                        cuda="nvidia-1gpu", hip="amd-1gpu").get(str(ml.vendor()))
            if _col:
                _deg = set(_la.degenerate(_col))
                _hit = sorted({k.split("/", 1)[0] for k in cells} & _deg)
                if _hit:
                    record["degenerate_lanes"] = _hit
                    record["degenerate_column"] = _col
        except Exception as exc:        # never lose a column over this check
            record["degenerate_check_error"] = f"{type(exc).__name__}: {exc}"[:200]
        # WAS THIS GPU RUN SERIALISED? (2026-09-19) Concurrent Metal jobs on one
        # M4 return NaN, constant and zero output -- a sabotage-free way to get
        # a wrong answer that still hashes stably. `verify_lanes.py` routes
        # every backend through `tools/mac_slot.py`, which takes
        # /tmp/mojolearn-metal-slot. RUNNING THIS FILE DIRECTLY TAKES NO LOCK,
        # and on 2026-09-19 a job doing exactly that held the GPU for 18
        # minutes while the slot read `free` -- so a second agent could not
        # tell it was there. The column now records what the lock said, so a
        # reader can see whether the run was serialised instead of assuming
        # it. Recorded, never refused: a legitimate diagnostic run is allowed
        # to be unlocked, it just should not be mistaken for a clean column.
        try:
            if str(ml.vendor()) != "cpu":
                _lk = pathlib.Path(os.environ.get("MOJOLEARN_METAL_LOCK",
                                                 "/tmp/mojolearn-metal-slot"))
                _pid = (_lk / "pid").read_text().strip() if (_lk / "pid").exists() else ""
                # THREE STATES, AND THE MIDDLE ONE IS THE HAZARD. `held by
                # another` means a second GPU job is live while this one
                # runs -- concurrent Metal on one M4 returns NaN, constant
                # and zero output that still hashes stably. `mac_slot.py`
                # marks its own children with MOJOLEARN_SLOT_TOKEN, so a run
                # under the slot is distinguishable from one merely racing it.
                _mine = bool(os.environ.get("MOJOLEARN_SLOT_TOKEN"))
                if not _lk.exists():
                    record["gpu_slot"] = "FREE (run not serialised)"
                elif _mine:
                    record["gpu_slot"] = f"held by this run (pid={_pid})" if _pid else "held by this run"
                else:
                    record["gpu_slot"] = (
                        f"HELD BY ANOTHER RUN (pid={_pid}) -- concurrent GPU work, "
                        "these cells are not evidence")
        except Exception as exc:
            record["gpu_slot_error"] = f"{type(exc).__name__}: {exc}"[:200]
        host_infer = _host_infer_lanes()
        if host_infer is not None:
            record["host_infer"] = "all" if host_infer is True else sorted(host_infer)
        for part in extra:
            record[f"{part}_protocol"] = _part_protocol(part, args.batch_alone)
            record[f"{part}_sabotage"] = extra_sabotage[part] or False
        if host is not None:
            record["host"] = host
        # every binding this process loaded, hashed, so the column is tied
        # to the BYTES it ran and not only to a commit (2026-09-14); a
        # column from an installed wheel and one from a source build of the
        # same commit are different evidence, and the 0.8.5 release legs
        # caught two packaging regressions a source column could not see
        try:
            from mojolearn._verify import binding_artifacts
            artifacts = {b["module"]: b for b in package.get("bindings", [])}
            artifacts.update({b["module"]: dict(module=b["module"], sha256=b["sha256"], size=b["size"])
                              for b in binding_artifacts()})
            package["bindings"] = list(artifacts.values())
        except Exception as exc:                        # never lose a column over provenance
            package["bindings_error"] = f"{type(exc).__name__}: {exc}"[:200]
        record["package"] = package
        record["resume_signature"] = signature
        atomic_json(args.json, record)
        # The untruncated text of every refusal, beside the JSON. A cell's
        # own error field is capped; this file never is.
        write_error_sidecar(args.json)

    print(f"# identity_break  mode={mode}  {time.strftime('%Y-%m-%d %H:%M:%S')}  "
          f"{platform.platform()}")
    print("# per lane: the train row, then an `infer` row (held-out rows), a `model` row "
          "(saved bytes) and a `batch` row (whole batch vs rows alone, a split and prefixes) where the lane has them")
    W = 22
    print(f"| {'lane':<{W}} | " + " | ".join(f"{f:<16}" for f in fixtures) + " |")
    print(f"|{'-'*(W+2)}|" + "|".join("-" * 18 for _ in fixtures) + "|")
    na = {}
    for name in lanes:
        row, row_infer, row_model, row_batch, row_rl = [], [], [], [], []
        row_extra = {}
        for f in fixtures:
            if f"{name}/{f}" in cells:
                cached = cells[f"{name}/{f}"]
                display = cached["hashes"][0] if cached["verdict"] == "STABLE" else cached["verdict"]
                row.append(f"{display:<16}")
                for col, target in (("infer", row_infer), ("model", row_model), ("batch", row_batch),
                                    ("rlpair", row_rl)) + tuple((part, row_extra.setdefault(part, [])) for part in extra):
                    if col == "rlpair" and name not in RLPAIR:
                        continue
                    verdict = cached.get(f"{col}_verdict", "REFUSED")
                    target.append(_shown(verdict, cached.get(col, [])))
                print(f"# REUSED {name}/{f} {cached['verdict']}", flush=True)
                continue
            timer = CellTimer(f"{name}/{f}")
            X, yc, yr = data[f]
            hs, parts, err = [], [], None
            infers, models, reloads, errs2 = [], [], [], []
            oracle_errors = []
            batches, errs3 = [], []
            extras = {part: dict(values=[], errors=[], notes=None) for part in extra}
            rls, errs4, rl_sides = [], [], None
            for repeat in range(args.repeats):
                try:
                    # each fit gets its own held-out bytes, so a lane cannot
                    # hand the next fit rows it wrote to
                    global _DUMP_TAG
                    _DUMP_TAG = f"{name}/{f}/{repeat}"
                    with timer.stage(f"repeat={repeat + 1}/train"):
                        p = LANES[name](ml, X, yc, yr, held[f].copy())
                    parts.append(dict(p))
                    hs.append(_train_hash(p))
                except NumericalMismatch as exc:
                    # Repeat the actual computation and retain its bytes. An
                    # oracle mismatch is a failed numerical check, never a
                    # STABLE reference and never an arbitrary REFUSED control.
                    parts.append(exc.parts)
                    hs.append(_train_hash(exc.parts))
                    oracle_errors.append(str(exc))
                    continue
                except Exception as exc:
                    err = _exc_text(None, exc)
                    if args.verbose:
                        traceback.print_exc()
                    break
                with timer.stage(f"repeat={repeat + 1}/infer-save-reload"):
                    inf, mod, rel, err2 = _probe_fit(p, name)
                infers.append(inf); models.append(mod); reloads.append(rel)
                if err2:
                    errs2.append(_clip_error(err2, f"{name}/{f} infer-model"))
                    if args.verbose:
                        traceback.print_exc()
                # the batch part runs after the infer and model columns so it
                # cannot change what they hash; its own copy of the held-out rows
                if args.no_batch:
                    bat, err3 = "n/a:skipped (--no-batch)", None
                else:
                    with timer.stage(f"repeat={repeat + 1}/batch"):
                        bat, err3 = _probe_batch(p, name, ml, held[f].copy(), args.batch_alone, batch_sabotage)
                batches.append(bat)
                if err3:
                    errs3.append(_clip_error(err3, f"{name}/{f} batch"))
                    if args.verbose:
                        print(err3)
                for part in extra:
                    with timer.stage(f"repeat={repeat + 1}/{part}"):
                        val, errp, notes = _probe_part(part, p, name, ml, held[f].copy(), args.batch_alone,
                                                       extra_sabotage[part])
                    extras[part]["values"].append(val)
                    if errp:
                        extras[part]["errors"].append(_clip_error(errp, f"{name}/{f} {part}"))
                        if args.verbose:
                            print(errp)
                    if extras[part]["notes"] is None:
                        extras[part]["notes"] = notes
                # the rlpair part last, on the lanes that declare it; a lane
                # that does not carries no rlpair key at all
                if name in RLPAIR:
                    if args.no_rlpair:
                        rl, err4, sd = "n/a:skipped (--no-rlpair)", None, []
                    else:
                        with timer.stage(f"repeat={repeat + 1}/rlpair"):
                            rl, err4, sd = _probe_rlpair(p, name, ml, held[f].copy(), rlpair_sabotage)
                    rls.append(rl)
                    rl_sides = rl_sides if rl_sides is not None else sd
                    if err4:
                        errs4.append(_clip_error(err4, f"{name}/{f} rlpair"))
                        if args.verbose:
                            print(err4)
            if err:
                cell = dict(verdict="REFUSED", error=_clip_error(err, f"{name}/{f} train"),
                            hashes=hs, parts=parts)
                shown = "REFUSED"
            elif oracle_errors:
                cell = dict(verdict="DIVERGENT", hashes=hs, parts=parts, oracle_errors=oracle_errors)
                shown = "DIVERGENT"
            elif len(set(hs)) == 1:
                cell = dict(verdict="STABLE", hashes=hs, parts=parts)
                shown = hs[0]
            else:
                cell = dict(verdict="MOVED", hashes=hs, parts=parts)
                shown = "MOVED " + hs[0][:8]
            # the two new columns; a REFUSED train column has no fit to probe
            if not err and not oracle_errors:
                iv = _column_verdict(infers)
                has_reload = all(r is not None for r in reloads)
                mv = _column_verdict(models, reloads if has_reload else None, infers)
                cell.update(infer=infers, infer_verdict=iv, model=models, model_verdict=mv)
                if any(r is not None for r in reloads):
                    cell["reload"] = reloads
                if errs2:
                    cell["probe_error"] = errs2[0]
                bv = _batch_column_verdict(batches)
                cell.update(batch=batches, batch_verdict=bv)
                if errs3:
                    cell["batch_error"] = errs3[0]
                row_infer.append(_shown(iv, infers))
                row_model.append(_shown(mv, models))
                row_batch.append("BATCH_MOVED" if bv == "BATCH_MOVED" else _shown(bv, batches))
                if name in RLPAIR:
                    rv = _rlpair_column_verdict(rls)
                    cell.update(rlpair=rls, rlpair_verdict=rv, rlpair_sides=rl_sides or [])
                    if errs4:
                        cell["rlpair_error"] = errs4[0]
                    row_rl.append("RLPAIR_MOVED" if rv == "RLPAIR_MOVED" else _shown(rv, rls))
                    if rv == "N/A":
                        na.setdefault(f"{name} rlpair", rls[0])
                if iv == "N/A":
                    na.setdefault(f"{name} infer", infers[0])
                if mv == "N/A":
                    na.setdefault(f"{name} model", models[0])
                if bv == "N/A":
                    na.setdefault(f"{name} batch", batches[0])
                for part in extra:
                    vals = extras[part]["values"]
                    pv = _batch_column_verdict(vals)
                    cell.update({part: vals, f"{part}_verdict": pv})
                    if extras[part]["errors"]:
                        cell[f"{part}_error"] = extras[part]["errors"][0]
                    if extras[part]["notes"]:
                        cell[f"{part}_notes"] = extras[part]["notes"]
                    row_extra.setdefault(part, []).append("BATCH_MOVED" if pv == "BATCH_MOVED" else _shown(pv, vals))
                    if pv == "N/A":
                        na.setdefault(f"{name} {part}", vals[0])
            else:
                row_infer.append("REFUSED"); row_model.append("REFUSED"); row_batch.append("REFUSED")
                for part in extra:
                    row_extra.setdefault(part, []).append("REFUSED")
                if name in RLPAIR:
                    row_rl.append("REFUSED")
            cell["timing"] = timer.result()
            cells[f"{name}/{f}"] = cell
            if args.json:
                dump(False)
            print(f"# CELL {name}/{f} {cell['verdict']} {cell['timing']['total_seconds']:.3f}s", flush=True)
            row.append(f"{shown:<16}")
        print(f"| {name:<{W}} | " + " | ".join(row) + " |", flush=True)
        for label, r in (("infer", row_infer), ("model", row_model), ("batch", row_batch), ("rlpair", row_rl)) + tuple(row_extra.items()):
            if any(not s.startswith("n/a") for s in r):
                print(f"| {name + ' ' + label:<{W}} | " + " | ".join(f"{s:<16}" for s in r) + " |", flush=True)

    refused = {k: v["error"] for k, v in cells.items() if v["verdict"] == "REFUSED"}
    moved = [k for k, v in cells.items() if v["verdict"] in ("MOVED", "DIVERGENT")]
    moved2 = [k for k, v in cells.items()
              if v.get("infer_verdict") == "MOVED" or v.get("model_verdict") in ("MOVED", "RELOAD-MOVED")]
    refused2 = {k: v["probe_error"] for k, v in cells.items() if v.get("probe_error")}
    moved3 = [k for k, v in cells.items() if v.get("batch_verdict") == "MOVED"]
    batch_moved = [k for k, v in cells.items() if v.get("batch_verdict") == "BATCH_MOVED"]
    refused3 = {k: v["batch_error"] for k, v in cells.items() if v.get("batch_error")}
    moved4 = [k for k, v in cells.items() if v.get("rlpair_verdict") == "MOVED"]
    rl_moved = [k for k, v in cells.items() if v.get("rlpair_verdict") == "RLPAIR_MOVED"]
    refused4 = {k: v["rlpair_error"] for k, v in cells.items() if v.get("rlpair_error")}
    print()
    print(f"cells={len(cells)} stable={len(cells)-len(refused)-len(moved)} "
          f"moved={len(moved)} refused={len(refused)}")
    for col in ("infer", "model", "batch", "rlpair") + tuple(extra):
        vs = [v.get(f"{col}_verdict") for v in cells.values() if v.get(f"{col}_verdict")]
        if col == "rlpair" and not vs:
            continue
        print(f"{col}: " + " ".join(f"{k.lower()}={vs.count(k)}" for k in
                                    ("STABLE", "MOVED", "RELOAD-MOVED", "BATCH_MOVED", "RLPAIR_MOVED", "REFUSED", "N/A")
                                    if vs.count(k)))
    if na:
        print("n/a: " + ", ".join(f"{k} {v}" for k, v in sorted(na.items())))
    for k in moved:
        print(f"MOVED   {k}: {cells[k]['hashes']}")
    for k in moved2:
        c = cells[k]
        print(f"MOVED   {k}: infer={c['infer_verdict']} {c['infer']} model={c['model_verdict']} {c['model']}"
              + (f" reload={c['reload']}" if c.get("reload") else ""))
    for k, e in refused.items():
        print(f"REFUSED {k}: {e}")
    for k, e in refused2.items():
        print(f"REFUSED {k}: {e}")
    for k in moved3:
        print(f"MOVED   {k}: batch={cells[k]['batch']}")
    # BATCH_MOVED is a defect under IDENTICAL only: FAST promises no bits and
    # DETERMINISTIC hands the GEMM to the vendor kernel, which may tile by
    # batch (checks/batch_invariance_check.mojo's tier table), so both RECORD
    batch_fails = bool(batch_moved) and mode == "identical"
    for k in batch_moved:
        first = next(b for b in cells[k]["batch"] if b.startswith("BATCH_MOVED"))
        print(f"{'BATCH_MOVED' if batch_fails else 'BATCH_MOVED (' + mode + ', recorded, not a failure)'} {k}: {first}")
    for k, e in refused3.items():
        print(f"REFUSED {k}: {e}")
    extra_fail = False
    for part in extra:
        for k, c in cells.items():
            pv = c.get(f"{part}_verdict")
            if pv == "MOVED":
                print(f"MOVED   {k}: {part}={c[part]}")
                extra_fail = True
            elif pv == "BATCH_MOVED":
                first = next(b for b in c[part] if b and b.startswith("BATCH_MOVED"))
                fails = mode == "identical"
                extra_fail = extra_fail or fails
                print(f"{'BATCH_MOVED' if fails else 'BATCH_MOVED (' + mode + ', recorded, not a failure)'} "
                      f"{k} {part}: {first}")
            if c.get(f"{part}_error"):
                print(f"REFUSED {k} {part}: {c[f'{part}_error']}")
            for note in c.get(f"{part}_notes") or []:
                if "REPORTED" in note:
                    print(f"REPORTED {k} {part}: {note}")
    for k in moved4:
        print(f"MOVED   {k}: rlpair={cells[k]['rlpair']}")
    # RLPAIR_MOVED is a defect under IDENTICAL only, as BATCH_MOVED is
    rl_fails = bool(rl_moved) and mode == "identical"
    for k in rl_moved:
        first = next(b for b in cells[k]["rlpair"] if b.startswith("RLPAIR_MOVED"))
        print(f"{'RLPAIR_MOVED' if rl_fails else 'RLPAIR_MOVED (' + mode + ', recorded, not a failure)'} {k}: {first}")
    for k, e in refused4.items():
        print(f"REFUSED {k}: {e}")
    if args.json:
        dump(True)
        print(f"wrote {args.json}")
        sidecar = write_error_sidecar(args.json)
        if sidecar:
            print(f"wrote {sidecar} ({len(_FULL_ERRORS)} refusal(s), untruncated)")
    elif _FULL_ERRORS:
        # No JSON to sit beside: the whole point is that nothing is thrown
        # away, so the full text goes to the log the run is read from.
        print()
        print(f"# FULL REFUSAL TEXT ({len(_FULL_ERRORS)}); pass --json to have it written beside the record")
        for key in sorted(_FULL_ERRORS):
            print(f"===== {key} =====")
            print(_FULL_ERRORS[key])
    refused_any = refused or refused2 or refused3 or refused4 or any(
        c.get(f"{part}_error") for c in cells.values() for part in extra)
    return 1 if (moved or moved2 or moved3 or batch_fails or moved4 or rl_fails or extra_fail
                 or (getattr(args, "fail_on_refused", False) and refused_any)) else 0


def _diff_column(cols, k, col):
    """One row of the second table: the per-column shown values and the
    verdict for `col` (infer, model or batch) of cell `k`. A column that
    lacks the key is `(no column)` and is never compared: a JSON written
    before the batch part (2026-09-14) has no `batch_verdict`, so its batch
    cells read NOT-COMPARED, or are counted as not compared when no JSON
    carries them, never DIVERGENT. A BATCH_MOVED from an IDENTICAL column is
    BATCH_MOVED (a failure); from a FAST or DETERMINISTIC column only, it is
    BATCH_MOVED-RECORDED (reported, not a failure)."""
    shown, hashes, missing, na, identical_bm = [], [], False, False, False
    for _, j in cols:
        c = j["cells"].get(k)
        if c is None:
            shown.append("(not run)"); continue
        if c.get("verdict") == "REFUSED":
            shown.append("REFUSED"); continue
        if c.get("verdict") == "DIVERGENT":
            shown.append("DIVERGENT"); hashes.append("DIVERGENT"); continue
        if f"{col}_verdict" not in c:
            shown.append("(no column)"); missing = True; continue
        v = c[f"{col}_verdict"]
        if v == "N/A":
            shown.append(c[col][0]); na = True; continue
        if v in ("MOVED", "RELOAD-MOVED", "REFUSED", "BATCH_MOVED", "RLPAIR_MOVED"):
            shown.append(v); hashes.append(v)
            identical_bm = identical_bm or (v in ("BATCH_MOVED", "RLPAIR_MOVED") and j.get("mode") == "identical")
            continue
        shown.append(c[col][0]); hashes.append(c[col][0])
    real = [h for h in hashes if h not in ("DIVERGENT", "MOVED", "RELOAD-MOVED", "REFUSED", "BATCH_MOVED", "RLPAIR_MOVED")]
    if "DIVERGENT" in hashes:
        verdict = "DIVERGENT"
    elif "RELOAD-MOVED" in hashes:
        verdict = "RELOAD-MOVED"
    elif "MOVED" in hashes:
        verdict = "MOVED"
    elif "BATCH_MOVED" in hashes:
        verdict = "BATCH_MOVED" if identical_bm else "BATCH_MOVED-RECORDED"
    elif "RLPAIR_MOVED" in hashes:
        verdict = "RLPAIR_MOVED" if identical_bm else "RLPAIR_MOVED-RECORDED"
    elif len(real) >= 2:
        verdict = f"IDENTICAL x{len(real)}" if len(set(real)) == 1 else "DIVERGENT"
    elif missing:
        verdict = "NOT-COMPARED"
    elif len(real) == 1:
        verdict = "ONE-COLUMN"
    elif na:
        verdict = "N/A"
    else:
        verdict = "REFUSED"
    return verdict, shown


def _real_count(verdict):
    """How many real hashes a diff verdict rests on: `IDENTICAL xK` is K,
    ONE-COLUMN is 1, REFUSED is 0; DIVERGENT, MOVED, RELOAD-MOVED and
    BATCH_MOVED already fail on their own and are not counted here; N/A,
    NOT-COMPARED and BATCH_MOVED-RECORDED are None (no requirement applies,
    as for a FAST column's infer and model cells)."""
    if verdict.startswith("IDENTICAL x"):
        return int(verdict.split("x", 1)[1])
    if verdict == "ONE-COLUMN":
        return 1
    if verdict == "REFUSED":
        return 0
    return None


#: the per-column verdicts that are already a failure on their own; a cell
#: carrying one is never OWED
_OWED_BLOCKING = ("DIVERGENT", "MOVED", "RELOAD-MOVED", "REFUSED", "BATCH_MOVED", "RLPAIR_MOVED")


def _owed_status(cols, key, col):
    """Whether a cell part that rests on fewer real hashes than
    --require-columns demands is OWED rather than short (2026-09-15,
    lane/cpu-gate-owed-cells). GPU records are taken only at PyPI releases,
    so a CPU lane can carry a cell part no committed GPU record hashes yet.
    DERIVED, never hand-listed: the part is OWED only when
      - every CPU column (a JSON with `host`) hashes it STABLE over at least
        two repeats, and at least one CPU column is given;
      - every other column either hashes it or has NO hash for it: the cell
        is absent (a lane not in that record), the part key is absent, or
        the part is n/a;
      - no column REFUSED the cell or the part and none reads MOVED,
        RELOAD-MOVED, BATCH_MOVED or RLPAIR_MOVED.
    Agreement among the columns that do hash it is the caller's verdict:
    DIVERGENT is never short and never OWED. Returns (missing column names,
    "") when OWED, else ([], the reason it is not)."""
    missing, cpu_seen = [], False
    for name, j in cols:
        c = j["cells"].get(key)
        is_cpu = bool(j.get("host"))
        if c is None:
            if is_cpu:
                return [], f"CPU column {name} has no cell"
            missing.append(name); continue
        if c.get("verdict") == "REFUSED":
            return [], f"column {name} REFUSED the cell"
        if col == "train":
            v, values = c.get("verdict"), c.get("hashes") or []
        elif f"{col}_verdict" not in c:
            if is_cpu:
                return [], f"CPU column {name} carries no {col} part"
            missing.append(name); continue
        else:
            v, values = c[f"{col}_verdict"], c.get(col) or []
        if v in _OWED_BLOCKING:
            return [], f"column {name} reads {v}"
        if v == "N/A":
            if is_cpu:
                return [], f"CPU column {name} reads N/A"
            missing.append(name); continue
        if is_cpu:
            cpu_seen = True
            if v != "STABLE" or len(values) < 2 or len(set(values)) != 1:
                return [], f"CPU column {name} is not STABLE over two or more repeats ({v}, {len(values)} repeat(s))"
    if not cpu_seen:
        return [], "no CPU column hashes it"
    if not missing:
        return [], "no column lacks a hash"
    return missing, ""


def diff(paths, require_columns=0, require_lanes=None, owed_json=None):
    cols = []
    for p in paths:
        with open(p) as fh:
            j = json.load(fh)
        cols.append((j.get("vendor") or os.path.basename(p), j))
    for name, j in cols:
        stale = stale_revision_lanes(j)
        dropped = [k for k in j["cells"] if k.split("/")[0] in stale]
        if dropped:
            for k in dropped:
                del j["cells"][k]
            print(f"NOTE: column {name} hashed {', '.join(stale)} at an older lane revision "
                  f"(LANE_REVISIONS); its {len(dropped)} cell(s) there are not compared and read as absent")
    keys = sorted(set(k for _, j in cols for k in j["cells"]))
    if require_lanes:
        # --lanes SCOPES the diff (2026-09-14 night): the verdicts, the
        # summaries and the exit status are over the named lanes only. Until
        # then it scoped only --require-columns, so a DIVERGENT cell on a lane
        # a CPU column does not cover failed the CPU gate's covered-lanes diff.
        outside = len(set(k.split("/")[0] for k in keys) - set(require_lanes))
        keys = [k for k in keys if k.split("/")[0] in set(require_lanes)]
        print(f"NOTE: --lanes scopes this diff to {len(set(require_lanes))} lane(s); {outside} other lane(s) the JSONs carry are not compared")
    names = [c for c, _ in cols]
    if require_columns and require_columns > len(cols):
        print(f"REQUIRE FAIL: --require-columns {require_columns} with {len(cols)} JSONs given")
    required = set(require_lanes) if require_lanes else (set(k.split("/")[0] for k in keys) if require_columns else set())
    short, owed = [], []
    if owed_json and not require_columns:
        raise SystemExit("REFUSING: --owed-json needs --require-columns (OWED is a cell short of that count)")

    def require(key, col, verdict):
        """Records a short cell part and returns the verdict to count and
        print: `OWED xK` for a part --owed-json admits (K real hashes, all
        equal), else the verdict unchanged."""
        if not require_columns or key.split("/")[0] not in required:
            return verdict
        n = _real_count(verdict)
        if n is not None and n < require_columns:
            why = ""
            if owed_json and n > 0:
                missing_cols, why = _owed_status(cols, key, col)
                if missing_cols:
                    lane_name, fixture = key.split("/", 1)
                    owed.append(dict(lane=lane_name, fixture=fixture, part=col, missing=missing_cols,
                                     present=[nm for nm, _ in cols if nm not in missing_cols], real_hashes=n))
                    return f"OWED x{n}"
            short.append((key, col, verdict, n, why))
        return verdict

    fixture_ns = set(((j.get("package") or {}).get("fixture_n", 20000), bool((j.get("package") or {}).get("wide")))
                     for _, j in cols)
    if len(fixture_ns) > 1:
        raise SystemExit(f"REFUSING TO DIFF: the columns hash different fixture sizes {sorted(fixture_ns)} "
                         f"({FIXTURE_N_ENV}, {WIDE_ENV}); their cells are different questions")
    for n_, (_, j) in zip(names, cols):
        if j.get("host"):
            h = j["host"]
            print(f"NOTE: column {n_} is a CPU column: cpu_model={h.get('cpu_model')!r} "
                  f"column={h.get('column')} families={sorted(h.get('families', {}))} commit={j.get('commit')}")
    for n, (_, j) in zip(names, cols):
        if j.get("batch_sabotage"):
            print(f"NOTE: column {n} ran with {BATCH_SABOTAGE_ENV} ON; its batch cells are a sabotage "
                  "run and must read BATCH_MOVED, they are not evidence")
        for part in EXTRA_PARTS:
            if j.get(f"{part}_sabotage"):
                print(f"NOTE: column {n} ran the {part} part with sabotage {j[f'{part}_sabotage']!r}; "
                      "its cells are not evidence")
        if j.get("rlpair_sabotage"):
            print(f"NOTE: column {n} ran with {RLPAIR_SABOTAGE_ENV} ON; its rlpair cells are a sabotage "
                  "run and must read RLPAIR_MOVED, they are not evidence")
    protocols = set(json.dumps(j.get("batch_protocol"), sort_keys=True) for _, j in cols if j.get("batch_protocol"))
    if len(protocols) > 1:
        print(f"NOTE: the columns ran different batch protocols {sorted(protocols)}; the batch hash is "
              "the whole-batch output and compares, but the verdicts rest on different row counts")
    rlp = set(json.dumps({k: v for k, v in j["rlpair_protocol"].items() if k != "enabled"}, sort_keys=True)
              for _, j in cols if j.get("rlpair_protocol"))
    if len(rlp) > 1:
        print(f"NOTE: the columns ran different rlpair protocols {sorted(rlp)}; the rlpair hash covers the "
              "decoded tokens, so a protocol difference reads DIVERGENT for that reason and no other")
    for n, (_, j) in zip(names, cols):
        if not j.get("complete", True):
            print(f"NOTE: column {n} is INCOMPLETE (the run was killed); lanes after the last one written are absent, not clean")
        if j.get("skipped"):
            print(f"NOTE: column {n} SKIPPED {j['skipped']} on request")
    # Refuse to blame the library for a fixture the vendors did not share.
    fx = [j.get("fixtures") for _, j in cols]
    if all(fx):
        for f in sorted(set(k for x in fx for k in x)):
            vals = [json.dumps(x.get(f), sort_keys=True) for x in fx]
            if len(set(vals)) != 1:
                print(f"FIXTURE MISMATCH on {f!r}: the columns were not handed the same bytes; "
                      f"cells on it are the fixture's divergence, not the library's")
                for n, x in zip(names, fx):
                    print(f"    {n}: {x.get(f)}")
    else:
        print("NOTE: a column carries no fixture hashes (older harness); fixture equality unchecked")
    hx = [j.get("heldout") for _, j in cols]
    if all(hx):
        for f in sorted(set(k for x in hx for k in x)):
            vals = [json.dumps(x.get(f), sort_keys=True) for x in hx]
            if len(set(vals)) != 1:
                print(f"HELD-OUT MISMATCH on {f!r}: the columns were not handed the same held-out bytes; "
                      f"infer and model cells on it are the fixture's divergence, not the library's")
                for n, x in zip(names, hx):
                    print(f"    {n}: {x.get(f)}")
    elif any(hx):
        print("NOTE: a column carries no held-out hashes (it predates the infer/model columns); "
              "its infer and model cells read NOT-COMPARED")
    print(f"| {'lane/fixture':<28} | {'verdict':<10} | " + " | ".join(f"{n:<16}" for n in names) + " |")
    print(f"|{'-'*30}|{'-'*12}|" + "|".join("-" * 18 for _ in names) + "|")
    bad, counts = 0, {}
    for k in keys:
        vals, shown = [], []
        for _, j in cols:
            c = j["cells"].get(k)
            if c is None:
                shown.append("(not run)"); continue
            if c["verdict"] == "REFUSED":
                shown.append("REFUSED"); continue
            if c["verdict"] in ("MOVED", "DIVERGENT"):
                shown.append(c["verdict"]); vals.append(c["verdict"]); continue
            shown.append(c["hashes"][0]); vals.append(c["hashes"][0])
        ran = [v for v in vals]
        if "DIVERGENT" in ran:
            verdict = "DIVERGENT"
        elif "MOVED" in ran:
            verdict = "MOVED"
        elif len(ran) < 2:
            verdict = "ONE-COLUMN" if len(ran) == 1 else "REFUSED"
        elif len(set(ran)) == 1:
            verdict = f"IDENTICAL x{len(ran)}"
        else:
            verdict = "DIVERGENT"
        if verdict in ("MOVED", "DIVERGENT"):
            bad += 1
        if verdict == "DIVERGENT":
            # localise: which named part disagrees
            per = {}
            for n, (_, j) in zip(names, cols):
                c = j["cells"].get(k)
                if c and c.get("parts"):
                    for pk, pv in c["parts"][0].items():
                        per.setdefault(pk, []).append(pv)
            diverging = [pk for pk, pv in per.items() if len(set(pv)) > 1]
            agreeing = [pk for pk, pv in per.items() if len(set(pv)) == 1]
            if per:
                shown[0] = f"parts differ: {','.join(diverging) or '?'}; agree: {','.join(agreeing) or '-'}"
        verdict = require(k, "train", verdict)
        counts[verdict.split(" ")[0]] = counts.get(verdict.split(" ")[0], 0) + 1
        print(f"| {k:<28} | {verdict:<10} | " + " | ".join(f"{s:<16}" for s in shown) + " |")
    print()
    print("summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    # THE SECOND TABLE: infer (held-out rows), model (saved bytes) and batch
    # (whole batch vs rows alone, a split and prefixes), cell by cell,
    # printed where at least one column carries the key. Cells no column
    # carries are counted as not compared, never as divergent.
    # Two summary lines, `summary (infer/model):` exactly as it read before
    # the batch part (the CPU gate greps it) and `summary (batch):` beside it;
    # until 2026-09-14 evening the batch branch merged the two into one
    # `summary (infer/model/batch):` line, which no committed gate grep matches.
    rows = []
    # the opt-in parts (2026-09-15) are their own groups and print only where
    # a column carries them, so a diff of records that never asked reads as
    # it always did
    # A THIRD LINE, `summary (rlpair):` (2026-09-15), over the lanes that
    # declare the rlpair part (RLPAIR) or whose cells carry it; the other
    # lanes have no rlpair question and are not counted at all.
    extra_cols = tuple(part for part in EXTRA_PARTS
                       if any(f"{part}_verdict" in c for _, j in cols for c in j["cells"].values()))
    uncarried = {"infer/model": 0, "batch": 0, "rlpair": 0, **{part: 0 for part in extra_cols}}
    counts2 = {"infer/model": {}, "batch": {}, "rlpair": {}, **{part: {} for part in extra_cols}}
    for k in keys:
        for col in ("infer", "model", "batch", "rlpair") + extra_cols:
            group = col if col in extra_cols or col in ("batch", "rlpair") else "infer/model"
            carried = any(f"{col}_verdict" in (j["cells"].get(k) or {}) for _, j in cols)
            if not carried:
                if col != "rlpair" or k.split("/")[0] in RLPAIR:
                    uncarried[group] += 1
                continue
            verdict, shown = _diff_column(cols, k, col)
            if verdict in ("MOVED", "DIVERGENT", "RELOAD-MOVED", "BATCH_MOVED", "RLPAIR_MOVED"):
                bad += 1
            verdict = require(k, col, verdict)
            c2 = counts2[group]
            c2[verdict.split(" ")[0]] = c2.get(verdict.split(" ")[0], 0) + 1
            rows.append((k, col, verdict, shown))
    print()
    if rows:
        print(f"| {'lane/fixture':<28} | {'column':<6} | {'verdict':<12} | " + " | ".join(f"{n:<16}" for n in names) + " |")
        print(f"|{'-'*30}|{'-'*8}|{'-'*14}|" + "|".join("-" * 18 for _ in names) + "|")
        for k, col, verdict, shown in rows:
            print(f"| {k:<28} | {col:<6} | {verdict:<12} | " + " | ".join(f"{s:<16}" for s in shown) + " |")
        print()
    for group in ("infer/model", "batch", "rlpair") + extra_cols:
        c2 = counts2[group]
        if uncarried[group]:
            c2["NOT-COMPARED"] = c2.get("NOT-COMPARED", 0) + uncarried[group]
            print(f"{group}: {uncarried[group]} column cells not compared (no JSON here carries them; "
                  f"they predate the {'columns' if group == 'infer/model' else group + ' part'})")
        print(f"summary ({group}): " + ", ".join(f"{k}={v}" for k, v in sorted(c2.items())))
    if require_columns:
        if require_columns > len(cols):
            bad += 1
        for key, col, verdict, n, why in short:
            print(f"REQUIRE FAIL {key} {col}: {verdict} rests on {n} real hash(es), "
                  f"--require-columns {require_columns} demands that many; a column that "
                  "should cover this lane is refusing, which is not a pass"
                  + (f" (not OWED: {why})" if why else ""))
        missing = sorted(required - set(k.split("/")[0] for k in keys))
        for lane_name in missing:
            print(f"REQUIRE FAIL {lane_name}: no JSON carries a cell for this lane")
        bad += len(short) + len(missing)
        print(f"require-columns {require_columns} over {sorted(required)}: "
              f"{'OK' if not short and not missing else str(len(short) + len(missing)) + ' short'}"
              + (f" ({len(owed)} OWED)" if owed_json else ""))
    if owed_json:
        for o in owed:
            print(f"OWED {o['lane']}/{o['fixture']} {o['part']}: no hash in {','.join(o['missing'])}; "
                  f"rests on {o['real_hashes']} ({','.join(o['present'])})")
        with open(owed_json, "w") as fh:
            json.dump(dict(columns=names, require_columns=require_columns,
                           lanes=sorted(required), owed=owed), fh, indent=1)
        print(f"summary (owed): OWED={len(owed)}; the next release record owes exactly these cell parts; "
              f"wrote {owed_json}")
    return 1 if bad else 0


#: the keys every part of one column must agree on before --merge joins them
MERGE_SAME = ("vendor", "commit", "mode", "repeats", "heldout_seed", "fixtures", "heldout",
              "batch_protocol", "batch_sabotage", "rlpair_protocol", "rlpair_sabotage", "lane_revisions") + tuple(
    f"{part}_{k}" for part in EXTRA_PARTS for k in ("protocol", "sabotage"))


def _build_digests(j):
    """The binding bytes one column part ran: every loaded GPU binding
    (`package.bindings`) and, on a CPU column, every host binding file
    (`host.families[*].sha256`, recorded since 2026-09-15). Sorted
    (module, sha256) pairs; empty when the part recorded neither."""
    dig = [(b["module"], b["sha256"]) for b in (j.get("package") or {}).get("bindings", [])]
    dig += [("host:" + name, fam["sha256"]) for name, fam in ((j.get("host") or {}).get("families") or {}).items()
            if fam.get("sha256")]
    return sorted(dig)


def _host_without_digests(j):
    """A CPU part's `host` object with the per-binding bytes removed, so two
    parts of one machine compare equal on everything but the build."""
    host = j.get("host")
    if not host:
        return host
    host = dict(host)
    host["families"] = {name: {k: v for k, v in fam.items() if k not in ("sha256", "size")}
                        for name, fam in (host.get("families") or {}).items()}
    return host


def merge(paths, out, allow_separate_builds=False):
    """ONE column from the parts a leg split across processes (two
    one-device processes on a two-GPU box, 2026-09-14 night). Refuses parts
    that are not the same column: any MERGE_SAME key differing, a
    different `package.par_devices`, or different binding digests (the
    parts must have loaded the SAME build, byte for byte), a sabotage part,
    and a cell two parts both carry with different contents. The merged
    JSON orders cells by LANES and FIXTURES, is `complete` only if every
    part is, and lists its parts under `merged_from`."""
    parts = []
    for p in paths:
        with open(p) as fh:
            parts.append((p, json.load(fh)))
    first = parts[0][1]
    for p, j in parts:
        for k in MERGE_SAME:
            if json.dumps(j.get(k), sort_keys=True) != json.dumps(first.get(k), sort_keys=True):
                raise SystemExit(f"REFUSING --merge: {p} differs from {parts[0][0]} on {k!r}")
        if j.get("batch_sabotage") or any(j.get(f"{part}_sabotage") for part in EXTRA_PARTS):
            raise SystemExit(f"REFUSING --merge: {p} is a batch sabotage run")
        if j.get("rlpair_sabotage"):
            raise SystemExit(f"REFUSING --merge: {p} is an rlpair sabotage run")
        pk, fk = j.get("package") or {}, first.get("package") or {}
        if pk.get("par_devices") != fk.get("par_devices"):
            raise SystemExit(f"REFUSING --merge: {p} ran par_devices={pk.get('par_devices')!r}, "
                             f"{parts[0][0]} ran {fk.get('par_devices')!r}")
        if json.dumps(_host_without_digests(j), sort_keys=True) != json.dumps(_host_without_digests(first), sort_keys=True):
            raise SystemExit(f"REFUSING --merge: {p} differs from {parts[0][0]} on 'host' (the machine, "
                             "the host bindings built or what they read back)")
        dig = _build_digests(j)
        same_build = dig and dig == _build_digests(first)
        if not same_build and not (allow_separate_builds and dig):
            raise SystemExit(f"REFUSING --merge: {p} loaded different (or unrecorded) binding bytes than "
                             f"{parts[0][0]}; the parts of one column must run one build")
    cells = {}
    for p, j in parts:
        for k, c in j["cells"].items():
            if k in cells and {a: b for a, b in cells[k].items() if a != "timing"} != {
                    a: b for a, b in c.items() if a != "timing"}:
                raise SystemExit(f"REFUSING --merge: cell {k} is in two parts with different contents")
            cells[k] = c
    lane_rank = {n: i for i, n in enumerate(LANES)}
    fx_rank = {f: i for i, f in enumerate(FIXTURES)}
    order = sorted(cells, key=lambda k: (lane_rank.get(k.split("/")[0], len(lane_rank)), k.split("/")[0],
                                         fx_rank.get(k.split("/", 1)[1], len(fx_rank))))
    record = {k: first[k] for k in first if k != "cells"}
    record["cells"] = {k: cells[k] for k in order}
    record["complete"] = all(j.get("complete", False) for _, j in parts)
    # A part whose leg hit its lease is `complete: false` with its finished
    # lanes on disk; a later part that ran exactly the missing lanes makes
    # the column whole. Complete by coverage means every lane of this
    # harness has a cell on every fixture the parts were handed.
    covered = all(f"{n}/{f}" in cells for n in LANES for f in (first.get("fixtures") or {}))
    record["complete_by_coverage"] = bool(covered and first.get("fixtures"))
    record["complete"] = record["complete"] or record["complete_by_coverage"]
    record["skipped"] = sorted(set(s for _, j in parts for s in j.get("skipped", [])))
    record["merged_from"] = [dict(file=os.path.basename(p), cells=len(j["cells"]), complete=j.get("complete"),
                                  platform=j.get("platform"), bindings=(j.get("package") or {}).get("bindings"))
                             for p, j in parts]
    record["merged_separate_builds"] = len(set(json.dumps(_build_digests(j)) for _, j in parts)) > 1
    with open(out, "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"merged {len(parts)} parts, {len(cells)} cells, complete={record['complete']} -> {out}")
    return 0


def _strip_driver_args(argv):
    """Remove arguments owned by the process supervisor from a child argv."""
    out, i = [], 0
    valued = {"--jobs", "--json", "--lanes"}
    while i < len(argv):
        arg = argv[i]
        if arg in valued:
            i += 2
            continue
        if any(arg.startswith(name + "=") for name in valued):
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


def _job_shards(lanes, count):
    """Contiguous, ordered lane shards; no estimator can cross a process."""
    count = min(count, len(lanes))
    return [lanes[len(lanes) * i // count:len(lanes) * (i + 1) // count]
            for i in range(count)]


def run_jobs(args, argv=None):
    """Run CPU lane shards in isolated processes and merge their records.

    This deliberately does not share fitted estimators, fixtures, Python
    modules, or native runtimes between workers. GPU backends are refused in
    every child: concurrent jobs on one accelerator invalidate the evidence.
    Part JSONs are durable checkpoints and are merged only after every child
    exits, in deterministic shard order.
    """
    if not args.json:
        raise SystemExit("REFUSING: --jobs > 1 requires --json for durable shard checkpoints")
    cap = min(8, os.cpu_count() or 1)
    if args.jobs > cap:
        raise SystemExit(f"REFUSING: --jobs {args.jobs} exceeds this host's conservative cap {cap}")
    requested = [n for n in args.lanes.split(",") if n]
    unknown = [n for n in requested if n not in LANES]
    if unknown:
        raise SystemExit(f"REFUSING: --lanes names no lane: {unknown}; lanes are {sorted(LANES)}")
    lanes = requested or [n for n in LANES if not n.startswith(RECORD_EXCLUDED_PREFIXES)]
    skip = set(filter(None, args.skip.split(",")))
    lanes = [n for n in lanes if n not in skip]
    if not lanes:
        raise SystemExit("REFUSING: --jobs has no lanes to run after --skip")
    shards = _job_shards(lanes, args.jobs)
    final = Path(args.json)
    part_dir = final.with_name(final.name + ".jobs")
    part_dir.mkdir(parents=True, exist_ok=True)
    parts = [part_dir / f"part-{i:02d}.json" for i in range(len(shards))]
    logs = [part_dir / f"part-{i:02d}.log" for i in range(len(shards))]
    if not args.resume:
        for path in parts + logs:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    base = _strip_driver_args(list(sys.argv[1:] if argv is None else argv))
    base = [arg for arg in base if arg != "--resume"]
    # A child verifies the actual loaded backend before its first fit. Adding
    # this even when the caller omitted it makes --jobs intrinsically CPU-only.
    if "--require-cpu" not in base:
        base.append("--require-cpu")
    running = []
    print(f"# JOBS {len(shards)} isolated CPU shards (cap={cap}); checkpoints={part_dir}", flush=True)
    try:
        for i, (shard, part, log) in enumerate(zip(shards, parts, logs)):
            cmd = [sys.executable, os.path.abspath(__file__)] + base + [
                "--jobs", "1", "--lanes", ",".join(shard), "--json", str(part)]
            if args.resume and part.is_file():
                cmd.append("--resume")
            stream = open(log, "a" if args.resume else "w")
            proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT)
            running.append((i, proc, stream))
            print(f"# JOB {i} pid={proc.pid} lanes={','.join(shard)}", flush=True)
        codes = {}
        for i, proc, stream in running:
            codes[i] = proc.wait()
            stream.close()
    except BaseException:
        for _, proc, stream in running:
            if proc.poll() is None:
                proc.terminate()
        for _, proc, stream in running:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            stream.close()
        raise
    for i, log in enumerate(logs):
        print(f"\n=== identity shard {i} exit={codes[i]} ===", flush=True)
        try:
            sys.stdout.write(log.read_text())
        except OSError as exc:
            print(f"REFUSED to read shard log: {exc}")
    missing = [str(p) for p in parts if not p.is_file()]
    if missing:
        print(f"REFUSING: shard records missing: {missing}", file=sys.stderr)
        return 1
    merge(parts, str(final))
    bad = [i for i, code in codes.items() if code]
    if bad:
        print(f"# JOBS FAILED shards={bad}; merged evidence retained at {final}", file=sys.stderr)
        return 1
    print(f"# JOBS COMPLETE merged={final}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default="")
    ap.add_argument("--require-backend", choices=("cpu", "metal", "cuda", "hip"),
                    help="refuse a different loaded backend before fitting")
    ap.add_argument("--require-cpu", action="store_true",
                    help="refuse any GPU backend before fitting (internal CPU oracle jobs)")
    ap.add_argument("--fail-on-refused", action="store_true",
                    help="exit nonzero if any requested stage refuses (used by iteration runner)")
    ap.add_argument("--resume", action="store_true",
                    help="reuse completed cells from --json only when sources, binaries and protocol match")
    ap.add_argument("--jobs", type=int, default=1, metavar="N",
                    help="CPU only: run independent lane shards in N isolated processes (requires --json; "
                         "each shard is resumable and the final record is merged in lane order)")
    ap.add_argument("--lanes", default="")
    ap.add_argument("--skip", default="", help="lanes to leave out, comma separated; each is reported as SKIPPED")
    ap.add_argument("--fixtures", default="")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--vendor", default=None,
                    help="the box label, ^[a-z0-9][a-z0-9_.-]*$ and not a placeholder; default "
                         "cpu-<cpu model slug> on a CPU-only install, platform.machine() elsewhere")
    ap.add_argument("--allow-fast", action="store_true")
    ap.add_argument("--batch-alone", type=int, default=BATCH_ALONE, metavar="N",
                    help="rows evaluated alone per batch call (default %(default)s); the split and the "
                         "prefixes are fixed, and the part's hash does not depend on N")
    ap.add_argument("--no-batch", action="store_true",
                    help="skip the batch part; its cells record n/a:skipped (--no-batch)")
    ap.add_argument("--batch-grad", action="store_true",
                    help="add the batchgrad part: per-row gradients and clause 9.2 aligned accumulation")
    ap.add_argument("--batch-scale", action="store_true",
                    help="add the batchscale part: B in {1, 17, 64, 256} inside B = 1024, and long-L prefixes")
    ap.add_argument("--ragged", action="store_true",
                    help="add the ragged part: lengths= right-padded batches against each row alone")
    ap.add_argument("--step-full", action="store_true",
                    help="add the stepfull part: one fresh-state forward pass over a sequence against the "
                         "same sequence decoded one token at a time with a carried state, compared bitwise "
                         "position by position")
    ap.add_argument("--no-rlpair", action="store_true",
                    help="skip the rlpair part; the lanes that declare it record n/a:skipped (--no-rlpair)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--diff", nargs="+", default=None, metavar="JSON",
                    help="compare JSONs cell by cell: the train column, then infer and model where carried")
    ap.add_argument("--require-columns", type=int, default=0, metavar="N",
                    help="with --diff: exit non-zero unless every compared cell of the lanes named by "
                         "--lanes (every lane when --lanes is empty) rests on at least N real hashes; "
                         "--lanes also scopes which cells --diff compares at all")
    ap.add_argument("--owed-json", default="", metavar="PATH",
                    help="with --diff --require-columns: a cell part short ONLY because a record has no hash "
                         "for it (cell absent, part absent or n/a) reads OWED xK instead of failing, when every "
                         "CPU column hashes it STABLE over two or more repeats, no column refused or moved it, "
                         "and the columns that hash it agree; the owed cell parts are written to PATH. Their "
                         "sabotage check is tools/cpu_identity_gate_check.py owed")
    ap.add_argument("--merge", nargs="+", default=None, metavar="JSON",
                    help="join the parts of ONE column (same vendor, commit, fixtures, build) into --json")
    ap.add_argument("--allow-separate-builds", action="store_true",
                    help="with --merge: admit parts whose binding digests differ (two legs, two builds of the "
                         "same commit on the same box type); every part's digests are kept under merged_from")
    args = ap.parse_args()
    if args.jobs < 1:
        raise SystemExit("REFUSING: --jobs must be positive")
    if args.jobs > 1:
        if args.merge or args.diff:
            raise SystemExit("REFUSING: --jobs cannot be combined with --merge or --diff")
        return run_jobs(args)
    if args.merge:
        if not args.json:
            raise SystemExit("REFUSING: --merge needs --json <out>")
        return merge(args.merge, args.json, args.allow_separate_builds)
    if args.diff:
        lanes = [n for n in args.lanes.split(",") if n] if args.lanes else None
        if lanes:
            unknown = [n for n in lanes if n not in LANES]
            if unknown:
                raise SystemExit(f"REFUSING: --lanes names no lane: {unknown}; lanes are {sorted(LANES)}")
        return diff(args.diff, args.require_columns, lanes, args.owed_json or None)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
