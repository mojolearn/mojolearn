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
            (DBSCAN, agglomerative, spectral: `fit` and `fit_predict`, and
            SpectralClustering.predict raises NotImplementedError by
            design); no-predict means KMeans has `fit` and `fit_predict`
            only, no predict or transform (`cluster.py`, verified
            2026-09-14); function means the lane is not an estimator
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
            length only). No sequence API takes padding or a ragged batch,
            so a sequence is compared alone against same-length peers. The
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
            The n/a reasons are transductive and no-predict as for infer, function
            (metrics, resampling, cross-validation), no-batch-axis
            (tokenizer), optimizer-step and training-step (the batch IS the
            arithmetic of a step), no-model (a trainer lane that returns
            no estimator) and batch-dependent-by-contract (UMAP.transform,
            whose module says query batching may change results). A method
            that refuses a batch of one BY NAME (the coordinate descent
            predict, mirroring cuML's cdPredict) is held to that refusal and
            its rows are asked in the smallest admitted batch instead
            (`_BatchRows` min_batch). `summary (infer/model):` and
            `summary (batch):` are separate lines of `--diff`. Cost is bounded: at most 1 + 16 + 3 extra calls per
            row call and 4 per length call, per cell per repeat; the
            declarations sit outside the lane bodies (`BATCH`), so no train,
            infer or model hash moves. IT TESTS the host call path's batch
            splitting through the bindings, on ONE box. IT DOES NOT TEST
            serving-scale B (checks/batch_invariance_check.mojo, B in
            {1, 17, 64}), padding or ragged batches, the backward pass, or
            any cross-vendor statement, which comes from `--diff` over
            records. THE SABOTAGE turns every hashed batch cell BATCH_MOVED,
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

THE LANES, 178 (2026-09-14; 46 on 2026-09-13, pca-whiten the same night, 71
on 2026-09-14 from the claim-surface census, logistic-multiclass and
tokenizer the same day when those two got their doors, 16 `par-*` lanes that
evening for the ordered multi-GPU drivers run on ONE device, and 15 lanes for
the doors workstream D opened: Cholesky, the kernel methods, the Gaussian
mixture, HDBSCAN, resampling, the training primitives and the KMeans arms,
then 15 more `par-*` lanes that night for the multi-GPU drivers the first 16
missed, then `par-forest-pool`, `par-gmm`, `par-resample` and `par-hdbscan` for the drivers the multigpu lane added, and `ivf` and `embedding` when IVFIndex and Embedding left `_NOT_YET`, then `ivf-euclidean` when its metric stopped being refused, then `embedding-sort` for PLAN_SORT, and `par-cholesky`, `par-kernel-ridge`, `par-nystroem` and `par-rbf-sampler` on 2026-09-15). One per public estimator
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
               gbdt-adapter-reg iforest-tuned (last, with iforest)
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
    2026-09-14 evening, workstream D (docs/lanes/LANE_BODY_*.py)
      cholesky kernel-ridge nystroem rbf-sampler gmm gmm-random-init hdbscan
               hdbscan-leaf bootstrap permutation-test monte-carlo
               training-primitives kmeans-sqrt kmeans-classic-pp
               kmeans-cosine (the refusal sentence is its cell)
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
    2026-09-14 night (lane/expose-ivf-embedding, the last two _NOT_YET doors)
      ivf embedding
    2026-09-14 night (fix/ivf-l2sqrt, metric='euclidean' no longer refused)
      ivf-euclidean
    2026-09-15 (lane/embedding-owed, PLAN_SORT through Embedding(plan="sort"))
      embedding-sort

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
held-out bytes, batch protocol, par_devices or loaded binding digests, and a
cell two parts carry differently (2026-09-14 night). Parts from two legs
(two builds of the same commit) need `--allow-separate-builds`, and the
merged JSON then says `merged_separate_builds: true` and keeps each part's
binding digests.

THE FOURTH COLUMN, a CPU (the CPU training lane, 2026-09-13; brief
docs/lanes/BRIEF_cpu_training_2026-09-13.md section 3.3). On an install whose
`mojolearn.vendor()` is 'cpu' (no GPU set, a host binding under
mojolearn/host/ built), `--vendor` defaults to `cpu-<cpu model slug>` read
from the machine, the JSON gains a `host` object (cpu model, arch, the host
bindings built and what each reads back as its kernel-matrix column), and a
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
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

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
        try:
            m = _backend.load_host_module(basename)
            fam = dict(column=str(getattr(m, prefix + "_column")()))
            detected = getattr(m, prefix + "_detected_column", None)
            if detected is not None:
                fam["detected_column"] = str(detected())
            sab = getattr(m, prefix + "_sabotage", None)
            if sab is not None:
                fam["sabotage"] = bool(sab())
        except Exception as exc:
            fam = dict(error=f"{type(exc).__name__}: {exc}"[:300])
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


def _fit(parts, est=None, probe="n/a:function"):
    f = Fit(parts)
    f.est = est
    f.probe = probe
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
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("rf-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("et-clf")
def _(ml, X, yc, yr, Xh=None):
    m = ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("et-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.ExtraTreesRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-symmetric")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss").fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-depthwise")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, grow_policy="Depthwise",
                            loss="Logloss").fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-lossguide")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GradientBoosting(n_estimators=20, max_leaves=32, grow_policy="Lossguide",
                            loss="Logloss").fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-rmse")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="RMSE").fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("kmeans")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KMeans(n_clusters=8, random_state=3).fit(X)
    # mojolearn's KMeans has fit and fit_predict only, no predict or transform
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_)), m, "n/a:no-predict")


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
    m = ml.DBSCAN(eps=0.9, min_samples=5).fit(X[:6000, :4])
    return _fit(dict(labels=_h(m.labels_)), m, "n/a:transductive")


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
    m = ml.AgglomerativeClustering(n_clusters=4).fit(X[:2000, :4])
    return _fit(dict(labels=_h(m.labels_)), m, "n/a:transductive")


@lane("spectral")
def _(ml, X, yc, yr, Xh=None):
    m = ml.SpectralClustering(n_clusters=4, random_state=3).fit(X[:2000, :4])
    # SpectralClustering.predict raises NotImplementedError by design
    return _fit(dict(labels=_h(m.labels_)), m, "n/a:transductive")


@lane("holtwinters")
def _(ml, X, yc, yr, Xh=None):
    series = (np.cumsum(X[:512, 0]) + 50.0).astype(np.float32)
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


def _same_bytes(name_a, a, name_b, b):
    """The forecasters' infer probe: two public entries the estimator
    documents as the same answer. Returns both for hashing when their bytes
    agree; raises, naming the pair and the byte count, when they do not, so
    the infer column reads REFUSED with the message instead of a hash that
    hides which of the two moved."""
    ba, bb = np.asarray(a).ravel().tobytes(), np.asarray(b).ravel().tobytes()
    if ba != bb:
        n = sum(x != y for x, y in zip(ba, bb)) + abs(len(ba) - len(bb))
        raise ValueError(f"{name_a} and {name_b} differ: {n} bytes of {max(len(ba), len(bb))}")
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


def _block_weights(lane, shapes, ones=()):
    """A weight dict for a sequence block. `ones` names get a vector of
    ones (the norms), the rest are hashed uniform on [-1/8, 1/8)."""
    return {name: (np.ones(shp, dtype=np.float32) if name in ones
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
                m, lambda e: (np.asarray(e.predict_logits(np.ascontiguousarray(Xh[:256, :8]))),))


@lane("byte-lm")
def _(ml, X, yc, yr, Xh=None):
    """SmallByteLanguageModelTrainer (LanguageModelTrainer is the same
    class) at the default b2-l32-d32 profile, 34944 parameters. Three
    AdamW steps on three (2, 33) windows of the fixture bytes (_ids),
    weights from _byte_lm_params. The identity claim is per shape by the
    trainer's own contract; this is one shape."""
    shape = ml.ByteLanguageModelConfig()
    named, _ = _byte_lm_params(shape)
    m = ml.SmallByteLanguageModelTrainer(named, data_schedule={"dataset": "identity_break", "order": "sequential"},
                                         shape=shape)
    ids = _ids(X, 3 * shape.batch, shape.length + 1)
    losses = [np.float64(m.train_step(ids[2 * k:2 * k + 2])["loss"]) for k in range(3)]
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
        ones=("block_norm.weight", "B_norm.weight", "C_norm.weight"))
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
        ones=("input_layernorm.weight", "post_attention_layernorm.weight"))
    blk = ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), dict(max_tokens=32))
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


@lane("samba")
def _(ml, X, yc, yr, Xh=None):
    """SambaStack, one Mamba-3 layer and one attention layer at d_model 32
    over a 256-byte vocabulary, weights from the stack's own seeded
    generator, three AdamW steps on three (2, 17) windows of the fixture
    bytes. SUPPORT_MATRIX says cross-vendor qualification of the training
    surface is open. The checkpoint is the model column."""
    cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)
    m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3)
    ids = _ids(X, 6, 17)
    losses = [np.float64(m.train_step(ids[2 * k:2 * k + 2, :-1], ids[2 * k:2 * k + 2, 1:])["loss"])
              for k in range(3)]
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
    """The fixture with NaN in columns 5, 6 and 7 of every eighth row, so a
    quantizer's nan_mode has something to place. No fixture carries a NaN
    (the other lanes would refuse it), so the GBDT nan lanes make their own."""
    Xn = X.copy()
    Xn[::8, 5:8] = np.float32(np.nan)
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
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="MultiClass",
                            class_weights=[1.0, 2.0, 0.5]).fit(X, t)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-onevsall")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="MultiClassOneVsAll").fit(X, _three_class_centered(X))
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-parametric-losses")
def _(ml, X, yc, yr, Xh=None):
    """The ten losses no lane above fits, eight trees of depth four each:
    four take a mandatory parameter; CrossEntropy takes a probability
    target; the rest a positive target."""
    y = _pos(yr)
    fits = {
        "Quantile": dict(), "MAE": dict(), "LogLinQuantile": dict(), "MAPE": dict(), "Poisson": dict(),
        "Lq": dict(loss_q=3.0), "Expectile": dict(loss_alpha=0.3),
        "Tweedie": dict(loss_variance_power=1.5), "Huber": dict(loss_delta=1.0),
    }
    parts, first = {}, None
    for loss, kw in fits.items():
        m = ml.GradientBoosting(n_estimators=8, max_depth=4, loss=loss, **kw).fit(X, y)
        parts[loss] = _h(m.predict(X))
        first = first or m
    ce = ml.GradientBoosting(n_estimators=8, max_depth=4, loss="CrossEntropy").fit(X, yc.astype(np.float32))
    parts["CrossEntropy"] = _h(ce.predict(X))
    return _fit(parts, first, lambda e: (e.predict(Xh),))


@lane("gbdt-lossguide-newtoncosine")
def _(ml, X, yc, yr, Xh=None):
    """NewtonCosine, the child-hessian and split-gain and leaf-count
    thresholds, column subsampling, random strength, Bernoulli bootstrap,
    gradient leaves with three iterations: one Lossguide fit."""
    m = ml.GradientBoosting(n_estimators=20, max_leaves=32, grow_policy="Lossguide", loss="Logloss",
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
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss", score_function="L2",
                            use_pointwise_searcher=True, bootstrap_type="Bayesian", bagging_temperature=0.5,
                            boost_from_average=True, od_type="Iter", od_wait=5, use_best_model=True
                            ).fit(X, yc, sample_weight=w, eval_set=(Xh, ych))
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("gbdt-exact-mae")
def _(ml, X, yc, yr, Xh=None):
    """The Exact (weighted quantile) leaf estimator and the Poisson bootstrap."""
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="MAE", leaf_estimation_method="Exact",
                            bootstrap_type="Poisson", subsample=0.6).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


@lane("gbdt-categorical-ctr")
def _(ml, X, yc, yr, Xh=None):
    """A CTR feature and a one-hot feature (disjoint: cat_features makes
    its own one-hot decision) with two CTR permutations, on the coded
    columns of _coded."""
    m = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss", cat_features=[0],
                            one_hot_features=[1], permutation_count=2,
                            ctr_estimation_permutation_id=0).fit(_coded(X), yc)
    return _fit(dict(predict=_h(m.predict(_coded(X)))), m, lambda e: (e.predict(_coded(Xh)),))


@lane("gbdt-nan-modes")
def _(ml, X, yc, yr, Xh=None):
    """nan_mode Min and Max on a fixture that actually carries NaN
    (_with_nan); on a NaN-free column the quantizer collapses both to
    Forbidden, which is why no lane above could reach them."""
    Xn = _with_nan(X)
    lo = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss", nan_mode="Min").fit(Xn, yc)
    hi = ml.GradientBoosting(n_estimators=20, max_depth=6, loss="Logloss", nan_mode="Max").fit(Xn, yc)
    return _fit(dict(min=_h(lo.predict(Xn)), max=_h(hi.predict(Xn))),
                lo, lambda e: (e.predict(_with_nan(Xh)),))


@lane("gbdt-adapter-clf")
def _(ml, X, yc, yr, Xh=None):
    """The sklearn-style classifier adapter, the only caller of the binary
    probability and class entries. It refuses save, so no model column."""
    m = ml.GradientBoostingClassifier(n_estimators=20, max_depth=6).fit(X, yc)
    return _fit(dict(predict=_h(m.predict(X)), proba=_h(m.predict_proba(X)),
                     decision=_h(m.decision_function(X[:512]))),
                m, lambda e: (e.predict(Xh), e.predict_proba(Xh), e.decision_function(Xh[:512])))


@lane("gbdt-adapter-reg")
def _(ml, X, yc, yr, Xh=None):
    m = ml.GradientBoostingRegressor(n_estimators=20, max_depth=6).fit(X, yr)
    return _fit(dict(predict=_h(m.predict(X))), m, lambda e: (e.predict(Xh),))


# -- neural

@lane("mamba2-dtlimit")
def _(ml, X, yc, yr, Xh=None):
    """The active dt clamp (seam S9); at the default (0, inf) it cannot
    move a bit, which is what the mamba2 lane measures."""
    dm, di, nh = 32, 64, 1
    cd, dip = di + 256, 2 * di + 256 + nh
    w = _block_weights("mamba2", {
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
        "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
        "out_proj.weight": (dm, di)},
        ones=("block_norm.weight", "norm.weight"))
    blk = ml.Mamba2Block(w, dt_limit=(0.01, 0.1))
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), {})
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


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
        ones=("input_layernorm.weight", "post_attention_layernorm.weight"))
    blk = ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv, window=8)
    parts = _block_fit(blk, _seq(X, 2, 16, dm), _seq(X, 2, 16, dm, skip=1024), dict(max_tokens=32))
    return _fit(parts, blk, lambda e: (np.asarray(e.forward(_seq(Xh, 2, 16, dm))),))


@lane("byte-lm-resident")
def _(ml, X, yc, yr, Xh=None):
    """The device-owned session (resident=True, lean step results), the
    path the byte-lm lane's docstring says is bit-equal to the stateless
    one and measured on one H100 only."""
    shape = ml.ByteLanguageModelConfig()
    named, _ = _byte_lm_params(shape)
    m = ml.SmallByteLanguageModelTrainer(named, data_schedule={"dataset": "identity_break", "order": "sequential"},
                                         shape=shape, resident=True, step_result="lean")
    ids = _ids(X, 3 * shape.batch, shape.length + 1)
    losses = [np.float64(m.train_step(ids[2 * k:2 * k + 2])["loss"]) for k in range(3)]
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
    for k in range(3):
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
    ids = _ids(X, 96, 17)
    losses = [np.float64(m.train_step(ids[32 * k:32 * k + 32, :-1], ids[32 * k:32 * k + 32, 1:])["loss"])
              for k in range(3)]
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
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_)), m, "n/a:no-predict")


@lane("kmeans-array")
def _(ml, X, yc, yr, Xh=None):
    m = ml.KMeans(n_clusters=8, init="array", init_centroids=np.ascontiguousarray(X[:8]), random_state=3).fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_)), m, "n/a:no-predict")


@lane("kmeans-weighted")
def _(ml, X, yc, yr, Xh=None):
    w = _hw((X.shape[0],), "kmeans:sample_weight", 0.5, 1.5)
    m = ml.KMeans(n_clusters=8, random_state=3).fit(X, sample_weight=w)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_)), m, "n/a:no-predict")


@lane("dbscan-brute-l1")
def _(ml, X, yc, yr, Xh=None):
    m = ml.DBSCAN(eps=0.9, min_samples=5, metric="manhattan", algorithm="brute").fit(X[:6000, :4])
    return _fit(dict(labels=_h(m.labels_)), m, "n/a:transductive")


@lane("dbscan-weighted")
def _(ml, X, yc, yr, Xh=None):
    w = _hw((6000,), "dbscan:sample_weight", 0.5, 1.5)
    m = ml.DBSCAN(eps=0.9, min_samples=5).fit(X[:6000, :4], sample_weight=w)
    return _fit(dict(labels=_h(m.labels_)), m, "n/a:transductive")


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
    m = ml.SpectralClustering(n_clusters=4, affinity="precomputed", random_state=3).fit(A)
    return _fit(dict(affinity=_h(A), labels=_h(m.labels_)), m, "n/a:transductive")


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


@lane("tokenizer")
def _(ml, X, yc, yr, Xh=None):
    """GPT2Tokenizer (python/mojolearn/tokenizer.py), host integers and
    tables through _mojolearn_tokenizer_host; no float arithmetic, so
    cross-vendor identity is by construction and what this measures is
    that the SAME binary bytes were built on every box. The fixture bytes
    are the first 4,096 bytes of X viewed as bytes (the byte-lm lane's
    derivation without _ids' modulus), encoded with <|endoftext|> allowed,
    then decoded back; the held-out probe encodes Xh's first 4,096 bytes
    (docs/lanes/BRIEF_expose_tokenizer_2026-09-14.md section 3)."""
    tok = ml.GPT2Tokenizer()
    raw = np.ascontiguousarray(X).tobytes()[:4096]
    ids = np.asarray(tok.encode_bytes(raw, allow_endoftext=True), dtype=np.int32)
    back = np.frombuffer(tok.decode_bytes(ids.tolist()), dtype=np.uint8)
    assert back.tobytes() == raw, "tokenizer lane: decode(encode(x)) != x"
    return _fit(dict(ids=_h(ids), decoded=_h(back), n_vocab=_h(np.int64(tok.n_vocab))),
                tok, lambda e: (np.asarray(e.encode_bytes(np.ascontiguousarray(Xh).tobytes()[:4096],
                                                          allow_endoftext=True), dtype=np.int32),))


@lane("cross-val")
def _(ml, X, yc, yr, Xh=None):
    """cross_val_score, which had no lane: three unshuffled folds of the
    sklearn-style boosting regressor (cross-validation clones through
    get_params, which the classical estimators do not implement), scored
    by the adapter's own score."""
    scores = ml.model_selection.cross_val_score(ml.GradientBoostingRegressor(n_estimators=8, max_depth=4), X, yr, cv=3)
    return _fit(dict(scores=_h(np.asarray(scores, dtype=np.float64))))



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


@lane("hdbscan")
def _(ml, X, yc, yr, Xh=None):
    """HDBSCAN (python/mojolearn/hdbscan.py), cuML's runner path at its
    defaults with excess-of-mass selection: the core distances, the
    Boruvka MST, single linkage, the condensed tree and the labels.
    Train hashes the labels, the core distances and the four integers the
    fit reports (cluster, outlier, Boruvka round and condensed cluster
    counts); the integer stages are where a divergence first shows."""
    m = ml.HDBSCAN(min_cluster_size=5).fit(X[:6000, :4])
    return _fit(dict(labels=_h(m.labels_), core=_h(m.core_distances_),
                     counts=_h(np.asarray([m.n_clusters_, m.n_outliers_, m.n_boruvka_rounds_, m.n_condensed_clusters_], dtype=np.int64))),
                m, "n/a:transductive")


@lane("hdbscan-leaf")
def _(ml, X, yc, yr, Xh=None):
    """The same fit under cluster_selection_method='leaf' with
    min_samples below min_cluster_size and allow_single_cluster, the
    other selection arm and the two knobs that change which condensed
    clusters become labels."""
    m = ml.HDBSCAN(min_cluster_size=8, min_samples=3, cluster_selection_method="leaf", allow_single_cluster=True).fit(X[:6000, :4])
    return _fit(dict(labels=_h(m.labels_), core=_h(m.core_distances_),
                     counts=_h(np.asarray([m.n_clusters_, m.n_outliers_, m.n_boruvka_rounds_, m.n_condensed_clusters_], dtype=np.int64))),
                m, "n/a:transductive")


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
                e, lambda m: (np.asarray(m.forward(idh)),))


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
                e, lambda m: (np.asarray(m.forward(idh)),))


@lane("kmeans-sqrt")
def _(ml, X, yc, yr, Xh=None):
    """metric='l2_sqrt_expanded': cuVS's L2SqrtExpanded, the root taken on
    the assignment's reduced distance (`metric_is_sqrt`, `identical_sqrt`,
    DEVIATION 2715) and so in the inertia, the row norms squared as under
    L2Expanded (DEVIATION 2716); a different inertia_ from the squared arm."""
    m = ml.KMeans(n_clusters=8, random_state=3, metric="l2_sqrt_expanded").fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_),
                     inertia=_h(np.float64(m.inertia_)), scales=_h(np.asarray([m.sum_scale_, m.weight_scale_], dtype=np.float64))),
                m, "n/a:no-predict")


@lane("kmeans-classic-pp")
def _(ml, X, yc, yr, Xh=None):
    """oversampling_factor=0.0: the classic sequential k-means++ seeding
    (detail/kmeans.cuh:910-915's `== 0` arm), a different algorithm from
    the scalable k-means|| the default selects."""
    m = ml.KMeans(n_clusters=8, random_state=3, oversampling_factor=0.0).fit(X)
    return _fit(dict(centers=_h(m.cluster_centers_), labels=_h(m.labels_), inertia=_h(np.float64(m.inertia_))),
                m, "n/a:no-predict")


@lane("kmeans-cosine")
def _(ml, X, yc, yr, Xh=None):
    """metric='cosine' is routed and REFUSED BY NAME on the Mojo host
    (cluster/impl/kmeans_params.mojo::validate): the expected cell is the
    refusal sentence on every column, never a hash. A column that hashes
    here means the refusal was lifted without a fused cosine arm."""
    try:
        m = ml.KMeans(n_clusters=8, random_state=3, metric="cosine").fit(X)
    except Exception as exc:
        # the expected outcome: the cell is the hash of the refusal sentence
        # (harness owner, 2026-09-14), so a record carries it STABLE and a
        # lifted refusal reads DIVERGENT against it instead of vanishing
        # into a permanent REFUSED count
        text = f"{type(exc).__name__}: {exc}"
        return _fit(dict(refusal=_h(np.frombuffer(text.encode(), dtype=np.uint8))))
    return _fit(dict(centers=_h(m.cluster_centers_)), m, "n/a:no-predict")


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
    par = fit_boosting(ml.GradientBoosting(**kw), X, yc, devices=_par_devices())
    plain = ml.GradientBoosting(**kw).fit(X, yc)
    _same_bytes("fit_boosting predict", par.predict(X[:2048]), "plain predict", plain.predict(X[:2048]))
    return _fit(dict(predict=_h(par.predict(X)), proba=_h(par.predict_proba(X))),
                par, lambda e: (e.predict(Xh), e.predict_proba(Xh)))


@lane("par-kmeans")
def _(ml, X, yc, yr, Xh=None):
    from mojolearn.parallel_classical import fit_kmeans
    par = fit_kmeans(ml.KMeans(n_clusters=8, random_state=3), X, devices=_par_devices())
    plain = ml.KMeans(n_clusters=8, random_state=3).fit(X)
    _same_bytes("fit_kmeans centers", par.cluster_centers_, "plain centers", plain.cluster_centers_)
    return _fit(dict(centers=_h(par.cluster_centers_), labels=_h(par.labels_)), par, "n/a:no-predict")


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
    par = fit_dbscan(ml.DBSCAN(eps=0.9, min_samples=5), X[:6000, :4], devices=_par_devices())
    plain = ml.DBSCAN(eps=0.9, min_samples=5).fit(X[:6000, :4])
    _same_bytes("fit_dbscan labels", par.labels_, "plain labels", plain.labels_)
    return _fit(dict(labels=_h(par.labels_)), par, "n/a:transductive")


@lane("par-scaler")
def _(ml, X, yc, yr, Xh=None):
    """fit_scaler and transform_scaler with four columns per shard, so the
    16-column fixture is four shards, held to the plain scaler."""
    from mojolearn.parallel_preprocessing import fit_scaler, transform_scaler
    par = fit_scaler(ml.StandardScaler(), X, devices=_par_devices(), columns_per_shard=4)
    plain = ml.StandardScaler().fit(X)
    t = transform_scaler(par, X[:256], devices=_par_devices(), columns_per_shard=4)
    _same_bytes("transform_scaler", t, "plain transform", plain.transform(X[:256]))
    return _fit(dict(mean=_h(par.mean_), var=_h(par.var_), transform=_h(t),
                     inverse=_h(transform_scaler(par, t, devices=_par_devices(), columns_per_shard=4, inverse=True))),
                par, lambda e: (e.transform(Xh[:256]),))


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


@lane("par-mlp")
def _(ml, X, yc, yr, Xh=None):
    """ParallelNeuralTrainer over the mlp lane's trainer: three logical
    shards of 64 rows, one ordered update per step, three steps; its
    contract is equality to its own one-device replay, not to a plain
    step over 192 rows, so the parts are hashed and not held to mlp."""
    from mojolearn.parallel_training import ParallelNeuralTrainer
    w = [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
         for s in ((16, 8), (16,), (3, 16), (3,))]
    m = ml.SmallMLPTrainer(*w, data_schedule={"dataset": "identity_break", "order": "sequential"})
    Xm = np.ascontiguousarray(X[:576, :8])
    t = _three_class(X[:576])
    with ParallelNeuralTrainer(m, devices=_par_devices(), logical_shards=3) as tr:
        for step in range(3):
            base = 192 * step
            tr.train_step([(Xm[base + 64 * k: base + 64 * (k + 1)], t[base + 64 * k: base + 64 * (k + 1)]) for k in range(3)])
        ck = tr.checkpoint()
    return _fit(dict(weights=_h(*[np.asarray(m.weights_[k]) for k in sorted(m.weights_)]),
                     logits=_h(np.asarray(m.predict_logits(Xm[:256]))),
                     shards=_h(np.int64(ck["logical_shards"]))),
                m, lambda e: (np.asarray(e.predict_logits(np.ascontiguousarray(Xh[:256, :8]))),))


@lane("par-samba")
def _(ml, X, yc, yr, Xh=None):
    """ParallelNeuralTrainer over the samba lane's stack, two logical shards
    of (2, 17) windows per step, two steps."""
    from mojolearn.parallel_training import ParallelNeuralTrainer
    cfg = _par_samba_config(ml)
    m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3)
    ids = _ids(X, 8, 17)
    with ParallelNeuralTrainer(m, devices=_par_devices(), logical_shards=2) as tr:
        for step in range(2):
            tr.train_step([(ids[4 * step + 2 * k: 4 * step + 2 * k + 2, :-1], ids[4 * step + 2 * k: 4 * step + 2 * k + 2, 1:])
                           for k in range(2)])
    params = m.parameters()
    return _fit(dict(logits=_h(np.asarray(m.forward(ids[:2, :-1]))),
                     params=_h(*[np.asarray(params[k]) for k in sorted(params)])),
                m, lambda e: (np.asarray(e.forward(_ids(Xh, 2, 16))),))


@lane("par-byte-lm")
def _(ml, X, yc, yr, Xh=None):
    """ParallelByteLanguageModelTrainer from the byte-lm lane's starting
    state, two logical shards on one device, three steps; the state after
    and the per-shard losses are the parts."""
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer
    shape = _par_byte_lm_config(ml)
    named, _ = _byte_lm_params(shape)
    seed = ml.SmallByteLanguageModelTrainer(named, data_schedule={"dataset": "identity_break", "order": "sequential"},
                                            shape=shape)
    ids = _ids(X, 6 * shape.batch, shape.length + 1)
    losses = []
    with ParallelByteLanguageModelTrainer(seed.state_dict(), devices=_par_devices(), logical_shards=2) as tr:
        for step in range(3):
            res = tr.train_step([ids[4 * step: 4 * step + 2], ids[4 * step + 2: 4 * step + 4]])
            losses.append([np.float64(v) for v in res["losses"]])      # a dict: losses, completed_steps
        state = tr.state_dict()
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
    _same_bytes("ReferenceShardedNeighbors distances", d, "plain distances", d0)
    _same_bytes("ReferenceShardedNeighbors indices", i, "plain indices", i0)
    _same_bytes("ReferenceShardedNeighbors predict", pr, "plain predict", m.predict(q))
    _same_bytes("ReferenceShardedNeighbors predict_proba", pp, "plain predict_proba", m.predict_proba(q))

    def probe(e):
        with _rsn(e) as rs:
            return rs.predict(Xh[:64]), rs.predict_proba(Xh[:64])
    return _fit(dict(dist=_h(d), idx=_h(i), predict=_h(pr), proba=_h(pp)), m, probe)


@lane("par-reference-knn-reg")
def _(ml, X, yc, yr, Xh=None):
    """ReferenceShardedNeighbors over the knn-reg lane's regressor, the
    merged neighbors' regression vote, held to the plain predict."""
    m = ml.KNeighborsRegressor(n_neighbors=8).fit(X[:4096], yr[:4096])
    q = np.ascontiguousarray(X[4096:4160])
    with _rsn(m) as rs:
        pr = rs.predict(q)
    _same_bytes("ReferenceShardedNeighbors predict", pr, "plain predict", m.predict(q))

    def probe(e):
        with _rsn(e) as rs:
            return (rs.predict(Xh[:64]),)
    return _fit(dict(predict=_h(pr)), m, probe)


@lane("par-graph-agglomerative")
def _(ml, X, yc, yr, Xh=None):
    """fit_graph over the agglomerative lane's fit, held to the plain labels
    and merge tree."""
    from mojolearn.parallel_graph import fit_graph
    A = np.ascontiguousarray(X[:2000, :4])
    par = fit_graph(ml.AgglomerativeClustering(n_clusters=4), A, devices=_par_devices())
    plain = ml.AgglomerativeClustering(n_clusters=4).fit(A)
    _same_bytes("fit_graph labels_", par.labels_, "plain labels_", plain.labels_)
    _same_bytes("fit_graph children_", par.children_, "plain children_", plain.children_)
    return _fit(dict(labels=_h(par.labels_), children=_h(par.children_)), par, "n/a:transductive")


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
    par = fit_boosting(ml.GradientBoosting(**kw), X, yc, devices=_par_devices())
    plain = ml.GradientBoosting(**kw).fit(X, yc)
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
    two logical shards of (2, 17) windows, two steps."""
    from mojolearn.parallel_training import ParallelNeuralTrainer
    cfg = _par_samba_config(ml)
    m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3, max_norm=0.5)
    ids = _ids(X, 8, 17)
    with ParallelNeuralTrainer(m, devices=_par_devices(), logical_shards=2) as tr:
        losses = []
        for step in range(2):
            res = tr.train_step([(ids[4 * step + 2 * k: 4 * step + 2 * k + 2, :-1], ids[4 * step + 2 * k: 4 * step + 2 * k + 2, 1:])
                                 for k in range(2)])
            losses.append([np.float64(v) for v in res["losses"]])
        grads = tr.export_gradients()
    params = m.parameters()
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
    labels and core distances."""
    from mojolearn.parallel_classical import fit_hdbscan
    par = fit_hdbscan(ml.HDBSCAN(min_cluster_size=5), X[:6000, :4], devices=_par_devices())
    plain = ml.HDBSCAN(min_cluster_size=5).fit(X[:6000, :4])
    _same_bytes("fit_hdbscan labels_", par.labels_, "plain labels_", plain.labels_)
    _same_bytes("fit_hdbscan core_distances_", par.core_distances_, "plain core_distances_", plain.core_distances_)
    return _fit(dict(labels=_h(par.labels_), core=_h(par.core_distances_),
                     counts=_h(np.asarray([par.n_clusters_, par.n_outliers_, par.n_boruvka_rounds_,
                                           par.n_condensed_clusters_], dtype=np.int64))),
                par, "n/a:transductive")


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
    BY NAME. The coordinate descent predict mirrors cuML's `cdPredict`
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
    for p in _prefix_lengths(call.full):
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
        calls = spec(ml, fit.est, Xh)
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
        return None, f"batch: {type(exc).__name__}: {exc}"
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
            "rf-clf", "et-clf", "gbdt-symmetric", "rf-clf-entropy-log2-noboot", "rf-clf-balanced-parallel",
            "et-clf-entropy-bestfirst", "gbdt-multiclass", "gbdt-onevsall", "gbdt-pointwise-l2-bayesian-eval",
            "par-forest", "par-boosting")
_batch_decl(_rows_calls("predict"),
            "rf-reg", "et-reg", "gbdt-depthwise", "gbdt-lossguide", "gbdt-rmse", "gbdt-ordered-rmse",
            "rf-reg-poisson", "rf-reg-gamma-ig", "et-reg-bootstrap-parallel", "gbdt-parametric-losses",
            "gbdt-lossguide-newtoncosine", "gbdt-exact-mae", "gbdt-adapter-reg", "par-forest-et")
_batch_decl(_rows_calls("predict", prep=_coded), "gbdt-feature-freq", "gbdt-categorical-ctr")
_batch_decl(_rows_calls("predict", prep=_with_nan), "gbdt-nan-modes")


def _batch_gbdt_adapter_clf(ml, e, Xh):
    return (_rows_calls("predict", "predict_proba")(ml, e, Xh)
            + _rows_calls("decision_function", sl=slice(0, 512))(ml, e, Xh))


_batch_decl(_batch_gbdt_adapter_clf, "gbdt-adapter-clf")


def _batch_iforest(ml, e, Xh):
    return (_rows_calls("score_samples")(ml, e, Xh)
            + _rows_calls("predict", sl=slice(0, 512))(ml, e, Xh))


_batch_decl(_batch_iforest, "iforest", "iforest-tuned", "par-iforest")

_batch_decl("n/a:no-predict", "kmeans", "kmeans-random", "kmeans-array", "kmeans-weighted", "kmeans-sqrt",
            "kmeans-classic-pp", "kmeans-cosine", "par-kmeans")
_batch_decl("n/a:transductive", "dbscan", "dbscan-brute-l1", "dbscan-weighted", "par-dbscan", "agglomerative",
            "spectral", "spectral-precomputed", "hdbscan", "hdbscan-leaf")


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
            "minmax-scaler-clip", "par-scaler", "rbf-sampler")
_batch_decl(_rows_calls("predict", sl=slice(0, 256)), "ols", "ridge", "ols-no-intercept",
            "ols-weighted", "ridge-no-intercept", "par-gram", "svr", "svr-linear")
# The coordinate descent predict refuses one row BY NAME, mirroring cuML's
# cdPredict (cd.cuh:341); see _BatchRows. Its rows are asked in windows of two.
CD_PREDICT_REFUSAL = "Parameter n_rows: number of rows cannot be less than two"
_batch_decl(_rows_calls("predict", sl=slice(0, 256), min_batch=2, refusal=CD_PREDICT_REFUSAL),
            "lasso", "elasticnet", "elasticnet-l2end-no-intercept", "par-cd")
_batch_decl(_rows_calls("predict_proba", sl=slice(0, 256)), "logistic", "logistic-l1", "logistic-elasticnet",
            "logistic-unpenalized-no-intercept", "par-logistic")
_batch_decl(_rows_calls("predict_proba", "predict", sl=slice(0, 256)), "logistic-multiclass")
_batch_decl(_rows_calls("decision_function", "predict", sl=slice(0, 256)), "svc", "svc-linear", "par-svm")
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


_batch_decl(_batch_gp, "gp", "gp-matern12", "gp-matern32", "gp-matern52-ard", "par-gp")
# UMAP.transform is batch-dependent BY ITS OWN CONTRACT: umap/transform.mojo's
# module docstring says "Query batching may change results (global sigma
# floor, edge weighting and RNG ordinals)", and the part measured it on the
# M4 (base fixture, 2026-09-14 night: held-out row 0 alone 0xbfb692af against
# 0xbf99e1c6 in the batch of 64). A BATCH_MOVED there is the documented
# algorithm, not a defect, so the part records the reason instead of failing
# every IDENTICAL run; the infer column still hashes the whole-batch transform.
_batch_decl("n/a:batch-dependent-by-contract (umap/transform.mojo: query batching may change results)",
            "umap", "par-graph-umap")
_batch_decl(_rows_calls("predict", sl=(slice(0, 64), slice(0, 4))), "kernel-ridge")
_batch_decl(_rows_calls("transform", sl=(slice(0, 64), slice(0, 4))), "nystroem")
_batch_decl(_rows_calls("score_samples", "predict", "predict_proba", sl=(slice(0, 64), slice(0, 4))), "gmm")
_batch_decl(_rows_calls("score_samples", sl=(slice(0, 64), slice(0, 4))), "gmm-random-init")


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
_batch_decl(_batch_gemm_transposed, "gemm-transposed")
_batch_decl("n/a:function", "metrics", "metrics-classification", "cross-val", "bootstrap", "permutation-test",
            "monte-carlo")
_batch_decl("n/a:no-batch-axis", "tokenizer")
_batch_decl("n/a:optimizer-step", "optim-sgd", "optim-adam-clip")
_batch_decl("n/a:training-step", "byte-lm-host-train")
_batch_decl("n/a:no-model", "par-byte-lm")


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


_batch_decl(_batch_ivf, "ivf", "ivf-euclidean")
_batch_decl(_batch_embedding, "embedding", "embedding-sort")
_batch_decl(_batch_cross_entropy, "cross-entropy-arms")
_batch_decl(_batch_training_primitives, "training-primitives")
_batch_decl(_rows_calls("predict_logits", sl=(slice(0, 256), slice(0, 8))), "mlp", "par-mlp")

#: the sequence models' held-out batch: 8 sequences of 16, so eight rows alone
SEQ_BATCH, SEQ_LEN = 8, 16


def _batch_byte_lm(ml, e, Xh):
    """logits `[B, L, vocab]`, each sequence alone against 16 of them, and
    every prefix length against the whole length (the model is causal and
    prefills from position 0, so position t may not read the length)."""
    shape = ml.ByteLanguageModelConfig()
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


_batch_decl(_batch_block, "mamba1", "mamba2", "mamba3", "mamba2-dtlimit", "transformer", "transformer-window")


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
_batch_decl(_batch_pq("radius_neighbors", sl=slice(0, 64), ragged=True, sort_results=True), "par-queries-radius")
_batch_decl(_batch_pq("score_samples", sl=(slice(0, 256), slice(0, 4))), "par-queries-kde")
_batch_decl(_batch_rsn("kneighbors", "predict", "predict_proba"), "par-reference-knn")
_batch_decl(_batch_rsn("predict"), "par-reference-knn-reg")
_batch_decl("n/a:transductive", "par-graph-agglomerative", "par-graph-spectral")
_batch_decl(_rows_calls("predict"), "par-ordered-rmse")
_batch_decl(_rows_calls("predict", prep=_coded), "par-feature-freq")
_batch_decl(_rows_calls("predict", "predict_proba"), "par-boosting-pointwise")
_batch_decl(lambda ml, e, Xh: [_BatchPrefix("forecast", FORECAST_HORIZON, lambda h: (e.forecast(h),), axis=0)],
            "par-holtwinters")
_batch_decl("n/a:no-model", "par-byte-lm-model-pool", "par-byte-lm-offload")
_batch_decl(_rows_calls("predict", "predict_proba"), "par-forest-pool")
_batch_decl(_rows_calls("score_samples", "predict", sl=(slice(0, 64), slice(0, 4))), "par-gmm")
_batch_decl("n/a:function", "par-resample")
_batch_decl("n/a:transductive", "par-hdbscan")
_batch_decl(_rows_calls("predict", sl=(slice(0, 64), slice(0, 4))), "par-kernel-ridge")
_batch_decl(_rows_calls("transform", sl=(slice(0, 64), slice(0, 4))), "par-nystroem")
_batch_decl(_rows_calls("transform", sl=slice(0, 256)), "par-rbf-sampler")
def _batch_par_cholesky(ml, e, Xh):
    """_batch_cholesky at the par-cholesky lane's 600-row factor."""
    B = np.ascontiguousarray(Xh[:600, :2])
    return [_BatchRows("solve (right-hand sides)", B.T,
                       lambda r: (np.asarray(e.solve(np.ascontiguousarray(r.T))).reshape(B.shape[0], -1).T,))]


_batch_decl(_batch_par_cholesky, "par-cholesky")


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
        return None, f"rlpair: {type(exc).__name__}: {exc}", sides


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
    shape = ml.ByteLanguageModelConfig()
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
_rlpair_decl(_rlpair_block_spec(), "mamba1", "mamba2", "mamba3", "mamba2-dtlimit")
_rlpair_decl(_rlpair_block_spec("kv"), "transformer", "transformer-window")
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


def _probe_fit(fit, name):
    """The infer and model columns of ONE fit. Returns (infer, model, reload,
    error) where infer and model are a hash or an `n/a:<reason>` string,
    reload is the held-out hash of the loaded-back file or None, and error is
    the exception text of whichever stage raised (with the stage name). A
    stage that raised leaves its column None, which reads REFUSED."""
    if not callable(fit.probe):
        return fit.probe, "n/a:no-save", None, None
    try:
        infer = _h(*fit.probe(fit.est))
    except Exception as exc:
        return None, None, None, f"infer: {type(exc).__name__}: {exc}"
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
            reload = _h(*fit.probe(back))
    except Exception as exc:
        return infer, None, None, f"model: {type(exc).__name__}: {exc}"
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
    import mojolearn as ml
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
    rlpair_protocol = dict(seqs=RLPAIR_SEQS, prompt=RLPAIR_PROMPT, decode=RLPAIR_DECODE,
                           trainer_split=list(RLPAIR_SPLIT) + ["n"], joiner=RLPAIR_JOINER,
                           join_at=RLPAIR_JOIN_AT, join_row=RLPAIR_JOIN_ROW, leaver=RLPAIR_LEAVER,
                           leave_after=RLPAIR_LEAVE_AFTER, block_vocab=RLPAIR_BLOCK_VOCAB,
                           logprob="training.cross_entropy(reduction='none')", enabled=not args.no_rlpair)
    if rlpair_sabotage:
        print(f"# {RLPAIR_SABOTAGE_ENV} is ON: the first sampler log-probability of every rlpair cell is "
              "perturbed by one low-bit flip; every rlpair cell with a hash MUST read RLPAIR_MOVED. "
              "This JSON is not evidence.")
    undeclared = [n for n in lanes if n not in BATCH]
    if undeclared and not args.no_batch:
        print(f"# WARNING: no batch declaration for {undeclared}; their batch part reads n/a:UNDECLARED")

    def dump(complete):
        record = dict(mode=mode, repeats=args.repeats, platform=platform.platform(),
                      vendor=vendor, commit=commit, commit_source=commit_source,
                      heldout_seed=HELDOUT_SEED, fixtures=fixture_hashes,
                      heldout=heldout_hashes, cells=cells, complete=complete,
                      skipped=sorted(skip), batch_protocol=batch_protocol,
                      batch_sabotage=batch_sabotage, rlpair_protocol=rlpair_protocol,
                      rlpair_sabotage=rlpair_sabotage)
        if host is not None:
            record["host"] = host
        # every binding this process loaded, hashed, so the column is tied
        # to the BYTES it ran and not only to a commit (2026-09-14); a
        # column from an installed wheel and one from a source build of the
        # same commit are different evidence, and the 0.8.5 release legs
        # caught two packaging regressions a source column could not see
        try:
            from mojolearn._verify import binding_artifacts
            package["bindings"] = [dict(module=b["module"], sha256=b["sha256"], size=b["size"])
                                   for b in binding_artifacts()]
        except Exception as exc:                        # never lose a column over provenance
            package["bindings_error"] = f"{type(exc).__name__}: {exc}"[:200]
        record["package"] = package
        with open(args.json, "w") as fh:
            json.dump(record, fh, indent=1)

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
        for f in fixtures:
            X, yc, yr = data[f]
            hs, parts, err = [], [], None
            infers, models, reloads, errs2 = [], [], [], []
            batches, errs3 = [], []
            rls, errs4, rl_sides = [], [], None
            for repeat in range(args.repeats):
                try:
                    # each fit gets its own held-out bytes, so a lane cannot
                    # hand the next fit rows it wrote to
                    global _DUMP_TAG
                    _DUMP_TAG = f"{name}/{f}/{repeat}"
                    p = LANES[name](ml, X, yc, yr, held[f].copy())
                    parts.append(dict(p))
                    hs.append(_h(np.frombuffer("|".join(f"{k}={v}" for k, v in sorted(p.items())).encode(), dtype=np.uint8)))
                except Exception as exc:
                    err = f"{type(exc).__name__}: {exc}"
                    if args.verbose:
                        traceback.print_exc()
                    break
                inf, mod, rel, err2 = _probe_fit(p, name)
                infers.append(inf); models.append(mod); reloads.append(rel)
                if err2:
                    errs2.append(err2[:300])
                    if args.verbose:
                        traceback.print_exc()
                # the batch part runs after the infer and model columns so it
                # cannot change what they hash; its own copy of the held-out rows
                if args.no_batch:
                    bat, err3 = "n/a:skipped (--no-batch)", None
                else:
                    bat, err3 = _probe_batch(p, name, ml, held[f].copy(), args.batch_alone, batch_sabotage)
                batches.append(bat)
                if err3:
                    errs3.append(err3[:300])
                    if args.verbose:
                        print(err3)
                # the rlpair part last, on the lanes that declare it; a lane
                # that does not carries no rlpair key at all
                if name in RLPAIR:
                    if args.no_rlpair:
                        rl, err4, sd = "n/a:skipped (--no-rlpair)", None, []
                    else:
                        rl, err4, sd = _probe_rlpair(p, name, ml, held[f].copy(), rlpair_sabotage)
                    rls.append(rl)
                    rl_sides = rl_sides if rl_sides is not None else sd
                    if err4:
                        errs4.append(err4[:300])
                        if args.verbose:
                            print(err4)
            if err:
                cell = dict(verdict="REFUSED", error=err[:300], hashes=hs, parts=parts)
                shown = "REFUSED"
            elif len(set(hs)) == 1:
                cell = dict(verdict="STABLE", hashes=hs, parts=parts)
                shown = hs[0]
            else:
                cell = dict(verdict="MOVED", hashes=hs, parts=parts)
                shown = "MOVED " + hs[0][:8]
            # the two new columns; a REFUSED train column has no fit to probe
            if not err:
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
            else:
                row_infer.append("REFUSED"); row_model.append("REFUSED"); row_batch.append("REFUSED")
                if name in RLPAIR:
                    row_rl.append("REFUSED")
            cells[f"{name}/{f}"] = cell
            row.append(f"{shown:<16}")
        print(f"| {name:<{W}} | " + " | ".join(row) + " |", flush=True)
        for label, r in (("infer", row_infer), ("model", row_model), ("batch", row_batch), ("rlpair", row_rl)):
            if any(not s.startswith("n/a") for s in r):
                print(f"| {name + ' ' + label:<{W}} | " + " | ".join(f"{s:<16}" for s in r) + " |", flush=True)
        if args.json:
            # written after EVERY lane so a hang that gets killed by the
            # caller's timeout still leaves the finished lanes on disk
            dump(False)

    refused = {k: v["error"] for k, v in cells.items() if v["verdict"] == "REFUSED"}
    moved = [k for k, v in cells.items() if v["verdict"] == "MOVED"]
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
    for col in ("infer", "model", "batch", "rlpair"):
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
    return 1 if (moved or moved2 or moved3 or batch_fails or moved4 or rl_fails) else 0


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
    real = [h for h in hashes if h not in ("MOVED", "RELOAD-MOVED", "REFUSED", "BATCH_MOVED", "RLPAIR_MOVED")]
    if "RELOAD-MOVED" in hashes:
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


def diff(paths, require_columns=0, require_lanes=None):
    cols = []
    for p in paths:
        with open(p) as fh:
            j = json.load(fh)
        cols.append((j.get("vendor") or os.path.basename(p), j))
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
    short = []

    def require(key, col, verdict):
        if not require_columns or key.split("/")[0] not in required:
            return
        n = _real_count(verdict)
        if n is not None and n < require_columns:
            short.append((key, col, verdict, n))

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
            if c["verdict"] == "MOVED":
                shown.append("MOVED"); vals.append("MOVED"); continue
            shown.append(c["hashes"][0]); vals.append(c["hashes"][0])
        ran = [v for v in vals]
        if "MOVED" in ran:
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
        counts[verdict.split(" ")[0]] = counts.get(verdict.split(" ")[0], 0) + 1
        require(k, "train", verdict)
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
    # A THIRD LINE, `summary (rlpair):` (2026-09-15), over the lanes that
    # declare the rlpair part (RLPAIR) or whose cells carry it; the other
    # lanes have no rlpair question and are not counted at all.
    uncarried = {"infer/model": 0, "batch": 0, "rlpair": 0}
    counts2 = {"infer/model": {}, "batch": {}, "rlpair": {}}
    for k in keys:
        for col in ("infer", "model", "batch", "rlpair"):
            group = col if col in ("batch", "rlpair") else "infer/model"
            carried = any(f"{col}_verdict" in (j["cells"].get(k) or {}) for _, j in cols)
            if not carried:
                if col != "rlpair" or k.split("/")[0] in RLPAIR:
                    uncarried[group] += 1
                continue
            verdict, shown = _diff_column(cols, k, col)
            if verdict in ("MOVED", "DIVERGENT", "RELOAD-MOVED", "BATCH_MOVED", "RLPAIR_MOVED"):
                bad += 1
            c2 = counts2[group]
            c2[verdict.split(" ")[0]] = c2.get(verdict.split(" ")[0], 0) + 1
            require(k, col, verdict)
            rows.append((k, col, verdict, shown))
    print()
    if rows:
        print(f"| {'lane/fixture':<28} | {'column':<6} | {'verdict':<12} | " + " | ".join(f"{n:<16}" for n in names) + " |")
        print(f"|{'-'*30}|{'-'*8}|{'-'*14}|" + "|".join("-" * 18 for _ in names) + "|")
        for k, col, verdict, shown in rows:
            print(f"| {k:<28} | {col:<6} | {verdict:<12} | " + " | ".join(f"{s:<16}" for s in shown) + " |")
        print()
    for group in ("infer/model", "batch", "rlpair"):
        c2 = counts2[group]
        if uncarried[group]:
            c2["NOT-COMPARED"] = c2.get("NOT-COMPARED", 0) + uncarried[group]
            print(f"{group}: {uncarried[group]} column cells not compared (no JSON here carries them; "
                  f"they predate the {group + ' part' if group != 'infer/model' else 'columns'})")
        print(f"summary ({group}): " + ", ".join(f"{k}={v}" for k, v in sorted(c2.items())))
    if require_columns:
        if require_columns > len(cols):
            bad += 1
        for key, col, verdict, n in short:
            print(f"REQUIRE FAIL {key} {col}: {verdict} rests on {n} real hash(es), "
                  f"--require-columns {require_columns} demands that many; a column that "
                  "should cover this lane is refusing, which is not a pass")
        missing = sorted(required - set(k.split("/")[0] for k in keys))
        for lane_name in missing:
            print(f"REQUIRE FAIL {lane_name}: no JSON carries a cell for this lane")
        bad += len(short) + len(missing)
        print(f"require-columns {require_columns} over {sorted(required)}: "
              f"{'OK' if not short and not missing else str(len(short) + len(missing)) + ' short'}")
    return 1 if bad else 0


#: the keys every part of one column must agree on before --merge joins them
MERGE_SAME = ("vendor", "commit", "mode", "repeats", "heldout_seed", "fixtures", "heldout",
              "batch_protocol", "batch_sabotage", "rlpair_protocol", "rlpair_sabotage")


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
        if j.get("batch_sabotage"):
            raise SystemExit(f"REFUSING --merge: {p} is a batch sabotage run")
        if j.get("rlpair_sabotage"):
            raise SystemExit(f"REFUSING --merge: {p} is an rlpair sabotage run")
        pk, fk = j.get("package") or {}, first.get("package") or {}
        if pk.get("par_devices") != fk.get("par_devices"):
            raise SystemExit(f"REFUSING --merge: {p} ran par_devices={pk.get('par_devices')!r}, "
                             f"{parts[0][0]} ran {fk.get('par_devices')!r}")
        dig = sorted((b["module"], b["sha256"]) for b in pk.get("bindings", []))
        same_build = dig and dig == sorted((b["module"], b["sha256"]) for b in fk.get("bindings", []))
        if not same_build and not (allow_separate_builds and dig):
            raise SystemExit(f"REFUSING --merge: {p} loaded different (or unrecorded) binding bytes than "
                             f"{parts[0][0]}; the parts of one column must run one build")
    cells = {}
    for p, j in parts:
        for k, c in j["cells"].items():
            if k in cells and json.dumps(cells[k], sort_keys=True) != json.dumps(c, sort_keys=True):
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
    record["merged_separate_builds"] = len(set(json.dumps(sorted((b["module"], b["sha256"]) for b in
                                                                 (j.get("package") or {}).get("bindings", [])))
                                               for _, j in parts)) > 1
    with open(out, "w") as fh:
        json.dump(record, fh, indent=1)
    print(f"merged {len(parts)} parts, {len(cells)} cells, complete={record['complete']} -> {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", default="")
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
    ap.add_argument("--no-rlpair", action="store_true",
                    help="skip the rlpair part; the lanes that declare it record n/a:skipped (--no-rlpair)")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--diff", nargs="+", default=None, metavar="JSON",
                    help="compare JSONs cell by cell: the train column, then infer and model where carried")
    ap.add_argument("--require-columns", type=int, default=0, metavar="N",
                    help="with --diff: exit non-zero unless every compared cell of the lanes named by "
                         "--lanes (every lane when --lanes is empty) rests on at least N real hashes; "
                         "--lanes also scopes which cells --diff compares at all")
    ap.add_argument("--merge", nargs="+", default=None, metavar="JSON",
                    help="join the parts of ONE column (same vendor, commit, fixtures, build) into --json")
    ap.add_argument("--allow-separate-builds", action="store_true",
                    help="with --merge: admit parts whose binding digests differ (two legs, two builds of the "
                         "same commit on the same box type); every part's digests are kept under merged_from")
    args = ap.parse_args()
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
        return diff(args.diff, args.require_columns, lanes)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
