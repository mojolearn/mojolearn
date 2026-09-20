# Handoff 2026-09-20: every algorithm on PyPI, CPU and GPU arms, in the verifier

Owner's goal: every algorithm in the repo is exposed on PyPI with a CPU arm and
a GPU arm, and each is in the verifier harness. Work tree:
`/Users/andrewhendel/mojolearn-wt/takeover-exposure`, branch
`lane/takeover-exposure`, which equals `origin/main` at the time of writing.

## Rules that changed today (owner's decisions)

- CPU training is PUBLIC. `fit` on a CPU-only install trains
  (`python/mojolearn/_cpu_reference.py` refuses nothing). Never bring the
  refusal back.
- Every cell is fitted ONCE. Table admission
  (`python/mojolearn/_verify_reference.py`, `ADMISSION_POLICY`) takes one fit
  as a value and admits it when a second device class carries the same hash.
  `tools/verification_matrix.py` counts a one-fit column.
  `tools/gap_column_leg.sh` fits once by default (`MOJOLEARN_GAP_REPEATS`).
- No new gates, refusals, probes or protocols in the verifier.

## State on main

270 lanes: 210 EXPOSED, 59 NOT APPLICABLE (every `par-*` driver; all 59 now
carry a two-device column, checked with `verify --par`), 1 OWED.
`python3 tools/lane_accounting.py --check` prints OK.

Landed today: the peer's NVIDIA columns (squashed), both decode sessions,
`language-model-config`, the last four `par-*` lanes
(`bench/results/identity_break/2026-09-20_takeover-last-gpu-columns/`), and
`SpectralEmbedding` (class, both bindings, lane, CPU == RTX 4090 on nine
fixtures), and the six QN objectives.

## Open, in priority order

1. **`gbdt-symmetric-eval`: CPU and GPU DISAGREE.** First GPU run of the lane.
   The held-out loss curve differs on all nine fixtures; the model differs on
   `dupes` and `negative`; `wide` refuses on the GPU because the shrink never
   cuts. Evidence: the `gbdt-symmetric-eval` JSON in the directory above versus
   the CPU column from lane/close-no-cpu-path-gbdt. Suspect
   `gbdt/host/gbdt_oracle_eval.mojo`. The device is the reference. The lane is
   back in `PUBLIC_PENDING_LANES`.
2. **Gradient boosting barely trains on a CPU.** The host side is
   per-configuration oracles (`gbdt/host/`). Measured against a Sep 19 binding:
   3 of 19 plain configurations trained; the DEFAULT regressor and classifier
   refused (Bayesian and Bernoulli bootstrap, `random_strength`, RMSE under
   Depthwise and Lossguide). Branch `origin/lane/gbdt-cpu-default-parity`
   (4 commits, in progress, NOT merged): a matrix script, the before matrix,
   and the default regressor training on a CPU. Not yet shown equal to a GPU.
3. **Six QN objectives: MERGED.** `mojolearn.svm.LinearSVC`, `LinearSVR` and
   `mojolearn.QNRegressor`, six lanes, CPU == RTX 4090 on all 54 cells. Left
   over: the classes have no save/load, no lane fits the l1 penalty on the
   hinge and QN losses, and `LinearSVR` defaults to `penalty='l1'` like the
   reference library, not like scikit-learn.
4. **Evidence taken against Sep 19 binaries, to redo from today's source.**
   The built bindings in the main checkout (`python/mojolearn/host`,
   `identical`, `.dylibs`) date from Sep 19 and are symlinked into the work
   tree. Three results used them: the `language-model-config` CPU column that
   is in the shipped table, the CPU-only public `fit` smoke run, and the
   gradient boosting matrix in item 2. One command redoes all three on a rented
   CPU pod, built from source. It was dry-run only and never rented:

       export RUNPOD_API_KEY=$(cat ~/.mojolearn_runpod_key)
       FAM=$(python3 python/mojolearn/host_surface.py --families --sep ,)
       bash tools/runpod_cpu_leg.sh --rent --lane today-source-cpu --build "$FAM" \
         --vcpu 16 --jobs 16 --out ~/mojolearn-evidence/takeover/cpu-today \
         --cmd 'python3 tools/identity_break.py --lanes language-model-config --repeats 2 --vendor cpu-x86 --json "$LEG_OUT/cpu-x86.language-model-config.json"; python3 tools/cpu_public_fit_smoke.py --out "$LEG_OUT/cpu_public_fit.tsv"; python3 -m mojolearn verify --all --no-models --json-out "$LEG_OUT/verify_all.json" > "$LEG_OUT/verify_all.txt" 2>&1'

   Everything that came from a rented pod today was built from source on the box.
5. **Chunked LM-head v2** is bound (`bindings/_mojolearn_training.mojo:749`)
   with no Python caller. Deferred: the performance lanes are still changing it.
6. Not started: 8061 of 17118 table parts rest on ONE device class;
   266 committed files carry a RunPod `consumerUserId`; `tools/docs_facts.py`
   exits 1 on main; eight manifest tests under
   `python/mojolearn/tests/test_cpu_training_gbdt*` and `test_cpu_training_gp`
   fail on main (drift from today's performance merges);
   `lane/par-sabotage-defines` (34 unrun Mojo arms) and
   `lane/checks-that-cannot-fail-sweep` stay unmerged.

## Commands

Admit lanes into the shipped table after their columns are committed:

    cd python && python -m mojolearn verify --all --emit-reference OUT.json \
      --reference-table mojolearn/verify_reference/table.json --lanes A,B --batch-checks
    cp OUT.json mojolearn/verify_reference/table.json
    # then delete the lanes from PUBLIC_PENDING_LANES in python/mojolearn/host_surface.py
    python3 tools/verification_matrix.py --write && python3 tools/lane_accounting.py --check

A GPU column for named lanes (one pod, every lane in one wrapper): copy
`~/mojolearn-evidence/takeover/wrap_gbdt_eval.sh`, then
`MOJOLEARN_GEMM_LEG_EXTRA=<wrapper> MOJOLEARN_GEMM_LEG_LOCAL_CARD=<an apple.card> sh tools/gemm_remote_leg.sh nvidia --rent --minutes 60 --allow-concurrent`
(two GPUs: `MOJOLEARN_GEMM_LEG_GPU_COUNT=2` and `MOJOLEARN_GAP_TWO_DEVICE=1`).
A leg runs the column in about 3 minutes; the rest of its 20 is compiling
families the R2 cache missed.

Pytest: `/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python`.
