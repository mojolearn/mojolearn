# Handoff 2026-09-20: every algorithm on PyPI, CPU and GPU arms, in the verifier

Owner's goal: every algorithm in the repo is exposed on PyPI with a CPU arm and
a GPU arm, and each is in the verifier harness. Work tree:
`/Users/andrewhendel/mojolearn-wt/takeover-exposure`, branch
`lane/takeover-exposure`, which equals `origin/main` at the time of writing.

## Release integration update (2026-09-20, after the original handoff)

The open-item list below is historical. The release integration is merged and
pushed to main (integration merge `c44fd9fde`). The native 0.8.9 candidate is
frozen at `a97676ba18a09fb577ef9faae45ab19a01eec848`; publication is pending
fresh platform builds and installed-wheel qualification.

- QN's six objective lanes are merged; all 54 CPU/NVIDIA cells agree.
- The CPU GBDT implementation was integrated and rebuilt. The public CPU
  configuration matrix now passes **41/41**, including non-symmetric policies.
- The device held-out cursor is initialized when `boost_from_average=False`.
  The restored `gbdt-symmetric-eval` lane agrees across fresh CPU and Metal
  builds, including every numeric part.
- New stochastic, multiclass-default, and ranking-default lanes have matching
  CPU/Metal evidence committed under `bench/results/identity_break/`.
- Explicit `boost_from_average=True` now works for its four supported losses;
  the seven release regressions pass on fresh CPU and on Metal.
- Accounting passes: **273 lanes**, all with table cells; **214 exposed**,
  **59 parallel lanes not applicable to a single-device invocation**; no
  public algorithm lacks a lane. The verification matrix matches 255 public
  API entries. This is coverage accounting, not a claim that every final
  wheel has already completed qualification.
- Fresh source-built x86 CPU bindings reproduce all nine language-model-config
  fixtures. The generic default-constructor smoke was interrupted during a
  long 1000-tree Ordered GBDT fit; it is not recorded as passing. The bounded
  41-configuration GBDT matrix is the completed CPU fit evidence.

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

1. **Merge `origin/lane/gbdt-cpu-default-parity` (tip 8a9f08dd4, finished, NOT
   merged).** The default `GradientBoostingRegressor()` and
   `GradientBoostingClassifier()` train on a CPU and match an RTX 4090 bit for
   bit: new lane `gbdt-stochastic-arms`, 36 of 36 cells, each fitted once,
   admitted into that branch's table. Public matrix on a CPU pod went from 10
   of 41 fits to 36 of 41 (`bench/results/gbdt_cpu_parity/2026-09-20/`). A
   merge into main conflicts in EIGHT files because the QN and
   SpectralEmbedding merges landed first: `README.md`, `SUPPORT_MATRIX.md`,
   `docs/BYTE_LM_CPU_TRAINING.md`, `docs/VERIFICATION_MATRIX.md`,
   `python/mojolearn/verify_reference/table.json` (generated: take main's and
   regenerate with the commands below), `python/mojolearn/host_surface.py`,
   `tools/identity_break.py` and `bindings/_mojolearn_gbdt_host.mojo` (both
   sides add entries; keep both). The merged binding and harness have not been
   compiled together, so build the gbdt host family and rerun
   `gbdt-stochastic-arms` on a CPU pod after resolving. I aborted the merge
   rather than push a hand-merged binding nobody had compiled.
   Still refused on a CPU after that branch: MultiClass and OneVsAll with any
   bootstrap or noise (so their default fit), the Poisson bootstrap on the
   non-symmetric policies, and the pointwise losses under Depthwise and
   Lossguide.
2. **`gbdt-symmetric-eval`: CPU and GPU DISAGREE, and the evidence points at
   the DEVICE.** On an RTX 4090 the held-out curve changes between identical
   fits in one process while the learn curve does not, differs from the learn
   curve when the eval set IS the learn set (equal on a CPU), and starts above
   ln 2. `fit_with_test` in `gbdt/methods/doc_parallel_boosting.mojo` fills
   the test cursor only under `boost_from_average`; the Ordered fit fills it
   unconditionally. Looks like a one-line device fix; not made (that file is
   under the performance lanes). Values and the diagnostic script:
   `bench/results/identity_break/2026-09-20_gbdt-cpu-default-parity/` on the
   branch in item 1. The lane stays in `PUBLIC_PENDING_LANES`. Separately, the
   device's default RMSE fit takes its leaves from the bootstrapped stats
   planes, which DEVIATION 64's argument does not cover; the CPU matches the
   device there.
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
