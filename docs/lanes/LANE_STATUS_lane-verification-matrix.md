# LANE STATUS: lane/verification-matrix (2026-09-16)

Andrew's question, Sep 16: "for every algorithm we ship, do we have all four
kinds of bitwise-identity verification?" The four kinds are a recorded GPU
column, a CPU verifier, a sabotage that has been SEEN to make the lane
divergent, and a declared batch-invariance part or a named `n/a:<reason>`.

Branch `lane/verification-matrix`, cut from origin/main `bfb8f725a`. Read-only
work. No kernel, no estimator and no lane changed. Nothing was rented and no
Metal job ran, because a detached script was recording the 0.8.6 Apple column.

## What landed

    tools/verification_matrix.py     the generator
    docs/VERIFICATION_MATRIX.md      its committed output
    docs/lanes/LANE_STATUS_lane-verification-matrix.md   this file

A GENERATOR, not a static document, so it cannot rot. It reads
`tools/identity_break.py` (the lane and BATCH tables),
`python/mojolearn/host_surface.py` (the CPU surface),
`python/mojolearn/_verify_reference.py` (the column admission rule, the same
one the shipped reference table uses), the committed columns under
`bench/results/`, and `__all__` of the package and of every public submodule.

    python3 tools/verification_matrix.py            # print
    python3 tools/verification_matrix.py --write    # rewrite the doc
    python3 tools/verification_matrix.py --check    # fail when the doc is stale
    python3 tools/verification_matrix.py --json     # the data

`--check` is NOT wired into any gate and should not be. This is an occasional
audit, not a per-commit tax. Re-run it after a batch of lanes merges.

## The answer, at this head

199 lanes (160 single-device, 39 `par-*` multi-GPU drivers) and 154 shipped
algorithms enumerated from the public API.

| | algorithms | lanes |
|---|---|---|
| all four kinds | 102 of 154 | 107 of 199 |
| gpu column | 139 | 194 (171 on all three classes) |
| cpu verifier | 140 | 169 |
| sabotage SEEN to move a build | 103 | 110 |
| batch part or named n/a | 140 | 199 |

Six shipped algorithms have no identity lane at all, and five of the six are
the saved-model host inference surface, measured by a different gate:
`HostForest`, `HostGBDT`, `host_model`, `host_predict`, `host_predict_proba`
(`tools/forest_host_gate.py` and `tools/classical_host_gate.py`, 25 and 21
committed recording sets). The sixth is a real hole,
`metrics.homogeneity_completeness_v_measure`, which nothing calls.

Eight more have no lane of their own but ARE exercised, through
`_public_est`'s CPU inference routing and the batchgrad part. They are
reported in their own bucket rather than counted as gaps.

## The claim this tool is most careful about

Sabotage "exists" and sabotage "has been seen to fail" are different claims
and the matrix keeps them apart. The only thing counted as proof is a
committed pair of columns whose HASHES DIFFER for that lane. The split at this
head is 110 `seen(build)`, 7 `seen(harness)`, 57 `declared`, 25 `none`.

`seen(harness)` means only the harness's own batch switch moved the cell. That
proves the batch PROBE can fail and says nothing about the implementation, so
it is never counted as a build sabotage. `declared` means the family declares
a define and no committed pair moves the lane here; several of those were
watched to fail in a CI gate run whose artifacts are not in the tree, which is
exactly the distinction worth keeping.

## Reconciling the Sep 15 evening starting facts

Every one was re-measured from the tree rather than trusted.

| stated Sep 15 | at `bfb8f725a` | why |
|---|---|---|
| 195 lanes | 199 | arima-exog added two, gp-optimize two more |
| 178 real batch parts, 17 named n/a, 0 undeclared | 182, 17, 0 | the four new lanes declared real parts |
| CPU verifier 167 of 195 | 169 of 199 | two more covered |
| the 28 missing are two-device lanes | 30 missing, 28 of them `par-*` | `gp-optimize` and `gp-optimize-restarts` are new and single-device |

`docs/lanes/LANE_STATUS_lane-cpu-training-par-wave3.md` still triages the 28
`par-*` lanes correctly and nothing here supersedes it.

## Ranked gaps, cheapest first

1. **`gp-optimize` and `gp-optimize-restarts`**, the only single-device lanes
   with no CPU verifier declared, and carried by Apple alone among the GPU
   columns. The CPU column exists
   (`bench/results/identity_break/2026-09-15_gp-optimize/x86-runpod/cpu.json`,
   63 of 63 STABLE) and its sabotage arm exists and MOVED (`cpu_gsab.json`,
   `-D MOJOLEARN_GP_GRAD_SABOTAGE=1`), so both lanes already read
   `seen(build)`. What is missing is the declaration in `host_surface.py` and
   an NVIDIA and an AMD column at the next record.
2. **`metrics.homogeneity_completeness_v_measure`**, one uncalled metric. Add
   it to the `metrics-classification` lane. Smallest real hole in the matrix.
3. **Five lanes with no GPU column at all**: `arima-exog`,
   `arima-exog-seasonal`, `gbdt-adapter-score-weighted`,
   `metrics-fowlkes-mallows`, `rf-score-weighted`. All merged after the last
   record. They are owed a column at the next release record, not a rental of
   their own.
4. **The 48 single-device lanes whose sabotage is `declared` but never seen to
   move in a committed pair**, the trees and GBDT families worst of all
   (`rf-clf`, `rf-reg`, `et-clf`, `et-reg`, `gbdt-symmetric`, `gbdt-rmse` and
   their arms), plus `knn`, `kde`, `logistic`, `pca-whiten`, `tsvd`,
   `gemm-pinned` and the optimizer lanes. The CPU gate builds a sabotage host
   set every run, so the evidence is produced and then thrown away. The cheap
   fix is to commit one sabotage column beside the clean one on the next CPU
   leg, not to invent new arms.
5. **`pca`, `byte-lm-host-infer` and `byte-lm-host-infer-threaded`**, where the
   only recorded move is the harness batch switch. Their build sabotage is
   unproven in the tree.
6. **The 28 `par-*` lanes with no CPU verifier**, already triaged in wave 3,
   and the `par-*` sabotage which follows from it.

## Resume steps for a session with none of this context

Read `docs/VERIFICATION_MATRIX.md` first; it states its own method. Then:

**1. Take a worktree and sync.**

    cd /Users/andrewhendel/CascadeProjects/mojolearn   # SHARED: never build or commit here
    git worktree add -b <branch> <scratch>/wt-vmatrix origin/main
    cd <scratch>/wt-vmatrix && git fetch origin main && git merge origin/main

**2. Re-measure. Costs one core and about twenty seconds.**

    nice -n 19 python3 tools/verification_matrix.py --check     # is the doc stale?
    nice -n 19 python3 tools/verification_matrix.py --write     # rewrite it
    nice -n 19 python3 tools/verification_matrix.py --json | python3 -m json.tool | head

**3. Close a gap.** Gap 1 and gap 2 above need no box at all. Gap 4 needs one
line added to the next RunPod CPU leg's command, committing the sabotage
column beside the clean one:

    bash tools/runpod_cpu_leg.sh --lane <name> --build <families> \
      --sabotage-build <families> \
      --cmd '... --json "$LEG_OUT/cpu-x86.json" && \
             MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage \
             MOJOLEARN_HOST_ALLOW_SABOTAGE=1 python3 tools/identity_break.py \
               ... --json "$LEG_OUT/cpu-x86.host-sabotage.json"'

The generator pairs `cpu-x86.host-sabotage.json` with `cpu-x86.json`
automatically when both are committed in the same directory at the same
commit, and the lane flips to `seen(build)` on the next run.

**4. Merge.** `git merge origin/main`, then

    python3 tools/docs_facts.py --check
    python3 packaging/wheel_ci.py pins .
    python3 packaging/wheel_ci.py inventory python/mojolearn
    python3 tools/verification_matrix.py --check

then `git push origin HEAD:<branch> && git push origin HEAD:main`, and confirm
it landed. Never wait on CI. Main only; never touch release/0.8.6.

## Rules this lane ran under

One core, `nice -n 19`, one process at a time, its own worktree. No Metal lock
and no Metal job. Nothing rented. No `git stash` in the shared checkout, no
`git add -A`, branch checked immediately before the commit.

## Pods

None rented on this branch.
