# Neural-only IDENTICAL source experiments

The [64-card idea list](../../docs/plans/NEURAL_IDENTICAL_AB_IDEAS_2026-10-06.md)
was authored before implementation fan-out. It covers neural GEMM, attention,
transformers/LMs, Mamba/Samba, recurrent networks, MLP/MoE, CNN/ResNet, neural
graph layers, embeddings, losses, optimizers and native neural ownership.
Classical estimators and classical forecasts are excluded.

This worktree was created from **main at
`fd6cf80453a6f18eb02e81566c824e7da106ccf0`** on branch
`ideas/neural-identical-ab-20261006-r3`. Other worktrees and active measurement
campaigns were not changed by this work.

## Numerical rule

Bits may change between A and B or between releases. For each selected
arm/version, all promised outputs, gradients, state and error decisions must
match across NVIDIA, AMD, Apple and host. An arithmetic revision needs the
same explicit graph in every affected host/device, backward, checkpoint and
decode path. Old-version equality is not the acceptance criterion. Preserving
model quality remains mandatory and is separate from bitwise agreement.

Every new source candidate is opt-in. Existing enabled optimizations are
identified as inherited; this change does not claim them as new wins. A
compiled flag, a no-op fallback or a standalone kernel is not proof that a
full model reaches a candidate. Shared primitives here are explicitly neural
experiments, not new defaults for classical callers.

## Delivery and evidence status

The user explicitly requested **no compilation or verification**. No builds,
tests, static checkers, candidate execution, benchmarks, remote GPU jobs or
measurement-board updates were performed. The document-authoring Python
scripts only wrote metadata/documents. Mamba host source was refreshed through
an explicit source-write-only mode that disables its verification phase. The inspection/planning CLI below was
written but not executed as a check.

Read [what model integration means](MODEL_INTEGRATION.md), the [implementation status](IMPLEMENTATION_STATUS.md) and per-lane source
handoffs before choosing an experiment:

- [GEMM](lanes/gemm.md), [machine-readable handoff](lanes/gemm.json)
- [Attention/transformer](lanes/attention.md), [machine-readable handoff](lanes/attention.json)
- [State-space/recurrent/CNN](lanes/state_cnn.md), [machine-readable handoff](lanes/state_cnn.json)
- [Embedding/loss/optimizer/runtime](lanes/training.md), [machine-readable handoff](lanes/training.json)

`wired_draft` means a new opt-in source branch reaches a real native/public caller, including newly exposed complete model/layer operations.
`component_draft` means real native code exists but full caller/lifetime/profile
integration remains unfinished. `reused_existing` records prior source and
its controls; it is not a new implementation or qualification. `pending`
means concrete programming remains. Every category is uncompiled,
unverified and unmeasured for this work. Full end-to-end campaign recipes are
also pending; the catalog maps their authoritative source locations and does
not manufacture dataset hashes, full-size coverage or timing results.

## Experiment index and concrete arm selection

The [experiment/file inventory](../../docs/plans/NEURAL_AB_EXPERIMENT_INVENTORY_2026-10-06.md)
lists all 64 NN cards, the 60 existing I/A/N/F manifests, and additional A/B suites
found in the checkout. [arms.json](arms.json) records exact selected controls,
callers, prior experiment relationships and limitations; [experiment_inventory.json](experiment_inventory.json)
is the combined machine-readable index.

These commands are available for a later user invocation; they only read
metadata or author a concrete arm configuration. There is no execute/build/check/run
subcommand:

```sh
python3 tools/neural_identical_ab.py list
python3 tools/neural_identical_ab.py list --include-existing
python3 tools/neural_identical_ab.py show NN52
python3 tools/neural_identical_ab.py plan NN52 --vendor amd --arm candidate
python3 tools/neural_identical_ab.py configure NN52 --vendor amd --arm candidate \
  --output /outside/repo/nn52-a.json --env-output /outside/repo/nn52-a.env
```

The optional environment file supplies `MOJOLEARN_MOJO_BUILD_FLAGS` to the
existing binding builders and clears declared conflicting runtime controls.
Use one frozen source and the same profile settings for **all** affected GPU
and host bindings. Select B separately with `--arm baseline`; do not disable
`is_defined` flags by setting them to zero. NN10 requires an explicit recorded
`--parameter MOJOLEARN_IDN_NEURAL_FILL_BLOCKS=...` hardware budget; NN53 and NN60
expose named `--variant` options. The selector does not build or install binaries.

For a future authorized run, the existing neural board accepts the resulting
JSON with `--neural-ab-config`. It propagates runtime environment and explicit
operation settings and retains them in the output. This connects new tape,
owned-snapshot, chunked-head, residual/dropout and MLP-session APIs to selectable
workloads. A recorded config means **configured**, not proof that a binary was
built with those flags or that every affected model reached a candidate. The
full workload's recipe/hash, build provenance, identity and quality requirements
still apply. Distinct operations use our own arms only and do not invent
opponent ratios from different boundaries.

The existing I/A/N/F frozen-source executor remains
[`tools/performance_ideas.py`](../../tools/performance_ideas.py). Its retained
evidence protocol and historical receipts are not relabeled as NN results.

The idea inventory is in [catalog.json](catalog.json), authored by
[catalog_source.py](catalog_source.py). Implementation facts live separately
in lane handoffs, so a planned card cannot silently become a claimed pass.
Multiple handoffs for a cross-owner card are retained together. Sub-arm fields
carry actual defines/environment/config requirements; absent information is
pending, not a guessed command. Each arithmetic profile specifies its own
host/device operations and permitted parameterization.

## Future admission, not run here

Before future timing, resolve every affected estimator/model to the saved full
workload and audit corpus/dataset hash/version/split, actual dimensions,
intrinsic row/sequence caps, settings, seeds, flags and complete timed boundary.
Preparation, fit/training, synchronization and consumed outputs belong inside
that boundary; report inference/prefill/decode, cold and repeated use separately.
Tiny fixtures/component timings do not complete this work.

Freeze source and reuse matching accepted build/identity evidence. The initial
full NVIDIA and AMD A/B uses one excluded warmup and one scored sample,
serial cells per GPU and separate vendor boxes. A default requires combined
improvement with neither voting vendor materially slower and no quality loss.
Apple/host establish same-version identity; Apple timing does not vote.
Evaluate interactions and the complete proposed configuration. Preserve losers,
failures, pending coverage and provenance; update boards through board tools
only when real results exist. No experiment in this worktree is promoted.

> Keep logs out of context: save complete output to files, use targeted rg/grep
> with bounded surrounding lines and short tails, and summarize exit status,
> coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
> never hide failures or infer full success from filtered output.
