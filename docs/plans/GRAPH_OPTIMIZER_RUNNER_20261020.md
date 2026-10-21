# Captured Program optimizer measurement runner

Internal engineering record. The optimizer remains caller opt-in and defaults
off. This is executable measurement infrastructure, not a performance result or
completion of full-model/pass coverage.

`tools/bench_board_neural.py` now accepts the explicit A/B operation
`program_optimize` with a required boolean `runtime.optimize`. The saved
`program-run` runner forwards that value to every `Program.run`, including its
initial equivalence check, and retains native inspection. An unrelated
operation cannot silently accept this option. `program_eager` still compares a
different public boundary and is not optimizer A/B.

Example board configuration (make separate immutable false/true files):

```json
{"id":"graph-optimizer","arm":"all-on","mode":"identical","compile_defines":[],"runtime":{"operation":"program_optimize","optimize":true}}
```

`tools/bench_graph_optimizer.py` provides the paired workflow. It consumes the
same retained native Program checkpoint and complete input bytes on both arms,
passes the aggregate boolean directly, checks the five named native pass
reports, and records the actual active passes. `--require-pass NAME` refuses a
workload that does not activate a required pass. It does not add redundant
operations to make a baseline slower.

## Existing complete control

The immutable `program-run` receipt from
`prepare_graph_optimizer_inputs.py` contains the entire saved synthetic
4096-by-256 input and the 256→512→16 MLP weights. The adapter captures the
existing fused-linear/ReLU/softmax graph without changing its expression. A
separate float64 dense-network oracle checks the complete output before saving
the reference. It is expected to activate zero passes and measures overhead,
not active-pass benefits.

```sh
python tools/bench_graph_optimizer.py prepare-control \
  --recipe /root/results-graph-opt/inputs/program-run/recipe.json \
  --out /root/results-graph-opt/workload/program-run
MOJOLEARN_VENDOR=cuda MOJOLEARN_NUMERIC_MODE=identical \
  /root/results-graph-opt/venv/bin/python tools/bench_graph_optimizer.py run \
  --vendor cuda \
  --workload /root/results-graph-opt/workload/program-run/workload.json \
  --binary-provenance /root/results-graph-opt/accepted-native-build.json \
  --out /root/results-graph-opt/nvidia-program-run
```

Copy the actual retained workload bytes to AMD and run with vendor `hip` using
that installed build's own provenance and a fresh output directory. The
provenance JSON requires `source_commit` and `binaries`, each containing the
actual native artifact `path` and `sha256`; loaded sequence/tokenizer hashes
must match. The installed-package, actual backend and identical-mode guards
are mandatory for timing. Neither the runner nor these instructions compile
or rent hardware. The orchestrator uses accepted matching native closures.

The default is one excluded warmup and one scored sample per arm. First calls
and native audit calls are also retained and explicitly not scored. A separate
noise assessment and all affected workloads are still necessary before any
promotion. Native audits and pass inspection run outside the performance
clock. Native execution includes input upload, graph planning/execution,
synchronization and host return. The repeated whole-operation clock also
includes consuming every output byte into SHA256. Cold restored inference
adds full host-input materialization, Session creation and checkpoint loading;
capture/fixture creation and package import are separately excluded. These
are inference clocks; this optimizer does not optimize fit/backward.

Source, immutable input/checkpoint bytes, actual loaded native binaries,
installed Python connector source and binary provenance are archived before
timing and uploaded using the existing content-hashed R2 evidence helper.
Outputs, raw times, complete failures and result records are retained and
uploaded afterwards. Transfer failure preserves local files and requires
retrying the transfer, not remeasuring. Rental logs/results must additionally
be archived to the Mac using the normal box archiver. No board/opponent cell
is rewritten by this own-only controller.

## Additional native Program workloads

The runner accepts a `mojolearn.graph-optimizer-workload.v1` JSON manifest:

* `name`, `model`, and `coverage` describe the actual model and full workload,
  including intrinsic dataset/corpus caps. They are reviewable declarations;
  array extent checks cannot prove that an external corpus was complete.
* `recipe` points to the full immutable input receipt; `inputs` lists the
  complete array names in the Program's public input order. Every receipt
  array, including labels, is hashed and checked. There is no row slicing or
  per-batch loop in this runner.
* `program` points to the native checkpoint. `expected` is an NPZ containing
  all `output_0`, `output_1`, … reference arrays. Whole output shapes, dtypes
  and bits are checked on cold, audit, warmup and every scored invocation.
* `files` maps all relative artifact paths to `bytes` and `sha256`. Include
  original model source/configuration, captured state and independent-oracle
  evidence as additional files; the runner archives them with the workload.
* Optional `quality` supports complete softmax accuracy/NLL with a `labels`
  receipt-array name. Other tasks retain exact complete output comparison and
  explicitly report that no task metric was declared; they do not inherit
  unrelated classification metrics.

`compare` checks full recorded outputs across reports, failed/incomplete arms,
requested/native toggle readback, pass inventories and native runtime audits.
It does not decide a promotion or claim all vendors were supplied.

## Coverage and evidence

The original CPU controller test run passed 34/34 tests, no skips, combining
new mutation/propagation/timer controls and existing neural worker protocol
tests. Two additional binary-provenance controls cover matching and changed
loaded bytes, and both passed. Five checks on the actual full-workload report
matched its retained native build, accepted an unchanged comparison and
rejected a changed tail-output digest, a missing sample and wrong native
toggle readback. Complete logs and JUnit are under
`~/mojolearn-evidence/finish-graph-20261020/optimizer-harness-*`.

The complete saved MLP control ran against the immutable stage54 CPU runtime
(reused stage51 sequence closure), with cold/repeated baseline and candidate
outputs, both native audits, and all five zero rewrite counts passing. The
complete 4096-by-16 float32 output hash was
`da6c27db289f043818493aa3cb7eda5c566bb1f58440cdba3e905a223cd4bfa2`.
The independent float64 oracle passed. Evidence is in
`optimizer-full-control/{workload.json,expected.npz,program.mjl}` and
`optimizer-full-control-cpu/{result.json,retained/}` under the same directory.
This was `--validation-only`, not a CPU performance race or GPU qualification.

No retained real full model currently demonstrates active reshape,
permutation, pruning, broadcast/ReLU or duplicate-elimination coverage.
Transformer attention layout now has a source-only capture adapter described
in [the native capture handoff](GRAPH_TRANSFORMER_CAPTURE_20261020.md), but its
RMSNorm/rotary/projection additions and complete-block equivalence remain
uncompiled and unverified. Sequential's native model-fit
ABI is also distinct from OpsProgram replay. Substituting either specialized
runner, or constructing artificial redundant nodes, would not fill the gap.
Actual source-derived captures, their complete datasets/checkpoints and
equivalence evidence remain required. NVIDIA/AMD timings, cross-vendor output
identity, native ownership qualification, meaningful active-pass workloads,
noise evidence and any default decision remain separate pending work.
