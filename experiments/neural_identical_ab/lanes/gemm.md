Dense-profile scope clarification: NN03/NN04 change dense projections and their
backward products. Mamba M2/M3 SSD/SISO internal contractions retain their
separately declared fixed 128-term graph in every column; their host oracles
keep incumbent helpers. Selecting the dense profile must not change those
host-only helper calls. Changing SSD/SISO internals is an optional separate
versioned experiment, not unfinished wiring of the selected dense profile.

# Neural IDENTICAL GEMM experiment source handoff

Branch `ideas/neural-identical-ab-20261006-r3`, from main
`fd6cf80453a6f18eb02e81566c824e7da106ccf0`.

`gemm.neural_dispatch` now supplies the model-facing GEMM and workspace API;
`gemm.neural_backward` routes all linear gradients through that same selection.
`gemm.host.neural_gemm` supplies host List/pointer and logically zero-padded
equivalents. `neural_profile.mojo` is pure host/device arithmetic with no GPU
imports; launchers moved to `neural_profile_device.mojo`. NN03/NN04 select
`MOJOLEARN_IDN_NEURAL_LEAF` and `MOJOLEARN_IDN_NEURAL_CHAINS` for all routed
forward/backward products, including packed byte-LM host kernels. Microbatch
admission now reads the selected leaf partition.

Real callers include training device tensors, native SmallMLP steps, byte-LM
training/head calls, transformer forward/backward, Mamba projections and CNN
GEMMs (the latter family edits are owned by the other lanes). SmallMLP's Python
`_matmul` now calls the training module's new `neural_gemm` boundary in IDENTICAL
mode for NN/NT/TN; it does no data arithmetic. GPU and CPU binding modules are
`bindings/neural_gemm_boundary.mojo` and `neural_gemm_boundary_host.mojo`; root
registered both under `neural_gemm`.

NN05/NN07 reach the transformer's actual gate/up projections through
`GemmWorkspace.run_pair`. The pair preserves independent outputs and chains;
common-input staging lives only for that pair, so no mutation generation can
silently stale across steps. NN06 fuses the native SmallMLP's own `_add` and
ReLU conventions into its projection producers; the existing activated-output
backward convention remains intact. NN12 retains scratch by model/context or
training-output handle, with explicit bounded retention and closing. NN11 now
also has a shared-memory stack arm in addition to register and global-state
capacity arms.

NN16's real model arm removes the existing zero fill of byte-LM `head_ws`
while preserving its exact selected workspace size. Its B is the existing
`_zeros` call, not the diagnostic component's synthetic scratch clear. Root
authored this small caller edit in `training/byte_lm_logits.mojo`.

The common schedule control is `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL`; pair
controls use `MOJOLEARN_IDN_NEURAL_PAIR_CONTROL`. Numeric-profile B omits its
NN03/NN04 switch. All switches remain off by default and honor ALL_OFF. One
GEMM scheduling family is selected per A/B; NN01's old instruction geometries
explicitly reject changed numerical profiles. NN08's supported NVIDIA async
schedule and other vendors' synchronous schedule share the declared arithmetic;
unsupported native AMD/Apple async variants remain a Modular capability ask.
NN10 requires an explicit recorded device-fill parameter. Other unchosen
epilogues, persistent transpose caches and individual extra schedule variants
are optional distinct experiments, not unfinished primary caller wiring.

NN13 reuses the sequence lane's existing same-chain tiled executor. NN14's
bounded convolution programming and its final source status belong to the
state/CNN lane. No compilation, static checks, tests, identity, quality or timing
was run during either programming stage. Source integration is not numerical
or performance evidence. Root owns the requested commit/push.


Dense-profile scope: NN03/NN04 change dense projections and their backward
products. Mamba M2/M3 SSD/SISO internal contractions retain their independently
declared fixed 128-term graph in every column. Their host helpers keep that
graph as well; changing those contractions is a separate arithmetic experiment.

Exact per-card paths, flags, controls and source status are in
[gemm.json](gemm.json). New source has no compilation, identity, quality or
performance evidence; no new default was promoted.

> Keep logs out of context: save complete output to files, use targeted rg/grep
> with bounded surrounding lines and short tails, and summarize exit status,
> coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
> never hide failures or infer full success from filtered output.

## Source integration continuation, 2026-10-06

The file-by-file A/B inventory is `gemm_integration_inventory.json` (top-level
`inventory`, separate from the planner's `experiments` handoff collection).
It records each selected arm, exact defines, caller paths, workspace policy,
legacy experiment relationships and additional placement limits. NN03 now
records leaf64 versus leaf128 with the selected profile enabled in both arms;
NN04 similarly compares chains2 versus chains1. These isolate the arithmetic
choice from absent-flag dispatcher changes. NN10 requires an explicit positive
recorded fill budget in both configurations.

Source fixes in this continuation share direct/cached GEMM admission, reject
NN11 schedule combinations that would silently skip its arm (NN02+NN11 global
fold slots remain supported), and make NN12 owned-workspace B actually allocate
and wait per operation. NN12's chosen scope is GemmWorkspace and DevPool owners;
Mamba/CNN/ByteLM-head manual caches retain their separate policies. NN59 now
explicitly FTZ-normalizes probability before comparing a Philox draw.

No source change or inventory entry is execution evidence. No compiler,
checker, test, identity, quality or timing run was performed.
