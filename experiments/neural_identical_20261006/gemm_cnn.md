# GEMM/CNN source-only handoff

Branch: `ideas/neural-identical-20261006-v2` in
`/Users/andrewhendel/CascadeProjects/mojolearn-neural-identical-ideas-v2`.
Base: `fd6cf80453a6f18eb02e81566c824e7da106ccf0`. Changes remain uncommitted;
no merge or push was performed.

All 18 assigned cards have explicit entries in `gemm_cnn.json`: 14 have newly
wired candidate controls, three reuse existing A/B implementations, and one
has a source-level structural rejection. These are source statuses, not successful
compilation, identity, quality, or performance results. NI04 is an existing
default versus its scalar rollback, not a newly implemented optimization.

The GEMM changes provide bounded streaming of exact leaf partials with a
persisted canonical fold stack, geometric scratch capacity growth that respects
the retained budget, an IDENTICAL-only one-page staging selector, and a shared
host/GPU leaf256 numerical version. The existing leaf64 and compact-tile arms
now respect `MOJOLEARN_IDN_ALL_OFF`. Training-forward workspace ownership is
wired by the transformer/training lane and is recorded here as NI01.

CNN changes provide budgeted saved columns/preactivation tapes versus native
recompute; 64-tap direct convolution; exact implicit-column convolution using
small shared operand tiles; a bounded-address col2im subarm; canonical bias
leaves without a ones buffer; no-pool convolution/ReLU forward and backward
fusion; a shared host/GPU normalization fold version; and fused device epoch
permutation with paired data/label gather. The Python change only plans native
buffer sizes and handles. New runtime numerical work is Mojo.

NI07 now has a real separate-buffer Q/K/V triplet in transformer attention,
including unequal query/KV widths, canonical tiled arithmetic and explicit
fallback for the caller's int15/sabotage routes. NI09 now fuses the shared
rounded bias seam into tiled linear GEMM output, reaching public CNN linear
calls and classifier heads; aliasing retains the separate-product route.
NI14 stages bounded shared tap pages to coalesce gradient reads while each
pixel keeps its exact kh/kw contribution order. Page-footprint and row-boundary
fallbacks use the existing bounded-address implementation.

NI13 is `rejected_source` for the incumbent path: CNN convolution and linear
forward already consume their resident weight layouts directly via OP_NT or
direct/implicit kernels. There is no per-call repack for a cache to eliminate.
`x_cnn/neural_weight_cache.mojo` remains a parked, real byte-preserving packing
helper for a separately justified future packed-layout consumer, with no
public integration or measured performance claim. This source-level rejection
is a decision, not unfinished work to add an unnecessary cache.

Changed paths owned by this lane:

- `gemm/checks/gemm_identical.mojo`
- `gemm/contract.mojo`
- `gemm/experiments/bounded_workspace.mojo`
- `gemm/experiments/neural_tiled.mojo`
- `x_cnn/ops.mojo` (conv/normalization/epoch sections)
- `x_cnn/device.mojo` (conv/bias/epoch sections)
- `x_cnn/neural_weight_cache.mojo`
- `x_cnn/neural_col2im.mojo`
- `bindings/_mojolearn_x_cnn.mojo`
- `bindings/_mojolearn_x_cnn_host.mojo`
- `python/mojolearn/_expansion_cnn.py` (buffer allocation metadata only)
- `experiments/neural_identical_20261006/gemm_cnn.json`
- `experiments/neural_identical_20261006/gemm_cnn.md`
- `tools/identical_wave_native_build.py` (bounded integration delegated by root)

The existing frozen native builder now accepts repeatable `--neural-idea`
and `--neural-variant` selections with `--recipe-role candidate|baseline`.
It reads lane records and the integration map using `git show` at `--sha`,
then uses the common source-plan/variant helpers only after both helper files
byte-match that commit. Required caller builders, explicit controls, runtime
environment, both arm source plans, and source-file hashes go into the normal
receipt. Manual `--define`, legacy recipes, and `--arm off` cannot be mixed
with neural selections. Existing legacy build behavior is retained in source.
The builder and helper code have not been executed, including plan-only mode,
imports, syntax checks, or other verification.

Root owns subsequent graph/dropout patches in the shared CNN source files.
The training lane owns `training/neural_gemm_workspace.mojo` and its callers.
It also owns NI07's integration in `transformer/impl/llama/modeling_llama.mojo`.
Root owns NI08 exact profile reporting in the linalg bindings/API shell;
the ledger records its explicit candidate/reference profile environment.

The caller-integration pass also closes NI02's CNN bypass: eligible streaming
products now enter the shared dispatcher before CNN's forced TN/Apple plans.
The caller and dispatcher share one resource predicate, so a forced plan
cannot accidentally consume the smaller streaming workspace. The reference
arm retains its existing plan choices.

Both CNN bindings now export `x_cnn_numerical_profile`. It records GEMM,
NI17 normalization, NI57 graph and the existing fold versions using one shared
Mojo definition. `_Layer` retains that string as `numerical_profile_` and
refuses a later mismatched binding. This is API-shell state provenance only;
unrecorded older models and external formats that omit the field still need
explicit migration policy. It does not establish identity or quality.

Every lane variant now has an explicit status and A/B controls. Each card
records source-declared `required_bindings` and `build_targets`, including GPU
and host scripts and the common `MOJOLEARN_MOJO_BUILD_FLAGS` define transport.
These declarations are not an executed build plan or an exhaustive full-workload
mapping. NI09/NI15 also affect graph-neural linear layers sharing the CNN native
binding and must include those models in their future workload coverage.

Validation: **NOT RUN, as explicitly requested**. No compilation, execution,
test, lint, syntax check, manifest validator, identity check, quality evaluation
or timing was run. The ledger was edited as metadata; no manifest validator or
candidate import was executed. No new build/test/GPU logs exist. The ledger and source are
the retained artifacts; old experiment evidence paths in existing comments
are historical, not evidence for this changed source.

Remaining work includes broader optional subarms/public caller coverage,
numerical-version/checkpoint coverage for external state formats, complete full
workload recipes and every qualification gate. Existing suspicious dispatch
rules remain recorded for a separate A/B removal with neighboring shapes and
a non-board full workload. Within-version NVIDIA/AMD/Apple/host identity is
required; cross-version output bits may change. No quality or speed gain has
been established and no default was promoted.
