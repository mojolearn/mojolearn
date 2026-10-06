# Neural IDENTICAL GEMM experiment source handoff

Branch: `ideas/neural-identical-ab-20261006-r3`. Fork baseline:
`main` at `fd6cf80453a6f18eb02e81566c824e7da106ccf0`. Changes remain
uncommitted. No build, test, checker, lint, identity, quality, benchmark or
candidate execution was run, as explicitly requested. There are no validation
logs or admitted performance results for these new sources.

Keep logs out of context: save complete output to files, use targeted rg/grep
with bounded surrounding lines and short tails, and summarize exit status,
coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
never hide failures or infer full success from filtered output.

## Actual scope

Six new Mojo modules provide explicit neural component APIs. No public model
imports them and no classical or existing default path changes. They are source
drafts, not compiled implementations or completed full-workload experiments.
Every new candidate requires `MOJOLEARN_NUMERIC_IDENTICAL`, its named
`MOJOLEARN_IDN_NEURAL_NNxx` define and an explicit candidate parameter.
`MOJOLEARN_IDN_ALL_OFF` disables all new candidate switches. Low-level arithmetic
helpers are internal mechanisms; the `_ab` surfaces apply the switch policy.

The authoritative per-card switches, entry points, remaining work and validation
states are in [gemm.json](gemm.json). The root plan-only selector is
`tools/neural_identical_ab.py`.

| Cards | Source and implemented mechanism | A/B selection |
| --- | --- | --- |
| NN01 | `gemm/experiments/neural_plans.mojo`: adapter to existing actual step geometries with returned route attribution | `neural_schedule_ab[CANDIDATE]`; candidate requires `MOJOLEARN_GEMM_ARM_TRIAL`; B shipped dispatch |
| NN02 | `neural_streaming.mojo`: bounded leaf groups plus persistent adjacent-pair tree stack, versus full leaf-plane materialization | `neural_streaming_ab[CANDIDATE,GROUP]` |
| NN03 | `neural_profile.mojo`: actual device kernel and host mirror with 64/128/256 logical leaves | `neural_profile_ab_{device,host}[LEAF,1,CANDIDATE]` |
| NN04 | `neural_profile.mojo`: two/four independent interleaved scalar FMA chains, fixed merge, common host/device body | `neural_profile_ab_{device,host}[128,CHAINS,CANDIDATE]` |
| NN05 | `neural_grouped.mojo`: existing grouped-grid arm and new shared-left load across 1–4 independent products | `neural_grouped_ab[JOBS,CANDIDATE,SHARE_LEFT]`; B independent launches |
| NN06 | `neural_epilogue.mojo`: fuse rounded product/bias/scale/residual/ReLU; retain ReLU preactivation and host/device derivative body | `neural_epilogue_ab[KIND,CANDIDATE]`; B one kernel per enabled stage |
| NN07 | `neural_grouped.mojo`: bounded immutable logical-view packing keyed by owner, mutation generation and strides | `NeuralOperandStage.prepare[CANDIDATE]`; B repacks |
| NN08 | `neural_plans.mojo`: explicit reuse of existing supported NVIDIA asynchronous copy pipeline | `neural_async_ab[CANDIDATE]`; unsupported candidate vendors fail explicitly |
| NN09 | `neural_tiled.mojo`: shared 8x16 output tiles with one/two pages, padding and invertible XOR layout | `neural_tiled_ab[CANDIDATE,DEPTH,PAD,SWIZZLE,False]` |
| NN10 | `neural_plans.mojo`: traffic/padded-work/device-fill estimate chooses existing flat/32/64/128 plans | `NeuralGemmPlan(...,cost_candidate=True)`; B `choose_gemm_plan` |
| NN11 | `neural_streaming.mojo`: logical-tree-height register capacity and exact global state-plane extent | `neural_fold_capacity_ab[CANDIDATE]`; independent `SPECIALIZE` in bounded streaming |
| NN12 | `neural_plans.mojo`: model-owned plan plus bounded retained scratch and oversized-call temporary route | `NeuralPlanWorkspace.run[CANDIDATE]`; B re-plan/allocate/synchronize per call |
| NN13 | Existing `sequence/gemm_tiled.mojo`, `sequence/ops.mojo`, `sequence/exec_device.mojo` reused per state/CNN owner | Existing `MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF` supplies B; no new source in this lane |
| NN14 | No bounded im2col producer added in this lane; state/CNN owner handles supplemental source | Pending here; consult state/CNN lane manifest |
| NN15 | `neural_tiled.mojo`: transpose physical output-thread ownership while keeping scalar product membership | `neural_tiled_ab[True,1,0,False,True]` with NN15 define |
| NN16 | `neural_streaming.mojo`: omit redundant scratch clear only where every read has an owned prior producer | `ELIDE_CLEAR=True` and NN16 define versus explicit clear control |

## Numerical contract and limitations

NN03 and NN04 intentionally permit new-version bits. Each selected profile uses
the same `neural_leaf` and `neural_cell` body in the host mirror and device
kernel, `rtf_mul_add`, explicit FTZ reads/results, fixed positive-zero chain
initialization, and an adjacent-pair fold with unchanged odd carries. Profile
partitioning depends only on k and declared constants. The 2/4-chain profile
explicitly merges unused positive-zero chains in ragged leaves. All numerical
profile consumers and host bindings must migrate together before real model
use. Sharing a source body is a contract implementation, not identity evidence.

Schedule candidates retain the incumbent `CONTRACT_K_LEAF_MIN` profile. NN09
staging tails never perform fake zero products, and all block lanes participate
in uniform barriers. NN07 copies raw words; only the consuming contraction
applies FTZ. NN06 uses pinned multiplication at the scale seam and separately
rounded additions. ReLU zero-derivative behavior must match an actual caller
before integration; no other activation has been implemented here.

NN10 is a whole-dispatch component draft. Isolating each implicated legacy rule
into its own removal experiment remains pending. `target_blocks` must come from
a recorded, supported device profile, not a guessed hardware property. Neighbor
shapes and one non-board dataset remain mandatory. NN11's second new variant
uses exact **global** state storage; a shared-memory stack variant is still
pending. NN16's clear is an explicit component B, not evidence that any public
model currently pays for a redundant clear.

Workspace and operand-stage objects belong to one model and one in-order
context. Callers must keep operands and owned buffers alive until completion,
use unique allocation owner tokens, increment mutation/settings generations,
and close on success or failure. Public lifecycle integration, partial-alias
guards and error-path qualification remain pending. No global cache is added.

## Remaining work

Map each candidate to exact saved full-workload datasets, hashes, dimensions,
estimator settings, mode, toggles and timed boundaries. Wire real neural callers
and complete host/profile migration where arithmetic changes. New code has no
accepted compile evidence, and therefore cannot yet use the repository's
already-compiled experimental merge exception.

Only after the user authorizes verification: compile frozen source, preserve
four-column identity/quality evidence, and measure full-dataset end-to-end A/B
on NVIDIA and AMD with the required warmup/sample policy. Retain cold/repeated
use separately, measure interactions, keep rejected candidates visible, and
update boards only through board tools. Existing I03/I04/I05/N03 component
winners and losers stay historical component evidence; none qualifies these
new neural experiments or changes a default.
