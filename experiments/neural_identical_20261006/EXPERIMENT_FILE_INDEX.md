# Neural experiment file index

This index lists all 60 neural A/B ideas, every variant recorded in their lane ledgers, and the existing neural experiments found in the older runners and recipe files. It identifies the kernel, caller, binding and orchestration files needed to locate each experiment. Trees and classical estimators are outside the scope.

Source snapshot: `e1a885b9b` on `ideas/neural-identical-20261006-v2`, including integration commit `e7386c1e0`; the branch started from `main` at `fd6cf80453a6f18eb02e81566c824e7da106ccf0`. New candidates are disabled by default. Compilation, model verification and measurements remain unperformed for this source delivery. Historical `build_passed` labels below describe the older manifests only.

Jump to the [60-card index](#new-and-reused-catalog-experiments),
[reused experiments](#existing-experiments-reused-by-the-new-catalog),
[legacy neural runner](#existing-neural-stage-runner),
[GEMM arms](#existing-gemm-runtime-arms),
[frozen recipes](#existing-frozen-candidate-recipes),
[older IDENTICAL manifests](#existing-identical-manifests-for-neural-and-shared-gemm),
[state and rollback harnesses](#existing-state-and-rollback-a-b-harnesses),
[FAST references](#existing-fast-neural-references), or
the [reverse source file lookup](#source-file-lookup-for-catalog-ideas).

## Reading the arms

For NI cards, **A is the candidate and B is the per-card reference**. An empty define list means omit the controls; a presence-tested define set to `0` may still enable its path. B is not the blanket `MOJOLEARN_IDN_ALL_OFF` configuration. Runtime environments must be clean of unrelated experiment settings.

**S** denotes intended unchanged arithmetic; **V** denotes a new arithmetic version. Within each version, host, NVIDIA, AMD and Apple bits must match. A and B may differ for V experiments. No status in this index proves speed or quality. Full affected workloads and relevant combinations still need NVIDIA/AMD acceptance and host/Apple identity evidence.

The 60 parent cards comprise 41 new wired candidates, 15 reused existing candidates, two partial parent ideas and two source-based rejections. NI38 and NI49 each also expose a separately wired new variant. Variants are not additional parent cards. NI13 and NI22 do not describe selectable new optimizations.

## Catalog and orchestration files

| File | Purpose |
| --- | --- |
| [docs/plans/NEURAL_IDENTICAL_EXPERIMENT_IDEAS_2026-10-06.md](../../docs/plans/NEURAL_IDENTICAL_EXPERIMENT_IDEAS_2026-10-06.md) | Full hypotheses, numerical contracts and required interaction experiments |
| [experiments/neural_identical_20261006/IMPLEMENTATION_STATUS.md](../../experiments/neural_identical_20261006/IMPLEMENTATION_STATUS.md) | Source status, incomplete designs and qualification limits |
| [experiments/neural_identical_20261006/integration.json](../../experiments/neural_identical_20261006/integration.json) | All idea IDs to affected binding families and full workload groups |
| [experiments/neural_identical_20261006/workload_requirements.json](../../experiments/neural_identical_20261006/workload_requirements.json) | Full dataset, provenance, quality and timing requirements |
| [tools/neural_identical_ideas.py](../../tools/neural_identical_ideas.py) | Shared list, show, plan, build-plan and queue-template interface |
| [tools/neural_identical_integration.py](../../tools/neural_identical_integration.py) | Frozen-builder plans and neural full-operation queue integration |
| [tools/performance_ideas.py](../../tools/performance_ideas.py) | Existing entrypoint: neural subcommand forwards to the common catalog |
| [tools/neural_experiments.py](../../tools/neural_experiments.py) | Existing entrypoint: ideas subcommand forwards to the common catalog; older stage matrix remains separate |
| [tools/identical_wave_native_build.py](../../tools/identical_wave_native_build.py) | Frozen native build selection via --neural-idea, --neural-variant and --recipe-role |
| [tools/performance_full_ab_queue.py](../../tools/performance_full_ab_queue.py) | Full-operation A/B jobs with frozen controls, artifacts and quality evidence |

Commands below are selection examples, not executed experiments:

```sh
python3 tools/performance_ideas.py neural show NI20
python3 tools/neural_experiments.py ideas plan NI20 --arm A
python3 tools/neural_identical_ideas.py plan NI20 --arm B
python3 tools/neural_identical_ideas.py plan NI38 --arm A --variant NI38=state_window
python3 tools/neural_identical_ideas.py plan NI49 --arm A --variant NI49=row_serial_scan
```

Compiler controls reach binding scripts through `MOJOLEARN_MOJO_BUILD_FLAGS`. GPU families use `bindings/build_<family>.sh`; host companions use `bindings/build_<family>_host.sh`. CPU-only `neural_host` uses `bindings/build_neural_host.sh`. The per-card integration map and lane `build_targets` give the affected set. Build plans and queue templates do not run these scripts.

## New and reused catalog experiments

| ID | Experiment | Origin and state | Arithmetic |
| --- | --- | --- | --- |
| [NI01](#ni01) | Neural GEMM scratch lifetime | New wired candidate, unverified | S |
| [NI02](#ni02) | Bounded streaming of GEMM partial planes | New wired candidate, unverified | S |
| [NI03](#ni03) | GEMM launch plans from resource costs | Existing candidate reused, unverified here | S |
| [NI04](#ni04) | Smaller exact AMD matrix tiles | Existing candidate reused, unverified here | S |
| [NI05](#ni05) | NVIDIA register tiles without partial workspace | Existing candidate reused, unverified here | S |
| [NI06](#ni06) | GEMM operand staging and page depth | New wired candidate, unverified | S |
| [NI07](#ni07) | Batched independent neural projections | New wired candidate, unverified | S |
| [NI08](#ni08) | Versioned GEMM leaf lengths and balanced folds | New wired candidate, unverified | V |
| [NI09](#ni09) | Epilogues adjacent to canonical GEMM | New wired candidate, unverified | S |
| [NI10](#ni10) | Retain convolution columns for backward | New wired candidate, unverified | S |
| [NI11](#ni11) | Direct exact convolution for bounded reductions | New wired candidate, unverified | S |
| [NI12](#ni12) | Implicit convolution with canonical tiles | New wired candidate, unverified | S |
| [NI13](#ni13) | Cache convolution weight layouts by generation | Rejected from source findings | S |
| [NI14](#ni14) | Convolution backward gather tiling | New wired candidate, unverified | S |
| [NI15](#ni15) | Dedicated canonical bias and parameter folds | New wired candidate, unverified | S |
| [NI16](#ni16) | Fuse convolution activation and residual passes | New wired candidate, unverified | S |
| [NI17](#ni17) | CNN pooling and normalization canonical reductions | New wired candidate, unverified | V |
| [NI18](#ni18) | CNN device epoch preparation | New wired candidate, unverified | S |
| [NI19](#ni19) | Reuse attention query key and value tiles | Existing candidate reused, unverified here | S |
| [NI20](#ni20) | Versioned online attention softmax | New wired candidate, unverified | V |
| [NI21](#ni21) | Attention save versus recompute by memory budget | Existing candidate reused, unverified here | S |
| [NI22](#ni22) | Sparse causal tile scheduling | Rejected from source findings | S |
| [NI23](#ni23) | Fuse RoPE and attention layout transforms | New wired candidate, unverified | S |
| [NI24](#ni24) | Training prefill without unused KV cache writes | New wired candidate, unverified | S |
| [NI25](#ni25) | Canonical parallel RMSNorm and LayerNorm | New wired candidate, unverified | S |
| [NI26](#ni26) | Fuse training SwiGLU and its saved outputs | New wired candidate, unverified | S |
| [NI27](#ni27) | Parameter and gradient arena views | New wired candidate, unverified | S |
| [NI28](#ni28) | Consolidate status collection transactionally | Existing candidate reused, unverified here | S |
| [NI29](#ni29) | Fuse optimizer update and state refusal scans | Existing candidate reused, unverified here | S |
| [NI30](#ni30) | Batched optimizers and canonical global clipping | Existing candidate reused, unverified here | S |
| [NI31](#ni31) | Stable segmented embedding gradients | Existing candidate reused, unverified here | S |
| [NI32](#ni32) | Reuse token validation and resident batches | New wired candidate, unverified | S |
| [NI33](#ni33) | Fused loss intermediates and logits gradient | New wired candidate, unverified | S |
| [NI34](#ni34) | Chunked LM head with canonical vocabulary fold | New wired candidate, unverified | V |
| [NI35](#ni35) | Versioned loss and gradient reduction trees | New wired candidate, unverified | V |
| [NI36](#ni36) | Live range based activation retention | New wired candidate, unverified | S |
| [NI37](#ni37) | Parallel causal depthwise convolution | Existing candidate reused, unverified here | S |
| [NI38](#ni38) | Versioned Mamba 1 affine chunk scan | Partial parent idea | S |
| [NI39](#ni39) | Cache Mamba exponentials and decays | New wired candidate, unverified | S |
| [NI40](#ni40) | Tile Mamba 2 SSD interactions | Existing candidate reused, unverified here | S |
| [NI41](#ni41) | Skip mathematically unused SSD triangle | Existing candidate reused, unverified here | S |
| [NI42](#ni42) | Tile Mamba 3 interactions on all supported GPUs | New wired candidate, unverified | S |
| [NI43](#ni43) | Linear work Mamba 3 angle gradient | New wired candidate, unverified | S |
| [NI44](#ni44) | Canonical Mamba parameter gradient folds | New wired candidate, unverified | V |
| [NI45](#ni45) | Mamba scratch arenas and generation keyed weights | Existing candidate reused, unverified here | S |
| [NI46](#ni46) | Fuse Mamba elementwise state preparation | New wired candidate, unverified | S |
| [NI47](#ni47) | State space chunk boundary contract | New wired candidate, unverified | S |
| [NI48](#ni48) | Reuse Samba forward stages in backward | New wired candidate, unverified | S |
| [NI49](#ni49) | Repair recurrent persistent scan before timing | Partial parent idea | S |
| [NI50](#ni50) | Tiled same chain neural sequence GEMM | Existing candidate reused, unverified here | S |
| [NI51](#ni51) | Versioned recurrent gradients and LayerNorm folds | New wired candidate, unverified | V |
| [NI52](#ni52) | Neural MLP device epoch shuffle and batches | New wired candidate, unverified | S |
| [NI53](#ni53) | MoE stable routing and grouped expert execution | New wired candidate, unverified | S |
| [NI54](#ni54) | Adafactor tiled and versioned factored statistics | New wired candidate, unverified | V |
| [NI55](#ni55) | Graph neural feature register tiles | New wired candidate, unverified | S |
| [NI56](#ni56) | Reuse graph normalization workspace | Existing candidate reused, unverified here | S |
| [NI57](#ni57) | Versioned graph neural edge reduction | New wired candidate, unverified | V |
| [NI58](#ni58) | GraphSAGE max feature tiling | New wired candidate, unverified | S |
| [NI59](#ni59) | One launch channel dropout | New wired candidate, unverified | S |
| [NI60](#ni60) | Tile dropout mask application | New wired candidate, unverified | S |

## GEMM and CNN

Machine-readable arm records: [experiments/neural_identical_20261006/gemm_cnn.json](../../experiments/neural_identical_20261006/gemm_cnn.json). Source handoff: [experiments/neural_identical_20261006/gemm_cnn.md](../../experiments/neural_identical_20261006/gemm_cnn.md).

### NI01

**Neural GEMM scratch lifetime**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI01_TRAINING_WORKSPACE=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)
- [gemm/experiments/bounded_workspace.mojo](../../gemm/experiments/bounded_workspace.mojo)
- [training/neural_gemm_workspace.mojo](../../training/neural_gemm_workspace.mojo)
- [training/dev_tensors.mojo](../../training/dev_tensors.mojo)

#### NI01 geometric_capacity

New wired candidate, unverified; arithmetic **S**. Select with `--variant NI01=geometric_capacity`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_NI01_TRAINING_WORKSPACE=1`; `MOJOLEARN_NI01_GEMM_GEOMETRIC_WORKSPACE=1`.
- **B compiler defines:** `MOJOLEARN_NI01_TRAINING_WORKSPACE=1`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo); [gemm/experiments/bounded_workspace.mojo](../../gemm/experiments/bounded_workspace.mojo); [training/neural_gemm_workspace.mojo](../../training/neural_gemm_workspace.mojo); [training/dev_tensors.mojo](../../training/dev_tensors.mojo)

### NI02

**Bounded streaming of GEMM partial planes**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI02_GEMM_STREAM_PARTIALS=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)

### NI03

**GEMM launch plans from resource costs**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)
- [checks/kernel_matrix_gemm.mojo](../../checks/kernel_matrix_gemm.mojo)

#### NI03 slack_two

Existing candidate reused, unverified here; arithmetic **S**. Select with `--variant NI03=slack_two`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_IDN_GEMM_GROUP_SLACK_2=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo); [checks/kernel_matrix_gemm.mojo](../../checks/kernel_matrix_gemm.mojo)

#### NI03 slack_eight

Existing candidate reused, unverified here; arithmetic **S**. Select with `--variant NI03=slack_eight`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_IDN_GEMM_GROUP_SLACK_8=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo); [checks/kernel_matrix_gemm.mojo](../../checks/kernel_matrix_gemm.mojo)

### NI04

**Smaller exact AMD matrix tiles**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_GEMM_MFMA16_OFF=1`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)

### NI05

**NVIDIA register tiles without partial workspace**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)

### NI06

**GEMM operand staging and page depth**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI06_GEMM_ONE_PAGE=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)
- [checks/kernel_matrix_gemm.mojo](../../checks/kernel_matrix_gemm.mojo)

### NI07

**Batched independent neural projections**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI07_GROUPED_PROJECTIONS=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [gemm/experiments/neural_tiled.mojo](../../gemm/experiments/neural_tiled.mojo)
- [gemm/experiments/grouped_jobs.mojo](../../gemm/experiments/grouped_jobs.mojo)
- [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo)

### NI08

**Versioned GEMM leaf lengths and balanced folds**

New wired candidate, unverified; arithmetic **V**.

- **A compiler defines:** `MOJOLEARN_NI08_GEMM_LEAF_256=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** `MOJOLEARN_EXPERIMENT_GEMM_PROFILE=mojolearn.identical.gemm.fp32.ni08-leaf256`.
- **B runtime environment:** `MOJOLEARN_EXPERIMENT_GEMM_PROFILE=mojolearn.identical.gemm.fp32.v1`.

**Implementation and caller files**

- [gemm/contract.mojo](../../gemm/contract.mojo)
- [gemm/host/gemm_oracle.mojo](../../gemm/host/gemm_oracle.mojo)
- [gemm/host/identical_gemm.mojo](../../gemm/host/identical_gemm.mojo)
- [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)
- [bindings/_mojolearn_linalg.mojo](../../bindings/_mojolearn_linalg.mojo)
- [bindings/_mojolearn_linalg_host.mojo](../../bindings/_mojolearn_linalg_host.mojo)
- [python/mojolearn/_linalg_impl.py](../../python/mojolearn/_linalg_impl.py)
- [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo)
- [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo)
- [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py)

#### NI08 existing_leaf64

Existing candidate reused, unverified here; arithmetic **V**. Select with `--variant NI08=existing_leaf64`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_IDN_GEMM_FOLD_LEAF_64=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** `MOJOLEARN_EXPERIMENT_GEMM_PROFILE=mojolearn.identical.gemm.fp32.i04-leaf64`.
- **B runtime environment:** `MOJOLEARN_EXPERIMENT_GEMM_PROFILE=mojolearn.identical.gemm.fp32.v1`.

**Files:** [gemm/contract.mojo](../../gemm/contract.mojo); [gemm/host/gemm_oracle.mojo](../../gemm/host/gemm_oracle.mojo); [gemm/host/identical_gemm.mojo](../../gemm/host/identical_gemm.mojo); [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo); [bindings/_mojolearn_linalg.mojo](../../bindings/_mojolearn_linalg.mojo); [bindings/_mojolearn_linalg_host.mojo](../../bindings/_mojolearn_linalg_host.mojo); [python/mojolearn/_linalg_impl.py](../../python/mojolearn/_linalg_impl.py); [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo); [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo); [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py)

Existing I04 component losses on AMD and NVIDIA retained; not re-decided or promoted here

### NI09

**Epilogues adjacent to canonical GEMM**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI09_TILED_BIAS=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [gemm/experiments/neural_tiled.mojo](../../gemm/experiments/neural_tiled.mojo)
- [gemm/experiments/rounded_epilogue.mojo](../../gemm/experiments/rounded_epilogue.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)
- [x_cnn/ops.mojo](../../x_cnn/ops.mojo)

### NI10

**Retain convolution columns for backward**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI10_BUDGETED_CNN_TAPE=1`.
- **B compiler defines:** `MOJOLEARN_NI10_RECOMPUTE_CNN_TAPE=1`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/ops.mojo](../../x_cnn/ops.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)
- [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo)
- [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo)
- [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo)
- [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py)

### NI11

**Direct exact convolution for bounded reductions**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI11_DIRECT_CONV_TAPS64=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/device.mojo](../../x_cnn/device.mojo)

### NI12

**Implicit convolution with canonical tiles**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI12_IMPLICIT_CONV=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/ops.mojo](../../x_cnn/ops.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)

### NI13

**Cache convolution weight layouts by generation**

Rejected from source findings; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI13_CNN_WEIGHT_GENERATION_CACHE=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/neural_weight_cache.mojo](../../x_cnn/neural_weight_cache.mojo)

No public integration is warranted for the incumbent repack-removal premise

If a future packed-layout consumer is separately justified, it needs an owning mutation-generation contract, byte budget, context release and comparison against current no-repack OP_NT including cold preparation

Parked helper has no compile, identity, quality or timing qualification

### NI14

**Convolution backward gather tiling**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI14_TILED_COL2IM=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/neural_col2im.mojo](../../x_cnn/neural_col2im.mojo)
- [x_cnn/ops.mojo](../../x_cnn/ops.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)

#### NI14 address_bounds_only

New wired candidate, unverified; arithmetic **S**. Select with `--variant NI14=address_bounds_only`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_NI14_BOUNDED_COL2IM=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [x_cnn/neural_col2im.mojo](../../x_cnn/neural_col2im.mojo); [x_cnn/ops.mojo](../../x_cnn/ops.mojo); [x_cnn/device.mojo](../../x_cnn/device.mojo)

### NI15

**Dedicated canonical bias and parameter folds**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI15_BIAS_NO_ONES=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/device.mojo](../../x_cnn/device.mojo)

### NI16

**Fuse convolution activation and residual passes**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI16_CONV_RELU_FUSED=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/ops.mojo](../../x_cnn/ops.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)

### NI17

**CNN pooling and normalization canonical reductions**

New wired candidate, unverified; arithmetic **V**.

- **A compiler defines:** `MOJOLEARN_NI17_BALANCED_NORM=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/ops.mojo](../../x_cnn/ops.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)
- [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo)
- [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo)
- [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo)
- [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py)

### NI18

**CNN device epoch preparation**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI18_FUSED_EPOCH_GATHER=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/ops.mojo](../../x_cnn/ops.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)

## Transformer training and embedding

Machine-readable arm records: [experiments/neural_identical_20261006/transformer_training.json](../../experiments/neural_identical_20261006/transformer_training.json). Source handoff: [experiments/neural_identical_20261006/transformer_training.md](../../experiments/neural_identical_20261006/transformer_training.md).

### NI19

**Reuse attention query key and value tiles**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`.
- **B runtime environment:** `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`.

**Implementation and caller files**

- [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo)

### NI20

**Versioned online attention softmax**

New wired candidate, unverified; arithmetic **V**.

- **A compiler defines:** `MOJOLEARN_IDN_ATTENTION_V2=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [transformer/impl/llama/attention_v2_model_contract.mojo](../../transformer/impl/llama/attention_v2_model_contract.mojo)
- [transformer/impl/llama/attention_v2_model_device.mojo](../../transformer/impl/llama/attention_v2_model_device.mojo)
- [transformer/impl/llama/attention_v2_model_host.mojo](../../transformer/impl/llama/attention_v2_model_host.mojo)
- [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo)
- [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo)
- [transformer/checks/transformer_oracle.mojo](../../transformer/checks/transformer_oracle.mojo)
- [transformer/checks/transformer_backward_oracle.mojo](../../transformer/checks/transformer_backward_oracle.mojo)
- [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo)
- [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo)
- [training/byte_lm_config.mojo](../../training/byte_lm_config.mojo)
- [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo)
- [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo)
- [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo)
- [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo)
- [transformer/ATTENTION_V2_CONTRACT.md](../../transformer/ATTENTION_V2_CONTRACT.md)

### NI21

**Attention save versus recompute by memory budget**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_ATTN_V1_PACKED_ESTASH=1`.
- **B compiler defines:** `MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD=1`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo)

### NI22

**Sparse causal tile scheduling**

Rejected from source findings; arithmetic **S**.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo)

Closed for the catalog hypothesis: no redundant skip arm was added. Reopen only after identifying actual computed masked work in a specific reachable route, while retaining complete backward and trace coverage.

NOT RUN: source rejection is not a measured performance result, compile claim or vendor identity result.

### NI23

**Fuse RoPE and attention layout transforms**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_ROPE_CACHE=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo)

### NI24

**Training prefill without unused KV cache writes**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_TRAIN_NO_DECODE_CACHE=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [training/byte_lm.mojo](../../training/byte_lm.mojo)
- [training/byte_lm_layer_pool.mojo](../../training/byte_lm_layer_pool.mojo)
- [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo)

### NI25

**Canonical parallel RMSNorm and LayerNorm**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_RMS_ROW_BLOCK=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo)
- [training/samba_ops.mojo](../../training/samba_ops.mojo)

### NI26

**Fuse training SwiGLU and its saved outputs**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_TRAIN_SWIGLU=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo)
- [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo)

### NI27

**Parameter and gradient arena views**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_LM_PARAM_VIEWS=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [training/byte_lm.mojo](../../training/byte_lm.mojo)
- [training/byte_lm_layer_pool.mojo](../../training/byte_lm_layer_pool.mojo)

### NI28

**Consolidate status collection transactionally**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_OPT_GATE_SCAN_OFF=1`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/opt_gate.mojo](../../training/opt_gate.mojo)
- [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo)

### NI29

**Fuse optimizer update and state refusal scans**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_TRAIN_LIVE_STATUS=1`; `MOJOLEARN_STEP_GLUE_TRIAL=1`.
- **B compiler defines:** `MOJOLEARN_STEP_GLUE_TRIAL=1`.
- **A runtime environment:** `MOJOLEARN_STEP_GLUE_ARM=noshadow`.
- **B runtime environment:** `MOJOLEARN_STEP_GLUE_ARM=noshadow`.

**Implementation and caller files**

- [training/byte_lm.mojo](../../training/byte_lm.mojo)
- [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo)

### NI30

**Batched optimizers and canonical global clipping**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_OPT_SGD_ONE_LAUNCH_OFF=1`; `MOJOLEARN_IDN_OPT_CLIP_BATCHED_OFF=1`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/opt_gate.mojo](../../training/opt_gate.mojo)
- [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo)

### NI31

**Stable segmented embedding gradients**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_EMB_RADIX_SORT_OFF=1`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [embedding/checks/embedding_sort.mojo](../../embedding/checks/embedding_sort.mojo)
- [embedding/checks/embedding_identical.mojo](../../embedding/checks/embedding_identical.mojo)
- [training/byte_lm.mojo](../../training/byte_lm.mojo)

### NI32

**Reuse token validation and resident batches**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_LM_OWNED_TOKENS=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [training/byte_lm.mojo](../../training/byte_lm.mojo)
- [embedding/checks/embedding_identical.mojo](../../embedding/checks/embedding_identical.mojo)

### NI33

**Fused loss intermediates and logits gradient**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_CE_GRAD_FUSED=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [training/checks/loss.mojo](../../training/checks/loss.mojo)

### NI34

**Chunked LM head with canonical vocabulary fold**

New wired candidate, unverified; arithmetic **V**.

- **A compiler defines:** `MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [training/chunked_lm_head_v2.mojo](../../training/chunked_lm_head_v2.mojo)
- [training/chunked_lm_head_host.mojo](../../training/chunked_lm_head_host.mojo)
- [training/byte_lm.mojo](../../training/byte_lm.mojo)
- [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo)
- [training/byte_lm_host_backward.mojo](../../training/byte_lm_host_backward.mojo)
- [training/byte_lm_config.mojo](../../training/byte_lm_config.mojo)
- [training/checks/chunked_lm_head_oracle.mojo](../../training/checks/chunked_lm_head_oracle.mojo)
- [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo)
- [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo)
- [python/mojolearn/_samba_impl.py](../../python/mojolearn/_samba_impl.py)
- [training/CHUNKED_LM_HEAD_V2.md](../../training/CHUNKED_LM_HEAD_V2.md)

#### NI34 existing-explicit-byte-config

Existing direct native configuration alternative. Configure these settings at the native caller; the generic selector does not apply `candidate_settings` or `baseline_settings` as a distinct runtime setting arm.

- A settings: `ByteConfig.chunked_lm_head_v2=true`.
- B settings: `ByteConfig.chunked_lm_head_v2=false`.

**Files:** [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo); [training/chunked_lm_head_v2.mojo](../../training/chunked_lm_head_v2.mojo); [training/chunked_lm_head_host.mojo](../../training/chunked_lm_head_host.mojo); [training/byte_lm.mojo](../../training/byte_lm.mojo); [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo); [training/byte_lm_host_backward.mojo](../../training/byte_lm_host_backward.mojo); [training/byte_lm_config.mojo](../../training/byte_lm_config.mojo); [training/checks/chunked_lm_head_oracle.mojo](../../training/checks/chunked_lm_head_oracle.mojo); [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo); [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo); [python/mojolearn/_samba_impl.py](../../python/mojolearn/_samba_impl.py); [training/CHUNKED_LM_HEAD_V2.md](../../training/CHUNKED_LM_HEAD_V2.md)

Existing direct native config alternative; primary A/B uses the guarded compile selector.

### NI35

**Versioned loss and gradient reduction trees**

New wired candidate, unverified; arithmetic **V**.

- **A compiler defines:** `MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [training/loss_reduction_v2.mojo](../../training/loss_reduction_v2.mojo)
- [training/checks/loss.mojo](../../training/checks/loss.mojo)
- [training/checks/loss_oracle.mojo](../../training/checks/loss_oracle.mojo)
- [training/loss_host_rows.mojo](../../training/loss_host_rows.mojo)
- [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo)
- [training/checks/loss_contract.mojo](../../training/checks/loss_contract.mojo)
- [training/byte_lm_config.mojo](../../training/byte_lm_config.mojo)
- [training/IDENTICAL_LOSS_CONTRACT.md](../../training/IDENTICAL_LOSS_CONTRACT.md)
- [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo)
- [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo)
- [python/mojolearn/_samba_impl.py](../../python/mojolearn/_samba_impl.py)

### NI36

**Live range based activation retention**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_SAMBA_FORWARD_TAPE=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo)
- [training/dev_tensors.mojo](../../training/dev_tensors.mojo)
- [training/byte_lm_layer_pool.mojo](../../training/byte_lm_layer_pool.mojo)
- [training/byte_lm.mojo](../../training/byte_lm.mojo)
- [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo)
- [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo)
- [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py)
- [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py)
- [python/mojolearn/_samba_impl.py](../../python/mojolearn/_samba_impl.py)

#### NI36 linear-backward-scratch-sharing

New wired candidate, unverified; arithmetic **S**. Select with `--variant NI36=linear-backward-scratch-sharing`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_IDN_TRAIN_BACKWARD_SCRATCH=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo); [training/dev_tensors.mojo](../../training/dev_tensors.mojo); [training/byte_lm_layer_pool.mojo](../../training/byte_lm_layer_pool.mojo); [training/byte_lm.mojo](../../training/byte_lm.mojo); [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo); [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo); [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py); [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py); [python/mojolearn/_samba_impl.py](../../python/mojolearn/_samba_impl.py)

Independent smaller arm; does not enable the primary saved-forward tape experiment.

## Mamba Samba and neural sequence

Machine-readable arm records: [experiments/neural_identical_20261006/sequence.json](../../experiments/neural_identical_20261006/sequence.json). Source handoff: [experiments/neural_identical_20261006/sequence.md](../../experiments/neural_identical_20261006/sequence.md).

### NI37

**Parallel causal depthwise convolution**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_MAMBA_CONV_CELL_OFF`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo)
- [mamba/impl/modeling/modeling_mamba.mojo](../../mamba/impl/modeling/modeling_mamba.mojo)
- [mamba/impl/modules/mamba2.mojo](../../mamba/impl/modules/mamba2.mojo)
- [mamba/host/gen/modeling_mamba.mojo](../../mamba/host/gen/modeling_mamba.mojo)
- [mamba/host/gen/mamba2.mojo](../../mamba/host/gen/mamba2.mojo)

### NI38

**Versioned Mamba 1 affine chunk scan**

Partial parent idea; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_M1_STATE_WINDOW`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/ops/identical_scan_window.mojo](../../mamba/impl/ops/identical_scan_window.mojo)
- [mamba/impl/ops/selective_scan_interface.mojo](../../mamba/impl/ops/selective_scan_interface.mojo)
- [mamba/impl/ops/selective_scan_backward.mojo](../../mamba/impl/ops/selective_scan_backward.mojo)
- [mamba/host/gen/selective_scan_interface.mojo](../../mamba/host/gen/selective_scan_interface.mojo)
- [tools/mamba_host_gen.py](../../tools/mamba_host_gen.py)
- [mamba/host/gen/device_optimizations.mojo](../../mamba/host/gen/device_optimizations.mojo)

The original V affine time-parallel scan remains pending: define an absolute-position augmented carry/checkpoint representation and matching forward/backward/prefill/decode/serialized-state contracts on host and all GPUs.

The implemented S alternative needs future full Mamba1/Samba forward/backward/decode quality and identity evidence, plus end-to-end timing including allocation, all window launches and final scratch drain.

Keep V and S experiment IDs/receipts distinguishable; never claim the original affine graph was implemented by the S alternative.

#### NI38 state_window

New wired candidate, unverified; arithmetic **S**. Select with `--variant NI38=state_window`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_IDN_M1_STATE_WINDOW`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [mamba/impl/ops/identical_scan_window.mojo](../../mamba/impl/ops/identical_scan_window.mojo); [mamba/impl/ops/selective_scan_interface.mojo](../../mamba/impl/ops/selective_scan_interface.mojo); [mamba/impl/ops/selective_scan_backward.mojo](../../mamba/impl/ops/selective_scan_backward.mojo); [mamba/host/gen/selective_scan_interface.mojo](../../mamba/host/gen/selective_scan_interface.mojo); [tools/mamba_host_gen.py](../../tools/mamba_host_gen.py); [mamba/host/gen/device_optimizations.mojo](../../mamba/host/gen/device_optimizations.mojo)

Implemented S subarm: selective_scan_fn dispatches to bounded state-parallel window kernels; host, carried state and backward checkpoint use the unchanged scalar graph. The original V affine proposal below remains pending.

#### NI38 original_versioned_affine_scan

Design pending; arithmetic **V**. Select with `--variant NI38=original_versioned_affine_scan`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** No implementation path recorded.

Not wired: absolute carry/checkpoint/serialization and derivative contracts still required.

### NI39

**Cache Mamba exponentials and decays**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_M2_YOFF_EXP_CACHE`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo)
- [mamba/host/gen/ssd_minimal.mojo](../../mamba/host/gen/ssd_minimal.mojo)

### NI40

**Tile Mamba 2 SSD interactions**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_M2_RETAIN_GL`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo)
- [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo)
- [mamba/impl/ops/mamba2_ssd_backward.mojo](../../mamba/impl/ops/mamba2_ssd_backward.mojo)
- [mamba/host/gen/ssd_minimal.mojo](../../mamba/host/gen/ssd_minimal.mojo)

#### NI40 existing_ssd_tiles

Existing candidate reused, unverified here; arithmetic **S**. Select with `--variant NI40=existing_ssd_tiles`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_M2_SSD_TILES_OFF`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo); [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo); [mamba/impl/ops/mamba2_ssd_backward.mojo](../../mamba/impl/ops/mamba2_ssd_backward.mojo); [mamba/host/gen/ssd_minimal.mojo](../../mamba/host/gen/ssd_minimal.mojo)

same-chain Ydiag/Cstate and paired backward tiling

### NI41

**Skip mathematically unused SSD triangle**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_M2_CB_LOWER_OFF`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo)
- [mamba/host/gen/ssd_minimal.mojo](../../mamba/host/gen/ssd_minimal.mojo)
- [mamba/impl/ops/mamba2_ssd_backward.mojo](../../mamba/impl/ops/mamba2_ssd_backward.mojo)

### NI42

**Tile Mamba 3 interactions on all supported GPUs**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_M3_YINTRA_RESOURCE_TILE`.
- **B compiler defines:** `MOJOLEARN_MAMBA3_LEGACY_YINTRA`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/ops/mamba3_siso.mojo](../../mamba/impl/ops/mamba3_siso.mojo)
- [mamba/host/gen/mamba3_siso.mojo](../../mamba/host/gen/mamba3_siso.mojo)

#### NI42 historical_routing_B

New wired candidate, unverified; arithmetic **S**. Select with `--variant NI42=historical_routing_B`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_IDN_M3_YINTRA_RESOURCE_TILE`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [mamba/impl/ops/mamba3_siso.mojo](../../mamba/impl/ops/mamba3_siso.mojo); [mamba/host/gen/mamba3_siso.mojo](../../mamba/host/gen/mamba3_siso.mojo)

required old-rule comparison, includes identical-route cells visibly

### NI43

**Linear work Mamba 3 angle gradient**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo)
- [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo)

#### NI43 existing_versioned_suffix

Existing candidate reused, unverified here; arithmetic **V**. Select with `--variant NI43=existing_versioned_suffix`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo); [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo)

existing canonical chunk suffix vs legacy serial suffix, unchanged defaults

### NI44

**Canonical Mamba parameter gradient folds**

New wired candidate, unverified; arithmetic **V**.

- **A compiler defines:** `MOJOLEARN_IDN_M2_GRAD_LEAF128`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/ops/mamba2_ssd_backward.mojo](../../mamba/impl/ops/mamba2_ssd_backward.mojo)
- [mamba/host/gen/mamba2_ssd_backward.mojo](../../mamba/host/gen/mamba2_ssd_backward.mojo)

### NI45

**Mamba scratch arenas and generation keyed weights**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_MAMBA_ARENA_OFF`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo)
- [mamba/impl/modules/afn_arena.mojo](../../mamba/impl/modules/afn_arena.mojo)
- [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo)
- [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo)

#### NI45 gemm_workspace

Existing candidate reused, unverified here; arithmetic **S**. Select with `--variant NI45=gemm_workspace`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo); [mamba/impl/modules/afn_arena.mojo](../../mamba/impl/modules/afn_arena.mojo); [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo); [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo)

Alternative explicitly recorded from source; unverified.

#### NI45 mamba3_session_stage_reuse

Existing candidate reused, unverified here; arithmetic **S**. Select with `--variant NI45=mamba3_session_stage_reuse`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_M3_SESSION_STAGE_REUSE_OFF`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo); [mamba/impl/modules/afn_arena.mojo](../../mamba/impl/modules/afn_arena.mojo); [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo); [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo)

Alternative explicitly recorded from source; unverified.

### NI46

**Fuse Mamba elementwise state preparation**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_MAMBA1_ELEMENTWISE_FUSED`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/modeling/modeling_mamba.mojo](../../mamba/impl/modeling/modeling_mamba.mojo)
- [mamba/impl/modeling/afn_mamba1_fused.mojo](../../mamba/impl/modeling/afn_mamba1_fused.mojo)
- [mamba/host/gen/modeling_mamba.mojo](../../mamba/host/gen/modeling_mamba.mojo)

### NI47

**State space chunk boundary contract**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [mamba/impl/modules/mamba3.mojo](../../mamba/impl/modules/mamba3.mojo)
- [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo)

#### NI47 versioned_absolute_boundary_schema

Design pending; arithmetic **V**. Select with `--variant NI47=versioned_absolute_boundary_schema`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [mamba/impl/modules/mamba3.mojo](../../mamba/impl/modules/mamba3.mojo); [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo)

Required if NI38 affine composition changes the state graph.

### NI48

**Reuse Samba forward stages in backward**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_SAMBA_FORWARD_TAPE`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo)
- [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo)
- [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py)
- [python/mojolearn/_samba_impl.py](../../python/mojolearn/_samba_impl.py)
- [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo)
- [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py)

### NI49

**Repair recurrent persistent scan before timing**

Partial parent idea; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [sequence/recurrent_scan.mojo](../../sequence/recurrent_scan.mojo)
- [sequence/recurrent.mojo](../../sequence/recurrent.mojo)
- [sequence/exec_device.mojo](../../sequence/exec_device.mojo)
- [sequence/dispatch.mojo](../../sequence/dispatch.mojo)

When authorized, diagnose the old cooperative SCAN with per-step state/gradient evidence; its original constant-prediction cause is still not established.

Qualify ROW_SERIAL_SCAN independently on full LSTM/GRU/RNN training and all host/GPU columns before timing or promotion.

Compare reduced launch overhead against reduced within-row parallelism; a scheduling alternative is not a measured performance win.

#### NI49 row_serial_scan

New wired candidate, unverified; arithmetic **S**. Select with `--variant NI49=row_serial_scan`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [sequence/recurrent_scan.mojo](../../sequence/recurrent_scan.mojo); [sequence/recurrent.mojo](../../sequence/recurrent.mojo); [sequence/exec_device.mojo](../../sequence/exec_device.mojo); [sequence/dispatch.mojo](../../sequence/dispatch.mojo)

Implemented independent repair candidate. Recurrent forward/backward select OP_CELL_*_SCAN, DeviceExec selects ordinary seq_kernel items, and shared op_cell_*_scan bodies own one entire row. No within-kernel inter-thread state dependency. B is the existing per-timestep model path with this define omitted.

#### NI49 original_failed_cooperative_scan

Partial parent idea; arithmetic **S**. Select with `--variant NI49=original_failed_cooperative_scan`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_IDN_SEQ_LSTM_SCAN`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [sequence/recurrent_scan.mojo](../../sequence/recurrent_scan.mojo); [sequence/recurrent.mojo](../../sequence/recurrent.mojo); [sequence/exec_device.mojo](../../sequence/exec_device.mojo); [sequence/dispatch.mojo](../../sequence/dispatch.mojo)

Original broken SCAN remains default OFF and not admitted; source repair/alternative does not erase recorded accuracy/r2 failure evidence.

### NI50

**Tiled same chain neural sequence GEMM**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [sequence/gemm_tiled.mojo](../../sequence/gemm_tiled.mojo)
- [sequence/exec_device.mojo](../../sequence/exec_device.mojo)
- [sequence/recurrent.mojo](../../sequence/recurrent.mojo)
- [sequence/mlp_fit.mojo](../../sequence/mlp_fit.mojo)

### NI51

**Versioned recurrent gradients and LayerNorm folds**

New wired candidate, unverified; arithmetic **V**.

- **A compiler defines:** `MOJOLEARN_IDN_SEQ_WGRAD_LEAF256`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [sequence/recurrent.mojo](../../sequence/recurrent.mojo)
- [sequence/layernorm.mojo](../../sequence/layernorm.mojo)
- [sequence/ops.mojo](../../sequence/ops.mojo)
- [sequence/exec.mojo](../../sequence/exec.mojo)
- [sequence/exec_device.mojo](../../sequence/exec_device.mojo)

#### NI51 layernorm_parameter_leaf32

New wired candidate, unverified; arithmetic **V**. Select with `--variant NI51=layernorm_parameter_leaf32`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** `MOJOLEARN_IDN_SEQ_LN_LEAF32`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [sequence/recurrent.mojo](../../sequence/recurrent.mojo); [sequence/layernorm.mojo](../../sequence/layernorm.mojo); [sequence/ops.mojo](../../sequence/ops.mojo); [sequence/exec.mojo](../../sequence/exec.mojo); [sequence/exec_device.mojo](../../sequence/exec_device.mojo)

Alternative explicitly recorded from source; unverified.

### NI52

**Neural MLP device epoch shuffle and batches**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_MLP_DEVICE_EPOCH_ORDER`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [sequence/mlp.mojo](../../sequence/mlp.mojo)
- [sequence/mlp_fit.mojo](../../sequence/mlp_fit.mojo)

#### NI52 existing_device_epochs

Existing candidate reused, unverified here; arithmetic **V**. Select with `--variant NI52=existing_device_epochs`. Fields omitted by the variant inherit the parent record.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_MLP_EPOCH_DEV_OFF`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Files:** [sequence/mlp.mojo](../../sequence/mlp.mojo); [sequence/mlp_fit.mojo](../../sequence/mlp_fit.mojo)

Alternative explicitly recorded from source; unverified.

### NI53

**MoE stable routing and grouped expert execution**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_IDN_MOE_STABLE_PACK`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [sequence/moe_group.mojo](../../sequence/moe_group.mojo)
- [sequence/exec_device.mojo](../../sequence/exec_device.mojo)
- [sequence/moe_reg.mojo](../../sequence/moe_reg.mojo)
- [sequence/moe_tiled.mojo](../../sequence/moe_tiled.mojo)
- [sequence/pyapi.mojo](../../sequence/pyapi.mojo)

### NI54

**Adafactor tiled and versioned factored statistics**

New wired candidate, unverified; arithmetic **V**.

- **A compiler defines:** `MOJOLEARN_IDN_AF_COL_CHUNK_FOLD`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [sequence/adafactor.mojo](../../sequence/adafactor.mojo)
- [sequence/adafactor_candidates.mojo](../../sequence/adafactor_candidates.mojo)
- [sequence/exec_device.mojo](../../sequence/exec_device.mojo)

## Graph neural and dropout

Machine-readable arm records: [experiments/neural_identical_20261006/neural_aux.json](../../experiments/neural_identical_20261006/neural_aux.json). Source handoff: [experiments/neural_identical_20261006/neural_aux.md](../../experiments/neural_identical_20261006/neural_aux.md).

### NI55

**Graph neural feature register tiles**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI55_GRAPH_FEATURE4=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/neural_aux_ops.mojo](../../x_cnn/neural_aux_ops.mojo)
- [x_cnn/neural_aux_contract.mojo](../../x_cnn/neural_aux_contract.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)

### NI56

**Reuse graph normalization workspace**

Existing candidate reused, unverified here; arithmetic **S**.

- **A compiler defines:** none (omit controls).
- **B compiler defines:** `MOJOLEARN_IDN_GRAPH_M_OFF=1`.
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/device.mojo](../../x_cnn/device.mojo)

### NI57

**Versioned graph neural edge reduction**

New wired candidate, unverified; arithmetic **V**.

- **A compiler defines:** `MOJOLEARN_NI57_GRAPH_TREE64=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/neural_aux_contract.mojo](../../x_cnn/neural_aux_contract.mojo)
- [x_cnn/ops.mojo](../../x_cnn/ops.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)
- [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo)

### NI58

**GraphSAGE max feature tiling**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI58_SAGE_FEATURE4=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/neural_aux_ops.mojo](../../x_cnn/neural_aux_ops.mojo)
- [x_cnn/neural_aux_contract.mojo](../../x_cnn/neural_aux_contract.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)

### NI59

**One launch channel dropout**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI59_DROPOUT_CHANNEL=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/neural_aux_ops.mojo](../../x_cnn/neural_aux_ops.mojo)
- [x_cnn/neural_aux_contract.mojo](../../x_cnn/neural_aux_contract.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)

### NI60

**Tile dropout mask application**

New wired candidate, unverified; arithmetic **S**.

- **A compiler defines:** `MOJOLEARN_NI60_DROPOUT_APPLY4=1`.
- **B compiler defines:** none (omit controls).
- **A runtime environment:** none (clean inherited experiment settings).
- **B runtime environment:** none (clean inherited experiment settings).

**Implementation and caller files**

- [x_cnn/neural_aux_ops.mojo](../../x_cnn/neural_aux_ops.mojo)
- [x_cnn/neural_aux_contract.mojo](../../x_cnn/neural_aux_contract.mojo)
- [x_cnn/device.mojo](../../x_cnn/device.mojo)

## Existing experiments reused by the new catalog

These are existing implementation paths, not 15 newly invented kernels. Their exact A/B controls and files are listed under the corresponding NI entries above. New integration repairs do not turn historical evidence into current qualification.

| Existing experiment | Catalog entry | Source files |
| --- | --- | --- |
| GEMM launch plans from resource costs | [NI03](#ni03) | [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo); [checks/kernel_matrix_gemm.mojo](../../checks/kernel_matrix_gemm.mojo) |
| Smaller exact AMD matrix tiles | [NI04](#ni04) | [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| NVIDIA register tiles without partial workspace | [NI05](#ni05) | [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| Reuse attention query key and value tiles | [NI19](#ni19) | [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo) |
| Attention save versus recompute by memory budget | [NI21](#ni21) | [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo) |
| Consolidate status collection transactionally | [NI28](#ni28) | [training/opt_gate.mojo](../../training/opt_gate.mojo); [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo) |
| Fuse optimizer update and state refusal scans | [NI29](#ni29) | [training/byte_lm.mojo](../../training/byte_lm.mojo); [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo) |
| Batched optimizers and canonical global clipping | [NI30](#ni30) | [training/opt_gate.mojo](../../training/opt_gate.mojo); [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo) |
| Stable segmented embedding gradients | [NI31](#ni31) | [embedding/checks/embedding_sort.mojo](../../embedding/checks/embedding_sort.mojo); [embedding/checks/embedding_identical.mojo](../../embedding/checks/embedding_identical.mojo); [training/byte_lm.mojo](../../training/byte_lm.mojo) |
| Parallel causal depthwise convolution | [NI37](#ni37) | [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo); [mamba/impl/modeling/modeling_mamba.mojo](../../mamba/impl/modeling/modeling_mamba.mojo); [mamba/impl/modules/mamba2.mojo](../../mamba/impl/modules/mamba2.mojo); [mamba/host/gen/modeling_mamba.mojo](../../mamba/host/gen/modeling_mamba.mojo); [mamba/host/gen/mamba2.mojo](../../mamba/host/gen/mamba2.mojo) |
| Tile Mamba 2 SSD interactions | [NI40](#ni40) | [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo); [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo); [mamba/impl/ops/mamba2_ssd_backward.mojo](../../mamba/impl/ops/mamba2_ssd_backward.mojo); [mamba/host/gen/ssd_minimal.mojo](../../mamba/host/gen/ssd_minimal.mojo) |
| Skip mathematically unused SSD triangle | [NI41](#ni41) | [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo); [mamba/host/gen/ssd_minimal.mojo](../../mamba/host/gen/ssd_minimal.mojo); [mamba/impl/ops/mamba2_ssd_backward.mojo](../../mamba/impl/ops/mamba2_ssd_backward.mojo) |
| Mamba scratch arenas and generation keyed weights | [NI45](#ni45) | [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo); [mamba/impl/modules/afn_arena.mojo](../../mamba/impl/modules/afn_arena.mojo); [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo); [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo) |
| Tiled same chain neural sequence GEMM | [NI50](#ni50) | [sequence/gemm_tiled.mojo](../../sequence/gemm_tiled.mojo); [sequence/exec_device.mojo](../../sequence/exec_device.mojo); [sequence/recurrent.mojo](../../sequence/recurrent.mojo); [sequence/mlp_fit.mojo](../../sequence/mlp_fit.mojo) |
| Reuse graph normalization workspace | [NI56](#ni56) | [x_cnn/device.mojo](../../x_cnn/device.mojo) |

Additional existing variants include NI03 `slack_two` and `slack_eight`, NI08 `existing_leaf64`, NI34 direct `ByteConfig.chunked_lm_head_v2`, NI40 `existing_ssd_tiles`, NI43 `existing_versioned_suffix`, NI45 `gemm_workspace` and `mamba3_session_stage_reuse`, and NI52 `existing_device_epochs`. Their exact controls and paths are retained in the variant entries. NI08 leaf64 retains recorded historical component losses; it is not promoted or proposed as an unchanged re-run.

## Existing neural stage runner

Configuration registry: [tools/neural_experiments.py](../../tools/neural_experiments.py) (`EXPERIMENTS` and `SETS`). Driver: [tools/neural_stage_timing.py](../../tools/neural_stage_timing.py). Each named configuration is compared with `baseline`; these old stage diagnostics require unchanged baseline bits and do not qualify V experiments or full-dataset performance. Names describe controls, not proof that a route differs on a particular build.

| Existing configuration | Environment delta from baseline | Files containing the implementation or caller | Comparison |
| --- | --- | --- | --- |
| `baseline` | none (clean inherited experiment settings) | [tools/neural_stage_timing.py](../../tools/neural_stage_timing.py) | Current compiled defaults |
| `no_retain_weights` | `MOJOLEARN_TRANSFORMER_RETAIN_WEIGHTS=0`; `MOJOLEARN_MAMBA3_RETAIN_WEIGHTS=0` | [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo); [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo) | Per-call weight uploads |
| `legacy_fresh_entry` | `MOJOLEARN_TRANSFORMER_SESSION_FRESH=0` | [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py) | Fresh session entry versus older entry |
| `legacy_everything` | `MOJOLEARN_TRANSFORMER_LEGACY_SETUP=1`; `MOJOLEARN_MAMBA3_LEGACY_SETUP=1` | [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py); [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py) | Older setup for both families |
| `no_stage_reset` | `MOJOLEARN_TRANSFORMER_STAGE_RESET=0` | [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo) | Stage initialization policy |
| `speculative_attn` | `MOJOLEARN_ATTN_SPECULATIVE=1` | [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo) | Speculative attention workspace path |
| `swiglu_fused` | `MOJOLEARN_SWIGLU_FUSED=1` | [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo) | Existing inference SwiGLU fusion; distinct from NI26 training tape arm |
| `no_layer_sync` | `MOJOLEARN_BYTE_LM_LAYER_SYNC=0` | [training/byte_lm.mojo](../../training/byte_lm.mojo) | Per-layer synchronization policy |
| `mamba3_legacy` | `MOJOLEARN_MAMBA3_LEGACY_SETUP=1` | [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py) | Older Mamba3 setup |
| `mamba3_no_retain_stages` | `MOJOLEARN_MAMBA3_RETAIN_STAGES=0` | [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo) | Forward stage retention |
| `norm_dw_own_ws` | `MOJOLEARN_TRANSFORMER_NORM_DW_OWN_WS=1` | [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo) | Norm parameter-gradient GEMM workspace |
| `s16_naive` | `MOJOLEARN_MAMBA3_S16_QK_ARM=naive` | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo); [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo) | Existing S16 Q/K backward schedule |
| `s16_shared` | `MOJOLEARN_MAMBA3_S16_QK_ARM=shared` | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo); [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo) | Existing S16 Q/K backward schedule |
| `s16_regs` | `MOJOLEARN_MAMBA3_S16_QK_ARM=regs` | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo); [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo) | Existing S16 Q/K backward schedule |
| `s16_regs2` | `MOJOLEARN_MAMBA3_S16_QK_ARM=regs2` | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo); [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo) | Existing S16 Q/K backward schedule |
| `s16_smem48` | `MOJOLEARN_MAMBA3_S16_QK_ARM=smem48` | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo); [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo) | Existing S16 Q/K backward schedule |
| `s16_regs2h` | `MOJOLEARN_MAMBA3_S16_QK_ARM=regs2h` | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo); [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo) | Existing S16 Q/K backward schedule |
| `s16_regsh` | `MOJOLEARN_MAMBA3_S16_QK_ARM=regsh` | [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo); [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo) | Existing S16 Q/K backward schedule |
| `opt_host` | `MOJOLEARN_OPTIMIZER_RESIDENT=0` | [python/mojolearn/_training_impl.py](../../python/mojolearn/_training_impl.py); [python/mojolearn/_x_sequence_optim.py](../../python/mojolearn/_x_sequence_optim.py) | Legacy nonresident optimizer comparison; not a new authorized host runtime route |
| `all_on` | `MOJOLEARN_ATTN_SPECULATIVE=1`; `MOJOLEARN_SWIGLU_FUSED=1`; `MOJOLEARN_BYTE_LM_LAYER_SYNC=0`; `MOJOLEARN_TRANSFORMER_STAGE_RESET=0` | [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo); [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo); [training/byte_lm.mojo](../../training/byte_lm.mojo); [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo) | Historical combination, not all new NI candidates |

The existing sets are `priority`, `s16`, `s16_apple`, `optimizer`, `default`, `nvidia` and `amd`. The runner covers `transformer-forward`, `mamba3-forward`, `samba-forward`, `samba-train-step`, `lm-forward` and `lm-train-step`. Its stage timing is diagnostic, and its historical Apple settings are not IDENTICAL performance votes.

## Existing GEMM runtime arms

The `gemm_step_arm_parse` selector and implementations are in [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo). They require `MOJOLEARN_GEMM_ARM_TRIAL=1` at compile time. The runtime choice is `MOJOLEARN_GEMM_ARM=<name>`; `shipped`, unset or empty is the reference. Without the trial build, a requested runtime arm does not establish that its kernel ran.

| Existing arm | Runtime selection |
| --- | --- |
| `shipped` | `MOJOLEARN_GEMM_ARM=shipped` |
| `lfold` | `MOJOLEARN_GEMM_ARM=lfold` |
| `half` | `MOJOLEARN_GEMM_ARM=half` |
| `half_ks16` | `MOJOLEARN_GEMM_ARM=half_ks16` |
| `quarter` | `MOJOLEARN_GEMM_ARM=quarter` |
| `head` | `MOJOLEARN_GEMM_ARM=head` |
| `half_head` | `MOJOLEARN_GEMM_ARM=half_head` |
| `ksplit` | `MOJOLEARN_GEMM_ARM=ksplit` |
| `ksplit_leaf` | `MOJOLEARN_GEMM_ARM=ksplit_leaf` |
| `tuned128` | `MOJOLEARN_GEMM_ARM=tuned128` |
| `kpack` | `MOJOLEARN_GEMM_ARM=kpack` |
| `kpack_wide` | `MOJOLEARN_GEMM_ARM=kpack_wide` |
| `kfoldv` | `MOJOLEARN_GEMM_ARM=kfoldv` |
| `kfoldv_leaf` | `MOJOLEARN_GEMM_ARM=kfoldv_leaf` |
| `kpack_pad` | `MOJOLEARN_GEMM_ARM=kpack_pad` |
| `kpack_padv` | `MOJOLEARN_GEMM_ARM=kpack_padv` |
| `kpack_hf` | `MOJOLEARN_GEMM_ARM=kpack_hf` |
| `kpack_gs` | `MOJOLEARN_GEMM_ARM=kpack_gs` |
| `kpack_hg` | `MOJOLEARN_GEMM_ARM=kpack_hg` |

Related files: [gemm/checks/gemm_step_arms_check.mojo](../../gemm/checks/gemm_step_arms_check.mojo); [tools/speed_gemm_arm.py](../../tools/speed_gemm_arm.py); [tools/neural_experiments.py](../../tools/neural_experiments.py). The stage runner exposes these via `--gemm-arms`. `MOJOLEARN_GEMM_ARM_SABOTAGE=1` is a deliberate bad-output reach diagnostic, not a performance candidate. Legacy routes with dimension-based assumptions still require the owner's neighboring-shape and non-board A/B removal policy; this index does not endorse those routes.

## Existing frozen candidate recipes

Registry: [tools/identical_candidate_recipes.json](../../tools/identical_candidate_recipes.json). Builder: [tools/identical_wave_native_build.py](../../tools/identical_wave_native_build.py) with `--candidate-recipe` and `--recipe-role`. These nine neural/shared-GEMM recipes retain `NOT_QUALIFIED_LEAVE_DISABLED`; source commits and evidence fields remain in the registry.

| Recipe | Candidate compiler defines | Reference compiler defines | Source files |
| --- | --- | --- | --- |
| `mamba3_parallel_angle/32` | `MOJOLEARN_IDN_M3_ANGLE_PARALLEL=1`; `MOJOLEARN_IDN_M3_ANGLE_BLOCK_32=1` | none (omit controls) | [mamba/impl/ops/mamba3_siso.mojo](../../mamba/impl/ops/mamba3_siso.mojo); [mamba/host/gen/mamba3_siso.mojo](../../mamba/host/gen/mamba3_siso.mojo); [mamba/checks/mamba3_oracle.mojo](../../mamba/checks/mamba3_oracle.mojo) |
| `mamba3_parallel_angle/64` | `MOJOLEARN_IDN_M3_ANGLE_PARALLEL=1` | none (omit controls) | [mamba/impl/ops/mamba3_siso.mojo](../../mamba/impl/ops/mamba3_siso.mojo); [mamba/host/gen/mamba3_siso.mojo](../../mamba/host/gen/mamba3_siso.mojo); [mamba/checks/mamba3_oracle.mojo](../../mamba/checks/mamba3_oracle.mojo) |
| `mamba3_parallel_angle/128` | `MOJOLEARN_IDN_M3_ANGLE_PARALLEL=1`; `MOJOLEARN_IDN_M3_ANGLE_BLOCK_128=1` | none (omit controls) | [mamba/impl/ops/mamba3_siso.mojo](../../mamba/impl/ops/mamba3_siso.mojo); [mamba/host/gen/mamba3_siso.mojo](../../mamba/host/gen/mamba3_siso.mojo); [mamba/checks/mamba3_oracle.mojo](../../mamba/checks/mamba3_oracle.mojo) |
| `gemm_group_slack/2` | `MOJOLEARN_IDN_GEMM_GROUP_SLACK_2=1` | none (omit controls) | [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `gemm_group_slack/8` | `MOJOLEARN_IDN_GEMM_GROUP_SLACK_8=1` | none (omit controls) | [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `gemm_group_s/half` | `MOJOLEARN_IDN_GEMM_GROUP_S_HALF=1` | none (omit controls) | [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `gemm_group_s/x2` | `MOJOLEARN_IDN_GEMM_GROUP_S_X2=1` | none (omit controls) | [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `gemm_group_tiles_body/1` | `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY=1` | none (omit controls) | [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) |
| `cross_entropy_fold256/256` | `MOJOLEARN_IDN_XENT_FOLD_BLOCK_256=1` | none (omit controls) | [x_cnn/ops.mojo](../../x_cnn/ops.mojo) |

The angle block sizes are separate compile arms. GEMM grouping recipes overlap NI03; the older Mamba angle and cross-entropy fold recipes are separate from NI43/NI35. The older recipe builder sets may cover diagnostic bindings only; the NI integration map adds affected full neural callers.

## Existing IDENTICAL manifests for neural and shared GEMM

These 22 earlier manifests contain related experiments or prototypes. Their source-status labels are reproduced as historical metadata, not renewed compilation or full-workload acceptance. Candidate lists in multi-arm campaigns may enumerate alternatives or repeat common defines; use each manifest's `compile_matrix.json` or campaign record for an individual arm. Do not concatenate contradictory values into one build.

### Existing I01

**Attribute existing GEMM schedules**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/I01/manifest.json](../../experiments/performance_ideas/I01/manifest.json).

- Recorded candidate defines: `MOJOLEARN_IDN_GEMM_GROUP_SLACK_2=1`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo); [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py)

**Individual arm records:** [experiments/performance_ideas/I01/compile_matrix.json](../../experiments/performance_ideas/I01/compile_matrix.json); [experiments/performance_ideas/I01/campaign.json](../../experiments/performance_ideas/I01/campaign.json)

### Existing I02

**Bound session GEMM scratch retention**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/I02/manifest.json](../../experiments/performance_ideas/I02/manifest.json).

- Recorded candidate defines: `MOJOLEARN_STEP_PHASE_TIMERS=1`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/bounded_workspace.mojo](../../gemm/experiments/bounded_workspace.mojo); [gemm/experiments/bounded_workspace_check.mojo](../../gemm/experiments/bounded_workspace_check.mojo); [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py)

**Individual arm records:** [experiments/performance_ideas/I02/compile_matrix.json](../../experiments/performance_ideas/I02/compile_matrix.json)

### Existing I03

**Batch independent products on separate grid jobs**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/I03/manifest.json](../../experiments/performance_ideas/I03/manifest.json).

- Recorded candidate defines: none (omit controls).
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/grouped_jobs.mojo](../../gemm/experiments/grouped_jobs.mojo); [gemm/experiments/grouped_jobs_check.mojo](../../gemm/experiments/grouped_jobs_check.mojo); [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py)

**Individual arm records:** [experiments/performance_ideas/I03/compile_matrix.json](../../experiments/performance_ideas/I03/compile_matrix.json)

The arm selection lives in the linked driver or campaign, rather than distinct top-level manifest defines.

### Existing I04

**Qualify a coherently shared opt-in 64-element IDENTICAL leaf version**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/I04/manifest.json](../../experiments/performance_ideas/I04/manifest.json).

- Recorded candidate defines: `MOJOLEARN_IDN_GEMM_FOLD_LEAF_64=1`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/contract.mojo](../../gemm/contract.mojo); [gemm/experiments/fold_profile_probe.mojo](../../gemm/experiments/fold_profile_probe.mojo); [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo); [gemm/experiments/profile_identity_check.mojo](../../gemm/experiments/profile_identity_check.mojo); [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py)

**Individual arm records:** [experiments/performance_ideas/I04/compile_matrix.json](../../experiments/performance_ideas/I04/compile_matrix.json)

### Existing I05

**Fuse bias after the explicit rounded product seam**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/I05/manifest.json](../../experiments/performance_ideas/I05/manifest.json).

- Recorded candidate defines: none (omit controls).
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/rounded_epilogue.mojo](../../gemm/experiments/rounded_epilogue.mojo); [gemm/experiments/rounded_epilogue_check.mojo](../../gemm/experiments/rounded_epilogue_check.mojo); [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py)

**Individual arm records:** [experiments/performance_ideas/I05/compile_matrix.json](../../experiments/performance_ideas/I05/compile_matrix.json)

The arm selection lives in the linked driver or campaign, rather than distinct top-level manifest defines.

### Existing I06

**qualify attention KV grid reuse across GQA tails**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/I06/manifest.json](../../experiments/performance_ideas/I06/manifest.json).

- Recorded candidate defines: `MOJOLEARN_ATTN_ARM_TRIAL=1`; `MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE=1`.
- Recorded reference defines: `MOJOLEARN_ATTN_ARM_TRIAL=1`.

**Implementation files:** [experiments/performance_ideas/I06/check.mojo](../../experiments/performance_ideas/I06/check.mojo); [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo)

**Individual arm records:** [experiments/performance_ideas/I06/compile_matrix.json](../../experiments/performance_ideas/I06/compile_matrix.json); [experiments/performance_ideas/I06/native_arms.json](../../experiments/performance_ideas/I06/native_arms.json)

### Existing I07

**exercise retained and recomputed attention backward lifetimes**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/I07/manifest.json](../../experiments/performance_ideas/I07/manifest.json).

- Recorded candidate defines: `MOJOLEARN_ATTN_ARM_TRIAL=1`.
- Recorded reference defines: `MOJOLEARN_ATTN_ARM_TRIAL=1`; `MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD=1`.

**Implementation files:** [experiments/performance_ideas/I07/check.mojo](../../experiments/performance_ideas/I07/check.mojo); [experiments/performance_ideas/I07/state_cost.mojo](../../experiments/performance_ideas/I07/state_cost.mojo)

**Individual arm records:** [experiments/performance_ideas/I07/compile_matrix.json](../../experiments/performance_ideas/I07/compile_matrix.json); [experiments/performance_ideas/I07/native_arms.json](../../experiments/performance_ideas/I07/native_arms.json)

### Existing I08

**attribute SSD tile reuse on multiple state cases and chunk tails**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/I08/manifest.json](../../experiments/performance_ideas/I08/manifest.json).

- Recorded candidate defines: `MOJOLEARN_IDN_M2_RETAIN_GL=1`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [experiments/performance_ideas/I08/check.mojo](../../experiments/performance_ideas/I08/check.mojo); [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo); [experiments/performance_ideas/I08/backward_check.mojo](../../experiments/performance_ideas/I08/backward_check.mojo); [mamba/impl/modules/mamba2_prefill_backward.mojo](../../mamba/impl/modules/mamba2_prefill_backward.mojo)

**Individual arm records:** [experiments/performance_ideas/I08/compile_matrix.json](../../experiments/performance_ideas/I08/compile_matrix.json); [experiments/performance_ideas/I08/native_arms.json](../../experiments/performance_ideas/I08/native_arms.json)

### Existing I09

**gate token-parallel recurrence on prefix and decode boundaries**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/I09/manifest.json](../../experiments/performance_ideas/I09/manifest.json).

- Recorded candidate defines: none (omit controls).
- Recorded reference defines: `MOJOLEARN_IDN_MAMBA_CONV_CELL_OFF=1`.

**Implementation files:** [experiments/performance_ideas/I09/check.mojo](../../experiments/performance_ideas/I09/check.mojo)

**Individual arm records:** [experiments/performance_ideas/I09/compile_matrix.json](../../experiments/performance_ideas/I09/compile_matrix.json); [experiments/performance_ideas/I09/native_arms.json](../../experiments/performance_ideas/I09/native_arms.json)

### Existing I10

**reduce ordered training status from per-tile contributions**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/I10/manifest.json](../../experiments/performance_ideas/I10/manifest.json).

- Recorded candidate defines: `MOJOLEARN_TRAIN_LIVE_STATUS=1`; `MOJOLEARN_STEP_GLUE_TRIAL=1`.
- Recorded reference defines: `MOJOLEARN_STEP_GLUE_TRIAL=1`.

**Implementation files:** [experiments/performance_ideas/I10/check.mojo](../../experiments/performance_ideas/I10/check.mojo); [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo); [training/byte_lm.mojo](../../training/byte_lm.mojo)

**Individual arm records:** [experiments/performance_ideas/I10/compile_matrix.json](../../experiments/performance_ideas/I10/compile_matrix.json); [experiments/performance_ideas/I10/native_arms.json](../../experiments/performance_ideas/I10/native_arms.json)

### Existing I11

**qualify canonical radix embedding updates under skew and vocabulary reuse**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/I11/manifest.json](../../experiments/performance_ideas/I11/manifest.json).

- Recorded candidate defines: none (omit controls).
- Recorded reference defines: `MOJOLEARN_IDN_EMB_RADIX_SORT_OFF=1`.

**Implementation files:** [experiments/performance_ideas/I11/check.mojo](../../experiments/performance_ideas/I11/check.mojo)

**Individual arm records:** [experiments/performance_ideas/I11/compile_matrix.json](../../experiments/performance_ideas/I11/compile_matrix.json); [experiments/performance_ideas/I11/native_arms.json](../../experiments/performance_ideas/I11/native_arms.json)

### Existing A01

**Isolate AMD smaller MFMA and band routing**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/A01/manifest.json](../../experiments/performance_ideas/A01/manifest.json).

- Recorded candidate defines: `MOJOLEARN_IDN_GEMM_MFMA16_OFF=1`; `MOJOLEARN_IDN_GEMM_AMD_BAND_MFMA_OFF=1`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)

**Individual arm records:** [experiments/performance_ideas/A01/compile_matrix.json](../../experiments/performance_ideas/A01/compile_matrix.json); [experiments/performance_ideas/A01/campaign.json](../../experiments/performance_ideas/A01/campaign.json)

### Existing A02

**Compare production one/two-page staging and bounded one/two/four-plane resource controls**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/A02/manifest.json](../../experiments/performance_ideas/A02/manifest.json).

- Recorded candidate defines: `MOJOLEARN_GEMM_ONE_PAGE=1`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo); [experiments/performance_ideas/A02/check.mojo](../../experiments/performance_ideas/A02/check.mojo); [gemm/experiments/bounded_staging.mojo](../../gemm/experiments/bounded_staging.mojo); [gemm/experiments/bounded_staging_check.mojo](../../gemm/experiments/bounded_staging_check.mojo)

**Individual arm records:** [experiments/performance_ideas/A02/compile_matrix.json](../../experiments/performance_ideas/A02/compile_matrix.json); [experiments/performance_ideas/A02/campaign.json](../../experiments/performance_ideas/A02/campaign.json)

### Existing A03

**Vary vector-aligned LDS operand strides**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/A03/manifest.json](../../experiments/performance_ideas/A03/manifest.json).

- Recorded candidate defines: `MOJOLEARN_IDN_GEMM_LDS_PAD_WORDS=0`; `MOJOLEARN_IDN_GEMM_LDS_PAD_WORDS=8`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)

**Individual arm records:** [experiments/performance_ideas/A03/compile_matrix.json](../../experiments/performance_ideas/A03/compile_matrix.json); [experiments/performance_ideas/A03/campaign.json](../../experiments/performance_ideas/A03/campaign.json)

### Existing A04

**Pair independent logical groups with supported XOR membership**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/A04/manifest.json](../../experiments/performance_ideas/A04/manifest.json).

- Recorded candidate defines: none (omit controls).
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/subwave_membership.mojo](../../gemm/experiments/subwave_membership.mojo); [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py)

**Individual arm records:** [experiments/performance_ideas/A04/compile_matrix.json](../../experiments/performance_ideas/A04/compile_matrix.json)

The arm selection lives in the linked driver or campaign, rather than distinct top-level manifest defines.

### Existing A05

**Halve the packed body row accumulator live range**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/A05/manifest.json](../../experiments/performance_ideas/A05/manifest.json).

- Recorded candidate defines: `MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE=1`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)

**Individual arm records:** [experiments/performance_ideas/A05/compile_matrix.json](../../experiments/performance_ideas/A05/compile_matrix.json); [experiments/performance_ideas/A05/campaign.json](../../experiments/performance_ideas/A05/campaign.json)

### Existing N01

**Isolate packed64 body tiles and their interaction**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/N01/manifest.json](../../experiments/performance_ideas/N01/manifest.json).

- Recorded candidate defines: `MOJOLEARN_GEMM_KPACK_RPT4=1`; `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY=1`; `MOJOLEARN_GEMM_KPACK_RPT4=1`; `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY=1`; `MOJOLEARN_IDN_GEMM_NV_STEP_KPACK_OFF=1`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)

**Individual arm records:** [experiments/performance_ideas/N01/compile_matrix.json](../../experiments/performance_ideas/N01/compile_matrix.json); [experiments/performance_ideas/N01/campaign.json](../../experiments/performance_ideas/N01/campaign.json)

### Existing N02

**Prove and dispatch a two-level logical fold stack**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/N02/manifest.json](../../experiments/performance_ideas/N02/manifest.json).

- Recorded candidate defines: `MOJOLEARN_GEMM_NV_FS4_OFF=1`; `MOJOLEARN_IDN_GEMM_FS2=1`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)

**Individual arm records:** [experiments/performance_ideas/N02/compile_matrix.json](../../experiments/performance_ideas/N02/compile_matrix.json); [experiments/performance_ideas/N02/campaign.json](../../experiments/performance_ideas/N02/campaign.json)

### Existing N03

**Pipeline supported asynchronous operand loads with explicit completion**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/N03/manifest.json](../../experiments/performance_ideas/N03/manifest.json).

- Recorded candidate defines: none (omit controls).
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/async_operand_pipeline.mojo](../../gemm/experiments/async_operand_pipeline.mojo); [gemm/experiments/async_operand_pipeline_check.mojo](../../gemm/experiments/async_operand_pipeline_check.mojo); [gemm/experiments/async_api_probe.mojo](../../gemm/experiments/async_api_probe.mojo); [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py)

**Individual arm records:** [experiments/performance_ideas/N03/compile_matrix.json](../../experiments/performance_ideas/N03/compile_matrix.json)

The arm selection lives in the linked driver or campaign, rather than distinct top-level manifest defines.

### Existing N04

**Separate contiguous and gather staging with counted caller passes**

Recorded status: `source_ready`. Registry: [experiments/performance_ideas/N04/manifest.json](../../experiments/performance_ideas/N04/manifest.json).

- Recorded candidate defines: `MOJOLEARN_GEMM_ARM_TRIAL=1`; `MOJOLEARN_GEMM_ARM_TRIAL=1`; `MOJOLEARN_GEMM_ARM_TRIAL=1`.
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py); [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo); [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo)

**Individual arm records:** [experiments/performance_ideas/N04/compile_matrix.json](../../experiments/performance_ideas/N04/compile_matrix.json); [experiments/performance_ideas/N04/campaign.json](../../experiments/performance_ideas/N04/campaign.json)

### Existing N05

**Batch compatible launches with fixed-address changing inputs**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/N05/manifest.json](../../experiments/performance_ideas/N05/manifest.json).

- Recorded candidate defines: none (omit controls).
- Recorded reference defines: none (omit controls).

**Implementation files:** [gemm/experiments/grouped_jobs.mojo](../../gemm/experiments/grouped_jobs.mojo); [gemm/experiments/changing_batch_check.mojo](../../gemm/experiments/changing_batch_check.mojo); [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py)

**Individual arm records:** [experiments/performance_ideas/N05/compile_matrix.json](../../experiments/performance_ideas/N05/compile_matrix.json)

The arm selection lives in the linked driver or campaign, rather than distinct top-level manifest defines.

### Existing N06

**Keep compact feature gradients in registers without query barriers**

Recorded status: `build_passed`. Registry: [experiments/performance_ideas/N06/manifest.json](../../experiments/performance_ideas/N06/manifest.json).

- Recorded candidate defines: none (omit controls).
- Recorded reference defines: none (omit controls).

**Implementation files:** [experiments/performance_ideas/N06/compact_grad.mojo](../../experiments/performance_ideas/N06/compact_grad.mojo); [experiments/performance_ideas/N06/check.mojo](../../experiments/performance_ideas/N06/check.mojo); [transformer/impl/llama/attention_v2.mojo](../../transformer/impl/llama/attention_v2.mojo); [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py)

**Individual arm records:** [experiments/performance_ideas/N06/compile_matrix.json](../../experiments/performance_ideas/N06/compile_matrix.json)

The arm selection lives in the linked driver or campaign, rather than distinct top-level manifest defines.

Examples of overlap: I01/N01 with NI03, I02 with NI01, I03/N05 with NI07, I04 with NI08 leaf64, I05 with NI09, I06 with NI19, I07 with NI21, I08 with NI40, I09 with NI37/NI38, I10 with NI29, and I11 with NI31. This is a relationship map, not an assertion that both routes or workloads are equivalent.

## Existing state and rollback A B harnesses

| Harness | Paired behavior | Related files |
| --- | --- | --- |
| [tools/lm_shards_ab_matrix.sh](../../tools/lm_shards_ab_matrix.sh) | Baseline/candidate binaries across one and two devices, logical shards and reversed order; strict state comparison | [tools/lm_shards_probe.py](../../tools/lm_shards_probe.py); [tools/lm_shards_ab_compare.py](../../tools/lm_shards_ab_compare.py); [training/byte_lm.mojo](../../training/byte_lm.mojo); [bindings/_mojolearn_byte_lm.mojo](../../bindings/_mojolearn_byte_lm.mojo) |
| [tools/byte_lm_pool_ab_matrix.sh](../../tools/byte_lm_pool_ab_matrix.sh) | Baseline/candidate fault builds; optimizer pool, replica/replay and rollback checks | [tools/byte_lm_optimizer_pool_check.py](../../tools/byte_lm_optimizer_pool_check.py); [tools/byte_lm_pool_ab_compare.py](../../tools/byte_lm_pool_ab_compare.py); [training/byte_lm_layer_pool.mojo](../../training/byte_lm_layer_pool.mojo); [bindings/_mojolearn_byte_lm.mojo](../../bindings/_mojolearn_byte_lm.mojo) |
| [tools/transformer_ab_input_sha.py](../../tools/transformer_ab_input_sha.py) | Companion input fingerprint for the existing host-thread comparison | [tools/host_threads_ab_check.py](../../tools/host_threads_ab_check.py); [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo) |

These existing harnesses were not run. They do not replace complete model quality and full-operation performance coverage; thread-resource policies must follow current AGENTS.md rather than inherited old diagnostic caps.

## Existing FAST neural references

These existing neural A/B sources were also found. They belong to Apple FAST, outside the active IDENTICAL campaign, and do not provide cross-vendor IDENTICAL acceptance. Their historical A/B convention may differ: the older Apple request files use A as baseline and B as candidate.

| Existing experiment | Manifest | Implementation files |
| --- | --- | --- |
| F07: FLASH and true GQA caller qualification with reach counters | [experiments/performance_ideas/F07/manifest.json](../../experiments/performance_ideas/F07/manifest.json) | [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo); [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo); [experiments/performance_ideas/F07/caller.py](../../experiments/performance_ideas/F07/caller.py) |
| F08: Independent LM backward fusion and view A/B training task | [experiments/performance_ideas/F08/manifest.json](../../experiments/performance_ideas/F08/manifest.json) | [experiments/performance_ideas/F08/caller.py](../../experiments/performance_ideas/F08/caller.py); [experiments/performance_ideas/apple_fast/lm_task.py](../../experiments/performance_ideas/apple_fast/lm_task.py); [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo); [training/byte_lm_afn_grad.mojo](../../training/byte_lm_afn_grad.mojo) |
| F09: Memory-bounded LM head in a fixed actual SGD task | [experiments/performance_ideas/F09/manifest.json](../../experiments/performance_ideas/F09/manifest.json) | [experiments/performance_ideas/F09/caller.py](../../experiments/performance_ideas/F09/caller.py); [training/chunked_lm_head_v2.mojo](../../training/chunked_lm_head_v2.mojo); [training/afn_optim.mojo](../../training/afn_optim.mojo) |
| F10: Independent SSD MMA and Mamba fusion caller arms | [experiments/performance_ideas/F10/manifest.json](../../experiments/performance_ideas/F10/manifest.json) | [experiments/performance_ideas/F10/caller.py](../../experiments/performance_ideas/F10/caller.py); [mamba/impl/modules/afn_ssd_mma.mojo](../../mamba/impl/modules/afn_ssd_mma.mojo); [mamba/impl/modeling/afn_mamba1_fused.mojo](../../mamba/impl/modeling/afn_mamba1_fused.mojo) |
| F20: Task-level optimizer fusion and FAST blocked LayerNorm qualification | [experiments/performance_ideas/F20/manifest.json](../../experiments/performance_ideas/F20/manifest.json) | [sequence/layernorm.mojo](../../sequence/layernorm.mojo); [training/afn_optim.mojo](../../training/afn_optim.mojo); [experiments/performance_ideas/F20/caller.py](../../experiments/performance_ideas/F20/caller.py) |

Older Apple neural request documents: [docs/apple-fast/ab-neural/README.md](../../docs/apple-fast/ab-neural/README.md); [docs/apple-fast/ab-neural/attn.md](../../docs/apple-fast/ab-neural/attn.md); [docs/apple-fast/ab-neural/gemm.md](../../docs/apple-fast/ab-neural/gemm.md); [docs/apple-fast/ab-neural/lm.md](../../docs/apple-fast/ab-neural/lm.md); [docs/apple-fast/ab-neural/mamba.md](../../docs/apple-fast/ab-neural/mamba.md); [docs/apple-fast/ab-neural/mlp.md](../../docs/apple-fast/ab-neural/mlp.md); [docs/apple-fast/ab-neural/optim.md](../../docs/apple-fast/ab-neural/optim.md); [docs/apple-fast/ab-neural/samba.md](../../docs/apple-fast/ab-neural/samba.md); [docs/apple-fast/ab-neural/tier.md](../../docs/apple-fast/ab-neural/tier.md); [docs/apple-fast/ab-neural/w2-epi.md](../../docs/apple-fast/ab-neural/w2-epi.md); [docs/apple-fast/ab-neural/w2-gemm2.md](../../docs/apple-fast/ab-neural/w2-gemm2.md); [docs/apple-fast/ab-neural/w2-lmgrad.md](../../docs/apple-fast/ab-neural/w2-lmgrad.md). Their existing common runner is [tools/afn_ab.sh](../../tools/afn_ab.sh). These are historical references, not new jobs to queue.

## Source file lookup for catalog ideas

This reverse index includes every implementation path explicitly recorded by a parent NI card or one of its variants. IDs after a colon identify the recorded variant. Shared runner and recipe files are listed at the start.

| Source file | Catalog ideas and variants |
| --- | --- |
| [bindings/_mojolearn_linalg.mojo](../../bindings/_mojolearn_linalg.mojo) | `NI08`, `NI08:existing_leaf64` |
| [bindings/_mojolearn_linalg_host.mojo](../../bindings/_mojolearn_linalg_host.mojo) | `NI08`, `NI08:existing_leaf64` |
| [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo) | `NI36`, `NI36:linear-backward-scratch-sharing`, `NI45`, `NI45:gemm_workspace`, `NI45:mamba3_session_stage_reuse`, `NI47`, `NI47:versioned_absolute_boundary_schema`, `NI48` |
| [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo) | `NI48` |
| [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo) | `NI20`, `NI34`, `NI34:existing-explicit-byte-config`, `NI35` |
| [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo) | `NI20`, `NI34`, `NI34:existing-explicit-byte-config`, `NI35` |
| [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo) | `NI20`, `NI36`, `NI36:linear-backward-scratch-sharing`, `NI48` |
| [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo) | `NI20` |
| [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo) | `NI08`, `NI08:existing_leaf64`, `NI10`, `NI17` |
| [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo) | `NI08`, `NI08:existing_leaf64`, `NI10`, `NI17` |
| [checks/kernel_matrix_gemm.mojo](../../checks/kernel_matrix_gemm.mojo) | `NI03`, `NI03:slack_eight`, `NI03:slack_two`, `NI06` |
| [embedding/checks/embedding_identical.mojo](../../embedding/checks/embedding_identical.mojo) | `NI31`, `NI32` |
| [embedding/checks/embedding_sort.mojo](../../embedding/checks/embedding_sort.mojo) | `NI31` |
| [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo) | `NI01`, `NI01:geometric_capacity`, `NI02`, `NI03`, `NI03:slack_eight`, `NI03:slack_two`, `NI04`, `NI05`, `NI06`, `NI08`, `NI08:existing_leaf64` |
| [gemm/contract.mojo](../../gemm/contract.mojo) | `NI08`, `NI08:existing_leaf64` |
| [gemm/experiments/bounded_workspace.mojo](../../gemm/experiments/bounded_workspace.mojo) | `NI01`, `NI01:geometric_capacity` |
| [gemm/experiments/grouped_jobs.mojo](../../gemm/experiments/grouped_jobs.mojo) | `NI07` |
| [gemm/experiments/neural_tiled.mojo](../../gemm/experiments/neural_tiled.mojo) | `NI07`, `NI09` |
| [gemm/experiments/rounded_epilogue.mojo](../../gemm/experiments/rounded_epilogue.mojo) | `NI09` |
| [gemm/host/gemm_oracle.mojo](../../gemm/host/gemm_oracle.mojo) | `NI08`, `NI08:existing_leaf64` |
| [gemm/host/identical_gemm.mojo](../../gemm/host/identical_gemm.mojo) | `NI08`, `NI08:existing_leaf64` |
| [mamba/host/gen/device_optimizations.mojo](../../mamba/host/gen/device_optimizations.mojo) | `NI38`, `NI38:state_window` |
| [mamba/host/gen/mamba2.mojo](../../mamba/host/gen/mamba2.mojo) | `NI37` |
| [mamba/host/gen/mamba2_ssd_backward.mojo](../../mamba/host/gen/mamba2_ssd_backward.mojo) | `NI44` |
| [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo) | `NI43`, `NI43:existing_versioned_suffix` |
| [mamba/host/gen/mamba3_siso.mojo](../../mamba/host/gen/mamba3_siso.mojo) | `NI42`, `NI42:historical_routing_B` |
| [mamba/host/gen/modeling_mamba.mojo](../../mamba/host/gen/modeling_mamba.mojo) | `NI37`, `NI46` |
| [mamba/host/gen/selective_scan_interface.mojo](../../mamba/host/gen/selective_scan_interface.mojo) | `NI38`, `NI38:state_window` |
| [mamba/host/gen/ssd_minimal.mojo](../../mamba/host/gen/ssd_minimal.mojo) | `NI39`, `NI40`, `NI40:existing_ssd_tiles`, `NI41` |
| [mamba/impl/modeling/afn_mamba1_fused.mojo](../../mamba/impl/modeling/afn_mamba1_fused.mojo) | `NI46` |
| [mamba/impl/modeling/modeling_mamba.mojo](../../mamba/impl/modeling/modeling_mamba.mojo) | `NI37`, `NI46` |
| [mamba/impl/modules/afn_arena.mojo](../../mamba/impl/modules/afn_arena.mojo) | `NI45`, `NI45:gemm_workspace`, `NI45:mamba3_session_stage_reuse` |
| [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo) | `NI37`, `NI40`, `NI40:existing_ssd_tiles`, `NI45`, `NI45:gemm_workspace`, `NI45:mamba3_session_stage_reuse` |
| [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo) | `NI45`, `NI45:gemm_workspace`, `NI45:mamba3_session_stage_reuse` |
| [mamba/impl/modules/mamba2.mojo](../../mamba/impl/modules/mamba2.mojo) | `NI37` |
| [mamba/impl/modules/mamba3.mojo](../../mamba/impl/modules/mamba3.mojo) | `NI47`, `NI47:versioned_absolute_boundary_schema` |
| [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo) | `NI43`, `NI43:existing_versioned_suffix` |
| [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo) | `NI39`, `NI40`, `NI40:existing_ssd_tiles`, `NI41` |
| [mamba/impl/ops/identical_scan_window.mojo](../../mamba/impl/ops/identical_scan_window.mojo) | `NI38`, `NI38:state_window` |
| [mamba/impl/ops/mamba2_ssd_backward.mojo](../../mamba/impl/ops/mamba2_ssd_backward.mojo) | `NI40`, `NI40:existing_ssd_tiles`, `NI41`, `NI44` |
| [mamba/impl/ops/mamba3_siso.mojo](../../mamba/impl/ops/mamba3_siso.mojo) | `NI42`, `NI42:historical_routing_B` |
| [mamba/impl/ops/selective_scan_backward.mojo](../../mamba/impl/ops/selective_scan_backward.mojo) | `NI38`, `NI38:state_window` |
| [mamba/impl/ops/selective_scan_interface.mojo](../../mamba/impl/ops/selective_scan_interface.mojo) | `NI38`, `NI38:state_window` |
| [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py) | `NI08`, `NI08:existing_leaf64`, `NI10`, `NI17` |
| [python/mojolearn/_linalg_impl.py](../../python/mojolearn/_linalg_impl.py) | `NI08`, `NI08:existing_leaf64` |
| [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py) | `NI36`, `NI36:linear-backward-scratch-sharing`, `NI48` |
| [python/mojolearn/_samba_impl.py](../../python/mojolearn/_samba_impl.py) | `NI34`, `NI34:existing-explicit-byte-config`, `NI35`, `NI36`, `NI36:linear-backward-scratch-sharing`, `NI48` |
| [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py) | `NI36`, `NI36:linear-backward-scratch-sharing`, `NI48` |
| [sequence/adafactor.mojo](../../sequence/adafactor.mojo) | `NI54` |
| [sequence/adafactor_candidates.mojo](../../sequence/adafactor_candidates.mojo) | `NI54` |
| [sequence/dispatch.mojo](../../sequence/dispatch.mojo) | `NI49`, `NI49:original_failed_cooperative_scan`, `NI49:row_serial_scan` |
| [sequence/exec.mojo](../../sequence/exec.mojo) | `NI51`, `NI51:layernorm_parameter_leaf32` |
| [sequence/exec_device.mojo](../../sequence/exec_device.mojo) | `NI49`, `NI49:original_failed_cooperative_scan`, `NI49:row_serial_scan`, `NI50`, `NI51`, `NI51:layernorm_parameter_leaf32`, `NI53`, `NI54` |
| [sequence/gemm_tiled.mojo](../../sequence/gemm_tiled.mojo) | `NI50` |
| [sequence/layernorm.mojo](../../sequence/layernorm.mojo) | `NI51`, `NI51:layernorm_parameter_leaf32` |
| [sequence/mlp.mojo](../../sequence/mlp.mojo) | `NI52`, `NI52:existing_device_epochs` |
| [sequence/mlp_fit.mojo](../../sequence/mlp_fit.mojo) | `NI50`, `NI52`, `NI52:existing_device_epochs` |
| [sequence/moe_group.mojo](../../sequence/moe_group.mojo) | `NI53` |
| [sequence/moe_reg.mojo](../../sequence/moe_reg.mojo) | `NI53` |
| [sequence/moe_tiled.mojo](../../sequence/moe_tiled.mojo) | `NI53` |
| [sequence/ops.mojo](../../sequence/ops.mojo) | `NI51`, `NI51:layernorm_parameter_leaf32` |
| [sequence/pyapi.mojo](../../sequence/pyapi.mojo) | `NI53` |
| [sequence/recurrent.mojo](../../sequence/recurrent.mojo) | `NI49`, `NI49:original_failed_cooperative_scan`, `NI49:row_serial_scan`, `NI50`, `NI51`, `NI51:layernorm_parameter_leaf32` |
| [sequence/recurrent_scan.mojo](../../sequence/recurrent_scan.mojo) | `NI49`, `NI49:original_failed_cooperative_scan`, `NI49:row_serial_scan` |
| [tools/mamba_host_gen.py](../../tools/mamba_host_gen.py) | `NI38`, `NI38:state_window` |
| [training/CHUNKED_LM_HEAD_V2.md](../../training/CHUNKED_LM_HEAD_V2.md) | `NI34`, `NI34:existing-explicit-byte-config` |
| [training/IDENTICAL_LOSS_CONTRACT.md](../../training/IDENTICAL_LOSS_CONTRACT.md) | `NI35` |
| [training/byte_lm.mojo](../../training/byte_lm.mojo) | `NI24`, `NI27`, `NI29`, `NI31`, `NI32`, `NI34`, `NI34:existing-explicit-byte-config`, `NI36`, `NI36:linear-backward-scratch-sharing` |
| [training/byte_lm_config.mojo](../../training/byte_lm_config.mojo) | `NI20`, `NI34`, `NI34:existing-explicit-byte-config`, `NI35` |
| [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo) | `NI20`, `NI34`, `NI34:existing-explicit-byte-config` |
| [training/byte_lm_host_backward.mojo](../../training/byte_lm_host_backward.mojo) | `NI34`, `NI34:existing-explicit-byte-config` |
| [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo) | `NI20`, `NI35` |
| [training/byte_lm_layer_pool.mojo](../../training/byte_lm_layer_pool.mojo) | `NI24`, `NI27`, `NI36`, `NI36:linear-backward-scratch-sharing` |
| [training/checks/chunked_lm_head_oracle.mojo](../../training/checks/chunked_lm_head_oracle.mojo) | `NI34`, `NI34:existing-explicit-byte-config` |
| [training/checks/loss.mojo](../../training/checks/loss.mojo) | `NI33`, `NI35` |
| [training/checks/loss_contract.mojo](../../training/checks/loss_contract.mojo) | `NI35` |
| [training/checks/loss_oracle.mojo](../../training/checks/loss_oracle.mojo) | `NI35` |
| [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo) | `NI28`, `NI29`, `NI30` |
| [training/chunked_lm_head_host.mojo](../../training/chunked_lm_head_host.mojo) | `NI34`, `NI34:existing-explicit-byte-config` |
| [training/chunked_lm_head_v2.mojo](../../training/chunked_lm_head_v2.mojo) | `NI34`, `NI34:existing-explicit-byte-config` |
| [training/dev_tensors.mojo](../../training/dev_tensors.mojo) | `NI01`, `NI01:geometric_capacity`, `NI36`, `NI36:linear-backward-scratch-sharing` |
| [training/loss_host_rows.mojo](../../training/loss_host_rows.mojo) | `NI35` |
| [training/loss_reduction_v2.mojo](../../training/loss_reduction_v2.mojo) | `NI35` |
| [training/neural_gemm_workspace.mojo](../../training/neural_gemm_workspace.mojo) | `NI01`, `NI01:geometric_capacity` |
| [training/neural_identical_experiments.mojo](../../training/neural_identical_experiments.mojo) | `NI20`, `NI23`, `NI24`, `NI25`, `NI26`, `NI27`, `NI32`, `NI33`, `NI34`, `NI34:existing-explicit-byte-config`, `NI35`, `NI36`, `NI36:linear-backward-scratch-sharing` |
| [training/opt_gate.mojo](../../training/opt_gate.mojo) | `NI28`, `NI30` |
| [training/samba_ops.mojo](../../training/samba_ops.mojo) | `NI25` |
| [transformer/ATTENTION_V2_CONTRACT.md](../../transformer/ATTENTION_V2_CONTRACT.md) | `NI20` |
| [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo) | `NI20`, `NI26` |
| [transformer/checks/transformer_backward_oracle.mojo](../../transformer/checks/transformer_backward_oracle.mojo) | `NI20` |
| [transformer/checks/transformer_oracle.mojo](../../transformer/checks/transformer_oracle.mojo) | `NI20` |
| [transformer/impl/llama/attention_v2_model_contract.mojo](../../transformer/impl/llama/attention_v2_model_contract.mojo) | `NI20` |
| [transformer/impl/llama/attention_v2_model_device.mojo](../../transformer/impl/llama/attention_v2_model_device.mojo) | `NI20` |
| [transformer/impl/llama/attention_v2_model_host.mojo](../../transformer/impl/llama/attention_v2_model_host.mojo) | `NI20` |
| [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo) | `NI19`, `NI21`, `NI22` |
| [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo) | `NI07`, `NI20`, `NI23`, `NI24`, `NI25`, `NI26` |
| [x_cnn/device.mojo](../../x_cnn/device.mojo) | `NI02`, `NI09`, `NI10`, `NI11`, `NI12`, `NI14`, `NI14:address_bounds_only`, `NI15`, `NI16`, `NI17`, `NI18`, `NI55`, `NI56`, `NI57`, `NI58`, `NI59`, `NI60` |
| [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo) | `NI10`, `NI17`, `NI57` |
| [x_cnn/neural_aux_contract.mojo](../../x_cnn/neural_aux_contract.mojo) | `NI55`, `NI57`, `NI58`, `NI59`, `NI60` |
| [x_cnn/neural_aux_ops.mojo](../../x_cnn/neural_aux_ops.mojo) | `NI55`, `NI58`, `NI59`, `NI60` |
| [x_cnn/neural_col2im.mojo](../../x_cnn/neural_col2im.mojo) | `NI14`, `NI14:address_bounds_only` |
| [x_cnn/neural_weight_cache.mojo](../../x_cnn/neural_weight_cache.mojo) | `NI13` |
| [x_cnn/ops.mojo](../../x_cnn/ops.mojo) | `NI09`, `NI10`, `NI12`, `NI14`, `NI14:address_bounds_only`, `NI16`, `NI17`, `NI18`, `NI57` |

## Remaining limits

NI38's original versioned affine-scan design remains pending; `state_window` is its implemented S alternative. NI49's historical cooperative-scan quality failure remains unresolved; `row_serial_scan` is a separate unverified repair candidate. NI13 has no incumbent repack to eliminate, and NI22 targets invisible attention work the incumbent already prunes. Broader optional variants retain the limits in their linked ledgers.

The underlying idea catalog also records required interactions. This file is an experiment and file inventory, not a measurement board: no speed ratios, quality passes, default promotions or build receipts are added.
