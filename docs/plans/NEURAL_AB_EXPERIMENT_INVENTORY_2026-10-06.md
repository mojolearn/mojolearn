# Neural A/B experiment and file inventory — 2026-10-06

Branch `ideas/neural-identical-ab-20261006-r3`, forked from main at `fd6cf80453a6f18eb02e81566c824e7da106ccf0`.

This document indexes all 64 new neural idea cards, their selected source arms and callers, the 60 existing I/A/N/F manifests found in this checkout, and additional named A/B suites. Existing classical and FAST entries are references; the new implementation scope is neural IDENTICAL only. Research variants beyond each selected arm are not claimed as implemented.

All new source remains uncompiled, unverified and unmeasured by request. No identity checks, quality evaluation, timing, board updates or default promotions were performed. Bits may differ across versions or A/B arms; within one selected version they must agree on NVIDIA, AMD, Apple and host. Source wiring does not prove that obligation.

## Selectors and model integration

[arms.json](../../experiments/neural_identical_ab/arms.json) contains concrete A/B compile defines, environment and runtime settings. [tools/neural_identical_ab.py](../../tools/neural_identical_ab.py) `configure` writes an arm JSON consumed by `bench_board_neural.py --neural-ab-config` and an optional shell environment consumed by the existing native builders. It never compiles or executes. The native build must apply the same profile flags to every affected GPU and host binding; a config file is not proof that installed binaries match it.

The retained [performance_ideas runner](../../tools/performance_ideas.py) still owns its I/A/N/F frozen-manifest evidence protocol. New NN selectors link reused IDs and sources; old component evidence is not copied into new model qualification. `neural_identical_ab.py list --include-existing` exposes both registries.

Explicit operations reach transformer tape/VJP and windowed cache writes, Mamba forward/VJP and owned weights, the residual/dropout layer, stateless MLP session batching, the chunked LM head, and configured Samba clipping/accumulation. Neural algorithm recipes can apply the CNN unpooled-block setting in both arms. Normal lanes remain available for ordinary model operations. Full input hashes, cap audits, all affected estimator workloads and relevant combinations are still future campaign prerequisites. A layer-only operation is labeled as such, not as a full model train step.

## New neural experiments

A is candidate; B is baseline. The exact controls below are compile defines unless prefixed `env:` or `runtime:`. `is_defined` flags are disabled by omission, never by setting them to zero. Hardware parameters such as NN10's fill budget require an explicit recorded value. Each linked lane inventory retains admission, interactions and limitations.

### NN01 — Attribute GEMM schedules at real neural callers

Files: [gemm/experiments/neural_plans.mojo](../../gemm/experiments/neural_plans.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo), [training/checks/optimizer_contract.mojo](../../training/checks/optimizer_contract.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py), [bindings/neural_gemm_boundary.mojo](../../bindings/neural_gemm_boundary.mojo), [bindings/neural_gemm_boundary_host.mojo](../../bindings/neural_gemm_boundary_host.mojo), [bindings/neural_gemm_boundary_common.mojo](../../bindings/neural_gemm_boundary_common.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN01=1`; `MOJOLEARN_GEMM_ARM_TRIAL=1`; `MOJOLEARN_IDN_NEURAL_GEMM_ARM=10`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN01=1`; `MOJOLEARN_GEMM_ARM_TRIAL=1`; `MOJOLEARN_IDN_NEURAL_GEMM_ARM=10`; `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: I01, N01, N04, A01.

- Legacy explicit geometry applies only where existing step-arm routing admits it; inherited shipped routes remain B. NN01 refuses NN03/NN04.

### NN02 — Stream GEMM partial planes into a bounded fold

Files: [gemm/experiments/neural_streaming.mojo](../../gemm/experiments/neural_streaming.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo), [training/checks/optimizer_contract.mojo](../../training/checks/optimizer_contract.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py), [bindings/neural_gemm_boundary.mojo](../../bindings/neural_gemm_boundary.mojo), [bindings/neural_gemm_boundary_host.mojo](../../bindings/neural_gemm_boundary_host.mojo), [bindings/neural_gemm_boundary_common.mojo](../../bindings/neural_gemm_boundary_common.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN02=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN02=1`; `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: I02.


### NN03 — Versioned GEMM leaf lengths

Files: [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo).

Callers: [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo), [training/checks/optimizer_contract.mojo](../../training/checks/optimizer_contract.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py), [bindings/neural_gemm_boundary.mojo](../../bindings/neural_gemm_boundary.mojo), [bindings/neural_gemm_boundary_host.mojo](../../bindings/neural_gemm_boundary_host.mojo), [bindings/neural_gemm_boundary_common.mojo](../../bindings/neural_gemm_boundary_common.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN03=1`; `MOJOLEARN_IDN_NEURAL_LEAF=64`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN03=1`; `MOJOLEARN_IDN_NEURAL_LEAF=128`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: I04.

- Dense projection graph only; M2/M3 SSD/SISO and sequence internal algorithms keep separately pinned profiles on every column. Additional leaf256 variant is selectable.

### NN04 — Versioned independent accumulator chains inside GEMM leaves

Files: [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo).

Callers: [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo), [training/checks/optimizer_contract.mojo](../../training/checks/optimizer_contract.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py), [bindings/neural_gemm_boundary.mojo](../../bindings/neural_gemm_boundary.mojo), [bindings/neural_gemm_boundary_host.mojo](../../bindings/neural_gemm_boundary_host.mojo), [bindings/neural_gemm_boundary_common.mojo](../../bindings/neural_gemm_boundary_common.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN04=1`; `MOJOLEARN_IDN_NEURAL_CHAINS=2`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN04=1`; `MOJOLEARN_IDN_NEURAL_CHAINS=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: none separately claimed.

- Dense projection graph only; specialized SSD/SISO/sequence internal profiles stay pinned. Additional chains4 variant is selectable.

### NN05 — Group independent projection jobs

Files: [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/experiments/neural_grouped.mojo](../../gemm/experiments/neural_grouped.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN05=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "model_options": "Llama gate/up same shape; existing options/INT15 fallback admission preserved.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN05=1`; `MOJOLEARN_IDN_NEURAL_PAIR_CONTROL=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "model_options": "Llama gate/up same shape; existing options/INT15 fallback admission preserved.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: I03, N05.

- Chosen model placement is compatible Llama gate/up projection pair; arbitrary QKV/dX/dW grouping is separate future placement. Pair route retains existing option/INT15 fallbacks.

### NN06 — Rounded neural GEMM epilogues

Files: [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [gemm/experiments/neural_epilogue.mojo](../../gemm/experiments/neural_epilogue.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN06=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN06=1`; `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: I05.

- Chosen model placement is native fused SmallMLP training forward; host and separate backward keep same declared arithmetic. Generic residual/scale epilogue placements remain component APIs.

### NN07 — Reuse transposed operand staging across neural jobs

Files: [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/experiments/neural_grouped.mojo](../../gemm/experiments/neural_grouped.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN07=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "model_options": "Llama gate/up same shape; existing options/INT15 fallback admission preserved.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN07=1`; `MOJOLEARN_IDN_NEURAL_PAIR_CONTROL=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "model_options": "Llama gate/up same shape; existing options/INT15 fallback admission preserved.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: N04.

- Chosen placement is pair-local shared A staging in Llama gate/up; generation-keyed cross-call stage component is not implicitly used by models.

### NN08 — Bounded asynchronous operand loading

Files: [gemm/experiments/neural_plans.mojo](../../gemm/experiments/neural_plans.mojo), [gemm/experiments/async_operand_pipeline.mojo](../../gemm/experiments/async_operand_pipeline.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo), [training/checks/optimizer_contract.mojo](../../training/checks/optimizer_contract.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py), [bindings/neural_gemm_boundary.mojo](../../bindings/neural_gemm_boundary.mojo), [bindings/neural_gemm_boundary_host.mojo](../../bindings/neural_gemm_boundary_host.mojo), [bindings/neural_gemm_boundary_common.mojo](../../bindings/neural_gemm_boundary_common.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN08=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN08=1`; `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: N03.

- NVIDIA asynchronous path only; AMD/Apple execute synchronous selected profile and require future upstream support for equivalent native async scheduling.

### NN09 — Shared-memory page count and bank layout

Files: [gemm/experiments/neural_tiled.mojo](../../gemm/experiments/neural_tiled.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo), [training/checks/optimizer_contract.mojo](../../training/checks/optimizer_contract.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py), [bindings/neural_gemm_boundary.mojo](../../bindings/neural_gemm_boundary.mojo), [bindings/neural_gemm_boundary_host.mojo](../../bindings/neural_gemm_boundary_host.mojo), [bindings/neural_gemm_boundary_common.mojo](../../bindings/neural_gemm_boundary_common.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN09=1`; `MOJOLEARN_IDN_NEURAL_STAGE_DEPTH=2`; `MOJOLEARN_IDN_NEURAL_STAGE_PAD=0`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN09=1`; `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL=1`; `MOJOLEARN_IDN_NEURAL_STAGE_DEPTH=2`; `MOJOLEARN_IDN_NEURAL_STAGE_PAD=0`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: A02, A03.


### NN10 — Neural GEMM cost-based dispatch

Files: [gemm/experiments/neural_plans.mojo](../../gemm/experiments/neural_plans.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo), [training/checks/optimizer_contract.mojo](../../training/checks/optimizer_contract.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py), [bindings/neural_gemm_boundary.mojo](../../bindings/neural_gemm_boundary.mojo), [bindings/neural_gemm_boundary_host.mojo](../../bindings/neural_gemm_boundary_host.mojo), [bindings/neural_gemm_boundary_common.mojo](../../bindings/neural_gemm_boundary_common.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN10=1`; `MOJOLEARN_IDN_NEURAL_FILL_BLOCKS={MOJOLEARN_IDN_NEURAL_FILL_BLOCKS}`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN10=1`; `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL=1`; `MOJOLEARN_IDN_NEURAL_FILL_BLOCKS={MOJOLEARN_IDN_NEURAL_FILL_BLOCKS}`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: I01.

- Selector must require positive recorded MOJOLEARN_IDN_NEURAL_FILL_BLOCKS for both arms; full dataset neighboring-shape/non-board evidence remains future work.

### NN11 — Specialize fold storage to the logical tree

Files: [gemm/experiments/neural_streaming.mojo](../../gemm/experiments/neural_streaming.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo), [training/checks/optimizer_contract.mojo](../../training/checks/optimizer_contract.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py), [bindings/neural_gemm_boundary.mojo](../../bindings/neural_gemm_boundary.mojo), [bindings/neural_gemm_boundary_host.mojo](../../bindings/neural_gemm_boundary_host.mojo), [bindings/neural_gemm_boundary_common.mojo](../../bindings/neural_gemm_boundary_common.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN11=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN11=1`; `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: N02.

- Standalone capacity family refuses NN01/NN08/NN09/NN10/NN15; NN02 global-capacity placement is the supported composition.

### NN12 — Model-owned GEMM plan and workspace reuse

Files: [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [gemm/experiments/neural_plans.mojo](../../gemm/experiments/neural_plans.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN12=1`; `MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS=16777216`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN12=1`; `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL=1`; `MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS=16777216`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: I02.

- Mamba custom idn_gemm_ws, CNN slot caches and ByteLM-head manual scratch ownership are excluded; NN40 separately changes Mamba growth.

### NN13 — Tile sequence-family same-chain GEMM

Files: [sequence/ops.mojo](../../sequence/ops.mojo), [sequence/exec_device.mojo](../../sequence/exec_device.mojo), [sequence/gemm_tiled.mojo](../../sequence/gemm_tiled.mojo).

Callers: [sequence/ops.mojo](../../sequence/ops.mojo), [sequence/exec_device.mojo](../../sequence/exec_device.mojo), [sequence/gemm_tiled.mojo](../../sequence/gemm_tiled.mojo), [sequence/recurrent.mojo](../../sequence/recurrent.mojo), [sequence/pyapi.mojo](../../sequence/pyapi.mojo), [sequence/exec.mojo](../../sequence/exec.mojo), [bindings/_mojolearn_x_sequence.mojo](../../bindings/_mojolearn_x_sequence.mojo), [bindings/_mojolearn_x_sequence_host.mojo](../../bindings/_mojolearn_x_sequence_host.mojo), [python/mojolearn/_x_sequence_rnn.py](../../python/mojolearn/_x_sequence_rnn.py), [sequence/mlp.mojo](../../sequence/mlp.mojo), [sequence/mlp_fit.mojo](../../sequence/mlp_fit.mojo).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_algos.py](../../tools/bench_board_algos.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF=1`; runtime: `{"numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json), [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_sequence_tiles.

- Reused inherited default; no new NN13 switch or new evidence. Extra sequence batching variants remain separate.
- Inherited NVIDIA/AMD route uses size-derived M/N guard; Apple timing does not vote. No classical forecast source or workload is newly changed or claimed.

### NN14 — Group convolution-lowered GEMMs with bounded im2col

Files: [x_cnn/ops.mojo](../../x_cnn/ops.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo), [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo), [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [x_cnn/ops.mojo](../../x_cnn/ops.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo), [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo), [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_algos.py](../../tools/bench_board_algos.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN14_BOUNDED_IM2COL=1`; runtime: `{"allocation_query": "x_cnn_bounded_im2col()", "api": "Conv2d.forward/backward; CNNClassifier.fit/predict", "numeric_mode": "identical", "saved_cols": "CNNClassifier query allocates one-word unused sentinel under NN14."}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"allocation_query": "x_cnn_bounded_im2col()", "api": "Conv2d.forward/backward; CNNClassifier.fit/predict", "numeric_mode": "identical", "saved_cols": "CNNClassifier query allocates one-word unused sentinel under NN14."}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json), [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_cnn_routes.

- State/CNN owner is authoritative; full input/output/preactivation/grow and previously retained owner capacity remain.
- 8 MiB bounds forward cols+y2 except an indivisible row; outputs/gradients/other retained caches are separate, not a total VRAM guarantee.
- NN14 suppresses NN45 epilogue fusion when combined.

### NN15 — Hardware-specific schedules under one profile

Files: [gemm/experiments/neural_tiled.mojo](../../gemm/experiments/neural_tiled.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [training/dev_tensors.mojo](../../training/dev_tensors.mojo), [training/mlp_ops.mojo](../../training/mlp_ops.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo), [training/checks/optimizer_contract.mojo](../../training/checks/optimizer_contract.mojo), [x_cnn/host/gemm_host.mojo](../../x_cnn/host/gemm_host.mojo), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py), [bindings/neural_gemm_boundary.mojo](../../bindings/neural_gemm_boundary.mojo), [bindings/neural_gemm_boundary_host.mojo](../../bindings/neural_gemm_boundary_host.mojo), [bindings/neural_gemm_boundary_common.mojo](../../bindings/neural_gemm_boundary_common.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN15=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN15=1`; `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: A04.

- Chosen arm changes generic output-thread mapping; vendor-specific tuned geometry variants are optional research, not claimed as wired.

### NN16 — Avoid unnecessary zero-tail and scratch passes

Files: [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [gemm/experiments/neural_streaming.mojo](../../gemm/experiments/neural_streaming.mojo), [gemm/neural_dispatch.mojo](../../gemm/neural_dispatch.mojo), [gemm/neural_backward.mojo](../../gemm/neural_backward.mojo), [gemm/host/neural_gemm.mojo](../../gemm/host/neural_gemm.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo).

Callers: [training/byte_lm_logits.mojo](../../training/byte_lm_logits.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN16=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_NEURAL_NN16=1`; `MOJOLEARN_IDN_NEURAL_GEMM_CONTROL=1`; runtime: `{"caller_workload": "Saved full workload for selected public caller; no smaller substitute or inherited component timing.", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json). Existing related IDs: none separately claimed.

- Chosen model arm removes real ByteLM-head cold scratch clear; standalone synthetic-clear component is diagnostic only.

### NN17 — Share attention K/V tiles across query heads

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py), [experiments/neural_identical_ab/workloads/NN17-full-gqa.json](../../experiments/neural_identical_ab/workloads/NN17-full-gqa.json).

A: `MOJOLEARN_ATTN_ARM_TRIAL=1`; `MOJOLEARN_NN17_GQA_FOUR_HEADS=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`; env: `MOJOLEARN_NUMERIC_MODE=identical`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`; runtime: `{"operation": "transformer_configured_forward", "transformer_config": {"batch": 1, "d_model": 512, "head_dim": 64, "intermediate": 1536, "length": 2048, "n_heads": 8, "n_kv": 2}}`.

B: `MOJOLEARN_ATTN_ARM_TRIAL=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`; env: `MOJOLEARN_NUMERIC_MODE=identical`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`; runtime: `{"operation": "transformer_configured_forward", "transformer_config": {"batch": 1, "d_model": 512, "head_dim": 64, "intermediate": 1536, "length": 2048, "n_heads": 8, "n_kv": 2}}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: I06, ATTN_RUNTIME_ARMS.

- Full GQA callers must actually reach HD64/TQ32/QRES/PF non-swizzled arm and GQA ratio divisible by four; otherwise record no-distinct-runtime-arm.
- Retain I06 two-head LOSER evidence; no unchanged repeat is proposed.
- Measure full forward+backward and complete default interactions.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.
- The ordinary full transformer fixture has GQA ratio one. The explicit saved NN17-full-gqa recipe uses ratio four in both arms and reaches the public configured model; it is a supplemental full block workload, not full LM corpus coverage. Its input hash, branch reach and all execution evidence remain pending.

### NN18 — Skip structurally masked attention tiles

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_ATTN_ARM_TRIAL=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled`; env: `MOJOLEARN_NUMERIC_MODE=identical`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

B: `MOJOLEARN_ATTN_ARM_TRIAL=1`; `MOJOLEARN_NN18_DENSE_TILE_CONTROL=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled`; env: `MOJOLEARN_NUMERIC_MODE=identical`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: ATTN_RUNTIME_ARMS, ATTN_MASKED_TAIL_REPAIR.

- Compile/identity/quality/timing unrun. Full tiled forward/backward model reach and refusal rate must accompany future measurements.
- More aggressive task compaction is an optional separate variant; this selected arm isolates existing structural tile bounds.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN19 — Retain versus recompute attention intermediates

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo), [experiments/performance_ideas/I07/state_cost.mojo](../../experiments/performance_ideas/I07/state_cost.mojo), [experiments/performance_ideas/I07/check.mojo](../../experiments/performance_ideas/I07/check.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_ATTN_ARM_TRIAL=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

B: `MOJOLEARN_ATTN_ARM_TRIAL=1`; `MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: I07, ATTN_V1_PACKED_ESTASH, ATTN_V1_ALIAS_Y_ESTASH, ATTN_APPLE_ERECOMP.

- Both model arms explicitly request the same supported estash schedule. A must report positive kept cells; B forced-recompute must report zero. HD64/QRES/PF/kvgrid and finite-regime reach are required.
- Reuse corrected I07 retained/recompute source and accepted prior receipts. New model source remains uncompiled/unexecuted; full workloads, quality, identity and memory lifetime/retention qualification are unrun.
- Keep earlier no-distinct-arm and confounded evidence separate.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN20 — Versioned streaming stable attention softmax

Files: [transformer/experiments/attention_summary_contract.mojo](../../transformer/experiments/attention_summary_contract.mojo), [transformer/experiments/attention_summary_tree.mojo](../../transformer/experiments/attention_summary_tree.mojo), [transformer/experiments/summary_model_contract.mojo](../../transformer/experiments/summary_model_contract.mojo), [transformer/experiments/summary_model.mojo](../../transformer/experiments/summary_model.mojo), [transformer/experiments/summary_model_host.mojo](../../transformer/experiments/summary_model_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [transformer/checks/transformer_oracle.mojo](../../transformer/checks/transformer_oracle.mojo), [transformer/checks/transformer_backward_oracle.mojo](../../transformer/checks/transformer_backward_oracle.mojo), [transformer/experiments/profile.mojo](../../transformer/experiments/profile.mojo), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [transformer/checks/transformer_oracle.mojo](../../transformer/checks/transformer_oracle.mojo), [transformer/checks/transformer_backward_oracle.mojo](../../transformer/checks/transformer_backward_oracle.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN20_BALANCED_SUMMARY_TREE=1`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

B: env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: ATTENTION_V2, ATTN_RUNTIME_ARMS, TRANSFORMER_HOST_FUSED_ATTENTION.

- No compilation, four-column identity, derivative/task quality or timing admission has been run. Existing attention_v2 large-shape failures remain visible and do not qualify this new graph.
- Public arm refuses softcap, INT15 and legacy planted-score calls. Dropout/nondefault-options training remain the existing API refusal.
- One owner per row is a concrete graph implementation, not a throughput claim; cooperative/tree scheduling is an optional later variant.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN21 — Fuse attention pointwise score transforms

Files: [transformer/experiments/attention_schedules.mojo](../../transformer/experiments/attention_schedules.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN21_SCALE_MASK=1`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=eager`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

B: env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=eager`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: ATTN_MASKED_TAIL_REPAIR.

- Eager/fallback scope only; an automatic fused-only full model run cannot establish candidate reach.
- Full affected model E2E, task quality, vendor+host bits and performance remain unrun.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN22 — Canonical dK/dV task geometry

Files: [transformer/experiments/attention_schedules.mojo](../../transformer/experiments/attention_schedules.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN22_EAGER_DKDV_PAIR=1`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=eager`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

B: env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=eager`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: ATTN_RUNTIME_ARMS, ATTN_LAUNCH_GEOMETRY.

- Eager/fallback scope only; an automatic fused-only full model run cannot establish candidate reach.
- Full affected model E2E, task quality, vendor+host bits and performance remain unrun.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN23 — Fuse attention backward row dot and pointwise gradients

Files: [transformer/experiments/attention_schedules.mojo](../../transformer/experiments/attention_schedules.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN23_ROWDOT_DS=1`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=eager`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

B: env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=eager`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: ATTN_RUNTIME_ARMS, ATTN_APPLE_MMA.

- Eager/fallback scope only; an automatic fused-only full model run cannot establish candidate reach.
- Full affected model E2E, task quality, vendor+host bits and performance remain unrun.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN24 — Versioned RMSNorm/LayerNorm fixed-lane reductions

Files: [transformer/experiments/norm_profile_contract.mojo](../../transformer/experiments/norm_profile_contract.mojo), [transformer/experiments/norm_profile.mojo](../../transformer/experiments/norm_profile.mojo), [transformer/experiments/attention_schedules.mojo](../../transformer/experiments/attention_schedules.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [transformer/checks/transformer_oracle.mojo](../../transformer/checks/transformer_oracle.mojo), [transformer/checks/transformer_backward_oracle.mojo](../../transformer/checks/transformer_backward_oracle.mojo), [transformer/experiments/profile.mojo](../../transformer/experiments/profile.mojo), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [transformer/checks/transformer_oracle.mojo](../../transformer/checks/transformer_oracle.mojo), [transformer/checks/transformer_backward_oracle.mojo](../../transformer/checks/transformer_backward_oracle.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN24_NORM_LANES8=1`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

B: env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: BWD_RMS_DH_DOT_FUSION, BWD_NORM2_RESIDUAL_FUSION, RESIDUAL1_NORM2_FUSION, RESIDUAL2_NEXT_NORM.

- Compilation, identity, task/gradient quality and full-workload timings unrun.
- Extended-options LayerNorm training remains the existing explicit refusal. Standalone LN derivative helper is an optional component, not newly advertised public training.
- Independent sequence/Mamba norm graphs remain their existing versions; transformer-side Samba norms are migrated together by root.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN25 — Separate norm scalar fold from parallel cell scaling

Files: [transformer/experiments/attention_schedules.mojo](../../transformer/experiments/attention_schedules.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN25_RMS_SPLIT_SCALE=1`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

B: env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: BWD_RMS_DH_DOT_FUSION, BWD_RMS_SHAPE_RULE_REMOVAL, STEP_GLUE_ROWS.

- RMSNorm launcher scope; existing fused residual-norm and non-RMSNorm variants remain outside this arm.
- Repeated scalar div/rsqrt in flat cell kernel may regress; record loss and full operation timing.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN26 — Training SwiGLU forward/backward pass fusion

Files: [transformer/experiments/attention_schedules.mojo](../../transformer/experiments/attention_schedules.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN26_TRAIN_SWIGLU=1`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

B: env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: SWIGLU_FORWARD_ONLY, BWD_GATED_SILU_FUSION.

- Plain gated SiLU training only; bias/GELU/ungated/forward-only calls preserve old routes.
- Saved sigmoid/derivative extension is not implemented; this arm reuses existing paired backward and existing SiLU saved stage.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN27 — Session-owned RoPE frequency and position state

Files: [transformer/experiments/attention_schedules.mojo](../../transformer/experiments/attention_schedules.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN27_QK_ROPE_PAIR=1`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

B: env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward or lm-forward; full saved affected-estimator recipes still required", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: none separately claimed.

- All compilation/identity/quality/timing unrun. Session-wide cross-layer RoPE ownership and scan elision are optional separate variants.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN28 — Remove training-only dead KV-cache writes

Files: [transformer/experiments/attention_schedules.mojo](../../transformer/experiments/attention_schedules.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

Callers: [training/byte_lm.mojo](../../training/byte_lm.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN28_DEAD_TRAINING_CACHE=1`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

B: env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: IDN_LLAMA_REFUSE_BATCH, IDN_ATTN_CACHE_NOWAIT.

- Root wired the four _byte_forward_loss full-prefill calls with explicit empty-cache reset and retain_kv_cache=False under NN28. All backward and trace stages remain populated.
- Compilation/identity/quality/timing unrun; broader owners are optional extension.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN29 — Bounded decode KV layout and append fusion

Files: [transformer/experiments/attention_schedules.mojo](../../transformer/experiments/attention_schedules.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN29_RING_PAIR=1`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward on full saved x; explicit consumed window cache, zero logical position each repeat", "operation": "transformer_windowed_forward", "transformer_window": 256}`.

B: env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "transformer-forward on full saved x; explicit consumed window cache, zero logical position each repeat", "operation": "transformer_windowed_forward", "transformer_window": 256}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: IDN_ATTN_CACHE_NOWAIT.

- Compilation/identity/quality/timing unrun. Direct-to-final projection/cache layout and multi-request batching are optional broader variants.
- Both arms use explicit window=256 model semantics, adjustable through the frozen full-workload recipe, never a hardware/benchmark dimension dispatch.
- Model/harness consumes saved K/V ring outputs; carried-state decode and nonzero starting-position continuation still require separate full-workload recipes.
- Existing window=0 comparator is deliberately not run for this operation; qualification remains pending matching window/cache reference.
- Do not combine a dead-cache NN28 caller with an experiment claiming ring-commit reach.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN30 — Backward residual and gradient buffer views

Files: [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

B: `MOJOLEARN_IDN_BWD_ENTRY_ALIAS_OFF=1`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"caller_workload": "lm-train-step (and all other affected transformer/Samba training recipes); forward-only board execution does not cover this candidate", "operation": "model_default"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: IDN_ATTN_SCAN_CACHE, IDN_ATTN_BWD_SCAN_REUSE, IDN_ATTN_SCRATCH_CACHE, IDN_ATTN_ONE_FLAG_WAIT, IDN_BWD_ENTRY_ALIAS.

- Existing IDN_BWD_ENTRY_ALIAS remains shipped ON; ALL_OFF disables it. No new default was promoted. Reuse accepted existing receipts for unchanged source.
- Further alias/ownership transfer is an optional broader variant.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN31 — Bounded model activation checkpoint policy

Files: [transformer/experiments/checkpoint_contract.mojo](../../transformer/experiments/checkpoint_contract.mojo), [training/neural_attention_owner.mojo](../../training/neural_attention_owner.mojo), [transformer/host/attention_tape.mojo](../../transformer/host/attention_tape.mojo), [transformer/experiments/profile.mojo](../../transformer/experiments/profile.mojo), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [training/neural_attention_owner.mojo](../../training/neural_attention_owner.mojo), [transformer/host/attention_tape.mojo](../../transformer/host/attention_tape.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [transformer/checks/transformer_oracle.mojo](../../transformer/checks/transformer_oracle.mojo), [transformer/checks/transformer_backward_oracle.mojo](../../transformer/checks/transformer_backward_oracle.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN31_BOUNDED_CHECKPOINTS=1`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"activation_budget_bytes": 536870912, "caller_workload": "transformer-forward full saved x; VJP cotangent is the same immutable full x; no optimizer/loss", "expect_retained": true, "minimum_replay_ops_per_byte": 0, "operation": "transformer_forward_vjp_tape"}`.

B: env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"activation_budget_bytes": 536870912, "caller_workload": "transformer-forward full saved x; VJP cotangent is the same immutable full x; no optimizer/loss", "expect_retained": false, "minimum_replay_ops_per_byte": 0, "operation": "transformer_forward_vjp_tape"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: TRANSFORMER_SESSION_REUSE.

- No compilation, stale-ticket/failure-lifetime execution, four-column identity, quality, memory-capacity measurement or timing has been run.
- Chosen public scope is default FP32 IDENTICAL zero-state prefill, including sliding windows. Nondefault options, dropout, INT15 and carried-state training remain explicit refusals.
- Per-block activation budget covers retained forward buffers including GEMM workspace. Immutable input/weight snapshots, transient forward/backward storage and process caches are outside that bound.
- Multi-layer global budget scheduling and direct ByteTrainer/Samba optimizer adoption are optional broader integrations; the chosen public TransformerBlock model route is implemented.
- Candidate must fit the explicitly frozen retention budget; otherwise harness refuses expected retained=true. Raising a budget requires a new recipe and retained memory evidence.
- Native retained metadata is recorded; ordinary forward quality does not qualify the VJP, stale-owner behavior or full model training.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN32 — Retain Samba attention forward state for backward

Files: [transformer/experiments/checkpoint_contract.mojo](../../transformer/experiments/checkpoint_contract.mojo), [training/neural_attention_owner.mojo](../../training/neural_attention_owner.mojo), [transformer/host/attention_tape.mojo](../../transformer/host/attention_tape.mojo), [transformer/experiments/profile.mojo](../../transformer/experiments/profile.mojo), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py).

Callers: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [bindings/_mojolearn_transformer_host.mojo](../../bindings/_mojolearn_transformer_host.mojo), [training/neural_attention_owner.mojo](../../training/neural_attention_owner.mojo), [transformer/host/attention_tape.mojo](../../transformer/host/attention_tape.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [transformer/checks/transformer_oracle.mojo](../../transformer/checks/transformer_oracle.mojo), [transformer/checks/transformer_backward_oracle.mojo](../../transformer/checks/transformer_backward_oracle.mojo), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN32_RETAIN_FORWARD=1`; env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"activation_budget_bytes": 536870912, "caller_workload": "transformer-forward full saved x; VJP cotangent is the same immutable full x; no optimizer/loss", "expect_retained": true, "minimum_replay_ops_per_byte": 0, "operation": "transformer_forward_vjp_tape"}`.

B: env: `MOJOLEARN_NUMERIC_MODE=identical`; runtime: `{"activation_budget_bytes": 536870912, "caller_workload": "transformer-forward full saved x; VJP cotangent is the same immutable full x; no optimizer/loss", "expect_retained": false, "minimum_replay_ops_per_byte": 0, "operation": "transformer_forward_vjp_tape"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json). Existing related IDs: I07, TRANSFORMER_SESSION_REUSE.

- No compilation, stale-ticket/failure-lifetime execution, four-column identity, quality, memory-capacity measurement or timing has been run.
- Chosen public scope is default FP32 IDENTICAL zero-state prefill, including sliding windows. Nondefault options, dropout, INT15 and carried-state training remain explicit refusals.
- Per-block activation budget covers retained forward buffers including GEMM workspace. Immutable input/weight snapshots, transient forward/backward storage and process caches are outside that bound.
- Multi-layer global budget scheduling and direct ByteTrainer/Samba optimizer adoption are optional broader integrations; the chosen public TransformerBlock model route is implemented.
- Candidate must fit the explicitly frozen retention budget; otherwise harness refuses expected retained=true. Raising a budget requires a new recipe and retained memory evidence.
- Native retained metadata is recorded; ordinary forward quality does not qualify the VJP, stale-owner behavior or full model training.
- Source-only by user request: compilation, host/NVIDIA/AMD/Apple identity, independent quality, timing, neighboring shapes, non-board workload and full affected-estimator coverage have not been executed.

### NN33 — Attribute SSD/SISO shared tile reuse

Files: [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo), [mamba/host/gen/ssd_minimal.mojo](../../mamba/host/gen/ssd_minimal.mojo).

Callers: [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py), [mamba/impl/modules/mamba2.mojo](../../mamba/impl/modules/mamba2.mojo), [mamba/impl/modules/afn_defines.mojo](../../mamba/impl/modules/afn_defines.mojo).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN33_YDIAG_ROWS8=1`; `MOJOLEARN_NN33_CSTATE_P16=1`; runtime: `{"numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: I08.

- Requires inherited SSD tiles and 20 KiB resource guard. Rows8 uses 512 lanes; no compilation/resource/runtime evidence claimed.

### NN34 — Versioned absolute-chunk selective scan

Files: [mamba/impl/ops/neural_scan_profile.mojo](../../mamba/impl/ops/neural_scan_profile.mojo), [mamba/impl/ops/neural_mamba_scan.mojo](../../mamba/impl/ops/neural_mamba_scan.mojo), [mamba/impl/ops/selective_scan_backward.mojo](../../mamba/impl/ops/selective_scan_backward.mojo), [mamba/impl/modeling/modeling_mamba.mojo](../../mamba/impl/modeling/modeling_mamba.mojo), [mamba/checks/mamba_oracle.mojo](../../mamba/checks/mamba_oracle.mojo), [mamba/checks/mamba_backward.mojo](../../mamba/checks/mamba_backward.mojo), [mamba/host/gen/neural_scan_profile.mojo](../../mamba/host/gen/neural_scan_profile.mojo), [mamba/host/gen/neural_mamba_scan.mojo](../../mamba/host/gen/neural_mamba_scan.mojo), [mamba/host/gen/selective_scan_backward.mojo](../../mamba/host/gen/selective_scan_backward.mojo), [mamba/host/gen/modeling_mamba.mojo](../../mamba/host/gen/modeling_mamba.mojo), [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo), [bindings/_mojolearn_neural_host.mojo](../../bindings/_mojolearn_neural_host.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py), [experiments/neural_identical_ab/NN34_AFFINE_PREFIX_CONTRACT.md](../../experiments/neural_identical_ab/NN34_AFFINE_PREFIX_CONTRACT.md).

Callers: [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py), [bindings/_mojolearn_neural_host.mojo](../../bindings/_mojolearn_neural_host.mojo), [mamba/impl/modeling/modeling_mamba.mojo](../../mamba/impl/modeling/modeling_mamba.mojo), [mamba/checks/mamba_backward.mojo](../../mamba/checks/mamba_backward.mojo).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN34_AFFINE_PREFIX=1`; runtime: `{"api": "Mamba1Block.forward/step/backward; Mamba1DecodeSession; allocate_state/export_state/load_state", "baseline_profile_id": 1, "candidate_profile_id": 2, "contract": "experiments/neural_identical_ab/NN34_AFFINE_PREFIX_CONTRACT.md", "numeric_mode": "identical", "state_arrays": ["conv_window", "h", "affine_boundary", "affine_a", "affine_b"], "state_metadata": ["profile_id", "absolute_position"]}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"api": "Mamba1Block.forward/step/backward; Mamba1DecodeSession; allocate_state/export_state/load_state", "baseline_profile_id": 1, "candidate_profile_id": 2, "contract": "experiments/neural_identical_ab/NN34_AFFINE_PREFIX_CONTRACT.md", "numeric_mode": "identical", "state_arrays": ["conv_window", "h", "affine_boundary", "affine_a", "affine_b"], "state_metadata": ["profile_id", "absolute_position"]}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: I09.

- Backward remains the existing zero-state prefill API; no state cotangent or independent-position carried-ragged API added.
- Each arm creates its own profile1/profile2 state; cross-profile state loading is rejected.
- VJP repeats up to32 prefix trees/chunk. Speed/gradient/quality is unproven.

### NN35 — Parallel causal depthwise-convolution cells

Files: [mamba/impl/modeling/modeling_mamba.mojo](../../mamba/impl/modeling/modeling_mamba.mojo), [mamba/impl/modules/mamba2.mojo](../../mamba/impl/modules/mamba2.mojo), [mamba/host/gen/modeling_mamba.mojo](../../mamba/host/gen/modeling_mamba.mojo), [mamba/host/gen/mamba2.mojo](../../mamba/host/gen/mamba2.mojo).

Callers: [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN35_CONV_PAIR=1`; runtime: `{"numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: I09.


### NN36 — Cache Mamba decay exponent values

Files: [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo), [tools/mamba_host_gen.py](../../tools/mamba_host_gen.py), [mamba/host/gen/ssd_minimal.mojo](../../mamba/host/gen/ssd_minimal.mojo).

Callers: [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py), [mamba/impl/modules/mamba2.mojo](../../mamba/impl/modules/mamba2.mojo).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN36_SHARED_DECAY=1`; runtime: `{"numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: I08.


### NN37 — Prune unused triangular SSD work

Files: [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo), [mamba/host/gen/ssd_minimal.mojo](../../mamba/host/gen/ssd_minimal.mojo).

Callers: [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py), [mamba/impl/modules/mamba2.mojo](../../mamba/impl/modules/mamba2.mojo).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN37_TRIANGLE_TASKS=1`; runtime: `{"numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: I08.

- Requires inherited CB lower route. Include full zero-fill launch and traffic in timed boundary.

### NN38 — Versioned shared Mamba-3 angle-gradient suffixes

Files: [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo), [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo).

Callers: [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py), [mamba/checks/mamba3_backward.mojo](../../mamba/checks/mamba3_backward.mojo), [python/mojolearn/_samba_impl.py](../../python/mojolearn/_samba_impl.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN38_CACHE_SUFFIX_SEEDS=1`; runtime: `{"numeric_mode": "identical", "operation": "mamba_forward_vjp"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical", "operation": "mamba_forward_vjp"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_m3_angle_suffix.

- Requires inherited angle suffix. This is schedule reuse, not a newly implemented suffix numerical profile.

### NN39 — Versioned Mamba parameter-gradient folds

Files: [mamba/impl/ops/neural_gradient_profile.mojo](../../mamba/impl/ops/neural_gradient_profile.mojo), [mamba/impl/ops/mamba2_ssd_backward.mojo](../../mamba/impl/ops/mamba2_ssd_backward.mojo), [mamba/host/gen/mamba2_ssd_backward.mojo](../../mamba/host/gen/mamba2_ssd_backward.mojo).

Callers: [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py), [mamba/checks/mamba2_backward.mojo](../../mamba/checks/mamba2_backward.mojo).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN39_M2_GRAD_TREE=1`; runtime: `{"numeric_mode": "identical", "operation": "mamba_forward_vjp"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical", "operation": "mamba_forward_vjp"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_m2_gradient_leaves.

- Version-to-version bits intentionally change; other gradient families keep incumbent graph.

### NN40 — Mamba immutable weight generations and workspace

Files: [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py).

Callers: [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [bindings/_mojolearn_mamba_host.mojo](../../bindings/_mojolearn_mamba_host.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py), [python/mojolearn/_samba_impl.py](../../python/mojolearn/_samba_impl.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN40_WS_HEADROOM=1`; `MOJOLEARN_NN40_OWNED_WEIGHTS=1`; env: `MOJOLEARN_MAMBA3_LEGACY_SETUP=0`; runtime: `{"after_parameter_update": "block.install_owned_weights()", "api": "Mamba3Block fresh forward/backward", "export": "block.export_owned_weights()", "install": "block.install_owned_weights()", "install_owned_weights": true, "metadata": "block.session_info()", "numeric_mode": "identical", "operation": "mamba3_owned_forward", "owned_weights": true, "release": "block.release_owned_weights()"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; env: `MOJOLEARN_MAMBA3_LEGACY_SETUP=0`; runtime: `{"after_parameter_update": null, "api": "Mamba3Block fresh forward/backward", "export": "block.export_owned_weights()", "install": null, "install_owned_weights": false, "metadata": "block.session_info()", "numeric_mode": "identical", "operation": "mamba3_owned_forward", "owned_weights": false, "release": "block.release_owned_weights()"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_mamba_workspace_sessions.

- Requires inherited Mamba workspace; generic NN12 controls do not change this custom owner.
- Define alone does not transfer weight ownership; explicit install/reinstall required after each intended update.
- Explicit carried-state/decode retains ordinary weights. Alternate Mamba3BlockInference host binding is outside owned-snapshot scope.
- MOJOLEARN_MAMBA3_LEGACY_SETUP=1 is incompatible. Host reports/info probes now exported; host stage reuse remains zero.

### NN41 — One-launch ordered recurrent inference/training segments

Files: [sequence/recurrent_scan.mojo](../../sequence/recurrent_scan.mojo), [sequence/recurrent.mojo](../../sequence/recurrent.mojo), [sequence/exec_device.mojo](../../sequence/exec_device.mojo).

Callers: [sequence/recurrent.mojo](../../sequence/recurrent.mojo), [sequence/pyapi.mojo](../../sequence/pyapi.mojo), [sequence/exec_device.mojo](../../sequence/exec_device.mojo), [sequence/exec.mojo](../../sequence/exec.mojo), [bindings/_mojolearn_x_sequence.mojo](../../bindings/_mojolearn_x_sequence.mojo), [bindings/_mojolearn_x_sequence_host.mojo](../../bindings/_mojolearn_x_sequence_host.mojo), [python/mojolearn/_x_sequence_rnn.py](../../python/mojolearn/_x_sequence_rnn.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_algos.py](../../tools/bench_board_algos.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_IDN_SEQ_LSTM_SCAN=1`; runtime: `{"numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_recurrent_scan_gradients.

- Applies for1<=H<=1024 and G*H<=4096; other shapes fall back.
- Inherited FAST quality failure remains: LSTM accuracy0.9608 to0.5002 and R2 0.9804 to-0.1043. No run establishes IDENTICAL cause/quality.
- If team_barrier cannot order device-memory loop stores/loads, wait for documented Modular support. No spin barrier/toolchain workaround.

### NN42 — Fuse recurrent gate pointwise updates

Files: [sequence/ops.mojo](../../sequence/ops.mojo), [sequence/recurrent.mojo](../../sequence/recurrent.mojo).

Callers: [sequence/recurrent.mojo](../../sequence/recurrent.mojo), [sequence/pyapi.mojo](../../sequence/pyapi.mojo), [sequence/exec_device.mojo](../../sequence/exec_device.mojo), [sequence/exec.mojo](../../sequence/exec.mojo), [bindings/_mojolearn_x_sequence.mojo](../../bindings/_mojolearn_x_sequence.mojo), [bindings/_mojolearn_x_sequence_host.mojo](../../bindings/_mojolearn_x_sequence_host.mojo), [python/mojolearn/_x_sequence_rnn.py](../../python/mojolearn/_x_sequence_rnn.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_algos.py](../../tools/bench_board_algos.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN42_INPUT_BIAS_FUSED=1`; runtime: `{"numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_recurrent_scan_gradients.

- Inactive whenever SEQ_LSTM_SCAN compiled, even runtime shape fallback; NN41+NN42 is not two effective changes.

### NN43 — Versioned recurrent weight and bias gradients

Files: [sequence/recurrent.mojo](../../sequence/recurrent.mojo), [sequence/checks/oracle.mojo](../../sequence/checks/oracle.mojo), [sequence/ops.mojo](../../sequence/ops.mojo).

Callers: [sequence/recurrent.mojo](../../sequence/recurrent.mojo), [sequence/pyapi.mojo](../../sequence/pyapi.mojo), [sequence/exec_device.mojo](../../sequence/exec_device.mojo), [sequence/exec.mojo](../../sequence/exec.mojo), [bindings/_mojolearn_x_sequence.mojo](../../bindings/_mojolearn_x_sequence.mojo), [bindings/_mojolearn_x_sequence_host.mojo](../../bindings/_mojolearn_x_sequence_host.mojo), [python/mojolearn/_x_sequence_rnn.py](../../python/mojolearn/_x_sequence_rnn.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_algos.py](../../tools/bench_board_algos.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN43_WGRAD_FIXED128=1`; runtime: `{"numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_recurrent_scan_gradients.

- Requires inherited blocked recurrent gradients; forward-only timing cannot exercise changed weight/bias folds.

### NN44 — MoE stable token grouping and grouped expert jobs

Files: [sequence/moe_group.mojo](../../sequence/moe_group.mojo), [sequence/moe_tiled.mojo](../../sequence/moe_tiled.mojo), [sequence/exec_device.mojo](../../sequence/exec_device.mojo).

Callers: [sequence/pyapi.mojo](../../sequence/pyapi.mojo), [sequence/exec_device.mojo](../../sequence/exec_device.mojo), [sequence/exec.mojo](../../sequence/exec.mojo), [bindings/_mojolearn_x_sequence.mojo](../../bindings/_mojolearn_x_sequence.mojo), [bindings/_mojolearn_x_sequence_host.mojo](../../bindings/_mojolearn_x_sequence_host.mojo), [python/mojolearn/_x_sequence_moe.py](../../python/mojolearn/_x_sequence_moe.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_algos.py](../../tools/bench_board_algos.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN44_STABLE_GROUP=1`; `MOJOLEARN_NN44_EXPERT_BISECT=1`; env: `MOJOLEARN_SEQ_MOE_TILED=1`; runtime: `{"numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; env: `MOJOLEARN_SEQ_MOE_TILED=1`; runtime: `{"numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_moe_device_group.

- Stable grouping O(experts*pairs) may lose.
- Expert bisection needs tiled products; both arms explicitly MOJOLEARN_SEQ_MOE_TILED=1.
- No MoE gradient/training API is claimed.

### NN45 — Fuse CNN activation/bias/residual passes

Files: [x_cnn/ops.mojo](../../x_cnn/ops.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo).

Callers: [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo), [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo), [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo), [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_algos.py](../../tools/bench_board_algos.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_XCNN_NO_DIRECT_CONV=1`; `MOJOLEARN_NN45_CONV_RELU=1`; runtime: `{"api": "CNNClassifier.fit/predict", "estimator_settings": {"pool_size": 1}, "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_XCNN_NO_DIRECT_CONV=1`; runtime: `{"api": "CNNClassifier.fit/predict", "estimator_settings": {"pool_size": 1}, "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_cnn_routes.

- Pooled blocks already fuse ReLU/maxpool and do not execute NN45.
- Without common direct-conv control, inherited one-leaf path bypasses NN45; native-default interactions need a separate future A/B.
- NN14 takes precedence and suppresses NN45.

### NN46 — Deterministic CNN gradient and pooling schedules

Files: [x_cnn/ops.mojo](../../x_cnn/ops.mojo).

Callers: [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo), [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo), [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo), [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_algos.py](../../tools/bench_board_algos.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN46_GATHER_BOUNDS=1`; runtime: `{"numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_cnn_routes.

- Reverse sabotage retains full bounds; no host scheduling speedup claimed.
- No new dWeight tree or shared gather tile selected.

### NN47 — Neural BatchNorm statistics and running-state fusion

Files: [x_cnn/ops.mojo](../../x_cnn/ops.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo).

Callers: [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo), [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo), [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo), [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_algos.py](../../tools/bench_board_algos.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN47_APPLY_RUNNING=1`; runtime: `{"api": "BatchNorm1d/BatchNorm2d and affected ResNet blocks", "numeric_mode": "identical", "training": true}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"api": "BatchNorm1d/BatchNorm2d and affected ResNet blocks", "numeric_mode": "identical", "training": true}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_batchnorm_apply_stats.

- Evaluation and existing moment reductions unchanged; no raw-moment variance.

### NN48 — Neural graph aggregation tile reuse

Files: [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/ops.mojo](../../x_cnn/ops.mojo).

Callers: [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo), [bindings/_mojolearn_x_cnn.mojo](../../bindings/_mojolearn_x_cnn.mojo), [bindings/_mojolearn_x_cnn_host.mojo](../../bindings/_mojolearn_x_cnn_host.mojo), [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py).

Workload/selector files: [experiments/performance_ideas/README.md](../../experiments/performance_ideas/README.md), [tools/bench_board_algos.py](../../tools/bench_board_algos.py).

A: `MOJOLEARN_NUMERIC_IDENTICAL=1`; `MOJOLEARN_NN48_CSR_TILES=1`; runtime: `{"api": "GCNConv.forward/backward; GraphSAGE mean forward/backward", "numeric_mode": "identical"}`.

B: `MOJOLEARN_NUMERIC_IDENTICAL=1`; runtime: `{"api": "GCNConv.forward/backward; GraphSAGE mean forward/backward", "numeric_mode": "identical"}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json](../../experiments/neural_identical_ab/lanes/state_cnn_integration_inventory.json). Existing related IDs: existing_neural_csr.

- GraphSAGE max excluded. Include zero-degree nodes and tails.
- No classical graph/PageRank source changed.

### NN49 — Stable embedding grouping with touched-row gradients

Files: [embedding/checks/embedding_identical.mojo](../../embedding/checks/embedding_identical.mojo).

Callers: [embedding/checks/embedding_identical.mojo](../../embedding/checks/embedding_identical.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN49_EMB_VECTOR4=1`.

B: incumbent, new flags absent.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- Route only runs when existing touched-row dispatch is active (n_positions < vocab); other shapes retain incumbent.
- Full LM/embedding callers and all-vendor qualification remain unrun; include zeroing/sort/readback in timing.

### NN50 — Reuse immutable token grouping across backward uses

Files: [embedding/owned_runs.mojo](../../embedding/owned_runs.mojo), [embedding/checks/embedding_identical.mojo](../../embedding/checks/embedding_identical.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

Callers: [embedding/owned_runs.mojo](../../embedding/owned_runs.mojo), [embedding/checks/embedding_identical.mojo](../../embedding/checks/embedding_identical.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN50_OWNED_RUNS=1`.

B: incumbent, new flags absent.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- Cold construction and repeated same/different-batch qualification intentionally unrun.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

### NN51 — Resident token and target validation

Files: [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/checks/loss.mojo](../../training/checks/loss.mojo).

Callers: [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/checks/loss.mojo](../../training/checks/loss.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN51_RESIDENT_TOKEN_VALIDATION=1`.

B: incumbent, new flags absent.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- Other model upload owners are optional extensions; selected byte-LM caller is wired.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

### NN52 — Cross-entropy elementwise pass fusion

Files: [training/checks/loss.mojo](../../training/checks/loss.mojo).

Callers: [training/checks/loss.mojo](../../training/checks/loss.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN52_CE_WEIGHT_GRAD=1`.

B: incumbent, new flags absent.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- Selected complete arm is backward weights/dLogits fusion; forward shift-exp/target fusion is an additional research variant.
- Aliased/unaliased buffers, full loss semantics and complete training quality unverified.

### NN53 — Stream LM head and exact loss without full logits

Files: [training/chunked_lm_head_v2.mojo](../../training/chunked_lm_head_v2.mojo), [training/byte_lm_config.mojo](../../training/byte_lm_config.mojo), [training/chunked_lm_head_profile.mojo](../../training/chunked_lm_head_profile.mojo), [training/chunked_lm_head_gemm_host.mojo](../../training/chunked_lm_head_gemm_host.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/byte_lm_host_backward.mojo](../../training/byte_lm_host_backward.mojo), [bindings/_mojolearn_byte_lm.mojo](../../bindings/_mojolearn_byte_lm.mojo), [bindings/_mojolearn_byte_lm_host.mojo](../../bindings/_mojolearn_byte_lm_host.mojo), [python/mojolearn/_byte_lm_config.py](../../python/mojolearn/_byte_lm_config.py), [python/mojolearn/_byte_lm_host.py](../../python/mojolearn/_byte_lm_host.py), [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

Callers: [training/chunked_lm_head_v2.mojo](../../training/chunked_lm_head_v2.mojo), [training/byte_lm_config.mojo](../../training/byte_lm_config.mojo), [training/chunked_lm_head_profile.mojo](../../training/chunked_lm_head_profile.mojo), [training/chunked_lm_head_gemm_host.mojo](../../training/chunked_lm_head_gemm_host.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [training/byte_lm_host.mojo](../../training/byte_lm_host.mojo), [training/byte_lm_host_backward.mojo](../../training/byte_lm_host_backward.mojo), [bindings/_mojolearn_byte_lm.mojo](../../bindings/_mojolearn_byte_lm.mojo), [bindings/_mojolearn_byte_lm_host.mojo](../../bindings/_mojolearn_byte_lm_host.mojo), [python/mojolearn/_byte_lm_config.py](../../python/mojolearn/_byte_lm_config.py), [python/mojolearn/_byte_lm_host.py](../../python/mojolearn/_byte_lm_host.py), [tools/bench_board_neural.py](../../tools/bench_board_neural.py), [python/mojolearn/_byte_lm_impl.py](../../python/mojolearn/_byte_lm_impl.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN53_HEAD_CHUNK512=1`; runtime: `{"chunked_lm_head_v2": true, "operation": "lm_chunked_head_v2"}`.

B: runtime: `{"chunked_lm_head_v2": true, "operation": "lm_chunked_head_v2"}`.

Additional selectable variants: `panel512`, `panel2048`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json), [experiments/neural_identical_ab/lanes/lm_head_integration_inventory.json](../../experiments/neural_identical_ab/lanes/lm_head_integration_inventory.json). Existing related IDs: none separately claimed.

- Both arms set chunked_lm_head_v2=True through the actual lm-train-step workload selector. Panel width is storage only, so its profile remains the existing -lmhead-chunk256-v2 identifier.
- The existing full LM board control shape has intrinsic limits; its name full does not establish every intended target workload or corpus coverage.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.
- New selected host loss uses the normative block-hidden composition; host task scheduling performance is unmeasured.
- Full workload corpus hash, shapes, model settings, cold/repeated boundaries and all quality/performance evidence remain future work.
- No source-only claim of compilation, bitwise identity, quality or performance.

### NN54 — Versioned loss reduction across tokens and microbatches

Files: [training/neural_ab_profiles.mojo](../../training/neural_ab_profiles.mojo), [training/neural_ab_profile_contract.mojo](../../training/neural_ab_profile_contract.mojo), [training/checks/loss.mojo](../../training/checks/loss.mojo), [training/checks/loss_oracle.mojo](../../training/checks/loss_oracle.mojo), [training/loss_host_rows.mojo](../../training/loss_host_rows.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo).

Callers: [training/neural_ab_profiles.mojo](../../training/neural_ab_profiles.mojo), [training/neural_ab_profile_contract.mojo](../../training/neural_ab_profile_contract.mojo), [training/checks/loss.mojo](../../training/checks/loss.mojo), [training/checks/loss_oracle.mojo](../../training/checks/loss_oracle.mojo), [training/loss_host_rows.mojo](../../training/loss_host_rows.mojo), [training/byte_lm_host_kernels.mojo](../../training/byte_lm_host_kernels.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN54_LOSS_PROFILE=1`.

B: incumbent, new flags absent.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- Selected arm changes row-total order only; denominator and vocabulary folds remain their existing selected GEMM contract.
- Alternate leaf sizes are additional research arms, not implemented switches.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

### NN55 — Fuse optimizer post-update status production

Files: [training/neural_ab_optimizer.mojo](../../training/neural_ab_optimizer.mojo), [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

Callers: [training/neural_ab_optimizer.mojo](../../training/neural_ab_optimizer.mojo), [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN56_GROUPED_ADAM=1`; `MOJOLEARN_NN55_BLOCK_STATUS=1`.

B: `MOJOLEARN_NN56_GROUPED_ADAM=1`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: I10.

- No status or rollback execution was run; faults retain incumbent route.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

### NN56 — Batch optimizer parameter groups with per-group scalars

Files: [training/neural_ab_optimizer.mojo](../../training/neural_ab_optimizer.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

Callers: [training/neural_ab_optimizer.mojo](../../training/neural_ab_optimizer.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN56_GROUPED_ADAM=1`.

B: incumbent, new flags absent.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- The selected complete model is byte-LM AdamW; general SGD/group-API expansion remains an optional separate idea.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

### NN57 — Versioned global gradient-norm reduction

Files: [training/neural_ab_profiles.mojo](../../training/neural_ab_profiles.mojo), [training/neural_ab_profile_contract.mojo](../../training/neural_ab_profile_contract.mojo), [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo), [training/checks/optimizer_oracle.mojo](../../training/checks/optimizer_oracle.mojo), [training/clip_multi_gpu.mojo](../../training/clip_multi_gpu.mojo).

Callers: [training/neural_ab_profiles.mojo](../../training/neural_ab_profiles.mojo), [training/neural_ab_profile_contract.mojo](../../training/neural_ab_profile_contract.mojo), [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo), [training/checks/optimizer_oracle.mojo](../../training/checks/optimizer_oracle.mojo), [training/clip_multi_gpu.mojo](../../training/clip_multi_gpu.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN57_NORM_PROFILE=1`; runtime: `{"operation": "samba_training_config", "samba_max_norm": 1.0}`.

B: runtime: `{"operation": "samba_training_config", "samba_max_norm": 1.0}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- This selected v2 arm preserves the two-level tensor norm structure; a flattened norm is an additional arithmetic experiment.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

### NN58 — Fuse accumulation with gradient finishing

Files: [training/neural_ab_pointwise.mojo](../../training/neural_ab_pointwise.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo).

Callers: [training/neural_ab_pointwise.mojo](../../training/neural_ab_pointwise.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/host/samba_ops_oracle.mojo](../../training/host/samba_ops_oracle.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN58_ACCUMULATE_STATUS=1`; runtime: `{"operation": "samba_training_config", "samba_accumulation_steps": 2}`.

B: runtime: `{"operation": "samba_training_config", "samba_accumulation_steps": 2}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- The new opt-in profile explicitly refuses computed nonfinite gradients; baseline input admission is retained. NN58+NN63 avoids pretending the independent fused status win composes for free.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

### NN59 — Fuse neural dropout RNG and pointwise consumers

Files: [training/residual_dropout_contract.mojo](../../training/residual_dropout_contract.mojo), [training/residual_dropout.mojo](../../training/residual_dropout.mojo), [training/residual_dropout_host.mojo](../../training/residual_dropout_host.mojo), [training/neural_ab_pointwise.mojo](../../training/neural_ab_pointwise.mojo), [bindings/residual_dropout_common.mojo](../../bindings/residual_dropout_common.mojo), [bindings/residual_dropout_boundary.mojo](../../bindings/residual_dropout_boundary.mojo), [bindings/residual_dropout_boundary_host.mojo](../../bindings/residual_dropout_boundary_host.mojo), [python/mojolearn/_residual_dropout_impl.py](../../python/mojolearn/_residual_dropout_impl.py), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo), [python/mojolearn/training.py](../../python/mojolearn/training.py).

Callers: [training/residual_dropout_contract.mojo](../../training/residual_dropout_contract.mojo), [training/residual_dropout.mojo](../../training/residual_dropout.mojo), [training/residual_dropout_host.mojo](../../training/residual_dropout_host.mojo), [training/neural_ab_pointwise.mojo](../../training/neural_ab_pointwise.mojo), [bindings/residual_dropout_common.mojo](../../bindings/residual_dropout_common.mojo), [bindings/residual_dropout_boundary.mojo](../../bindings/residual_dropout_boundary.mojo), [bindings/residual_dropout_boundary_host.mojo](../../bindings/residual_dropout_boundary_host.mojo), [python/mojolearn/_residual_dropout_impl.py](../../python/mojolearn/_residual_dropout_impl.py), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo), [python/mojolearn/training.py](../../python/mojolearn/training.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN59_DROPOUT_RESIDUAL=1`; runtime: `{"dropout_probability": 0.1, "offset": 0, "operation": "residual_dropout_forward_vjp", "seed": 7, "stream": 0}`.

B: `MOJOLEARN_NN59_DROPOUT_RESIDUAL=1`; `MOJOLEARN_NN59_CONTROL=1`; runtime: `{"dropout_probability": 0.1, "offset": 0, "operation": "residual_dropout_forward_vjp", "seed": 7, "stream": 0}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/gemm_integration_inventory.json](../../experiments/neural_identical_ab/lanes/gemm_integration_inventory.json), [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- Explicit public residual layer; no existing transformer/CNN behavior is silently changed. Caller supplies identical p/seed/stream/offset to forward and backward.
- Existing transformer dropout refusal is preserved. This is an explicit callable layer; arbitrary CNN/transformer residual-model adoption is an additional experiment.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.
- The selected operation is a complete explicit layer and its VJP; it is not a transformer model train step or model-quality evidence.

### NN60 — Parameter/gradient views with generation-safe ownership

Files: [training/byte_lm.mojo](../../training/byte_lm.mojo).

Callers: [training/byte_lm.mojo](../../training/byte_lm.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN60_BLOCK_VIEWS=1`.

B: incumbent, new flags absent.

Additional selectable variants: `blocks`, `emb_head`, `combined`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- Other model/offload owners are separate extensions; no new default enabled.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

### NN61 — One final neural step status/readback boundary

Files: [training/byte_lm.mojo](../../training/byte_lm.mojo), [core/step_glue.mojo](../../core/step_glue.mojo), [experiments/performance_ideas/I10/native_arms.json](../../experiments/performance_ideas/I10/native_arms.json).

Callers: [training/byte_lm.mojo](../../training/byte_lm.mojo), [core/step_glue.mojo](../../core/step_glue.mojo), [experiments/performance_ideas/I10/native_arms.json](../../experiments/performance_ideas/I10/native_arms.json).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_TRAIN_LIVE_STATUS=1`; `MOJOLEARN_STEP_GLUE_TRIAL=1`.

B: `MOJOLEARN_STEP_GLUE_TRIAL=1`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: I10.

- Broader all-model drain consolidation is an optional extension.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

### NN62 — Live-range neural scratch and activation arenas

Files: [training/neural_ab_lifetime.mojo](../../training/neural_ab_lifetime.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

Callers: [training/neural_ab_lifetime.mojo](../../training/neural_ab_lifetime.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN62_LIFETIME_ARENA=1`.

B: incumbent, new flags absent.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- Other activation/stage intervals are additional arena applications, not needed to select this concrete complete-model arm.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.

### NN63 — Deterministic neural multi-device shard merge

Files: [training/neural_ab_shards.mojo](../../training/neural_ab_shards.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/accumulate_multi_gpu.mojo](../../training/accumulate_multi_gpu.mojo).

Callers: [training/neural_ab_shards.mojo](../../training/neural_ab_shards.mojo), [training/samba_ops.mojo](../../training/samba_ops.mojo), [training/accumulate_multi_gpu.mojo](../../training/accumulate_multi_gpu.mojo).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN63_CANONICAL_SHARD_MERGE=1`; runtime: `{"operation": "samba_training_config", "samba_accumulation_steps": 2}`.

B: runtime: `{"operation": "samba_training_config", "samba_accumulation_steps": 2}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- No new collective/transport backend. Different partition/topology recipes must preserve logical leaves; no count-invariance evidence claimed.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.
- This recipe reaches canonical accumulation on one device. Existing training/accumulate_multi_gpu.mojo owns multi-device transport; topology coverage remains a separate future workload.

### NN64 — Neural inference state batching with isolated sessions

Files: [training/neural_ab_lifetime.mojo](../../training/neural_ab_lifetime.mojo), [training/neural_session_mlp.mojo](../../training/neural_session_mlp.mojo), [training/neural_session_mlp_host.mojo](../../training/neural_session_mlp_host.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo), [python/mojolearn/_training_impl.py](../../python/mojolearn/_training_impl.py), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py).

Callers: [training/neural_ab_lifetime.mojo](../../training/neural_ab_lifetime.mojo), [training/neural_session_mlp.mojo](../../training/neural_session_mlp.mojo), [training/neural_session_mlp_host.mojo](../../training/neural_session_mlp_host.mojo), [bindings/_mojolearn_training.mojo](../../bindings/_mojolearn_training.mojo), [bindings/_mojolearn_training_host.mojo](../../bindings/_mojolearn_training_host.mojo), [python/mojolearn/_training_impl.py](../../python/mojolearn/_training_impl.py), [python/mojolearn/_mlp_impl.py](../../python/mojolearn/_mlp_impl.py).

Workload/selector files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py).

A: `MOJOLEARN_NN64_SESSION_PACK=1`; runtime: `{"operation": "mlp_inference_sessions", "session_count": 4}`.

B: runtime: `{"operation": "mlp_inference_sessions", "session_count": 4}`.

Arm/caller notes: [experiments/neural_identical_ab/lanes/training_integration_inventory.json](../../experiments/neural_identical_ab/lanes/training_integration_inventory.json). Existing related IDs: none separately claimed.

- Selected arm is stateless feedforward inference: no cache or RNG exists to merge. Stateful decode/cache batching remains a separate unimplemented research direction in the broad idea list.
- Compilation, same-version cross-vendor identity, quality evaluation and end-to-end timing were not run by request.
- The saved MLP input fixture is reused in full through the GPU mlp-forward lane. The explicit operation uses the selected training binding. The CPU-only MLPInference lane remains separate; host arithmetic qualification needs its own matching binding recipe.

## Existing I/A/N/F manifest experiments

Statuses below are retained metadata from before this continuation, not newly verified results. Read each manifest's timing contract, coverage and retained failures. A build status or component result does not establish full-dataset model qualification. F entries are Apple FAST references. A/N entries retain their original vendor scope; they are not automatically four-column numerical profiles.

### A01 — Isolate AMD smaller MFMA and band routing

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/A01/manifest.json](../../experiments/performance_ideas/A01/manifest.json), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo), [experiments/performance_ideas/A01/campaign.json](../../experiments/performance_ideas/A01/campaign.json), [experiments/performance_ideas/A01/compile_matrix.json](../../experiments/performance_ideas/A01/compile_matrix.json).

A defines: `MOJOLEARN_IDN_GEMM_MFMA16_OFF=1`, `MOJOLEARN_IDN_GEMM_AMD_BAND_MFMA_OFF=1`.

B defines: none in manifest.

### A02 — Compare production one/two-page staging and bounded one/two/four-plane resource controls

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/A02/manifest.json](../../experiments/performance_ideas/A02/manifest.json), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo), [experiments/performance_ideas/A02/check.mojo](../../experiments/performance_ideas/A02/check.mojo), [gemm/experiments/bounded_staging.mojo](../../gemm/experiments/bounded_staging.mojo), [gemm/experiments/bounded_staging_check.mojo](../../gemm/experiments/bounded_staging_check.mojo), [experiments/performance_ideas/A02/campaign.json](../../experiments/performance_ideas/A02/campaign.json), [experiments/performance_ideas/A02/compile_matrix.json](../../experiments/performance_ideas/A02/compile_matrix.json).

A defines: `MOJOLEARN_GEMM_ONE_PAGE=1`.

B defines: none in manifest.

### A03 — Vary vector-aligned LDS operand strides

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/A03/manifest.json](../../experiments/performance_ideas/A03/manifest.json), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo), [experiments/performance_ideas/A03/campaign.json](../../experiments/performance_ideas/A03/campaign.json), [experiments/performance_ideas/A03/compile_matrix.json](../../experiments/performance_ideas/A03/compile_matrix.json).

A defines: `MOJOLEARN_IDN_GEMM_LDS_PAD_WORDS=0`, `MOJOLEARN_IDN_GEMM_LDS_PAD_WORDS=8`.

B defines: none in manifest.

### A04 — Pair independent logical groups with supported XOR membership

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/A04/manifest.json](../../experiments/performance_ideas/A04/manifest.json), [gemm/experiments/subwave_membership.mojo](../../gemm/experiments/subwave_membership.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [experiments/performance_ideas/A04/compile_matrix.json](../../experiments/performance_ideas/A04/compile_matrix.json), [experiments/performance_ideas/A04/time.mojo](../../experiments/performance_ideas/A04/time.mojo).

A defines: none in manifest.

B defines: none in manifest.

### A05 — Halve the packed body row accumulator live range

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/A05/manifest.json](../../experiments/performance_ideas/A05/manifest.json), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo), [experiments/performance_ideas/A05/campaign.json](../../experiments/performance_ideas/A05/campaign.json), [experiments/performance_ideas/A05/compile_matrix.json](../../experiments/performance_ideas/A05/compile_matrix.json).

A defines: `MOJOLEARN_IDN_GEMM_COMPACT_LIVE_TILE=1`.

B defines: none in manifest.

### A06 — Qualify AMD exact fused low-dimensional neighbor selection

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/A06/manifest.json](../../experiments/performance_ideas/A06/manifest.json), [experiments/performance_ideas/A06/check.mojo](../../experiments/performance_ideas/A06/check.mojo), [neighbors/checks/fused_slot_merge_check.mojo](../../neighbors/checks/fused_slot_merge_check.mojo), [neighbors/estimator.mojo](../../neighbors/estimator.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [experiments/performance_ideas/A06/compile_matrix.json](../../experiments/performance_ideas/A06/compile_matrix.json), [experiments/performance_ideas/A06/time.mojo](../../experiments/performance_ideas/A06/time.mojo).

A defines: none in manifest.

B defines: none in manifest.

### A07 — Qualify bounded degree graph and exact integer node histogram tasks

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/A07/manifest.json](../../experiments/performance_ideas/A07/manifest.json), [experiments/performance_ideas/A07/check.mojo](../../experiments/performance_ideas/A07/check.mojo), [experiments/performance_ideas/A07/histogram_check.mojo](../../experiments/performance_ideas/A07/histogram_check.mojo), [experiments/performance_ideas/A07/histogram_tasks.mojo](../../experiments/performance_ideas/A07/histogram_tasks.mojo), [experiments/performance_ideas/N07/streamed_histogram.mojo](../../experiments/performance_ideas/N07/streamed_histogram.mojo), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [neighbors/checks/ball_cover_canonical_order.mojo](../../neighbors/checks/ball_cover_canonical_order.mojo), [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo), [experiments/performance_ideas/A07/production_check.mojo](../../experiments/performance_ideas/A07/production_check.mojo), [experiments/performance_ideas/A07/campaign.json](../../experiments/performance_ideas/A07/campaign.json), [experiments/performance_ideas/A07/compile_matrix.json](../../experiments/performance_ideas/A07/compile_matrix.json), [experiments/performance_ideas/A07/time.mojo](../../experiments/performance_ideas/A07/time.mojo).

A defines: `MOJOLEARN_RBC_CANON_MERGE=1`, `MOJOLEARN_RBC_CANON_DEGREE_BUCKETS=1`.

B defines: none in manifest.

### A08 — Requalify promoted 256-row accumulation with fit interactions

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/A08/manifest.json](../../experiments/performance_ideas/A08/manifest.json), [experiments/performance_ideas/A08/check.mojo](../../experiments/performance_ideas/A08/check.mojo), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [cluster/checks/reduce_by_key.mojo](../../cluster/checks/reduce_by_key.mojo), [cluster/impl/detail/kmeans.mojo](../../cluster/impl/detail/kmeans.mojo), [experiments/performance_ideas/A08/campaign.json](../../experiments/performance_ideas/A08/campaign.json), [experiments/performance_ideas/A08/compile_matrix.json](../../experiments/performance_ideas/A08/compile_matrix.json), [experiments/performance_ideas/A08/time.mojo](../../experiments/performance_ideas/A08/time.mojo).

A defines: `MOJOLEARN_IDN_KMEANS_ACC_ROWS_256_OFF=1`, `MOJOLEARN_IDN_KMEANS_ACC_ROWS_4096=1`, `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_1=1`, `MOJOLEARN_IDN_KMEANS_ACC_ROWS_256_OFF=1`, `MOJOLEARN_IDN_KMEANS_CONV_CHUNK_1=1`.

B defines: none in manifest.

### F01 — Actual PCA caller GEMM geometry

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F01/manifest.json](../../experiments/performance_ideas/F01/manifest.json), [experiments/performance_ideas/F01/caller.py](../../experiments/performance_ideas/F01/caller.py), [experiments/performance_ideas/apple_fast/pair.py](../../experiments/performance_ideas/apple_fast/pair.py), [experiments/apple_fast/gemm/scoped_dispatch.mojo](../../experiments/apple_fast/gemm/scoped_dispatch.mojo), [experiments/performance_ideas/F01/compile_audit.json](../../experiments/performance_ideas/F01/compile_audit.json).

A defines: `MOJOLEARN_SCOPED_GEMM_AUDIT`, `MOJOLEARN_SCOPED_GEMM_G1_TALL`, `MOJOLEARN_SCOPED_GEMM_G1_DENSE`, `MOJOLEARN_SCOPED_GEMM_G1_GRAM`, `MOJOLEARN_SCOPED_GEMM_G2_NARROW`, `MOJOLEARN_SCOPED_GEMM_SPLIT`, `MOJOLEARN_SCOPED_GEMM_PCA`.

B defines: `MOJOLEARN_SCOPED_GEMM_AUDIT`.

### F02 — Share GEMM infrastructure through independent fused and unfused caller adapters

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F02/manifest.json](../../experiments/performance_ideas/F02/manifest.json), [experiments/apple_fast/gemm/scoped_dispatch.mojo](../../experiments/apple_fast/gemm/scoped_dispatch.mojo), [x_decomp/lu_fast_mma.mojo](../../x_decomp/lu_fast_mma.mojo), [experiments/performance_ideas/F02/caller.py](../../experiments/performance_ideas/F02/caller.py), [cholesky/checks/fast_trsm.mojo](../../cholesky/checks/fast_trsm.mojo), [bindings/_mojolearn_gp.mojo](../../bindings/_mojolearn_gp.mojo), [bindings/_mojolearn_x_decomp.mojo](../../bindings/_mojolearn_x_decomp.mojo), [core/gemm.mojo](../../core/gemm.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo), [bindings/_mojolearn_scoped_gemm_probe.mojo](../../bindings/_mojolearn_scoped_gemm_probe.mojo), [experiments/performance_ideas/F02/compile_audit.json](../../experiments/performance_ideas/F02/compile_audit.json).

A defines: `MOJOLEARN_LU_FAST_SHARED_SUB`.

B defines: none in manifest.

### F03 — Actual LM bounded resident session versus stateless calls

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F03/manifest.json](../../experiments/performance_ideas/F03/manifest.json), [experiments/performance_ideas/F03/caller.py](../../experiments/performance_ideas/F03/caller.py), [experiments/performance_ideas/apple_fast/lm_task.py](../../experiments/performance_ideas/apple_fast/lm_task.py), [bindings/_mojolearn_byte_lm.mojo](../../bindings/_mojolearn_byte_lm.mojo), [experiments/performance_ideas/F03/compile_audit.json](../../experiments/performance_ideas/F03/compile_audit.json).

A defines: none in manifest.

B defines: none in manifest.

### F04 — Pair independent gather completions with owned buffers

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F04/manifest.json](../../experiments/performance_ideas/F04/manifest.json), [resample/estimator.mojo](../../resample/estimator.mojo), [experiments/performance_ideas/F04/caller.py](../../experiments/performance_ideas/F04/caller.py), [experiments/performance_ideas/F04/compile_audit.json](../../experiments/performance_ideas/F04/compile_audit.json).

A defines: `MOJOLEARN_RESAMPLE_FAST_GATHER`, `MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR`.

B defines: `MOJOLEARN_RESAMPLE_FAST_GATHER`.

### F05 — Narrow softmax at full optimizer and line-search caller

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F05/manifest.json](../../experiments/performance_ideas/F05/manifest.json), [experiments/performance_ideas/F05/caller.py](../../experiments/performance_ideas/F05/caller.py), [experiments/apple_fast/gemm/softmax_narrow.mojo](../../experiments/apple_fast/gemm/softmax_narrow.mojo), [experiments/performance_ideas/F05/compile_audit.json](../../experiments/performance_ideas/F05/compile_audit.json).

A defines: `MOJOLEARN_SOFTMAX_FAST_G2_NARROW`, `MOJOLEARN_SOFTMAX_G2_AUDIT`.

B defines: `MOJOLEARN_SOFTMAX_G2_AUDIT`.

### F06 — Independent MCD batch bounds, active-candidate compaction and exact-support covariance reuse

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F06/manifest.json](../../experiments/performance_ideas/F06/manifest.json), [x_decomp/mcd_bmma.mojo](../../x_decomp/mcd_bmma.mojo), [bindings/_mojolearn_x_decomp.mojo](../../bindings/_mojolearn_x_decomp.mojo), [experiments/performance_ideas/F06/caller.py](../../experiments/performance_ideas/F06/caller.py), [x_decomp/mcd_fast.mojo](../../x_decomp/mcd_fast.mojo), [x_decomp/mcd_experiments.mojo](../../x_decomp/mcd_experiments.mojo), [experiments/performance_ideas/F06/native_check.mojo](../../experiments/performance_ideas/F06/native_check.mojo), [experiments/performance_ideas/F06/compile_audit.json](../../experiments/performance_ideas/F06/compile_audit.json), [experiments/performance_ideas/F06/coverage.md](../../experiments/performance_ideas/F06/coverage.md).

A defines: `MOJOLEARN_MCD_FAST_BOUND_BATCH`.

B defines: none in manifest.

### F07 — FLASH and true GQA caller qualification with reach counters

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F07/manifest.json](../../experiments/performance_ideas/F07/manifest.json), [transformer/impl/llama/afn_apple_fast.mojo](../../transformer/impl/llama/afn_apple_fast.mojo), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo), [experiments/performance_ideas/F07/caller.py](../../experiments/performance_ideas/F07/caller.py), [experiments/performance_ideas/F07/compile_audit.json](../../experiments/performance_ideas/F07/compile_audit.json).

A defines: `MOJOLEARN_AFN_ATTN_FLASH`, `MOJOLEARN_AFN_ATTN_AUDIT`.

B defines: `MOJOLEARN_AFN_ATTN_AUDIT`.

### F08 — Independent LM backward fusion and view A/B training task

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F08/manifest.json](../../experiments/performance_ideas/F08/manifest.json), [experiments/performance_ideas/F08/caller.py](../../experiments/performance_ideas/F08/caller.py), [experiments/performance_ideas/apple_fast/lm_task.py](../../experiments/performance_ideas/apple_fast/lm_task.py), [training/byte_lm_afn.mojo](../../training/byte_lm_afn.mojo), [training/byte_lm_afn_grad.mojo](../../training/byte_lm_afn_grad.mojo), [experiments/performance_ideas/F08/compile_audit.json](../../experiments/performance_ideas/F08/compile_audit.json).

A defines: `MOJOLEARN_AFN_LM_BWD_NOSYNC`.

B defines: `MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF`.

### F09 — Memory-bounded LM head in a fixed actual SGD task

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F09/manifest.json](../../experiments/performance_ideas/F09/manifest.json), [experiments/performance_ideas/F09/caller.py](../../experiments/performance_ideas/F09/caller.py), [training/chunked_lm_head_v2.mojo](../../training/chunked_lm_head_v2.mojo), [training/afn_optim.mojo](../../training/afn_optim.mojo), [experiments/performance_ideas/F09/compile_audit.json](../../experiments/performance_ideas/F09/compile_audit.json).

A defines: none in manifest.

B defines: none in manifest.

### F10 — Independent SSD MMA and Mamba fusion caller arms

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F10/manifest.json](../../experiments/performance_ideas/F10/manifest.json), [experiments/performance_ideas/F10/caller.py](../../experiments/performance_ideas/F10/caller.py), [mamba/impl/modules/afn_ssd_mma.mojo](../../mamba/impl/modules/afn_ssd_mma.mojo), [mamba/impl/modeling/afn_mamba1_fused.mojo](../../mamba/impl/modeling/afn_mamba1_fused.mojo), [experiments/performance_ideas/F10/compile_audit.json](../../experiments/performance_ideas/F10/compile_audit.json).

A defines: `MOJOLEARN_AFN_MAMBA2_SSD_MMA`.

B defines: none in manifest.

### F11 — Apple FAST opt-in TSQR norm and grid scheduling

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F11/manifest.json](../../experiments/performance_ideas/F11/manifest.json), [x_decomp/tsqr_device.mojo](../../x_decomp/tsqr_device.mojo), [experiments/performance_ideas/F11/caller.py](../../experiments/performance_ideas/F11/caller.py), [decomposition/impl/linalg/detail/pca.mojo](../../decomposition/impl/linalg/detail/pca.mojo), [bindings/_mojolearn_estimators.mojo](../../bindings/_mojolearn_estimators.mojo), [experiments/performance_ideas/F11/compile_audit.json](../../experiments/performance_ideas/F11/compile_audit.json).

A defines: `MOJOLEARN_DECOMP_FAST_TSQR_NORM`.

B defines: none in manifest.

### F12 — Qualify resident boosting partition and leaf-input reuse

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F12/manifest.json](../../experiments/performance_ideas/F12/manifest.json), [experiments/performance_ideas/F12/caller.py](../../experiments/performance_ideas/F12/caller.py), [gbdt/methods/sym_iter_fast.mojo](../../gbdt/methods/sym_iter_fast.mojo), [gbdt/methods/leaves_estimation/apple_fast_est.mojo](../../gbdt/methods/leaves_estimation/apple_fast_est.mojo), [gbdt/methods/ordered_fast_switches.mojo](../../gbdt/methods/ordered_fast_switches.mojo), [gbdt/methods/pointwise_scores_calcer.mojo](../../gbdt/methods/pointwise_scores_calcer.mojo), [gbdt/methods/pointwise_kernels.mojo](../../gbdt/methods/pointwise_kernels.mojo), [gbdt/methods/kernel/compute_point_hist2_loop.mojo](../../gbdt/methods/kernel/compute_point_hist2_loop.mojo), [experiments/performance_ideas/F12/compile_audit.json](../../experiments/performance_ideas/F12/compile_audit.json).

A defines: `MOJOLEARN_SYM_REUSE_PARTITION`.

B defines: none in manifest.

### F13 — FAST-only shared row forest layout at divergent public callers

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F13/manifest.json](../../experiments/performance_ideas/F13/manifest.json), [core/forest_inference.mojo](../../core/forest_inference.mojo), [experiments/performance_ideas/F13/caller.py](../../experiments/performance_ideas/F13/caller.py), [xtrees/shap_device.mojo](../../xtrees/shap_device.mojo), [bindings/_mojolearn_trees.mojo](../../bindings/_mojolearn_trees.mojo), [bindings/_mojolearn_x_trees.mojo](../../bindings/_mojolearn_x_trees.mojo), [experiments/performance_ideas/F13/compile_audit.json](../../experiments/performance_ideas/F13/compile_audit.json).

A defines: `MOJOLEARN_FOREST_FAST_SHARED_ROWS`.

B defines: none in manifest.

### F14 — Compare exact grouped query rows and fixed-probe bounded IVF list tasks

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F14/manifest.json](../../experiments/performance_ideas/F14/manifest.json), [neighbors/impl/detail/fast_topk_knn.mojo](../../neighbors/impl/detail/fast_topk_knn.mojo), [experiments/performance_ideas/F14/caller.py](../../experiments/performance_ideas/F14/caller.py), [ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo](../../ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo), [ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo](../../ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo), [experiments/performance_ideas/F14/ann_caller.py](../../experiments/performance_ideas/F14/ann_caller.py), [experiments/performance_ideas/F14/compile_audit.json](../../experiments/performance_ideas/F14/compile_audit.json).

A defines: `MOJOLEARN_KNN_FAST_QUERY_GROUP4`.

B defines: none in manifest.

### F15 — Independent mini-batch labeling and bounded stopping work

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F15/manifest.json](../../experiments/performance_ideas/F15/manifest.json), [x_cluster/minibatch_fast.mojo](../../x_cluster/minibatch_fast.mojo), [experiments/performance_ideas/F15/caller.py](../../experiments/performance_ideas/F15/caller.py), [dbscan/estimator.mojo](../../dbscan/estimator.mojo), [hdbscan/impl/detail/fast_apple.mojo](../../hdbscan/impl/detail/fast_apple.mojo), [experiments/performance_ideas/F15/compile_audit.json](../../experiments/performance_ideas/F15/compile_audit.json).

A defines: `MOJOLEARN_X_CLUSTER_FAST_W2_MBK_LABRG`.

B defines: none in manifest.

### F16 — Integrate compensated production Kalman tail with independent gradient prerequisite

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F16/manifest.json](../../experiments/performance_ideas/F16/manifest.json), [arima/impl/fast_eval_ws.mojo](../../arima/impl/fast_eval_ws.mojo), [arima/impl/fast_eval_df.mojo](../../arima/impl/fast_eval_df.mojo), [arima/checks/product_df_quality_check.mojo](../../arima/checks/product_df_quality_check.mojo), [experiments/performance_ideas/F16/caller.py](../../experiments/performance_ideas/F16/caller.py), [experiments/performance_ideas/apple_fast/arima_task.py](../../experiments/performance_ideas/apple_fast/arima_task.py), [experiments/performance_ideas/F16/compile_audit.json](../../experiments/performance_ideas/F16/compile_audit.json), [experiments/performance_ideas/F16/compile_matrix.json](../../experiments/performance_ideas/F16/compile_matrix.json), [experiments/performance_ideas/F16/prerequisite.py](../../experiments/performance_ideas/F16/prerequisite.py).

A defines: `MOJOLEARN_ARIMA_FAST_PRODUCT_DF_TAIL=1`.

B defines: none in manifest.

### F17 — Actual panel search A/B for stacked candidates and held evaluation state

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F17/manifest.json](../../experiments/performance_ideas/F17/manifest.json), [experiments/performance_ideas/F17/caller.py](../../experiments/performance_ideas/F17/caller.py), [experiments/performance_ideas/apple_fast/arima_task.py](../../experiments/performance_ideas/apple_fast/arima_task.py), [arima/impl/batched_arima.mojo](../../arima/impl/batched_arima.mojo), [arima/impl/fast_eval_ws.mojo](../../arima/impl/fast_eval_ws.mojo), [experiments/performance_ideas/F17/compile_audit.json](../../experiments/performance_ideas/F17/compile_audit.json).

A defines: none in manifest.

B defines: `MOJOLEARN_ARIMA_FAST_BATCH_GRAD_OFF`.

### F18 — Resident KDE immutable input ownership and direct preparation

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F18/manifest.json](../../experiments/performance_ideas/F18/manifest.json), [kde/resident_fit.mojo](../../kde/resident_fit.mojo), [python/mojolearn/density.py](../../python/mojolearn/density.py), [bindings/_mojolearn_estimators.mojo](../../bindings/_mojolearn_estimators.mojo), [experiments/performance_ideas/F18/caller.py](../../experiments/performance_ideas/F18/caller.py), [experiments/performance_ideas/F18/compile_audit.json](../../experiments/performance_ideas/F18/compile_audit.json).

A defines: `MOJOLEARN_KDE_FAST_DIRECT_PREP`.

B defines: `MOJOLEARN_KDE_FAST_DIRECT_PREP_OFF`.

### F19 — Coalesced tiled gather with GPU-generated indices and direct owned outputs

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F19/manifest.json](../../experiments/performance_ideas/F19/manifest.json), [resample/gather_fast.mojo](../../resample/gather_fast.mojo), [resample/estimator.mojo](../../resample/estimator.mojo), [experiments/performance_ideas/F19/caller.py](../../experiments/performance_ideas/F19/caller.py), [experiments/performance_ideas/F04/caller.py](../../experiments/performance_ideas/F04/caller.py), [bindings/_mojolearn_resample.mojo](../../bindings/_mojolearn_resample.mojo), [python/mojolearn/resample.py](../../python/mojolearn/resample.py), [experiments/performance_ideas/F19/compile_audit.json](../../experiments/performance_ideas/F19/compile_audit.json).

A defines: `MOJOLEARN_RESAMPLE_FAST_GATHER`, `MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR`, `MOJOLEARN_RESAMPLE_FAST_TILED_GATHER`.

B defines: `MOJOLEARN_RESAMPLE_FAST_GATHER`, `MOJOLEARN_RESAMPLE_FAST_WAIT_PAIR`.

### F20 — Task-level optimizer fusion and FAST blocked LayerNorm qualification

Mode `fast`; vendors apple; retained status `source_ready`.

Files: [experiments/performance_ideas/F20/manifest.json](../../experiments/performance_ideas/F20/manifest.json), [sequence/layernorm.mojo](../../sequence/layernorm.mojo), [training/afn_optim.mojo](../../training/afn_optim.mojo), [experiments/performance_ideas/F20/caller.py](../../experiments/performance_ideas/F20/caller.py), [experiments/performance_ideas/F20/compile_audit.json](../../experiments/performance_ideas/F20/compile_audit.json).

A defines: `MOJOLEARN_AFN_OPT_MULTITENSOR`.

B defines: none in manifest.

### I01 — Attribute existing GEMM schedules

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/I01/manifest.json](../../experiments/performance_ideas/I01/manifest.json), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [experiments/performance_ideas/I01/campaign.json](../../experiments/performance_ideas/I01/campaign.json), [experiments/performance_ideas/I01/compile_matrix.json](../../experiments/performance_ideas/I01/compile_matrix.json).

A defines: `MOJOLEARN_IDN_GEMM_GROUP_SLACK_2=1`.

B defines: none in manifest.

### I02 — Bound session GEMM scratch retention

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/I02/manifest.json](../../experiments/performance_ideas/I02/manifest.json), [gemm/experiments/bounded_workspace.mojo](../../gemm/experiments/bounded_workspace.mojo), [gemm/experiments/bounded_workspace_check.mojo](../../gemm/experiments/bounded_workspace_check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [experiments/performance_ideas/I02/compile_matrix.json](../../experiments/performance_ideas/I02/compile_matrix.json), [experiments/performance_ideas/I02/time.mojo](../../experiments/performance_ideas/I02/time.mojo).

A defines: `MOJOLEARN_STEP_PHASE_TIMERS=1`.

B defines: none in manifest.

### I03 — Batch independent products on separate grid jobs

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/I03/manifest.json](../../experiments/performance_ideas/I03/manifest.json), [gemm/experiments/grouped_jobs.mojo](../../gemm/experiments/grouped_jobs.mojo), [gemm/experiments/grouped_jobs_check.mojo](../../gemm/experiments/grouped_jobs_check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [experiments/performance_ideas/I03/compile_matrix.json](../../experiments/performance_ideas/I03/compile_matrix.json).

A defines: none in manifest.

B defines: none in manifest.

### I04 — Qualify a coherently shared opt-in 64-element IDENTICAL leaf version

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/I04/manifest.json](../../experiments/performance_ideas/I04/manifest.json), [gemm/contract.mojo](../../gemm/contract.mojo), [gemm/experiments/fold_profile_probe.mojo](../../gemm/experiments/fold_profile_probe.mojo), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/experiments/profile_identity_check.mojo](../../gemm/experiments/profile_identity_check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [experiments/performance_ideas/I04/compile_matrix.json](../../experiments/performance_ideas/I04/compile_matrix.json), [experiments/performance_ideas/I04/time.mojo](../../experiments/performance_ideas/I04/time.mojo).

A defines: `MOJOLEARN_IDN_GEMM_FOLD_LEAF_64=1`.

B defines: none in manifest.

### I05 — Fuse bias after the explicit rounded product seam

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/I05/manifest.json](../../experiments/performance_ideas/I05/manifest.json), [gemm/experiments/rounded_epilogue.mojo](../../gemm/experiments/rounded_epilogue.mojo), [gemm/experiments/rounded_epilogue_check.mojo](../../gemm/experiments/rounded_epilogue_check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [experiments/performance_ideas/I05/compile_matrix.json](../../experiments/performance_ideas/I05/compile_matrix.json).

A defines: none in manifest.

B defines: none in manifest.

### I06 — qualify attention KV grid reuse across GQA tails

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I06/manifest.json](../../experiments/performance_ideas/I06/manifest.json), [experiments/performance_ideas/I06/check.mojo](../../experiments/performance_ideas/I06/check.mojo), [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo), [experiments/performance_ideas/I06/compile_matrix.json](../../experiments/performance_ideas/I06/compile_matrix.json), [experiments/performance_ideas/I06/coverage.md](../../experiments/performance_ideas/I06/coverage.md), [experiments/performance_ideas/I06/native_arms.json](../../experiments/performance_ideas/I06/native_arms.json).

A defines: `MOJOLEARN_ATTN_ARM_TRIAL=1`, `MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE=1`.

B defines: `MOJOLEARN_ATTN_ARM_TRIAL=1`.

### I07 — exercise retained and recomputed attention backward lifetimes

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I07/manifest.json](../../experiments/performance_ideas/I07/manifest.json), [experiments/performance_ideas/I07/check.mojo](../../experiments/performance_ideas/I07/check.mojo), [experiments/performance_ideas/I07/state_cost.mojo](../../experiments/performance_ideas/I07/state_cost.mojo), [experiments/performance_ideas/I07/compile_matrix.json](../../experiments/performance_ideas/I07/compile_matrix.json), [experiments/performance_ideas/I07/coverage.md](../../experiments/performance_ideas/I07/coverage.md), [experiments/performance_ideas/I07/native_arms.json](../../experiments/performance_ideas/I07/native_arms.json).

A defines: `MOJOLEARN_ATTN_ARM_TRIAL=1`.

B defines: `MOJOLEARN_ATTN_ARM_TRIAL=1`, `MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD=1`.

### I08 — attribute SSD tile reuse on multiple state cases and chunk tails

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I08/manifest.json](../../experiments/performance_ideas/I08/manifest.json), [experiments/performance_ideas/I08/check.mojo](../../experiments/performance_ideas/I08/check.mojo), [mamba/impl/modules/ssd_minimal.mojo](../../mamba/impl/modules/ssd_minimal.mojo), [experiments/performance_ideas/I08/backward_check.mojo](../../experiments/performance_ideas/I08/backward_check.mojo), [mamba/impl/modules/mamba2_prefill_backward.mojo](../../mamba/impl/modules/mamba2_prefill_backward.mojo), [experiments/performance_ideas/I08/compile_matrix.json](../../experiments/performance_ideas/I08/compile_matrix.json), [experiments/performance_ideas/I08/coverage.md](../../experiments/performance_ideas/I08/coverage.md), [experiments/performance_ideas/I08/native_arms.json](../../experiments/performance_ideas/I08/native_arms.json), [experiments/performance_ideas/I08/time.mojo](../../experiments/performance_ideas/I08/time.mojo).

A defines: `MOJOLEARN_IDN_M2_RETAIN_GL=1`.

B defines: none in manifest.

### I09 — gate token-parallel recurrence on prefix and decode boundaries

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I09/manifest.json](../../experiments/performance_ideas/I09/manifest.json), [experiments/performance_ideas/I09/check.mojo](../../experiments/performance_ideas/I09/check.mojo), [experiments/performance_ideas/I09/compile_matrix.json](../../experiments/performance_ideas/I09/compile_matrix.json), [experiments/performance_ideas/I09/coverage.md](../../experiments/performance_ideas/I09/coverage.md), [experiments/performance_ideas/I09/native_arms.json](../../experiments/performance_ideas/I09/native_arms.json), [experiments/performance_ideas/I09/time.mojo](../../experiments/performance_ideas/I09/time.mojo).

A defines: none in manifest.

B defines: `MOJOLEARN_IDN_MAMBA_CONV_CELL_OFF=1`.

### I10 — reduce ordered training status from per-tile contributions

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I10/manifest.json](../../experiments/performance_ideas/I10/manifest.json), [experiments/performance_ideas/I10/check.mojo](../../experiments/performance_ideas/I10/check.mojo), [training/checks/optimizer.mojo](../../training/checks/optimizer.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo), [experiments/performance_ideas/I10/compile_matrix.json](../../experiments/performance_ideas/I10/compile_matrix.json), [experiments/performance_ideas/I10/coverage.md](../../experiments/performance_ideas/I10/coverage.md), [experiments/performance_ideas/I10/native_arms.json](../../experiments/performance_ideas/I10/native_arms.json), [experiments/performance_ideas/I10/time.mojo](../../experiments/performance_ideas/I10/time.mojo).

A defines: `MOJOLEARN_TRAIN_LIVE_STATUS=1`, `MOJOLEARN_STEP_GLUE_TRIAL=1`.

B defines: `MOJOLEARN_STEP_GLUE_TRIAL=1`.

### I11 — qualify canonical radix embedding updates under skew and vocabulary reuse

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I11/manifest.json](../../experiments/performance_ideas/I11/manifest.json), [experiments/performance_ideas/I11/check.mojo](../../experiments/performance_ideas/I11/check.mojo), [experiments/performance_ideas/I11/compile_matrix.json](../../experiments/performance_ideas/I11/compile_matrix.json), [experiments/performance_ideas/I11/coverage.md](../../experiments/performance_ideas/I11/coverage.md), [experiments/performance_ideas/I11/native_arms.json](../../experiments/performance_ideas/I11/native_arms.json), [experiments/performance_ideas/I11/time.mojo](../../experiments/performance_ideas/I11/time.mojo).

A defines: none in manifest.

B defines: `MOJOLEARN_IDN_EMB_RADIX_SORT_OFF=1`.

### I12 — gate solver pass fusion on accepted iterates and refusal semantics

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I12/manifest.json](../../experiments/performance_ideas/I12/manifest.json), [experiments/performance_ideas/I12/check.mojo](../../experiments/performance_ideas/I12/check.mojo), [glm/impl/qn/glm_base.mojo](../../glm/impl/qn/glm_base.mojo), [glm/impl/qn/qn_linesearch.mojo](../../glm/impl/qn/qn_linesearch.mojo), [experiments/performance_ideas/I12/trials_check.mojo](../../experiments/performance_ideas/I12/trials_check.mojo), [experiments/performance_ideas/I12/sgd_check.mojo](../../experiments/performance_ideas/I12/sgd_check.mojo), [x_linear/device.mojo](../../x_linear/device.mojo), [x_linear/sgd.mojo](../../x_linear/sgd.mojo), [experiments/performance_ideas/I12/compile_matrix.json](../../experiments/performance_ideas/I12/compile_matrix.json), [experiments/performance_ideas/I12/coverage.md](../../experiments/performance_ideas/I12/coverage.md), [experiments/performance_ideas/I12/native_arms.json](../../experiments/performance_ideas/I12/native_arms.json), [experiments/performance_ideas/I12/time.mojo](../../experiments/performance_ideas/I12/time.mojo).

A defines: `MOJOLEARN_IDN_QN_EXACT_TRIALS=1`.

B defines: none in manifest.

### I13 — implement compact degree buckets for bounded RBC stable merges

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I13/manifest.json](../../experiments/performance_ideas/I13/manifest.json), [experiments/performance_ideas/I13/check.mojo](../../experiments/performance_ideas/I13/check.mojo), [neighbors/checks/ball_cover_canonical_order.mojo](../../neighbors/checks/ball_cover_canonical_order.mojo), [neighbors/checks/rbc_canonical_merge_check.mojo](../../neighbors/checks/rbc_canonical_merge_check.mojo), [experiments/performance_ideas/I13/compile_matrix.json](../../experiments/performance_ideas/I13/compile_matrix.json), [experiments/performance_ideas/I13/coverage.md](../../experiments/performance_ideas/I13/coverage.md), [experiments/performance_ideas/I13/native_arms.json](../../experiments/performance_ideas/I13/native_arms.json), [experiments/performance_ideas/I13/time.mojo](../../experiments/performance_ideas/I13/time.mojo).

A defines: `MOJOLEARN_RBC_CANON_DEGREE_BUCKETS=1`.

B defines: `MOJOLEARN_RBC_CANON_MERGE=1`.

### I14 — qualify gated convergence at first fixed point on graph topologies

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I14/manifest.json](../../experiments/performance_ideas/I14/manifest.json), [experiments/performance_ideas/I14/check.mojo](../../experiments/performance_ideas/I14/check.mojo), [experiments/performance_ideas/I14/hdbscan_stages.mojo](../../experiments/performance_ideas/I14/hdbscan_stages.mojo), [experiments/performance_ideas/I14/compile_matrix.json](../../experiments/performance_ideas/I14/compile_matrix.json), [experiments/performance_ideas/I14/coverage.md](../../experiments/performance_ideas/I14/coverage.md), [experiments/performance_ideas/I14/native_arms.json](../../experiments/performance_ideas/I14/native_arms.json), [experiments/performance_ideas/I14/time.mojo](../../experiments/performance_ideas/I14/time.mojo).

A defines: `MOJOLEARN_IDN_DBSCAN_CC_CHUNK16=1`.

B defines: `MOJOLEARN_IDN_DBSCAN_CC_GATED_OFF=1`, `MOJOLEARN_IDN_HDB_ONE_SYNC_OFF=1`, `MOJOLEARN_IDN_HDB_CONDENSE_TWO_READS_OFF=1`, `MOJOLEARN_IDN_HDB_SELECT_ONE_READ_OFF=1`, `MOJOLEARN_IDN_HDB_MR_FUSED_GUARD_OFF=1`, `MOJOLEARN_IDN_HDB_SOFT_LEAN_OFF=1`, `MOJOLEARN_IDN_HDB_PREDICT_DEVICE_CAST_OFF=1`, `MOJOLEARN_IDN_HDB_PREDICT_LEAN_OFF=1`.

### I15 — qualify streaming exact neighbor merge at ties and batch tails

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I15/manifest.json](../../experiments/performance_ideas/I15/manifest.json), [experiments/performance_ideas/I15/check.mojo](../../experiments/performance_ideas/I15/check.mojo), [neighbors/impl/detail/knn_brute_force.mojo](../../neighbors/impl/detail/knn_brute_force.mojo), [experiments/performance_ideas/I15/certified_caller.mojo](../../experiments/performance_ideas/I15/certified_caller.mojo), [experiments/performance_ideas/I15/compile_matrix.json](../../experiments/performance_ideas/I15/compile_matrix.json), [experiments/performance_ideas/I15/coverage.md](../../experiments/performance_ideas/I15/coverage.md), [experiments/performance_ideas/I15/native_arms.json](../../experiments/performance_ideas/I15/native_arms.json), [experiments/performance_ideas/I15/time.mojo](../../experiments/performance_ideas/I15/time.mojo).

A defines: `MOJOLEARN_KNN_SELECT_TRIAL=1`, `MOJOLEARN_IDN_KNN_CERTIFIED_REACH=1`.

B defines: `MOJOLEARN_KNN_SELECT_TRIAL=1`, `MOJOLEARN_IDN_KNN_CERTIFIED_REACH=1`, `MOJOLEARN_KNN_CERTIFIED_MMA_OFF=1`.

### I16 — attribute IVF grouped and staged scans under skewed list occupancy

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I16/manifest.json](../../experiments/performance_ideas/I16/manifest.json), [experiments/performance_ideas/I16/check.mojo](../../experiments/performance_ideas/I16/check.mojo), [ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo](../../ivf/impl/neighbors/ivf_flat/ivf_balanced_tasks.mojo), [ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo](../../ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo), [experiments/performance_ideas/I16/compile_matrix.json](../../experiments/performance_ideas/I16/compile_matrix.json), [experiments/performance_ideas/I16/coverage.md](../../experiments/performance_ideas/I16/coverage.md), [experiments/performance_ideas/I16/native_arms.json](../../experiments/performance_ideas/I16/native_arms.json), [experiments/performance_ideas/I16/time.mojo](../../experiments/performance_ideas/I16/time.mojo).

A defines: `MOJOLEARN_IVF_BALANCED_TASKS=1`.

B defines: none in manifest.

### I17 — qualify loss-guide frontier retention at varying live leaf capacities

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I17/manifest.json](../../experiments/performance_ideas/I17/manifest.json), [experiments/performance_ideas/I17/check.mojo](../../experiments/performance_ideas/I17/check.mojo), [gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo](../../gbdt/methods/greedy_subsets_searcher/greedy_search_helper_depthwise.mojo), [experiments/performance_ideas/I17/caller_check.mojo](../../experiments/performance_ideas/I17/caller_check.mojo), [experiments/performance_ideas/I17/compile_matrix.json](../../experiments/performance_ideas/I17/compile_matrix.json), [experiments/performance_ideas/I17/coverage.md](../../experiments/performance_ideas/I17/coverage.md), [experiments/performance_ideas/I17/native_arms.json](../../experiments/performance_ideas/I17/native_arms.json), [experiments/performance_ideas/I17/time.mojo](../../experiments/performance_ideas/I17/time.mojo).

A defines: `MOJOLEARN_IDN_GBDT_FRONTIER_RESIDENT=1`.

B defines: none in manifest.

### I18 — Retain bounded exact forest histograms and subtract compatible siblings

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I18/manifest.json](../../experiments/performance_ideas/I18/manifest.json), [experiments/performance_ideas/I18/check.mojo](../../experiments/performance_ideas/I18/check.mojo), [ensemble/decisiontree/batched_levelalgo/exact_histogram_subtract.mojo](../../ensemble/decisiontree/batched_levelalgo/exact_histogram_subtract.mojo), [ensemble/decisiontree/batched_levelalgo/retained_count_histograms.mojo](../../ensemble/decisiontree/batched_levelalgo/retained_count_histograms.mojo), [ensemble/decisiontree/batched_levelalgo/builder.mojo](../../ensemble/decisiontree/batched_levelalgo/builder.mojo), [experiments/performance_ideas/I18/forest_check.mojo](../../experiments/performance_ideas/I18/forest_check.mojo), [experiments/performance_ideas/I18/compile_matrix.json](../../experiments/performance_ideas/I18/compile_matrix.json), [experiments/performance_ideas/I18/coverage.md](../../experiments/performance_ideas/I18/coverage.md), [experiments/performance_ideas/I18/paired_identity.py](../../experiments/performance_ideas/I18/paired_identity.py), [experiments/performance_ideas/I18/test_paired_identity.py](../../experiments/performance_ideas/I18/test_paired_identity.py), [experiments/performance_ideas/I18/time.mojo](../../experiments/performance_ideas/I18/time.mojo).

A defines: `MOJOLEARN_TREE_EXACT_SIBLING_HIST=1`, `MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT=1`.

B defines: `MOJOLEARN_TREE_EXACT_SIBLING_HIST_AUDIT=1`.

### I19 — qualify reusable stable radix scratch and bounded key widths

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I19/manifest.json](../../experiments/performance_ideas/I19/manifest.json), [experiments/performance_ideas/I19/check.mojo](../../experiments/performance_ideas/I19/check.mojo), [experiments/performance_ideas/I19/ragged_float.mojo](../../experiments/performance_ideas/I19/ragged_float.mojo), [experiments/performance_ideas/I19/float_check.mojo](../../experiments/performance_ideas/I19/float_check.mojo), [x_prep/ragged_quantile.mojo](../../x_prep/ragged_quantile.mojo), [experiments/performance_ideas/I19/quantile_check.mojo](../../experiments/performance_ideas/I19/quantile_check.mojo), [x_prep/ragged_categories.mojo](../../x_prep/ragged_categories.mojo), [x_prep/ragged_select.mojo](../../x_prep/ragged_select.mojo), [core/stable_radix_digits.mojo](../../core/stable_radix_digits.mojo), [experiments/performance_ideas/I19/categories_check.mojo](../../experiments/performance_ideas/I19/categories_check.mojo), [experiments/performance_ideas/I19/select_check.mojo](../../experiments/performance_ideas/I19/select_check.mojo), [experiments/performance_ideas/I19/compile_matrix.json](../../experiments/performance_ideas/I19/compile_matrix.json), [experiments/performance_ideas/I19/coverage.md](../../experiments/performance_ideas/I19/coverage.md), [experiments/performance_ideas/I19/native_arms.json](../../experiments/performance_ideas/I19/native_arms.json), [experiments/performance_ideas/I19/time.mojo](../../experiments/performance_ideas/I19/time.mojo).

A defines: `MOJOLEARN_IDN_RAGGED_FLOAT_RADIX=1`.

B defines: none in manifest.

### I20 — Retain bounded source-owned KDE partial storage under the existing chunked fold

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/I20/manifest.json](../../experiments/performance_ideas/I20/manifest.json), [experiments/performance_ideas/I20/check.mojo](../../experiments/performance_ideas/I20/check.mojo), [kde/impl/chunk_workspace.mojo](../../kde/impl/chunk_workspace.mojo), [kde/impl/neighbors/kernel_density.mojo](../../kde/impl/neighbors/kernel_density.mojo), [kde/resident_fit.mojo](../../kde/resident_fit.mojo), [kde/checks/reused_workspace_check.mojo](../../kde/checks/reused_workspace_check.mojo), [experiments/performance_ideas/I20/compile_matrix.json](../../experiments/performance_ideas/I20/compile_matrix.json), [experiments/performance_ideas/I20/coverage.md](../../experiments/performance_ideas/I20/coverage.md), [experiments/performance_ideas/I20/time.mojo](../../experiments/performance_ideas/I20/time.mojo).

A defines: `MOJOLEARN_IDN_KDE_PARTIAL_POOL=1`.

B defines: none in manifest.

### I21 — gate fused GMM components on complete EM state and likelihood

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I21/manifest.json](../../experiments/performance_ideas/I21/manifest.json), [experiments/performance_ideas/I21/check.mojo](../../experiments/performance_ideas/I21/check.mojo), [mixture/checks/estep.mojo](../../mixture/checks/estep.mojo), [experiments/performance_ideas/I21/component_check.mojo](../../experiments/performance_ideas/I21/component_check.mojo), [mixture/checks/mstep.mojo](../../mixture/checks/mstep.mojo), [experiments/performance_ideas/I21/compile_matrix.json](../../experiments/performance_ideas/I21/compile_matrix.json), [experiments/performance_ideas/I21/coverage.md](../../experiments/performance_ideas/I21/coverage.md), [experiments/performance_ideas/I21/native_arms.json](../../experiments/performance_ideas/I21/native_arms.json), [experiments/performance_ideas/I21/time.mojo](../../experiments/performance_ideas/I21/time.mojo).

A defines: `MOJOLEARN_IDN_GMM_COMPONENT_BATCH=1`, `MOJOLEARN_IDN_GMM_CENTER_PAIR=1`.

B defines: none in manifest.

### I22 — qualify blocked TSQR factors and retained reflector applies across tails

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I22/manifest.json](../../experiments/performance_ideas/I22/manifest.json), [experiments/performance_ideas/I22/check.mojo](../../experiments/performance_ideas/I22/check.mojo), [x_decomp/tsqr_device.mojo](../../x_decomp/tsqr_device.mojo), [experiments/performance_ideas/I22/reuse_check.mojo](../../experiments/performance_ideas/I22/reuse_check.mojo), [experiments/performance_ideas/I22/campaign.py](../../experiments/performance_ideas/I22/campaign.py), [experiments/performance_ideas/I22/compile_matrix.json](../../experiments/performance_ideas/I22/compile_matrix.json), [experiments/performance_ideas/I22/coverage.md](../../experiments/performance_ideas/I22/coverage.md), [experiments/performance_ideas/I22/native_arms.json](../../experiments/performance_ideas/I22/native_arms.json), [experiments/performance_ideas/I22/time.mojo](../../experiments/performance_ideas/I22/time.mojo).

A defines: `MOJOLEARN_IDN_TSQR_REUSE=1`, `MOJOLEARN_IDN_TSQR_STRIP_UPDATE=1`.

B defines: none in manifest.

### I23 — qualify independent time-series fits and batch-gradient selection state

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I23/manifest.json](../../experiments/performance_ideas/I23/manifest.json), [experiments/performance_ideas/I23/check.mojo](../../experiments/performance_ideas/I23/check.mojo), [arima/impl/batched_arima.mojo](../../arima/impl/batched_arima.mojo), [experiments/performance_ideas/I23/compile_matrix.json](../../experiments/performance_ideas/I23/compile_matrix.json), [experiments/performance_ideas/I23/coverage.md](../../experiments/performance_ideas/I23/coverage.md), [experiments/performance_ideas/I23/native_arms.json](../../experiments/performance_ideas/I23/native_arms.json), [experiments/performance_ideas/I23/time.mojo](../../experiments/performance_ideas/I23/time.mojo).

A defines: none in manifest.

B defines: `MOJOLEARN_ARIMA_ID_BATCH_GRAD_OFF=1`.

### I24 — fuse resident confusion and PRF input counts in one device pass

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/I24/manifest.json](../../experiments/performance_ideas/I24/manifest.json), [experiments/performance_ideas/I24/check.mojo](../../experiments/performance_ideas/I24/check.mojo), [metrics/impl/classification_joint.mojo](../../metrics/impl/classification_joint.mojo), [experiments/performance_ideas/I24/compile_matrix.json](../../experiments/performance_ideas/I24/compile_matrix.json), [experiments/performance_ideas/I24/coverage.md](../../experiments/performance_ideas/I24/coverage.md), [experiments/performance_ideas/I24/time.mojo](../../experiments/performance_ideas/I24/time.mojo).

A defines: `MOJOLEARN_METRICS_JOINT_COUNTS=1`.

B defines: none in manifest.

### N01 — Isolate packed64 body tiles and their interaction

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/N01/manifest.json](../../experiments/performance_ideas/N01/manifest.json), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo), [experiments/performance_ideas/N01/campaign.json](../../experiments/performance_ideas/N01/campaign.json), [experiments/performance_ideas/N01/compile_matrix.json](../../experiments/performance_ideas/N01/compile_matrix.json).

A defines: `MOJOLEARN_GEMM_KPACK_RPT4=1`, `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY=1`, `MOJOLEARN_GEMM_KPACK_RPT4=1`, `MOJOLEARN_IDN_GEMM_GROUP_TILES_BODY=1`, `MOJOLEARN_IDN_GEMM_NV_STEP_KPACK_OFF=1`.

B defines: none in manifest.

### N02 — Prove and dispatch a two-level logical fold stack

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/N02/manifest.json](../../experiments/performance_ideas/N02/manifest.json), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo), [experiments/performance_ideas/N02/campaign.json](../../experiments/performance_ideas/N02/campaign.json), [experiments/performance_ideas/N02/compile_matrix.json](../../experiments/performance_ideas/N02/compile_matrix.json).

A defines: `MOJOLEARN_GEMM_NV_FS4_OFF=1`, `MOJOLEARN_IDN_GEMM_FS2=1`.

B defines: none in manifest.

### N03 — Pipeline supported asynchronous operand loads with explicit completion

Mode `identical`; vendors nvidia; retained status `build_passed`.

Files: [experiments/performance_ideas/N03/manifest.json](../../experiments/performance_ideas/N03/manifest.json), [gemm/experiments/async_operand_pipeline.mojo](../../gemm/experiments/async_operand_pipeline.mojo), [gemm/experiments/async_operand_pipeline_check.mojo](../../gemm/experiments/async_operand_pipeline_check.mojo), [gemm/experiments/async_api_probe.mojo](../../gemm/experiments/async_api_probe.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [experiments/performance_ideas/N03/compile_matrix.json](../../experiments/performance_ideas/N03/compile_matrix.json).

A defines: none in manifest.

B defines: none in manifest.

### N04 — Separate contiguous and gather staging with counted caller passes

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/N04/manifest.json](../../experiments/performance_ideas/N04/manifest.json), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/checks/gemm_identical.mojo](../../gemm/checks/gemm_identical.mojo), [experiments/performance_ideas/N04/campaign.json](../../experiments/performance_ideas/N04/campaign.json), [experiments/performance_ideas/N04/compile_matrix.json](../../experiments/performance_ideas/N04/compile_matrix.json).

A defines: `MOJOLEARN_GEMM_ARM_TRIAL=1`, `MOJOLEARN_GEMM_ARM_TRIAL=1`, `MOJOLEARN_GEMM_ARM_TRIAL=1`.

B defines: none in manifest.

### N05 — Batch compatible launches with fixed-address changing inputs

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/N05/manifest.json](../../experiments/performance_ideas/N05/manifest.json), [gemm/experiments/grouped_jobs.mojo](../../gemm/experiments/grouped_jobs.mojo), [gemm/experiments/changing_batch_check.mojo](../../gemm/experiments/changing_batch_check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [experiments/performance_ideas/N05/compile_matrix.json](../../experiments/performance_ideas/N05/compile_matrix.json).

A defines: none in manifest.

B defines: none in manifest.

### N06 — Keep compact feature gradients in registers without query barriers

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/N06/manifest.json](../../experiments/performance_ideas/N06/manifest.json), [experiments/performance_ideas/N06/compact_grad.mojo](../../experiments/performance_ideas/N06/compact_grad.mojo), [experiments/performance_ideas/N06/check.mojo](../../experiments/performance_ideas/N06/check.mojo), [transformer/impl/llama/attention_v2.mojo](../../transformer/impl/llama/attention_v2.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [experiments/performance_ideas/N06/compile_matrix.json](../../experiments/performance_ideas/N06/compile_matrix.json), [experiments/performance_ideas/N06/time.mojo](../../experiments/performance_ideas/N06/time.mojo).

A defines: none in manifest.

B defines: none in manifest.

### N07 — Stream bounded feature chunks with exact private integer histograms

Mode `identical`; vendors nvidia, amd, apple; retained status `source_ready`.

Files: [experiments/performance_ideas/N07/manifest.json](../../experiments/performance_ideas/N07/manifest.json), [experiments/performance_ideas/N07/streamed_histogram.mojo](../../experiments/performance_ideas/N07/streamed_histogram.mojo), [experiments/performance_ideas/N07/check.mojo](../../experiments/performance_ideas/N07/check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo](../../ensemble/decisiontree/batched_levelalgo/kernels/builder_kernels_impl.mojo), [experiments/performance_ideas/N07/production_check.mojo](../../experiments/performance_ideas/N07/production_check.mojo), [experiments/performance_ideas/N07/compile_matrix.json](../../experiments/performance_ideas/N07/compile_matrix.json), [experiments/performance_ideas/N07/time.mojo](../../experiments/performance_ideas/N07/time.mojo).

A defines: `MOJOLEARN_IDN_RF_STREAM_REPLICAS=1`.

B defines: none in manifest.

### N08 — Fuse canonical radius distance decisions into device CSR scan/fill

Mode `identical`; vendors nvidia, amd, apple; retained status `build_passed`.

Files: [experiments/performance_ideas/N08/manifest.json](../../experiments/performance_ideas/N08/manifest.json), [experiments/performance_ideas/N08/fused_threshold.mojo](../../experiments/performance_ideas/N08/fused_threshold.mojo), [experiments/performance_ideas/N08/check.mojo](../../experiments/performance_ideas/N08/check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [experiments/performance_ideas/N08/compile_matrix.json](../../experiments/performance_ideas/N08/compile_matrix.json), [experiments/performance_ideas/N08/time.mojo](../../experiments/performance_ideas/N08/time.mojo).

A defines: none in manifest.

B defines: none in manifest.

## Other existing suites and A/B files found

### Existing runtime neural toggles

Runtime schedule/ownership screening. Existing same-baseline digest policy is specific to those schedule arms, not the acceptance rule for new arithmetic versions.

Names: `baseline`, `no_retain_weights`, `legacy_fresh_entry`, `legacy_everything`, `no_stage_reset`, `speculative_attn`, `swiglu_fused`, `no_layer_sync`, `mamba3_legacy`, `mamba3_no_retain_stages`, `norm_dw_own_ws`, `s16_naive`, `s16_shared`, `s16_regs`, `s16_regs2`, `s16_smem48`, `s16_regs2h`, `s16_regsh`, `opt_host`, `all_on`.

Files: [tools/neural_experiments.py](../../tools/neural_experiments.py), [tools/neural_stage_timing.py](../../tools/neural_stage_timing.py).

### Existing IDENTICAL GEMM profiles

Synthetic/component screening; retained profile vendor subsets do not qualify the new full-neural callers.

Names: `one-page`, `group-slack2`, `group-slack8`, `body-tiles`, `packed64`.

Files: [experiments/identical_speed/profiles.json](../../experiments/identical_speed/profiles.json), [experiments/identical_speed/selected-batch.json](../../experiments/identical_speed/selected-batch.json), [tools/identical_speed_ab.py](../../tools/identical_speed_ab.py).

### Resident callpath

Existing shared infrastructure and scaler adapters; classical reference only, no new classical runtime work here. I1/I2/I3 are local names, not performance_ideas I01/I02/I03.

Names: `I1 persistent call`, `I2 batch/per-item wait`, `I3 batch/grouped wait`.

Files: [experiments/identical_callpath/README.md](../../experiments/identical_callpath/README.md), [experiments/identical_callpath/COVERAGE.md](../../experiments/identical_callpath/COVERAGE.md), [core/identical_callpath.mojo](../../core/identical_callpath.mojo), [experiments/identical_callpath/__init__.mojo](../../experiments/identical_callpath/__init__.mojo), [experiments/identical_callpath/compile_probe.mojo](../../experiments/identical_callpath/compile_probe.mojo), [experiments/identical_callpath/families.mojo](../../experiments/identical_callpath/families.mojo), [experiments/identical_callpath/host_gate.mojo](../../experiments/identical_callpath/host_gate.mojo), [experiments/identical_callpath/identity_gate.mojo](../../experiments/identical_callpath/identity_gate.mojo), [experiments/identical_callpath/minmax.mojo](../../experiments/identical_callpath/minmax.mojo), [experiments/identical_callpath/primitive_gate.mojo](../../experiments/identical_callpath/primitive_gate.mojo), [experiments/identical_callpath/session.mojo](../../experiments/identical_callpath/session.mojo), [experiments/identical_callpath/standard.mojo](../../experiments/identical_callpath/standard.mojo), [experiments/identical_callpath/storage.mojo](../../experiments/identical_callpath/storage.mojo), [experiments/identical_callpath/transport_gate.mojo](../../experiments/identical_callpath/transport_gate.mojo).

### Byte-LM pool and logical/device shards

Existing state/rollback/partition matrices. These scripts execute checks and some overwrite installed bindings; none were run in this task.

Names: `pool baseline/candidate`, `logical/device shard baseline/candidate`.

Files: [tools/byte_lm_pool_ab_matrix.sh](../../tools/byte_lm_pool_ab_matrix.sh), [tools/byte_lm_pool_ab_compare.py](../../tools/byte_lm_pool_ab_compare.py), [tools/byte_lm_optimizer_pool_check.py](../../tools/byte_lm_optimizer_pool_check.py), [tools/lm_shards_ab_matrix.sh](../../tools/lm_shards_ab_matrix.sh), [tools/lm_shards_ab_compare.py](../../tools/lm_shards_ab_compare.py), [tools/lm_shards_probe.py](../../tools/lm_shards_probe.py).

### Apple neural schedule families

Existing Apple-specific exploration; Apple does not vote on new IDENTICAL promotions.

Names: `attention arms`, `GEMM geometry`, `neural stage A/B`.

Files: [tools/apple_speed_neural/ab.sh](../../tools/apple_speed_neural/ab.sh), [tools/apple_speed_neural/attn_arms.sh](../../tools/apple_speed_neural/attn_arms.sh), [tools/apple_speed_neural/attn_trial.sh](../../tools/apple_speed_neural/attn_trial.sh), [tools/apple_speed_neural/gemm_geom.sh](../../tools/apple_speed_neural/gemm_geom.sh), [tools/apple_speed_neural/profile.sh](../../tools/apple_speed_neural/profile.sh).

### Legacy GEMM native experiment modules

Includes shared/new neural files and old probes/checks/build orchestration. File presence is not an execution or qualification claim.

Names: see per-file declarations.

Files: [gemm/experiments/async_api_probe.mojo](../../gemm/experiments/async_api_probe.mojo), [gemm/experiments/async_operand_pipeline.mojo](../../gemm/experiments/async_operand_pipeline.mojo), [gemm/experiments/async_operand_pipeline_check.mojo](../../gemm/experiments/async_operand_pipeline_check.mojo), [gemm/experiments/bounded_staging.mojo](../../gemm/experiments/bounded_staging.mojo), [gemm/experiments/bounded_staging_check.mojo](../../gemm/experiments/bounded_staging_check.mojo), [gemm/experiments/bounded_workspace.mojo](../../gemm/experiments/bounded_workspace.mojo), [gemm/experiments/bounded_workspace_check.mojo](../../gemm/experiments/bounded_workspace_check.mojo), [gemm/experiments/changing_batch_check.mojo](../../gemm/experiments/changing_batch_check.mojo), [gemm/experiments/compile_matrix.py](../../gemm/experiments/compile_matrix.py), [gemm/experiments/fold_profile_probe.mojo](../../gemm/experiments/fold_profile_probe.mojo), [gemm/experiments/grouped_jobs.mojo](../../gemm/experiments/grouped_jobs.mojo), [gemm/experiments/grouped_jobs_check.mojo](../../gemm/experiments/grouped_jobs_check.mojo), [gemm/experiments/native_build.py](../../gemm/experiments/native_build.py), [gemm/experiments/neural_epilogue.mojo](../../gemm/experiments/neural_epilogue.mojo), [gemm/experiments/neural_grouped.mojo](../../gemm/experiments/neural_grouped.mojo), [gemm/experiments/neural_plans.mojo](../../gemm/experiments/neural_plans.mojo), [gemm/experiments/neural_profile.mojo](../../gemm/experiments/neural_profile.mojo), [gemm/experiments/neural_profile_device.mojo](../../gemm/experiments/neural_profile_device.mojo), [gemm/experiments/neural_streaming.mojo](../../gemm/experiments/neural_streaming.mojo), [gemm/experiments/neural_tiled.mojo](../../gemm/experiments/neural_tiled.mojo), [gemm/experiments/profile_campaign.py](../../gemm/experiments/profile_campaign.py), [gemm/experiments/profile_check.mojo](../../gemm/experiments/profile_check.mojo), [gemm/experiments/profile_identity_check.mojo](../../gemm/experiments/profile_identity_check.mojo), [gemm/experiments/rounded_epilogue.mojo](../../gemm/experiments/rounded_epilogue.mojo), [gemm/experiments/rounded_epilogue_check.mojo](../../gemm/experiments/rounded_epilogue_check.mojo), [gemm/experiments/subwave_membership.mojo](../../gemm/experiments/subwave_membership.mojo).

### Neural and full-workload orchestration

Saved recipe locations and existing orchestration. Neural-only builders are in scope; full algorithms registry also includes classical estimators.

Names: see per-file declarations.

Files: [tools/bench_board_neural.py](../../tools/bench_board_neural.py), [tools/bench_board_algos.py](../../tools/bench_board_algos.py), [tools/bench_board.py](../../tools/bench_board.py), [tools/performance_ideas.py](../../tools/performance_ideas.py), [tools/performance_full_ab_queue.py](../../tools/performance_full_ab_queue.py), [tools/performance_ideas_gpu_queue.py](../../tools/performance_ideas_gpu_queue.py), [tools/neural_family_screen.py](../../tools/neural_family_screen.py), [tools/neural_family_screen_body.sh](../../tools/neural_family_screen_body.sh), [tools/neural_family_screen_summary.py](../../tools/neural_family_screen_summary.py), [tools/bench_neural_decode.py](../../tools/bench_neural_decode.py).

### State/CNN reference: I08

Inherited shared SSD tiles/lower triangle/retained G*L; old evidence does not qualify new arms.

Names: `MOJOLEARN_IDN_M2_SSD_TILES_OFF=1`, `MOJOLEARN_IDN_M2_CB_LOWER_OFF=1`, `MOJOLEARN_IDN_M2_RETAIN_GL=1`.

Files: [experiments/performance_ideas/I08/manifest.json](../../experiments/performance_ideas/I08/manifest.json), [experiments/performance_ideas/I08/native_arms.json](../../experiments/performance_ideas/I08/native_arms.json), [experiments/performance_ideas/I08/compile_matrix.json](../../experiments/performance_ideas/I08/compile_matrix.json), [experiments/performance_ideas/I08/coverage.md](../../experiments/performance_ideas/I08/coverage.md), [experiments/performance_ideas/I08/check.mojo](../../experiments/performance_ideas/I08/check.mojo), [experiments/performance_ideas/I08/backward_check.mojo](../../experiments/performance_ideas/I08/backward_check.mojo), [experiments/performance_ideas/I08/time.mojo](../../experiments/performance_ideas/I08/time.mojo).

### State/CNN reference: I09

Inherited token-parallel convolution and scan backlog. Component/planted-weight evidence is not full new-profile acceptance.

Names: `MOJOLEARN_IDN_MAMBA_CONV_CELL_OFF=1`.

Files: [experiments/performance_ideas/I09/manifest.json](../../experiments/performance_ideas/I09/manifest.json), [experiments/performance_ideas/I09/native_arms.json](../../experiments/performance_ideas/I09/native_arms.json), [experiments/performance_ideas/I09/compile_matrix.json](../../experiments/performance_ideas/I09/compile_matrix.json), [experiments/performance_ideas/I09/coverage.md](../../experiments/performance_ideas/I09/coverage.md), [experiments/performance_ideas/I09/check.mojo](../../experiments/performance_ideas/I09/check.mojo), [experiments/performance_ideas/I09/time.mojo](../../experiments/performance_ideas/I09/time.mojo).

### State/CNN reference: existing_sequence_tiles

Default-on NVIDIA/AMD sequence tiles; exact NN13 reuse.

Names: `MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF=1`.

Files: [sequence/gemm_tiled.mojo](../../sequence/gemm_tiled.mojo), [sequence/exec_device.mojo](../../sequence/exec_device.mojo), [sequence/README.md](../../sequence/README.md).

### State/CNN reference: existing_m3_angle_suffix

Inherited fixed64 default-on suffix; NN38 reuses seeds.

Names: `MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF=1`.

Files: [mamba/impl/modules/mamba3_backward.mojo](../../mamba/impl/modules/mamba3_backward.mojo), [mamba/host/gen/mamba3_backward.mojo](../../mamba/host/gen/mamba3_backward.mojo).

### State/CNN reference: existing_m2_gradient_leaves

Existing fixed256 row leaves/serial final merge; NN39 changes final merge only.

Names: see per-file declarations.

Files: [mamba/impl/ops/mamba2_ssd_backward.mojo](../../mamba/impl/ops/mamba2_ssd_backward.mojo), [mamba/host/gen/mamba2_ssd_backward.mojo](../../mamba/host/gen/mamba2_ssd_backward.mojo).

### State/CNN reference: existing_mamba_workspace_sessions

Retained workspace, byte-validated borrowed prefill and copied decode weights preexist.

Names: `MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF=1`.

Files: [mamba/impl/modules/idn_gemm_ws.mojo](../../mamba/impl/modules/idn_gemm_ws.mojo), [bindings/_mojolearn_mamba.mojo](../../bindings/_mojolearn_mamba.mojo), [python/mojolearn/_mamba_impl.py](../../python/mojolearn/_mamba_impl.py).

### State/CNN reference: existing_recurrent_scan_gradients

Existing scan/per-step cells/blocked gradients. FAST constant-predictor failure unresolved.

Names: `MOJOLEARN_IDN_SEQ_LSTM_SCAN=1`, `MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF=1`.

Files: [sequence/recurrent.mojo](../../sequence/recurrent.mojo), [sequence/recurrent_scan.mojo](../../sequence/recurrent_scan.mojo), [sequence/ops.mojo](../../sequence/ops.mojo), [sequence/checks/oracle.mojo](../../sequence/checks/oracle.mojo), [tools/sequence_apple_ab.sh](../../tools/sequence_apple_ab.sh).

### State/CNN reference: existing_moe_device_group

Existing any-expert a.i6=2 device grouping/tiled products; B atomic scatter and linear expert lookup. FAST MMA/register alternatives are not IDENTICAL equivalents.

Names: see per-file declarations.

Files: [sequence/moe_group.mojo](../../sequence/moe_group.mojo), [sequence/moe_tiled.mojo](../../sequence/moe_tiled.mojo), [sequence/pyapi.mojo](../../sequence/pyapi.mojo), [sequence/moe_reg.mojo](../../sequence/moe_reg.mojo).

### State/CNN reference: existing_cnn_routes

Existing direct conv/layout/pool fusion. Historical Apple FAST or scoped CNN results are not new IDENTICAL NVIDIA+AMD acceptance.

Names: `MOJOLEARN_XCNN_NO_DIRECT_CONV=1`, `MOJOLEARN_XCNN_NO_TILED_LAYOUT=1`, `MOJOLEARN_XCNN_TILED_ROWS=1`.

Files: [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/ops.mojo](../../x_cnn/ops.mojo), [tools/apple_speed_cnn/ab.sh](../../tools/apple_speed_cnn/ab.sh), [tools/identical_wave_cnn_sgd_compare.py](../../tools/identical_wave_cnn_sgd_compare.py), [tools/identical_wave_cnn_sgd_gate.py](../../tools/identical_wave_cnn_sgd_gate.py).

### State/CNN reference: existing_batchnorm_apply_stats

B applies and updates running state separately, both share inherited centered moment statistics.

Names: see per-file declarations.

Files: [x_cnn/ops.mojo](../../x_cnn/ops.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [x_cnn/host/ops_host.mojo](../../x_cnn/host/ops_host.mojo).

### State/CNN reference: existing_neural_csr

B per-cell SpMM; graph preparation/degree semantics/edge order inherited.

Names: see per-file declarations.

Files: [x_cnn/ops.mojo](../../x_cnn/ops.mojo), [x_cnn/device.mojo](../../x_cnn/device.mojo), [python/mojolearn/_expansion_cnn.py](../../python/mojolearn/_expansion_cnn.py).

Additional attention-family records are retained in [experiments/neural_identical_ab/lanes/attention_integration_inventory.json](../../experiments/neural_identical_ab/lanes/attention_integration_inventory.json):

### I06 — Two-query-head grouped-query page reuse

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo), [experiments/performance_ideas/I06/native_arms.json](../../experiments/performance_ideas/I06/native_arms.json), [experiments/performance_ideas/I06/check.mojo](../../experiments/performance_ideas/I06/check.mojo), [experiments/performance_ideas/I06/coverage.md](../../experiments/performance_ideas/I06/coverage.md), [experiments/performance_ideas/I06/manifest.json](../../experiments/performance_ideas/I06/manifest.json), [experiments/performance_ideas/I06/compile_matrix.json](../../experiments/performance_ideas/I06/compile_matrix.json).

A: `MOJOLEARN_ATTN_ARM_TRIAL=1`; `MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`.

B: `MOJOLEARN_ATTN_ARM_TRIAL=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_kvgrid_r32`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`.

- Corrected two-head component LOSER: MI325X candidate/baseline 1.211/1.212 and L40S 1.147/1.108.
- Earlier ratio-three GQA case had no distinct arm; earlier bundled kvgrid comparison was confounded.
- NN17 is four-head sharing, not a relabeling of this loss.

### I07 — Retained forward exponents versus backward recomputation

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo), [experiments/performance_ideas/I07/native_arms.json](../../experiments/performance_ideas/I07/native_arms.json), [experiments/performance_ideas/I07/check.mojo](../../experiments/performance_ideas/I07/check.mojo), [experiments/performance_ideas/I07/state_cost.mojo](../../experiments/performance_ideas/I07/state_cost.mojo), [experiments/performance_ideas/I07/coverage.md](../../experiments/performance_ideas/I07/coverage.md), [experiments/performance_ideas/I07/manifest.json](../../experiments/performance_ideas/I07/manifest.json), [experiments/performance_ideas/I07/compile_matrix.json](../../experiments/performance_ideas/I07/compile_matrix.json).

A: `MOJOLEARN_ATTN_ARM_TRIAL=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`.

B: `MOJOLEARN_ATTN_ARM_TRIAL=1`; `MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`.

- Corrected retained/recompute component ratios MI325X .560/.586; L40S .666/.665. No full-model promotion follows.
- Earlier kept_cells=0 in both arms was invalid evidence; retained A must prove positive kept cells, B zero.

### ATTN_V1_PACKED_ESTASH — Packed retained exponent storage

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: `MOJOLEARN_ATTN_ARM_TRIAL=1`; `MOJOLEARN_ATTN_V1_PACKED_ESTASH=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`.

B: `MOJOLEARN_ATTN_ARM_TRIAL=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`.

- Supported native geometry and finite-regime admission still required.

### ATTN_V1_ALIAS_Y_ESTASH — Reuse output storage for retained exponents

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: `MOJOLEARN_ATTN_ARM_TRIAL=1`; `MOJOLEARN_ATTN_V1_ALIAS_Y_ESTASH=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`.

B: `MOJOLEARN_ATTN_ARM_TRIAL=1`; `MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32`; env: `MOJOLEARN_TRANSFORMER_ATTN_PATH=fused`.

- The explicit override requests alias storage; source also has inherited vendor default. This baseline compares recompute, not an isolated alias-versus-packed storage experiment.
- Lifetime and allocation reach must be recorded, not inferred from a define.

### ATTENTION_V2 — Prior fixed-tile online summary profile

Files: [transformer/impl/llama/attention_v2.mojo](../../transformer/impl/llama/attention_v2.mojo), [transformer/ATTENTION_V2_CONTRACT.md](../../transformer/ATTENTION_V2_CONTRACT.md), [transformer/checks/attention_v2_forward_bench.mojo](../../transformer/checks/attention_v2_forward_bench.mojo), [transformer/checks/attention_v2_backward_bench.mojo](../../transformer/checks/attention_v2_backward_bench.mojo), [transformer/checks/attention_v1_memory_profile_check.mojo](../../transformer/checks/attention_v1_memory_profile_check.mojo), [tools/attention_v2_oracle.py](../../tools/attention_v2_oracle.py).

A: see source control.

B: see source control.

- Component executable selection only; no claimed public model switch.
- Prior exact fixtures were qualified on Apple/NVIDIA/AMD; retained contract says large-shape memory/speed tradeoff is not promotable.
- NN20 balanced absolute summary tree is a different graph; old qualification and large-shape failures do not qualify NN20.

### ATTN_RUNTIME_ARMS — Existing stash, tiling, query residency and retained-state runtime family

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo), [core/step_phase.mojo](../../core/step_phase.mojo).

A: `MOJOLEARN_ATTN_ARM_TRIAL=1`; env: `MOJOLEARN_ATTN_ARM=stash_tiled`.

B: `MOJOLEARN_ATTN_ARM_TRIAL=1`; env: `MOJOLEARN_ATTN_ARM=baseline`.

Other recorded runtime arms: `bwd_stash`, `fwd_sstash`, `bwd_stash_tiled`, `stash`, `stash_tiled_ztiled`, `stash_tiled_pf`, `stash_tiled_fgrid_r32`, `stash_tiled_fgrid_r32_qres`, `stash_tiled_fgrid_r32_qres_pf_kvrecompute`, `stash_tiled_fgrid_r32_qres_pf_kvsplit`, `stash_tiled_fgrid_r32_qres_pf_zlag`, `stash_tiled_fgrid_r32_qres_pf_estash_kvgrid_r32`, `stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32`.

- Runtime names only take effect in a trial-compiled build; keep per-arm geometry and refusals in evidence.

### ATTN_LAUNCH_GEOMETRY — Existing explicit forward/dQ/dKdV and retained-exponent tile arms

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: see source control.

B: `MOJOLEARN_ATTN_TILE_RULE_OFF=1`.

- Explicit overrides can suppress the hardware/row-rule dispatcher. Preserve each experiment as a separate arm; do not bundle all listed flags.

### ATTN_STICKY — Direction-specific sticky refusal latch

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

A: `MOJOLEARN_ATTN_STICKY=1`.

B: see source control.

- OFF: prior 700-step H100 control regressed throughput by 3.6%; intermittent refusals mean a permanent latch loses work.

### ATTN_MASKED_TAIL_REPAIR — Existing exact omitted-tail replay

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: `MOJOLEARN_ATTN_REPAIR_MASKED_TAIL=1`.

B: `MOJOLEARN_ATTN_LEGACY_CORNER=1`.

- Historical correctness repair: old legacy arm may refuse supported inputs.
- MOJOLEARN_ATTN_NO_BWD_CORNER and *_SABOTAGE are diagnostic bit-changing arms, excluded from quality-preserving candidates.

### IDN_ATTN_SCAN_CACHE — Reuse finite-scan scratch

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: see source control.

B: `MOJOLEARN_IDN_ATTN_SCAN_CACHE_OFF=1`.

- Inherited IDENTICAL default; ALL_OFF disables it. No new promotion.

### IDN_ATTN_BWD_SCAN_REUSE — Reuse admitted forward finite-scan result

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: see source control.

B: `MOJOLEARN_IDN_ATTN_BWD_SCAN_REUSE_OFF=1`.

- Inherited IDENTICAL default; ALL_OFF disables it. No new promotion.

### IDN_ATTN_SCRATCH_CACHE — Retain attention scratch owners

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: see source control.

B: `MOJOLEARN_IDN_ATTN_SCRATCH_CACHE_OFF=1`.

- Inherited IDENTICAL default; ALL_OFF disables it. No new promotion.

### IDN_ATTN_ONE_FLAG_WAIT — Share scan/status completion wait

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: see source control.

B: `MOJOLEARN_IDN_ATTN_ONE_FLAG_WAIT_OFF=1`.

- Inherited IDENTICAL default; ALL_OFF disables it. No new promotion.

### ATTN_SPECULATIVE — Speculative fused enqueue admission

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: env: `MOJOLEARN_ATTN_SPECULATIVE=1`.

B: env: `MOJOLEARN_ATTN_SPECULATIVE=0`.

- Record refusal/replay rate as well as timing.

### ATTN_AMD_MFMA_DQ — AMD dQ matrix-instruction trial

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: `MOJOLEARN_ATTN_DQ_MFMA=1`.

B: see source control.

- Vendor-specific physical kernel; still owes same arithmetic graph and four-column admission.

### ATTN_APPLE_MMA — Apple dKdV and zdot matrix-instruction trials

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: `MOJOLEARN_ATTN_DKDV_AMMA=1`.

B: see source control.

- Apple never votes IDENTICAL performance. Prior zdot M4 component was slower: 2119 ms versus 1808–1860 ms per step.

### ATTN_AMD_MODE_FLUSH — AMD hardware flush-mode seam spelling

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: see source control.

B: `MOJOLEARN_ATTN_NO_MODE_FLUSH=1`.

- AMD-only physical spelling; software-seam control retained.

### ATTN_APPLE_ERECOMP — Apple gated exponent recomputation

Files: [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: `MOJOLEARN_ATTN_APPLE_ERECOMP=1`.

B: see source control.

- Admission can deny the route; no source define proves execution. Apple timing cannot promote IDENTICAL switches.

### IDN_LLAMA_REFUSE_BATCH — Batch ordered nonfinite refusals

Files: [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

A: see source control.

B: `MOJOLEARN_IDN_LLAMA_REFUSE_BATCH_OFF=1`.

- Inherited IDENTICAL default; retains first refusal ordering.

### IDN_ATTN_CACHE_NOWAIT — Omit redundant same-stream cache wait

Files: [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

A: see source control.

B: `MOJOLEARN_IDN_ATTN_CACHE_NOWAIT_OFF=1`.

- Alias B flag MOJOLEARN_ATTN_CACHE_WAIT=1; inherited IDENTICAL default.

### SWIGLU_FORWARD_ONLY — Existing forward-only SiLU multiply fusion

Files: [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

A: env: `MOJOLEARN_SWIGLU_FUSED=1`.

B: env: `MOJOLEARN_SWIGLU_FUSED=0`.

- No saved silu intermediate: requires forward_only; NN26 separately implements training storage.

### BWD_GATED_SILU_FUSION — Existing backward gated SiLU fusion

Files: [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

A: `MOJOLEARN_BWD_GATED_SILU_FUSED_TRIAL=1`.

B: `MOJOLEARN_BWD_GATED_SILU_SPLIT_TRIAL=1`.


### BWD_RMS_DH_DOT_FUSION — Existing RMS backward dh and row-dot fusion

Files: [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

A: `MOJOLEARN_BWD_NORM_FUSED_TRIAL=1`.

B: `MOJOLEARN_BWD_NORM_SPLIT_TRIAL=1`.

- Forced fused source option now actually forces fusion; prior explicit arm still obeyed exact-size thresholds.

### BWD_RMS_SHAPE_RULE_REMOVAL — Remove benchmark-dimension backward RMS dispatcher

Files: [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [transformer/impl/llama/fused_attention.mojo](../../transformer/impl/llama/fused_attention.mojo).

A: see source control.

B: `MOJOLEARN_BWD_NORM_LEGACY_SHAPE_RULE=1`.

- Old Apple m>=8192,dm>=768 / NVIDIA m>=32768,dm>=768 kept only in explicit historical B.
- A uses row grid >= 2 * reported SM/CU; unknown hardware splits. Existing fixed-order arithmetic is unchanged.
- New source is uncompiled/unverified/unmeasured. Must time neighboring sizes and one non-board full dataset on both voting vendors before a performance claim.
- The removed rule also governed FAST with AFN_LM_BWD_FUSE enabled; no new NN switch enables FAST.

### BWD_NORM2_RESIDUAL_FUSION — Post-attention RMS backward and residual gradient fusion

Files: [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

A: `MOJOLEARN_BWD_NORM2_RESIDUAL_FUSED_TRIAL=1`.

B: `MOJOLEARN_BWD_NORM2_RESIDUAL_SPLIT_TRIAL=1`.


### RESIDUAL1_NORM2_FUSION — Attention residual plus post-attention RMS forward

Files: [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo).

A: `MOJOLEARN_FUSE_RESIDUAL1_NORM2=1`.

B: see source control.

- Apple already has an inherited admitted default under its working-set cap; explicit force is not a distinct Apple arm everywhere. NN24 migrated both spellings.

### RESIDUAL2_NEXT_NORM — Cross-layer residual plus next RMS norm

Files: [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [training/byte_lm.mojo](../../training/byte_lm.mojo).

A: see source control.

B: `MOJOLEARN_DISABLE_RESIDUAL2_NEXT_NORM=1`.

- Existing Apple-specific working-set rule, not a new default.

### STEP_GLUE_ROWS — Existing RMS launch row geometry

Files: [core/step_glue.mojo](../../core/step_glue.mojo), [transformer/impl/llama/modeling_llama.mojo](../../transformer/impl/llama/modeling_llama.mojo), [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo), [checks/kernel_matrix.mojo](../../checks/kernel_matrix.mojo).

A: `MOJOLEARN_STEP_GLUE_TRIAL=1`; env: `MOJOLEARN_STEP_GLUE_ARM=rows16`.

B: `MOJOLEARN_STEP_GLUE_TRIAL=1`; env: `MOJOLEARN_STEP_GLUE_ARM=shipped`.

- Additional separate choices rows8 and rows4; rows tokens can compose with optskip/noshadow in documented parser order.
- Some columns have inherited shipped rows; compare explicit metadata, not an assumed all-vendor default.

### IDN_BWD_ENTRY_ALIAS — Read-only incoming-gradient view

Files: [transformer/checks/transformer_backward.mojo](../../transformer/checks/transformer_backward.mojo).

A: see source control.

B: `MOJOLEARN_IDN_BWD_ENTRY_ALIAS_OFF=1`.

- Inherited IDENTICAL default; traced/materialized paths retain copies; ALL_OFF disables.

### TRANSFORMER_HOST_FUSED_ATTENTION — Host staged versus fused attention

Files: [transformer/checks/transformer_oracle.mojo](../../transformer/checks/transformer_oracle.mojo).

A: see source control.

B: `MOJOLEARN_TRANSFORMER_HOST_STAGED_ATTENTION=1`.

- Incumbent shortcut disabled under NN03/04; NN20 selects its own coherent host graph first.

### TRANSFORMER_SESSION_REUSE — Existing public native session and fresh-prefill reuse

Files: [python/mojolearn/_transformer_impl.py](../../python/mojolearn/_transformer_impl.py), [bindings/_mojolearn_transformer.mojo](../../bindings/_mojolearn_transformer.mojo).

A: env: `MOJOLEARN_TRANSFORMER_LEGACY_SETUP=0`.

B: env: `MOJOLEARN_TRANSFORMER_LEGACY_SETUP=1`.

- Cold construction and repeated calls are different boundaries; saved tape API explicitly requires its native owner.

## Retained limits and evidence

NN41's inherited recurrent-scan quality failure and NN20's earlier attention-profile nonpromotion remain open evidence; new wiring does not erase either. Explicit dimension-targeted norm scheduling found during this source pass has a general hardware-cost candidate and an explicit historical B; neighboring shapes and a non-board workload remain required before promotion.

This index describes experiments found in the inspected registries and named suites; it does not claim every historical branch or every compiler option in the repository was enumerated. The machine-readable companion is [experiment_inventory.json](../../experiments/neural_identical_ab/experiment_inventory.json), with selector source in [write_inventory.py](../../experiments/neural_identical_ab/write_inventory.py). Model scope and outstanding research variants remain in [MODEL_INTEGRATION.md](../../experiments/neural_identical_ab/MODEL_INTEGRATION.md).
