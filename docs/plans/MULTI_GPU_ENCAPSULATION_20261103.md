# Multi-GPU encapsulation (2026-11-03)

Owner decision (Andrew, 2026-11-03): encapsulate the single-GPU code so the
multi-GPU layer composes over it instead of being interleaved with it.
Fable plans; Opus lanes implement; the orchestrator compiles once and runs the
identity checks once. Bits must not change in single-device mode.

## Where we are

Five multi-GPU mechanisms exist. Two are clean, three are not.

- **Clean:** process-pool sharding (`bindings/parallel_shard_native.mojo`,
  `native_process_pool.mojo`) and the neural trainers
  (`training/byte_lm_parallel.mojo` wraps `List[ByteTrainer]`). Each shard runs
  the plain single-device owner. Leave them alone.
- **Kind A, the worst coupling:** classical families read an ambient
  thread-local device plan (`core/device_plan.mojo`: `device_plan_setting`,
  `device_plan_device`, `device_plan_root`) deep inside kernels, and shard
  implementations live inside the single-device files. About 47 branch sites
  over about 20 files:
  - `svm/impl/distance/kernel_matrices.mojo` (`distribute` :526, branch
    :544-552, `_kernel_rows` :632/:671), `svm/impl/smosolver.mojo:848`;
  - `isolation_forest/impl/isolation_forest.mojo` (:915, :971-983,
    `_fit_tree_shards` :1665/:1690);
  - `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo` (comptime `distribute` :3135, branch
    :3213-3224, `launch_feature_shards` :7030),
    `gbdt/methods/pointwise_scores_calcer.mojo` :375/:431/:466;
  - `resample/estimator.mojo` (count :1029, branches :1500/:1843/:2103,
    per-rank contexts :1202/:1241/:1321);
  - `neighbors/estimator.mojo` (:535/:759/:703/:790/:1334),
    `dbscan/impl/runner.mojo` (10 sites, :783-:1347),
    `hierarchy/.../connectivities.mojo:176/:233`,
    `cluster/impl/detail/kmeans.mojo:1586`,
    `cluster/impl/detail/min_cluster_distance_compute.mojo:208`,
    `glm/impl/qn/glm_base.mojo:1795`, `mixture/estimator.mojo` (5 dispatch
    calls; the dispatcher in `mixture/multi_gpu.mojo:118-149` already owns the
    single-device fallback: this is the target pattern);
  - `core/gemm.mojo:512/:606`, `core/gram_splitk.mojo:1208/:1227`,
    `core/householder_qr.mojo:702-730`, `gemm/tn_centered.mojo:82`,
    `core/neural_context.mojo:92`, `mamba/impl/mamba_shell_impl.mojo:673`;
  - `solver/impl/cd.mojo:1715/:1728`: single-device fast kernels are switched
    off unless the device count is `""` or `"1"`.
- **Kind E:** the optimizer reads `MOJOLEARN_OPTIMIZER_DEVICE_COUNT` with a raw
  `getenv` (`training/optimizer_multi_gpu.mojo:61`, `clip_multi_gpu.mojo:71`,
  `accumulate_multi_gpu.mojo:31`, `optimizer_public.mojo` :399/:447/:512/:680,
  `optimizer_public_common.mojo:294`, `bindings/samba_state_native.mojo:421`,
  `samba_step_native.mojo:40`). At `optimizer_public.mojo:679-680` the variable
  named `multi` selects the single-device route.
- **Autograd distribute/partition:** composes one `OpsSession[E]` per device
  over `Exec` and adds no branches to single-GPU code, but reaches into
  OpsSession privates 46 times (`_drop_temps`, `_slot`, `_add`, `_op_run`,
  `tape._nodes`, `opt_states`, `int_words`, `opts`, `_opt_index`) across
  `autograd/distributed_program.mojo`, `distributed_entries.mojo`,
  `partition_stage.mojo`, `partition_entries.mojo`, and duplicates the
  epoch/minibatch loop of `train_program_control`
  (`program_runtime.mojo:131-344`).

## Target shape

1. **One explicit `DevicePlan` value per family entry.** The binding entry
   reads the thread-scoped plan ONCE (`device_plan_setting`/`device_plan_device`
   at the family's public native entry) and passes a value down. No kernel or
   estimator file below the entry calls `device_plan_*`.
2. **One dispatcher per family in its `multi_gpu.mojo`:**
   `<family>_dispatch(plan, ...)`. With `plan.single()` it calls the single-device
   entry unchanged; otherwise it runs the shard plan. The gmm dispatcher
   (`mixture/multi_gpu.mojo:118-149`) is the model.
3. **Shard code leaves the single-device files.** `_kernel_rows`,
   `_fit_tree_shards`, `launch_feature_shards`, the resample per-rank owners,
   the QR panel split and the Gram split-k shard path move into the family's
   `multi_gpu.mojo` (or `core/*_multi_gpu.mojo`). The single-device file keeps
   no `distribute` parameter and no device-count branch.
4. **Kernel-choice gates keep their exact test.** `cd.mojo`, `smosolver.mojo`
   and `tn_centered.mojo` select kernels on "exactly one device". Replace the
   string test by `plan.single()` whose definition is literally
   `setting == "" or setting == "1"`. Same truth table, so same bits.
5. **The optimizer reads the plan, not the environment.** The bindings
   (`samba_state_native`, `samba_step_native`) read the environment once at
   entry, enter the thread plan, and the optimizer uses `plan.count`.
   Rename `multi` at `optimizer_public.mojo:679` to what it is (`single`).
6. **Autograd executor interface.** A public `ProgramExecutor[E]` over
   `OpsSession` with `load_params`, `prepare_dataset`,
   `gradient(range, loss, rng_key)`, `param_slice`, `grad_slice`,
   `opt_step(range)`, `evaluate_node`, `forward`, `read_loss`. The distributed
   and partitioned programs use only this interface; the 46 private accesses go.
   The two fit loops keep their arithmetic order (the distributed fold is a
   host `identical_mul_add` over shard scalars and row order differs: they are
   NOT bit-equal to `Program.fit` and must not be rewritten as "one shard");
   share only the non-arithmetic scaffolding (epoch order, minibatch ranges,
   checkpoint cadence).

## Rules that bind every lane

- Lanes never compile, never run tests on the laptop, never rent or ssh. Code,
  commit, push, reply (CLAUDE.md). The orchestrator compiles the integration
  branch once and runs the identity table admit once.
- No bit changes in single-device mode. A hoisted branch keeps the same
  condition and the same call in the same order. Multi-device bits may change
  only where the plan already says they are not promised (autograd distributed).
- GPU-path rules stay: no host loops over rows, no Python compute, no new host
  readbacks. Moving code is not an excuse to add a download.
- Each lane edits only its family's files (listed in its brief) plus the
  family's `multi_gpu.mojo` and the family's binding entry. Shared files
  (`core/device_plan.mojo`) are edited by lane `dp-plan-value` only; the other
  lanes code against the interface declared below and say so in their reply.
- Every lane's reply lists the bindings the orchestrator must rebuild.

## Shared interface (lane dp-plan-value adds it; others code against it)

```mojo
# core/device_plan.mojo
struct DevicePlan(Copyable, Movable):
    var setting: String        # the raw setting ("" or "1" means one device)
    var count: Int             # devices in the plan, >= 1
    var root: Int              # the root device id
    var devices: List[Int]     # device ids in plan order
    fn single(self) -> Bool    # setting == "" or setting == "1"  (exact old test)
    fn device(self, rank: Int) -> Int

def device_plan_current(name: String, default: String = "") raises -> DevicePlan
# reads the thread-scoped plan once; `name` is the family's count setting
# (the setting is per family, so a no-argument form could not reproduce it)
def plan_for_single() -> DevicePlan      # setting "", one device, for callers outside any entry
```

## Lanes (phase 1, in parallel; phase 2 after the compile is green)

| lane | files | moves |
|---|---|---|
| dp-plan-value | `core/device_plan.mojo`, `core/multi_gpu.mojo` | `DevicePlan`, `device_plan_current`; no behavior change |
| dp-svm | `svm/impl/distance/kernel_matrices.mojo`, `svm/impl/smosolver.mojo`, `svm/multi_gpu.mojo` (new), the svm binding entry | `_kernel_rows` and the `distribute` branch out; `one_device` gate from `plan.single()` |
| dp-iforest | `isolation_forest/impl/isolation_forest.mojo`, `isolation_forest/multi_gpu.mojo` (new), its binding entry (`bindings/_mojolearn_trees.mojo`) | `_fit_tree_shards` out |
| dp-gbdt | `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo`, `gbdt/methods/pointwise_scores_calcer.mojo`, `gbdt/methods/pointwise_multi_gpu.mojo`, its binding entry | `launch_feature_shards` out; the three pointwise sites through the dispatcher |
| dp-resample | `resample/estimator.mojo`, `resample/multi_gpu.mojo` (new), its binding entry | per-rank contexts and the three branches out |
| dp-core | `core/gemm.mojo`, `core/gram_splitk.mojo`, `core/householder_qr.mojo`, `gemm/tn_centered.mojo`, `core/gram_multi_gpu.mojo`, `solver/impl/cd.mojo` | plan passed in; split-k shard and QR panel split out; gates via `plan.single()` |
| dp-neighbors | `neighbors/estimator.mojo`, `dbscan/impl/runner.mojo`, `hierarchy/impl/cluster/detail/connectivities.mojo`, `cluster/impl/detail/{kmeans,min_cluster_distance_compute}.mojo`, `glm/impl/qn/glm_base.mojo`, `mixture/estimator.mojo`, `core/neural_context.mojo`, `mamba/impl/mamba_shell_impl.mojo`, their `multi_gpu.mojo` | every `device_plan_*` read replaced by a plan value from the entry; dispatchers own the fallback |
| dp-optimizer | `training/optimizer_multi_gpu.mojo`, `clip_multi_gpu.mojo`, `accumulate_multi_gpu.mojo`, `optimizer_public.mojo`, `optimizer_public_common.mojo`, `bindings/samba_state_native.mojo`, `samba_step_native.mojo` | environment read once at the binding entry; plan value below; `multi` renamed |

Phase 2 (one lane, after phase 1 is green): `ProgramExecutor[E]` in
`autograd/`, then `distributed_*` and `partition_*` rewritten against it.

## Acceptance

- `rg -n 'device_plan_(setting|device|root)\(' core bindings providers` matches
  only `core/device_plan.mojo` and the binding entries.
- `rg -n 'getenv\("MOJOLEARN_OPTIMIZER_DEVICE_COUNT"' .` matches only
  `training/optimizer_device_plan.mojo` (integration 2026-11-03: the samba
  bindings never call `training/` directly; the public optimizer entries and
  the native dispatch tables are entries of their own, so one helper builds
  the plan for all of them, read once per entry). Owed: the Python-side
  reads in `bindings/small_mlp_native.mojo` and `parallel_mlp_native.mojo`.
- The integration compile is green on NVIDIA, AMD and Apple; the identity
  table admit shows no changed digest on any single-device lane.
- Phase 2: `rg -n '\._(slot|add|op_run|drop_temps|opt_index)|tape\._nodes|\.opt_states' autograd/distributed_* autograd/partition_*` matches nothing.
