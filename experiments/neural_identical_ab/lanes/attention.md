# NN17–NN32 source handoff

Branch `ideas/neural-identical-ab-20261006-r3`, forked from `main` at
`fd6cf80453a6f18eb02e81566c824e7da106ccf0`. This continuation adds source
integration after root's `d39587b1d` snapshot. Root owns the final commit.

The selected native/public scope for all 16 cards is now written or attributed
to existing native source. These are **uncompiled, unverified source drafts**.
No Mojo invocation, builds, tests, static/syntax/lint checks, identity checks,
quality runs, benchmarks or candidate execution occurred. No new performance
claim, quality claim, default promotion or qualification is implied.

All new switches are OFF and IDENTICAL/`!MOJOLEARN_IDN_ALL_OFF` gated. Existing
shipped switches are explicitly attributed. Each numerical profile shares
its scalar graph across host/NVIDIA/AMD/Apple; cross-version bits may change.
The necessary evidence that implementations meet this contract is unrun.

| Card | Concrete selected implementation | Scope / optional broader variants |
|---|---|---|
| NN17 | Four adjacent query heads share the same K/V page in the actual tiled forward kernel, 64 logical rows and 16 positions/head. Existing backward consumes that forward graph. | Existing HD64/TQ32/QRES/PF non-swizzled reach and GQA ratio divisible by four. Retain I06 two-head losses. |
| NN18 | Existing structural K-page bounds are the A arm; new native dense-page B control traverses every page with the same per-row masks, admission and replay. | `MOJOLEARN_NN18_DENSE_TILE_CONTROL=1` only in B. Both arms select the same tiled model path. General task compaction is optional. |
| NN19 | Reuse actual I07 retained/recompute fused attention source and native model callers. | Retain corrected component evidence, invalid-arm history and pending full-task qualification. |
| NN20 | Public model host/GPU forward, backward, trace, prefill, decode and window paths now select the shared stable-summary graph. Finite/overflow/index admission and profile metadata are native. | Fixed absolute 32-key leaves and every merge level anchored to absolute key zero. Existing attention_v2 failures remain. Softcap/INT15/legacy plants refuse. Cooperative scheduling is optional. |
| NN21 | Eager score scale+mask share a launch, retaining both stage writes and the required +0 mask add. | Existing softcap/plants/sabotage use old route. Both A/B arms force eager. |
| NN22 | Eager dK/dV share traversal, preserving each original h-then-t chain. | Both arms force eager; broader fused geometry is optional. |
| NN23 | Eager row-dot and dS consumers share one row owner, with every stage retained. | Both arms force eager. Timing may lose; no result assumed. |
| NN24 | Actual RMS forward/backward, residual RMS, q/k RMS and RMS/centered-LN inference select the same eight logical lanes on host/device. Decode, checkpoint and shared Samba transformer norms migrate together. | Existing extended-options training refusal stays explicit. Standalone LN backward helper is not newly advertised public training. Mamba's independent norm graph stays its own profile. |
| NN25 | RMS sumsq then flat output scaling in actual norm launcher. | Repeated scalar div/rsqrt cost unmeasured. Other normalization scheduling variants optional. |
| NN26 | Actual training SiLU/gate fusion retains both saved stages; existing paired backward reused. | Saved-sigmoid expansion optional. |
| NN27 | Actual Q/K RoPE launch shares position-table loads for independent rounded rotations. | Cross-layer table sharing/scan elision optional. |
| NN28 | Root wired four ByteLM `_byte_forward_loss` prefill calls to explicit dead-cache omission, resetting empty cache and retaining backward/trace stages. | Other training owners optional. |
| NN29 | Actual paired K/V window-ring write selects each slot's last new token. Existing linear paired append retained. | Projection-to-cache layout and request batching optional. |
| NN30 | Existing shipped backward read-only incoming-gradient view and OFF control reused. | No new default. Further ownership transfer optional. |
| NN31 | Public native tape budget/cost admission chooses actual retained stages or release+replay. Original native ledger also models shared layer leases and outstanding ticket reset refusal. | Public budget is per block's retained stage allocations, not a total-memory guarantee. Global layer-group scheduling optional. |
| NN32 | Existing GPU TransformerSession and matching CPU session expose actual forward-tape/backward VJP, native snapshots, epoch/cookie checks, mutation invalidation, single-use consume and stale-close protection. | Default FP32 IDENTICAL zero-state prefill, including windows. Direct ByteTrainer/Samba optimizer adoption is an optional extension. |

`attention.json` records exact flags, source paths, public APIs, prior evidence,
A/B reach and remaining measurement coverage. NN20 model B is incumbent
attention; the standalone component's left-fold control is separately named.
NN24 model B is the incumbent norm; its standalone LN helper makes no claim
of matching every historical derivative. NN20 overrides older attention
schedule cards when combined; combinations must report actual reach.

The new public API is `TransformerBlock.forward_with_tape(x,
activation_budget_bytes=..., minimum_replay_ops_per_byte=...) -> (y, tape)`,
followed by `backward_from_tape(tape, dy)` returning input plus nine weight
gradients. All tensor work and snapshots are native. Python checks arguments,
allocates output buffers and passes addresses. `tape.retained`, `stage_bytes`
and `profile` are native metadata. `tape.close()` releases only its own
current generation. Tapes cannot be serialized. Later block/session calls
invalidate tickets before changing weights or storage. Array edits after
forward do not change its native snapshot. Configuration changes refuse.

The byte budget counts explicit retained forward stages and GEMM workspace.
It excludes immutable input/weight snapshots, transient current forward and
backward storage, and shared process caches. Budget misses replay the same
native arithmetic. NN31 applies the optional shape-derived replay-cost
threshold; NN32 retains under the bound. With both switches off, the API
performs a real replay B arm. The API remains IDENTICAL-only.

Pure host/device arithmetic lives in `norm_profile_contract.mojo`,
`attention_summary_contract.mojo` and `summary_model_contract.mojo`.
`summary_model.mojo` owns device packing/scratch/lifetime fences;
`summary_model_host.mojo` uses the same scalar operations on native lists.
`transformer_arithmetic_profile()` labels attention, norm and neural GEMM
versions. The bindings expose it; block serialization retains and admits the
label. State/decode admission rejects mismatched tags and an experimental
profile refuses an untagged nonempty KV cache. Existing weights can migrate
by explicitly constructing a new block.

All transformer device/host projection, backward and norm-parameter GEMMs
now use the neural dispatcher. NN05/07 reach the real gate/up pair through
`GemmWorkspace.run_pair`. NN03/04 bypass incumbent fused QK shortcuts on
both host and device; NN20 declares its own common scalar QK graph. Root
routed ByteLM optimized host inference through the common transformer oracle
for NN20/24 in `training/byte_lm_host.mojo` and migrated the independent Samba
host transformer norm helpers.

Future admission remains full-workload A/B on NVIDIA and AMD, common bits
also on host and Apple, and task/gradient quality. Include preparation,
synchronization, outputs, cold/repeated and train/inference/decode boundaries.
Keep existing accepted receipts; retain I06 losses, I07 invalid-arm history
and attention_v2 nonpromotion. No unsupported toolchain workaround was added;
if a future cooperative/async variant needs Modular support, record the ask
and wait.

Keep logs out of context: save complete output to files, use targeted rg/grep
with bounded surrounding lines and short tails, and summarize exit status,
coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
never hide failures or infer full success from filtered output.

## Source integration inventory continuation

`attention_integration_inventory.json` records every NN17–NN32 A/B arm, exact
bare compile definitions, environment, runtime operation, model callers,
required reach, historical experiment families and limitations. It also retains
existing I06/I07 controls, old v2 nonpromotion, launch geometries, caches, waits,
norm/activation fusions and session experiments. These are source records, not
new execution evidence.

The public runner now accepts `--neural-ab-config`, clears declared managed
environment variables, applies IDENTICAL before imports, freezes configuration
metadata and propagates it to workers. Requested compile definitions stay
separate from loaded-binary and kernel-reach evidence. Ordinary model runs
remain available; nondefault operations are explicitly named and own-only.

For NN31/32, `transformer_forward_vjp_tape` consumes full saved inputs and all
forward/input-gradient/weight-gradient outputs; native retained-stage metadata
is recorded, and unexpected retention refuses the declared arm. The existing
forward quality reference does not qualify the VJP or a training operation.
The chosen 512 MiB retention budget must fit the frozen workload or be replaced
by an explicit new memory-qualified recipe.

For NN29, `transformer_windowed_forward` passes a positive `transformer_window`
(model parameter, currently 256 in both recipe arms), consumes full input and
explicit K/V ring state, and resets logical position to zero per repeat. It
covers full prefill cache commit; carried decode needs its own full-workload
recipe. The old window-zero quality comparator is refused as a qualification
for this operation. NN17's ordinary board fixture has GQA ratio one and cannot
reach four-head reuse. The explicit [full GQA block recipe](../workloads/NN17-full-gqa.json)
selects the same ratio-four model in both arms through `transformer_configured_forward`.
The harness applies those declared dimensions before writing the complete input,
weights and shape metadata. This supplemental block workload does not replace
LM corpus training or all affected-estimator coverage; input hashes, execution,
branch reach and quality/performance evidence remain pending.

Source tracing corrected NN20/24 imports that had landed inside long module
docstrings, moved NN19 to explicit matched estash requests, and made serialized
arithmetic tags reflect the actual FAST/IDENTICAL binding mode. It also removed
an existing backward RMS dispatcher tied to exact board dimensions: the new
rule uses independent row blocks versus reported SM/CU capacity, with the old
rule retained only under `MOJOLEARN_BWD_NORM_LEGACY_SHAPE_RULE=1` as historical B.
Neighboring/non-board full-workload A/B remains unrun, including the pre-existing
FAST fusion route to which that dispatch rule also applied.

The harness source also connects NN40 explicit owned-weight installation and
NN53 public chunked-head configuration; root and the other lanes own those
inventories. All compilation, syntax/static checks, tests, identity, quality,
execution and timing remain unrun at the user's request. Root owns commit/push.
