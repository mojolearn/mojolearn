# NN17–NN32 attention/transformer lane

Branch `ideas/neural-identical-ab-20261006-r3`, base `main` commit
`fd6cf80453a6f18eb02e81566c824e7da106ccf0`. Source drafts are uncommitted.
No Mojo invocation, compilation, execution, tests, static checks, identity
verification, quality assessment or timing was performed, as requested.
Source reading/editing and metadata generation are not qualification.

All new switches are OFF. They require IDENTICAL and respect
`MOJOLEARN_IDN_ALL_OFF`. No FAST route or numerical default was promoted.
Cross-version bits may change; each arm still needs common host, NVIDIA,
AMD and Apple arithmetic and full-task quality before any promotion.

| Card | Delivered source | Remaining scope |
|---|---|---|
| NN17 | Four adjacent query heads share K/V staging using 64 logical rows (16 positions/head). | GQA ratio divisible by four, HD64/QRES/PF non-swizzled reach required. Existing I06 two-head losses remain; all new evidence pending. |
| NN18 | Existing fused causal/window tile bounds reused and attributed. | No new isolated task-compaction implementation. Existing replay/refusal conventions retained. |
| NN19 | Existing I07 retained/recompute implementation reused. | Full model lifetime, capacity and E2E qualification; corrected component wins are insufficient. |
| NN20 | New common host/device stable-summary component: fixed key leaves, streaming adjacent-pair tree/odd carries, forward and canonical backward; A/B left-fold control. Existing v2 nonpromotion retained. | Model/profile/checkpoint/decode integration and complete admission remain pending; component B is not v1/v2. |
| NN21 | Eager score scale+mask fused while storing both stages and preserving +0 mask add. | Softcap/plants/sabotage use old path; no invented bias support. |
| NN22 | Eager dK/dV shared traversal with two original ordered chains. | Fused-route geometry extensions and complete model reach remain pending. |
| NN23 | Eager row dot followed by its dS consumers in one row owner. | Serial stores may lose; full-operation evidence required. |
| NN24 | Standalone RMS/centered LayerNorm profile with one/eight fixed logical lanes; common host/device forward, backward and parameter helpers/launchers. | Complete model profile migration remains pending; component B is not claimed to equal every incumbent LayerNorm derivative. |
| NN25 | Original RMS sumsq row fold followed by a flat cell scaling kernel. | Recomputed scalar cost and fused residual/non-RMS variants need separate coverage. |
| NN26 | Training SiLU+gate product fused, both saved stages retained; existing paired backward reused. | Saved-sigmoid variant not implemented. |
| NN27 | Q/K rotations share existing position-table cells in one launch. | Session-wide cross-layer table sharing/scan elision is not implemented. |
| NN28 | Explicit `retain_kv_cache=False` capability omits dead decode-cache writes while keeping every backward/trace stage. | Root integrates byte-LM full prefill callers; other training owners pending. |
| NN29 | Paired K/V window-ring writes; existing linear paired append retained. | Direct-to-final layout and multi-request batching pending. |
| NN30 | Existing shipped read-only backward entry gradient view attributed. | Further ownership transfer is pending; no new switch/default. |
| NN31 | Native retained-state byte/cost ledger and actual retain/release/replay choice, bounded lifetime and reset refusal for every outstanding ticket. | Full model traversal/group scheduling and memory outside retained forward stages remain pending. |
| NN32 | Native moved-in context/weights/RoPE owner, device input snapshot, actual forward/backward, generation tickets, mutation invalidation, abort cleanup and canonical parameter gradient output. | Public Samba/LM binding adoption and optimizer hooks pending; default full-prefill FP32 profile only. |

Machine-readable flags, both-arm settings, runtime reach and pending coverage
are in `attention.json`. NN21/22/23 require the eager attention route in both
arms; automatic fused-only execution must be recorded as no candidate reach.
NN17 requires the exact same non-swizzled forward/backward schedule in both
arms. NN28 changes only storage proven dead by an explicit training caller.
NN24 is component-only and must not be selected as a production model arm.

Future measurements must map each full workload to exact dataset/model
settings and hashes, include preparation, synchronization and consumed
outputs, and separate train/inference/prefill/decode/cold/repeated boundaries.
Use a frozen commit, one excluded warmup and one scored sample per vendor,
NVIDIA and AMD jointly for timing decisions, Apple and host for identity.
Reuse existing accepted compilation/identity receipts instead of rerunning
unchanged candidates. Retain prior I06 losses and I07 invalid-arm receipts.

No new unsupported Mojo feature was required by the implemented schedules.
If native graph capture, async offload or other missing toolchain support
is needed by later work, record an upstream ask and wait for Modular.

Keep logs out of context: save complete output to files, use targeted rg/grep
with bounded surrounding lines and short tails, and summarize exit status,
coverage, failures, and evidence paths. Expand only relevant diagnostic blocks;
never hide failures or infer full success from filtered output.

NN20 now has a separate source-only arithmetic component in
`transformer/experiments/attention_summary_tree.mojo`. NN31 and NN32 now
have concrete native ownership components in
`training/neural_attention_owner.mojo`; these are not public model routes.
No added source has been compiled, executed or verified. The checkpoint
budget covers explicit retained forward buffers and their GEMM workspace,
not total device memory or the transient current forward allocation.
