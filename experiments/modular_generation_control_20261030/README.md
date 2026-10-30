# Modular generation controls — 2026-10-30

This experiment separates numerical sampling policy, the selected execution
route, and the existing KV upload-retention candidate. It makes no performance
claim and promotes no default. Slower and failed measurements remain part of the
experiment record. A correctness failure is repaired or explicitly rejected
before the affected configuration executes again.

The default remains `MOJOLEARN_EXPERIMENT_GENERATION_ROUTE=auto`, preserving the
existing route precedence. `MOJOLEARN_EXPERIMENT_RESIDENT_SAMPLING_FP32=0` and
`MOJOLEARN_EXPERIMENT_SAMPLING_KV_UPLOAD_REUSE=0` remain the default-off candidates.
Numerical policy belongs to the stream and its checkpoint. Route and upload
policy are captured at each public read; changing them does not reinterpret the
stream's numerical policy or RNG position.

## Controls and actual applicability

| Explicit route | Sampling policy | KV toggle | Actual behavior |
| --- | --- | --- | --- |
| `auto` | Existing greedy / Float64 / FP32 | 0 or 1 | Existing resident-first precedence; witness reports the branch that executed. |
| `legacy` | Greedy, Float64, or FP32 | 0 | Existing native per-forward loop and host selector; resident forward reloads KV each step. |
| `legacy` | Greedy, Float64, or FP32 | 1 | Skip subsequent intra-read KV reload batches only after private/disjoint state and unexposed-weight admission. Successful-step exports remain unchanged. |
| `resident` | Greedy or FP32 | 0 or 1 | One admitted resident native token loop. KV is already retained within the read. The toggle has no additional effect; there is no fabricated off-arm upload. |
| `resident` | Float64 sampling | 0 or 1 | Unsupported; explicit admission refusal, not a silent fallback. |
| `native_engine` | Greedy, Float64, or FP32 | 0 or 1 | Existing native engine with per-endpoint transitions; this KV candidate is inapplicable. Transformer runs require the existing resident-forward opt-out described below. |

The explicit selector takes precedence over `MOJOLEARN_GENERATION_NATIVE_ENGINE`.
That older setting still governs the existing `auto` branch. Explicit `legacy`
therefore makes FP32 host selection and upload reuse independently testable; it
selects an existing route and does not add transfers to the resident route.

`native_engine` retains its existing admission contract. In particular, a model
that qualifies for the resident Transformer program is refused by engine
admission unless `MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD=0` selects the supported
nonresident configuration. Distributed execution and legacy session setup retain
their existing refusals. Mamba is not a Transformer KV-cache implementation;
this candidate does not infer an upload optimization for it.

The saved mojolearn-162m checkpoint has no final norm. Its resident route is
inadmissible under this freeze's unchanged default-off
`MOJOLEARN_IDN_LM_RESIDENT_GENERATE` compile-time switch. Source support behind
that switch is not evidence that this binary executes it. Enabling that separate
route would require consistent host/GPU compiler inputs and new qualification;
this experiment does not change the switch or add a synthetic norm. SmolLM2
provides the currently applicable resident workload.

Explicit `resident` also refuses incompatible/aliased state, unsupported entries,
and exposed sampling weights instead of silently executing the legacy route.
Automatic selection retains its existing safe fallback behavior. A one-step
legacy read cannot remove an intermediate reload; its status is `single_step`.
A multi-step read that stops after its first selection reports
`eligible_no_second_forward`, not an executed reuse benefit.

## Ownership and implementation

`core/generation_kv_policy.mojo` contains separate scalar route controls and KV
policy. The shared `_resident_acquire_read` helper in
`bindings/causal_lm_public.mojo` owns session acquisition for greedy and FP32
resident reads, independently of either selector. It always reloads host KV at
PUBLIC read boundaries, even when session pointers and shapes match. Mutable
bytes are never treated as stable because their pointers match.

The retained legacy loan remains bounded to one read. Its existing disjoint-span,
parameter-exposure, cache-generation, and busy-owner checks are preserved. Live
parameter exposure invalidates the loan before another reused forward; actual
reuse before invalidation is recorded separately. Shared acquisition writes
owners into caller-held slots before a GIL-releasing load or close, preserving
failure cleanup when any individual operation fails. Publication, committed
sampling recovery, checkpoint copies, and private result ownership are unchanged.
No algorithm computation or callbacks are moved into Python or provider wrappers.

## Executed-route evidence

Set `MOJOLEARN_EXPERIMENT_GENERATION_ROUTE_TRACE=1` only for an excluded warmup.
The backwards-compatible `MOJOLEARN_GENERATION_ROUTE` prefix reports executed
`route`, `capacity`, `produced`, and `retained_kv`. Appended fields include
`requested_route`, `engine_requested`, `sampling_policy`, `greedy`,
`kv_requested`, `kv_policy`, and `kv_uploads_skipped`.

`retained_kv=1` now requires a reload actually skipped, not merely an eligible
loan. `kv_uploads_skipped` counts whole per-forward KV reload batches avoided;
it is neither a count of individual DMA calls nor a byte count. Intrinsically
resident reads report `already_resident` and zero for this legacy counter.
Failures stay failures in raw harness results; absence of a successful route
witness does not establish a completed execution.

## Native loop and CPython boundary audit

All three generation drivers are native Mojo loops, but that fact alone does
not establish equal boundary costs:

- `legacy_forward` performs native CPython metadata/C-API work and provider
  calls between tokens. Its host selector is native, not Python computation.
- `native_engine` keeps its traversal and selector native but the admitted
  endpoint still manages CPython GIL release/reacquisition per call. It must not
  be described as eliminating those transitions from the full token operation.
- `resident_greedy` and `resident_sampling` admit/pin owners and create one
  native session group before entering the complete resident read. The token
  loop runs inside the admitted native operation, rather than returning through
  the CPython-facing provider for each token. Boundary setup/publication still
  uses CPython C APIs and may release/reacquire the GIL while loading/exporting.
  The claim is about the interior token loop, not zero CPython use end to end.

Relevant boundaries are `providers/python/transformer_resident_pyapi.mojo`
(`_transformer_resident_lm_execute`), `providers/python/neural_descriptor_pyapi.mojo`,
`transformer/native_generation_engine.mojo`, and the common-session descriptor.
The next NVIDIA harness snapshots existing provider operation counters outside
the clock, alongside route witnesses. This distinguishes observed native-group
execution from source-only claims about endpoint transitions; it does not invent
Python callback counts or measured GIL time.

## Validation and measurement scope

The supported development selection is `host:tokenizer`, standard optimized
settings through the machine compile slot. Unchanged groups are reused only with
matching source closure, compiler, flags, target, and dependency receipts.
Focused installed GPU tests cover explicit dispatch, real skipped-reload counts,
no-effect resident combinations, unchanged parameter/sampling semantics,
checkpoint/read transitions, mutation visibility, failure cleanup, and existing
alias/owner contracts. These fixtures do not establish speed or full identity.

The next NVIDIA-only experiment uses both retained full checkpoints and the
actual read-128/read-1 generation boundaries. Planned meaningful cells are:
legacy Float64/FP32 × KV off/on; resident FP32 × KV off/on (no effect retained);
supported native engine Float64/FP32 × KV off/on (inapplicable KV retained);
and the intended combined automatic configuration. Every invocation records the
resident-forward selector and the frozen source. Unsupported resident Float64
is an admission negative control, not a manufactured workload.

Full-workload A/B, completed/failed cells, raw outputs, quality, actual route,
versions, source and artifact hashes, and retention receipts must remain visible.
No NVIDIA-only timing result promotes a default without the required AMD,
quality, and within-release identity evidence. Existing active frozen rounds
are not mutated by this next-round implementation.

The aea1d9a candidate passed 32 installed GPU checks, then failed actual
concurrency checks: shared-model reads replayed a stale busy flag and overlapping
reads of one stream segfaulted. Those sources, logs and state bytes remain
retained; no candidate timings were started. The repair candidate removes stale
lifecycle replay and claims the original stream before capturing its native
owners. Read, info and checkpoint calls now refuse overlap on the same stream;
independent streams retain separate claims. Frozen repair `99c25a6d` passed
both selected optimized Linux compilations, all 43 installed NVIDIA checks
(zero skips), and four isolated concurrency probes. Independent streams on
legacy/resident routes preserved serial outputs, checkpoint/info and cleared
ownership. Overlapping reads on the same stream were refused while the owning
read remained correct. This proves overlapping public-call ownership, not
simultaneous GPU kernels. The raw prerequisite archive (including state arrays)
is recorded in `experiment.json`. The full-workload frozen matrix is running;
AMD/Apple qualification and promotion evidence remain pending. Defaults have
not changed, and the original aea failures remain retained.
