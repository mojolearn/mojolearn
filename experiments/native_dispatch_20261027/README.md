# Generation native dispatch lane, 2026-10-27

Status: source implementation and non-building checks complete; Mojo compilation,
public runtime qualification, within-release identity and performance pending.
`MOJOLEARN_GENERATION_NATIVE_SESSION=1` opts into the candidate at model admission.
It is off by default. No performance improvement is claimed.

## Concrete repair

The ordinary GPU transformer session-forward path previously packed Python lists
and called an admitted compiled extension with `py_call` on each layer/token.
The candidate passes native integer vectors to a cached C function pointer. It
does not use `PyObject_Call`, vectorcall, or CPython function-object dispatch for
that repeated call. Its table is obtained once through a documented versioned
PyCapsule, and its opaque session handle is admitted once per session owner.
The provider retains the owning extension, capsule and session. Replacing the
session reconnects its handle. No CPython or Mojo object layout is inferred.

The unchanged `_transformer_run_session` owns arithmetic. The original Python
entry and new ABI share its execution/cleanup helper. The ABI reserves `busy`
before releasing the GIL, executes without it, and finalizes after reacquiring
it on success and failure. Complete native error bytes return through a compiled
native sink, avoiding truncation and cross-library allocation/free. The caller
keeps all buffer owners until synchronization finishes.

The candidate targets fp32 GPU `transformer_session_forward` on Apple, NVIDIA and
AMD in `identical`. It does not select a different resident route, change any
model/kernel/settings, enable Apple `fast`, or replace CPU, Mamba, int15 or
distributed execution. Existing routes remain present. Metadata descriptors
are now constructed as native vectors; legacy entries receive Python lists at
their boundary. Generation still has other Python-object metadata and compiled
extension calls. This is not a claim that all generation is CPython-free.

## Physical provider/native boundary

This is an incremental provider boundary. Existing native code stays under
`transformer/`; there is no new runtime directory or whole-library folder move.
Migrating the rest of the library is separate work.

| Location | Responsibility |
| --- | --- |
| `providers/python/generation_session_pyapi.mojo` | Capsule validation, session admission, Python ownership, GIL transition, public error return |
| `transformer/session_abi.mojo` and `.h` | Versioned native function signature, borrowed vectors, opaque-handle and error contracts |
| `transformer/native_session_call.mojo` | Typed native invocation and native error storage; no Python/provider imports |
| `transformer/session_native_dispatch.mojo` | Native ABI producer and shared session execution/cleanup |
| `bindings/_mojolearn_transformer.mojo` | Existing public Python door; setup-only capsule and handle entries |

`tools/check_generation_native_boundary.py` traverses local Mojo imports from
the native modules, including the transformer kernel implementation. It refuses
Python/provider dependencies, CPython helper tokens and unresolved local imports.
It is a regression tripwire, not a whole-program proof: upstream std/MAX package
internals, indirect external ABI implementations and arbitrary obfuscation are
outside this source checker. Its tests include transitive/aliased imports and a
direct CPython C-symbol negative control.

Existing `adapters/native/abi.h` already gives Torch/MAX whole-model training a
native descriptor ABI. `adapters/torch/registration.cpp` owns tensor admission,
framework allocation and stream synchronization and calls
`mojolearn_adapter_train`; `adapters/max_ops/__init__.mojo` connects the same
whole-model entry. These are the precedents for shared native execution. Their
model/program descriptors and ownership rules should be consolidated deliberately,
not replaced with a second framework API or removed per-operation adapter ops.

Framework-specific capture/import/registration remains a separate concern.
`python/mojolearn/adapters/_torch_compile.py` still performs graph export in
Python and explicitly records this incomplete migration.
`autograd/torch_program_pyapi.mojo` admits Torch tensors and native synchronization
entries; its GPU program call still passes Python descriptor containers to an
admitted compiled entry. Its CPU program execution already calls native Mojo
`program_execute_run`. Native execution can share descriptors while providers
retain differing allocation, lifetime, device and synchronization contracts.
An MPS tensor is not a host pointer or a CUDA pointer; preserve its supported
host-buffer boundary until a supported direct interface exists.

## Remaining dispatch inventory (source frequency, not measured cost)

| Priority | Site | Classification and next scope |
| --- | --- | --- |
| 1 | `causal_lm_public._read` → `_run` → `causal_layer_run` | O(tokens × layers) calls to admitted compiled extension entries. Candidate repairs normal GPU session-forward only. CPU, int15 and Mamba entries need their own native contracts and coverage. |
| 1 | `_resident_forward_ok`, `causal_lm_resident_binding` | A final norm is required today. The byte-LM/162M checkpoint format declares `norm_name=None` (`models/config.py`), so it cannot use this resident route. Supporting an absent norm requires a separately qualified native resident change; inserting RMSNorm would change the model. No profile establishes that dispatch causes its timing regression. |
| 2 | `causal_model_embedding`, `causal_model_head` | Per-token compiled embedding/norm/head entry calls, host-buffer preparation and transfers remain. A whole-model native program is a larger repair than optimizing call envelopes. Preserve timed transfers and consumed outputs. |
| 2 | `small_mlp_native._native_step`, `_native_resident_step`, `_gemm_into`, optimizer/loss entries | Checked native extension calls per step or constituent operation; resident multi-step paths already fuse work. Trace selected route before estimating frequency. Reuse whole-model training descriptors where semantics agree. |
| 3 | `autograd/torch_program_pyapi` GPU `py_call(run, program, descriptors, addresses)` | One checked native program call, not an interpreted Python model loop. Provider allocation/device calls and synchronization surround it. Replace the program boundary after mapping owned descriptors to the existing execution plan. |
| 3 | `causal_lm_public` metadata and `generation_buffers.py_ints` | Native CPython list/dict/tuple/buffer operations still exist. These are not automatically Python callbacks. Immutable metadata can be retained; mutable cache positions, newly allocated outputs, exposed weights and externally mutable buffers need freshness/lifetime rules. |
| 4 | `parallel_call` in causal generation | `parallel_causal_initialize` verifies a compiled native router. It is not proof of an interpreted Python callback; RPC/process and host-buffer ownership make this a separate provider. This lane leaves it unchanged. |

The documented `Array.__len__` metadata callback is handled by the other lane.
This audit does not classify all `py_call` symbols as interpreted Python, and
does not claim that checking a builtin function type alone audits all its callees.

## Validation and integration requirements

Non-building checks: 25 pytest tests and 3 subtests passed (generation boundary,
native dispatch reach, existing execution boundary). Native local source closure:
193 modules, zero violations. Initial checker false positives and the initial
test assertion failure are retained in the evidence directory. No Mojo build,
GPU execution, rental, teardown, release or active measurement change occurred.

Root must compile the changed tokenizer/public and transformer groups once,
including native function-pointer list/capsule ABI and newly registered native
owner types. Then run the existing public generation tests with the candidate
off/on on Apple/NVIDIA/AMD `identical`, plus CPU regressions. The companion
`public_runtime_probe.py --output <report.json>` requires the freshly integrated
installed GPU build: it exercises repeated exact forwards, sampled tokens,
state positions, reference stability, a poisoned old-dispatch negative control,
closed-session refusal and owner replacement. It fails when the candidate API
is absent; it was authored and syntax-checked, not executed in this lane.

Additional required checks: malformed/wrong-version capsule admission, wrong
session type, busy/concurrent calls, failures during/after kernel execution,
all buffer owners and provider/module teardown, optional transformer option
tails, checkpoint/resume, exposed-weight mutation, and int15/Mamba/CPU unchanged
routes. Compare retained output hashes across all three supported vendors within
the same source freeze, reusing unaffected evidence. Tiny probes do not qualify
full-workload performance or every error path.

Before promoting this candidate, freeze and run full affected saved workloads
on NVIDIA and AMD with the switch off/on, same models/data/settings/timed
boundary, one excluded warmup and one scored sample under owner policy. Include
the normless byte-LM/162M workload and saved nonresident streaming/sampling
transformer recipes; map exact dataset/version/hash, shape, seeds and numeric
settings first. Measure fit/inference/cold/repeated usage where affected, retain
quality and failures, and apply the existing combined-vendor/noise rules.
Resident-only runs cannot establish this candidate's benefit. No opponents need
rerunning solely for a private dispatch change when their comparability evidence
is still valid. Apple identity remains required; Apple timing does not vote.

Evidence: `~/mojolearn-evidence/native-dispatch-20261027/`.
Primary API reference: [CPython capsule API](https://docs.python.org/3/c-api/capsule.html).
