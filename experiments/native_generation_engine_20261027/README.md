# Native operation-local generation engine (2026-10-27)

This source candidate removes Python-object state and CPython dispatch from the
nonresident fp32 Transformer token/layer execution loop. It is **off by default**:
load the model with `MOJOLEARN_NEURAL_NATIVE_DISPATCH=1` and
`MOJOLEARN_GENERATION_NATIVE_SESSION=1`, then select
`MOJOLEARN_GENERATION_NATIVE_ENGINE=1` for public forward/read/generate operations.
No compilation, product runtime, speedup, or whole-29 completion is claimed here.

The existing common primitive descriptor, checked Transformer session table,
shared buffer admission, stable native error/result storage, and existing sampler
are reused. There is no new public adapter and no arbitrary extension-function
unwrapping. Admission is added to the existing Python ownership boundary;
`transformer/native_generation_engine.mojo` has a transitive native-only closure.

The provider retains model/weight/cache/session owners for one operation, admits
route and buffer metadata, and reserves each distinct session under the GIL. The
native engine owns hidden/logit/token storage, position/cached-token counters,
layer calls, done decisions and sampling. On return or failure, the provider
reacquires the GIL, releases its reservations and publishes progress. Layer
admission/reservation errors are retained as native strings and raised at their
original execution position: earlier layers still advance before a later refusal.
Repeated blocks may share a session; that owner is reserved/released only once.
No session owned by another call is unlocked. Dynamic capacity refusal stays
outside execution invalidation. A stream's last successful token progress survives
failure, and the public stream remains unusable after a failed read as before.

The sequence is unchanged: embedding, ordered Transformer session calls, optional
final normalization, head, then the existing sampler. No norm is inserted into
normless models. The engine adds native buffer allocations and a returned-array
copy inside the public operation; removing CPython does not establish a speedup.
Admission still uses native CPython metadata operations and some redundant route
preflight reads; it is not a claim of literally one field load at every boundary.
Those operations are outside repeated token/layer execution. The Python wrapper
and public consumed-output semantics are unchanged.

## Coverage and remaining work

`coverage.json` carries all29 recipe IDs and a separate route map. This tranche
changes only nonresident fp32 Transformer CausalLM execution. The normless162M
route can use that engine without changing model math. SmolLM2 normally selects
existing whole-generation resident execution; runtime route evidence is needed.

Explicit selection refuses distributed, resident-per-forward, packed/int15,
foreign weight/state and non-Transformer branches rather than silently measuring
mixed orchestration. Their ordinary supported defaults remain available. CPU has
no admitted sibling table here. Mamba1/2/3, Samba repeated orchestration,
Transformer fresh/backward dispatch, tape and other recipe owners are **not** made
complete by this change. Existing resident whole-operation numerical owners are
not new provider wrappers. Text tokenizer/chunk processing also remains separate.

The whole-operation session reservation prevents concurrent mutation/close from
interleaving during a read. Existing busy/closed refusal and ownership must be
checked under actual concurrent public calls; source checks cannot qualify them.
The API does not gain a guarantee for concurrent mutation of public model fields.
Snapshots refresh for every public operation/read.

## Local integration validation

Root compiles changed source closure(s), including tokenizer host binding and
Transformer sibling ABI, then runs `public_runtime_probe.py --output <report>`
against the rebuilt local Mac runtime. It is authored, not executed in this lane.
The probe compares bits and counters over8 forwards, multi-token reads/resume,
owner replacement, later-layer cache refusal/partial progress, recoverable capacity
refusal and closed-session replacement. It checks the actual private native-route
diagnostic. Run the retained native-session return regression as appropriate for
changed closure; inline output-owner corruption must not recur.

Concurrent contention, tied-session alias execution, allocation/error injection,
all existing supported routes and other vendors remain unqualified. The authored
probe is bounded correctness, not performance. Full-workload evidence remains
required for performance claims/default promotion. No new measurements should
silently use a mixed route while the owner gate requires complete native execution.
