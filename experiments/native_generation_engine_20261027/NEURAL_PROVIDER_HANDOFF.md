# Neural provider lane checkpoint — 2026-10-27

Branch: `lane/neural-provider-completion-20261027`.
Worktree: `/Users/andrewhendel/mojolearn-wt/neural-provider-completion-20261027`.
This handoff is committed alongside the current WIP. Its enclosing commit is the checkpoint identifier.
Root owns integration, compilation and runtime. This lane ran source checks only; no compiler, runtime, GPU, measurement or release command.

## Latest pushed tranches before this checkpoint

- `17524f78f6`: CPU inference/decode provider, ten original CPU helpers relocated byte-for-byte.
- `414dc1d55c`: packed GPU Transformer/Samba provider, Transformer table 36 operations; fixes extracted CPU CausalLM route lookup.
- `76ef5c0bd6`: separate CPU training-host backward/admin/owned-weight provider tables, preserving each original numerical owner.
- `f9740239e0`: CPU Mamba public fresh/state admission and training-host state table.
- `54b66b950f`: generation uses common per-layer native execution lifecycle; removes old pre-reservation loop. The pure native engine enters GIL-held; common endpoints release it around numerical calls/native lock waits.
- `fc617ceb09`: shared `admit_native_buffer_exporter` before buffer pinning; exact native exporters, recursive memoryview owner check, static ndarray check, direct ctypes arrays without buffer hooks.
- `a4d8037930`: generation pins foreign-buffer metadata once, delayed native copies/writeback, shared alias counters and deferred admission errors.
- `a569c7fba7`: CPU Samba primitive/training/optimizer provider; nine original primitive bodies relocated without numerical edits. CPU carried Transformer and Samba training routed through provider candidate.

## Current WIP in this checkpoint

Extends the native generation engine to Mamba1/2 on CPU/GPU, including state descriptors, Mamba2 reports, counters and foreign-buffer shapes. Source tests reported 18 passed before this checkpoint. This is NOT yet ready for activation.

**Known required correction found immediately before the checkpoint instruction:** the existing local `_run_snapshot` always passes `False` for block `step`, even when `_run_prepared(..., remote_step=True)` is called during generation. `remote_step` controls distributed execution only. Current WIP instead selects decode operations when `progress.produced != 0` and forwards `remote_step` to the native engine. Remove that local route change: preserve the original forward operation at length 1. Do not infer equivalent bits. Update the authored source assertion to enforce the original local route. No correction was made after the user's immediate checkpoint instruction.

## Evidence and validation limits

Continuation correction (2026-10-27): removed the local decode selection and
the engine's `step` argument. Both direct local execution and streamed token
continuations retain the original forward operation and its length parameter.
The eight focused generation source checks pass; this correction remains
uncompiled and has no runtime or numerical qualification yet. Full check log:
`/Users/andrewhendel/mojolearn-evidence/provider-migration-resume-20261027/mamba-route-source-tests.log`.

All full logs retained under `/Users/andrewhendel/mojolearn-evidence/neural-provider-completion-20261027/`.

- `packed-provider-source-tests.log`: 32 passed.
- `cpu-training-source-tests.log`: 34 passed.
- `cpu-state-source-tests.log`: 35 passed.
- `generation-lifecycle-source-tests.log`: 15 passed.
- `exporter-admission-source-tests.log`: 3 passed.
- `generation-buffers-source-tests.log`: 19 passed.
- `cpu-primitive-source-tests.log`: 36 passed.
- `generation-mamba-source-tests.log`: 18 passed, WIP semantic issue above remains.
- Corresponding `*-layout.log`: source architecture guard, not runtime validation.
- `host-helper-relocation.json`, `cpu-primitive-relocation.json`: original-body relocation hashes/indent comparison.
- `*-commit.log` / `*-push.log`: complete Git/check-hook outputs.

The source tests check architecture/contracts, not execution or numerical identity. Root integrates/compiler-fixes separately; do not overwrite those fixes by replacing entire files from this worktree.

## Remaining scoped inventory

- Correct current Mamba generation route issue above; compile latest source and validate actual public paths through root.
- Packed generation head and distributed generation remain explicit engine refusals; legacy/session-fresh-off routes still require audit/migration. Ordinary packed Transformer/Samba numerical paths have provider operations.
- Provider/default selectors remain experimental; root controls activation. Baseline toggle is retained for the requested Mac comparison. No completeness or performance claim.
- Samba foreign weights still resolve early; extend delayed borrowed-buffer descriptors to preserve weight/state aliases across layers and microbatches.
- Audit exact one-time metadata admission end-to-end: eligibility preflight plus admission still duplicates reads; some binding-side ownership/optimizer preparation/publication loops still use Python objects/CPython builtins.
- Audit binding replacement, old session owners, lock ordering, session close/invalidation, failed partial state, and actual Session/tape public transitive paths. Do not equate a Python-free numerical folder with a Python-free public operation.
- Separate CPU host tape APIs retain original supported behavior; immutable tape candidate availability differs by host binding. Confirm all supported public routes in root validation.
- Framework lane owns Torch/MAX/ImportedProgram/ByteLM and graph metadata. Library lane owns classical and shared fit lifecycle. No lane compilation, no measurements, no GPU validation, no release.

Keep logs out of context: full outputs to files, targeted bounded excerpts, report exit status, actual coverage, failures and evidence. Never claim runtime success from source checks.
