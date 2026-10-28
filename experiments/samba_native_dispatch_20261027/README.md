# Default Samba dispatch continuation, 2026-10-27

This continuation starts at integrated `c295db1c5` and preserves the disabled-by-default `MOJOLEARN_NEURAL_NATIVE_DISPATCH` candidate. Shared source remains vendor-neutral; only local Mac compiled validation is requested. No kernels, algorithms, settings or resident defaults change.

## Audit result and repair

Default board Samba dropout is zero. Its stateless forward/train-step may take existing whole-model resident execution. Stateful decode uses ordinary carried-state Mamba3 plus attention blocks and embedding/norm/head. The preceding descriptor candidate already covers those numerical entries. The owned Mamba3 route requires `MOJOLEARN_NN40_OWNED_WEIGHTS`, an opt-in compile flag absent from the inspected default build scripts; it is not evidence of an uncovered default recipe. No unused owned-session/tape ABI is added just to increase coverage counts.

A real repeated setup remained: `neural_block_owner` constructs a temporary CausalLM-shaped owner and re-enters admission for each standalone block call and each Samba attention layer. That admission reconnected a provider and consequently its native session handle. The continuation stores only immutable typed providers on the retained block and reuses them for the exact admitted module, compiled factory and ABI name. Transformer also requires the exact session-admission function. Module/capsule/factory references remain retained. Changed factories reconnect through the normal capsule checks. Existing route, numeric mode, vendor, arithmetic profile and native-function checks still run before reuse; no shape, dimension, weight or mutable state is cached here. Candidate off still selects the original routes.

The source audit also found a connection-list omission in the preceding candidate: Samba's import-time symbol list did not expose `neural_primitives_api_v1`; its plain entries namespace lacked the canonical module used for identity matching. The Mamba training symbol list omitted `mamba_state_api_v1`. Those import-only wiring gaps are repaired. This adds no Python product execution. The previous 29-row map is corrected: standalone Transformer/Mamba1/2 **do** enter causal admission through `neural_block_owner`; saying they never connected the candidate was incorrect. Their benchmark controller's per-token calls remain application/testing work, while Samba's package-owned layer loop is runtime work.

## Remaining scope

Repeated Python-backed metadata is handled by the separate operation-local metadata lane. Optional owned Mamba3, int15, tape, nonresident Samba training CE/dropout/optimizer calls keep their original behavior; they have not been converted by this patch. No existing whole-native route needs a cosmetic wrapper. The default recipe coverage map is source analysis, not proof of runtime path selection or an all-29 Python-free result. No profile establishes a timing gain.

## Validation and integration

35 source tests plus 3 subtests pass across provider reuse, descriptor/ABI source architecture, native dispatch reach and execution boundary tests. New `public_provider_probe.py` is AST-valid but not executed by this lane. It exercises mixed Mamba3/attention Samba decode, bitwise outputs and state counters, stable provider/session-link identity across calls, a changed compiled factory on the same module, refusal of a Python factory without invoking it, and recovery. Root must execute it against the rebuilt installed/local development runtime; source tests do not qualify ABI behavior.

Root's c295 compilation separately found missing `py_get` import in `bindings/samba_training_native.mojo`; root is integrating that fix. This branch intentionally does not duplicate that concurrent edit. Integration needs the changed import-time Python bridge files alongside the compiled tokenizer owner. Training/Mamba/Transformer numerical source is unchanged in this continuation, so reuse accepted group compilation when its complete inputs match. No vendor runs, release, default promotion, active campaign changes or full-workload timing were performed here.

Evidence: `~/mojolearn-evidence/samba-native-dispatch-20261027/`.
