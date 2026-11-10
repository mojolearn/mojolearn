# Samba optimizer owner integration

This repairs Samba's consumers of the migrated public AdamW owner. Its direct
construction and restore formerly initialized retired Python control fields;
steps and snapshots then read those fields and bypassed the native owner.

Construction and restore now install OptimizerOwnerControl through the existing
OptimizerOwner boundary wrapper. An OptimizerOwnerLease spans each step,
snapshot and restore, including GIL-released calls. The native control stores
the actual handle, driver, freshness, pristine and parameter/gradient residency
state. The wrapper pins its compiled provider until close succeeds. Opening a
slot attaches this owner before moment upload; failed initialization closes it
without abandoning an unowned handle. A previously owned slot refuses a
different compiled provider. Public optimizer finalization now owns Samba's slot
through the same lifecycle as standalone AdamW.

Open/download/upload/close use OptimizerNativeDriver. The existing packed
optimizer operations 25–27 remain in the same native primitive descriptor and
call per_call_step/resident_step/resident_step_io (CPU uses its existing
optimizer_step_host_into). They receive the typed SambaOptimizerPlan from
core/samba_optimizer_plan.mojo. No high-level PublicStep repacking, new tensor
copy or numerical algorithm is introduced. The model/session kernel choices,
resident and accumulation switches, uploads/downloads and optimizer IO masks are
retained. The legacy resident session is admitted to its existing native
descriptor when that session path is selected; the model's native-dispatch
switch itself is unchanged.

Writable moment aliases retain the same preparation and copyback behavior.
Snapshot downloads into admitted owners; restore uploads new moments before
publishing those owners. Parameters, offsets, initialization flags, moments,
outputs and provider images remain pinned through native calls. Successful
training marks the native optimizer state non-pristine, including the path that
materializes moments without a resident slot. This prevents a later resident
open from mistaking existing moments for initial zeros.

Removed: Samba's _res/_res_binding/_where and control-flag mirrors, resident
optimizer Python entry dispatch, the Python optimizer parameter list, its
provider-side list decoder and the temporary Python moment-owner list. Public
_resident_wanted remains the actual constructor parameter. Two import-time
capability booleans retain the original optional step-IO/maximize routing.

The live bindings/providers/Python consumer review includes _res, _where, _KIND
and all six retired control flags. Remaining matching names belong to CNN batch
normalization, matrix resident helpers, model-selection stores, tree/ANN kind
metadata, and the optimizer's intentional external legacy-archive admission.
Full source evidence is retained under
~/mojolearn-evidence/boundary-ridge-native-20261110/optimizer-all-control-name-followup.txt.
This source review is not runtime or performance evidence.

Root owes the training owner/host and classical/tokenizer dependent closure
compilation, installed Samba constructor/step/state/rebound-owner/device-owner
tests (test-lane commit 25ca9e4679), and native owner lifecycle coverage. GPU
performance is unmeasured. No lane compile/test/identity/timing or box work ran;
no A/B define or kernel-bit change is claimed. Broader Samba configuration,
model-state and non-optimizer boundaries remain separate migration work.
