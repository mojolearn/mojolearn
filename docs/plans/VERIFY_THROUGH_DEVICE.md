# Verify through `device=` (Andrew, 2026-11-24)

Decision: `python -m mojolearn verify` drives every public endpoint through the public `device=` parameter, once
with `device="cpu"` and once with `device="gpu"`, takes the output digest and compares it with the reference. That
is the one place the CPU and GPU sides of an endpoint are checked. It gates the release: it changes what the
release's one verification runs.

State on `lane/w3-integrate` 3cda8a05b3 (coverage audit of the round-6 gate):

- Verify selects the CPU side with the environment variable `MOJOLEARN_VENDOR=cpu` in a separate process
  (`python/mojolearn/_verify_recipe.py`); no lane passes `device=`. The public parameter is tested only by
  dedicated pytest files (about 99 files mention it).
- Public functions (metrics, cross-validation tools, splitters' helpers) have no `device` parameter.
- 4,494 public endpoints: 1,591 mapped to a lane, 953 not numerical, 1,950 unmapped. Unmapped with numbers:
  `score` 73, `fit_transform` 49, `fit_predict` 16, `state_dict` / `load_state_dict` 29 each; the rest are
  `get_params`, `set_params`, `check`, `digest_`, `training_digest_`, attribute access and pickling across about
  200 classes. 70 lanes have their reference OWED.
- Autograd, ops and training: about 230 endpoints, 178 mapped, 45 unmapped, all through the environment switch.
- Fault injection exists: `tools/coverage_sabotage.py` (1,369 native registrations have a sabotage arm),
  `_verify_api_faults.py`, `_verify_light_faults.py`, `verify --self-test`.

## Work

1. **`device=` on every public function** that computes (metrics, model-selection tools, anything else the
   inventory finds). `device=None` keeps today's choice; `"cpu"` and `"gpu"` select the side. The word is read in
   UNPACK and becomes a typed Mojo value; nothing is looked up per call. A function with no GPU implementation
   refuses `device="gpu"` with clear words: it never runs the CPU route in its place (the selector refuses, never
   widens). Classes whose constructor lacks `device` gain it the same way.
2. **Verify passes `device=`.** Each lane runs its endpoint with `device="cpu"` and `device="gpu"`; the digests are
   compared with the reference columns as today. The `MOJOLEARN_VENDOR=cpu` switch inside verify goes in the same
   change; the variable itself stays in the product (see the inventory below); its 80 verify-only sites go ("remove what a change made
   unnecessary").
3. **Coverage.** Every public endpoint with a numerical result gets a lane where reasonably possible, in this order:
   `score`, `fit_transform`, `fit_predict`, `state_dict` / `load_state_dict`, pickling round trips, the 45 autograd /
   ops / training endpoints, then `digest_` / `training_digest_` / `check`. `get_params`, `set_params` and attribute
   access are covered by one generic lane per mechanism, not one per class. The 70 OWED references are recorded.
   What is left unmapped is listed with its reason.
4. **Fault injection.** Every new or changed lane has a sabotage arm that it catches (a fault in the code the lane
   reaches moves the digest), on the CPU side and on the GPU side. A lane that cannot fail is not a lane.
5. **Delete what verify now covers.** The dedicated `device=` pytest files and the pytest files that re-check an
   endpoint's numbers go in the same merge. Kept in pytest: refusal of a bad device word, of a device array handed
   to a CPU model, of callbacks; exception types and words; returned object types.

## What the inventory found (lane vd-inventory, 2026-11-24; tables in `docs/plans/verify_through_device/`)

- `MOJOLEARN_VENDOR` stays in the product: it chooses cuda or hip and the CPU-only set, and the native process pool
  writes it into each child (`bindings/native_process_pool_door.mojo:163`). Of its 362 sites, 80 go once verify
  passes `device=` (`vendor_switch_uses.tsv`); the rest stay. Item 2 removes those 80, not the variable.
- Two sabotage patches fault one side by reading that variable. With both sides in one process they fault both or
  neither: re-cut each so the fault sits in the code of one side.
- `mojolearn.check` (`_check_both.py`) starts the same CPU child process as verify: it moves to `device=` with it.
- 43 `parallel_*` entries take `devices=` (a GPU list) and have no CPU side: they keep `devices=`, verify takes
  their one digest, and they are listed as single-sided.
- 197 functions and 75 computing constructors have no `device` (`functions_without_device.tsv`): lane vd-functions.
- `device="cpu"` is refused by 15 estimator classes (9 not connected, 6 compiled constructors without `device`):
  connect them in lane vd-functions.
- 210 endpoints have no CPU side and 152 no GPU side: verify takes one digest and the table names the missing
  side; each missing side is an owed implementation, listed, not hidden.
- Fault arms: 655 mapped endpoints have an arm on the CPU side only (GPU arms reach 258 lanes); 96 have none
  (`ops.Session` 85, `ops.Program` 9). Item 4 closes both.
- Unmapped: 1,341 endpoints fall to five generic lanes (params 391, digest 391, attribute access 251, `check` 196,
  pickling 78); 606 need lanes of their own (`score` 73, `fit_transform` 49, `call` 43, state dicts 58, ...).
- pytest: 1,332 files, 11,147 tests: 601 tooling, 511 boundary, 181 duplicate, 39 device. The classes are a first
  pass from names and contents: each duplicate file is read before it is deleted.

## Lanes

- `vd-inventory` (reads only): per public endpoint: has `device`, CPU and GPU implementation, lane, sabotage arm;
  per pytest file: boundary (stays) or duplicated by a lane (goes).
- `vd-functions`: item 1.
- `vd-verify` (after the round-6 fix lanes land, they edit the same verify files): items 2 to 5, with sub-lanes per
  family.

Done when the coverage audit shows no numerical endpoint unmapped without a listed reason, every lane's two
digests match the reference on CPU and NVIDIA, the sabotage run catches every arm, and the verify smoke
(`tools/ops/wheel_verify_smoke.sh`) passes from the built wheel.
