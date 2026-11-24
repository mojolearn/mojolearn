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
   change; if nothing else needs the variable it is deleted from the product too ("remove what a change made
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

## Lanes

- `vd-inventory` (reads only): per public endpoint: has `device`, CPU and GPU implementation, lane, sabotage arm;
  per pytest file: boundary (stays) or duplicated by a lane (goes).
- `vd-functions`: item 1.
- `vd-verify` (after the round-6 fix lanes land, they edit the same verify files): items 2 to 5, with sub-lanes per
  family.

Done when the coverage audit shows no numerical endpoint unmapped without a listed reason, every lane's two
digests match the reference on CPU and NVIDIA, the sabotage run catches every arm, and the verify smoke
(`tools/ops/wheel_verify_smoke.sh`) passes from the built wheel.
