# After 0.8.63: verified once counts everywhere

Owner order (Andrew, 2026-10-31): work verified once is reused automatically, and publishing never
needs a waiver for verified work. 0.8.63 shipped with `docs/release/0.8.63-owner-waiver.json` and
`docs/release/0.8.63-identity-reuse-waiver.json` only because the tools below were not connected.
Each item ends when its acceptance holds; no item reruns verified work to prove itself.

## 1. The release ledger reads the evidence that exists

`tools/release.py` and `tools/release_requirements.py` count only their own per-lane identity runs
and `lq` gate receipts. Make the ledger admit, as passes with provenance:

- the committed identity table admission (`bench/results/reference_admissions/<date>/`,
  `bench/results/identity_break/retained/`) for every `identity-<vendor>/<lane>` row it covers;
- the endpoint and light references (`_verification_api_reference.py`,
  `_verification_light_reference.py`) for the public-API rows;
- the installed wheel self-test receipt (`self-test-<platform>/self-test.log`) for the platform's
  candidate row.

Acceptance: a release whose admitted table and receipts cover its rows passes `release.py`'s
pre-publish check with no `--owner-waiver`.

## 2. Reuse without stops

- A finished leg whose begun compile runs were never recorded is recorded automatically from its
  own output (0.8.63 needed `record_compile_requirements` by hand for cuda-sm_89 at 9e79e6e65085).
- `cross-compile` reuses its pass when no kernel source changed since the passing commit (it reran
  on every 0.8.63 freeze, about 30 minutes each).
- A build-tooling change that cannot alter compiled bytes (0.8.63: `packaging/portable_math/wheel.py`
  audit logic) does not invalidate a whole-leg take.

Acceptance: refreezing a release for a Python/data-only commit launches no build and no cross-compile.

## 3. Identity attribution is fine enough to reuse

`tools/lane_select.py` still marks 32 lanes per GPU column (via `host_model` routes) and 432 on the CPU
(via the generic `classical_native_call` dispatcher) for a change to one host k-means file. Resolve the
dispatcher's operations per lane and the `host_model` exports per lane, each as a derived rule with a
refusal test. Also attribute `adapters/max_ops/__init__.mojo` (built only into the MAX ops package).

Acceptance: `identity_columns.py changed-lanes --since 0412c8ee2f` reports only the k-means host family
on cpu and nothing on metal/cuda/hip; admission accepts the 0.8.63 records with no reuse waiver.

## 4. The GPU gate needs nothing outside the repo

The NVIDIA/AMD gate runs through `~/mojolearn-ops/lq`, a script outside this repository, and its AMD
runner (`amd_runner.sh`) no longer exists anywhere. Move `lq` and the box helpers into `tools/ops/`,
run the gate through the supported job wrappers (`tools/nvidia_central.sh submit`,
`tools/do_amd_steward.sh run`), and build the autograd gate executables in the same job.

Acceptance: `release.py` runs both gates on freshly rented boxes with no file outside the repo.

## 5. Known product debt found during 0.8.63 (not blocking)

- The 18 `par-*` multi-GPU drivers fail on one device ("native provider requires an exact compiled
  module"); multi-GPU verification is retired, the features remain undocumented.
- Two compiler issues are recorded in `docs/MODULAR_ASKS.md`: host bytes depend on the build machine's
  CPU, and the Linux x86-64 host build dropped the k-means++ seeding compare (worked around in source).
- 10 `lane_select` tests and several packaging tests fail identically on clean main.
