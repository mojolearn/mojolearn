# Eigh stable rotation cache, 2026-10-04

Baseline is freshly fetched main `d4bb2b795`; private branch
`lane/apple-fast-eigh-cache`. The four tangent/Rayleigh commits through
`131a0d78a` were cherry-picked onto that baseline, then the new opt-in was
added. Main defaults and IDENTICAL remain unchanged.

Manager measured repaired tangent on M3: 43796.817 -> 51254.836 ms (+17.03%),
eigenvalue error 6.08669e-5 -> 3.52564e-7, residual 5.37499e-5 -> 4.82917e-6.
Quality passed, speed failed. This is now recorded beside the source switch
and in EXPERIMENTS.md; it is not a default candidate on those measurements.

`MOJOLEARN_EIGH_TANGENT_CACHE` implies the repaired tangent/Rayleigh path,
FAST + Apple only. The old fused round recomputed the same pair coefficient
(sqrt/divisions and diagonal loads) for every matrix block and every V row:
O(n²) coefficient calculations per round for only n/2 distinct pairs. The
new first kernel computes those n/2 coefficients in parallel. The second
uses the same stable FMA rotation expressions and cached coefficients.

All cross-block coefficient dependencies are now immutable. Each matrix
pair-of-pairs owns a disjoint 2x2 block and its mirror, and each V row/pair
owns disjoint entries, so the update is safely in place. This also removes
the per-sweep ping-pong copy. No host fallback, serial device algorithm,
tolerance change, reduced sweep budget, or change to normalized compensated
Rayleigh extraction. Existing original-matrix/AV buffers remain necessary.
The tradeoff is two launches per round versus the failed fused arm's one,
but main already uses two launches per round. Large n should benefit most.

Required binding: **x_decomp**, plus existing FAST base. Manager compiles
arm A with no define and B with `-D MOJOLEARN_EIGH_TANGENT_CACHE` using
`MOJOLEARN_NUMERIC_MODE=fast`, `MOJOLEARN_COMPILE_JOBS=1`,
`MOJOLEARN_MOJO_BUILD_FLAGS` and `bindings/build_x_decomp.sh`.
No local compilation or tests were run; static diff checks only.

Quality must run before scoring. Use `tools/apple_fast_eigh_quality.py`
with the installed A and B arms: default sizes 31,128,257 and board,
indefinite,repeated,gram fixtures; B must pass against A **and the saved
repaired 131a0d78a B.json**, not merely the less accurate main result.
The checker measures actual eigenvalue error, AV-VΛ residual and VᵀV-I
orthogonality against float64 references. Reuse repaired artifacts only
with matching fixture sizes/kinds and checker source; otherwise obtain an
untimed repaired-arm quality artifact. Do not re-time repaired or opponents.

Example candidate check (manager sets the existing repaired JSON path):

```sh
MOJOLEARN_VENDOR=apple MOJOLEARN_NUMERIC_MODE=fast PYTHONPATH=python OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$HOME/board-0834/cache/venv/bin/python" tools/apple_fast_eigh_quality.py --output "$HOME/mq/out/gap26-eigh-cache-quality/B.json" --compare "$EIGH_REPAIRED_QUALITY"
```

After small quality passes, use the same checker on `--sizes 4096 --kinds
board` before accepting the large row, and compare errors to the repaired
metrics above as well as main. Preserve the existing stricter opponent
quality hold; improvement over main alone does not remove that hold.

Only after these gates, exact one-run-per-arm M3 timing proposal:

```sh
AFC_FAMILY=algos bash tools/afc_ab_def.sh gap26-eigh-cache-synthetic x_decomp eigh synthetic 1 1 "" "-D MOJOLEARN_EIGH_TANGENT_CACHE"
```

Manager should stage/validate precompiled A.so/B.so and set AFC_SKIP_BUILD=1
through verified_arms.py, as for earlier candidates. No queue was submitted.
Record measured outcome beside EIGH_TANGENT_CACHE and in EXPERIMENTS.md.

## Executable staging, quality receipt and conditional timing

Manager has compiled both arms at `14764dbb846b882443858cff3c05c3d95b1b9876`.
`tools/eigh_cache_verified.py` validates/stages that source using the manager's
verified_arms.py. Its default repaired-small reference is exactly the old
helper's `<tag>-quality/B.json` for tag `gap26-eigh-rayleigh`:
`~/mq/out/gap26-eigh-rayleigh-quality/B.json`. The helper cannot verify that
remote file locally; it refuses absent or mismatched fixture keys at runtime.

Exact first M3 CMD, on this branch:

```sh
MOJOLEARN_NUMERIC_MODE=fast "$HOME/board-0834/cache/venv/bin/python" tools/eigh_cache_verified.py quality gap26-eigh-cache
```

Small A/B quality must pass against both current main and the saved repaired
reference. Then it runs board4096 A/B quality and compares against both main
and repaired. Supply `--repaired-board /path/to/existing.json` to reuse an
existing matching reference; otherwise it verifies old source131a0d78a's
manifest and B.so hash in `~/mq/verified-arms/<full-old-sha>/x_decomp/`, checks
the original checker/linalg import surface, and generates a **quality-only**
repaired board4096 reference. It never reruns any old scored tag.

Exact separate conditional timing CMD:

```sh
MOJOLEARN_NUMERIC_MODE=fast "$HOME/board-0834/cache/venv/bin/python" tools/eigh_cache_verified.py timing gap26-eigh-cache gap26-eigh-cache-synthetic
```

The receipt is `~/mq/out/gap26-eigh-cache-quality/PASS.json`, issued only
when all 12 small cases and board4096 pass all three metrics against BOTH
references. Same existing checker thresholds: finite errors <=2e-4, sorted
eigenvalues, each candidate metric <=max(reference*1.1,reference+5e-8).
Receipt binds source, define, compiled manifest, checker/helper hashes, and
all saved result/reference JSON hashes. Timing verifies the receipt before
calling verified_arms.py; that validates compiled hashes/source scope and
refuses existing scored race.log, then runs one arm pair with AFC_SKIP_BUILD.
This admits a speed experiment, not removal of the stricter opponent-quality
hold or automatic promotion.

Quality logs and metrics live at
`~/mq/out/gap26-eigh-cache-quality/{A-small,B-small,repaired-board,A-board,B-board}.{log,json}`;
repaired-small.json is a preserved reference copy. Scored evidence lands in
`~/mq/out/race-gap26-eigh-cache-synthetic/race.log`.

## Resume after the captured main baseline failure

Main A-board4096 produced residual 5.374987821e-5, eigenvalue error 6.08669e-5
and orthogonality .00117131286. The last exceeds the original 2e-4 absolute
quality threshold. The old helper stopped before B-board; this is baseline
failure evidence, not a reason to skip candidate assessment.

The updated helper preserves that failure and still requires candidate B
to satisfy **all original absolute gates**, sorted eigenvalues and the
unchanged per-metric comparison to BOTH main and repaired references.
No tolerance was widened. Baseline failure is explicitly printed as
`EIGH-BASELINE-FAILURE` and remains in the hashed evidence.

Exact M3 serial-queue resume CMD:

```sh
MOJOLEARN_NUMERIC_MODE=fast "$HOME/board-0834/cache/venv/bin/python" tools/eigh_cache_verified.py resume gap26-eigh-cache
```

This specific resume requires existing A-small, B-small, repaired-board and
A-board captures, with matching log/JSON cases and metrics; it refuses if
B-board was already captured or attempted. It validates staged A/B binaries
against source14764dbb's manifest, the repaired manifest/reference and the
unchanged checker. It writes RESUME.json pinning inherited artifact hashes,
then executes **only missing B-board4096**. No prior GPU capture or scored
run is repeated. It issues PASS.json only after candidate absolute and both
reference gates pass; the separate timing command still requires that receipt.

The legacy failed run had no start receipt. Its provenance is explicitly
manager-authorized evidence, corroborated by staged source/binary hashes
and complete matching logs, rather than a claim that retrospective hashing
proves historical execution. New resume receipts preserve that distinction.
All GPU quality and scored work must run on the serial M3 queue; no heavy
maintenance or backup may overlap the measured window. Historical one-pair
speed verdicts with unaudited overlap remain HOLD-measurement.
