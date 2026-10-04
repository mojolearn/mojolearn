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
