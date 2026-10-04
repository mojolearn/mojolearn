# Scoped adapter r2: pinned quality readiness

Compiled source C is `28f06e1923cfa164fd068c31b0f986ff0126d306`
(`lane/apple-fast-scoped-gemm-r2`). This source exists locally and its owner
recorded it pushed. H is the final commit of `lane/apple-fast-scoped-gemm-readiness`.
No cloud state, binaries, numerical quality or timing were checked here.

The harness may run C binaries under H only after C is proven an ancestor of H,
all changed paths match the explicit harness-only allowlist, and the tracked
checkout is clean. Native sources, production Python, build configuration and
unreviewed tools may not drift. No numerical tolerance, split policy, fixture or
selector was changed. The probe exercises real decomp/PCA entrances, not full
estimators; a quality PASS alone does not admit a default or board update.

## Build on M2

Use C in a full isolated source checkout. Only binding `scoped_gemm_probe` is
needed. `bindings/build_scoped_gemm_probe.sh` is compile-only and takes candidate
flags through **MOJOLEARN_MOJO_BUILD_FLAGS**, not BUILD_EXTRA_DEFINES.
The all-profile exact flags are:

```
A: -D MOJOLEARN_SCOPED_GEMM_AUDIT
B: -D MOJOLEARN_SCOPED_GEMM_AUDIT -D MOJOLEARN_SCOPED_GEMM_G1_TALL -D MOJOLEARN_SCOPED_GEMM_G1_DENSE -D MOJOLEARN_SCOPED_GEMM_G1_GRAM -D MOJOLEARN_SCOPED_GEMM_G2_NARROW -D MOJOLEARN_SCOPED_GEMM_SPLIT -D MOJOLEARN_SCOPED_GEMM_PCA
```

For each arm, run through the compile semaphore, using its exact quoted flags:

```
bash ~/mojolearn-evidence/compile_slot.sh env MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_MOJO_BUILD_FLAGS='ARM_FLAGS' bash bindings/build_scoped_gemm_probe.sh
```

`ARM_FLAGS` above is a placeholder for one complete line's flags, not a variable
or a literal compiler input. Clear unrelated target-column/build environment
settings; preserve compiler version and environment in the build record. The
script compiles with `-j 1` and stages
`python/mojolearn/_mojolearn_scoped_gemm_probe.so`. Copy A before B overwrites it.

Stage `~/mq/verified-arms/C/scoped_gemm_probe/{A.so,B.so,manifest.json}` with
source_sha C, binding `scoped_gemm_probe`, numeric_mode `fast`, exact ordered
`defines_A`/`defines_B` above, and SHA256 `hashes.A`/`hashes.B`.

## Fresh source and ABI

The helper loads each binary directly by its absolute verified artifact path
with importlib in a separate Python process. It does not import mojolearn,
install a package binary, assume an original .so exists, or need an IDENTICAL
ibase. NumPy and platform runtime dependencies must already exist in the pinned
runner's board-0834 venv; no installation/build fallback is allowed.

Both artifact hashes are checked before capture, each subprocess checks its own
hash before native load, and each hash is checked again after capture. ABI must
be 1, FAST numeric mode 0, enabled 1; flags must equal 0 for A and the exact
profile mask for B. Exact route counters and split/stride metadata remain gated.
Native module loading remains the runtime proof of linked dependency readiness;
a metadata READY receipt does not establish that proof.

## Metadata preflight and serial submission

From clean H, generate the full 28-case metadata spec locally:

```
python3 tools/scoped_gemm_preflight_spec.py 28f06e1923cfa164fd068c31b0f986ff0126d306 scoped-r2-all-q-v1 all > /OWNED_EVIDENCE/scoped-r2-all-q-v1.json
```

The generator inserts H from HEAD, validates tools-only compatibility, and
includes exact pair source/binding/mode/defines plus all native source paths.
The tag above is proposed, not reserved; check for duplicates on M3.

Manager stages H and C in the mirror, complete artifacts, spec and reviewed
policy in a serial transfer window. Deploy policy additively: preserve newer
main entries (this branch predates some unrelated policy registrations). Use an
already prepared runner branch with the policy, then run read-only metadata
preflight and retain its READY receipt. The eventual command is:

```
python tools/apple_fast_pinned_job.py H tools/scoped_gemm_quality.py 28f06e1923cfa164fd068c31b0f986ff0126d306 scoped-r2-all-q-v1 all
```

`H` and `/OWNED_EVIDENCE` are placeholders. Do not submit either literally.
The helper revalidates provenance/artifacts during execution; no numerical
work happens during spec generation or metadata preflight.

## Quality receipt

`~/mq/out/TAG-quality/report.json` retains source_sha C, harness_source H,
helper SHA256, exact manifest, all four A/B packet and metadata hashes, profile,
contract `scoped-actual-decomp-pca-quality-v1`, route counts, per-case independent
FP64 errors, failures and PASS/HOLD. Directory creation is exclusive; failures
remain preserved and require fresh tags for authorized infrastructure repair.

The existing relative Frobenius bound 5e-6 and zero allowance for relative or
maximum-absolute error regression are unchanged. PCA mean/input restoration
checks and one-split atomic controls are unchanged. Atomic nondeterminism can
cause HOLD; preserve it without loosening thresholds or replaying scored work.

Any later timing helper must verify the report hash, C/H, manifest/binary
hashes, profile, contract, gates, packet hashes, and positive reach for its
operation. This change creates no timing helper or admission. Actual estimator
quality and useful speed must still be established separately.

Local validation: five metadata/source-policy tests and seven existing preflight
tests passed; Python/shell syntax and git diff checks passed. No numerical or
native execution, compiler, queue mutation, cloud access or main mutation.
