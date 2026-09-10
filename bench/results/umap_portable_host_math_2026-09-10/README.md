# IDENTICAL UMAP portable host math evidence

Production change `163aaf81` routes six UMAP host modules to portable
binary64 exp/log/log2/pow seams only in IDENTICAL. Gate correction `20061ad5`
separates the explicitly declared x**0/1**p signaling-NaN policy from finite
libm accuracy admission. Production arithmetic did not change in that fix.

Root executed all local builds under the shared lock and the matching Linux
host/H100 runs. Source hashes for all ten relevant production/gate/script
files match the remote checkout; `manifest.json` records these, artifact
SHA256s and host/compiler/GPU metadata. The NVIDIA device is H10080GB,
driver580.126.09, Mojo1.0.0(ed45d567). No AMD execution occurred in this
continuation, and no performance/opponent measurement was taken.

## Primitive

Both hosts produce the same 262,378-pair input hash
**11634649235927139599** and portable-output hash **13353115118729037508**.
Both report worst1037ULP and worst normal relative error
**1.2169443322931572e-13** against their host libm. The declared admission is
2e-12 relative with an absolute allowance of two smallest binary64 subnormals;
these measured results do not imply a tight-ULP or correctly rounded pow.
Subnormal input/output and signed/special-value behavior are part of the gate.

The initial Linux gate failed one policy case: signaling NaN**0 returns1 in
our explicit contract while glibc returns a quiet NaN. That original failure
is retained in `linux-h100/primitive-initial-policy-failure.log`. The corrected
gate checks our policy exactly and records this as one expected libm policy
difference; Apple reports zero. Finite accuracy admission was not relaxed.
The Apple final corrected gate is `apple/primitive-final.log`.

## UMAP stages

| Capture | Exact cells | Apple ↔ Linux/H100 differences |
|---|---:|---:|
| Existing small identity fixture | 186 | 0 |
| Existing broader identity fixture | 690 | 0 |
| New host-only curve/graph/layout/transform stages | 116 across17stages | 0 |
| Public transform fixture | 21 | 0 |

The small and broader Apple fixtures also match their pre-change captures
exactly, with zero changed cells in either fixture. This is measured fixture
evidence, not a promise that all previously platform-dependent outputs stay
unchanged on every input.

The new host gate retains the existing independent default/custom curve and
serial optimizer numerical references, verifies dense/CSR graph and layout
identity, repeated transform output, frozen training coordinates and refusal
cases. It executes no FAST optimizer. The existing full pipeline captures
also exercise the default IDENTICAL device optimizer, whose arithmetic was
not changed by this host-math work.

Reproduce gates using `tools/check_umap_portable_host_math.sh OUT` in an
activated toolchain. Reproduce full-word comparisons using:

```sh
python3 tools/compare_umap_portable_host_math.py \
  bench/results/umap_portable_host_math_2026-09-10
```

`comparison.json` records exact cell counts and hashes; it validates required
stage/index sets and completion markers and rejects nonfinite or duplicate
records. It uses no floating-point tolerance for cross-host comparisons.

This closes the named UMAP host seams for the measured captures. It does not
close all primitive gaps or certify a physical AMD run. The remaining actual
Float32 log2 tree call is inventoried in
`docs/lanes/PORTABLE_PRIMITIVE_AUDIT_2026-09-10.md` and was not modified/tested.
The new primitive does not modify existing shared portable functions or tree
behavior. PCA evidence from the parent archive is deliberately excluded.
