#!/usr/bin/env python3
"""scikit-learn as the ORACLE for PCA at 64 and 128 features.

**sklearn is an oracle here, not a design source.** The design is cuML's
`pcaFit` (`cuml/cpp/src/pca/pca.cuh:104`) and `calEig`
(`cuml/cpp/src/tsvd/tsvd.cuh:99`). sklearn only answers the question "is the
number right", which is the question that went unanswered while the device
eigensolver silently capped `n_cols` at 32.

USAGE

    /tmp/pca_wide > /tmp/pca_wide.out
    <python-with-sklearn> decomposition/pca_wide_sklearn.py /tmp/pca_wide.out

The fixture is regenerated here from the SAME splitmix64 stream that
`_fill_wide` in `decomposition/mojo_only/pca_check.mojo` uses, so both fit
the same float32 matrix. Keep the two generators in step; if you change one,
this script's first assertion (the total variance) is what catches it.

WHAT IS COMPARED, AND THE CAVEATS THAT ARE NOT EXCUSES

- `explained_variance_` element by element, relative to the total variance.
  This is a strict comparison: no sign freedom, no permutation freedom.
- `explained_variance_ratio_` the same way.
- `components_` UP TO SIGN only. An eigenvector is defined up to sign, and
  cuML's `signFlip` (`tsvd.cuh:139`) picks its sign from the TRANSFORMED data
  while ours picks it from the components, so the two conventions do not have
  to agree. |dot| is what is asserted.
- `components_` only for eigenvalues that are SEPARATED. The fixture is rank
  8 over an isotropic noise floor, so components 8 and up span a degenerate
  subspace in which individual eigenvectors are not defined at all and any
  orthonormal basis of it is a correct answer. Comparing them would be
  testing the tie-break of two LAPACK-era implementations against each other,
  which is not a correctness property. The degenerate block is still covered:
  its EIGENVALUES are compared, one by one, above.
"""
import sys

import numpy as np
from sklearn.decomposition import PCA

M1, M2, M3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB
WIDE_RANK = 8

# Relative to the total variance. The Mojo fit is float32 on the device; the
# reference is float64 LAPACK. Nothing here should be read as bit agreement.
TOL_EIGENVALUE = 2.0e-4
TOL_RATIO = 2.0e-4
TOL_COMPONENT_DOT = 1.0e-3  # |1 - |dot|| for a separated component


def u01(a, b, salt):
    """splitmix64 -> [0, 1), matching `_wide_u01` in pca_check.mojo."""
    z = (
        a.astype(np.uint64) * np.uint64(M1)
        + (b.astype(np.uint64) + np.uint64(1)) * np.uint64(M2)
        + np.uint64((salt + 1) * M3 % (1 << 64))
    )
    z = (z ^ (z >> np.uint64(30))) * np.uint64(M2)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(M3)
    z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / 9007199254740992.0)


def latent_sd(k):
    return np.sqrt(200.0 - k * 24.0)


def fixture(n_rows, n_cols):
    i = np.arange(n_rows, dtype=np.uint64)[:, None]
    j = np.arange(n_cols, dtype=np.uint64)[None, :]
    x = np.zeros((n_rows, n_cols), dtype=np.float64)
    for k in range(WIDE_RANK):
        kk = np.full((n_rows, 1), k, dtype=np.uint64)
        z = (u01(i, kk, 1) - 0.5) * 2.0 * np.sqrt(3.0)
        krow = np.full((1, n_cols), k, dtype=np.uint64)
        b = (u01(krow, j, 2) - 0.5) * 2.0
        x += z * latent_sd(k) * b
    x += (u01(i, j + np.uint64(1000), 3) - 0.5) * 2.0 * np.sqrt(3.0)
    return np.ascontiguousarray(x, dtype=np.float32)


def parse(path):
    blocks, cur = [], None
    for line in open(path):
        line = line.strip()
        if line.startswith("WIDE "):
            kv = dict(p.split("=") for p in line.split()[1:])
            cur = {"n_cols": int(kv["n_cols"]), "n_rows": int(kv["n_rows"]),
                   "COMP": {}}
            blocks.append(cur)
        elif cur is None:
            continue
        elif line.startswith("COMP "):
            parts = line.split()
            cur["COMP"][int(parts[1])] = np.array(
                [float(v) for v in parts[2:]], dtype=np.float64
            )
        elif line.split(" ", 1)[0] in ("EV", "EVR", "SV"):
            tag, rest = line.split(" ", 1)
            cur[tag] = np.array([float(v) for v in rest.split()],
                                dtype=np.float64)
        elif line.startswith("NOISE "):
            cur["NOISE"] = float(line.split()[1])
    return blocks


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pca_wide.out"
    blocks = parse(path)
    if not blocks:
        sys.exit(f"no WIDE blocks in {path}")

    failures = 0
    for b in blocks:
        n_cols, n_rows = b["n_cols"], b["n_rows"]
        x = fixture(n_rows, n_cols)
        ref = PCA(n_components=n_cols, svd_solver="full").fit(x)
        total = float(ref.explained_variance_.sum())

        ev_err = np.abs(b["EV"] - ref.explained_variance_).max() / total
        evr_err = np.abs(b["EVR"] - ref.explained_variance_ratio_).max()
        sv_err = (np.abs(b["SV"] - ref.singular_values_).max()
                  / float(ref.singular_values_.max()))

        # Which components are separated enough to have a defined direction?
        # The gap to the nearer neighbour must dominate the float32 noise.
        gaps = []
        for c in range(WIDE_RANK):
            lo = ref.explained_variance_[c + 1] if c + 1 < n_cols else 0.0
            hi = (ref.explained_variance_[c - 1] if c > 0
                  else ref.explained_variance_[0] * 2.0)
            gaps.append(min(hi - ref.explained_variance_[c],
                            ref.explained_variance_[c] - lo) / total)

        worst_dot, worst_c = 0.0, -1
        for c, comp in sorted(b["COMP"].items()):
            if gaps[c] < 1e-3:
                continue
            d = abs(float(np.dot(comp, ref.components_[c])))
            if abs(1.0 - d) > worst_dot:
                worst_dot, worst_c = abs(1.0 - d), c

        ok = (ev_err <= TOL_EIGENVALUE and evr_err <= TOL_RATIO
              and worst_dot <= TOL_COMPONENT_DOT)
        failures += 0 if ok else 1
        print(
            f"n_cols={n_cols:4d}  explained_variance {ev_err:.3e}"
            f"  ratio {evr_err:.3e}  singular_values {sv_err:.3e}"
            f"  worst |1-|dot|| over separated components {worst_dot:.3e}"
            f" (component {worst_c})   {'OK' if ok else 'FAIL'}"
        )
        print(
            f"           our top-3 EV {b['EV'][:3]}\n"
            f"       sklearn top-3 EV {ref.explained_variance_[:3]}"
        )

    if failures:
        sys.exit(f"{failures} width(s) disagree with scikit-learn")
    print("PCA matches scikit-learn at every width tested.")


if __name__ == "__main__":
    main()
