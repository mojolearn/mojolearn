#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Lane idn-all (2026-10-04): the box checks the nine IDENTICAL speed lanes owe
that a board race cannot express. Run in a built branch tree, IDENTICAL mode
(an `lq add <box> CMD` line; never on the laptop):

    python tools/idn_all_checks.py lu-nan        # LU on an input that overflows to NaN: device vs host column
    python tools/idn_all_checks.py sgd-nan       # an SGD fit with NaN / infinity in X or in y must raise
    python tools/idn_all_checks.py import-smoke  # import the package and every module the lanes changed
    python tools/idn_all_checks.py pca-id        # PCA / TruncatedSVD fit (round-robin eigh): device vs host column
    python tools/idn_all_checks.py all           # the four above, one verdict line each, then IDN_ALL_CHECKS

Each prints one verdict line last (`LU_NAN ...`, `SGD_NAN ...`, `IMPORT_OK`,
`PCA_ID ...`) and exits non-zero on a failure. The digests in the LU_NAN and
PCA_ID lines are comparable across boxes (same fixture, same bytes hashed).
No timing, no opponent."""
import hashlib
import json
from pathlib import Path
import os
import subprocess
import sys

MODULES = (
    "mojolearn", "mojolearn.linear_model", "mojolearn.decomposition", "mojolearn.embedding", "mojolearn.linalg",
    "mojolearn._expansion_cnn", "mojolearn._expansion_decomp", "mojolearn._expansion_linear",
    "mojolearn._expansion_prep", "mojolearn._expansion_trees", "mojolearn._surface_decomp",
    "mojolearn._surface_prep",
)


def _bytes(o):
    if hasattr(o, "tobytes"):
        return o.tobytes()
    import numpy as np
    return np.asarray(o).tobytes()


def _lu_fixture(n):
    """Finite Wilkinson growth matrix with two growing trailing columns.

    Partial pivoting leaves the tied lower entries in place. Repeated growth
    overflows the trailing columns during elimination, then inf/inf produces
    NaN. The old 48 * 1e37 fixture already overflowed before entering LU.
    """
    import numpy as np
    i = np.arange(n, dtype=np.int64)
    a = np.where(i[:, None] > i[None, :], -1, 0).astype(np.float32)
    a[i, i] = 1
    a[:, -1] = 1
    a[:-1, -2] = 1
    a *= np.float32(1e30)
    assert np.isfinite(a).all(), 'LU overflow must originate inside the solver'
    b = (((i[:, None] * 17 + np.arange(3)[None, :] * 5) % 13) - 6).astype(np.float32)
    return np.ascontiguousarray(a), np.ascontiguousarray(b)


def _lu_child():
    import warnings
    import numpy as np
    import mojolearn as ml
    from mojolearn import _expansion_decomp as la
    warnings.simplefilter("ignore")
    flags = int(la._Kit(la._mode(None))._raw().x_decomp_idn_flags())
    resident = bool(flags & 2)
    expected_resident = os.environ.get("MOJOLEARN_IDN_ALL_OFF") != "1"
    assert resident == expected_resident, 'compiled LU route does not match comparison arm'
    contract = "resident-raw-solve" if resident else "public-solve-refusal"
    artifacts = os.environ.get("MOJOLEARN_IDN_GATE_ARTIFACTS")
    destination = Path(artifacts) / ("lu-" + ml.vendor()) if artifacts else None
    if destination:
        destination.mkdir(parents=True, exist_ok=False)
    h = hashlib.sha256()
    nan_words = 0
    rows = []
    for n in (64, 257):
        a, b = _lu_fixture(n)
        lu, piv = la.lu_factor(a)
        # Both arms preserve the public finite-factor boundary. The OFF
        # solve route composes public lu_factor -> lu_solve and consequently
        # refuses overflowed factors. The ON resident route returns raw solve
        # words. This difference is explicit; neither arm normalizes NaNs.
        try:
            la.lu_solve((lu, piv), b)
        except ValueError as exc:
            refusal = str(exc)
            if 'finite' not in refusal:
                raise
        else:
            raise AssertionError('lu_solve accepted nonfinite input factors')
        values = {'lu': lu, 'piv': piv}
        if resident:
            values['solve'] = la.solve(a, b)
        else:
            try:
                la.solve(a, b)
            except ValueError as exc:
                assert str(exc) == refusal, 'OFF solve refused for a different reason'
            else:
                raise AssertionError('OFF solve accepted nonfinite intermediate factors')
        current_nan_words = int(np.isnan(np.frombuffer(_bytes(lu), dtype="<f4")).sum())
        assert current_nan_words > 0, 'vacuous LU NaN-word fixture'
        words = {}
        for name, part in values.items():
            raw = _bytes(part)  # Unmodified factor, pivot and ON solve bytes.
            h.update(raw)
            words[name] = {'sha256': hashlib.sha256(raw).hexdigest(),
                           'shape': list(np.asarray(part).shape), 'dtype': str(np.asarray(part).dtype)}
        h.update(contract.encode() + b'\0' + refusal.encode() + b'\0')
        nan_words += current_nan_words
        row = {'n': n, 'contract': contract, 'lu_solve_refusal': refusal,
               'nan_words': current_nan_words, 'outputs': words}
        if destination:
            path = destination / (str(n) + '.npz')
            np.savez(path, **values)
            row['artifact'] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        rows.append(row)
    if destination:
        (destination / 'result.json').write_text(json.dumps({'status': 'PASS', 'vendor': ml.vendor(),
            'compiled_flags': flags, 'contract': contract, 'digest': h.hexdigest(), 'fixtures': rows}, indent=2) + '\n')
    print("DIGEST %s nan_words=%d contract=%s" % (h.hexdigest(), nan_words, contract))


def _pca_child():
    import numpy as np
    from mojolearn.decomposition import PCA, TruncatedSVD
    h = hashlib.sha256()
    parts = 0
    for n, d, k in ((4096, 220, 16), (1500, 33, 33), (900, 7, 3)):
        i = np.arange(n * d, dtype=np.int64).reshape(n, d)
        X = ((((i * 2654435761) % 1000003) - 500001).astype(np.float32) / np.float32(977.0)
             * (1.0 + (np.arange(d) % 11)).astype(np.float32))
        X = np.ascontiguousarray(X, dtype=np.float32)
        for est in (PCA(n_components=k), TruncatedSVD(n_components=min(k, d - 1))):
            est.fit(X)
            for name in ("components_", "explained_variance_", "explained_variance_ratio_", "singular_values_",
                         "mean_"):
                v = getattr(est, name, None)
                if v is not None:
                    h.update(_bytes(v))
                    parts += 1
            h.update(_bytes(est.transform(X[:64])))
    print("DIGEST %s parts=%d" % (h.hexdigest(), parts))


def _run_child(extra_env, what="lu-nan"):
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical", **extra_env)
    p = subprocess.run([sys.executable, os.path.abspath(__file__), what, "--child"], env=env,
                       capture_output=True, text=True)
    artifacts = os.environ.get("MOJOLEARN_IDN_GATE_ARTIFACTS")
    if artifacts:
        folder = Path(artifacts)
        folder.mkdir(parents=True, exist_ok=True)
        tag = what + "-" + env.get("MOJOLEARN_VENDOR", "auto")
        (folder / (tag + ".stdout.log")).write_text(p.stdout)
        (folder / (tag + ".stderr.log")).write_text(p.stderr)
        (folder / (tag + ".rc.json")).write_text(json.dumps({'rc': p.returncode, 'vendor': env.get('MOJOLEARN_VENDOR')}) + "\n")
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("DIGEST ")]
    if p.returncode != 0 or not line:
        return None, (p.stderr or p.stdout).strip().splitlines()[-1:] or ["no output"]
    return line[-1].split()[1:], None


def lu_nan():
    dev, e1 = _run_child({})
    host, e2 = _run_child({"MOJOLEARN_VENDOR": "cpu"})
    if dev is None or host is None:
        print("LU_NAN ERROR device=%s host=%s" % (e1, e2))
        return 1
    vacuous = dev[1] == "nan_words=0" or host[1] == "nan_words=0"
    ok = dev == host and not vacuous
    print("LU_NAN device=%s host=%s %s %s" % (dev[0], host[0], " ".join(dev[1:]),
                                              "MATCH" if ok else ("VACUOUS" if vacuous else "DIFFER")))
    return 0 if ok else 1


def pca_id():
    dev, e1 = _run_child({}, "pca-id")
    host, e2 = _run_child({"MOJOLEARN_VENDOR": "cpu"}, "pca-id")
    if dev is None or host is None:
        print("PCA_ID ERROR device=%s host=%s" % (e1, e2))
        return 1
    ok = dev[0] == host[0]
    print("PCA_ID device=%s host=%s %s %s" % (dev[0], host[0], " ".join(dev[1:]), "MATCH" if ok else "DIFFER"))
    return 0 if ok else 1


def sgd_nan():
    import numpy as np
    from mojolearn import SGDClassifier, SGDRegressor
    n, d = 4096, 8
    i = np.arange(n * d, dtype=np.int64).reshape(n, d)
    X = (((i * 37) % 101) - 50).astype(np.float32) / np.float32(50)
    yr = (X[:, 0] - X[:, 3]).astype(np.float32)
    yc = (yr > 0).astype(np.int32)
    bad = 0
    cases = []
    for name, val in (("nan", np.nan), ("inf", np.inf)):
        Xb = X.copy()
        Xb[n - 3, d - 1] = val
        yb = yr.copy()
        yb[n // 2] = val
        cases += [("reg X " + name, SGDRegressor, Xb, yr), ("reg y " + name, SGDRegressor, X, yb),
                  ("clf X " + name, SGDClassifier, Xb, yc)]
    for what, cls, Xa, ya in cases:
        for kw in ({}, {"shuffle": False}):
            try:
                cls(max_iter=3, random_state=7, **kw).fit(Xa, ya)
            except Exception as e:  # the binding's Error or the Python refusal: either is the refusal
                msg = str(e)
                if "NaN" in msg or "infinity" in msg or "finite" in msg:
                    continue
                print("SGD_NAN wrong error for %s %s: %s" % (what, kw, msg[:200]))
                bad += 1
                continue
            print("SGD_NAN no error for %s %s" % (what, kw))
            bad += 1
    # the clean fit must still pass (the check is not a blanket refusal)
    SGDRegressor(max_iter=3, random_state=7).fit(X, yr)
    SGDClassifier(max_iter=3, random_state=7).fit(X, yc)
    print("SGD_NAN %s cases=%d bad=%d" % ("ok" if bad == 0 else "FAIL", 2 * len(cases), bad))
    return 0 if bad == 0 else 1


def import_smoke():
    import importlib
    for m in MODULES:
        importlib.import_module(m)
    import mojolearn
    print("IMPORT_OK %d modules mojolearn %s" % (len(MODULES), getattr(mojolearn, "__version__", "?")))
    return 0


def main(argv):
    if len(argv) >= 2 and argv[1] == "--child" and argv[0] in ("lu-nan", "pca-id"):
        (_lu_child if argv[0] == "lu-nan" else _pca_child)()
        return 0
    table = {"import-smoke": import_smoke, "lu-nan": lu_nan, "sgd-nan": sgd_nan, "pca-id": pca_id}
    if len(argv) == 1 and argv[0] == "all":
        got = {}
        for name, fn in table.items():   # glue: the four checks, each its own verdict line
            try:
                got[name] = fn()
            except Exception as e:
                print("%s raised: %s" % (name, str(e)[:300]))
                got[name] = 1
        print("IDN_ALL_CHECKS " + " ".join("%s=%s" % (k, "ok" if v == 0 else "FAIL") for k, v in got.items()))
        return 0 if not any(got.values()) else 1
    if len(argv) != 1 or argv[0] not in table:
        print(__doc__)
        return 2
    return table[argv[0]]()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
