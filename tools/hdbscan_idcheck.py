#!/usr/bin/env python3
"""Same-bits check of HDBSCAN's device tree stages (cgr2-hdbscan,
2026-10-03): the condensed tree, the stabilities, the EOM / leaf / epsilon
selection, the labels, the probabilities and the prediction data, now all
built on the device, against the host column's walk.

Each case fits a fixed seeded fixture twice, in separate subprocesses: once
with MOJOLEARN_VENDOR unset (the device column) and once with
MOJOLEARN_VENDOR=cpu (the host column), digests the fitted arrays' bytes,
and prints

    IDCHECK <case> device=<digest> host=<digest> MATCH|DIFFER

The device digest is comparable across boxes (NVIDIA = AMD = Apple = host).
Usage: python tools/hdbscan_idcheck.py [case ...]
"""
import hashlib
import json
import os
import subprocess
import sys

CASES = ["eom-pd", "leaf-pd", "eps", "single", "maxcs"]


def _fixture(n=8000, d=4, k=12, seed=11):
    import numpy as np
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-10.0, 10.0, (k, d)).astype(np.float32)
    lab = rng.integers(0, k, n)
    scale = rng.uniform(0.3, 1.5, k).astype(np.float32)
    # Elementwise only (no BLAS), so the fixture is the same bytes everywhere.
    X = centers[lab] + scale[lab][:, None] * rng.standard_normal((n, d)).astype(np.float32)
    noise = rng.random(n) < 0.05
    X[noise] = rng.uniform(-14.0, 14.0, (int(noise.sum()), d)).astype(np.float32)
    return np.ascontiguousarray(X.astype(np.float32))


def _digest(est):
    import numpy as np
    h = hashlib.sha256()
    for nm in ("labels_", "probabilities_", "core_distances_"):
        h.update(np.ascontiguousarray(getattr(est, nm)).tobytes())
    h.update(str((est.n_clusters_, est.n_outliers_, est.n_condensed_clusters_)).encode())
    pd = getattr(est, "_prediction_data", None)
    if pd:
        for k in sorted(pd):
            v = pd[k]
            if hasattr(v, "tobytes"):
                h.update(k.encode())
                h.update(np.ascontiguousarray(v).tobytes())
            else:
                h.update((k + "=" + str(v)).encode())
    return h.hexdigest()[:16]


def _run_case(case):
    import mojolearn as ml
    X = _fixture()
    if case == "eom-pd":
        e = ml.HDBSCAN(min_cluster_size=15, prediction_data=True).fit(X)
    elif case == "leaf-pd":
        e = ml.HDBSCAN(min_cluster_size=15, cluster_selection_method="leaf",
                       prediction_data=True).fit(X)
    elif case == "eps":
        e = ml.HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.6).fit(X)
    elif case == "single":
        e = ml.HDBSCAN(min_cluster_size=400, allow_single_cluster=True).fit(X)
    elif case == "maxcs":
        e = ml.HDBSCAN(min_cluster_size=10, max_cluster_size=500).fit(X)
    else:
        raise ValueError(case)
    return _digest(e)


def main():
    if len(sys.argv) > 2 and sys.argv[1] == "--child":
        try:
            print(json.dumps({"digest": _run_case(sys.argv[2])}))
        except Exception as ex:  # reported as the digest, so a refusal shows
            print(json.dumps({"digest": "ERROR:" + type(ex).__name__ + ":" + str(ex)[:120]}))
        return 0
    cases = sys.argv[1:] or CASES
    bad = 0
    for case in cases:
        out = {}
        for col in ("device", "host"):
            env = dict(os.environ)
            env.pop("MOJOLEARN_VENDOR", None)
            if col == "host":
                env["MOJOLEARN_VENDOR"] = "cpu"
            p = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", case],
                               env=env, capture_output=True, text=True, timeout=1800)
            dig = "ERROR:rc=%d" % p.returncode
            for line in p.stdout.splitlines()[::-1]:
                if line.startswith("{"):
                    dig = json.loads(line)["digest"]
                    break
            if dig.startswith("ERROR"):
                sys.stderr.write(p.stderr[-2000:])
            out[col] = dig
        ok = out["device"] == out["host"] and not out["device"].startswith("ERROR")
        bad += 0 if ok else 1
        print(f"IDCHECK {case} device={out['device']} host={out['host']} {'MATCH' if ok else 'DIFFER'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
