#!/usr/bin/env python3
"""AgglomerativeClustering on the device vs the host column (lane hr2-mds-agglo).

Runs every linkage (ward, complete, average, single through the x_cluster
route via compute_distances=True, average on a precomputed matrix) and the
legacy single-linkage route, once on the GPU binding and once under
MOJOLEARN_VENDOR=cpu, each in its own process, and prints one line per case:

    AGGLOCHECK <case> n=<n> gpu=<digest> host=<digest> MATCH|DIFFER ari_sklearn=<ari> gpu_ms=<t> v0_ms=<t>

The digest is sha256 of children_ (and distances_ where the case keeps them).
gpu_ms is the device fit; v0_ms the same fit with MOJOLEARN_XC_AGGLO_LOOP_V0=1
(main's host merge loop on the GPU binding), x_cluster cases only. A final
line `AGGLOCHECK summary match=<k>/<n>` closes the run.

Usage: python3 tools/agglo_device_check.py [n_rows]
"""
import hashlib
import json
import os
import subprocess
import sys

CASES = [
    ("ward", dict(linkage="ward")),
    ("complete", dict(linkage="complete")),
    ("average", dict(linkage="average")),
    ("single-x", dict(linkage="single", compute_distances=True)),
    ("average-pre", dict(linkage="average", metric="precomputed")),
    ("single-legacy", dict(linkage="single")),
]


def data(n, seed=7):
    import numpy as np
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 6, size=(8, 6))
    lab = rng.integers(0, 8, size=n)
    x = centers[lab] + rng.normal(0, 1, size=(n, 6))
    return x.astype("float32"), lab


def child(case, n, v0):
    import time
    import numpy as np
    from mojolearn import AgglomerativeClustering
    x, _ = data(n)
    kw = dict(CASES)[case]
    X = x
    if kw.get("metric") == "precomputed":
        d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1)).astype("float32")
        X = d
    m = AgglomerativeClustering(n_clusters=8, **kw)
    t0 = time.perf_counter()
    m.fit(X)
    ms = (time.perf_counter() - t0) * 1e3
    m2 = AgglomerativeClustering(n_clusters=8, **kw)
    t0 = time.perf_counter()
    m2.fit(X)
    ms = min(ms, (time.perf_counter() - t0) * 1e3)
    h = hashlib.sha256(np.asarray(m.children_, dtype="<i4").tobytes())
    if hasattr(m, "distances_"):
        h.update(np.asarray(m.distances_, dtype="<f4").tobytes())
    print(json.dumps(dict(d=h.hexdigest()[:16], ms=ms, labels=np.asarray(m.labels_).tolist())))


def run(case, n, env):
    e = dict(os.environ)
    e.update(env)
    r = subprocess.run([sys.executable, __file__, "--child", case, str(n)], env=e,
                       capture_output=True, text=True, timeout=3000)
    if r.returncode != 0:
        return dict(d="ERR:" + (r.stderr.strip().splitlines() or ["?"])[-1][:120], ms=-1, labels=None)
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        child(sys.argv[2], int(sys.argv[3]), False)
        return
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    from sklearn.cluster import AgglomerativeClustering as SK
    from sklearn.metrics import adjusted_rand_score
    x, _ = data(n)
    ok = 0
    for case, kw in CASES:
        g = run(case, n, {})
        h = run(case, n, {"MOJOLEARN_VENDOR": "cpu"})
        v0 = run(case, n, {"MOJOLEARN_XC_AGGLO_LOOP_V0": "1"}) if case != "single-legacy" else dict(ms=-1)
        skw = {k: v for k, v in kw.items() if k != "compute_distances"}
        Xs = x
        if skw.get("metric") == "precomputed":
            import numpy as np
            Xs = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).sum(-1))
        sk = SK(n_clusters=8, **skw).fit(Xs)
        ari = adjusted_rand_score(sk.labels_, g["labels"]) if g.get("labels") else float("nan")
        same = g["d"] == h["d"] and not g["d"].startswith("ERR")
        ok += same
        print("AGGLOCHECK %s n=%d gpu=%s host=%s %s ari_sklearn=%.4f gpu_ms=%.1f v0_ms=%.1f" % (
            case, n, g["d"], h["d"], "MATCH" if same else "DIFFER", ari, g["ms"], v0["ms"]), flush=True)
    print("AGGLOCHECK summary match=%d/%d" % (ok, len(CASES)))


if __name__ == "__main__":
    main()
