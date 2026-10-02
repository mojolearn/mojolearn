#!/usr/bin/env python3
"""Old path vs new path for the classical pass, one process per arm.

    python tools/classical_pass_ab.py case <name> <size> --out f.json   # one arm (env set by caller)
    python tools/classical_pass_ab.py all --out dir [--ivf-off-so path]  # every case, both arms

Each case builds its seeded input, times ours, and records a SHA-256 of every
output. "digest" sizes run old AND new (the old paths are the slow serial
programs, so these are reduced shapes); "full" sizes are the board's shape and
run the new path only. A new-path digest that differs from the old path's is a
bug in that fix. The digests also compare across vendors.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

OLD_ENV = {
    "lu": {"MOJOLEARN_XD_LU_PIVOT_SERIAL": "1"},
    "sgd-reg": {"MOJOLEARN_X_LINEAR_SGD_HOST": "0"},
    "sgd-clf": {"MOJOLEARN_X_LINEAR_SGD_HOST": "0"},
    "lars": {"MOJOLEARN_X_LINEAR_LARS_GRID_GRAM": "0"},
    "ivf": {},  # compile time: the old arm swaps in the -D MOJOLEARN_IVF_IDENTICAL_SCAN_OFF binding
}
# (digest size, full size)
SIZES = {"lu": (1024, 8192), "sgd-reg": (20_000, 1_000_000), "sgd-clf": (20_000, 1_000_000),
         "lars": (200_000, 1_000_000), "ivf": (40_000, 400_000)}
D = 220  # Istella's width


def sha(*arrays):
    import numpy as np
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def host_independent_xw(X, w):
    """X @ w as the SAME bits on every host: float64 elementwise IEEE
    operations over the columns, j ascending, no BLAS. `X @ w` (sgemv)
    gives bits that depend on the OpenBLAS kernel and on its THREAD COUNT
    (the row split moves the kernel's blocking): on the 20-vCPU MI325X box,
    OPENBLAS_NUM_THREADS 3, 6, 7, 12, 16, 32 and 48 each gave a different y
    at 20000 rows, and a 64-CPU view reproduced the L4 pod's SGD-reg and
    LARS digests exactly (bench/results/host-cpu-identity-20260930). The
    library was never host-dependent; the harness's y was."""
    import numpy as np
    acc = np.zeros(X.shape[0], np.float64)
    for j in range(X.shape[1]):
        acc = acc + X[:, j].astype(np.float64) * float(w[j])
    return acc


def run_case(name, n):
    import numpy as np
    import mojolearn as ml
    rng = np.random.default_rng(7)
    if name == "lu":
        A = (rng.standard_normal((n, n)) + 2 * np.sqrt(n) * np.eye(n)).astype(np.float32)
        B = rng.standard_normal((n, 64)).astype(np.float32)
        t = time.perf_counter()
        from mojolearn._expansion_decomp import lu_factor, lu_solve
        lu, piv = lu_factor(A, numeric_mode="identical")
        x = lu_solve((lu, piv), B, numeric_mode="identical")
        x = np.asarray(x)
        ms = (time.perf_counter() - t) * 1000
        res = float(np.max(np.abs(A.astype(np.float64) @ x - B)) / np.max(np.abs(B)))
        return {"ms": ms, "digest": sha(lu, piv, x), "quality": {"rel_residual": res}}
    X = rng.standard_normal((n, D)).astype(np.float32)
    w = rng.standard_normal(D).astype(np.float32)
    y = (host_independent_xw(X, w) + 0.1 * rng.standard_normal(n)).astype(np.float32)
    inputs = sha(X, y)
    if name in ("sgd-reg", "sgd-clf"):
        kw = dict(penalty="l2", alpha=1e-4, max_iter=100, tol=None, shuffle=True, random_state=7,
                  learning_rate="constant", eta0=0.005, numeric_mode="identical")
        if name == "sgd-reg":
            est = ml.SGDRegressor(loss="squared_error", **kw)
        else:
            est = ml.SGDClassifier(loss="hinge", **kw); y = (y > 0).astype(np.int32)
        t = time.perf_counter(); est.fit(X, y); ms = (time.perf_counter() - t) * 1000
        return {"ms": ms, "digest": sha(est.coef_, est.intercept_), "inputs": inputs,
                "quality": {"score": float(est.score(X[:20000], y[:20000]))}}
    if name == "lars":
        est = ml.Lars(n_nonzero_coefs=500, fit_intercept=True, random_state=7, numeric_mode="identical")
        t = time.perf_counter(); est.fit(X, y); ms = (time.perf_counter() - t) * 1000
        return {"ms": ms, "digest": sha(est.coef_, est.intercept_), "inputs": inputs,
                "quality": {"score": float(est.score(X[:20000], y[:20000]))}}
    if name == "ivf":
        q = rng.standard_normal((n // 100, D)).astype(np.float32)
        idx = ml.IVFIndex(n_lists=min(1024, n // 4), n_probes=32, n_neighbors=10, kmeans_n_iters=20,
                          metric="sqeuclidean", random_state=7, numeric_mode="identical")
        t = time.perf_counter(); idx.fit(X); out = idx.search(q); ms = (time.perf_counter() - t) * 1000
        dist, ind = (out if isinstance(out, tuple) else (out, None))
        return {"ms": ms, "digest": sha(dist, ind) if ind is not None else sha(dist), "inputs": sha(X, q)}
    raise SystemExit("unknown case " + name)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("case"); c.add_argument("name"); c.add_argument("size", type=int); c.add_argument("--out", required=True)
    a = sub.add_parser("all"); a.add_argument("--out", required=True); a.add_argument("--ivf-off-so")
    a.add_argument("--only", default=",".join(SIZES))
    args = ap.parse_args()
    if args.cmd == "case":
        r = run_case(args.name, args.size)
        r.update(case=args.name, size=args.size, env={k: v for k, v in os.environ.items() if k.startswith("MOJOLEARN_X")})
        Path(args.out).write_text(json.dumps(r, indent=2) + "\n"); print(json.dumps(r), flush=True)
        return
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    import mojolearn
    ivf_so = Path(os.environ["CLASSICAL_IVF_SO"]) if os.environ.get("CLASSICAL_IVF_SO") else None
    summary = []
    for name in args.only.split(","):
        small, full = SIZES[name]
        for arm, size in (("old", small), ("new", small), ("new", full)):
            tag = "%s-%s-%d" % (name, arm, size)
            env = dict(os.environ); swapped = None
            if arm == "old":
                env.update(OLD_ENV[name])
                if name == "ivf":
                    if not args.ivf_off_so or ivf_so is None:
                        summary.append({"tag": tag, "status": "skipped: no --ivf-off-so or CLASSICAL_IVF_SO"}); continue
                    swapped = ivf_so.with_suffix(".so.new"); shutil.copy2(ivf_so, swapped); shutil.copy2(args.ivf_off_so, ivf_so)
            print("###", tag, flush=True)
            try:
                p = subprocess.run([sys.executable, __file__, "case", name, str(size), "--out", str(out / (tag + ".json"))],
                                   env=env, capture_output=True, text=True, timeout=5400)
                (out / (tag + ".log")).write_text(p.stdout + p.stderr)
                rec = json.loads((out / (tag + ".json")).read_text()) if p.returncode == 0 else {"status": "FAILED rc=%d" % p.returncode, "tail": p.stderr[-1500:]}
            except subprocess.TimeoutExpired:
                rec = {"status": "TIMEOUT 5400 s"}
            finally:
                if swapped is not None:
                    shutil.copy2(swapped, ivf_so); swapped.unlink()
            rec["tag"] = tag; summary.append(rec); print(json.dumps({k: rec.get(k) for k in ("tag", "ms", "digest", "status")}), flush=True)
        by = {r["tag"]: r for r in summary}
        o, nw = by.get("%s-old-%d" % (name, small), {}), by.get("%s-new-%d" % (name, small), {})
        summary.append({"tag": name + "-verdict", "same_digest": bool(o.get("digest")) and o.get("digest") == nw.get("digest"),
                        "speedup_at_digest_size": (o["ms"] / nw["ms"]) if o.get("ms") and nw.get("ms") else None})
        print(json.dumps(summary[-1]), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
