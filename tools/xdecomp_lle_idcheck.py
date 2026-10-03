"""Same-bits check of LocallyLinearEmbedding's ltsa / hessian / modified
(x_decomp/lle_local.mojo, cgr-decomp): each case is fitted on a fixed seeded
fixture in a device subprocess and in a MOJOLEARN_VENDOR=cpu subprocess, and
the embedding digests compared. Prints one line per case:
IDCHECK <case> device=<digest> host=<digest> MATCH|DIFFER (exit 1 on a DIFFER)."""
import hashlib
import os
import subprocess
import sys

CASES = {"ltsa": 8, "hessian": 10, "modified": 8}


def child(case):
    import numpy as np
    import mojolearn as ml
    r = np.random.default_rng(7)
    t = r.random(400) * 3 * np.pi
    X = np.stack([t * np.cos(t), r.random(400) * 10, t * np.sin(t)], 1).astype(np.float32)
    e = ml.LocallyLinearEmbedding(n_neighbors=CASES[case], n_components=2, method=case,
                                           eigen_solver="dense").fit(X)
    emb = np.ascontiguousarray(np.asarray(e.embedding_, dtype=np.float32))
    print(hashlib.sha256(emb.tobytes()).hexdigest()[:16])


def run(case, cpu):
    env = dict(os.environ)
    if cpu:
        env["MOJOLEARN_VENDOR"] = "cpu"
    p = subprocess.run([sys.executable, __file__, "--child", case], env=env, capture_output=True, text=True)
    if p.returncode:
        return "ERROR:" + (p.stderr.strip().splitlines() or ["?"])[-1][:120].replace(" ", "_")
    return p.stdout.strip().splitlines()[-1]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
        sys.exit(0)
    bad = 0
    for case in CASES:
        d, h = run(case, False), run(case, True)
        ok = d == h and not d.startswith("ERROR")
        bad += not ok
        print(f"IDCHECK lle-{case} device={d} host={h} {'MATCH' if ok else 'DIFFER'}")
    sys.exit(1 if bad else 0)
