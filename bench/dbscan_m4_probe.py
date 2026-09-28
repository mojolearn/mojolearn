# DIAGNOSTIC ONLY (lane/cluster-apple-prof): DBSCAN taxi 100k, GPU vs CPU
# column, rbc and brute, with identity traces diffed stage by stage.
import os, subprocess, sys, tempfile, hashlib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
from bench.x_cluster_speed import load

def run(column, algo, trace, n):
    code = f'''
import os, sys, hashlib, numpy as np
sys.path.insert(0, "bench"); sys.path.insert(0, "python")
from x_cluster_speed import load
import mojolearn as ml
if "{column}" == "cpu":
    from mojolearn import _backend
    g = _backend.binding
    def hb(name, mode=None):
        base = _backend._HOST_MODULES.get(name)
        if name == "_mojolearn" or base is None:
            return g(name, mode)
        return _backend.load_host_module(base)
    _backend.binding = hb
x = np.ascontiguousarray(load("taxi", {n}))
e = ml.DBSCAN(eps=0.5, min_samples=10, algorithm="{algo}").fit(x)
lab = np.asarray(e.labels_.to_numpy() if hasattr(e.labels_, "to_numpy") else e.labels_)
print("DBPROBE {column} {algo} {n}", hashlib.sha256(lab.tobytes()).hexdigest()[:16], int(lab.max()) + 1, int((lab < 0).sum()))
'''
    env = dict(os.environ, MOJOLEARN_IDENTITY_TRACE=trace)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    print(r.stdout.strip() or r.stderr[-800:], flush=True)

d = tempfile.mkdtemp()
for n in (100000, 30000):
    for algo in ("rbc", "brute"):
        tg, tc = f"{d}/g_{algo}_{n}.trace", f"{d}/c_{algo}_{n}.trace"
        run("gpu", algo, tg, n)
        run("cpu", algo, tc, n)
        r = subprocess.run([sys.executable, "tools/identity_trace_diff.py", tg, tc], capture_output=True, text=True)
        print("TRACEDIFF", algo, n, (r.stdout + r.stderr).strip()[-1500:], flush=True)
