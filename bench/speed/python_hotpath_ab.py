# SPDX-License-Identifier: Apache-2.0
"""Interleaved A/B of every audited cell: arm P = MOJOLEARN_HOTPATH=python (the
reference routines), arm N = the hotpath seams. Arms alternate within ONE
process, `--rounds` rounds each; per arm the minimum and the spread (max/min)
are reported, a side whose spread exceeds 1.10 is marked `u`, and the result
hash of the two arms must be equal (`same`)."""
import hashlib, json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import python_hotpath_cells as B
from mojolearn._array import Array


def digest(value):
    h = hashlib.sha256()

    def walk(v):
        if isinstance(v, Array):
            h.update(repr((v.shape, v.dtype, v.order)).encode()); h.update(v.tobytes())
        elif isinstance(v, (list, tuple)):
            h.update(b"[%d" % len(v))
            if v and all(type(x) in (int, float, str, bool) for x in v):
                h.update(pickle.dumps(list(v), protocol=4))
            else:
                for x in v:
                    walk(x)
        elif hasattr(v, "classes") and hasattr(v, "codes"):
            walk(list(v))
        else:
            h.update(repr(v).encode())
    walk(value)
    return h.hexdigest()[:16]


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", default="1000000,4000000")
    p.add_argument("--json", default="")
    p.add_argument("--only", default="")
    p.add_argument("--skip", default="")
    p.add_argument("--rounds", type=int, default=5)
    a = p.parse_args(sys.argv[1:])
    sizes = [int(s) for s in a.sizes.split(",")]
    import numpy as np
    rows = []
    for name, make in B.CELLS:
        if a.only and a.only not in name:
            continue
        if a.skip and any(k in name for k in a.skip.split(",")):
            continue
        for n in sizes:
            made = make(n, np.random.default_rng(20260917))
            scale = 1.0
            if isinstance(made, tuple):
                made, scale = made
            t = {"P": [], "N": []}
            hashes = {}
            err = None
            for _ in range(a.rounds):
                for arm in ("P", "N"):
                    if arm == "P":
                        os.environ["MOJOLEARN_HOTPATH"] = "python"
                    else:
                        os.environ.pop("MOJOLEARN_HOTPATH", None)
                    try:
                        t0 = time.perf_counter(); out = made(); dt = (time.perf_counter() - t0) * 1e3 * scale
                    except Exception as exc:
                        err = "%s: %s" % (type(exc).__name__, str(exc)[:80]); break
                    t[arm].append(dt)
                    hashes.setdefault(arm, digest(out))
                    del out
                if err:
                    break
            os.environ.pop("MOJOLEARN_HOTPATH", None)
            if err:
                row = {"cell": name, "n": n, "error": err}
            else:
                row = {"cell": name, "n": n, "same": hashes["P"] == hashes["N"]}
                for arm in ("P", "N"):
                    lo, hi = min(t[arm]), max(t[arm])
                    row[arm + "_ms"] = round(lo, 2); row[arm + "_spread"] = round(hi / lo, 3)
                    row[arm + "_gate"] = "ok" if hi / lo <= 1.10 else "u"
            rows.append(row)
            print(json.dumps(row), flush=True)
    if a.json:
        json.dump({"rounds": a.rounds, "python": sys.version.split()[0], "rows": rows}, open(a.json, "w"), indent=1)


main()
