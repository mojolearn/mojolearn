"""make_small_rows.py <rows-full dir> <rows-small dir> [N]: a small copy of the prepared board data for identity checks.

Row blocks (every array whose first dim equals len(X) or len(Xq)) keep their first N rows (N/4 for the query side);
other blocks (graphs, text, time series) are copied as they are. Same input bits on every box, so device and host
digests from it compare across NVIDIA, AMD, Apple and the CPU column.
"""
import json, os, shutil, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 50_000
os.makedirs(dst, exist_ok=True)
for f in sorted(os.listdir(src)):
    s, d = os.path.join(src, f), os.path.join(dst, f)
    if not f.endswith(".npz"):
        if not os.path.exists(d):
            shutil.copy2(s, d)
        continue
    with np.load(s) as z:
        A = {k: z[k] for k in z.files}
    nx = A["X"].shape[0] if "X" in A and A["X"].ndim >= 1 else None
    nq = A["Xq"].shape[0] if "Xq" in A and A["Xq"].ndim >= 1 else None
    if nx is None or nx <= N:
        shutil.copy2(s, d)
        print(f"copy  {f}")
        continue
    out = {}
    for k, v in A.items():
        if v.ndim >= 1 and v.shape[0] == nx and k != "Xq" and not k.endswith("q"):
            out[k] = np.ascontiguousarray(v[:N])
        elif nq is not None and v.ndim >= 1 and v.shape[0] == nq:
            out[k] = np.ascontiguousarray(v[: max(1, min(nq, N // 4))])
        else:
            out[k] = v
    np.savez(d, **out)
    print(f"slice {f} X {nx}->{out['X'].shape[0]}" + (f" Xq {nq}->{out['Xq'].shape[0]}" if nq else ""))
open(os.path.join(dst, "SMALL_ROWS"), "w").write(json.dumps({"src": src, "N": N}))
