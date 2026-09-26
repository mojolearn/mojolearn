#!/usr/bin/env python3
"""Hand fixes on top of migrate.py for Mojo 1.2 nightly. Usage: fixups.py <root>"""
import sys

root = sys.argv[1]
EDITS = [
    ("neighbors/impl/detail/knn_brute_force.mojo",
     '''                top_k[largest=False, target="gpu"](
                    TileTensor(dv, row_major(rows, n_index)),''',
     '''                var tin = TileTensor(dv, row_major(rows, n_index))
                top_k[largest=False, target="gpu", KEngine=type_of(tin).Engine](
                    tin,'''),
    ("checks/vendor_correctness_check.mojo",
     '''    top_k[largest=False, target="gpu"](
        TileTensor(vals, row_major(batch, n)),''',
     '''    var tin = TileTensor(vals, row_major(batch, n))
    top_k[largest=False, target="gpu", KEngine=type_of(tin).Engine](
        tin,'''),
    ("neighbors/impl/ball_cover/registers.mojo",
     "from max.gpu.primitives.warp import lane_id, shuffle_idx, vote",
     "from max.gpu import lane_id\nfrom max.gpu.primitives.warp import shuffle_idx, vote"),
    ("neighbors/impl/matrix/detail/select_warpsort.mojo",
     "from max.gpu.primitives.warp import lane_id, shuffle_xor",
     "from max.gpu import lane_id\nfrom max.gpu.primitives.warp import shuffle_xor"),
]
for rel, old, new in EDITS:
    p = f"{root}/{rel}"
    t = open(p).read()
    n = t.count(old)
    if n != 1:
        print(f"SKIP {rel}: {n} matches")
        continue
    open(p, "w").write(t.replace(old, new))
    print(f"ok {rel}")
