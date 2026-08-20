"""Prepare the raw-float side of the INFERENCE benchmark: column-major X
(kept features only, prep order) and the borders in the device layout
`BinarizeFloatFeatureImpl` reads (`binarize.cu:44-52`): per feature, one
float holding the border COUNT, then the sorted border values.

    pixi run -e bench python tools/predict_prep.py <dir> <name> <border>

Must be run AFTER tools/interleaved_prep.py for the same <dir>/<name>: the
kept-feature filter and ordering are re-derived from the SAME borders tsv,
so column i of the colmajor file is fold-file feature i exactly.
"""
import sys
import numpy as np

d, name, B = sys.argv[1], sys.argv[2], int(sys.argv[3])

x = np.load("%s/%s_X.npy" % (d, name))
grid = {}
for line in open("%s/%s_borders_%d.tsv" % (d, name, B)):
    line = line.strip()
    if line:
        fi, bv = line.split("\t")
        grid.setdefault(int(fi), []).append(float(bv))
kept = sorted(f for f in grid if grid[f])

xk = np.ascontiguousarray(x[:, kept].T).astype(np.float32)  # kept-major
xk.tofile("%s/%s_X_colmajor_%d.f32" % (d, name, B))

flat = []
for f in kept:
    borders = np.sort(np.array(grid[f], dtype=np.float32))
    flat.append(np.array([len(borders)], dtype=np.float32))
    flat.append(borders)
np.concatenate(flat).tofile("%s/%s_borders_%d.f32bin" % (d, name, B))
print(name, "B", B, "kept", len(kept), "colmajor+borders written")
