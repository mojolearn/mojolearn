#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Pack the per-arm files of an inference record for the repository.

Lane lane/lowbit-quality, 2026-09-29. Runs on the pod, beside the records.

    python3 bench/lowbit_quality/pack_records.py <record dir> <out dir>

`infer_eval.py` writes one `arm_<key>.pt` per arm: the float64 nll of every
scored position and the arm's top-1 id there. This writes the same numbers
as `arm_<key>.npz` (numpy, compressed): `nll` float64, unchanged bit for
bit, and `top` as uint16 (every id is below 49152). It reads each file back
and refuses to go on when a value differs. JSON files are copied as they
are. Nothing is recomputed.
"""
import glob
import hashlib
import json
import os
import shutil
import sys

import numpy as np
import torch


def main(argv):
    src, dst = argv
    os.makedirs(dst, exist_ok=True)
    index = {}
    for path in sorted(glob.glob(os.path.join(src, "arm_*.pt"))):
        d = torch.load(path, map_location="cpu")
        nll = d["nll"].numpy().astype(np.float64)
        top = d["top"].numpy()
        if top.max() > 65535 or top.min() < 0:
            raise SystemExit(f"{path}: a top-1 id does not fit uint16")
        out = os.path.join(dst, os.path.basename(path)[:-3] + ".npz")
        np.savez_compressed(out, nll=nll, top=top.astype(np.uint16))
        back = np.load(out)
        if not (np.array_equal(back["nll"].view(np.uint64), nll.view(np.uint64))
                and np.array_equal(back["top"].astype(np.int64), top.astype(np.int64))):
            raise SystemExit(f"{out}: read back differs from {path}")
        index[os.path.basename(out)] = dict(
            positions=int(nll.size), mean_nll=float(nll.mean()),
            nll_sha256=hashlib.sha256(nll.tobytes()).hexdigest(),
            top_sha256=hashlib.sha256(top.astype(np.uint16).tobytes()).hexdigest(),
            bytes=os.path.getsize(out))
    for path in sorted(glob.glob(os.path.join(src, "*.json"))):
        shutil.copy(path, os.path.join(dst, os.path.basename(path)))
    with open(os.path.join(dst, "per_arm_index.json"), "w") as fh:
        json.dump(dict(source=src, format="npz: nll float64 (per scored position), top uint16 (the arm's top-1 id)",
                       arms=index), fh, indent=1)
    print("packed", len(index), "arms from", src, "into", dst,
          "bytes", sum(v["bytes"] for v in index.values()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
