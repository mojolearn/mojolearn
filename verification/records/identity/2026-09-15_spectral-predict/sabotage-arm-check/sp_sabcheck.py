"""CPU-only: predict the pod-saved spectral models through one metrics host
binary and print label and embedding hashes, the sabotage read-back and the
held-out label hashes recorded by the saving process.

usage: MOJOLEARN_HOST_DIR=<dir> MOJOLEARN_HOST_ALLOW_SABOTAGE=1 sp_sabcheck.py <models_dir> <tag>
"""
import hashlib
import json
import os
import sys

import numpy as np

from mojolearn import _backend, _classical_host

models, tag = sys.argv[1], sys.argv[2]
h = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
mod = _backend.load_host_module("_mojolearn_metrics_host")
out = {"tag": tag, "host_dir": os.environ.get("MOJOLEARN_HOST_DIR"), "sabotage": bool(mod.metrics_host_sabotage())}
for kind in ("base", "ties"):
    rec = json.load(open(os.path.join(models, f"agree.{kind}.json")))
    for name in ("spectral", "spectral-precomputed"):
        path = os.path.join(models, f"{name}-{kind}.npz")
        q = np.load(os.path.join(models, f"{name}-{kind}.queries.npy"))
        host = _classical_host.host_model(path)
        labels, emb = (np.asarray(a) for a in host._predict_embedding(q))
        out[f"{name}/{kind}"] = dict(labels=h(labels), embedding=h(emb),
                                     labels_equal_recorded=h(labels) == rec[name]["heldout_labels"],
                                     changed_labels=None)
        out[f"{name}/{kind}"]["hist"] = np.bincount(labels, minlength=4).tolist()
print(json.dumps(out, indent=1, sort_keys=True))
