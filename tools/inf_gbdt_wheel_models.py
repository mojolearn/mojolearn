"""lane/inference-gbdt-modes (2026-09-15) installed-wheel check.
dump: source tree, CPU reference fits, save model.npz + held-out rows per lane/fixture.
check: an installed package (PYTHONPATH=<target>), host_model(model).predict[_proba] hashed
exactly as identity_break._h and compared with the infer hash of every committed GPU column.

    python3 wheel_models.py dump <outdir>      (PYTHONPATH=<worktree>/python, MOJOLEARN_HOST_DIR set)
    python3 wheel_models.py check <outdir>     (PYTHONPATH=<installed target>)
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np

WT = os.environ.get("WT", ".")
LANES = ["gbdt-ordered-rmse", "gbdt-feature-freq", "gbdt-pointwise-l2-bayesian-eval", "gbdt-categorical-ctr"]
CODED = {"gbdt-feature-freq", "gbdt-categorical-ctr"}
PROBA = {"gbdt-pointwise-l2-bayesian-eval"}
COLUMNS = ["apple-m4", "nvidia-h100-sm_90a", "amd-mi325x-gfx942"]


def _h(*arrays):
    m = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        m.update(str(a.dtype).encode())
        m.update(str(a.shape).encode())
        m.update(a.tobytes())
    return m.hexdigest()[:16]


def _ib():
    spec = importlib.util.spec_from_file_location("ib", os.path.join(WT, "tools", "identity_break.py"))
    ib = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ib)
    return ib


def dump(out):
    ib = _ib()
    import mojolearn as ml
    from mojolearn._cpu_reference import reference_training
    os.makedirs(out, exist_ok=True)
    index = []
    with reference_training():
        for fx in ib.FIXTURES:
            X, yc, yr = ib.fixture(fx)
            Xh = ib.heldout(fx)
            for lane in LANES:
                f = ib.LANES[lane](ml, X, yc, yr, Xh)
                stem = f"{lane}.{fx}"
                f.est.save(os.path.join(out, stem + ".npz"))
                xh = ib._coded(Xh) if lane in CODED else Xh
                np.save(os.path.join(out, stem + ".xh.npy"), np.ascontiguousarray(xh, dtype=np.float32))
                index.append(dict(lane=lane, fixture=fx, source_infer=_h(*f.probe(f.est))))
                print("dumped", stem, flush=True)
    with open(os.path.join(out, "index.json"), "w") as fh:
        json.dump(index, fh, indent=1)


def check(out):
    import mojolearn
    print("mojolearn from", mojolearn.__file__)
    from mojolearn._forest_host import binary_path
    print("forest binary", binary_path())
    cols = {c: json.load(open(os.path.join(WT, "bench/results/identity_break/2026-09-14_166-lanes", c + ".json")))
            for c in COLUMNS}
    index = json.load(open(os.path.join(out, "index.json")))
    equal = differ = 0
    for row in index:
        stem = f"{row['lane']}.{row['fixture']}"
        host = mojolearn.host_model(os.path.join(out, stem + ".npz"))
        xh = np.load(os.path.join(out, stem + ".xh.npy"))
        parts = (host.predict(xh),) + ((host.predict_proba(xh),) if row["lane"] in PROBA else ())
        got = _h(*parts)
        want = {c: cols[c]["cells"][f"{row['lane']}/{row['fixture']}"]["infer"] for c in COLUMNS}
        ok = all(all(v == got for v in w) for w in want.values())
        equal += ok
        differ += not ok
        print(f"{stem:45s} {host.estimator:32s} {'IDENTICAL' if ok else 'DIFFER'} {got} "
              + " ".join(f"{c}={w[0]}" for c, w in want.items()))
    print(f"summary: IDENTICAL={equal} DIFFER={differ}")
    return 0 if differ == 0 and equal else 1


#: lane/inference-gbdt-ctr-tables (2026-09-15): the two CTR table lanes, whose
#: models are the GPU column's saved files (identity_break
#: MOJOLEARN_IDENTITY_GBDT_CTR_MODELS), with the held-out transform and the
#: probe each lane hashes.
CTR_LANES = {
    "gbdt-categorical-ctr-tables": ("_ctr_tables_xh", True),
    "gbdt-tensor-ctr-tables": ("_tensor_ctr_xh", False),
}


def check_saved(models, gpu_json):
    """`check-saved <models dir> <GPU column JSON>`: every `<lane>.<fixture>.npz`
    of the two CTR table lanes, predicted through the installed package's
    `host_model` on the lane's held-out rows and hashed as the lane hashes it,
    against that column's infer cell."""
    ib = _ib()
    import mojolearn
    print("mojolearn from", mojolearn.__file__)
    from mojolearn._forest_host import binary_path
    print("forest binary", binary_path())
    col = json.load(open(gpu_json))
    equal = differ = 0
    for name in sorted(os.listdir(models)):
        lane, _, rest = name.partition(".")
        fx = rest[:-len(".npz")] if rest.endswith(".npz") else None
        if lane not in CTR_LANES or fx is None:
            continue
        transform, proba = CTR_LANES[lane]
        xh = getattr(ib, transform)(ib.heldout(fx))
        host = mojolearn.host_model(os.path.join(models, name))
        got = _h(host.predict(xh), *((host.predict_proba(xh),) if proba else ()))
        want = col["cells"][f"{lane}/{fx}"]["infer"]
        ok = all(v == got for v in want)
        equal += ok
        differ += not ok
        print(f"{lane}/{fx:14s} {'IDENTICAL' if ok else 'DIFFER'} host={got} gpu={want[0]}")
    print(f"summary: IDENTICAL={equal} DIFFER={differ}")
    return 0 if differ == 0 and equal else 1


if __name__ == "__main__":
    if sys.argv[1] == "check-saved":
        sys.exit(check_saved(sys.argv[2], sys.argv[3]))
    sys.exit(dump(sys.argv[2]) if sys.argv[1] == "dump" else check(sys.argv[2]))
