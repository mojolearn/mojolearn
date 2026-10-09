#!/usr/bin/env python3
"""tools/neural_fixtures.py -- the neural-layer lanes' inputs and initial state, as files.

WHY (Andrew 2026-10-09: "what do our neural cells have to do with pytorch? they should be
separate"). The layer lanes of tools/bench_board_algos.py (kind "layer") used to build their
input tensors, their initial weights and the backward seed dy with torch inside OUR arm:
torch.Generator(SEED) + torch.randn/randint for x, nn.<Layer>(**p) default init after
torch.manual_seed(SEED) for the state, torch.Generator(SEED + 1) for dy. So our arm could not
run without torch installed, though mojolearn never imports torch.

This tool is now the ONLY place those tensors come from torch. `generate` runs once, on a box
with the CPU torch wheel (the board's torch pin, tools/bench_board.py DEFAULT_TORCH_SPEC), and
calls the harness's own builders (bench_board_algos._layer_inputs, the same seed, shapes, batch
rule and init calls the stored torch scores were made with), then writes every array as .npy
plus a manifest.json (lane, dataset, variant, shapes, dtypes, sha256 per file, torch version,
seed, smoke flag). The ours arms (ours, ours-fast) read the files through `load_for_lane`,
which verifies every sha256 and never imports torch. The torch arms keep building their own
tensors and assert they equal these bytes (bench_board_algos._neural_fixture_drift_check), so a
drift in torch's generator is caught instead of silently compared.

Layout (DIR defaults to <dirname(--data)>/neural-fixtures, i.e. beside rows-full/rows-small
under the box's algos-data mirror; MOJOLEARN_NEURAL_FIXTURES overrides):
    DIR/manifest.json
    DIR/<lane>/<dataset>/<variant>/x.npy        (absent for the graph lanes: x is the dataset's X)
    DIR/<lane>/<dataset>/<variant>/dy.npy       (absent for forward-only lanes)
    DIR/<lane>/<dataset>/<variant>/state/<torch state_dict key>.npy
<variant> is "nb64" (full) / "nb4" (smoke: the harness shrinks the batch, never the layer), "n<N>"
for the graph lanes (the dy shape follows the node count of the data dir), "bytes" for a
recurrent layer lane fed from the byte corpus.

    python3 tools/neural_fixtures.py generate --out DIR [--data DATA_DIR ...] [--lanes a,b]
    python3 tools/neural_fixtures.py verify --dir DIR
    python3 tools/neural_fixtures.py lanes

The R2 transport (tar + upload from the generating box, staged beside rows-full on every lq box)
is tools/neural_fixtures_box.sh.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time

SCHEMA = "mojolearn.neural-fixtures/1"
MANIFEST = "manifest.json"
DIRNAME = "neural-fixtures"
ENV_DIR = "MOJOLEARN_NEURAL_FIXTURES"
HERE = os.path.dirname(os.path.abspath(__file__))
MISSING_HINT = ("neural fixtures missing: run tools/neural_fixtures.py generate "
                "(tools/neural_fixtures_box.sh generate on a box, then stage) ")
GRAPH_TASKS = ("gcn", "sage")
RECURRENT_TASKS = ("lstm", "gru", "rnn")


class FixturesMissing(RuntimeError):
    """The fixture set (or this lane's entry) is not on this box: an infrastructure
    refusal (status not_ready), never a reason to fall back to torch."""
    infrastructure = True


class FixtureMismatch(RuntimeError):
    """A fixture file's bytes do not match the manifest's sha256 (or a shape does not
    match what the arm needs): refuse, never compare against unknown inputs."""
    infrastructure = True


# ---- shared rules (the harness's _layer_inputs uses layer_batch too) ---------

def layer_batch(D):
    """The layer lanes' batch: a smoke (D["_cap"] set) shrinks the batch, never the layer."""
    return 4 if D.get("_cap") else 64


def variant(task, D):
    if task in GRAPH_TASKS:
        return "n%d" % int(D["X"].shape[0])
    if task in RECURRENT_TASKS:
        return "bytes"
    return "nb%d" % layer_batch(D)


def default_dir(data):
    """MOJOLEARN_NEURAL_FIXTURES, else <dirname(data)>/neural-fixtures: one set beside the
    rows-full and rows-small mirrors (the fixtures do not depend on the row cap, except the
    graph lanes' dy, which the n<N> variant keys)."""
    env = os.environ.get(ENV_DIR)
    if env:
        return os.path.abspath(env)
    return os.path.join(os.path.dirname(os.path.abspath(data.rstrip("/"))), DIRNAME)


def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def entry_key(lane, dataset, var):
    return "%s/%s/%s" % (lane, dataset, var)


# ---- reading (the ours arms; numpy only, never torch) -----------------------

_CACHE = {}


def read_manifest(fixdir):
    path = os.path.join(fixdir, MANIFEST)
    if not os.path.isfile(path):
        raise FixturesMissing(MISSING_HINT + "(no %s)" % path)
    with open(path, encoding="utf-8") as fh:
        man = json.load(fh)
    if man.get("schema") != SCHEMA:
        raise FixtureMismatch("neural fixtures %s: schema %r, expected %r"
                              % (path, man.get("schema"), SCHEMA))
    return man


def _load_file(fixdir, rec):
    import numpy as np
    path = os.path.join(fixdir, rec["path"])
    if not os.path.isfile(path):
        raise FixturesMissing(MISSING_HINT + "(no %s)" % path)
    got = sha256_file(path)
    if got != rec["sha256"]:
        raise FixtureMismatch("neural fixture %s: sha256 %s, manifest %s" % (path, got, rec["sha256"]))
    a = np.load(path, allow_pickle=False)
    if list(a.shape) != list(rec["shape"]) or str(a.dtype) != rec["dtype"]:
        raise FixtureMismatch("neural fixture %s: %s %s, manifest %s %s"
                              % (path, a.dtype, list(a.shape), rec["dtype"], rec["shape"]))
    return a


def load(fixdir, lane, dataset, var):
    """{"x": array|None, "dy": array|None, "state": {key: array}, "entry": manifest entry,
    "manifest_sha256": ..., "torch_version": ...}; every file sha256-verified once per process."""
    fixdir = os.path.abspath(fixdir)
    ck = (fixdir, lane, dataset, var)
    if ck in _CACHE:
        return _CACHE[ck]
    man = read_manifest(fixdir)
    key = entry_key(lane, dataset, var)
    ent = man.get("entries", {}).get(key)
    if ent is None:
        raise FixturesMissing(MISSING_HINT + "(%s has no entry %s; it holds %d)"
                              % (os.path.join(fixdir, MANIFEST), key, len(man.get("entries", {}))))
    files = ent["files"]
    out = {"x": _load_file(fixdir, files["x"]) if "x" in files else None,
           "dy": _load_file(fixdir, files["dy"]) if "dy" in files else None,
           "state": {k: _load_file(fixdir, r) for k, r in files.get("state", {}).items()},
           "entry": ent, "dir": fixdir,
           "manifest_sha256": sha256_file(os.path.join(fixdir, MANIFEST)),
           "torch_version": man.get("torch_version"), "seed": man.get("seed")}
    _CACHE[ck] = out
    return out


def load_for_lane(fixdir, lane, spec, dataset, D):
    """The ours arms' entry point: the fixture of (lane, dataset, variant(task, D))."""
    if not fixdir:
        raise FixturesMissing(MISSING_HINT + "(no fixture directory: pass --neural-fixtures)")
    return load(fixdir, lane, dataset, variant(spec["task"], D))


def provenance(fx):
    return {"dir": fx["dir"], "manifest_sha256": fx["manifest_sha256"],
            "entry": entry_key(fx["entry"]["lane"], fx["entry"]["dataset"], fx["entry"]["variant"]),
            "generated_with_torch": fx["torch_version"], "seed": fx["seed"]}


def take_dy(fx, shape):
    """The backward seed: the fixture's dy, which must have the forward output's shape."""
    dy = fx["dy"]
    if dy is None:
        raise FixtureMismatch("neural fixture %s has no dy (a forward-only entry)"
                              % entry_key(fx["entry"]["lane"], fx["entry"]["dataset"], fx["entry"]["variant"]))
    if tuple(int(v) for v in shape) != tuple(dy.shape):
        raise FixtureMismatch("neural fixture dy %s != forward output %s (%s)"
                              % (tuple(dy.shape), tuple(shape), fx["entry"]["lane"]))
    return dy


# ---- writing ------------------------------------------------------------------

def _save(fixdir, rel, a):
    import numpy as np
    path = os.path.join(fixdir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    a = np.array(a, order="C")  # np.ascontiguousarray makes a 0-d tensor (num_batches_tracked) 1-d
    tmp = path + ".tmp.npy"
    np.save(tmp, a, allow_pickle=False)
    os.replace(tmp, path)
    return {"path": rel, "shape": list(a.shape), "dtype": str(a.dtype),
            "bytes": os.path.getsize(path), "sha256": sha256_file(path)}


def write_entry(fixdir, lane, dataset, var, *, task, params, smoke, x=None, dy=None, state=None,
                extra=None):
    """Save one (lane, dataset, variant) and return its manifest entry."""
    base = "%s/%s/%s" % (lane, dataset, var)
    files = {}
    if x is not None:
        files["x"] = _save(fixdir, base + "/x.npy", x)
    if dy is not None:
        files["dy"] = _save(fixdir, base + "/dy.npy", dy)
    files["state"] = {k: _save(fixdir, "%s/state/%s.npy" % (base, k), v)
                      for k, v in sorted((state or {}).items())}
    ent = {"lane": lane, "dataset": dataset, "variant": var, "task": task, "params": params,
           "smoke": bool(smoke), "files": files}
    ent.update(extra or {})
    return ent


def write_manifest(fixdir, entries, **meta):
    man = {"schema": SCHEMA, "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "generator": "tools/neural_fixtures.py", "entries": {}}
    man.update(meta)
    for e in entries:
        man["entries"][entry_key(e["lane"], e["dataset"], e["variant"])] = e
    path = os.path.join(fixdir, MANIFEST)
    with open(path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, sort_keys=True, default=str)
    os.replace(path + ".tmp", path)
    return man


def verify(fixdir):
    """Re-hash every file against the manifest; returns the list of problems."""
    man = read_manifest(fixdir)
    bad = []
    for key, ent in sorted(man["entries"].items()):
        recs = [r for n, r in ent["files"].items() if n != "state"] + list(ent["files"]["state"].values())
        for r in recs:
            p = os.path.join(fixdir, r["path"])
            if not os.path.isfile(p):
                bad.append("%s: missing %s" % (key, r["path"]))
            elif sha256_file(p) != r["sha256"]:
                bad.append("%s: sha256 mismatch %s" % (key, r["path"]))
    return bad


# ---- generate (THE ONLY torch use; runs once on a box with the CPU torch wheel) ----

def _harness():
    spec = importlib.util.spec_from_file_location("bench_board_algos", os.path.join(HERE, "bench_board_algos.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def layer_lanes(A):
    return [l for l in A.LANE_ORDER if A.LANES[l]["kind"] == "layer"]


def _cases(A, lane, data_dirs, smoke_rows):
    """(dataset, D, smoke) the ours arm can meet for this lane."""
    s = A.LANES[lane]
    out = []
    for ds in s["datasets"]:
        if ds == "synthetic" and s["task"] not in RECURRENT_TASKS:
            out.append((ds, A.lane_arrays(lane, {}), False))
            out.append((ds, A.lane_arrays(lane, {"_cap": int(smoke_rows)}), True))
            continue
        for data in data_dirs:
            try:
                B, _rec = A._load_block(lane, ds, data)
            except (FileNotFoundError, OSError) as exc:
                print("NF-SKIP lane=%s dataset=%s data=%s %s" % (lane, ds, data, exc), flush=True)
                continue
            out.append((ds, A.lane_arrays(lane, B), bool(B.get("_cap"))))
    return out


def generate(out, data_dirs, lanes=None, smoke_rows=4096):
    import torch
    A = _harness()
    os.makedirs(out, exist_ok=True)
    meta = {"torch_version": torch.__version__, "seed": A.SEED, "dy_seed": A.SEED + 1,
            "harness_sha256": sha256_file(os.path.join(HERE, "bench_board_algos.py")),
            "commit": os.environ.get("MOJOLEARN_REPO_COMMIT", "unknown"),
            "data_dirs": [os.path.abspath(d) for d in data_dirs]}
    try:
        import torch_geometric
        meta["torch_geometric_version"] = torch_geometric.__version__
    except ImportError:
        meta["torch_geometric_version"] = None
    entries, seen = [], set()
    for lane in (lanes or layer_lanes(A)):
        s = A.LANES[lane]
        t = s["task"]
        if t in GRAPH_TASKS and meta["torch_geometric_version"] is None:
            print("NF-SKIP lane=%s torch_geometric not installed" % lane, flush=True)
            continue
        for ds, D, smoke in _cases(A, lane, data_dirs, smoke_rows):
            var = variant(t, D)
            if (lane, ds, var) in seen:
                continue
            seen.add((lane, ds, var))
            # the harness's own builder: g = Generator(SEED) draws x, torch.manual_seed(SEED),
            # then make() is nn.<Layer>(**p) with its default init (bench_board_algos._build_layer)
            make, x_cpu, _extra = A._layer_inputs(lane, D, torch)
            ref = make()
            state = {k: v.detach().cpu().numpy().copy() for k, v in ref.state_dict().items()}
            dy = None
            if not s.get("forward_only"):
                if t in GRAPH_TASKS:
                    yshape = (int(x_cpu.shape[0]), int(s["params"]["out_channels"]))
                elif t in RECURRENT_TASKS:
                    with torch.no_grad():
                        yshape = tuple(ref(x_cpu)[0].shape)
                elif t in ("dropout2d", "layernorm", "batchnorm1d", "batchnorm2d"):
                    yshape = tuple(x_cpu.shape)
                else:
                    with torch.no_grad():
                        y = ref(x_cpu)
                    yshape = tuple((y[0] if isinstance(y, tuple) else y).shape)
                g = torch.Generator().manual_seed(A.SEED + 1)
                dy = torch.randn(yshape, generator=g).numpy()
            x = None if t in GRAPH_TASKS else x_cpu.detach().cpu().numpy()
            ent = write_entry(out, lane, ds, var, task=t, params=s["params"], smoke=smoke,
                              x=x, dy=dy, state=state,
                              extra={"in_channels": int(x_cpu.shape[1])} if t in GRAPH_TASKS else None)
            entries.append(ent)
            print("NF-WROTE %s files=%d" % (entry_key(lane, ds, var),
                                            len(ent["files"]["state"]) + ("x" in ent["files"]) + ("dy" in ent["files"])),
                  flush=True)
    man = write_manifest(out, entries, **meta)
    print("NF-MANIFEST entries=%d torch=%s dir=%s" % (len(man["entries"]), meta["torch_version"], out), flush=True)
    return man


def main(argv=None):
    p = argparse.ArgumentParser(prog="neural_fixtures", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--out", required=True)
    g.add_argument("--data", action="append", default=[],
                   help="an algos-data mirror (rows-full, rows-small); repeat for each the races use "
                        "(the graph lanes key their dy by the node count)")
    g.add_argument("--lanes", default="")
    g.add_argument("--smoke-rows", type=int, default=4096,
                   help="any positive value: a smoke only switches the batch to 4")
    v = sub.add_parser("verify")
    v.add_argument("--dir", required=True)
    sub.add_parser("lanes")
    args = p.parse_args(argv)
    if args.cmd == "generate":
        generate(args.out, args.data, [l for l in args.lanes.split(",") if l] or None, args.smoke_rows)
        return 0
    if args.cmd == "verify":
        bad = verify(args.dir)
        for b in bad[:20]:
            print("NF-BAD " + b)
        print("NF-VERIFY %s problems=%d" % ("ok" if not bad else "FAIL", len(bad)))
        return 1 if bad else 0
    A = _harness()
    for l in layer_lanes(A):
        print(l, A.LANES[l]["task"], ",".join(A.LANES[l]["datasets"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
