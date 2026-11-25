# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE COMPOSED-MODEL LANE (lane session-ops-verify, Andrew 2026-11-25): one generic lane, `ops-composed-cases`, driven
# by data. There is no verify lane per composition: a composition is a CASE, a data file, and adding a case is adding
# a file (no code). The design is docs/plans/SESSION_OPS_PARITY.md "Verify design".
#
# WHERE THE CASES LIVE: `python/mojolearn/verify_reference/composed-*.json` (in a wheel `mojolearn/verify_reference/`;
# every packer and pyproject's package-data already ship `verify_reference/*.json`, so a new case file needs no
# packaging change). This fragment reads them beside the harness first (a wheel's mojolearn/_identity_break.py), else
# from the checkout's python tree (tools/identity_break.py). A lane that finds no case REFUSES: a lane that runs
# nothing cannot fail.
#
# THE CASE FORMAT (schema "mojolearn.composed-cases.v1"): a file is {"schema", "cases": [case, ...]}; related cases that
# exercise the same implementation share a file. A case is {"name", "about", "data": {name: spec}, "do": [statement]}.
# Data specs (float32 unless a dtype is named): {"hashed": shape, "lo", "hi"} (the harness's hashed stream, seeded
# "composed:<case>:<name>"), {"value": nested list, "dtype"}, {"zeros": shape}, {"ones": shape},
# {"arange": n, "shape": shape}. A statement is a dict whose "op" is one of the verbs below. Values in a statement:
# "@name" or "@name.path" (a dict key, a list index or an attribute: "@prog.params", "@p.w") reads the environment;
# "@@text" is the literal "@text"; {"tuple": [...]} is a tuple (shapes, permutations); lists and dicts resolve inside.
# The graph is written as calls of the public `ops.Session` / `Program` methods by name, so any op the package has
# (including ones added later) is usable in a case with no change here.
#   session {as, record, do}          a `with ops.Session(record=..., numeric_mode="identical")` block
#   call {fn: "obj.method", args, kw, as}  call a method of an environment object; `as` a name or a list (unpacks)
#   map {fn, over, kw, as}            call fn(value, **kw) for each value of a dict or list (parameters, inputs)
#   let {as, value}                   bind a resolved value
#   read {fn, of, as}                 fn on each handle of a tree ("s.numpy", "s.grad"), each result as an ndarray
#   array {of, dtype, as}             np.asarray
#   np {fn, args, kw, as}             a NumPy function on host arrays (subtract, negative, ...): oracles only
#   repeat {n, var, do}               the body n times, `var` = 0..n-1
#   def {name, do} / use {name, with} a macro: `use` binds `with` into the environment and runs the body
#   path {name, as}                   a file name in the case's temporary directory
#   roundtrip {of, as}                every array of a tree written to an .npz file and read back (a checkpoint)
#   state_save {on, opt, keys, as}    optimizer_state_dict with the state tensors named in keys read back as arrays;
#                                     also binds <as>_arrays, those arrays in (state key, keys) order
#   find {of, in, as} / pick {from, at, as}  positions of handles in a list (Program.params) / the items at them
#   state_load {on, opt, from, keys}  load_optimizer_state_dict from that saved form
#   digest {part, of}                 hash a tree's arrays into part "<case>:<part>" (statements append in order)
#   same {a, b, what}                 every array of two trees the same bytes, else REFUSE
#   differ {a, b, what}               the two trees not all equal, else REFUSE
#   zero {of, index, what}            the indexed block (ints and [start, stop] pairs) exactly zero, else REFUSE
#   close {a, b, rtol, atol, what}    float64 closeness, else REFUSE
#   oracle {kind: "weighted_ce", logits, codes, weight, as}  float64 sum(w * ce) / sum(w)
#   fd_grad {program, inputs, wrt, grad, eps, rtol, atol, what}  central differences of a captured program's first
#                                     output (a scalar) against a gradient, element by element (tiny tensors only)
# Lane stream-fits (2026-11-25), for the streamed fits and their fresh-process resumes:
#   make {cls, args, kw, as}          call a public class or classmethod by its dotted name under the package
#                                     ("Sequential", "ops.BatchSource", "GRURegressor", "Sequential.from_checkpoint")
#   shards {as, lengths, F, targets, seal}  a batch shard directory (mojolearn-batch-shards v1) in the case's
#                                     temporary directory: one shard per list of record lengths, steps (total, F)
#                                     from the hashed stream, int64 offsets; targets {kind: "dense" (width W),
#                                     "classes" (classes K, width W: whole numbers in [0, K)) or "steps" (width w
#                                     a step, padded to the longest record)}; sealed unless seal is false; binds the
#                                     directory's path
#   pickle {dump, as} / pickle {load, as}  an object pickled to a file of the case's temporary directory (binds the
#                                     path) / read back
#   fresh {pass, do, keep}            run `do` in a FRESH PROCESS on this side (a child Python that loads this
#                                     harness): the names in `pass` (paths, numbers, arrays) go in, the arrays
#                                     named in `keep` come back into this case's environment
#
# A fragment of tools/identity_break.py (loaded by `_load_lane_fragments`; np, os, json, tempfile, _h, _fit, _hw,
# _same_bytes, lane and _batch_decl are that namespace's). Device-agnostic (docs/plans/VERIFY_THROUGH_DEVICE.md
# contract 1): no `device=`; the harness's view gives `ops.Session` its side. Session and Program expose no `digest_`
# or `training_digest_`: the runner hashes the outputs and states a case names. Every reference is OWED.
#
# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv, group session-ops-verify): the RUNNER's arms
# coverage-autograd-session_ops_verify_drop_accumulate_{gpu,cpu} (autograd/session_ops_verify.mojo: the tape drops
# every second gradient contribution to a tensor used on two paths, on one side) and the tagger case's kept arms
# coverage-autograd-nn_composed_{scan_key,group_state}_{gpu,cpu} (autograd/nn_composed.mojo), and the checkpoint cases'
# arms coverage-autograd-nn_ckpt_{fresh_mask,skip_node}_{gpu,cpu} (autograd/nn_ckpt.mojo; composed-checkpoint.json).
# Lane stream-fits-2: the streamed recurrent fit's arms coverage-sequence-stream_fit_{cw_batch,val_ahead}_{gpu,cpu}
# (sequence/recurrent.mojo; case gru-stream-cw-val-resume in composed-stream.json).

_CC_SCHEMA = "mojolearn.composed-cases.v1"
_CC_PREFIX = "composed-"


def _cc_files():
    """The case files, in name order: beside the harness (a wheel), else the checkout's python tree."""
    here = os.path.dirname(os.path.abspath(__file__))
    for base in (os.path.join(here, "verify_reference"),
                 os.path.join(os.path.dirname(here), "python", "mojolearn", "verify_reference")):
        if os.path.isdir(base):
            names = sorted(n for n in os.listdir(base) if n.startswith(_CC_PREFIX) and n.endswith(".json"))
            if names:
                return [os.path.join(base, n) for n in names]  # driver-loop: over the case files
    raise ValueError("ops-composed-cases: no composed-*.json case file found under verify_reference")


def _cc_data(case, name, spec):
    dtype = np.dtype(spec.get("dtype", "float32"))
    if "hashed" in spec:
        return _hw(tuple(spec["hashed"]), f"composed:{case}:{name}", spec["lo"], spec["hi"]).astype(dtype)
    if "value" in spec:
        return np.ascontiguousarray(np.array(spec["value"], dtype=dtype))
    if "zeros" in spec:
        return np.zeros(tuple(spec["zeros"]), dtype)
    if "ones" in spec:
        return np.ones(tuple(spec["ones"]), dtype)
    if "arange" in spec:
        return np.arange(spec["arange"], dtype=dtype).reshape(tuple(spec.get("shape", (spec["arange"],))))
    raise ValueError(f"ops-composed-cases: case {case!r} data {name!r}: unknown spec {sorted(spec)}")


def _cc_ref(env, text, where):
    parts = text[1:].split(".")
    if parts[0] not in env:
        raise ValueError(f"{where}: unknown name {parts[0]!r}")
    v = env[parts[0]]
    for p in parts[1:]:  # small-loop(a reference's path)
        if isinstance(v, dict):
            v = v[p]
        elif isinstance(v, (list, tuple)) and p.lstrip("-").isdigit():
            v = v[int(p)]
        else:
            v = getattr(v, p)
    return v


def _cc_val(env, x, where):
    if isinstance(x, str):
        if x.startswith("@@"):
            return x[1:]
        return _cc_ref(env, x, where) if x.startswith("@") else x
    if isinstance(x, list):
        return [_cc_val(env, v, where) for v in x]  # driver-loop: over a statement's values
    if isinstance(x, dict):
        if set(x) == {"tuple"}:
            return tuple(_cc_val(env, v, where) for v in x["tuple"])  # driver-loop: over a tuple's values
        return {k: _cc_val(env, v, where) for k, v in x.items()}  # driver-loop: over a dict's values
    return x


def _cc_bind(env, names, value):
    if names is None:
        return
    if isinstance(names, list):
        value = list(value)
        if len(value) != len(names):
            raise ValueError(f"ops-composed-cases: {len(names)} names for {len(value)} results")
        for n, v in zip(names, value):  # driver-loop: over the results
            if n is not None:
                env[n] = v
        return
    env[names] = value


def _cc_method(env, fn, where):
    obj, _, name = fn.rpartition(".")
    if not obj:
        raise ValueError(f"{where}: fn {fn!r} must be 'object.method'")
    return getattr(_cc_ref(env, "@" + obj, where), name)


def _cc_leaves(tree):
    """The arrays of a tree in a fixed order (dict insertion order, then list order)."""
    if isinstance(tree, dict):
        return [a for v in tree.values() for a in _cc_leaves(v)]  # driver-loop: over a tree
    if isinstance(tree, (list, tuple)):
        return [a for v in tree for a in _cc_leaves(v)]  # driver-loop: over a tree
    return [np.asarray(tree)]


def _cc_apply(fn, tree):
    if isinstance(tree, dict):
        return {k: _cc_apply(fn, v) for k, v in tree.items()}  # driver-loop: over a tree
    if isinstance(tree, (list, tuple)):
        return [_cc_apply(fn, v) for v in tree]  # driver-loop: over a tree
    return fn(tree)


def _cc_index(spec):
    return tuple(slice(*s) if isinstance(s, list) else s for s in spec)  # small-loop(index axes)


def _cc_run(ml, env, body, ctx):
    for i, st in enumerate(body):  # driver-loop: over a case's statements
        op = st.get("op")
        where = f"ops-composed-cases: case {ctx['case']!r} statement {op!r} #{i}"
        verb = _CC_VERBS.get(op)
        if verb is None:
            raise ValueError(f"{where}: unknown verb")
        verb(ml, env, st, ctx, where)


def _cc_session(ml, env, st, ctx, where):
    with ml.ops.Session(record=bool(st.get("record", False)),
                        numeric_mode=st.get("numeric_mode", "identical")) as s:
        env[st["as"]] = s
        _cc_run(ml, env, st["do"], ctx)


def _cc_call(ml, env, st, ctx, where):
    fn = _cc_method(env, st["fn"], where)
    _cc_bind(env, st.get("as"), fn(*_cc_val(env, st.get("args", []), where), **_cc_val(env, st.get("kw", {}), where)))


def _cc_map(ml, env, st, ctx, where):
    fn = _cc_method(env, st["fn"], where)
    kw = _cc_val(env, st.get("kw", {}), where)
    over = _cc_val(env, st["over"], where)
    if isinstance(over, dict):
        _cc_bind(env, st["as"], {k: fn(v, **kw) for k, v in over.items()})  # glue: one call per entry
    else:
        _cc_bind(env, st["as"], [fn(v, **kw) for v in over])  # glue: one call per entry


def _cc_let(ml, env, st, ctx, where):
    env[st["as"]] = _cc_val(env, st["value"], where)


def _cc_read(ml, env, st, ctx, where):
    fn = _cc_method(env, st["fn"], where)
    env[st["as"]] = _cc_apply(lambda h: np.asarray(fn(h)), _cc_val(env, st["of"], where))


def _cc_array(ml, env, st, ctx, where):
    env[st["as"]] = np.asarray(_cc_val(env, st["of"], where), dtype=st.get("dtype"))


def _cc_np(ml, env, st, ctx, where):
    fn = getattr(np, st["fn"])
    env[st["as"]] = fn(*_cc_val(env, st.get("args", []), where), **_cc_val(env, st.get("kw", {}), where))


def _cc_repeat(ml, env, st, ctx, where):
    for k in range(int(st["n"])):  # driver-loop: over a case's repeated block (training steps)
        env[st["var"]] = k
        _cc_run(ml, env, st["do"], ctx)


def _cc_def(ml, env, st, ctx, where):
    ctx["macros"][st["name"]] = st["do"]


def _cc_use(ml, env, st, ctx, where):
    if st["name"] not in ctx["macros"]:
        raise ValueError(f"{where}: unknown macro {st['name']!r}")
    env.update(_cc_val(env, st.get("with", {}), where))
    _cc_run(ml, env, ctx["macros"][st["name"]], ctx)


def _cc_path(ml, env, st, ctx, where):
    env[st["as"]] = os.path.join(ctx["tmp"], st["name"])


def _cc_roundtrip(ml, env, st, ctx, where):
    tree = _cc_val(env, st["of"], where)
    leaves = _cc_arrays(tree)
    path = os.path.join(ctx["tmp"], f"roundtrip-{ctx['serial']}.npz")
    ctx["serial"] += 1
    np.savez(path, *leaves)
    with np.load(path) as z:
        back = [np.array(z[f"arr_{k}"]) for k in range(len(leaves))]  # driver-loop: over the saved arrays
    env[st["as"]] = _cc_restore(tree, back)


def _cc_arrays(tree):
    """The non-scalar leaves of a tree (arrays), in walk order; numbers, words and None stay in the tree."""
    if isinstance(tree, dict):
        return [a for v in tree.values() for a in _cc_arrays(v)]  # driver-loop: over a tree
    if isinstance(tree, (list, tuple)):
        return [a for v in tree for a in _cc_arrays(v)]  # driver-loop: over a tree
    if isinstance(tree, (int, float, str, bool)) or tree is None:
        return []
    return [np.asarray(tree)]


def _cc_restore(tree, back):
    """The tree with each non-scalar leaf replaced, in walk order, by its read-back copy."""
    if isinstance(tree, dict):
        return {k: _cc_restore(v, back) for k, v in tree.items()}  # driver-loop: over a tree
    if isinstance(tree, (list, tuple)):
        return type(tree)(_cc_restore(v, back) for v in tree)  # driver-loop: over a tree (tuples stay tuples)
    if isinstance(tree, (int, float, str, bool)) or tree is None:
        return tree
    return back.pop(0)


def _cc_state_save(ml, env, st, ctx, where):
    s = _cc_ref(env, "@" + st["on"], where)
    keys = set(st["keys"])
    sd = s.optimizer_state_dict(_cc_val(env, st["opt"], where))
    state = {k: {n: (np.asarray(s.numpy(v)) if n in keys else v) for n, v in entry.items()}  # glue: per state
             for k, entry in sd["state"].items()}
    env[st["as"]] = {"epoch": sd["epoch"], "param_groups": sd["param_groups"], "state": state}
    env[st["as"] + "_arrays"] = [state[k][n] for k in sorted(state) for n in st["keys"]]  # driver-loop: over states


def _cc_find(ml, env, st, ctx, where):
    pool = list(_cc_val(env, st["in"], where))
    env[st["as"]] = _cc_apply(pool.index, _cc_val(env, st["of"], where))


def _cc_pick(ml, env, st, ctx, where):
    pool = list(_cc_val(env, st["from"], where))
    env[st["as"]] = _cc_apply(lambda k: pool[int(k)], _cc_val(env, st["at"], where))


def _cc_state_load(ml, env, st, ctx, where):
    s = _cc_ref(env, "@" + st["on"], where)
    keys = set(st["keys"])
    saved = _cc_val(env, st["from"], where)
    state = {k: {n: (s.tensor(v) if n in keys else v) for n, v in entry.items()}  # glue: per state
             for k, entry in saved["state"].items()}
    s.load_optimizer_state_dict(_cc_val(env, st["opt"], where),
                                {"epoch": saved["epoch"], "param_groups": saved["param_groups"], "state": state})


def _cc_digest(ml, env, st, ctx, where):
    key = f"{ctx['case']}:{st['part']}"
    ctx["parts"].setdefault(key, []).extend(_cc_leaves(_cc_val(env, st["of"], where)))


def _cc_same(ml, env, st, ctx, where):
    a, b = _cc_leaves(_cc_val(env, st["a"], where)), _cc_leaves(_cc_val(env, st["b"], where))
    if len(a) != len(b):
        raise ValueError(f"{where}: {st.get('what', '')}: {len(a)} arrays against {len(b)}")
    for k, (x, y) in enumerate(zip(a, b)):  # driver-loop: over the compared arrays
        _same_bytes(f"{ctx['case']} {st.get('what', '')} [{k}] a", x, f"{ctx['case']} {st.get('what', '')} [{k}] b", y)


def _cc_differ(ml, env, st, ctx, where):
    a, b = _cc_leaves(_cc_val(env, st["a"], where)), _cc_leaves(_cc_val(env, st["b"], where))
    if len(a) == len(b) and all(x.shape == y.shape and np.array_equal(x, y) for x, y in zip(a, b)):
        raise ValueError(f"{where}: {st.get('what', 'the two values must differ')}: they are equal")


def _cc_zero(ml, env, st, ctx, where):
    v = np.asarray(_cc_val(env, st["of"], where))
    block = v[_cc_index(st.get("index", []))]
    if np.any(block != 0):
        raise ValueError(f"{where}: {st.get('what', 'the block must be zero')}: {int(np.count_nonzero(block))} "
                         "nonzero values")


def _cc_close(ml, env, st, ctx, where):
    a = np.asarray(_cc_val(env, st["a"], where), np.float64)
    b = np.asarray(_cc_val(env, st["b"], where), np.float64)
    rtol, atol = float(st.get("rtol", 1e-5)), float(st.get("atol", 0.0))
    if a.shape != b.shape and a.size != b.size:
        raise ValueError(f"{where}: {st.get('what', '')}: shapes {a.shape} and {b.shape}")
    a, b = a.reshape(-1), b.reshape(-1)
    worst = np.abs(a - b) - (atol + rtol * np.abs(b))
    if not np.all(worst <= 0):
        k = int(np.argmax(worst))
        raise ValueError(f"{where}: {st.get('what', 'values must agree')}: element {k} is {a[k]!r}, want {b[k]!r} "
                         f"(rtol {rtol}, atol {atol})")


def _cc_oracle(ml, env, st, ctx, where):
    if st["kind"] != "weighted_ce":
        raise ValueError(f"{where}: unknown oracle {st['kind']!r}")
    z = np.asarray(_cc_val(env, st["logits"], where), np.float64)
    z = z.reshape(-1, z.shape[-1])
    codes = np.asarray(_cc_val(env, st["codes"], where)).reshape(-1).astype(np.int64)
    w = np.asarray(_cc_val(env, st["weight"], where), np.float64).reshape(-1)
    top = z.max(axis=1)
    ce = top + np.log(np.exp(z - top[:, None]).sum(axis=1)) - z[np.arange(z.shape[0]), codes]
    env[st["as"]] = np.array([(w * ce).sum() / w.sum()], np.float64)


def _cc_fd_grad(ml, env, st, ctx, where):
    prog = _cc_val(env, st["program"], where)
    inputs = [np.array(a, dtype=np.float32) for a in _cc_val(env, st["inputs"], where)]  # glue: copies
    wrt = int(st["wrt"])
    g = np.asarray(_cc_val(env, st["grad"], where), np.float64).reshape(-1)
    eps = float(st.get("eps", 1e-2))
    rtol, atol = float(st.get("rtol", 2e-2)), float(st.get("atol", 1e-3))
    base = inputs[wrt].reshape(-1)
    if base.size != g.size:
        raise ValueError(f"{where}: the gradient has {g.size} values, input {wrt} has {base.size}")

    def value(arrays):
        out = prog.run(*arrays)
        out = out[0] if isinstance(out, tuple) else out
        return float(np.asarray(out, np.float64).reshape(-1)[0])
    for k in range(base.size):  # driver-loop: over the input's elements (a tiny oracle)
        plus, minus = [a.copy() for a in inputs], [a.copy() for a in inputs]  # glue: per probe
        plus[wrt].reshape(-1)[k] += np.float32(eps)
        minus[wrt].reshape(-1)[k] -= np.float32(eps)
        fd = (value(plus) - value(minus)) / (2.0 * eps)
        if not abs(g[k] - fd) <= atol + rtol * abs(fd):
            raise ValueError(f"{where}: {st.get('what', 'gradient')}: element {k} is {g[k]!r}, central "
                             f"difference {fd!r} (a gradient contribution lost or doubled?)")


def _cc_make(ml, env, st, ctx, where):
    obj = ml
    for part in st["cls"].split("."):  # small-loop(a dotted public name)
        obj = getattr(obj, part)
    _cc_bind(env, st.get("as"), obj(*_cc_val(env, st.get("args", []), where), **_cc_val(env, st.get("kw", {}),
                                                                                         where)))


def _cc_shards(ml, env, st, ctx, where):
    """A sealed batch shard directory (autograd/batch_source.mojo's format), written with numpy.save alone."""
    root = os.path.join(ctx["tmp"], st["as"] + "-shards")
    os.makedirs(root)
    width = int(st["F"])
    spec = st.get("targets")
    longest = max(max(lengths) for lengths in st["lengths"])  # glue: the fixture's record lengths
    lines = ["mojolearn-batch-shards v1"]
    for k, lengths in enumerate(st["lengths"]):  # driver-loop: over the fixture's shards
        stem = f"part{k}"
        key = f"composed:{ctx['case']}:{st['as']}"
        steps = _hw((sum(lengths), width), f"{key}:steps{k}", -1.0, 1.0)
        offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        np.save(os.path.join(root, stem + ".steps.npy"), np.ascontiguousarray(steps, np.float32))
        np.save(os.path.join(root, stem + ".offsets.npy"), offsets)
        if spec is not None:
            n = len(lengths)
            kind = spec["kind"]
            if kind == "dense":
                target = _hw((n, int(spec["width"])), f"{key}:targets{k}", -1.0, 1.0)
            elif kind == "classes":
                classes = int(spec["classes"])
                draw = _hw((n, int(spec.get("width", 1))), f"{key}:targets{k}", 0.0, float(classes))
                target = np.clip(np.floor(np.asarray(draw, np.float64)), 0, classes - 1)
            elif kind == "steps":
                target = _hw((n, int(spec["width"]) * longest), f"{key}:targets{k}", -1.0, 1.0)
            else:
                raise ValueError(f"{where}: unknown targets kind {kind!r}")
            np.save(os.path.join(root, stem + ".targets.npy"), np.ascontiguousarray(target, np.float32))
        lines.append("shard " + stem)
    with open(os.path.join(root, "index.txt"), "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
    if st.get("seal", True) and ml.ops.BatchSource.seal(root) != len(st["lengths"]):
        raise ValueError(f"{where}: BatchSource.seal did not seal every shard")
    env[st["as"]] = root


def _cc_pickle(ml, env, st, ctx, where):
    import pickle
    if "dump" in st:
        path = os.path.join(ctx["tmp"], f"pickle-{ctx['serial']}.pkl")
        ctx["serial"] += 1
        with open(path, "wb") as fh:
            pickle.dump(_cc_val(env, st["dump"], where), fh)
        env[st["as"]] = path
        return
    with open(_cc_val(env, st["load"], where), "rb") as fh:
        env[st["as"]] = pickle.load(fh)  # the file this case wrote above


_CC_FRESH_CHILD = r'''
import importlib
import importlib.util
import pickle
import sys
import numpy as np
name, harness, payload, side, out = sys.argv[1:6]
if name.startswith("mojolearn."):
    ib = importlib.import_module(name)
else:
    spec = importlib.util.spec_from_file_location("_cc_fresh_harness", harness)
    ib = importlib.util.module_from_spec(spec)
    sys.modules["_cc_fresh_harness"] = ib
    spec.loader.exec_module(ib)
import mojolearn
from mojolearn import _verify_device as vd
with open(payload, "rb") as fh:
    job = pickle.load(fh)
env = dict(job["env"])
with vd.side(side):
    ctx = {"case": job["case"], "tmp": job["tmp"], "parts": {}, "macros": dict(job["macros"]), "serial": 1000}
    ib._cc_run(vd.view(mojolearn, side), env, job["do"], ctx)
np.savez(out, **{k: np.asarray(env[k]) for k in job["keep"]})
'''


def _cc_fresh(ml, env, st, ctx, where):
    """The statements in a fresh process on this side (the resume a user makes after a crash): the passed names go
    in by pickle, the kept arrays come back by .npz."""
    import pickle
    import subprocess
    import sys
    serial = ctx["serial"]
    ctx["serial"] += 1
    script = os.path.join(ctx["tmp"], f"fresh-{serial}.py")
    payload = os.path.join(ctx["tmp"], f"fresh-{serial}.pkl")
    out = os.path.join(ctx["tmp"], f"fresh-{serial}.npz")
    with open(script, "w", encoding="utf-8") as fh:
        fh.write(_CC_FRESH_CHILD)
    job = {"case": ctx["case"], "tmp": ctx["tmp"], "macros": ctx["macros"], "do": st["do"], "keep": st["keep"],
           "env": {k: env[k] for k in st.get("pass", [])}}  # glue: the passed names
    with open(payload, "wb") as fh:
        pickle.dump(job, fh)
    done = subprocess.run([sys.executable, script, __name__, os.path.abspath(__file__), payload, _lane_side(), out],
                          capture_output=True, text=True, timeout=900, check=False)
    if done.returncode != 0:
        raise ValueError(f"{where}: the fresh process failed: " + (done.stderr or done.stdout)[-2000:])
    with np.load(out) as z:
        for k in st["keep"]:  # driver-loop: over the kept arrays
            env[k] = np.array(z[k])


_CC_VERBS = {
    "session": _cc_session, "call": _cc_call, "map": _cc_map, "let": _cc_let, "read": _cc_read,
    "array": _cc_array, "np": _cc_np, "repeat": _cc_repeat, "def": _cc_def, "use": _cc_use, "path": _cc_path,
    "roundtrip": _cc_roundtrip, "state_save": _cc_state_save, "state_load": _cc_state_load,
    "digest": _cc_digest, "same": _cc_same, "differ": _cc_differ, "zero": _cc_zero, "close": _cc_close,
    "oracle": _cc_oracle, "fd_grad": _cc_fd_grad, "find": _cc_find, "pick": _cc_pick,
    "make": _cc_make, "shards": _cc_shards, "pickle": _cc_pickle, "fresh": _cc_fresh,  # lane stream-fits
}


@lane("ops-composed-cases")
def _(ml, X, yc, yr, Xh=None):
    """Every composed-model case under verify_reference/composed-*.json, run in file and case order on this side.
    Each case builds its graph from public Session / Program calls, trains, saves, resumes or replays as its
    statements say, REFUSES when one of its checks fails, and adds its named parts ("<case>:<part>") to the digest.
    The two sides' digests must match each other and the reference (OWED)."""
    parts = {}
    seen = set()
    for path in _cc_files():  # driver-loop: over the case files
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if doc.get("schema") != _CC_SCHEMA:
            raise ValueError(f"ops-composed-cases: {os.path.basename(path)} has schema {doc.get('schema')!r}, "
                             f"not {_CC_SCHEMA!r}")
        for case in doc["cases"]:  # driver-loop: over a file's cases
            name = case["name"]
            if name in seen:
                raise ValueError(f"ops-composed-cases: two cases named {name!r}")
            seen.add(name)
            env = {k: _cc_data(name, k, spec) for k, spec in case.get("data", {}).items()}  # glue: per array
            with tempfile.TemporaryDirectory(prefix="ib-composed-") as tmp:
                ctx = {"case": name, "tmp": tmp, "parts": {}, "macros": {}, "serial": 0}
                _cc_run(ml, env, case["do"], ctx)
            if not ctx["parts"]:
                raise ValueError(f"ops-composed-cases: case {name!r} digests nothing")
            parts.update({k: _h(*v) for k, v in ctx["parts"].items()})  # glue: one hash per part
    return _fit(parts)


_batch_decl("n/a:training-step (data-driven composed models: seeded training steps, replays and resumes through the "
            "ops session and Program; digested whole)", "ops-composed-cases")
