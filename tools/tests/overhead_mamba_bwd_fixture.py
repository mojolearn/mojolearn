# SPDX-License-Identifier: Apache-2.0
"""Light same-bits and same-refusals fixture for the recurrent block backward.

Lane overhead-mamba-bwd (2026-10-28) replaced the block reconstruction in the
Mamba-1/2/3 backward with a direct current-weight check. This fixture runs on
the CPU column in identical mode at tiny sizes and writes, per operation, the
SHA-256 of the forward output and of every backward gradient, plus the
exception type and message of each refusal case. Run it on the before and
after builds and compare the two JSON files with --compare.

Usage (CPU column; MOJOLEARN_HOST_DIR holds the host bindings under test):
  MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=<dir> PYTHONPATH=python \
      python tools/tests/overhead_mamba_bwd_fixture.py --out <file.json>
  python tools/tests/overhead_mamba_bwd_fixture.py --compare <before.json> <after.json>
"""
import argparse
import hashlib
import json
import sys

B, L, D = 2, 8, 32


def _shapes(kind, d):
    di = 2 * d
    if kind == "mamba1":
        r = (d + 15) // 16
        return {"norm.weight": (d,), "in_proj.weight": (2 * di, d), "conv1d.weight": (di, 1, 4),
                "conv1d.bias": (di,), "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
                "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,), "out_proj.weight": (d, di)}
    nh = di // 64
    if kind == "mamba2":
        cd = di + 256
        return {"block_norm.weight": (d,), "in_proj.weight": (2 * di + 256 + nh, d),
                "conv1d.weight": (cd, 1, 4), "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,),
                "D": (nh,), "norm.weight": (di,), "out_proj.weight": (d, di)}
    dip = 2 * di + 256 + 3 * nh + 32
    return {"block_norm.weight": (d,), "in_proj.weight": (dip, d), "dt_bias": (nh,),
            "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128),
            "C_bias": (nh, 128), "D": (nh,), "out_proj.weight": (d, di)}


def _weights(np, kind, d, seed, conv2d=False):
    rng = np.random.default_rng(seed)
    out = {}
    for name, shape in _shapes(kind, d).items():
        if name.endswith("norm.weight"):
            v = 1.0 + 0.05 * rng.standard_normal(shape)
        elif name == "A_log":
            v = np.log(1.0 + rng.random(shape))
        else:
            v = 0.1 * rng.standard_normal(shape)
        v = np.ascontiguousarray(v, dtype=np.float32)
        if conv2d and name == "conv1d.weight":
            v = np.ascontiguousarray(v.reshape(shape[0], 4))
        out[name] = v
    return out


def _digest(np, value):
    a = np.ascontiguousarray(np.asarray(value), dtype=np.float32)
    return hashlib.sha256(a.tobytes()).hexdigest()


def _block(ml, kind, weights, **kw):
    cls = {"mamba1": ml.Mamba1Block, "mamba2": ml.Mamba2Block, "mamba3": ml.Mamba3Block}[kind]
    return cls(weights, **kw)


def _inputs(np, d, seed):
    rng = np.random.default_rng(seed)
    x = np.ascontiguousarray(0.5 * rng.standard_normal((B, L, d)), dtype=np.float32)
    g = np.ascontiguousarray(0.1 * rng.standard_normal((B, L, d)), dtype=np.float32)
    return x, g


def _same_bits(np, ml):
    ops = {}
    cases = [("mamba1", {}, False), ("mamba1_conv2d", {}, True),
             ("mamba2", {}, False), ("mamba2_dt_limit", {"dt_limit": (0.001, 0.5)}, False),
             ("mamba3", {}, False)]
    for label, kw, conv2d in cases:
        kind = label.split("_")[0]
        block = _block(ml, kind, _weights(np, kind, D, 7, conv2d), **kw)
        x, g = _inputs(np, D, 11)
        row = {"forward": _digest(np, block.forward(x))}
        grads = block.backward(x, g)
        for name in sorted(grads):  # test-only digest over returned gradients
            row["grad:" + name] = _digest(np, grads[name])
        # A second backward on the same block reuses the same admitted weights.
        again = block.backward(x, g)
        row["repeat_equal"] = all(_digest(np, again[n]) == row["grad:" + n] for n in grads)
        ops[label + ".backward"] = row
    return ops


def _refusal(fn):
    try:
        fn()
    except BaseException as exc:  # test fixture records every refusal type
        return {"type": type(exc).__name__, "message": str(exc)[:300]}
    return {"type": None, "message": "no refusal"}


def _refusals(np, ml):
    out = {}
    for kind in ("mamba1", "mamba2", "mamba3"):
        weights = _weights(np, kind, D, 7)
        names = list(_shapes(kind, D))
        x, g = _inputs(np, D, 11)

        def mutated(edit, kind=kind, weights=weights):
            block = _block(ml, kind, weights)
            edit(block)
            return block

        def set_w(index, value):
            def edit(block):
                w = list(block.__dict__["_w"])
                w[index] = value
                block.__dict__["_w"] = w
            return edit

        def wrong_shape(block, names=names, weights=weights):
            k = names.index("out_proj.weight")
            set_w(k, np.zeros((D + 1, 2 * D), dtype=np.float32))(block)

        def wrong_rank(block, names=names):
            set_w(names.index("D"), np.zeros((2, 2, 2), dtype=np.float32))(block)

        def wrong_dtype(block, names=names, weights=weights):
            set_w(1, weights[names[1]].astype(np.float64))(block)

        def none_weight(block):
            set_w(0, None)(block)

        def short_list(block):
            block.__dict__["_w"] = list(block.__dict__["_w"])[:-1]

        def fortran_weight(block, names=names, weights=weights):
            set_w(1, np.asfortranarray(weights[names[1]]))(block)

        def bad_dmodel(block):
            block.__dict__["d_model"] = 48

        cases = {
            "wrong_shape": lambda: mutated(wrong_shape).backward(x, g),
            "wrong_rank": lambda: mutated(wrong_rank).backward(x, g),
            "wrong_dtype": lambda: mutated(wrong_dtype).backward(x, g),
            "none_weight": lambda: mutated(none_weight).backward(x, g),
            "short_weight_list": lambda: mutated(short_list).backward(x, g),
            "nonfinite_grad": lambda: _block(ml, kind, weights).backward(
                x, np.full_like(g, np.inf)),
            "grad_shape": lambda: _block(ml, kind, weights).backward(x, g[:, :-1]),
            "bad_mode": lambda: _block(ml, kind, weights, numeric_mode="fast").backward(x, g),
            "bad_d_model": lambda: mutated(bad_dmodel).backward(
                np.zeros((B, L, 48), dtype=np.float32), np.zeros((B, L, 48), dtype=np.float32)),
        }
        if kind == "mamba2":
            def bad_limit(block):
                block.__dict__["dt_limit"] = ("low", "high")

            def short_limit(block):
                block.__dict__["dt_limit"] = (0.0,)
            cases["bad_dt_limit"] = lambda: mutated(bad_limit).backward(x, g)
            cases["short_dt_limit"] = lambda: mutated(short_limit).backward(x, g)
        for name, fn in cases.items():  # test-only refusal cases
            out[kind + "." + name] = _refusal(fn)
        # Fortran-order weights are admitted by copy: a success, compared by digest.
        try:
            grads = mutated(fortran_weight).backward(x, g)
            out[kind + ".fortran_weight"] = {"type": None, "message": "ok",
                                             "digest": _digest(np, grads[names[1]])}
        except BaseException as exc:  # test fixture records every refusal type
            out[kind + ".fortran_weight"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
    return out


def run(path):
    import numpy as np
    import mojolearn as ml
    result = {"sizes": {"B": B, "L": L, "d_model": D}, "same_bits": _same_bits(np, ml),
              "refusals": _refusals(np, ml)}
    with open(path, "w") as fh:
        json.dump(result, fh, indent=1, sort_keys=True)
    print("wrote", path, "ops", len(result["same_bits"]), "refusals", len(result["refusals"]))


def compare(before_path, after_path):
    before = json.load(open(before_path))
    after = json.load(open(after_path))
    bad = 0
    for section in ("same_bits", "refusals"):
        keys = sorted(set(before[section]) | set(after[section]))
        for key in keys:  # test-only comparison of two recorded fixture runs
            same = before[section].get(key) == after[section].get(key)
            bad += not same
            print(section, key, "SAME" if same else "DIFFER")
    print("differences", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        sys.exit(compare(*a.compare))
    run(a.out)
