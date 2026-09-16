#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A step-by-step decode is the one-shot prefill, BITWISE, position by position
(lane/stateful-cpu-decoding, 2026-09-16).

This is the check the decode surface exists for. `TransformerBlockInference`,
`Mamba1BlockInference`, `Mamba2BlockInference`, `Mamba3BlockInference` and
`SambaInference` each run a sequence twice on the CPU host route:

  (a) ONE fresh-state forward pass over the whole sequence, `state=None`;
  (b) the SAME sequence decoded one token at a time through `allocate_state`
      and `step`, carrying the state.

and (a) and (b) are compared as bits, position by position. A model that
decodes incrementally and does not answer (a) is a different model wearing
the same name, so a mismatch prints the FIRST differing position with both
values and their bit patterns, never a count.

    python3 tools/step_vs_full_check.py
    python3 tools/step_vs_full_check.py --sabotage --l 12

`--sabotage` is the arm that proves the comparison can fail: before one step
it moves ONE carried cell by ONE ULP and requires the per-position check to
fire. A cell is scanned for because a single ULP in a single cached
component is often absorbed by the rounding downstream of it (measured: the
transformer's k_cache absorbs it at every one of the first thirty-two cells
while v_cache[7] does not), and an arm that silently absorbs its own
perturbation is indistinguishable from a pass.

It needs the shipped neural host binding
(`sh bindings/build_neural_host.sh`), and nothing else: no GPU, no reference
binding.
"""
import argparse
import sys

import numpy as np

import mojolearn as ml

_T_NAMES = ("input_layernorm.weight", "post_attention_layernorm.weight",
            "q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
            "gate_proj.weight", "up_proj.weight", "down_proj.weight")


def _block_weights(dm, nh=2, nkv=1, it=64, seed=1):
    hd = dm // nh
    shapes = ((dm,), (dm,), (nh * hd, dm), (nkv * hd, dm), (nkv * hd, dm),
              (dm, nh * hd), (it, dm), (it, dm), (dm, it))
    rng = np.random.default_rng(seed)
    return {n: (rng.standard_normal(s) * 0.1).astype(np.float32)
            for n, s in zip(_T_NAMES, shapes)}


def _mamba_weights(kind, dm, seed=21):
    di = 2 * dm
    rng = np.random.default_rng(seed)
    if kind == "mamba1":
        r = -(-dm // 16)
        shapes = {"norm.weight": (dm,), "in_proj.weight": (2 * di, dm),
                  "conv1d.weight": (di, 1, 4), "conv1d.bias": (di,),
                  "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r),
                  "dt_proj.bias": (di,), "A_log": (di, 16), "D": (di,),
                  "out_proj.weight": (dm, di)}
        ones = ("norm.weight",)
    elif kind == "mamba2":
        nh = di // 64
        cd, dip = di + 256, 2 * di + 256 + nh
        shapes = {"block_norm.weight": (dm,), "in_proj.weight": (dip, dm),
                  "conv1d.weight": (cd, 1, 4), "conv1d.bias": (cd,),
                  "dt_bias": (nh,), "A_log": (nh,), "D": (nh,),
                  "norm.weight": (di,), "out_proj.weight": (dm, di)}
        ones = ("block_norm.weight", "norm.weight")
    else:
        nh = di // 64
        shapes = {"block_norm.weight": (dm,),
                  "in_proj.weight": (2 * di + 256 + 3 * nh + 32, dm),
                  "dt_bias": (nh,), "B_norm.weight": (128,), "C_norm.weight": (128,),
                  "B_bias": (nh, 128), "C_bias": (nh, 128), "D": (nh,),
                  "out_proj.weight": (dm, di)}
        ones = ("block_norm.weight", "B_norm.weight", "C_norm.weight")
    return {n: (np.ones(s, np.float32) if n in ones
                else (rng.standard_normal(s) * 0.1).astype(np.float32))
            for n, s in shapes.items()}


def first_difference(full, stepped):
    """The first position at which the two disagree as bits, with the cell
    and both values, or None. PER POSITION, never a count."""
    full = np.ascontiguousarray(np.asarray(full, dtype=np.float32))
    stepped = np.ascontiguousarray(np.asarray(stepped, dtype=np.float32))
    if full.shape != stepped.shape:
        return ("shape", full.shape, stepped.shape)
    fb, sb = full.view(np.uint32), stepped.view(np.uint32)
    for t in range(full.shape[1]):
        if fb[:, t, :].tobytes() == sb[:, t, :].tobytes():
            continue
        row, unit = (int(v) for v in np.argwhere(fb[:, t, :] != sb[:, t, :])[0])
        return (t, row, unit, float(full[row, t, unit]), int(fb[row, t, unit]),
                float(stepped[row, t, unit]), int(sb[row, t, unit]))
    return None


def report(name, diff, positions):
    if diff is None:
        print(f"{name}: BITWISE EQUAL at every one of {positions} positions")
        return True
    if diff[0] == "shape":
        print(f"{name}: SHAPE MISMATCH full={diff[1]} stepped={diff[2]}")
        return False
    t, row, unit, fv, fbits, sv, sbits = diff
    print(f"{name}: FIRST DIFFERING POSITION t={t} (row {row}, unit {unit}) "
          f"full={fv!r} 0x{fbits:08x}  stepped={sv!r} 0x{sbits:08x}")
    return False


def _ulp(buf, index):
    """Move one cell of a carried state buffer up by one ULP, in place.
    Returns the bit pattern before and after, so a perturbation that did not
    land is distinguishable from one that landed and was absorbed."""
    view = np.asarray(buf).reshape(-1).view(np.uint32)
    before = int(view[index])
    view[index] = before + 1
    return before, int(np.asarray(buf).reshape(-1).view(np.uint32)[index])


class _Case:
    """One model's two runs. `pieces` names the carried buffers the sabotage
    arm may perturb, so each model is perturbed in its OWN state."""

    def __init__(self, name, make, x, pieces, alloc, run_full):
        self.name = name
        self.make = make
        self.x = x
        self.pieces = pieces
        self.alloc = alloc
        self.run_full = run_full

    def run(self, perturb=None):
        model = self.make()
        full = np.asarray(self.run_full(model, self.x))
        state = self.alloc(model)
        out = np.empty_like(full)
        landed = None
        for t in range(full.shape[1]):
            if perturb is not None and t == perturb[0]:
                landed = _ulp(self.pieces(state)[perturb[1]], perturb[2])
            out[:, t:t + 1, :] = self.step(model, t, state)
        return full, out, landed

    def step(self, model, t, state):
        raise NotImplementedError


class _BlockCase(_Case):
    def step(self, model, t, state):
        return np.asarray(model.step(np.ascontiguousarray(self.x[:, t:t + 1, :]), state))


class _SambaCase(_Case):
    def step(self, model, t, state):
        return np.asarray(model.step(np.ascontiguousarray(self.x[:, t:t + 1]), state)) \
            .reshape((self.x.shape[0], 1, -1))


def build_cases(b, l, dm):
    x = np.random.default_rng(3).standard_normal((b, l, dm)).astype(np.float32)
    cases = []
    for window in (0, 8):
        cases.append(_BlockCase(
            f"transformer(window={window})",
            lambda w=window: ml.TransformerBlockInference(
                _block_weights(dm), n_heads=2, n_kv_heads=1, window=w),
            x,
            lambda st: [st.k_cache, st.v_cache],
            lambda m: m.allocate_state(b, l),
            lambda m, xx: m.forward(np.ascontiguousarray(xx)),
        ))
    for kind, cls in (("mamba1", ml.Mamba1BlockInference),
                      ("mamba2", ml.Mamba2BlockInference),
                      ("mamba3", ml.Mamba3BlockInference)):
        cases.append(_BlockCase(
            kind,
            lambda k=kind, c=cls: c(_mamba_weights(k, dm)),
            x,
            lambda st: [st.conv_window if hasattr(st, "conv_window") else st.theta, st.h],
            lambda m: m.allocate_state(b),
            lambda m, xx: m.forward(np.ascontiguousarray(xx)),
        ))
    cfg = ml.SambaConfig(vocab=64, d_model=dm, layers=("mamba3", "attention"),
                         n_heads=2, intermediate=2 * dm, tie_embeddings=True)
    rng = np.random.default_rng(5)
    sw = {n: (rng.standard_normal(s) * 0.05).astype(np.float32) for n, s in cfg.registry()}
    for n in sw:
        if n.endswith("norm.weight") or n.endswith("norm_f.weight") or n.endswith("_norm.weight"):
            sw[n] = np.ones(sw[n].shape, np.float32)
    ids = np.random.default_rng(7).integers(0, 64, (b, l)).astype(np.int32)
    cases.append(_SambaCase(
        "samba",
        lambda: ml.SambaInference(cfg, sw),
        ids,
        lambda st: [st.layers[0].h, st.layers[1].v_cache],
        lambda m: m.allocate_state(b, l),
        lambda m, xx: m.forward(np.ascontiguousarray(xx)),
    ))
    return cases


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--b", type=int, default=2)
    ap.add_argument("--l", type=int, default=16)
    ap.add_argument("--d-model", type=int, default=32)
    ap.add_argument("--sabotage", action="store_true",
                    help="move ONE carried cell by ONE ULP and require the check to fire")
    ap.add_argument("--cells", type=int, default=128,
                    help="how many cells of each carried piece the sabotage arm may try")
    args = ap.parse_args(argv)
    cases = build_cases(args.b, args.l, args.d_model)
    at = max(1, args.l // 2)
    ok = True
    for case in cases:
        if not args.sabotage:
            full, out, _ = case.run()
            ok &= report(case.name, first_difference(full, out), full.shape[1])
            continue
        fired = False
        n_pieces = len(case.pieces(case.alloc(case.make())))
        for piece in range(n_pieces):
            for cell in range(args.cells):
                full, out, landed = case.run((at, piece, cell))
                if landed is None or landed[0] == landed[1]:
                    print(f"{case.name}: the perturbation of piece {piece} cell {cell} "
                          "DID NOT LAND")
                    continue
                diff = first_difference(full, out)
                if diff is not None:
                    print(f"{case.name}: piece {piece} cell {cell} 0x{landed[0]:08x} -> "
                          f"0x{landed[1]:08x} (+1 ULP) at step {at}")
                    report(case.name + " [sabotaged]", diff, full.shape[1])
                    fired = True
                    break
            if fired:
                break
        if not fired:
            print(f"{case.name}: NO one-ULP perturbation of {n_pieces} carried pieces "
                  f"({args.cells} cells each) changed any position; this arm cannot fail "
                  "and is not evidence")
            ok = False
    print("PASS step_vs_full" if ok else "FAIL step_vs_full")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
