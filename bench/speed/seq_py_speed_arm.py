#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""OUR arm of the sequence-model speed lanes, through the PUBLIC PYTHON API.

    MOJOLEARN_NUMERIC_MODE=identical \\
      python3 bench/speed/seq_py_speed_arm.py --lane mamba2 --rounds 10
    python3 bench/speed/seq_py_speed_arm.py --lane transformer --size smoke
    python3 bench/speed/seq_py_speed_arm.py --lane mamba1 --rows large

NOTHING IN THIS FILE HAS RUN. Written 2026-09-07 for the NVIDIA H100
identity-cost grid by an agent forbidden to execute anything but
`py_compile`; the first run is a BUILD, not a benchmark.

WHAT THIS IS (DEVIATION 2100)
-----------------------------
`bench/speed/seq_speed_main.mojo` drives the Mojo entry points of the
Mamba-1 and Llama blocks and has NO Mamba-2 or Mamba-3 lane. This file
drives all four blocks the way an outside user reaches them --
`mojolearn.mamba.Mamba1Block / Mamba2Block / Mamba3Block` and
`mojolearn.transformer.TransformerBlock` (`python/mojolearn/_mamba_impl.py`,
`_transformer_impl.py`), the bindings `_mojolearn_mamba` and
`_mojolearn_transformer` -- and prints the FSPEED contract
(`tools/fast_speed_table.py`) so its lines sit in one table beside the
Mojo driver's and beside `tools/speed_torch_seq.py`'s. Lanes: `mamba1`,
`mamba2`, `mamba3`, `transformer`.

THE MODE IS NOT CHOSEN HERE. It is whatever `MOJOLEARN_NUMERIC_MODE`
selected when `mojolearn` was imported, printed on the header exactly as
`bench/speed/forest_speed_arm.py` prints it (`spec.numeric_mode_label`,
DEVIATION 1896), and then READ BACK from the binary the block actually
holds (`numeric_mode_used()`, `vendor_used()`, `_mode.py`); a read-back
that disagrees with the header refuses the lane rather than print one
number under the wrong tier. The grid runs this under `identical`.

THE ROWS, AND WHY THE TAGS LOOK LIKE THE DRIVER'S (DEVIATION 2104)
-------------------------------------------------------------------
The shape table is `tools/speed_torch_seq.py`'s, which is the driver's
`seq_shape_*` ladder PARSED (never copied) plus that file's Python-side
`py_rows()`: the grid's two LARGE rows on seed-7 bytes,
`narrow.b8_l4096_d512` (B 8, L 4096, d_model 512) and
`wide.b8_l1024_d2048` (B 8, L 1024, d_model 2048), the same two tags on
all four lanes. For `mamba1` and `transformer` the `shipped` set is the
driver's rows (same tags, same `seq_seed` bytes, same witnesses, so a
number here can be cross-checked against the Mojo driver's) PLUS the two
large rows; `--rows large` restricts to the two. For `mamba2` and
`mamba3` the `shipped` set IS the two large rows and `smoke` is the
corpus shape `lane.b2_l4_d32`. The driver's rows are not renamed and not
touched. A row outside what the API admits is REFUSED by name
(`check_bounds`), never shrunk.

THE LANE TOKEN (DEVIATION 2103). `--lane mamba1` is the leg's spelling
and prints `lane=mamba` on every line, because the torch opponent the leg
runs for it is `speed_torch_seq.py --lane mamba` and
`tools/fast_speed_table.py` pairs ours against theirs by the (lane, shape)
on the LINES, not by the file name. `--lane mamba` is accepted as the same
thing. The dump for FSPEED-AGREE is written under that token too. If a leg
ALSO runs the Mojo driver's `mamba` lane into the same table, the two
`ours` implementations share a bucket on the driver's tags; that is the
leg's call to make and this file says so here rather than rename anything.

WHAT ONE ROUND IS, AND WHAT IT INCLUDES (DEVIATION 2111)
---------------------------------------------------------
One `forward` call of the block, timed on the host with `perf_counter`,
from the numpy arrays a user would hold to the numpy array a user gets
back. The bindings return HOST arrays and synchronize before they return
(`bindings/_mojolearn_mamba.mojo`: "every device buffer below is still
alive when `mamba_block_forward` returns because that function
synchronizes before it does, and the explicit transfers at the end hold
them past the last download"), so the return IS the synchronization, the
same proof `forest_speed_arm.py::_our_sync` names. Inside that call, on
every round: the dtype/shape checks, the weights read out of numpy into
host lists and uploaded, the refusal scan that downloads every weight and
scans it for non-finite values (`mamba_refuse_bad_inputs` and siblings,
contract section 6; on the transformer the weights are refused once at
upload, DEVIATION 1875), the arithmetic, and the download of the output
and the state. All of it stays INSIDE the timer, for DEVIATION 1840's
reason: a user of this surface pays it, and a benchmark that deletes a
cost the user cannot delete measures a library that does not exist. The
sub-lane numbers in the Mojo driver are where the kernels alone are
priced. State is zeroed per call by passing no state (a fresh zero state
is allocated inside the call, the surface's own `state=None` path), except
the transformer decode row, where the prior context's KV cache is built
ONCE by running the block on the context tokens and RESTORED before every
round, outside the timer (DEVIATION 2112, the driver's DEVIATION 1853
applied to `TransformerState`).

THE HASH (DEVIATION 2110). `hash=` is `spec.hash_predictions`'s sha256
over dtype + shape + the exact output bytes, truncated to 16 hex digits --
the template's recipe -- and NOT the driver's byte-at-a-time FNV-1a64 over
the whole output, which is a Python loop over 134 MB at the narrow row.
The strided FNV-1a64 witness of the output (`speed_torch_seq.witness_hash`,
the driver's `_witness_hash`) is printed beside it as a note so the three
files still share one fingerprint spelling. Under `identical` the column
is where a reader checks that the promise held; it is recorded, never
enforced.

THE WEIGHTS ARE THE SAME BYTES ON EVERY SIDE. Generated by
`speed_torch_seq.hashed_tensor_numpy`, the pure-numpy copy of the hash
spec (DEVIATION 2105) that the torch process cross-checks bitwise against
the corpus primitives at every start, with the corpus tensor ids and
ranges of each family (`speed_torch_seq.m1_ranges` etc., cross-checked
there against the corpus's own functions). Every tensor's witness is
printed as `FSPEED-WEIGHTS`, the same line the driver and the opponent
print. This file imports NO torch and NO corpus: `speed_torch_seq` loads
its corpus lazily from its own `main` (DEVIATION 2102) precisely so this
import is safe in the process that holds MAX's runtime.
"""

import argparse
import os
import sys
import time

import numpy as np

# `tools/` is not a package; the spec and the shape table are imported by
# path, and `python/` goes on the path so an in-repo `mojolearn` is
# importable the way `forest_speed_arm.py` arranges it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (os.path.join(_ROOT, "tools"), os.path.join(_ROOT, "python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import speed_gbdt_arm as spec           # noqa: E402  header helpers, the hash
import speed_torch_seq as seqspec       # noqa: E402  shapes, rows, witness, generator

FAMILY = "seq"
ARM = "ours"

#: CLI lane -> (FSPEED lane token, py_rows kind). DEVIATION 2103.
LANES = {
    "mamba1": ("mamba", "mamba1"),
    "mamba": ("mamba", "mamba1"),
    "mamba2": ("mamba2", "mamba2"),
    "mamba3": ("mamba3", "mamba3"),
    "transformer": ("transformer", "llama"),
}

OUR_ENTRY_POINTS = {
    "mamba1": "mojolearn.mamba.Mamba1Block(weights).forward(x) -> mamba/ via _mojolearn_mamba",
    "mamba2": "mojolearn.mamba.Mamba2Block(weights).forward(x) -> mamba/ via _mojolearn_mamba",
    "mamba3": "mojolearn.mamba.Mamba3Block(weights).forward(x) -> mamba/ via _mojolearn_mamba",
    "llama": "mojolearn.transformer.TransformerBlock(weights, n_heads=, n_kv_heads=, "
             "head_dim=).forward(x[, state]) -> transformer/ via _mojolearn_transformer",
}


def gen(seed, tid, n, lo, hi):
    return seqspec.hashed_tensor_numpy(seed, tid, n, lo, hi)


def emit_header(lane, dev, n_rounds, size):
    print("FSPEED-HEADER family=%s lane=%s arm=%s mode=%s device=%s rounds=%d size=%s"
          % (FAMILY, lane, ARM, spec.numeric_mode_label(), dev, n_rounds, size))
    sys.stdout.flush()


def emit_warmup(lane, tag, ms):
    """Six decimals, the seq family's precision (the driver's `Ledger` and
    `speed_torch_seq.time_arm` both print `%.6f`); the forest spec's three
    would round a sub-millisecond smoke row to nothing."""
    print("FSPEED-WARMUP lane=%s arm=%s shape=%s ms=%.6f" % (lane, ARM, tag, ms))
    sys.stdout.flush()


def emit_round(lane, tag, index, ms, digest):
    print("FSPEED lane=%s arm=%s shape=%s round=%d ms=%.6f hash=%s"
          % (lane, ARM, tag, index, ms, digest or "-"))
    sys.stdout.flush()


def refuse(lane, reason):
    spec.emit_refused(lane, ARM, reason)


def note(lane, text):
    print("FSPEED-NOTE lane=%s arm=%s %s" % (lane, ARM, " ".join(str(text).split())))
    sys.stdout.flush()


def device_string():
    """The driver's rule: `MOJOLEARN_SPEED_DEVICE` from the leg first (the
    vendor's own tool, spaces squeezed), the spec's nvidia-smi probe else."""
    d = os.environ.get("MOJOLEARN_SPEED_DEVICE", "").strip()
    return d.replace(" ", "_") if d else spec.device_string()


# --------------------------------------------------------------------------
# Weights and inputs, per family, in the public API's key set.
# --------------------------------------------------------------------------
def build_llama(row, lane, tag, samples):
    """The nine weights under `TransformerBlock._W_NAMES`, from
    `speed_torch_seq.llama_spec` (the driver's ranges), witnessed under the
    driver's tensor names; `x`, and `ctx.x` on the decode row."""
    api_name = {"norm1.weight": "input_layernorm.weight",
                "norm2.weight": "post_attention_layernorm.weight"}
    seed = row["seed"]
    W = {}
    for name, n, lo, hi, shape in seqspec.llama_spec(row):
        flat = gen(seed, seqspec.LLAMA_TIDS[name], n, lo, hi)
        seqspec.emit_witness(lane, tag, name, flat, samples)
        W[api_name.get(name, name)] = flat.reshape(shape)
    b, l, dm = row["b"], row["l"], row["d_model"]
    x = gen(seed, seqspec.LLAMA_TIDS["x"], b * l * dm, -2.0, 2.0)
    seqspec.emit_witness(lane, tag, "x", x, samples)
    xc = None
    if row["ctx"] > 0:
        cx = gen(seed, seqspec.TID_CTX_X, b * row["ctx"] * dm, -2.0, 2.0)
        seqspec.emit_witness(lane, tag, "ctx.x", cx, samples)
        xc = cx.reshape(b, row["ctx"], dm)
    return W, x.reshape(b, l, dm), xc


def build_mamba(kind, row, lane, tag, samples):
    """The Mamba-1/2/3 weights under the block's `_W_NAMES`, from the
    corpus ids and ranges transcribed in `speed_torch_seq`; witnessed in
    the same order the driver and the opponent print them."""
    dm, b, l = row["d_model"], row["b"], row["l"]
    if kind == "mamba1":
        ids, shapes, ranges, names = (seqspec.M1_TENSOR_IDS, seqspec.m1_shapes(dm, b, l),
                                      seqspec.m1_ranges(dm), seqspec.M1_NAMES)
    elif kind == "mamba2":
        ids, shapes, ranges, names = (seqspec.M2_TENSOR_IDS, seqspec.m2_shapes(dm, b, l),
                                      seqspec.m2_ranges(dm), seqspec.M2_NAMES)
    else:
        ids, shapes, ranges, names = (seqspec.M3_TENSOR_IDS, seqspec.m3_shapes(dm, b, l),
                                      seqspec.m3_ranges(dm), seqspec.M3_NAMES)
    seed = row["seed"]
    W = {}
    for name in names:
        shape = shapes[name]
        lo, hi = ranges[name]
        flat = gen(seed, ids[name], int(np.prod(shape)), lo, hi)
        seqspec.emit_witness(lane, tag, name, flat, samples)
        W[name] = flat.reshape(shape)
    lo, hi = ranges["x"]
    x = gen(seed, ids["x"], b * l * dm, lo, hi)
    seqspec.emit_witness(lane, tag, "x", x, samples)
    return W, x.reshape(b, l, dm), None


def make_block(kind, W, row):
    """The public constructor. `mojolearn` is imported HERE, after the
    weights exist, so an import failure is one lane's refusal and not a
    traceback before the first line of the table."""
    import mojolearn.mamba
    import mojolearn.transformer

    if kind == "mamba1":
        return mojolearn.mamba.Mamba1Block(W)
    if kind == "mamba2":
        return mojolearn.mamba.Mamba2Block(W)
    if kind == "mamba3":
        return mojolearn.mamba.Mamba3Block(W)
    return mojolearn.transformer.TransformerBlock(
        W, n_heads=row["n_heads"], n_kv_heads=row["n_kv"], head_dim=row["head_dim"])


# --------------------------------------------------------------------------
# One row.
# --------------------------------------------------------------------------
def run_row(kind, lane, row, args, samples):
    tag = row["name"]
    bad = seqspec.check_bounds(kind, row)
    if bad:
        refuse(lane, "shape %s refused, not shrunk: %s" % (tag, bad))
        return
    if kind == "llama":
        W, x, xc = build_llama(row, lane, tag, samples)
    else:
        W, x, xc = build_mamba(kind, row, lane, tag, samples)

    try:
        blk = make_block(kind, W, row)
    except Exception as exc:                        # noqa: BLE001
        refuse(lane, "%s: %s" % (exc.__class__.__name__, " ".join(str(exc).split())[:200]))
        return

    # THE TIER, READ BACK FROM THE BINARY THE BLOCK HOLDS, against the label
    # the header printed. `numeric_mode_used()` is the compile-time answer of
    # the loaded .so, not the string in the environment.
    try:
        used = str(blk.numeric_mode_used())
        vendor = str(blk.vendor_used())
    except Exception as exc:                        # noqa: BLE001
        refuse(lane, "mode read-back failed: %s: %s" % (exc.__class__.__name__, str(exc)[:160]))
        return
    note(lane, "binary read-back shape=%s numeric_mode_used=%s vendor_used=%s entry=%s"
         % (tag, used, vendor, OUR_ENTRY_POINTS[kind]))
    if used.upper() != spec.numeric_mode_label():
        refuse(lane, "header says mode=%s but the loaded binary reports %s at shape %s; "
                     "nothing is timed under a label the binary disagrees with"
               % (spec.numeric_mode_label(), used, tag))
        return

    # THE STATE. Fresh zeros per call (state=None) everywhere except the
    # transformer decode row, whose prior context is computed ONCE and
    # restored before every round (DEVIATION 2112).
    state = None
    snapshot = None
    if kind == "llama" and row["ctx"] > 0:
        b = row["b"]
        state = blk.allocate_state(b, row["ctx"] + row["l"])
        try:
            blk.forward(xc, state)                  # untimed: fills the cache
        except Exception as exc:                    # noqa: BLE001
            refuse(lane, "context prefill at shape %s: %s: %s"
                   % (tag, exc.__class__.__name__, str(exc)[:160]))
            return
        snapshot = (state.k_cache.copy(), state.v_cache.copy(), int(state.cached_tokens))

    def restore():
        if snapshot is not None:
            np.copyto(state.k_cache, snapshot[0], casting="no")
            np.copyto(state.v_cache, snapshot[1], casting="no")
            state.cached_tokens = snapshot[2]

    def call():
        return blk.forward(x, state) if state is not None else blk.forward(x)

    note(lane, "one round at shape %s = one public forward() including the weight "
               "host-list copy + upload, the non-finite refusal scan and the output "
               "download; the return is the synchronization (DEVIATION 2111)" % tag)

    # One untimed-in-the-table warm-up, printed, then N timed rounds.
    try:
        restore()
        t0 = time.perf_counter()
        y = call()
        emit_warmup(lane, tag, (time.perf_counter() - t0) * 1000.0)
    except Exception as exc:                        # noqa: BLE001
        refuse(lane, "warm-up at shape %s: %s: %s"
               % (tag, exc.__class__.__name__, " ".join(str(exc).split())[:200]))
        return

    hashes = []
    last = None
    for i in range(1, args.rounds + 1):
        restore()
        try:
            t0 = time.perf_counter()
            y = call()
            ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:                    # noqa: BLE001
            refuse(lane, "round %d at shape %s: %s: %s"
                   % (i, tag, exc.__class__.__name__, " ".join(str(exc).split())[:200]))
            return
        last = np.ascontiguousarray(y, dtype=np.float32)
        digest = spec.hash_predictions(last)        # DEVIATION 2110
        hashes.append(digest)
        emit_round(lane, tag, i, ms, digest)
    if len(set(hashes)) > 1:
        print("FSPEED-NOTE lane=%s arm=%s hash moved across rounds: %s %s"
              % (lane, ARM, hashes[0], next(h for h in hashes if h != hashes[0])))
    if last is not None:
        flat = last.reshape(-1)
        note(lane, "output witness shape=%s n=%d hash=%s"
             % (tag, flat.size, seqspec.hex16(seqspec.witness_hash(flat, samples))))
        dump_dir = os.environ.get("MOJOLEARN_SPEED_DUMP_DIR", "")
        if dump_dir:
            path = os.path.join(dump_dir, "seq.%s.%s.f32.bin" % (lane, tag))
            try:
                os.makedirs(dump_dir, exist_ok=True)
                flat.tofile(path)
                note(lane, "dump shape=%s n=%d path=%s" % (tag, flat.size, path))
            except OSError as exc:
                refuse(lane, "dump at shape %s could not be written: %s" % (tag, exc))
        else:
            spec.emit_refused(lane, "agree",
                              "MOJOLEARN_SPEED_DUMP_DIR unset, so no output was dumped "
                              "and no FSPEED-AGREE can be computed for shape %s" % tag)
    sys.stdout.flush()


# --------------------------------------------------------------------------
# CLI.
# --------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        prog="seq_py_speed_arm",
        description="mojolearn's Mamba-1/2/3 and transformer blocks through the "
                    "public Python API, in the numeric mode MOJOLEARN_NUMERIC_MODE "
                    "selected at import (printed on the header, read back from the "
                    "binary), one lane per process, FSPEED lines",
    )
    p.add_argument("--lane", required=True, choices=tuple(LANES))
    p.add_argument("--rounds", type=int,
                   default=int(os.environ.get("MOJOLEARN_SPEED_ROUNDS", "10")))
    p.add_argument("--size", default=os.environ.get("MOJOLEARN_SPEED_SIZE", "shipped"),
                   choices=("shipped", "smoke"))
    p.add_argument("--rows", default="all", choices=("all", "large", "driver"),
                   help="mamba1/transformer only: all (the driver's rows plus the two "
                        "large seed-7 rows, the default), large (the two large rows), "
                        "driver (the driver's rows). mamba2/mamba3 have only the "
                        "Python-side rows.")
    p.add_argument("--list-rows", action="store_true",
                   help="print the rows the lane would run and exit")
    return p


def rows_for(kind, args):
    """The lane's rows: the driver's (parsed, `seq_seed` bytes) for the two
    families the driver has, plus/or the Python-side rows (seed 7)."""
    py = seqspec.py_rows(kind)
    if kind in ("mamba2", "mamba3"):
        rows = py
    else:
        driver, consts = seqspec.load_shapes()
        fam = seqspec.FAM_LLAMA if kind == "llama" else seqspec.FAM_MAMBA
        driver = [r for r in driver if r["family"] == fam]
        for r in driver:
            r["seed"] = seqspec.seq_seed(r["i"], consts["seed_base"])
            r["kind"] = kind
        rows = {"all": driver + py, "large": py, "driver": driver}[args.rows]
    if args.size == "smoke":
        rows = [r for r in rows if r["smoke"] == 1]
    return rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.rounds < 1:
        raise SystemExit("seq_py_speed_arm: --rounds must be >= 1")
    lane, kind = LANES[args.lane]
    _, consts = seqspec.load_shapes()
    samples = consts["witness_samples"]
    rows = rows_for(kind, args)

    if args.list_rows:
        for r in rows:
            print("%s b=%d l=%d ctx=%d d_model=%d n_heads=%d n_kv=%d head_dim=%d "
                  "intermediate=%d smoke=%d seed=%#x"
                  % (r["name"], r["b"], r["l"], r["ctx"], r["d_model"], r["n_heads"],
                     r["n_kv"], r["head_dim"], r["intermediate"], r["smoke"], r["seed"]))
        return 0

    emit_header(lane, device_string(), args.rounds, args.size)
    note(lane, "entry=%s; our arm through the public Python API (DEVIATION 2100); "
               "cli_lane=%s rows=%s" % (OUR_ENTRY_POINTS[kind], args.lane, args.rows))
    if not rows:
        refuse(lane, "no rows selected for size=%s rows=%s" % (args.size, args.rows))
        print("FSPEED-DONE lane=%s arm=%s mode=%s" % (lane, ARM, spec.numeric_mode_label()))
        return 1
    for row in rows:
        # ONE ROW MAY NOT TAKE THE RUN DOWN; the lease is an hour.
        try:
            run_row(kind, lane, row, args, samples)
        except Exception as exc:                    # noqa: BLE001
            refuse(lane, "row %s: %s: %s"
                   % (row["name"], exc.__class__.__name__, " ".join(str(exc).split())[:200]))
    print("FSPEED-DONE lane=%s arm=%s mode=%s" % (lane, ARM, spec.numeric_mode_label()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
