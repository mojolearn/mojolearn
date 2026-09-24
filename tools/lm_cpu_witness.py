#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The CPU witness column of the GPT-3 Small run (docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md,
section 6, items 5 and 6): replay recorded GPU arithmetic from a checkpoint with
the HOST byte LM binding (`_mojolearn_byte_lm_host`) and compare float32 bits
and sha256 digests to the run's chain.

    python3 tools/lm_cpu_witness.py plan     --recipe R --manifest M --step 101 --shard 0 [--heldout-index 0]
    python3 tools/lm_cpu_witness.py loss     --recipe R --manifest M --tokens SRC --checkpoint CKPT \\
                                             --chain CHAIN --step 101 --shard 0 --out loss.json \\
                                             [--threaded] [--perturb-token] [--perturb-param-bit INDEX]
    python3 tools/lm_cpu_witness.py heldout  --recipe R --manifest M --tokens SRC --checkpoint CKPT \\
                                             [--chain CHAIN] --out heldout.json [--heldout-index 0] [--threaded]
    python3 tools/lm_cpu_witness.py gradient --recipe R --manifest M --tokens SRC --checkpoint CKPT \\
                                             --chain CHAIN --step 101 --out gradient.json \\
                                             [--shards 0:64] [--deadline-seconds S]

`--tokens` is a directory holding `tokens.i32` (or its `tokens.i32.partNN`
files), or a JSON file {"tokens.i32.part00": URL, ...} of presigned GET URLs,
from which ONLY the byte ranges a check needs are fetched (HTTP Range).
`--manifest` is the stream's `manifest.json`; its sha256 must be the recipe's
`tokens_manifest_sha256`.

WHAT EACH CHECK COMPARES.
  loss      the mean cross entropy of shard s of global step N+1 (the ids
            `TokenBatches.ids((N)*K + s)` under the recipe's schedule
            `lm-segment.shards.v1`), computed on the host from checkpoint N's
            parameters, against `losses_f32_hex[s]` of chain line N+1. Equal
            float32 bits or FAIL.
  gradient  the full optimizer step's summed gradient: every shard's
            gradient from `byte_lm_host_train_step`, combined by the device's
            ordered left fold (total = g0, then ftz(ftz(total) + ftz(g_k))),
            hashed under the recipe's scheme and compared to the chain line's
            `gradient_sha256`. The host step's loss for each shard is compared
            to `losses_f32_hex[s]` as it goes. If the deadline ends the replay
            before all K shards, the record says PARTIAL: the per-shard losses
            were compared, the fold prefix and every shard gradient's hash are
            recorded for a later GPU one-shard replay, and the summed gradient
            was NOT compared. The chain records no per-shard gradient hash, so
            nothing short of the whole fold can be compared to it.
  heldout   the loss of one held-out batch at the checkpoint, recorded with
            its float32 bits so every vendor can be compared to it later.

THE HELD-OUT BATCH. The plan's "shard 013" is FineWeb-Edu sample-10BT parquet
shard 013, which `tools/fineweb_tokens.py --held-out-shard` tokenized last
and recorded as the stream's `validation_range` (bench/results/
fineweb_tokens_2026-09-22). The recipe names no held-out batch, so this tool
defines held-out batch h (default 0) by the training schedule's own rule
(`TokenBatches`, train-range-modulo.v1) applied to the validation range
[vlo, vhi): row b reads ids [vlo + (h*B*L + b*L) % (vhi - vlo - L - 1) :
+ L + 1]. Batch 0 is the first B*L + 1 ids of shard 013.

EVERY RECORD ALSO CHECKS ITS INPUTS. The checkpoint's state is hashed under
the recipe's scheme and held to chain line N's `state_sha256` (so a wrong or
corrupt checkpoint cannot pass as a witness); the checkpoint's data schedule
must be the recipe's; the manifest must be the recipe's; the token ids are
hashed and recorded.

NEGATIVE CONTROLS. `--perturb-token` adds one to the first input id of row 0
(mod the vocabulary) and `--perturb-param-bit I` flips the lowest mantissa
bit of parameter I; the record is marked `control` and the check must read
FAIL. A witness that has never been seen to fail is not a witness.

The hashing is `tools/lm_segment.py`'s (`_hash_arrays`, `_hash_gradient`),
imported, not rewritten. Nothing here is imported by `mojolearn verify`.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lm_segment as seg  # noqa: E402

SCHEMA = "mojolearn.lm-cpu-witness.v1"
PART_BYTES = 2_000_000_000  # tokens.i32.partNN, bench/results/fineweb_tokens_2026-09-22
HOST_BASENAME = "_mojolearn_byte_lm_host"


# ---------------------------------------------------------------- pure helpers (unit tested)

def load_chain(path):
    """Chain lines by global step, through lm_segment's reader."""
    return seg._chain_index(path)


def recipe_dims(recipe):
    shape = recipe["shape"]
    return dict(batch=int(shape[0]), length=int(shape[1]), vocab=int(shape[8]), K=int(recipe["logical_shards"]))


def check_manifest(recipe, manifest_bytes):
    """The stream's manifest must be the recipe's; returns it parsed."""
    digest = seg._sha(manifest_bytes)
    want = recipe["data"]["tokens_manifest_sha256"]
    if digest != want:
        raise SystemExit("manifest sha256 %s is not the recipe's %s" % (digest, want))
    manifest = json.loads(manifest_bytes)
    if manifest["sha256"] != recipe["data"]["tokens_sha256"]:
        raise SystemExit("manifest names token stream %s, the recipe %s" % (manifest["sha256"], recipe["data"]["tokens_sha256"]))
    if list(manifest["train_range"]) != list(recipe["data"]["train_range"]):
        raise SystemExit("manifest train range %s is not the recipe's %s" % (manifest["train_range"], recipe["data"]["train_range"]))
    return manifest


def row_starts(index, batch, length, lo, hi):
    """`TokenBatches.ids(index)`: the first token of each of the `batch` rows,
    each row `length + 1` ids, over the range [lo, hi)."""
    modulus = hi - lo - length - 1
    if modulus <= 0:
        raise ValueError("the range is shorter than one row")
    return [lo + (index * batch * length + b * length) % modulus for b in range(batch)]


def train_shard_index(step, shard, K):
    """Chain line `step` (the step COMPLETED, 1-based) was computed from shards
    `(step - 1) * K + k` (lm_segment.SCHEDULE, `batch_index`)."""
    if step < 1 or not 0 <= shard < K:
        raise ValueError("step must be >= 1 and shard in [0, K)")
    return (step - 1) * K + shard


def train_rows(recipe, step, shard):
    d = recipe_dims(recipe)
    lo, hi = recipe["data"]["train_range"]
    return row_starts(train_shard_index(step, shard, d["K"]), d["batch"], d["length"], int(lo), int(hi))


def heldout_rows(recipe, manifest, index=0):
    d = recipe_dims(recipe)
    vr = manifest.get("validation_range")
    if not vr:
        raise SystemExit("the token manifest records no validation_range")
    return row_starts(index, d["batch"], d["length"], int(vr[0]), int(vr[1]))


def byte_segments(first_token, count, part_bytes=None):
    """[(part, byte_lo, byte_hi_exclusive)] covering tokens [first, first+count)
    of a stream split into parts of `part_bytes` bytes."""
    part_bytes = part_bytes or PART_BYTES
    lo, hi = 4 * first_token, 4 * (first_token + count)
    out = []
    while lo < hi:
        part = lo // part_bytes
        end = min(hi, (part + 1) * part_bytes)
        out.append((part, lo - part * part_bytes, end - part * part_bytes))
        lo = end
    return out


def ordered_fold_step(total, g, np):
    """One step of training/byte_lm_parallel.mojo's fold: ftz(ftz(total) + ftz(g)),
    one float32 rounding (fma(1, a, b) is an IEEE add). `total` None starts
    the fold with a copy of g (the device copies g[0] unflushed)."""
    if total is None:
        return np.array(g, dtype=np.float32, copy=True)
    with np.errstate(over="ignore"):
        return _ftz(_ftz(total, np) + _ftz(np.asarray(g, dtype=np.float32), np), np)


def _ftz(x, np):
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    sub = ((b & np.uint32(0x7F800000)) == 0) & ((b & np.uint32(0x007FFFFF)) != 0)
    return np.where(sub, b & np.uint32(0x80000000), b).view(np.float32)


def f32_hex(value):
    return "%08x" % seg._f32_bits(value)


def verdict_of(got_hex, want_hex, control):
    """PASS when the bits agree; a control must NOT agree, and a control that
    reads FAIL is recorded as EXPECTED FAIL (the witness can fail)."""
    same = got_hex == want_hex
    if control:
        return "FAIL (EXPECTED, control)" if not same else "PASS (CONTROL DID NOT FAIL: the witness is blind)"
    return "PASS" if same else "FAIL"


# ---------------------------------------------------------------- token source

class Tokens:
    """Reads token ranges from a local stream (tokens.i32 or its parts) or
    from presigned part URLs by HTTP Range."""

    def __init__(self, source):
        p = Path(source)
        self.urls = None
        self.dir = None
        self.fetched = []
        if p.is_file() and p.suffix == ".json":
            self.urls = json.loads(p.read_text())
        elif p.is_dir():
            self.dir = p
        else:
            raise SystemExit("--tokens must be a directory or a JSON of part URLs")

    def _read_part(self, part, a, b):
        name = "tokens.i32.part%02d" % part
        if self.dir is not None:
            whole = self.dir / "tokens.i32"
            path, off = (whole, part * PART_BYTES) if whole.exists() else (self.dir / name, 0)
            with open(path, "rb") as fh:
                fh.seek(off + a)
                data = fh.read(b - a)
        else:
            url = self.urls[name]
            data = None
            for attempt in range(1, 6):
                try:
                    req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (a, b - 1)})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        if r.status != 206:
                            raise OSError("HTTP %d, not a partial response" % r.status)
                        data = r.read()
                    break
                except OSError as e:
                    if attempt == 5:
                        raise
                    print("range fetch %s %d-%d failed (%s), retrying" % (name, a, b, e), flush=True)
                    time.sleep(3 * attempt)
        if len(data) != b - a:
            raise SystemExit("short read of %s [%d, %d): %d bytes" % (name, a, b, len(data)))
        self.fetched.append(dict(part=name, byte_range=[a, b]))
        return data

    def rows(self, starts, length):
        """int32 ids [len(starts), length + 1] as little-endian bytes."""
        out = bytearray()
        for s in starts:
            for part, a, b in byte_segments(s, length + 1):
                out += self._read_part(part, a, b)
        return bytes(out)


# ---------------------------------------------------------------- host binding and state

def _box():
    info = dict(host=platform.node(), machine=platform.machine(), system=platform.system(),
                python=platform.python_version(), cpus=os.cpu_count())
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                info["cpu"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return info


def host_binding():
    import mojolearn
    from mojolearn import _backend
    host = _backend.load_host_module(HOST_BASENAME)
    info = dict(package_version=getattr(mojolearn, "__version__", None), package_file=mojolearn.__file__,
                binding_file=host.__file__, binding_sha256=seg._sha_file(host.__file__),
                vendor=str(host.byte_lm_host_vendor()), numeric_mode=int(host.byte_lm_host_numeric_mode()))
    return host, info


def open_checkpoint(args, recipe, manifest, chain):
    """Load, hash under the recipe's scheme, hold to chain line N, check the
    data schedule. Returns (state, record)."""
    t0 = time.perf_counter()
    state, file_sha = seg.load_checkpoint(args.checkpoint)
    load_s = time.perf_counter() - t0
    step = int(state["completed_steps"])
    scheme = seg.hash_scheme_of(recipe)
    t1 = time.perf_counter()
    digest = seg._hash_arrays(state, scheme)
    rec = dict(file=Path(args.checkpoint).name, sha256=file_sha, step=step, state_sha256=digest,
               hash_scheme=scheme, load_seconds=round(load_s, 2), hash_seconds=round(time.perf_counter() - t1, 2))
    batches = seg.ManifestBatches(args.manifest, recipe["shape"][0], recipe["shape"][1])
    schedule = seg.data_schedule(recipe, batches)
    rec["data_schedule_is_recipe"] = seg._canonical(state["data_schedule"]) == seg._canonical(schedule)
    if not rec["data_schedule_is_recipe"]:
        raise SystemExit("REFUSED: the checkpoint's data schedule is not the recipe's")
    if chain is not None and step in chain:
        want = chain[step]["state_sha256"]
        rec["chain_state_sha256"] = want
        rec["state_matches_chain"] = want == digest
        if want != digest:
            raise SystemExit("REFUSED: checkpoint %s state %s is not chain line %d's %s"
                             % (rec["file"], digest, step, want))
    else:
        rec["state_matches_chain"] = None
    return state, rec


def _np():
    import numpy as np
    return np


def _ids_array(raw, recipe):
    np = _np()
    d = recipe_dims(recipe)
    ids = np.frombuffer(raw, dtype="<i4").reshape(d["batch"], d["length"] + 1).copy()
    if ids.min() < 0 or ids.max() >= d["vocab"]:
        raise SystemExit("token id outside [0, vocab)")
    return ids


def _params_view(state, writable=False):
    return seg._bytes_of(state["parameters"], writable=writable)


def _addr(buf):
    """Address of a numpy array or a mojolearn Array."""
    from mojolearn._buffer import addr_ro
    return addr_ro(buf, name="witness")


def _perturb(args, state, ids, rec):
    """Apply a negative control in place; record what was changed."""
    if args.perturb_token:
        old = int(ids[0, 0])
        ids[0, 0] = (old + 1) % int(args._vocab)
        rec["control"] = dict(kind="token", row=0, position=0, old=old, new=int(ids[0, 0]))
    if args.perturb_param_bit is not None:
        mv = _params_view(state, writable=True).cast("I")
        i = int(args.perturb_param_bit)
        old = mv[i]
        mv[i] = old ^ 1
        rec["control"] = dict(kind="parameter", index=i, old_bits="%08x" % old, new_bits="%08x" % mv[i])


def _host_loss(host, recipe, state, ids, threaded):
    shape = list(recipe["shape"])
    bits = host.byte_lm_host_loss([_addr(state["parameters"]), int(ids.ctypes.data)], shape, 1 if threaded else 0, 0)
    return int(bits) & 0xFFFFFFFF


def _write(path, rec):
    try:
        import resource
        rec["max_rss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1 << 20), 2)  # KiB on Linux
    except (ImportError, OSError):
        pass
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(rec, indent=1) + "\n")


def _common(args):
    recipe = seg.load_recipe(args.recipe)
    recipe_sha = seg._sha_file(args.recipe)
    manifest = check_manifest(recipe, Path(args.manifest).read_bytes())
    chain = load_chain(args.chain) if getattr(args, "chain", None) else None
    args._vocab = recipe["shape"][8]
    return recipe, recipe_sha, manifest, chain


def _say(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ---------------------------------------------------------------- commands

def cmd_plan(args):
    recipe = seg.load_recipe(args.recipe)
    manifest = check_manifest(recipe, Path(args.manifest).read_bytes())
    d = recipe_dims(recipe)
    out = dict(step=args.step, shard=args.shard, index=train_shard_index(args.step, args.shard, d["K"]),
               train_rows=train_rows(recipe, args.step, args.shard),
               heldout_index=args.heldout_index, heldout_rows=heldout_rows(recipe, manifest, args.heldout_index))
    out["train_segments"] = [byte_segments(s, d["length"] + 1) for s in out["train_rows"]]
    out["heldout_segments"] = [byte_segments(s, d["length"] + 1) for s in out["heldout_rows"]]
    print(json.dumps(out, indent=1))
    return 0


def cmd_loss(args):
    recipe, recipe_sha, manifest, chain = _common(args)
    if args.step not in chain:
        raise SystemExit("the chain has no line for step %d" % args.step)
    line = chain[args.step]
    host, binfo = host_binding()
    state, ck = open_checkpoint(args, recipe, manifest, chain)
    if ck["step"] != args.step - 1:
        raise SystemExit("REFUSED: checkpoint is at step %d; chain line %d needs step %d" % (ck["step"], args.step, args.step - 1))
    d = recipe_dims(recipe)
    tokens = Tokens(args.tokens)
    starts = train_rows(recipe, args.step, args.shard)
    ids = _ids_array(tokens.rows(starts, d["length"]), recipe)
    rec = dict(schema=SCHEMA, check="loss", step=args.step, shard=args.shard,
               shard_index=train_shard_index(args.step, args.shard, d["K"]), row_starts=starts,
               ids_sha256=seg._sha(ids.tobytes()), fetched=tokens.fetched,
               chain_batch_index=line.get("batch_index"), chain_label=line.get("label"),
               recipe_sha256=recipe_sha, checkpoint=ck, binding=binfo, box=_box(), threaded=bool(args.threaded))
    _perturb(args, state, ids, rec)
    if "control" in rec:
        rec["ids_sha256_after_control"] = seg._sha(ids.tobytes())
    want = line["losses_f32_hex"][args.shard]
    _say("loss: step %d shard %d from %s (state %s), %s path" % (args.step, args.shard, ck["file"], ck["state_sha256"][:16],
                                                               "threaded" if args.threaded else "reference"))
    t0 = time.perf_counter()
    bits = _host_loss(host, recipe, state, ids, args.threaded)
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    got = "%08x" % bits
    rec.update(want_f32_hex=want, got_f32_hex=got, want=seg._bits_f32(int(want, 16)), got=seg._bits_f32(bits),
               verdict=verdict_of(got, want, "control" in rec), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _write(args.out, rec)
    _say("%s: want %s got %s (%.6f vs %.6f) in %.1f s -> %s" % (rec["verdict"], want, got, rec["want"], rec["got"], rec["seconds"], args.out))
    return 0 if rec["verdict"].startswith("PASS") != ("control" in rec) else 1


def cmd_heldout(args):
    recipe, recipe_sha, manifest, chain = _common(args)
    host, binfo = host_binding()
    state, ck = open_checkpoint(args, recipe, manifest, chain)
    d = recipe_dims(recipe)
    tokens = Tokens(args.tokens)
    starts = heldout_rows(recipe, manifest, args.heldout_index)
    ids = _ids_array(tokens.rows(starts, d["length"]), recipe)
    rec = dict(schema=SCHEMA, check="heldout", heldout_index=args.heldout_index, row_starts=starts,
               validation_range=manifest["validation_range"],
               definition=("FineWeb-Edu shard 013 = the stream's validation_range; batch h reads rows "
                           "[vlo + (h*B*L + b*L) % (vhi - vlo - L - 1) : + L + 1], b = 0..B-1 (train-range-modulo.v1 "
                           "over the validation range)"),
               ids_sha256=seg._sha(ids.tobytes()), fetched=tokens.fetched, recipe_sha256=recipe_sha,
               checkpoint=ck, binding=binfo, box=_box(), threaded=bool(args.threaded))
    _say("heldout: batch %d at %s (state %s)" % (args.heldout_index, ck["file"], ck["state_sha256"][:16]))
    t0 = time.perf_counter()
    bits = _host_loss(host, recipe, state, ids, args.threaded)
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec.update(loss_f32_hex="%08x" % bits, loss=seg._bits_f32(bits), verdict="RECORDED",
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if args.expect_f32_hex:
        rec["expect_f32_hex"] = args.expect_f32_hex
        rec["verdict"] = "PASS" if args.expect_f32_hex == rec["loss_f32_hex"] else "FAIL"
    _write(args.out, rec)
    _say("%s: held-out loss %s (%.6f) in %.1f s -> %s" % (rec["verdict"], rec["loss_f32_hex"], rec["loss"], rec["seconds"], args.out))
    return 1 if rec["verdict"] == "FAIL" else 0


def _shard_range(text, K):
    """A:B, or a comma list of shard indices (for parallel processes that
    save their gradients and leave the fold to `fold`)."""
    if "," in text:
        out = [int(x) for x in text.split(",") if x.strip()]
        if not out or any(not 0 <= x < K for x in out) or len(set(out)) != len(out):
            raise SystemExit("--shards list must name distinct shards in [0, %d)" % K)
        return out
    a, sep, b = text.partition(":")
    a, b = (int(a), int(b)) if sep else (int(a), int(a) + 1)
    if not 0 <= a < b <= K:
        raise SystemExit("--shards A:B needs 0 <= A < B <= %d" % K)
    return list(range(a, b))


def cmd_gradient(args):
    np = _np()
    recipe, recipe_sha, manifest, chain = _common(args)
    if args.step not in chain:
        raise SystemExit("the chain has no line for step %d" % args.step)
    line = chain[args.step]
    host, binfo = host_binding()
    state, ck = open_checkpoint(args, recipe, manifest, chain)
    if ck["step"] != args.step - 1:
        raise SystemExit("REFUSED: checkpoint is at step %d; chain line %d needs step %d" % (ck["step"], args.step, args.step - 1))
    d = recipe_dims(recipe)
    K = d["K"]
    shards = _shard_range(args.shards, K)
    folds = shards == list(range(len(shards)))  # a prefix of the left fold from shard 0
    if not folds and not args.save_grads:
        raise SystemExit("shards that are not a prefix 0..j of the fold must be saved (--save-grads) for `fold`")
    scheme = seg.hash_scheme_of(recipe)
    opt = recipe["optimizer"]
    lr = seg._bits_f32(int(recipe["schedule"]["table_f32_hex"][args.step - 1], 16))
    scalars = [lr, opt["betas"][0], opt["betas"][1], opt["eps"], opt["weight_decay"]]
    n = len(_params_view(state)) // 4
    grad = np.zeros(n, dtype=np.float32)
    outs = [np.zeros(n, dtype=np.float32) for _ in range(3)]
    tokens = Tokens(args.tokens)
    rec = dict(schema=SCHEMA, check="gradient", step=args.step, K=K, hash_scheme=scheme, lr_f32_hex=f32_hex(lr),
               chain_lr_f32_hex=line["lr_f32_hex"], chain_gradient_sha256=line["gradient_sha256"],
               recipe_sha256=recipe_sha, checkpoint=ck, binding=binfo, box=_box(), shards=[],
               deadline_seconds=args.deadline_seconds)
    addrs = [_addr(state["parameters"]), _addr(state["m"]), _addr(state["v"])]
    total = None
    t_start = time.perf_counter()
    for s in shards:
        starts = train_rows(recipe, args.step, s)
        ids = _ids_array(tokens.rows(starts, d["length"]), recipe)
        _say("gradient: step %d shard %d" % (args.step, s))
        t0 = time.perf_counter()
        bits = host.byte_lm_host_train_step(
            addrs + [int(ids.ctypes.data), int(grad.ctypes.data)] + [int(o.ctypes.data) for o in outs],
            list(recipe["shape"]), scalars, ck["step"])
        sec = time.perf_counter() - t0
        got = "%08x" % (int(bits) & 0xFFFFFFFF)
        want = line["losses_f32_hex"][s]
        row = dict(shard=s, ids_sha256=seg._sha(ids.tobytes()), loss_want=want, loss_got=got,
                   loss_verdict="PASS" if got == want else "FAIL",
                   gradient_sha256=seg._hash_gradient(grad, scheme), seconds=round(sec, 1))
        if folds:
            total = ordered_fold_step(total, grad, np)
            row["prefix_sha256"] = seg._hash_gradient(total, scheme)
        if args.save_grads:
            Path(args.save_grads).mkdir(parents=True, exist_ok=True)
            tmp = Path(args.save_grads) / ("grad_%02d.f32.tmp" % s)
            grad.tofile(str(tmp))
            os.replace(tmp, Path(args.save_grads) / ("grad_%02d.f32" % s))
        rec["shards"].append(row)
        _say("shard %d: loss %s want %s %s, gradient %s, %.1f s" % (s, got, want, row["loss_verdict"], row["gradient_sha256"][:16], sec))
        _write(args.out, dict(rec, verdict="RUNNING"))
        elapsed = time.perf_counter() - t_start
        per = elapsed / len(rec["shards"])
        if args.deadline_seconds and elapsed + per > args.deadline_seconds and s != shards[-1]:
            _say("deadline: %.0f s used, %.0f s a shard; stopping after shard %d" % (elapsed, per, s))
            break
    done = [r["shard"] for r in rec["shards"]]
    losses_ok = all(r["loss_verdict"] == "PASS" for r in rec["shards"])
    rec["seconds_per_shard"] = round((time.perf_counter() - t_start) / len(done), 1)
    rec["shards_done"] = len(done)
    if folds and done == list(range(K)):
        rec["gradient_sha256"] = seg._hash_gradient(total, scheme)
        grad_ok = rec["gradient_sha256"] == line["gradient_sha256"]
        rec["verdict"] = "PASS" if grad_ok and losses_ok else "FAIL"
    else:
        rec["verdict"] = ("PARTIAL: %d of %d shards; per-shard losses %s; the summed gradient was NOT compared here"
                          % (len(done), K, "all PASS" if losses_ok else "FAIL"))
        if not losses_ok:
            rec["verdict"] = "FAIL (per-shard loss)"
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(args.out, rec)
    _say("%s -> %s" % (rec["verdict"], args.out))
    return 1 if rec["verdict"].startswith("FAIL") else 0


def cmd_fold(args):
    """The ordered left fold over saved shard gradients, optionally continuing
    a prefix folded elsewhere (the chained fold: `--prefix` holds
    fold(g_0..g_{A-1}) and `--shards A:B` continues it). When the fold reaches
    shard K-1 from shard 0, its hash is compared to the chain line's
    `gradient_sha256`."""
    np = _np()
    recipe = seg.load_recipe(args.recipe)
    chain = load_chain(args.chain)
    line = chain[args.step]
    K = recipe_dims(recipe)["K"]
    scheme = seg.hash_scheme_of(recipe)
    shards = _shard_range(args.shards, K)
    if shards != list(range(shards[0], shards[-1] + 1)):
        raise SystemExit("--shards for a fold must be contiguous A:B")
    if shards[0] != 0 and not args.prefix:
        raise SystemExit("a fold that starts at shard %d needs --prefix (the fold of shards 0..%d)" % (shards[0], shards[0] - 1))
    total = None
    rec = dict(schema=SCHEMA, check="fold", step=args.step, K=K, hash_scheme=scheme, shards=[shards[0], shards[-1] + 1],
               chain_gradient_sha256=line["gradient_sha256"], rows=[], box=_box())
    if args.prefix:
        total = np.fromfile(args.prefix, dtype="<f4")
        rec["prefix"] = dict(file=Path(args.prefix).name, sha256=seg._sha_file(args.prefix),
                             hash=seg._hash_gradient(total, scheme))
    t0 = time.perf_counter()
    for s in shards:
        path = Path(args.grads) / ("grad_%02d.f32" % s)
        g = np.fromfile(str(path), dtype="<f4")
        if total is not None and g.size != total.size:
            raise SystemExit("%s has %d values, the fold %d" % (path, g.size, total.size))
        total = ordered_fold_step(total, g, np)
        rec["rows"].append(dict(shard=s, gradient_sha256=seg._hash_gradient(g, scheme)))
    rec["fold_seconds"] = round(time.perf_counter() - t0, 1)
    rec["result_sha256"] = seg._hash_gradient(total, scheme)
    if args.save_prefix:
        total.tofile(args.save_prefix)
        rec["saved"] = dict(file=Path(args.save_prefix).name, sha256=seg._sha_file(args.save_prefix))
    if shards[-1] == K - 1:
        rec["verdict"] = "PASS" if rec["result_sha256"] == line["gradient_sha256"] else "FAIL"
    else:
        rec["verdict"] = "PREFIX (shards 0..%d; not compared)" % shards[-1]
    rec["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _write(args.out, rec)
    _say("%s: fold %s, chain %s -> %s" % (rec["verdict"], rec["result_sha256"], line["gradient_sha256"], args.out))
    return 1 if rec["verdict"] == "FAIL" else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, chain=True, tokens=True):
        p.add_argument("--recipe", required=True)
        p.add_argument("--manifest", required=True, help="the token stream's manifest.json")
        if tokens:
            p.add_argument("--tokens", required=True, help="a tokens directory, or a JSON of presigned part URLs")
            p.add_argument("--checkpoint", required=True)
            p.add_argument("--out", required=True)
        if chain:
            p.add_argument("--chain", required=chain == "required", default=None)

    p = sub.add_parser("plan")
    common(p, chain=False, tokens=False)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--heldout-index", type=int, default=0)

    p = sub.add_parser("loss")
    common(p, chain="required")
    p.add_argument("--step", type=int, required=True, help="the chain line (the step completed), N+1 for checkpoint N")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--threaded", action="store_true", help="the host's threaded forward (same bits by contract)")
    p.add_argument("--perturb-token", action="store_true", help="NEGATIVE CONTROL: first input id of row 0 plus one")
    p.add_argument("--perturb-param-bit", type=int, default=None, help="NEGATIVE CONTROL: flip the low bit of parameter I")

    p = sub.add_parser("heldout")
    common(p, chain=True)
    p.add_argument("--heldout-index", type=int, default=0)
    p.add_argument("--threaded", action="store_true")
    p.add_argument("--expect-f32-hex", default=None, help="another column's held-out loss bits to hold this one to")

    p = sub.add_parser("gradient")
    common(p, chain="required")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--shards", default="0:64", help="A:B (a prefix 0:B is folded here) or a comma list (saved for `fold`)")
    p.add_argument("--save-grads", default=None, help="write each shard's gradient as DIR/grad_NN.f32")
    p.add_argument("--deadline-seconds", type=float, default=0, help="stop after the shard that would cross this")

    p = sub.add_parser("fold")
    p.add_argument("--recipe", required=True)
    p.add_argument("--chain", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--grads", required=True, help="the directory of grad_NN.f32 files")
    p.add_argument("--shards", required=True, help="A:B, contiguous")
    p.add_argument("--prefix", default=None, help="the fold of shards 0..A-1, float32 file")
    p.add_argument("--save-prefix", default=None, help="write the result here (a prefix for the next owner)")
    p.add_argument("--out", required=True)

    args = ap.parse_args(argv)
    return dict(plan=cmd_plan, loss=cmd_loss, heldout=cmd_heldout, gradient=cmd_gradient,
                fold=cmd_fold)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
