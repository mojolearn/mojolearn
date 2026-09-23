#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""One segment of a checkpoint-to-checkpoint language-model training run.

    python3 tools/lm_segment.py recipe  --out recipe.json --shape B L DM H KV HD FF LAYERS V \\
                                        --shards 64 --steps 5000 --peak-lr 6e-4 --warmup 250 \\
                                        --tokens DIR [--checkpoint-every 100] [--boundaries 1000,2000,...]
    python3 tools/lm_segment.py init    --recipe recipe.json --tokens DIR --out ckpt_00000000.blm
    python3 tools/lm_segment.py keys    --recipe recipe.json --from-step 1000 --steps 1000 [--boundary 2000]
    python3 tools/lm_segment.py run     --recipe recipe.json --tokens DIR --from CKPT --steps S --out DIR \\
                                        [--devices 0,1] [--route A --segment 2 --label nvidia-h100] \\
                                        [--boundary GLOBAL_STEP] [--expect-chain CHAIN.jsonl] \\
                                        [--upload-urls URLS.json] [--record-window A:B] [--zero-moments]
                                        [--control none|shards=N|swap=A,B|split[=S]|ulp=S,I]
    python3 tools/lm_segment.py compare CHAIN_A.jsonl CHAIN_B.jsonl [...]

THE RUN IS A CHAIN OF SEGMENTS, EACH ON WHATEVER HARDWARE. A segment starts
from a checkpoint (`--from`), runs S optimizer steps of K logical shards with
`ParallelByteLanguageModelTrainer` on the devices of one box, and ends at a
checkpoint. K, the shape, the optimizer, the learning-rate table and the data
schedule are the RECIPE and never change between segments; the device count
and the vendor do. Steps are numbered globally: a checkpoint is
`ckpt_<global step>.blm`, so nothing about the cadence depends on which
segment a step is in (docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md, section 6).

WHAT IS WRITTEN, EVERY STEP: one line of `chain.jsonl` with the global step,
every shard's loss (float32 bits), the learning-rate bits the device used,
the sha256 of the complete state (parameters, m, v, flags), the sha256 of
the summed gradient, the step's seconds, and the sha256 of the previous line,
so the log is a chain. `--expect-chain` holds every line to another run's
line at the same global step (route B against route A; an arrival replay
against the sender), and the segment STOPS at the first disagreement.

WHAT IS WRITTEN, ON A CADENCE: a `mojolearn.byte-lm-stream.v1` checkpoint
(`_byte_lm_checkpoint`, the 1.95 GB durable format at the 162M shape) at
every global step that is a multiple of `checkpoint_every`, at `--boundary`
minus two (for the receiver's arrival replay), and at the segment's end.
Every file's size and sha256 go to `manifest.tsv`. With `--upload-urls`, a
file is PUT to its presigned URL as soon as it is written and its sha256
verified, so nothing lives only on the box; a key with no URL is refused
BEFORE the first step so a missing upload cannot pass silently.

THE LEARNING-RATE SCHEDULE IS A TABLE, NOT A FORMULA. `recipe` evaluates
warmup and cosine decay ONCE, on the machine that writes the recipe, and
stores one float32 bit pattern per step. A box reads its step's bits from
the table and sends them to the device through `set_lr`, which returns the
bits it stored. Nothing on a box evaluates `cos`, whose last bit depends on
the platform's libm. Shard gradients are SUMMED in the device fold, not
averaged, and the peak rate is written for that sum.

THE SEED CHECKPOINT IS BYTES, NOT A GENERATOR. `init` draws the weights
once with NumPy's `default_rng(seed)` and writes the step-0 checkpoint;
every route starts from that file's bytes and its sha256 is in the
manifest, so a NumPy version cannot become part of the claim.

NEGATIVE CONTROLS. `--zero-moments` restores the checkpoint with zeroed
AdamW moments and must FAIL an `--expect-chain` at the first step; a recipe
whose K, shape, schedule or tokens differ from the checkpoint's is refused
by name before any device work. `mojolearn.cross_vendor` holds the one-ulp
control for the multi-vendor segment. `--control` runs one of the plan's
controls (docs/GPT3_SMALL_SIX_SEGMENT_PLAN.md, section 6 item 7) against
the real chain; see `parse_control`. A control is STAMPED into every chain
line (`control`) and into segment.json, writes no checkpoint and refuses
`--upload-urls`, so its output can never be mistaken for a real run's.

Nothing here is imported by `mojolearn verify`; it is the driver of one
evidence-producing run.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))

RECIPE_SCHEMA = "mojolearn.lm-segment.recipe.v1"
CHAIN_SCHEMA = "mojolearn.lm-segment.chain.v1"
SCHEDULE = "lm-segment.shards.v1"  # shard k of optimizer step s reads TokenBatches.ids(s*K + k)
ARRAYS = ("parameters", "m", "v", "flags")


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _sha_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                return h.hexdigest()
            h.update(block)


def _f32_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _bits_f32(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _box():
    info = dict(host=platform.node(), machine=platform.machine(), system=platform.system(),
                python=platform.python_version())
    for cmd in (["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
                ["rocm-smi", "--showproductname", "--showuniqueid", "--csv"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
            if out:
                info[cmd[0]] = out
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def _commit():
    for name in ("commit.txt",):
        if Path(name).is_file():
            return Path(name).read_text().split()[0]
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip() or None
    except OSError:
        return None


# ---------------------------------------------------------------- the recipe

def lr_table(peak, warmup, total, final_ratio):
    """Float32 bits per optimizer step t = 1..total. Linear warmup from
    peak/warmup at t = 1 to peak at t = warmup, then cosine decay to
    final_ratio * peak at t = total. Evaluated in float64 once, here."""
    if not (0 < warmup <= total) or peak <= 0 or not (0 < final_ratio <= 1):
        raise ValueError("lr_table: need 0 < warmup <= total, peak > 0, 0 < final_ratio <= 1")
    final = peak * final_ratio
    out = []
    for t in range(1, total + 1):
        if t <= warmup:
            lr = peak * t / warmup
        elif total == warmup:
            lr = final
        else:
            frac = (t - warmup) / (total - warmup)
            lr = final + (peak - final) * 0.5 * (1.0 + math.cos(math.pi * frac))
        bits = _f32_bits(lr)
        if bits == 0 or _bits_f32(bits) <= 0:
            raise ValueError("lr_table: a learning rate rounded to zero at step %d" % t)
        out.append("%08x" % bits)
    return out


class ManifestBatches:
    """What `recipe` and `init` need of a token stream, from its manifest.json
    alone: the identity (sha256, manifest sha256, train range, vocabulary)
    and the schedule, with no 12 GB file on this machine. `ids()` is
    deliberately absent; a box that trains opens the real `TokenBatches`,
    which re-hashes the stream against the same manifest."""

    def __init__(self, manifest_path, batch, length):
        raw = Path(manifest_path).read_bytes()
        self.manifest = json.loads(raw)
        if self.manifest.get("schema") != "mojolearn.byte-lm.tokens.v1":
            raise SystemExit("%s is not a mojolearn.byte-lm.tokens.v1 manifest" % manifest_path)
        self.manifest_sha256 = _sha(raw)
        self.sha256 = self.manifest["sha256"]
        self.batch, self.length = int(batch), int(length)
        lo, hi = self.manifest.get("train_range") or [0, int(self.manifest["tokens"])]
        self.lo, self.hi = int(lo), int(hi)
        self.modulus = self.hi - self.lo - self.length - 1
        self.vocabulary = dict(self.manifest["vocabulary"])

    def data_schedule(self, **extra):
        v = self.manifest["vocabulary"]
        out = dict(schema="mojolearn.byte-lm.tokens.v1", schedule="train-range-modulo.v1", tokens_sha256=self.sha256,
                   corpus_sha256=self.manifest["source"]["sha256"], batch=self.batch, length=self.length,
                   train_range=[self.lo, self.hi],
                   vocabulary=dict(schema=v["schema"], sha256=v["sha256"], n_vocab=v["n_vocab"]))
        out.update(extra)
        return out


def _batches_for(recipe_or_shape, tokens, batch, length):
    """`TokenBatches` over a tokens directory, or `ManifestBatches` over a
    manifest.json path (or a directory holding only the manifest)."""
    from mojolearn import lm_corpus
    t = Path(tokens)
    if t.is_file():
        return ManifestBatches(t, batch, length)
    if not (t / "tokens.i32").exists() and (t / "manifest.json").exists():
        return ManifestBatches(t / "manifest.json", batch, length)
    return lm_corpus.TokenBatches(t, batch, length)


def cmd_recipe(args):
    batches = _batches_for(None, args.tokens, args.shape[0], args.shape[1])
    if args.steps * args.shards * args.shape[0] * args.shape[1] > batches.modulus:
        print("NOTE: the run reads the train range more than once (%d tokens per pass, %d needed)"
              % (batches.modulus, args.steps * args.shards * args.shape[0] * args.shape[1]))
    shape = list(args.shape)
    shape[8] = int(batches.vocabulary["n_vocab"])
    boundaries = [int(x) for x in args.boundaries.split(",")] if args.boundaries else []
    if any(b <= 0 or b > args.steps for b in boundaries) or boundaries != sorted(set(boundaries)):
        raise SystemExit("--boundaries must be increasing global steps within the run")
    if any(b % args.checkpoint_every for b in boundaries):
        raise SystemExit("every boundary must be a multiple of --checkpoint-every so it IS a cadence checkpoint")
    recipe = dict(
        schema=RECIPE_SCHEMA,
        shape=shape, logical_shards=args.shards, steps=args.steps, seed=args.seed,
        optimizer=dict(betas=[args.beta1, args.beta2], eps=args.eps, weight_decay=args.weight_decay),
        schedule=dict(kind="warmup-linear-cosine", peak_lr=args.peak_lr, warmup_steps=args.warmup,
                      final_ratio=args.final_ratio, table_f32_hex=lr_table(args.peak_lr, args.warmup, args.steps,
                                                                            args.final_ratio)),
        data=dict(schedule=SCHEDULE, tokens_sha256=batches.sha256, tokens_manifest_sha256=batches.manifest_sha256,
                  train_range=[batches.lo, batches.hi], vocabulary=batches.vocabulary),
        checkpoint_every=args.checkpoint_every, boundaries=boundaries, hash_scheme=args.hash_scheme,
        reduction="ordered_sum", written_by=dict(box=_box(), commit=_commit(), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
    )
    Path(args.out).write_text(json.dumps(recipe, indent=1) + "\n")
    print("wrote %s: shape %s, K=%d, %d steps, %d tokens, lr table %d entries, sha256 %s"
          % (args.out, shape, args.shards, args.steps, args.steps * args.shards * shape[0] * shape[1],
             len(recipe["schedule"]["table_f32_hex"]), _sha_file(args.out)[:16]))
    return 0


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise SystemExit("%s is not a %s" % (path, RECIPE_SCHEMA))
    if len(recipe["schedule"]["table_f32_hex"]) != recipe["steps"]:
        raise SystemExit("recipe: the learning-rate table has %d entries for %d steps"
                         % (len(recipe["schedule"]["table_f32_hex"]), recipe["steps"]))
    return recipe


def data_schedule(recipe, batches):
    """The trainer's `data_schedule`, kept in every checkpoint: the recipe's
    identity. A checkpoint whose schedule differs is refused on resume."""
    return batches.data_schedule(
        seed=recipe["seed"], logical_shards=recipe["logical_shards"], shard_schedule=SCHEDULE,
        lr_table_sha256=_sha("".join(recipe["schedule"]["table_f32_hex"]).encode()),
        recipe_steps=recipe["steps"])


def open_batches(recipe, tokens_dir):
    b = _batches_for(recipe, tokens_dir, recipe["shape"][0], recipe["shape"][1])
    if b.sha256 != recipe["data"]["tokens_sha256"]:
        raise SystemExit("tokens %s are not the recipe's (sha256 %s, recipe %s)"
                         % (tokens_dir, b.sha256[:16], recipe["data"]["tokens_sha256"][:16]))
    if int(b.vocabulary["n_vocab"]) != recipe["shape"][8]:
        raise SystemExit("the tokens' vocabulary has %s ids, the recipe's shape has %d"
                         % (b.vocabulary["n_vocab"], recipe["shape"][8]))
    return b


# ---------------------------------------------------------------- checkpoints

def checkpoint_name(step):
    return "ckpt_%08d.blm" % step


def save_checkpoint(trainer, directory, *, manifest, upload):
    """Stream one checkpoint from a raw export, verify, pin, upload."""
    from mojolearn import _byte_lm_checkpoint as ck
    step = trainer.step_
    raw = trainer.export_raw()
    state = dict(trainer._state)
    state.update(raw)
    state["completed_steps"] = step
    state["next_batch_index"] = step
    name = checkpoint_name(step)
    path = Path(directory) / name
    t0 = time.perf_counter()
    digest = ck.save(path, state)
    seconds = time.perf_counter() - t0
    size = path.stat().st_size
    manifest.pin(name, size, digest)
    upload(name, path)
    return dict(step=step, file=name, bytes=size, sha256=digest, save_seconds=round(seconds, 3))


def load_checkpoint(path, *, zero_moments=False):
    from mojolearn import _byte_lm_checkpoint as ck
    digest = _sha_file(path)
    state = ck.load(path)
    if zero_moments:
        for key in ("m", "v"):
            arr = state[key]
            mv = _bytes_of(arr, writable=True)
            mv[:] = bytes(len(mv))
    return state, digest


class Manifest:
    """`manifest.tsv`: name, bytes, sha256 of every artifact this segment wrote."""

    def __init__(self, directory):
        self.path = Path(directory) / "manifest.tsv"
        self.rows = {}

    def pin(self, name, size, digest):
        self.rows[name] = (size, digest)
        lines = ["%s\t%d\t%s\n" % (k, v[0], v[1]) for k, v in sorted(self.rows.items())]
        tmp = self.path.with_suffix(".tsv.tmp")
        tmp.write_text("".join(lines))
        os.replace(tmp, self.path)


class Uploader:
    """PUT each written file to its presigned URL, verified by sha256 first.
    Refuses at construction if any expected key has no URL."""

    def __init__(self, urls_path, expected_keys, log):
        self.urls = json.loads(Path(urls_path).read_text()) if urls_path else None
        self.log = log
        self.uploaded = []
        if self.urls is not None:
            missing = [k for k in expected_keys if k not in self.urls]
            if missing:
                raise SystemExit("--upload-urls has no URL for %d expected key(s): %s ..."
                                 % (len(missing), missing[:4]))

    def __call__(self, name, path):
        if self.urls is None:
            return
        url = self.urls.get(name)
        if url is None:
            raise SystemExit("no upload URL for %s" % name)
        for attempt in range(1, 4):
            t0 = time.perf_counter()
            r = subprocess.run(["curl", "-sS", "--fail", "--retry", "2", "-T", str(path), url],
                               capture_output=True, text=True)
            if r.returncode == 0:
                self.uploaded.append(dict(name=name, seconds=round(time.perf_counter() - t0, 2)))
                self.log("uploaded %s in %.1f s" % (name, time.perf_counter() - t0))
                return
            self.log("upload of %s failed (attempt %d): %s" % (name, attempt, r.stderr.strip()[:200]))
            time.sleep(5 * attempt)
        raise SystemExit("upload of %s failed three times; the segment stops rather than run unrecorded" % name)


def expected_keys(recipe, from_step, steps, boundary):
    keys = ["chain.jsonl", "manifest.tsv", "segment.json"]
    every = int(recipe["checkpoint_every"])
    last = from_step + steps
    for step in range(from_step + 1, last + 1):
        if step % every == 0 or step == last or (boundary and step == boundary - 2):
            keys.append(checkpoint_name(step))
    return keys


def cmd_keys(args):
    recipe = load_recipe(args.recipe)
    for k in expected_keys(recipe, args.from_step, args.steps, args.boundary):
        print(k)
    return 0


# ---------------------------------------------------------------- init

def cmd_init(args):
    import numpy as np
    from mojolearn import _byte_lm_checkpoint as ck
    from mojolearn import LanguageModelTrainer as Trainer, LanguageModelConfig as Shape
    recipe = load_recipe(args.recipe)
    batches = open_batches(recipe, args.tokens)
    shape = Shape(*recipe["shape"])
    rng = np.random.default_rng(recipe["seed"])
    weights = rng.normal(0, .02, shape.n_total).astype(np.float32)
    for entry in Trainer.parameter_registry(shape):
        if "norm" in entry["name"]:
            weights[entry["offset"]:entry["offset"] + entry["size"]] += np.float32(1)
    opt = recipe["optimizer"]
    first_lr = _bits_f32(int(recipe["schedule"]["table_f32_hex"][0], 16))
    trainer = Trainer(weights, shape=shape, data_schedule=data_schedule(recipe, batches), lr=first_lr,
                      betas=tuple(opt["betas"]), eps=opt["eps"], weight_decay=opt["weight_decay"])
    state = trainer.state_dict()
    digest = ck.save(args.out, state)
    print("wrote %s: step 0, %d parameters, init sha256 %s, file sha256 %s, numpy %s"
          % (args.out, shape.n_total, _sha(weights.tobytes())[:16], digest[:16], np.__version__))
    return 0


# ---------------------------------------------------------------- run

def _window(text):
    a, sep, b = text.partition(":")
    if not sep:
        raise argparse.ArgumentTypeError("a window is A:B, global step indices, B exclusive")
    a, b = int(a), int(b)
    if a < 0 or b <= a:
        raise argparse.ArgumentTypeError("a window needs 0 <= A < B")
    return a, b


def _bytes_of(obj, writable=False):
    """A flat `'B'` memoryview of any array the package hands out. NumPy
    arrays and bytes export the buffer protocol; mojolearn's own `Array`
    (what `export_raw` and the token batches return on a box without NumPy,
    and what the wheel's `empty` builds) cannot on Python 3.10 and 3.11
    (`python/mojolearn/_array.py`, DEVIATION 2305), so it is read through
    the package's `_buffer` at its address. The T3 rehearsal's first segment
    (Python 3.11 venv on the box) crashed at its first hash on exactly this.
    The caller keeps `obj` alive while the view is in use."""
    from mojolearn._array import Array
    if isinstance(obj, Array):
        from mojolearn._buffer import addr, addr_ro, memory_at
        a = addr(obj, name="array") if writable else addr_ro(obj, name="array")
        return memory_at(a, obj.nbytes, writable=writable)
    return memoryview(obj).cast("B")


HASH_SLICES = 8


def _sha_sliced(view, slices=HASH_SLICES, pool=None):
    """sha256 of the sha256s of `slices` equal byte ranges of `view`, the
    ranges hashed in parallel threads (hashlib releases the interpreter
    lock for a large buffer). At 1.95 GB of state a single sha256 cost 8 s
    on an H100 pod's host, a fifth of a one-device step and two fifths of a
    two-device step (bench/results/lm_t1_2026-09-22); this is the same
    strength of digest at about a quarter of the wall time on four or more
    cores. A chain line names its scheme, and two chains compare only under
    the same scheme."""
    view = _bytes_of(view)
    n = len(view)
    bounds = [(n * i // slices, n * (i + 1) // slices) for i in range(slices)]
    if pool is None:
        parts = [_sha(view[a:b]) for a, b in bounds]
    else:
        parts = list(pool.map(lambda ab: _sha(view[ab[0]:ab[1]]), bounds))
    return _sha("".join(parts).encode())


_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        from concurrent.futures import ThreadPoolExecutor
        _POOL = ThreadPoolExecutor(max_workers=HASH_SLICES)
    return _POOL


SCHEME_V1 = "sha256.v1"                       # one sha256 over the four arrays in order (the T1 and T2 chains)
SCHEME_V2 = "sliced-sha256-%d.v2" % HASH_SLICES  # sha256 of eight sliced sha256s per array, threaded
DEFAULT_SCHEME = SCHEME_V1


def hash_scheme_of(recipe):
    """THE RECIPE NAMES THE HASH SCHEME, so a run cannot change it between
    segments (T2 segment 3 once failed on exactly that: a replay hashed
    with a newer scheme against a chain hashed with the older). A recipe
    without the field is the first scheme."""
    scheme = recipe.get("hash_scheme", DEFAULT_SCHEME)
    if scheme not in (SCHEME_V1, SCHEME_V2):
        raise SystemExit("recipe: unknown hash_scheme %r" % scheme)
    return scheme


def _hash_arrays(raw, scheme=DEFAULT_SCHEME):
    """The state hash over parameters, m, v, flags in order, under `scheme`."""
    h = hashlib.sha256()
    if scheme == SCHEME_V1:
        for key in ARRAYS:
            h.update(_bytes_of(raw[key]))
        return h.hexdigest()
    for key in ARRAYS:
        h.update(_sha_sliced(raw[key], pool=_pool()).encode())
    return h.hexdigest()


def _hash_gradient(view, scheme=DEFAULT_SCHEME):
    if scheme == SCHEME_V1:
        return _sha(_bytes_of(view))
    return _sha_sliced(view, pool=_pool())


def _chain_index(path):
    """Lines of an expected chain by global step."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("schema") == CHAIN_SCHEMA:
                # A line written before the field existed was hashed under the
                # first scheme; compared as such, not as a missing value.
                row.setdefault("hash_scheme", SCHEME_V1)
                out[int(row["step"])] = row
    if not out:
        raise SystemExit("%s holds no chain lines" % path)
    return out


COMPARED = ("state_sha256", "gradient_sha256", "losses_f32_hex", "lr_f32_hex", "hash_scheme")


class ChainWriter:
    """`chain.jsonl`: one canonical JSON line per completed step, each carrying
    the sha256 of the previous line, held to an expected chain when given.
    `write(row)` returns False when the segment must stop (a disagreement)."""

    def __init__(self, path, first, expect, say):
        self.path, self.expect, self.say = Path(path), expect, say
        self.prev, self.done, self.verdict, self.disagreements = None, first, "PASS", []
        self.first = first
        if self.path.exists():
            lines = [l for l in self.path.read_text().splitlines() if l.strip()]
            if lines:
                prev_row = json.loads(lines[-1])
                if int(prev_row["step"]) != first:
                    raise SystemExit("REFUSED: %s ends at step %s, the checkpoint is at %d; use a fresh --out or the matching checkpoint"
                                     % (self.path, prev_row["step"], first))
                self.prev = _sha(lines[-1].encode())
        self.fh = self.path.open("a")

    def write(self, row):
        row = dict(row, prev=self.prev)
        line = _canonical(row)
        self.fh.write(line + "\n")
        self.fh.flush()
        self.prev = _sha(line.encode())
        completed = int(row["step"])
        self.done = completed
        K = len(row["losses_f32_hex"])
        mean_loss = sum(_bits_f32(int(h, 16)) for h in row["losses_f32_hex"]) / max(K, 1)
        self.say("step %d lr %s loss %.4f state %s grad %s %.2f s (+%.2f s hashing)"
                 % (completed, row["lr_f32_hex"], mean_loss, row["state_sha256"][:16], row["gradient_sha256"][:16],
                    row.get("seconds", 0.0), row.get("hash_seconds", 0.0)))
        if self.expect is None:
            return True
        want = self.expect.get(completed)
        if want is None:
            self.say("step %d: the expected chain has no line (not compared)" % completed)
            return True
        diff = [k for k in COMPARED if want.get(k) != row.get(k)]
        if diff:
            self.disagreements.append(dict(step=completed, fields=diff, expected={k: want.get(k) for k in diff},
                                           got={k: row.get(k) for k in diff}))
            self.verdict = "FAIL"
            self.say("DISAGREE at step %d on %s; the segment stops here" % (completed, diff))
            return False
        self.say("step %d agrees with the expected chain" % completed)
        return True

    def close(self, last):
        self.fh.close()
        if self.expect is not None and self.verdict == "PASS":
            compared = sum(1 for s in range(self.first + 1, last + 1) if s in self.expect)
            if compared == 0:
                self.verdict = "FAIL"
                self.disagreements.append(dict(reason="NOTHING WAS COMPARED: the expected chain covers none of these steps"))
        return self.verdict, self.disagreements, self.done


def _block(text):
    a, sep, b = text.partition(":")
    if not sep:
        raise SystemExit("--live-shards is A:B, this box's contiguous block of shards, B exclusive")
    a, b = int(a), int(b)
    if a < 0 or b <= a:
        raise SystemExit("--live-shards needs 0 <= A < B")
    return list(range(a, b))


def _run_live(args, recipe, batches, state, devices, chain, table, out, manifest, upload, segment, say,
              first, last, wants_checkpoint):
    """The multi-vendor segment: this box is one worker of a chained
    `mojolearn.cross_vendor` group, and, as the coordinator, also the fold's
    host. Every agreed step becomes a chain line of the same shape a one-box
    segment writes (the total's hash is the summed gradient's hash, since the
    host fold and the device fold are the same bits), so route A's live
    segment and route B's compare line for line, and checkpoints come from
    this worker's replica."""
    import threading
    from mojolearn.cross_vendor import Coordinator, Worker
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Par
    K = int(recipe["logical_shards"])
    block = _block(args.live_shards)
    if any(k < 0 or k >= K for k in block):
        raise SystemExit("--live-shards must lie inside 0..%d" % (K - 1))
    segment["live"] = dict(role=args.live_role, shards=[block[0], block[-1] + 1], workers=args.live_workers,
                           port=args.live_port, address=args.live_address, protocol="chained")
    lr_for_step = lambda step: _bits_f32(int(table[step], 16))  # noqa: E731
    batches_fn = lambda step, k: batches.ids(step * K + k)  # noqa: E731
    last_commit = [time.perf_counter()]
    stop = {"flag": False}

    scheme = hash_scheme_of(recipe)

    def make_commit(trainer):
        def on_commit(completed, row):
            now = time.perf_counter()
            # the chain line's hashes under the RECIPE's scheme, from this
            # replica's state and the total's bytes, so a live segment's chain
            # compares with any other segment's (the group's own agreement is
            # the coordinator's plain sha256 in coordinator.jsonl)
            t0 = time.perf_counter()
            state_digest = _hash_arrays(trainer.export_raw(), scheme)
            grad_digest = _hash_gradient(row["total_bytes"], scheme) if row.get("total_bytes") is not None else row["total_sha256"]
            hash_seconds = time.perf_counter() - t0
            line = dict(schema=CHAIN_SCHEMA, step=completed, route=args.route, segment=args.segment, label=args.label,
                        lr_f32_hex=table[completed - 1], losses_f32_hex=["%08x" % _f32_bits(x) for x in row["losses"]],
                        state_sha256=state_digest, gradient_sha256=grad_digest, hash_scheme=scheme,
                        group_state_sha256=row["state"], group_total_sha256=row["total_sha256"],
                        batch_index=[(completed - 1) * K, completed * K], seconds=round(now - last_commit[0], 4),
                        hash_seconds=round(hash_seconds, 3), live=segment["live"])
            last_commit[0] = time.perf_counter()
            if not chain.write(line):
                stop["flag"] = True
                raise SystemExit("the live segment stops at a disagreement")
            if wants_checkpoint(completed) and args.live_role == "coordinator":
                info = save_checkpoint(trainer, out, manifest=manifest, upload=upload)
                segment["checkpoints"].append(info)
                say("checkpoint %s (%d bytes, sha256 %s, %.1f s)" % (info["file"], info["bytes"], info["sha256"][:16], info["save_seconds"]))
        return on_commit

    errors = []
    coord = None
    if args.live_role == "coordinator":
        coord = Coordinator(host="0.0.0.0", port=args.live_port, workers=args.live_workers, logical_shards=K,
                            steps=last, chained=True, timeout=args.live_timeout, accept_timeout=args.live_timeout,
                            on_step=lambda r: _append_json(out / "coordinator.jsonl", r))
        t = threading.Thread(target=lambda: _safe(coord.run, errors), name="coordinator")
        t.start()
        coord.ready.wait(30)
        address = ("127.0.0.1", coord.bound_port)
        say("coordinator listening on port %d for %d workers, K=%d, until step %d" % (coord.bound_port, args.live_workers, K, last))
    else:
        host, port = args.live_address.rsplit(":", 1)
        address = (host, int(port))
    lock = threading.Lock()
    workers = []
    trainer = Par(state, devices=(devices[0],), logical_shards=1, pool_optimizer=False)
    workers.append(Worker(trainer=_Locked(trainer, lock) if args.live_local_extra else trainer, shards=block,
                          batches=batches_fn, address=address, name=args.label, chained=True,
                          lr_for_step=lr_for_step, on_commit=make_commit(trainer),
                          connect_timeout=args.live_timeout, timeout=args.live_timeout))
    if args.live_local_extra:  # a test: a second worker in this process, same device under a lock
        extra = Par(state, devices=(devices[0],), logical_shards=1, pool_optimizer=False)
        workers.append(Worker(trainer=_Locked(extra, lock), shards=_block(args.live_local_extra), batches=batches_fn,
                              address=address, name=args.label + "-extra", chained=True, lr_for_step=lr_for_step,
                              connect_timeout=args.live_timeout, timeout=args.live_timeout))
    threads = [threading.Thread(target=lambda w=w: _safe(w.run, errors), name=w.name) for w in workers]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if coord is not None:
        t.join()
    trainer.close()
    if errors and not stop["flag"]:
        segment["live_errors"] = [str(e) for e in errors]
        say("LIVE ERROR: %s" % "; ".join(str(e)[:200] for e in errors))
        chain.verdict = "FAIL"
        chain.disagreements.append(dict(reason="live group error", errors=[str(e) for e in errors]))


def _safe(fn, errors):
    try:
        fn()
    except BaseException as e:  # noqa: BLE001
        errors.append(e)


def _append_json(path, row):
    with open(path, "a") as fh:
        fh.write(_canonical(row) + "\n")


class _Locked:
    """A trainer whose device calls take turns under one lock (one Metal job at a time)."""

    def __init__(self, trainer, lock):
        self._t, self._lock = trainer, lock

    def __getattr__(self, name):
        attr = getattr(self._t, name)
        if not callable(attr):
            return attr

        def call(*a, **k):
            with self._lock:
                return attr(*a, **k)
        return call


# ---------------------------------------------------------------- controls

CONTROL_KINDS = ("none", "shards", "swap", "split", "ulp")


def parse_control(text):
    """`--control`, one of:

      none         the plain step, stamped: must PASS (the harness's own check)
      shards=N     run N logical shards, the first N of the recipe's K at every
                   step (1 <= N < K); the data of the kept shards is unchanged,
                   so what differs is the fold's shard count alone (`k63` is
                   shards=63)
      swap=A,B     exchange the shards in fold positions A and B (both read
                   their own recipe batch, in each other's slot). The fold is
                   an ordered left fold, total = g0, then total = total + g_k,
                   so the loss vector moves and, unless {A, B} = {0, 1} (float
                   addition commutes, so g0 + g1 == g1 + g0 bit for bit), the
                   rounding order of the sum moves too
      split[=S]    the split step on one device with NO edit: shards 0..S-1
                   folded on the device, the fold exported, shard S's gradient
                   downloaded and folded back in by `fold_add`, shards S+1..K-1
                   folded on the device, the total applied by `apply_gradient`
                   (S defaults to K-1). Must PASS: it proves the ulp harness
      ulp=S,I      the split step with ONE ulp added to element I of shard S's
                   gradient at the segment's first step (I may be `auto`: the
                   largest-magnitude element of that gradient whose one-ulp
                   change survives the fold's rounding). S >= 1

    Returns a dict with `kind` and `text` (the canonical stamp)."""
    text = (text or "").strip()
    kind, sep, rest = text.partition("=")
    kind = kind.strip()
    if kind == "k63" and not sep:
        kind, sep, rest = "shards", "=", "63"
    if kind not in CONTROL_KINDS:
        raise SystemExit("--control: unknown control %r (one of %s)" % (text, ", ".join(CONTROL_KINDS)))

    def ints(value, n, what):
        parts = [x.strip() for x in value.split(",")]
        if len(parts) != n:
            raise SystemExit("--control %s needs %d comma-separated values" % (what, n))
        try:
            return [int(x) for x in parts]
        except ValueError:
            raise SystemExit("--control %s: %r is not integers" % (what, value))
    if kind == "none":
        if sep:
            raise SystemExit("--control none takes no value")
        return dict(kind="none", text="none")
    if kind == "shards":
        (n,) = ints(rest, 1, "shards=N")
        if n < 1:
            raise SystemExit("--control shards=N needs N >= 1")
        return dict(kind="shards", shards=n, text="shards=%d" % n)
    if kind == "swap":
        a, b = ints(rest, 2, "swap=A,B")
        if a == b or a < 0 or b < 0:
            raise SystemExit("--control swap=A,B needs two different nonnegative positions")
        a, b = min(a, b), max(a, b)
        return dict(kind="swap", a=a, b=b, text="swap=%d,%d" % (a, b))
    if kind == "split":
        if not sep:
            return dict(kind="split", shard=None, text="split")
        (sh,) = ints(rest, 1, "split=S")
        return dict(kind="split", shard=sh, text="split=%d" % sh)
    parts = [x.strip() for x in rest.split(",")]
    if len(parts) != 2:
        raise SystemExit("--control ulp=S,I needs a shard and an element index (or auto)")
    (sh,) = ints(parts[0], 1, "ulp=S,I")
    if parts[1] == "auto":
        index = "auto"
    else:
        (index,) = ints(parts[1], 1, "ulp=S,I")
        if index < 0:
            raise SystemExit("--control ulp=S,I needs I >= 0")
    return dict(kind="ulp", shard=sh, index=index, text="ulp=%d,%s" % (sh, index))


def control_plan(control, K):
    """(K run, slots, split shard): the logical shard count the trainer is
    built with, `slots[k]` = the recipe's shard (0..K-1) whose batch fills
    fold position k, and the split shard for `split`/`ulp` (None otherwise).
    Refuses a control that does not fit this recipe's K."""
    kind = (control or {}).get("kind", "none")
    slots = list(range(K))
    if kind == "shards":
        n = control["shards"]
        if not 1 <= n < K:
            raise SystemExit("--control shards=%d: need 1 <= N < K=%d (the first N of the recipe's shards)" % (n, K))
        return n, list(range(n)), None
    if kind == "swap":
        a, b = control["a"], control["b"]
        if b >= K:
            raise SystemExit("--control swap=%d,%d: positions must lie in 0..%d" % (a, b, K - 1))
        slots[a], slots[b] = slots[b], slots[a]
        return K, slots, None
    if kind in ("split", "ulp"):
        sh = control.get("shard")
        sh = K - 1 if sh is None else sh
        if K < 2 or not 1 <= sh < K:
            raise SystemExit("--control %s: the shard must lie in 1..%d (shard 0 opens the fold)" % (control["text"], K - 1))
        return K, slots, sh
    return K, slots, None


def control_stamp(control, zero_moments):
    """The string every chain line of a control carries, or None for a real run."""
    parts = []
    if zero_moments:
        parts.append("zero-moments")
    if control is not None:
        parts.append(control["text"])
    return "+".join(parts) or None


def _zero_moments(state):
    for key in ("m", "v"):
        mv = _bytes_of(state[key], writable=True)
        mv[:] = bytes(len(mv))


def _f32_normal(x, np):
    return np.isfinite(x) & (np.abs(x) >= np.finfo(np.float32).tiny)


def ulp_edit(gradient, prefix, index):
    """Add one ulp (the bit pattern plus one, away from zero) to element
    `index` of `gradient` IN PLACE, and say what the fold's next add
    (prefix + gradient, one float32 rounding, `cross_vendor.fold_pair`)
    makes of it. `index` "auto" picks the largest-magnitude element whose
    change survives that add, every operand and result normal (the fold's
    flush-to-zero never touches it). `gradient` is a writable float32
    buffer, `prefix` the exported fold of the shards before it (bytes)."""
    import numpy as np
    gb = np.frombuffer(_bytes_of(gradient, writable=True), dtype="<u4")
    gf = gb.view("<f4")
    pf = np.frombuffer(prefix if isinstance(prefix, (bytes, bytearray)) else bytes(prefix), dtype="<f4")
    if len(pf) != len(gf):
        raise SystemExit("ulp: the fold prefix has %d elements, the gradient %d" % (len(pf), len(gf)))
    if index == "auto":
        eb = gb + np.uint32(1)
        ef = eb.view("<f4")
        with np.errstate(over="ignore", invalid="ignore"):
            before, after = pf + gf, pf + ef
            ok = (_f32_normal(gf, np) & _f32_normal(ef, np) & _f32_normal(pf, np) & _f32_normal(before, np)
                  & _f32_normal(after, np) & (before != after))
        if not ok.any():
            raise SystemExit("ulp auto: no element's one-ulp change survives the fold")
        index = int(np.argmax(np.where(ok, np.abs(gf), np.float32(-1))))
        del eb, ef, before, after, ok
    index = int(index)
    if not 0 <= index < len(gf):
        raise SystemExit("ulp: element %d is outside the gradient (%d elements)" % (index, len(gf)))
    old = int(gb[index])
    if (old & 0x7F800000) == 0x7F800000:
        raise SystemExit("ulp: element %d is not finite" % index)
    p = np.float32(pf[index])
    unedited = np.float32(p + np.float32(gf[index]))
    gb[index] = np.uint32(old + 1)
    edited = np.float32(p + np.float32(gf[index]))
    return dict(index=index, gradient_bits_before="%08x" % old, gradient_bits_after="%08x" % (old + 1),
                prefix_bits="%08x" % _f32_bits(float(p)),
                predicted_total_bits_unedited="%08x" % _f32_bits(float(unedited)),
                predicted_total_bits_edited="%08x" % _f32_bits(float(edited)),
                survives_fold=bool(unedited != edited))


def split_step(trainer, shards, split, *, index=None):
    """One step as a sequence of per-shard calls (the live worker's calls,
    on one device): shards 0..split-1 folded on the device, the fold
    exported, shard `split`'s gradient downloaded, edited when `index` is
    given (`ulp_edit`), the device fold reset to the exported prefix and the
    gradient added by `fold_add`, shards split+1.. folded on the device, the
    total exported and applied. Returns (losses, total bytes, edit detail)."""
    import array
    losses = []
    trainer.fold_reset(None)
    for k in range(split):
        losses.append(trainer.shard_gradient_fold(shards[k]))
    prefix = trainer.fold_export()
    loss, g = trainer.shard_gradient(shards[split])
    losses.append(loss)
    detail = None
    if index is not None:
        detail = ulp_edit(g, prefix, index)
    trainer.fold_reset(prefix)
    trainer.fold_add(g)
    for k in range(split + 1, len(shards)):
        losses.append(trainer.shard_gradient_fold(shards[k]))
    total = trainer.fold_export()
    if detail is not None:
        detail["observed_total_bits"] = "%08x" % struct.unpack_from("<I", total, 4 * detail["index"])[0]
        detail["shard"] = split
    trainer.apply_gradient(array.array("f", total))
    return losses, total, detail


def cmd_run(args):
    from mojolearn.parallel_training import ParallelByteLanguageModelTrainer as Par
    recipe = load_recipe(args.recipe)
    K = int(recipe["logical_shards"])
    every = int(recipe["checkpoint_every"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_fh = (out / "log.txt").open("a")

    def say(msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    batches = open_batches(recipe, args.tokens)
    if isinstance(batches, ManifestBatches):
        raise SystemExit("REFUSED: `run` needs the token stream itself (tokens.i32 beside its manifest), not the manifest alone")
    schedule = data_schedule(recipe, batches)
    control = parse_control(args.control) if args.control is not None else None
    stamp = control_stamp(control, args.zero_moments)
    K_run, slots, split = control_plan(control, K)
    if stamp is not None:
        if args.upload_urls:
            raise SystemExit("REFUSED: a control (%s) never uploads; its output must not reach a run's prefix" % stamp)
        if args.live_role:
            raise SystemExit("REFUSED: a control runs on one box, not in a live group")
        args.no_checkpoints = True  # a control's state is never a checkpoint anyone could resume
    if split is not None and "," in args.devices:
        raise SystemExit("REFUSED: --control %s is the one-device split step; pass one device" % control["text"])
    state, from_digest = load_checkpoint(args.from_ckpt)
    first = int(state["completed_steps"])
    if _canonical(state["data_schedule"]) != _canonical(schedule):
        raise SystemExit("REFUSED: the checkpoint's data schedule is not this recipe's over these tokens "
                         "(K, shape, seed, learning-rate table or token stream differ)")
    last = first + args.steps
    if last > recipe["steps"]:
        raise SystemExit("REFUSED: steps %d..%d exceed the recipe's %d" % (first, last, recipe["steps"]))
    if args.boundary is not None and not (first < args.boundary <= last):
        raise SystemExit("--boundary must be a global step inside this segment")
    devices = tuple(int(x) for x in args.devices.split(","))
    if args.live_role and not args.live_shards:
        raise SystemExit("--live-role needs --live-shards A:B")
    if args.live_role == "worker" and not args.live_address:
        raise SystemExit("--live-role worker needs --live-address HOST:PORT")
    if args.live_role == "worker":
        args.no_checkpoints = True  # the coordinator's replica writes the checkpoints
    expect = _chain_index(args.expect_chain) if args.expect_chain else None
    manifest = Manifest(out)
    keys = expected_keys(recipe, first, args.steps, args.boundary) if not args.no_checkpoints else ["chain.jsonl", "manifest.tsv", "segment.json"]
    upload = Uploader(args.upload_urls, keys, say)
    say("segment: route %s segment %s label %s, global steps %d..%d, K=%d, devices %s, from %s (sha256 %s)%s%s"
        % (args.route, args.segment, args.label, first, last, K, devices, Path(args.from_ckpt).name,
           from_digest[:16], " ZEROED MOMENTS (a control that must fail)" if args.zero_moments else "",
           " CONTROL %s (K run %d)" % (stamp, K_run) if stamp else ""))
    segment = dict(schema="mojolearn.lm-segment.run.v1", route=args.route, segment=args.segment, label=args.label,
                   from_checkpoint=dict(file=Path(args.from_ckpt).name, sha256=from_digest, step=first),
                   first_step=first, last_step=last, devices=list(devices), logical_shards=K,
                   recipe_sha256=_sha_file(args.recipe), tokens_sha256=batches.sha256, box=_box(),
                   commit=_commit(), zero_moments=bool(args.zero_moments), expect_chain=args.expect_chain,
                   checkpoints=[], utc_start=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if stamp is not None:
        segment["control"] = dict(control or {}, stamp=stamp, zero_moments=bool(args.zero_moments),
                                  logical_shards_run=K_run, slots=slots, split_shard=split)
    chain = ChainWriter(out / "chain.jsonl", first, expect, say)
    table = recipe["schedule"]["table_f32_hex"]
    scheme = hash_scheme_of(recipe)
    if expect is not None:
        theirs = {row.get("hash_scheme") or SCHEME_V1 for row in expect.values()}
        if theirs != {scheme}:
            raise SystemExit("REFUSED: the expected chain was hashed under %s and this recipe names %s; the two cannot be compared"
                             % (sorted(theirs), scheme))
    segment["hash_scheme"] = scheme
    if expect is not None and first in expect:
        # the checkpoint's own state against the expected chain's line at its
        # step, BEFORE any control touches it: a replay from the wrong bytes
        # would make every later verdict meaningless
        from_state = _hash_arrays(state, scheme)
        segment["from_state"] = dict(sha256=from_state, expected=expect[first]["state_sha256"],
                                     agrees=from_state == expect[first]["state_sha256"])
        say("the checkpoint's state %s the expected chain's step %d (%s)"
            % ("AGREES with" if segment["from_state"]["agrees"] else "DISAGREES with", first, from_state[:16]))
    if args.zero_moments:
        _zero_moments(state)

    def in_window(step):
        return any(a <= step < b for a, b in args.record_window)

    def wants_checkpoint(completed):
        return (not args.no_checkpoints) and (
            completed % every == 0 or completed == last or (args.boundary and completed == args.boundary - 2))

    if args.live_role:
        _run_live(args, recipe, batches, state, devices, chain, table, out, manifest, upload, segment, say,
                  first, last, wants_checkpoint)
        state = None
    else:
        if split is not None:
            # the live worker's trainer: one device, full optimizer replica, per-shard calls
            trainer_cm = Par(state, devices=(devices[0],), logical_shards=1, pool_optimizer=False)
        else:
            trainer_cm = Par(state, devices=devices, logical_shards=K_run)
        with trainer_cm as trainer:
            del state
            for s in range(first, last):
                completed = s + 1
                lr_hex = table[s]  # the rate used to reach step s+1
                lr_used = trainer.set_lr(_bits_f32(int(lr_hex, 16)))
                if "%08x" % _f32_bits(lr_used) != lr_hex:
                    raise SystemExit("learning rate bits drifted between the table and the device")
                shards = [batches.ids(s * K + k) for k in slots]
                t0 = time.perf_counter()
                if split is not None:
                    edit = control["index"] if (control["kind"] == "ulp" and s == first) else None
                    losses, total, detail = split_step(trainer, shards, split, index=edit)
                    if detail is not None:
                        segment["control"]["edit"] = dict(detail, step=completed)
                        say("ulp: shard %d element %d bits %s -> %s, prefix %s, total predicted %s unedited / %s edited, observed %s"
                            % (detail["shard"], detail["index"], detail["gradient_bits_before"], detail["gradient_bits_after"],
                               detail["prefix_bits"], detail["predicted_total_bits_unedited"],
                               detail["predicted_total_bits_edited"], detail["observed_total_bits"]))
                else:
                    losses = trainer.train_step(shards)["losses"]
                step_seconds = time.perf_counter() - t0
                t1 = time.perf_counter()
                raw = trainer.export_raw()
                state_digest = _hash_arrays(raw, scheme)
                grad_digest = _hash_gradient(total if split is not None else trainer.export_gradients(), scheme)
                hash_seconds = time.perf_counter() - t1
                row = dict(schema=CHAIN_SCHEMA, step=completed, route=args.route, segment=args.segment, label=args.label,
                           lr_f32_hex=lr_hex, losses_f32_hex=["%08x" % _f32_bits(x) for x in losses],
                           state_sha256=state_digest, gradient_sha256=grad_digest, hash_scheme=scheme,
                           batch_index=[s * K, s * K + K], seconds=round(step_seconds, 4),
                           hash_seconds=round(hash_seconds, 3))
                if stamp is not None:
                    row["control"] = stamp
                    if slots != list(range(K)):
                        row["shard_slots"] = slots
                if in_window(s):
                    row["window"] = _window_witness(trainer, raw, shards, say)
                if not chain.write(row):
                    break
                if wants_checkpoint(completed):
                    info = save_checkpoint(trainer, out, manifest=manifest, upload=upload)
                    segment["checkpoints"].append(info)
                    say("checkpoint %s (%d bytes, sha256 %s, %.1f s)" % (info["file"], info["bytes"], info["sha256"][:16], info["save_seconds"]))
    verdict, disagreements, done = chain.close(last)
    segment.update(verdict=verdict, disagreements=disagreements, steps_completed=done - first, last_completed=done,
                   utc_end=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), uploads=upload.uploaded)
    manifest.pin("chain.jsonl", chain.path.stat().st_size, _sha_file(chain.path))
    (out / "segment.json").write_text(json.dumps(segment, indent=1, default=str) + "\n")
    upload("chain.jsonl", chain.path)
    upload("segment.json", out / "segment.json")
    upload("manifest.tsv", manifest.path)
    say("%s: %s" % (verdict, out))
    return 0 if verdict == "PASS" else 1


def _window_witness(trainer, raw, shards, say):
    """Inside a record window: the per-array hashes of the post-step state,
    so a divergence names the array, and the sha256 of every shard's ids,
    so a divergence can be told from a data mismatch. Per-shard gradients
    belong to the one-shard replays `mojolearn.cross_vendor` and
    `tools/par_lm_xvendor.py` take from the boundary checkpoints."""
    return dict(parameters=_sha(_bytes_of(raw["parameters"])),
                m=_sha(_bytes_of(raw["m"])), v=_sha(_bytes_of(raw["v"])),
                flags=_sha(_bytes_of(raw["flags"])),
                shards_sha256=[_sha(_bytes_of(x)) for x in shards])


# ---------------------------------------------------------------- compare

def cmd_compare(args):
    chains = [(p, _chain_index(p)) for p in args.chains]
    steps = sorted(set.union(*(set(c) for _, c in chains)))
    compared, bad = 0, []
    for step in steps:
        rows = [(p, c[step]) for p, c in chains if step in c]
        if len(rows) < 2:
            continue
        for key in COMPARED:
            compared += 1
            vals = {p: _canonical(r.get(key)) for p, r in rows}
            if len(set(vals.values())) != 1:
                bad.append("step %d %s: %s" % (step, key, {Path(p).parent.name or p: v[:20] for p, v in vals.items()}))
    for b in bad:
        print("DISAGREE", b)
    print("steps=%d compared=%d disagreements=%d" % (len(steps), compared, len(bad)))
    if compared == 0:
        print("NOTHING WAS COMPARED: the chains share no step")
        return 1
    return 1 if bad else 0


def cmd_manifests(args):
    """Rows with the same name across manifests must agree on bytes and sha256
    (route B's checkpoints against route A's)."""
    tables = []
    for p in args.manifests:
        rows = {}
        for line in Path(p).read_text().splitlines():
            if line.strip():
                name, size, digest = line.rstrip("\n").split("\t")
                rows[name] = (int(size), digest)
        tables.append((p, rows))
    names = sorted(set.union(*(set(r) for _, r in tables)))
    compared, bad = 0, []
    for name in names:
        if not name.startswith("ckpt_"):
            continue
        present = [(p, r[name]) for p, r in tables if name in r]
        if len(present) < 2:
            continue
        compared += 1
        if len({v for _, v in present}) != 1:
            bad.append("%s: %s" % (name, {p: v[1][:16] for p, v in present}))
    for b in bad:
        print("DISAGREE", b)
    print("checkpoints compared=%d disagreements=%d" % (compared, len(bad)))
    if compared == 0:
        print("NOTHING WAS COMPARED: the manifests share no checkpoint")
        return 1
    return 1 if bad else 0


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("recipe")
    r.add_argument("--out", required=True)
    r.add_argument("--shape", nargs=9, type=int, required=True, metavar=("B", "L", "DM", "H", "KV", "HD", "FF", "LAYERS", "V"),
                   help="V is replaced by the tokens' n_vocab")
    r.add_argument("--tokens", required=True, help="a tokens directory (tokens.i32 + manifest.json), or a manifest.json alone")
    r.add_argument("--shards", type=int, required=True, help="K logical shards per optimizer step")
    r.add_argument("--steps", type=int, required=True, help="optimizer steps in the whole run")
    r.add_argument("--seed", type=int, default=93261)
    r.add_argument("--peak-lr", type=float, required=True)
    r.add_argument("--warmup", type=int, required=True)
    r.add_argument("--final-ratio", type=float, default=0.1)
    r.add_argument("--beta1", type=float, default=0.9)
    r.add_argument("--beta2", type=float, default=0.95)
    r.add_argument("--eps", type=float, default=1e-8)
    r.add_argument("--weight-decay", type=float, default=0.1)
    r.add_argument("--checkpoint-every", type=int, default=100)
    r.add_argument("--boundaries", default="", help="comma-separated global steps ending each segment")
    r.add_argument("--hash-scheme", choices=(SCHEME_V1, SCHEME_V2), default=SCHEME_V1,
                   help="the chain's hash scheme for the WHOLE run (v2 is about five times faster at 162M)")

    i = sub.add_parser("init")
    i.add_argument("--recipe", required=True)
    i.add_argument("--tokens", required=True)
    i.add_argument("--out", required=True)

    k = sub.add_parser("keys")
    k.add_argument("--recipe", required=True)
    k.add_argument("--from-step", type=int, required=True)
    k.add_argument("--steps", type=int, required=True)
    k.add_argument("--boundary", type=int, default=None)

    ru = sub.add_parser("run")
    ru.add_argument("--recipe", required=True)
    ru.add_argument("--tokens", required=True)
    ru.add_argument("--from", dest="from_ckpt", required=True, help="the checkpoint to start from")
    ru.add_argument("--steps", type=int, required=True)
    ru.add_argument("--out", required=True)
    ru.add_argument("--devices", default="0")
    ru.add_argument("--route", default="A")
    ru.add_argument("--segment", default="1")
    ru.add_argument("--label", default=platform.node())
    ru.add_argument("--boundary", type=int, default=None, help="the global step ending this segment (writes step-2)")
    ru.add_argument("--expect-chain", default=None, help="hold every step to this chain's line; stop on the first difference")
    ru.add_argument("--upload-urls", default=None, help="JSON {file name: presigned PUT URL} for every expected key")
    ru.add_argument("--record-window", type=_window, action="append", default=[], metavar="A:B")
    ru.add_argument("--zero-moments", action="store_true", help="NEGATIVE CONTROL: restore with zeroed AdamW moments")
    ru.add_argument("--no-checkpoints", action="store_true", help="a replay: write the chain only")
    ru.add_argument("--control", default=None, metavar="none|shards=N|swap=A,B|split[=S]|ulp=S,I",
                    help="NEGATIVE CONTROL (see parse_control): stamped into every chain line and segment.json, "
                         "no checkpoints, no uploads")
    ru.add_argument("--live-role", choices=("coordinator", "worker"), default=None,
                    help="the multi-vendor segment: this box is one worker of a chained cross_vendor group")
    ru.add_argument("--live-shards", default=None, metavar="A:B", help="this box's contiguous block of shards")
    ru.add_argument("--live-port", type=int, default=7777, help="coordinator: the port to listen on")
    ru.add_argument("--live-workers", type=int, default=2, help="coordinator: workers in the group, this one included")
    ru.add_argument("--live-address", default=None, metavar="HOST:PORT", help="worker: the coordinator")
    ru.add_argument("--live-timeout", type=float, default=7200.0, help="seconds to wait for peers and for a step")
    ru.add_argument("--live-local-extra", default=None, metavar="A:B",
                    help="TEST ONLY: a second worker in this process, this block, sharing the device under a lock")

    c = sub.add_parser("compare")
    c.add_argument("chains", nargs="+")

    m = sub.add_parser("manifests")
    m.add_argument("manifests", nargs="+")

    args = ap.parse_args(argv)
    return dict(recipe=cmd_recipe, init=cmd_init, keys=cmd_keys, run=cmd_run, compare=cmd_compare,
                manifests=cmd_manifests)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
