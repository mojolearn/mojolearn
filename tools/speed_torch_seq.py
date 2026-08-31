# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE OPPONENT ARM for the sequence-model speed lane: torch on CUDA, and
`mamba-ssm`'s fused `selective_scan_cuda` where it is installable.

    python3 tools/speed_torch_seq.py --lane transformer
    python3 tools/speed_torch_seq.py --lane mamba --size smoke
    python3 tools/speed_torch_seq.py --lane attention --dump-dir /tmp/seqdump

**NOTHING IN THIS FILE HAS BEEN RUN.** It was written on 2026-08-25 by an
agent that was forbidden to execute anything. Treat the first invocation as
a debugging session, not as a measurement.

WHAT THIS IS FOR
================
`bench/speed/seq_speed_main.mojo` times the FAST (default, NOT
`-D MOJOLEARN_NUMERIC_IDENTICAL=1`) path of `transformer/` and `mamba/` on
one GPU. This file times what an NVIDIA user would actually run, on the SAME
shapes, with the SAME weights, printing the SAME line format, so the two
outputs can be concatenated and read as one table.

TF32 IS THE WHOLE ARGUMENT AND IT IS MEASURED BOTH WAYS
========================================================
On Ampere and later torch may satisfy an FP32 matmul with TF32 tensor cores:
ten explicit mantissa bits instead of twenty-three. It is much faster and it
is NOT FP32. Measured in this repository on an H100, the same GEMM ran at
44.4 TFLOP/s with `allow_tf32=False` and 207.5 TFLOP/s with it on -- about
5x. Every attention and MLP arm here is GEMM dominated, so a single
unlabelled torch number would be somewhere in a 5x band for reasons that
have nothing to do with either implementation.

So both are timed and both are reported as SEPARATE ARMS, with the setting
in the arm name: `torch-gpu-fp32` (allow_tf32 False) and `torch-gpu-tf32`
(allow_tf32 True). `tools/vendor_gemm_price.py` made the same call for the
same reason and its docstring is the longer argument.

THE ATTENTION BACKEND IS NAMED, NEVER CHOSEN SILENTLY
======================================================
Our port is EAGER: contract section 6 pins `eager_attention_forward` and
excludes FlashAttention, SDPA and paged attention, because an online
softmax's rescale count is the KV tile count, which is an execution-plan
quantity. So:

  * `torch-gpu-fp32` / `torch-gpu-tf32` are EAGER torch -- matmul, additive
    mask, explicit max/exp/sum/divide, matmul -- which is the apples-to-
    apples arm and the one `FSPEED-AGREE` is computed against.
  * `torch-gpu-sdpa-math-fp32`, `torch-gpu-sdpa-efficient-fp32` and
    `torch-gpu-flash-fp32` force ONE named SDPA backend each. A backend that
    cannot serve the dtype or the shape emits `FSPEED-REFUSED` and the run
    continues.
  * `torch-gpu-flash-bf16` is the same fused kernel at bfloat16, which is
    what a served model actually runs. **IT IS A DIFFERENT PRECISION AND A
    DIFFERENT ALGORITHM AND IT IS NAMED SO.** It is here because a report
    that omits it is not answering the question anybody asked, and it is
    separated because a report that conflates it with the FP32 arms is
    lying.

FlashAttention on CUDA does not accept FP32 inputs at all, so
`torch-gpu-flash-fp32` is EXPECTED to refuse. That refusal is a finding
worth printing, not an error to hide: it is the reason nobody deploys FP32
attention, and it is why our FP32 identity contract has no fused opponent.

THE MAMBA OPPONENT, AND WHICH ONE RAN
======================================
The real opponent is `mamba_ssm.ops.selective_scan_interface.selective_scan_fn`,
which dispatches to the fused `selective_scan_cuda` extension: arm
`mamba-ssm-cuda`. If that package is not importable, the fallback is the
PURE-PYTORCH sequential reference scan that `mamba/corpus/gen_corpus.py`
already contains verbatim (`selective_scan_ref`, a Python loop over the
sequence): arm `torch-ref-scan-gpu`. **A sequential PyTorch scan is not the
thing anyone deploys**, and every line says which one ran. If only
`torch-ref-scan-gpu` appears in an output, the mamba comparison has no
native opponent in it and the markdown must say so.

THE WEIGHTS ARE THE SAME ON BOTH SIDES
=======================================
Every tensor comes from the same hashed generator the Mojo driver uses:
`value = f32(lo + (hi-lo) * top24(splitmix64(key + i)) * 2^-24)` with
`key = splitmix64(seed ^ (tensor_id << 32))`. The implementation is
`mamba/corpus/gen_corpus.py::hashed_unit`, IMPORTED rather than re-spelled.
The tensor ids and ranges are the same ids and ranges the Mojo side uses,
and every tensor's WITNESS HASH is printed as `FSPEED-WEIGHTS` by both
sides. Two sides that agree on every witness agree on the generator. If a
witness disagrees, the run is void and no ratio from it means anything.

THE SHAPE TABLE IS PARSED, NOT COPIED. It is read out of
`bench/speed/seq_speed_main.mojo`'s `seq_shape_*` if-ladders, which are the
single source of truth, exactly as `tools/vendor_gemm_price.py` parses
`bench/gemm_shapes.mojo`. The parser refuses a line it does not recognize
rather than guessing, because a silently mis-parsed shape is the failure the
whole arrangement exists to avoid.

WHAT IS NOT COMPARABLE, IN ONE PLACE
=====================================
Read `bench/speed/SEQ_SPEED.md` before quoting any ratio from this. The
short list: our block entry validates every weight on the host on every
call and torch does not; our attention materializes seven score-sized
buffers and synchronizes per head and torch's does not; our GEMM is the
identity lane's pinned kernel with its pins compiled off, not a tuned one;
and torch's modules are one library call where ours are thirty.

Deviations: 1852 (the lean eager llama forward written here rather than
reusing `transformer/corpus/gen_corpus.py::block_forward`, which recomputes
the rotary tables in numpy on every call), 1855 (`map_range_fast`), 1856
(the corpus generators' import-time determinism switches are turned back
off here).
"""

import argparse
import os
import re
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRIVER_MOJO = os.path.join(REPO, "bench", "speed", "seq_speed_main.mojo")

FAM_LLAMA, FAM_MAMBA = 0, 1

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
M64 = (1 << 64) - 1

#: `seq_speed_main.mojo::WITNESS_SAMPLES`. The two must agree or every
#: witness disagrees at once, which is the loud failure and not the quiet one.
WITNESS_SAMPLES = 4096

#: Bytes of output this file is willing to run a byte-at-a-time FNV over per
#: round. FNV-1a64 is sequential by construction (`core/identity_trace.mojo`
#: says why a word-at-a-time variant is a DIFFERENT function), so a
#: whole-output hash of the 8.4 MB Llama-8B t512 row would cost several
#: seconds per round in Python. Over the budget the round prints `hash=-`,
#: which the line format explicitly allows, and an output witness is printed
#: instead so torch non-determinism is still visible.
FNV_BYTE_BUDGET = 1 << 18

#: `seq_speed_main.mojo::SEQ_SEED_BASE` and `TID_CTX_X`. TRANSCRIBED, and
#: the witness lines are what catch a transcription that has drifted.
SEQ_SEED_BASE = 0x53657153706564FF
TID_CTX_X = 20

#: `transformer/corpus/gen_corpus.py`, the pinned FP32 constants.
RMS_EPS = 1e-6
ROPE_THETA = 10000.0
MASK_FILL = -3.4028234663852886e38   # 0xFF7FFFFF, masking_utils.py:601-603
UNMASKED_FILL = 0.0                  # ADDED and may not be elided

#: `mamba/corpus/gen_corpus.py`.
MAMBA_EPS = 1e-5
D_STATE = 16
D_CONV = 4
EXPAND = 2

#: The ten llama tensor ids. Identical in
#: `transformer/corpus/gen_corpus.py::TENSOR_IDS` and in
#: `transformer/original/transformer_fixture.mojo::TID_*`.
LLAMA_TIDS = {
    "x": 1, "norm1.weight": 2, "norm2.weight": 3,
    "q_proj.weight": 4, "k_proj.weight": 5, "v_proj.weight": 6,
    "o_proj.weight": 7, "gate_proj.weight": 8, "up_proj.weight": 9,
    "down_proj.weight": 10,
}


# ---------------------------------------------------------------------------
# Parsing bench/speed/seq_speed_main.mojo, so there is ONE shape table
# ---------------------------------------------------------------------------
def _ladder(src, fn):
    """Evaluate one `seq_shape_*` if-ladder for every i, from the source.

    Modeled on `tools/vendor_gemm_price.py::_ladder`, which does the same
    job for `bench/gemm_shapes.mojo`. It is copied rather than imported
    because that one knows about `OP_NT`/`OP_TN`/`OP_NN` and this one must
    not; the shared part is thirty lines of regex and the coupling would be
    worse than the duplication. Both REFUSE an unrecognized line rather than
    guessing at it.
    """
    m = re.search(r"^def %s\(i: Int\)[^\n]*:\n(.*?)(?=\n\ndef |\n\n#|\n\ncomptime |\Z)"
                  % fn, src, re.S | re.M)
    if not m:
        raise SystemExit("speed_torch_seq: cannot find %s in %s" % (fn, DRIVER_MOJO))
    body = m.group(1)
    rules = []
    default = None
    in_doc = False
    for raw in body.split("\n"):
        line = raw.strip()
        if not line:
            continue
        if in_doc:
            if line.endswith('"""'):
                in_doc = False
            continue
        if line.startswith('"""'):
            # a one-line docstring closes on the same line
            if not (len(line) > 3 and line.endswith('"""')):
                in_doc = True
            continue
        if line.startswith("#"):
            continue
        eq = re.match(r"if i == (\d+):$", line)
        rng = re.match(r"if i >= (\d+) and i <= (\d+):$", line)
        ret = re.match(r"return (.+?)(?:\s*#.*)?$", line)
        if eq:
            rules.append(("pending", ("eq", int(eq.group(1)))))
        elif rng:
            rules.append(("pending", ("rng", int(rng.group(1)), int(rng.group(2)))))
        elif ret:
            val = ret.group(1).strip()
            if val.startswith("String("):
                val = val[len("String("):].rstrip(")").strip('"')
            else:
                val = int(val.replace("_", ""))
            if rules and rules[-1][0] == "pending":
                rules[-1] = (rules[-1][1], val)
            else:
                default = val
        else:
            raise SystemExit("speed_torch_seq: unparsed line in %s: %r" % (fn, line))
    if default is None:
        raise SystemExit("speed_torch_seq: %s has no trailing default return" % fn)

    def evaluate(i):
        for pred, val in rules:
            if pred[0] == "eq" and i == pred[1]:
                return val
            if pred[0] == "rng" and pred[1] <= i <= pred[2]:
                return val
        return default

    return evaluate


def load_shapes():
    src = open(DRIVER_MOJO).read()
    cnt = re.search(r"comptime SEQ_SHAPE_COUNT = (\d+)", src)
    if not cnt:
        raise SystemExit("speed_torch_seq: no SEQ_SHAPE_COUNT in %s" % DRIVER_MOJO)
    n = int(cnt.group(1))
    fields = ["family", "name", "b", "l", "ctx", "d_model", "n_heads",
              "n_kv", "head_dim", "intermediate", "smoke"]
    ev = {f: _ladder(src, "seq_shape_" + f) for f in fields}
    # The Mojo side spells the seed base and the context tensor id as
    # comptime constants; re-read them rather than trusting the copies above.
    base = re.search(r"comptime SEQ_SEED_BASE: UInt64 = (0x[0-9A-Fa-f]+)", src)
    tidc = re.search(r"comptime TID_CTX_X = (\d+)", src)
    wsam = re.search(r"comptime WITNESS_SAMPLES = (\d+)", src)
    if not (base and tidc and wsam):
        raise SystemExit("speed_torch_seq: SEQ_SEED_BASE / TID_CTX_X / "
                         "WITNESS_SAMPLES not found in %s" % DRIVER_MOJO)
    consts = dict(seed_base=int(base.group(1), 16),
                  tid_ctx_x=int(tidc.group(1)),
                  witness_samples=int(wsam.group(1)))
    # THE TRANSCRIBED COPIES ARE CHECKED AGAINST THE PARSED ONES AND THE
    # PROCESS DIES IF THEY DISAGREE. The constants at the top of this file are
    # documentation; the driver is the source of truth. A silent drift here
    # would produce two sides generating different weights while every other
    # line looked right, which is precisely the failure the witness hashes
    # exist to catch -- and this catches it a hundred times faster.
    if (consts["seed_base"] != SEQ_SEED_BASE
            or consts["tid_ctx_x"] != TID_CTX_X
            or consts["witness_samples"] != WITNESS_SAMPLES):
        raise SystemExit(
            "speed_torch_seq: this file's transcribed constants disagree with "
            "%s. parsed seed_base=%#x tid_ctx_x=%d witness_samples=%d; here "
            "%#x %d %d. Fix the copies at the top of this file."
            % (DRIVER_MOJO, consts["seed_base"], consts["tid_ctx_x"],
               consts["witness_samples"], SEQ_SEED_BASE, TID_CTX_X,
               WITNESS_SAMPLES))
    rows = [{f: ev[f](i) for f in fields} for i in range(n)]
    for i, r in enumerate(rows):
        r["i"] = i
    return rows, consts


# ---------------------------------------------------------------------------
# The hashed generator. `hashed_unit` is IMPORTED from the mamba corpus so
# there is one implementation of the spec on this side of the fence.
# ---------------------------------------------------------------------------
def _load_module(alias, path):
    """Load one `gen_corpus.py` under an EXPLICIT module name.

    BOTH corpora are files called `gen_corpus.py`. A plain
    `sys.path.insert` plus `import gen_corpus` loads whichever came first
    and then hands the SAME module out for the second import, because
    `sys.modules` is keyed by name -- so the transformer cross-check would
    silently be run against the mamba generator and would fail for a reason
    that has nothing to do with either. Distinct aliases, loaded by path.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(alias, path)
    if spec is None or spec.loader is None:
        raise ImportError("no loader for %s" % path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    spec.loader.exec_module(mod)
    return mod


try:
    mamba_corpus = _load_module(
        "mojolearn_mamba_corpus",
        os.path.join(REPO, "mamba", "corpus", "gen_corpus.py"))
except Exception as e:                          # pragma: no cover
    raise SystemExit("speed_torch_seq: cannot import mamba/corpus/gen_corpus.py "
                     "(needed for the hash spec and the reference scan): %s" % e)

import numpy as np                              # noqa: E402


def map_range_fast(f_unit, lo, hi):
    """`f32(lo + (hi - lo) * f)` in float64, rounded ONCE.

    DEVIATION 1855. `gen_corpus.py::map_range` does the same arithmetic and
    then asserts exactness ELEMENT BY ELEMENT against `fractions.Fraction`.
    That is right for a corpus of a few thousand values and impossible here:
    the Llama-8B `down_proj` alone is 58.7 million elements and the assert is
    a Python loop. The dyadic check on (lo, hi) is kept -- it is what makes
    the float64 evaluation exact in the first place -- and the per-element
    assert is applied to a bounded SAMPLE instead of to everything.
    """
    from fractions import Fraction
    assert mamba_corpus._is_dyadic_small(lo) and mamba_corpus._is_dyadic_small(hi), (lo, hi)
    span = float(hi) - float(lo)
    v64 = float(lo) + span * f_unit
    flo, fspan = Fraction(lo), Fraction(hi) - Fraction(lo)
    step = max(1, f_unit.size // 256)
    for j in range(0, f_unit.size, step):
        fv = float(f_unit[j])
        exact = flo + fspan * Fraction(fv)
        assert Fraction(float(v64[j])) == exact, (
            "float64 evaluation is not exact", lo, hi, fv, v64[j])
    return v64.astype(np.float32)


def gen(seed, tid, n, lo, hi, chunk=1 << 22):
    """One tensor of the spec, as a flat float32 numpy array, IN CHUNKS.

    `gen_corpus.py::hashed_unit` is the same three lines and is not called
    directly, for one reason: it materializes `np.arange(n, dtype=uint64)`
    and four uint64 temporaries of that size at once. At the Llama-8B
    `down_proj` row that is 58.7 million elements, so roughly 2.3 GB of
    peak host memory for ONE tensor of nine, and a rented box with 32 GB of
    RAM would be at real risk of the OOM killer taking the lease with it.
    The two primitives (`splitmix64_scalar` for the key, `splitmix64_array`
    for the body) ARE imported, so the spec still has one implementation
    here; only the loop around them is local. Part of DEVIATION 1855.
    """
    key = mamba_corpus.splitmix64_scalar((seed ^ (tid << 32)) & M64)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        idx = np.arange(s, e, dtype=np.uint64)
        with np.errstate(over="ignore"):
            h = mamba_corpus.splitmix64_array(np.uint64(key) + idx)
        f = (h >> np.uint64(40)).astype(np.float64) * (2.0 ** -24)
        out[s:e] = map_range_fast(f, lo, hi)
        del idx, h, f
    return out


def seq_seed(row_index, seed_base):
    return (seed_base + 0x1000 * row_index) & M64


def fnv1a64(buf):
    """FNV-1a64 over bytes, in order. `core/identity_trace.mojo`'s function."""
    h = FNV_OFFSET
    for byte in buf:
        h = ((h ^ byte) * FNV_PRIME) & M64
    return h


def hex16(v):
    return "%016x" % (v & M64)


def witness_hash(arr, samples):
    """`seq_speed_main.mojo::_witness_hash`, spelled the same way.

    The LENGTH is folded first, little endian, then a fixed strided sample of
    the elements' float32 bytes at stride `max(1, n // samples)`. A stride
    rather than a prefix so a transposition or a permutation still moves it.
    """
    a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    n = int(a.size)
    h = FNV_OFFSET
    for i in range(8):
        h = ((h ^ ((n >> (8 * i)) & 0xFF)) * FNV_PRIME) & M64
    stride = max(1, n // samples)
    return fnv1a64(a[::stride].tobytes())


def emit_witness(lane, tag, name, arr, samples):
    a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    print("FSPEED-WEIGHTS lane=%s shape=%s tensor=%s n=%d hash=%s"
          % (lane, tag, name, a.size, hex16(witness_hash(a, samples))))


# ---------------------------------------------------------------------------
# The device, named out loud, or nothing
# ---------------------------------------------------------------------------
def require_accelerator(torch):
    """The GPU torch can actually see, or a refusal. Never the CPU.

    DEVIATION 1936, 2026-08-28. This was `require_cuda` and it refused
    everything that was not CUDA or ROCm, so on Apple silicon the attention,
    mlp, rmsnorm, transformer, mamba and selective_scan lanes had NO OPPONENT
    AT ALL -- while `tools/speed_gemm_arm.py`, the sibling arm in the same
    family, has always driven `torch.mps` and produced an `mps-default` row
    on the same box in the same run. One of the two was simply never taught.

    THE REFUSAL ITSELF WAS ALWAYS RIGHT AND IS KEPT WORD FOR WORD for the
    case it was written for: a CPU forward timed here "would be a perfectly
    good number for the wrong device and nothing downstream could tell". MPS
    is not that case. It is the GPU on the box, it is the only GPU backend
    any of these opponents ship for Apple silicon, and on this project's own
    thesis it is the processor the comparison is about.

    Returns (name, build, is_hip, devstr). `devstr` is what the caller must
    build its `torch.device` from -- it is no longer safe to assume "cuda".
    """
    if torch.cuda.is_available():
        hip = getattr(torch.version, "hip", None)
        name = torch.cuda.get_device_name(0)
        build = ("ROCm " + str(hip)) if hip else ("CUDA " + str(getattr(torch.version, "cuda", "?")))
        return name, build, bool(hip), "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "Apple MPS", "MPS " + str(torch.__version__), False, "mps"
    raise SystemExit(
        "speed_torch_seq: REFUSED. No CUDA and no MPS device visible to torch.\n"
        "A CPU forward timed here would be a perfectly good number for the wrong\n"
        "device and nothing downstream could tell. torch %s" % torch.__version__)


#: The device string the arms are running on, set once by `main` from
#: `require_accelerator`. A module global rather than a parameter because the
#: timing helper below is called from a dozen lanes and threading it through
#: every one of them would be a wider edit than the defect deserves; it is
#: written exactly once, before any arm runs.
_DEVSTR = "cuda"


def sync(torch, devstr):
    """Drain the device the arm is actually on.

    An unsynchronized timing measures the enqueue, not the work, and
    `torch.cuda.synchronize()` is a silent no-op when the tensors are on MPS.
    """
    if devstr == "cuda":
        torch.cuda.synchronize()
    elif devstr == "mps":
        torch.mps.synchronize()


def set_tf32(torch, on):
    """Set every TF32 switch this torch has, explicitly, both ways.

    Never left at the default: torch's default has moved between releases and
    a benchmark whose precision depends on which wheel got installed is not a
    benchmark. The newer `fp32_precision` spelling is set when present and
    the old `allow_tf32` booleans when they are.
    """
    touched = []
    try:
        torch.backends.cuda.matmul.allow_tf32 = bool(on)
        touched.append("cuda.matmul.allow_tf32")
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = bool(on)
        touched.append("cudnn.allow_tf32")
    except Exception:
        pass
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32" if on else "ieee"
        touched.append("cuda.matmul.fp32_precision")
    except Exception:
        pass
    return touched


# ---------------------------------------------------------------------------
# The llama opponent. DEVIATION 1852: written here rather than reusing
# `transformer/corpus/gen_corpus.py::block_forward`, which rebuilds the rotary
# tables in NUMPY on the host on every call. That is correct for a corpus and
# fatal for a timing harness: at L=512 it is tens of thousands of float64
# trig evaluations on the CPU inside what is supposed to be a GPU
# measurement. The steps below are that function's steps, in its order, with
# the tables hoisted; `--crosscheck` compares the two at the small row.
# ---------------------------------------------------------------------------
class LlamaEager:
    def __init__(self, torch, dev, cfg, W, dtype):
        self.t = torch
        self.dev = dev
        self.cfg = cfg
        self.dtype = dtype
        self.W = {k: v.to(dtype) for k, v in W.items()}
        hd = cfg["head_dim"]
        p_max = cfg["ctx"] + cfg["l"]
        half = hd // 2
        # LRE:108 inv_freq = 1 / (base ** (arange(0, dim, 2) / dim)), FP32.
        e32 = (np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd)).astype(np.float32)
        inv32 = (np.float32(1.0) / (np.float32(ROPE_THETA) ** e32)).astype(np.float32)
        pos = np.arange(p_max, dtype=np.float32)
        ang = (pos[:, None] * inv32[None, :]).astype(np.float32)
        cos = torch.from_numpy(np.cos(ang)).to(dev).to(dtype)
        sin = torch.from_numpy(np.sin(ang)).to(dev).to(dtype)
        # LRE:123 emb = cat((freqs, freqs)) -- a COPY, materialized once here
        self.cos = torch.cat((cos, cos), dim=-1)   # [p_max, hd]
        self.sin = torch.cat((sin, sin), dim=-1)
        self.eps = float(RMS_EPS)
        # Contract S-constant / DEVIATION 802: head_dim ** -0.5 in FP32.
        self.scale = float(np.float32(1.0) / np.sqrt(np.float32(hd)))
        self.half = half

    def _rms(self, x, w):
        # LRN:62-67, the reference's `variance` is the MEAN of the squares.
        var = (x * x).sum(-1, keepdim=True) / x.shape[-1]
        return w * (x * self.t.rsqrt(var + self.eps))

    def _rot_half(self, x):
        h = x.shape[-1] // 2
        return self.t.cat((-x[..., h:], x[..., :h]), dim=-1)

    def project(self, h, B, L):
        """q/k/v, RoPE'd, head-major. Shared by the eager and SDPA arms."""
        F = self.t.nn.functional
        c = self.cfg
        H, HKV, hd = c["n_heads"], c["n_kv"], c["head_dim"]
        q = F.linear(h, self.W["q_proj.weight"]).reshape(B, L, H, hd).transpose(1, 2)
        k = F.linear(h, self.W["k_proj.weight"]).reshape(B, L, HKV, hd).transpose(1, 2)
        v = F.linear(h, self.W["v_proj.weight"]).reshape(B, L, HKV, hd).transpose(1, 2)
        p0 = c["ctx"]
        cos = self.cos[p0:p0 + L].unsqueeze(0).unsqueeze(0)
        sin = self.sin[p0:p0 + L].unsqueeze(0).unsqueeze(0)
        q = q * cos + self._rot_half(q) * sin
        k = k * cos + self._rot_half(k) * sin
        return q, k, v

    def attention_eager(self, q, kfull, vfull, B, L):
        """EAF:204-210 spelled out, the mask ADDITIVE and by ABSOLUTE
        position, exactly as `eager_attention_forward` does it. This is the
        arm our port is comparable with."""
        c = self.cfg
        H, HKV, hd = c["n_heads"], c["n_kv"], c["head_dim"]
        n_rep = H // HKV
        S = kfull.shape[2]
        idx = self.t.arange(H, device=self.dev) // n_rep
        krep = kfull.index_select(1, idx)
        vrep = vfull.index_select(1, idx)
        scores = self.t.matmul(q, krep.transpose(2, 3)) * self.scale
        qp = self.t.arange(c["ctx"], c["ctx"] + L, device=self.dev).view(L, 1)
        kp = self.t.arange(S, device=self.dev).view(1, S)
        allowed = kp <= qp
        fill = self.t.where(
            allowed,
            self.t.tensor(UNMASKED_FILL, dtype=self.dtype, device=self.dev),
            self.t.tensor(MASK_FILL, dtype=self.dtype, device=self.dev))
        masked = scores + fill
        mx = masked.max(-1, keepdim=True).values
        e = self.t.exp(masked - mx)
        w = e / e.sum(-1, keepdim=True)
        ctxv = self.t.matmul(w, vrep)
        return ctxv.transpose(1, 2).reshape(B * L, H * hd)

    def attention_sdpa(self, q, kfull, vfull, B, L):
        """`F.scaled_dot_product_attention` -- a DIFFERENT ALGORITHM from the
        eager arm whenever the backend fuses the softmax, which is the point
        of measuring it separately."""
        c = self.cfg
        H, HKV, hd = c["n_heads"], c["n_kv"], c["head_dim"]
        n_rep = H // HKV
        S = kfull.shape[2]
        idx = self.t.arange(H, device=self.dev) // n_rep
        krep = kfull.index_select(1, idx)
        vrep = vfull.index_select(1, idx)
        qp = self.t.arange(c["ctx"], c["ctx"] + L, device=self.dev).view(L, 1)
        kp = self.t.arange(S, device=self.dev).view(1, S)
        mask = (kp <= qp).view(1, 1, L, S)
        o = self.t.nn.functional.scaled_dot_product_attention(
            q, krep, vrep, attn_mask=mask, scale=self.scale)
        return o.transpose(1, 2).reshape(B * L, H * hd)

    def mlp(self, h2):
        F = self.t.nn.functional
        g = F.linear(h2, self.W["gate_proj.weight"])
        u = F.linear(h2, self.W["up_proj.weight"])
        # LMLP:175 ACT2FN["silu"]; contract S20 pins ATen's ONE-division form.
        one = self.t.ones((), dtype=self.dtype, device=self.dev)
        s = g / (one + self.t.exp(-g))
        return F.linear(s * u, self.W["down_proj.weight"])

    def block(self, x, kv, B, L, sdpa=False):
        """One decoder layer. `kv` is `(k, v)` for the prior context or None.

        Returns `(out, o_proj, down_proj, norm1_out, norm2_out)` so every
        sub-lane can take its own output, and its own INPUT, from one
        implementation instead of four.

        `norm2_out` is returned and not recomputed by the mlp lane, and that
        is not tidiness. `llama_mlp_forward` on our side reads
        `stages.norm2_out`, which is `rms(x + o_proj(attn), norm2.weight)`.
        An opponent that fed the mlp `rms(x, norm2.weight)` instead would
        have the same shape, the same cost and a completely different
        answer, and `FSPEED-AGREE` would report a large difference for a
        harness bug rather than for anything about either implementation.
        """
        F = self.t.nn.functional
        h = self._rms(x, self.W["norm1.weight"])
        q, k, v = self.project(h, B, L)
        if kv is not None:
            kfull = self.t.cat((kv[0], k), dim=2)
            vfull = self.t.cat((kv[1], v), dim=2)
        else:
            kfull, vfull = k, v
        ctxv = (self.attention_sdpa(q, kfull, vfull, B, L) if sdpa
                else self.attention_eager(q, kfull, vfull, B, L))
        o = F.linear(ctxv, self.W["o_proj.weight"])
        r1 = x + o
        h2 = self._rms(r1, self.W["norm2.weight"])
        dn = self.mlp(h2)
        return r1 + dn, o, dn, h, h2


# ---------------------------------------------------------------------------
# The timing harness. One warm-up set, then N timed rounds, every one with an
# explicit `torch.cuda.synchronize()`. An unsynchronized timing measures the
# enqueue rate and nothing else.
# ---------------------------------------------------------------------------
def time_arm(torch, lane, arm, tag, call, rounds, warmups, out_of):
    """Run `call` and print its warm-up, its rounds and its hash lines.

    `out_of` maps the call's return value to the tensor the round is hashed
    and compared on. Returns that tensor from the LAST round, for
    `FSPEED-AGREE`.

    TORCH NEEDS MORE THAN ONE WARM-UP and this file uses five, because lazy
    module init, cuBLAS handle creation and the autotuner all land on the
    first call and some of them land on the second. `bench/lanes_price_main.mojo`
    uses one on our side because a Mojo kernel has no autotune cache.
    """
    for _ in range(warmups):
        r = call()
    sync(torch, _DEVSTR)
    t0 = time.perf_counter()
    r = call()
    sync(torch, _DEVSTR)
    print("FSPEED-WARMUP lane=%s arm=%s shape=%s ms=%.6f"
          % (lane, arm, tag, (time.perf_counter() - t0) * 1000.0))
    last = None
    hashes = []
    for i in range(1, rounds + 1):
        t0 = time.perf_counter()
        r = call()
        sync(torch, _DEVSTR)
        ms = (time.perf_counter() - t0) * 1000.0
        last = out_of(r)
        a = last.detach().to(torch.float32).contiguous().cpu().numpy()
        nbytes = a.size * 4
        if nbytes <= FNV_BYTE_BUDGET:
            h = hex16(fnv1a64(a.tobytes()))
            hashes.append(h)
        else:
            h = "-"
        print("FSPEED lane=%s arm=%s shape=%s round=%d ms=%.6f hash=%s"
              % (lane, arm, tag, i, ms, h))
    if len(hashes) > 1 and len(set(hashes)) > 1:
        print("FSPEED-NOTE lane=%s arm=%s hash moved across rounds: %s %s"
              % (lane, arm, hashes[0], next(x for x in hashes if x != hashes[0])))
    if last is not None:
        # An OUTPUT WITNESS, always, including when the round hash printed
        # `-` because the output was over `FNV_BYTE_BUDGET`. Without it the
        # big rows would have no fingerprint at all, and "the arm produced
        # nothing detectable" would look exactly like "the arm produced the
        # right thing". Same strided rule as the weight witness.
        a = last.detach().to(torch.float32).contiguous().cpu().numpy()
        note(lane, arm, "output witness shape=%s n=%d hash=%s"
             % (tag, a.size, hex16(witness_hash(a, WITNESS_SAMPLES))))
    return last


def undeterminize(torch):
    """Put torch's global determinism switches back where a benchmark needs
    them. DEVIATION 1856.

    BOTH `gen_corpus.py` files call `torch.use_deterministic_algorithms(True)`
    and `torch.set_num_threads(1)` AT IMPORT TIME. That is right for a corpus
    generator and wrong here twice over: deterministic algorithms select
    slower kernels, and on CUDA a deterministic matmul RAISES unless
    `CUBLAS_WORKSPACE_CONFIG` is set. This is the FAST arm on both sides. It
    is called after EVERY import of a corpus, not once at startup, because
    the transformer corpus is imported lazily by the cross-check and would
    otherwise silently re-arm determinism for every timed round after it.
    """
    try:
        torch.use_deterministic_algorithms(False)
        torch.set_num_threads(os.cpu_count() or 1)
    except Exception:
        pass


def one_line(s):
    """Collapse a message to ONE line.

    `FSPEED-REFUSED ... reason=<one line>` is a one-line contract and a
    torch exception is routinely several, including a whole CUDA backtrace.
    A multi-line reason would put unparseable text into the middle of the
    table."""
    return " ".join(str(s).split())


def refuse(lane, arm, reason):
    print("FSPEED-REFUSED lane=%s arm=%s reason=%s" % (lane, arm, one_line(reason)))


def note(lane, arm, text):
    print("FSPEED-NOTE lane=%s arm=%s %s" % (lane, arm, one_line(text)))


# ---------------------------------------------------------------------------
# FSPEED-AGREE against the Mojo dump
# ---------------------------------------------------------------------------
def agree(lane, tag, ours_path, theirs):
    """Compare the Mojo driver's dumped output with the eager FP32 arm's.

    A REPORT LINE AND NOT A GATE. Two implementations of the same block in
    FP32 will not agree bitwise and are not expected to; what this catches is
    the failure where a speed number was taken for a block that computes
    something else entirely, which is the failure that makes a whole
    benchmark worthless without ever looking wrong.
    """
    if not ours_path or not os.path.exists(ours_path):
        refuse(lane, "agree", "no mojo dump at %s (set MOJOLEARN_SPEED_DUMP_DIR "
                              "on the mojo side and --dump-dir here)" % ours_path)
        return
    ours = np.fromfile(ours_path, dtype=np.float32)
    th = np.ascontiguousarray(theirs, dtype=np.float32).reshape(-1)
    if ours.size != th.size:
        refuse(lane, "agree", "shape %s: mojo dumped %d floats, torch produced %d"
               % (tag, ours.size, th.size))
        return
    d = np.abs(ours.astype(np.float64) - th.astype(np.float64))
    den = np.maximum(np.abs(th.astype(np.float64)), 1e-30)
    print("FSPEED-AGREE lane=%s max_abs_diff=%.6g max_rel_diff=%.6g n=%d"
          % (lane, float(d.max()) if d.size else 0.0,
             float((d / den).max()) if d.size else 0.0, ours.size))


# ---------------------------------------------------------------------------
# The llama lanes
# ---------------------------------------------------------------------------
def llama_weights(torch, dev, row, seed, samples, lane, tag):
    dm = row["d_model"]
    H, HKV, hd, it = row["n_heads"], row["n_kv"], row["head_dim"], row["intermediate"]
    qw, kw = H * hd, HKV * hd
    s_o = mamba_corpus.fan_in_scale(qw)
    s_d = mamba_corpus.fan_in_scale(it)
    spec = [
        ("norm1.weight", dm, 0.5, 1.5, (dm,)),
        ("norm2.weight", dm, 0.5, 1.5, (dm,)),
        ("q_proj.weight", qw * dm, -0.5, 0.5, (qw, dm)),
        ("k_proj.weight", kw * dm, -0.5, 0.5, (kw, dm)),
        ("v_proj.weight", kw * dm, -0.5, 0.5, (kw, dm)),
        ("o_proj.weight", dm * qw, -s_o, s_o, (dm, qw)),
        ("gate_proj.weight", it * dm, -0.25, 0.25, (it, dm)),
        ("up_proj.weight", it * dm, -0.25, 0.25, (it, dm)),
        ("down_proj.weight", dm * it, -s_d, s_d, (dm, it)),
    ]
    W = {}
    for name, n, lo, hi, shape in spec:
        flat = gen(seed, LLAMA_TIDS[name], n, lo, hi)
        emit_witness(lane, tag, name, flat, samples)
        W[name] = torch.from_numpy(flat.reshape(shape)).to(dev)
    x = gen(seed, LLAMA_TIDS["x"], row["b"] * row["l"] * dm, -2.0, 2.0)
    emit_witness(lane, tag, "x", x, samples)
    xt = torch.from_numpy(x.reshape(row["b"] * row["l"], dm)).to(dev)
    xc = None
    if row["ctx"] > 0:
        cx = gen(seed, TID_CTX_X, row["b"] * row["ctx"] * dm, -2.0, 2.0)
        emit_witness(lane, tag, "ctx.x", cx, samples)
        xc = torch.from_numpy(cx.reshape(row["b"] * row["ctx"], dm)).to(dev)
    return W, xt, xc


def run_llama_row(torch, dev, lane, row, args, consts):
    tag = row["name"]
    B, L, ctxlen, dm = row["b"], row["l"], row["ctx"], row["d_model"]
    seed = seq_seed(row["i"], consts["seed_base"])
    samples = consts["witness_samples"]
    W, x, xc = llama_weights(torch, dev, row, seed, samples, lane, tag)
    cfg = dict(n_heads=row["n_heads"], n_kv=row["n_kv"], head_dim=row["head_dim"],
               intermediate=row["intermediate"], d_model=dm, ctx=ctxlen, l=L)

    dumped = os.path.join(args.dump_dir, "seq.%s.%s.f32.bin" % (lane, tag)) if args.dump_dir else ""
    eager_out = None

    for arm, tf32, dtype, backend in _llama_arms(torch, lane, args):
        try:
            set_tf32(torch, tf32)
            m = LlamaEager(torch, dev, cfg, W, dtype)
            xin = x.to(dtype)
            kv = None
            if ctxlen > 0:
                # The prior context, built the same way the Mojo side builds
                # it: by RUNNING the block on `ctx` tokens. Untimed.
                mc = LlamaEager(torch, dev, dict(cfg, l=ctxlen, ctx=0), W, dtype)
                hc = mc._rms(xc.to(dtype), mc.W["norm1.weight"])
                _, kc, vc = mc.project(hc, B, ctxlen)
                kv = (kc, vc)
            sdpa = backend is not None

            if lane == "transformer":
                call = lambda: m.block(xin, kv, B, L, sdpa=sdpa)
                pick = lambda r: r[0]
            elif lane == "attention":
                h = m._rms(xin, m.W["norm1.weight"])

                def call(m=m, h=h, kv=kv, sdpa=sdpa):
                    q, k, v = m.project(h, B, L)
                    kf = torch.cat((kv[0], k), dim=2) if kv is not None else k
                    vf = torch.cat((kv[1], v), dim=2) if kv is not None else v
                    c = (m.attention_sdpa(q, kf, vf, B, L) if sdpa
                         else m.attention_eager(q, kf, vf, B, L))
                    return torch.nn.functional.linear(c, m.W["o_proj.weight"])
                pick = lambda r: r
            elif lane == "mlp":
                # The REAL mlp input: norm2(x + o_proj(attn)), taken from one
                # untimed whole-block call, which is what our port's
                # `stages.norm2_out` holds when `llama_mlp_forward` reads it.
                h2 = m.block(xin, kv, B, L, sdpa=sdpa)[4]
                call = lambda m=m, h2=h2: m.mlp(h2)
                pick = lambda r: r
            else:  # rmsnorm
                call = lambda m=m: m._rms(xin, m.W["norm1.weight"])
                pick = lambda r: r

            if backend is None:
                out = time_arm(torch, lane, arm, tag, call, args.rounds, args.warmups, pick)
            else:
                with _sdpa_ctx(torch, backend):
                    out = time_arm(torch, lane, arm, tag, call, args.rounds, args.warmups, pick)
            if arm == "torch-gpu-fp32":
                eager_out = out
        except Exception as e:
            refuse(lane, arm, "%s at shape %s: %s" % (type(e).__name__, tag, str(e)[:180]))

    if eager_out is not None:
        agree(lane, tag, dumped, eager_out.detach().to(torch.float32).contiguous().cpu().numpy())
    else:
        refuse(lane, "agree", "the eager fp32 arm did not produce an output at "
                              "shape %s, so nothing can be compared" % tag)


def _llama_arms(torch, lane, args):
    """(arm name, tf32, dtype, sdpa backend or None).

    The eager arms come first so `FSPEED-AGREE` has an output even if a
    fused backend takes the process down a path that refuses."""
    arms = [("torch-gpu-fp32", False, torch.float32, None),
            ("torch-gpu-tf32", True, torch.float32, None)]
    if lane in ("transformer", "attention"):
        arms += [
            ("torch-gpu-sdpa-math-fp32", False, torch.float32, "math"),
            ("torch-gpu-sdpa-efficient-fp32", False, torch.float32, "efficient"),
            # EXPECTED TO REFUSE: FlashAttention on CUDA does not take FP32.
            # The refusal is the finding; see this file's docstring.
            ("torch-gpu-flash-fp32", False, torch.float32, "flash"),
        ]
        if args.bf16:
            arms += [("torch-gpu-flash-bf16", False, torch.bfloat16, "flash")]
    return arms


def _sdpa_ctx(torch, which):
    """Force ONE named SDPA backend, or fail loudly. Never let torch choose:
    an unlabelled `scaled_dot_product_attention` number is a number for
    whichever kernel that wheel happened to prefer."""
    from torch.nn.attention import SDPBackend, sdpa_kernel
    m = {"math": SDPBackend.MATH,
         "efficient": SDPBackend.EFFICIENT_ATTENTION,
         "flash": SDPBackend.FLASH_ATTENTION}
    return sdpa_kernel(m[which])


# ---------------------------------------------------------------------------
# The mamba lanes
# ---------------------------------------------------------------------------
def mamba_params(torch, dev, row, seed, samples, lane, tag):
    dm = row["d_model"]
    di = EXPAND * dm
    dt_rank = (dm + 15) // 16
    ranges = mamba_corpus.default_ranges(dm, di, dt_rank, D_STATE, D_CONV)
    shapes = mamba_corpus.shapes_for(dm, di, dt_rank, D_STATE, D_CONV, row["b"], row["l"])
    P = {}
    for name in ["norm.weight", "in_proj.weight", "conv1d.weight", "conv1d.bias",
                 "x_proj.weight", "dt_proj.weight", "dt_proj.bias", "A_log",
                 "D", "out_proj.weight"]:
        shape = shapes[name]
        n = int(np.prod(shape))
        lo, hi = ranges[name]
        flat = gen(seed, mamba_corpus.TENSOR_IDS[name], n, lo, hi)
        emit_witness(lane, tag, name, flat, samples)
        P[name] = torch.from_numpy(flat.reshape(shape)).to(dev)
    xn = gen(seed, mamba_corpus.TENSOR_IDS["x"], row["b"] * row["l"] * dm, -2.0, 2.0)
    emit_witness(lane, tag, "x", xn, samples)
    x = torch.from_numpy(xn.reshape(row["b"], row["l"], dm)).to(dev)
    return P, x, di, dt_rank


def mamba_prefix(torch, P, x, di, dt_rank):
    """Everything before the scan, as `selective_scan` needs it.

    Returns `(u, delta, A, Bt, Ct, D, gate)` with `u`, `delta` in
    `[B, d_inner, L]` and `delta` ALREADY SOFTPLUSED, because our
    `selective_scan_fn` REFUSES `delta_bias` and `delta_softplus`
    (DEVIATION 723: seam S14 belongs to the block). The opponent must be
    called the same way or it is doing more work than we are.
    """
    F = torch.nn.functional
    L = x.shape[1]
    h = mamba_corpus.rmsnorm(x, P["norm.weight"], MAMBA_EPS)
    proj = F.linear(h, P["in_proj.weight"]).transpose(1, 2)
    A = -torch.exp(P["A_log"])
    hs, gate = proj.chunk(2, dim=1)
    conv = F.conv1d(hs, P["conv1d.weight"], P["conv1d.bias"],
                    padding=D_CONV - 1, groups=di)[:, :, :L]
    u = F.silu(conv)
    xdbl = F.linear(u.transpose(1, 2), P["x_proj.weight"])
    dt_low, Bm, Cm = torch.split(xdbl, [dt_rank, D_STATE, D_STATE], dim=-1)
    dt_raw = torch.matmul(P["dt_proj.weight"], dt_low.transpose(1, 2))
    delta = F.softplus(dt_raw + P["dt_proj.bias"][None, :, None])
    return u, delta, A, Bm.transpose(1, 2), Cm.transpose(1, 2), P["D"], gate


def run_mamba_row(torch, dev, lane, row, args, consts):
    tag = row["name"]
    B, L, dm = row["b"], row["l"], row["d_model"]
    seed = seq_seed(row["i"], consts["seed_base"])
    samples = consts["witness_samples"]
    P, x, di, dt_rank = mamba_params(torch, dev, row, seed, samples, lane, tag)
    dumped = os.path.join(args.dump_dir, "seq.%s.%s.f32.bin" % (lane, tag)) if args.dump_dir else ""

    fused = None
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as fused
    except Exception as e:
        refuse(lane, "mamba-ssm-cuda",
               "mamba_ssm is not importable (%s). The only opponent left is the "
               "PURE-PYTORCH sequential reference scan, which is NOT what anyone "
               "deploys." % str(e)[:120])

    ref_out = None
    for arm, tf32 in (("torch-ref-scan-gpu", False), ("torch-ref-scan-gpu-tf32", True)):
        try:
            set_tf32(torch, tf32)
            if lane == "mamba":
                call = lambda: mamba_corpus.block_forward(P, x, torch.float32)
                pick = lambda r: r["block.out"]
            else:
                u, delta, A, Bt, Ct, Dv, gate = mamba_prefix(torch, P, x, di, dt_rank)

                def call(u=u, delta=delta, A=A, Bt=Bt, Ct=Ct, Dv=Dv):
                    with mamba_corpus._scan_ref_dtype(torch.float32):
                        return mamba_corpus.selective_scan_ref(
                            u, delta, A, Bt, Ct, D=Dv, z=None, delta_bias=None,
                            delta_softplus=False, return_last_state=True)
                # THE LAYOUTS DIFFER AND THE TRANSPOSE IS NOT COSMETIC. The
                # scan reference returns `[B, d_inner, L]`; our
                # `selective_scan_fn` writes `skip_out` as `[M, d_inner]`
                # token-major (`selective_scan_interface.mojo:569`). Compared
                # flat without this, `FSPEED-AGREE` would report a huge
                # difference between two implementations that agree.
                pick = lambda r: r[0].transpose(1, 2).contiguous()
            out = time_arm(torch, lane, arm, tag, call, args.rounds, args.warmups, pick)
            if arm == "torch-ref-scan-gpu":
                ref_out = out
        except Exception as e:
            refuse(lane, arm, "%s at shape %s: %s" % (type(e).__name__, tag, str(e)[:180]))

    if fused is not None:
        arm = "mamba-ssm-cuda"
        try:
            set_tf32(torch, False)
            u, delta, A, Bt, Ct, Dv, gate = mamba_prefix(torch, P, x, di, dt_rank)
            if lane == "mamba":
                # The block with the fused scan in it. NOTE that this arm
                # folds the gate INTO the scan (`z=gate`), which is what a
                # deployment does and which our port refuses to do because
                # seam S12 is a recorded stage of its own (DEVIATION 723).
                # So this arm does the same MATH in fewer kernels, and that
                # difference is part of what is being measured.
                def call(u=u, delta=delta, A=A, Bt=Bt, Ct=Ct, Dv=Dv, gate=gate):
                    g = fused(u, delta, A, Bt, Ct, Dv, z=gate, delta_bias=None,
                              delta_softplus=False, return_last_state=False)
                    o = torch.nn.functional.linear(g.transpose(1, 2), P["out_proj.weight"])
                    return x + o
                pick = lambda r: r
                note(lane, arm, "this arm fuses the z gate into selective_scan_cuda; "
                                "our port keeps S12 as its own kernel (DEVIATION 723)")
            else:
                def call(u=u, delta=delta, A=A, Bt=Bt, Ct=Ct, Dv=Dv):
                    return fused(u, delta, A, Bt, Ct, Dv, z=None, delta_bias=None,
                                 delta_softplus=False, return_last_state=True)
                pick = lambda r: r[0].transpose(1, 2).contiguous()
            time_arm(torch, lane, arm, tag, call, args.rounds, args.warmups, pick)
        except Exception as e:
            refuse(lane, arm, "%s at shape %s: %s" % (type(e).__name__, tag, str(e)[:180]))

    if ref_out is not None:
        agree(lane, tag, dumped, ref_out.detach().to(torch.float32).contiguous().cpu().numpy())
    else:
        refuse(lane, "agree", "the reference arm did not produce an output at "
                              "shape %s" % tag)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
LANES = ("transformer", "attention", "mlp", "rmsnorm", "mamba", "selective_scan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, choices=LANES)
    ap.add_argument("--rounds", type=int,
                    default=int(os.environ.get("MOJOLEARN_SPEED_ROUNDS", "10")))
    ap.add_argument("--warmups", type=int, default=5,
                    help="untimed torch warm-ups per arm; more than one because "
                         "lazy module init and cuBLAS handle creation land on the "
                         "first call and the autotuner can land on the second")
    ap.add_argument("--size", default=os.environ.get("MOJOLEARN_SPEED_SIZE", "shipped"),
                    choices=("shipped", "smoke"))
    ap.add_argument("--row", type=int, default=-1)
    ap.add_argument("--dump-dir", default=os.environ.get("MOJOLEARN_SPEED_DUMP_DIR", ""))
    ap.add_argument("--bf16", action="store_true", default=True,
                    help="also time the bfloat16 flash arm (a DIFFERENT precision)")
    ap.add_argument("--no-bf16", dest="bf16", action="store_false")
    ap.add_argument("--crosscheck", action="store_true", default=True)
    ap.add_argument("--no-crosscheck", dest="crosscheck", action="store_false")
    args = ap.parse_args()

    try:
        import torch
    except ImportError:
        raise SystemExit("speed_torch_seq: REFUSED. torch is not importable.")

    undeterminize(torch)   # DEVIATION 1856; see the function's docstring
    name, build, is_hip, devstr = require_accelerator(torch)
    global _DEVSTR
    _DEVSTR = devstr
    rows, consts = load_shapes()
    lane = args.lane
    fam = FAM_MAMBA if lane in ("mamba", "selective_scan") else FAM_LLAMA
    dev = torch.device(devstr)

    print("FSPEED-HEADER family=seq lane=%s arm=torch mode=FAST device=%s "
          "rounds=%d size=%s" % (lane, name.replace(" ", "_"), args.rounds, args.size))
    note(lane, "torch", "build=%s torch=%s tf32_switches=%s deterministic=off"
         % (build, torch.__version__, ",".join(set_tf32(torch, False)) or "none"))
    if is_hip:
        note(lane, "torch", "this is a ROCm build; the arm names still say cuda "
                            "because torch's device does. TF32 on CDNA3 is XF32 "
                            "and the same switch reaches it.")

    if args.crosscheck and fam == FAM_LLAMA:
        _crosscheck_llama(torch, dev, rows, consts, lane)
        undeterminize(torch)   # the cross-check imported a corpus; re-arm off

    for row in rows:
        if row["family"] != fam:
            continue
        if args.row >= 0 and row["i"] != args.row:
            continue
        if args.size == "smoke" and row["smoke"] != 1:
            continue
        try:
            if fam == FAM_LLAMA:
                run_llama_row(torch, dev, lane, row, args, consts)
            else:
                run_mamba_row(torch, dev, lane, row, args, consts)
        except Exception as e:
            refuse(lane, "torch", "row %d %s: %s: %s"
                   % (row["i"], row["name"], type(e).__name__, str(e)[:180]))
    print("FSPEED-DONE lane=%s arm=torch" % lane)
    return 0


def _crosscheck_llama(torch, dev, rows, consts, lane):
    """`LlamaEager` against `transformer/corpus/gen_corpus.py::block_forward`
    on the smallest row, in float64 on the CPU.

    DEVIATION 1852 is the decision to write a second eager forward here; this
    is the price of it. Without this check the two can drift and the drift is
    invisible, because a wrong-but-plausible opponent still prints
    milliseconds. It runs on the corpus-shaped row only, where the whole
    thing is a few hundred floats.
    """
    small = [r for r in rows if r["family"] == FAM_LLAMA and r["smoke"] == 1]
    if not small:
        note(lane, "torch", "crosscheck SKIPPED: no smoke-sized llama row")
        return
    row = small[0]
    try:
        xf_corpus = _load_module(
            "mojolearn_transformer_corpus",
            os.path.join(REPO, "transformer", "corpus", "gen_corpus.py"))
        undeterminize(torch)
    except Exception as e:
        note(lane, "torch", "crosscheck REFUSED: transformer/corpus/gen_corpus.py "
                            "not importable (%s)" % str(e)[:120])
        return
    try:
        seed = seq_seed(row["i"], consts["seed_base"])
        cpu = torch.device("cpu")
        W, x, _ = llama_weights(torch, cpu, row, seed, consts["witness_samples"],
                                lane + ".crosscheck", row["name"])
        B, L, dm = row["b"], row["l"], row["d_model"]
        cfg = dict(n_heads=row["n_heads"], n_kv=row["n_kv"], head_dim=row["head_dim"],
                   intermediate=row["intermediate"], d_model=dm, ctx=0, l=L)
        mine = LlamaEager(torch, cpu, cfg, W, torch.float64).block(
            x.to(torch.float64), None, B, L)[0]
        ccfg = dict(B_L=(B, L), d_model=dm, n_heads=row["n_heads"],
                    n_kv_heads=row["n_kv"], head_dim=row["head_dim"],
                    intermediate_size=row["intermediate"],
                    scale_f32=float(np.float32(1.0) / np.sqrt(np.float32(row["head_dim"]))))
        P = {k: v.to(torch.float64) for k, v in W.items()}
        theirs = xf_corpus.block_forward(ccfg, P, x.to(torch.float64),
                                         list(range(L)), torch.float64)["residual2.out"]
        d = float((mine - theirs).abs().max())
        note(lane, "torch", "crosscheck LlamaEager vs transformer/corpus "
                            "block_forward at %s: max_abs_diff=%.6g" % (row["name"], d))
    except Exception as e:
        note(lane, "torch", "crosscheck REFUSED: %s: %s" % (type(e).__name__, str(e)[:160]))


if __name__ == "__main__":
    sys.exit(main())
