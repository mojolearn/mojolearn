#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A public pretrained LLM through mojolearn's GPU Mamba2Block, IDENTICAL mode.

    state-spaces/mamba2-130m  ->  24 x mojolearn.Mamba2Block (GPU, profile
    mojolearn.identical.mamba2.fp32.v1)  ->  norm_f and the tied head on the
    host with order-defined arithmetic only  ->  greedy tokens.

Run it on two vendors and `--compare` the two JSON files.

WHAT RUNS WHERE
  weights         the published pytorch_model.bin stores FLOAT16; every tensor
                  is widened to float32 on load, which is exact (every
                  binary16 value is a binary32 value) and round-trip checked
  embedding       host, exact row copies (`E[ids]`), no arithmetic
  24 blocks       GPU, `Mamba2Block.forward` with no state (a fresh zero-state
                  prefill each call); the block is norm + mixer + residual,
                  so block 24's output is already `hidden + residual`, the
                  tensor mamba_ssm hands to norm_f (see VERIFY 2 below)
  norm_f          host: elementwise float32 numpy ufuncs, the sum of squares
                  as a SEQUENTIAL `np.cumsum(..., axis=-1)[..., -1]`
  tied head       host: `np.cumsum(E * h_t, axis=1)[:, -1]` per row, chunked
  greedy          `np.argmax` over the first 50277 logits (first index wins)
  generation      full re-prefill of the growing sequence each step, so every
                  step is the same prefill contract; no decode state, no EOS
                  stop (a fixed token count keeps two runs comparable)

NOTHING IN THE IDENTITY PATH USES np.sum, np.mean, np.dot, `@`, einsum or BLAS.
Elementwise IEEE multiply/add/divide/sqrt are correctly rounded on every
platform numpy runs on, and `np.cumsum` is `np.add.accumulate`, whose each
output depends on the previous one, so its order is the index order. The two
remaining host hazards are named and checked rather than assumed away: a
subnormal operand (where an FTZ/DAZ flag set by some runtime could change a
bit) is COUNTED at every host stage and the run refuses to claim identity if
any appears, and FMA contraction cannot happen because every arithmetic step
is its own ufunc call.

THE THREE VERIFY ITEMS (checked statically at every run; the run refuses if
the tree it imports from does not carry them):
  1. eps. mamba_ssm builds both norms at 1e-5 (mixer_seq_simple.py:36/:134
     norm_epsilon, layer_norm.py:957 RMSNorm, mamba2.py:144 the gated norm);
     config.json names no norm_epsilon, so the default holds. mojolearn:
     mamba_fixture.mojo:46 RMS_EPS = 1e-5 feeds modeling_mamba.mojo:787, the
     one rms kernel mamba2.mojo reuses for S1-S3 and S21.
  2. residual. mamba_ssm Block (block.py:57-67) adds hidden + residual inside
     the fused norm and the mixer reads the normed sum; MixerModel.forward
     (mixer_seq_simple.py:208-217) runs the same fused add into norm_f. A
     Mamba2Block returns x + out_proj (mamba2.mojo:1244, HF :630), so block
     i's output equals mamba_ssm's residual entering block i+1, and block
     24's output IS norm_f's input. No host residual add exists here.
  3. DEVIATIONS 2712/2713 (SUPPORT_MATRIX.md:134). Before the fix
     m2_ydiag_kernel read X_d rows T..Q-1, which is EVERY sequence here
     (T < 256). The fix bounds the read (ssd_minimal.mojo:486); the guard
     text is required in the source this run imports.

Exit codes: 0 ok; 2 refused input or not comparable; 3 the requested mode is
refused by the library (fast, for this identical-only lane); 4 the run
failed (non-finite, subnormal on the host, binding error); 1 --compare found
a divergence.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import io
import json
import os
import pickle
import platform
import struct
import subprocess
import sys
import time
import unicodedata
import zipfile

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCHEMA = "mojolearn.mamba2_pretrained_identity.v1"
TORCH_SCHEMA = "mojolearn.mamba2_pretrained_torch_reference.v1"
PROFILE = "mojolearn.identical.mamba2.fp32.v1"

# ---------------------------------------------------------------- the pins
# tools/mamba2_pretrained_leg.sh reads these lines with sed; keep the
# `NAME = "value"` shape at column 0.
MAMBA2_130M_REPO = "state-spaces/mamba2-130m"
MAMBA2_130M_REVISION = "3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49"
# FILL THIS IN (64 lowercase hex). Empty refuses every run. The local HF cache
# stores this file as blob f786aff903d8b6f2cef67ff2b1db06a9f44e7780340e6ea6d3dead16b52fc501,
# which is the git-LFS sha256 oid; confirm with `shasum -a 256` before pinning.
MAMBA2_130M_PYTORCH_MODEL_SHA256 = "f786aff903d8b6f2cef67ff2b1db06a9f44e7780340e6ea6d3dead16b52fc501"
MAMBA2_130M_CONFIG_SHA256 = "b4ff95783684446fdaa9633244f7d2f8f9b3809a29f70f24ddb0803f344bc561"
GPT_NEOX_20B_REPO = "EleutherAI/gpt-neox-20b"
GPT_NEOX_20B_REVISION = "c292233c833e336628618a88a648727eb3dff0a7"
GPT_NEOX_20B_TOKENIZER_SHA256 = "c24618a1b3e6a38167beff1c72cffd126c3a66254347304b50547d12c5f25624"

DEFAULT_PROMPT = "The capital of France is"

D_MODEL = 768
N_LAYER = 24
VOCAB = 50277
VOCAB_PADDED = 50288
NHEADS = 24
CONV_DIM = 1792
D_IN_PROJ = 3352
F32 = np.dtype("<f4")
EPS = np.float32(1e-5)
EPS_BITS = 0x3727C5AC
HEAD_CHUNK = 4096

# The source facts VERIFY 1-3 rest on, as text the imported tree must carry.
SOURCE_FACTS = (
    ("mamba/checks/mamba_fixture.mojo", "comptime RMS_EPS: Float32 = 1e-5",
     "VERIFY 1: the rms kernel's eps is 1e-5"),
    ("mamba/impl/modeling/modeling_mamba.mojo",
     "var rstd = ftz(identical_rsqrt(ftz(mean + RMS_EPS)))",
     "VERIFY 1: mamba_rms_norm adds RMS_EPS before the 1/sqrt"),
    ("mamba/impl/modules/mamba2.mojo",
     "# ---- S1-S3: the block RMSNorm, the REUSED Mamba-1 kernel.",
     "VERIFY 1: the block norm is mamba_rms_norm"),
    ("mamba/impl/modules/mamba2.mojo",
     "#      RMSNorm machinery over d_ssm = d_inner.",
     "VERIFY 1: the gated norm is mamba_rms_norm"),
    ("mamba/impl/modules/mamba2.mojo",
     "# ---- S22: residual (HF :630), the REUSED Mamba-1 kernel.",
     "VERIFY 2: the block returns x + out_proj"),
    ("mamba/impl/modules/ssd_minimal.mojo",
     "if c * qv + jj < t_work or SAB_2712_UNBOUNDED:",
     "VERIFY 3: m2_ydiag_kernel reads X_d only below T (DEVIATIONS 2712/2713)"),
)
# Everything whose change could move the binding's bits without moving the
# commit we record (an uncommitted edit): mtime-checked against the .so.
BINDING_SOURCES = (
    "bindings/_mojolearn_mamba.mojo",
    "mamba/impl/modules/mamba2.mojo",
    "mamba/impl/modules/ssd_minimal.mojo",
    "mamba/impl/modeling/modeling_mamba.mojo",
    "checks/numerics.mojo",
)

EXIT_OK, EXIT_DIVERGENT, EXIT_REFUSED, EXIT_MODE_REFUSED, EXIT_FAILED = 0, 1, 2, 3, 4


class Refusal(Exception):
    def __init__(self, message, code=EXIT_REFUSED):
        super().__init__(message)
        self.code = code


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_f32(a):
    a = np.ascontiguousarray(a)
    if a.dtype != F32:
        raise Refusal(f"internal: hashing a {a.dtype} array, want little-endian float32", EXIT_FAILED)
    return hashlib.sha256(a.tobytes()).hexdigest()


# ------------------------------------------------------- torch-free loader
#: The storage types admitted. The published mamba2-130m file stores its
#: tensors as HalfStorage (measured 2026-09-15); they are widened to float32.
_STORAGE_DTYPES = {"FloatStorage": np.dtype("<f4"), "HalfStorage": np.dtype("<f2")}


class _StorageType:
    def __init__(self, name):
        self.name = name


class _TensorRef:
    __slots__ = ("storage", "offset", "size", "stride")

    def __init__(self, storage, offset, size, stride):
        self.storage, self.offset, self.size, self.stride = storage, offset, size, stride


def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad=False,
                       backward_hooks=None, metadata=None):
    return _TensorRef(storage, int(storage_offset), tuple(int(s) for s in size),
                      tuple(int(s) for s in stride))


class _RestrictedUnpickler(pickle.Unpickler):
    """Resolves exactly the globals a plain float32 state dict needs and
    refuses every other one by name, so no code in the archive runs."""

    def find_class(self, module, name):
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild_tensor_v2
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        raise pickle.UnpicklingError(f"refusing pickle global {module}.{name}")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        _, stype, key, location, numel = pid
        sname = stype.name if isinstance(stype, _StorageType) else repr(stype)
        return (sname, str(key), str(location), int(numel))


def load_checkpoint(path):
    if sys.byteorder != "little":
        raise Refusal("this harness hashes native float32 bytes and assumes a little-endian host")
    tensors = collections.OrderedDict()
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        pkl = [n for n in names if n == "data.pkl" or n.endswith("/data.pkl")]
        if len(pkl) != 1:
            raise Refusal(f"{path}: expected one data.pkl in the torch zip, found {sorted(pkl)}")
        prefix = pkl[0][: -len("data.pkl")]
        bo_name = prefix + "byteorder"
        byteorder = zf.read(bo_name).decode().strip() if bo_name in names else None
        if byteorder not in (None, "little"):
            raise Refusal(f"{path}: byteorder record says {byteorder!r}, want little")
        state = _RestrictedUnpickler(io.BytesIO(zf.read(pkl[0]))).load()
        if not isinstance(state, dict):
            raise Refusal(f"{path}: the pickle is a {type(state).__name__}, want a state dict")
        raw_cache = {}
        noncontiguous = []
        source_dtypes = collections.Counter()
        for name, ref in state.items():
            if not isinstance(ref, _TensorRef):
                raise Refusal(f"{path}: entry {name} is a {type(ref).__name__}, not a tensor")
            sname, key, _loc, numel = ref.storage
            dt = _STORAGE_DTYPES.get(sname)
            if dt is None:
                raise Refusal(f"{path}: {name} is stored as {sname}; only FloatStorage and HalfStorage "
                              "(widened exactly to float32) are admitted")
            source_dtypes[sname] += 1
            isz = dt.itemsize
            if key not in raw_cache:
                raw_cache[key] = zf.read(prefix + "data/" + key)
            raw = raw_cache[key]
            if len(raw) != numel * isz:
                raise Refusal(f"{path}: storage {key} has {len(raw)} bytes for {numel} x {sname}")
            size, stride, off = ref.size, ref.stride, ref.offset
            if len(size) != len(stride) or off < 0 or any(s < 0 for s in stride):
                raise Refusal(f"{path}: {name} has size {size} stride {stride} offset {off}")
            count, last = 1, off
            for n_, s_ in zip(size, stride):
                count *= n_
                last += (n_ - 1) * s_
            if count == 0 or last >= numel:
                raise Refusal(f"{path}: {name} addresses past its storage")
            base = np.frombuffer(raw, dtype=dt, count=numel)
            view = np.lib.stride_tricks.as_strided(
                base[off:], shape=size, strides=tuple(s_ * isz for s_ in stride), writeable=False)
            if not view.flags.c_contiguous:
                noncontiguous.append(name)
            src = np.array(view, dtype=dt, order="C", copy=True)  # byte copy
            if dt == F32:
                tensors[name] = src
            else:
                # IEEE binary16 -> binary32 is EXACT: every half value, signed
                # zero, subnormal and infinity included, is a float32 value, so
                # the widening rounds nothing. Proven per tensor by the round
                # trip, not assumed.
                wide = src.astype(F32)
                if wide.astype(dt).tobytes() != src.tobytes():
                    raise Refusal(f"{path}: widening {name} from float16 did not round-trip")
                tensors[name] = wide
    return tensors, {"byteorder_record": byteorder, "tensors": len(tensors),
                     "source_storage_types": dict(source_dtypes),
                     "widening": "float16 -> float32, exact, round-trip verified per tensor",
                     "noncontiguous_made_contiguous": noncontiguous}


def expected_shapes():
    shapes = {"backbone.embedding.weight": (VOCAB_PADDED, D_MODEL),
              "backbone.norm_f.weight": (D_MODEL,), "lm_head.weight": (VOCAB_PADDED, D_MODEL)}
    for i in range(N_LAYER):
        p = f"backbone.layers.{i}."
        shapes.update({
            p + "norm.weight": (D_MODEL,), p + "mixer.dt_bias": (NHEADS,),
            p + "mixer.A_log": (NHEADS,), p + "mixer.D": (NHEADS,),
            p + "mixer.in_proj.weight": (D_IN_PROJ, D_MODEL),
            p + "mixer.conv1d.weight": (CONV_DIM, 1, 4), p + "mixer.conv1d.bias": (CONV_DIM,),
            p + "mixer.norm.weight": (2 * D_MODEL,), p + "mixer.out_proj.weight": (D_MODEL, 2 * D_MODEL),
        })
    return shapes


def check_state(state):
    want = expected_shapes()
    missing = sorted(set(want) - set(state))
    extra = sorted(set(state) - set(want))
    wrong = sorted(k for k in want if k in state and state[k].shape != want[k])
    if missing or extra or wrong:
        raise Refusal(f"checkpoint tensors disagree with mamba2-130m: missing {missing[:5]}, "
                      f"unexpected {extra[:5]}, wrong shape {wrong[:5]}")
    for k, v in state.items():
        if not np.isfinite(v).all():
            raise Refusal(f"checkpoint tensor {k} holds a non-finite value")
    if state["lm_head.weight"].tobytes() != state["backbone.embedding.weight"].tobytes():
        raise Refusal("lm_head.weight is not byte-identical to the embedding; tie_embeddings is not what ran")


def check_config(cfg):
    want = {"d_model": 768, "d_intermediate": 0, "n_layer": 24, "vocab_size": 50277,
            "ssm_cfg": {"layer": "Mamba2"}, "attn_layer_idx": [], "attn_cfg": {},
            "rms_norm": True, "residual_in_fp32": True, "fused_add_norm": True,
            "pad_vocab_size_multiple": 16, "tie_embeddings": True}
    problems = [f"{k}={cfg.get(k, '<absent>')!r} (want {v!r})" for k, v in want.items()
                if cfg.get(k, "<absent>") != v]
    unknown = sorted(set(cfg) - set(want) - {"norm_epsilon"})
    if unknown:
        problems.append(f"unknown keys {unknown} (any of them could change the arithmetic)")
    if "norm_epsilon" in cfg and float(cfg["norm_epsilon"]) != 1e-5:
        problems.append(f"norm_epsilon={cfg['norm_epsilon']!r} (the block profile pins 1e-5)")
    # ssm_cfg == {"layer": "Mamba2"} means every Mamba2 constructor default
    # holds: d_state 128, d_conv 4, expand 2, headdim 64, ngroups 1,
    # chunk_size 256, dt_limit (0, inf), rmsnorm True, norm_before_gate False,
    # D_has_hdim False, bias False, conv_bias True (mamba2.py:41-59), which
    # is the profile's section 3 exactly.
    if problems:
        raise Refusal("config.json is not the profile this block certifies: " + "; ".join(problems))


# ---------------------------------------------------------------- tokenizer
_WS = frozenset("\t\n\x0b\x0c\r \x85\xa0     　"
                + "".join(chr(c) for c in range(0x2000, 0x200B)))


def _is_letter(ch):
    return unicodedata.category(ch)[0] == "L"


def _is_number(ch):
    return unicodedata.category(ch)[0] == "N"


def _is_other(ch):
    return not (ch in _WS or _is_letter(ch) or _is_number(ch))


def gpt2_pretokenize(text):
    """The ByteLevel pre-tokenizer's regex, as a scanner, alternatives in the
    regex's own order: 's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+|
    ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+ (first alternative that matches wins)."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == "'" and i + 1 < n:
            if text[i + 1:i + 3] in ("re", "ve", "ll"):
                out.append(text[i:i + 3]); i += 3; continue
            if text[i + 1] in "stmd":
                out.append(text[i:i + 2]); i += 2; continue
        matched = False
        for pred in (_is_letter, _is_number, _is_other):
            if pred(ch):
                k = i
            elif ch == " " and i + 1 < n and pred(text[i + 1]):
                k = i + 1
            else:
                continue
            while k < n and pred(text[k]):
                k += 1
            out.append(text[i:k]); i = k; matched = True
            break
        if matched:
            continue
        k = i
        while k < n and text[k] in _WS:
            k += 1
        if k < n and k - i >= 2:
            out.append(text[i:k - 1]); i = k - 1
        else:
            out.append(text[i:k]); i = k
    return out


def _bytes_to_unicode():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1))
          + list(range(ord("\xae"), ord("\xff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class NeoXTokenizer:
    """Byte-level BPE from tokenizer.json with no `tokenizers` package. It
    only chooses the prompt ids, which are recorded and compared; the
    identity claim does not depend on it matching HF (the torch reference
    phase checks that when `tokenizers` is importable)."""

    def __init__(self, path):
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        pre, model = d.get("pre_tokenizer") or {}, d.get("model") or {}
        problems = []
        if (d.get("normalizer") or {}).get("type") != "NFC":
            problems.append(f"normalizer {d.get('normalizer')!r}")
        if (pre.get("type"), pre.get("add_prefix_space"), pre.get("use_regex")) != ("ByteLevel", False, True):
            problems.append(f"pre_tokenizer {pre!r}")
        if model.get("type") != "BPE" or model.get("dropout") not in (None, 0.0):
            problems.append(f"model type {model.get('type')!r} dropout {model.get('dropout')!r}")
        if problems:
            raise Refusal("tokenizer.json is not the GPT-NeoX byte-level BPE this code implements: "
                          + "; ".join(problems))
        self.vocab = model["vocab"]
        self.inv = {v: k for k, v in self.vocab.items()}
        self.ranks = {}
        for r, m in enumerate(model["merges"]):
            a, b = m.split(" ") if isinstance(m, str) else m
            self.ranks[(a, b)] = r
        self.added = {a["content"]: int(a["id"]) for a in d.get("added_tokens", [])}
        self.added_inv = {v: k for k, v in self.added.items()}
        self.added_by_len = sorted(self.added, key=len, reverse=True)
        self.byte_enc = _bytes_to_unicode()
        self.byte_dec = {v: k for k, v in self.byte_enc.items()}
        self._cache = {}

    def _bpe(self, word):
        if word in self._cache:
            return self._cache[word]
        parts = list(word)
        while len(parts) > 1:
            best, best_rank = None, None
            for j in range(len(parts) - 1):
                r = self.ranks.get((parts[j], parts[j + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = j, r
            if best is None:
                break
            a, b = parts[best], parts[best + 1]
            merged, j = [], 0
            while j < len(parts):
                if j < len(parts) - 1 and parts[j] == a and parts[j + 1] == b:
                    merged.append(a + b); j += 2
                else:
                    merged.append(parts[j]); j += 1
            parts = merged
        self._cache[word] = parts
        return parts

    def _encode_plain(self, text, ids):
        for piece in gpt2_pretokenize(text):
            mapped = "".join(self.byte_enc[b] for b in piece.encode("utf-8"))
            for tok in self._bpe(mapped):
                if tok not in self.vocab:
                    raise Refusal(f"tokenizer: BPE produced {tok!r}, which is not in the vocabulary")
                ids.append(self.vocab[tok])

    def encode(self, text):
        text = unicodedata.normalize("NFC", text)
        ids, i, start = [], 0, 0
        while i < len(text):
            hit = next((c for c in self.added_by_len if text.startswith(c, i)), None)
            if hit is None:
                i += 1
                continue
            self._encode_plain(text[start:i], ids)
            ids.append(self.added[hit])
            i += len(hit)
            start = i
        self._encode_plain(text[start:], ids)
        return ids

    def decode(self, ids):
        out, buf = [], bytearray()
        for t in ids:
            if t in self.added_inv:
                out.append(buf.decode("utf-8", errors="replace")); buf = bytearray()
                out.append(self.added_inv[t])
            elif t in self.inv:
                buf.extend(self.byte_dec[c] for c in self.inv[t])
            else:
                out.append(buf.decode("utf-8", errors="replace")); buf = bytearray()
                out.append(f"<unknown id {t}>")
        out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


# ------------------------------------------------------ host arithmetic
class SubnormalCensus:
    TINY = np.finfo(np.float32).smallest_normal

    def __init__(self):
        self.counts = collections.Counter()

    def check(self, stage, a):
        if a.dtype != F32:
            raise Refusal(f"host stage {stage} produced {a.dtype}; the host path must stay float32", EXIT_FAILED)
        if not np.isfinite(a).all():
            raise Refusal(f"host stage {stage} produced a non-finite value", EXIT_FAILED)
        self.counts[stage] += int(np.count_nonzero((a != 0) & (np.abs(a) < self.TINY)))

    def total(self):
        return sum(self.counts.values())


def host_rmsnorm(h, w, census):
    """mamba_ssm rms_norm_ref (layer_norm.py:120-121): rstd = 1/sqrt(mean(x^2)
    + eps), out = (x * rstd) * weight. h is (L, d_model) float32."""
    sq = np.multiply(h, h)
    acc = np.cumsum(sq, axis=-1)
    mean = np.divide(acc[:, -1], np.float32(D_MODEL))
    rstd = np.divide(np.float32(1.0), np.sqrt(np.add(mean, EPS)))
    scaled = np.multiply(h, rstd[:, None])
    out = np.multiply(scaled, w)
    for stage, a in (("norm_f.square", sq), ("norm_f.cumsum", acc), ("norm_f.mean", mean),
                     ("norm_f.rstd", rstd), ("norm_f.scaled", scaled), ("norm_f.out", out)):
        census.check(stage, a)
    return out


def host_logits(E, h_t, census):
    """logit[v] = serial sum over j ascending of E[v, j] * h_t[j]."""
    out = np.empty(E.shape[0], dtype=F32)
    for r0 in range(0, E.shape[0], HEAD_CHUNK):
        prod = np.multiply(E[r0:r0 + HEAD_CHUNK], h_t)
        acc = np.cumsum(prod, axis=1)
        out[r0:r0 + HEAD_CHUNK] = acc[:, -1]
        census.check("head.product", prod)
        census.check("head.cumsum", acc)
    return out


def greedy(logits):
    v = logits[:VOCAB]
    if not np.isfinite(v).all():
        raise Refusal("non-finite logit", EXIT_FAILED)
    return int(np.argmax(v))  # comparisons only; the first maximal index wins


# ------------------------------------------------------------ provenance
def git_info():
    info = {"commit": None, "commit_source": None, "tree_dirty_paths": None}
    try:
        commit = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=20)
        if commit.returncode == 0:
            info["commit"], info["commit_source"] = commit.stdout.strip(), "git rev-parse"
            st = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True,
                                text=True, timeout=60)
            if st.returncode == 0:
                info["tree_dirty_paths"] = len([x for x in st.stdout.splitlines() if x.strip()])
    except (OSError, subprocess.SubprocessError):
        pass
    if info["commit"] is None and os.environ.get("MOJOLEARN_GATE_COMMIT"):
        info["commit"], info["commit_source"] = os.environ["MOJOLEARN_GATE_COMMIT"], "MOJOLEARN_GATE_COMMIT (no .git)"
    return info


def mojo_source_sha256():
    """tools/do_extra_leg.sh's source_sha_recipe, reproduced: sha256 of the
    `sha256  ./path` lines of every *.mojo outside .pixi and bench/results,
    paths in byte order."""
    paths = []
    for root, dirs, files in os.walk(REPO):
        rel = os.path.relpath(root, REPO)
        if rel == ".":
            dirs[:] = [d for d in dirs if d not in (".pixi", ".git")]
        if rel.replace(os.sep, "/") == "bench":
            dirs[:] = [d for d in dirs if d != "results"]
        for f in files:
            if f.endswith(".mojo"):
                p = "./" + os.path.relpath(os.path.join(root, f), REPO).replace(os.sep, "/")
                paths.append(p)
    paths.sort(key=lambda s: s.encode("utf-8"))
    listing = "".join(f"{sha256_file(os.path.join(REPO, p[2:]))}  {p}\n" for p in paths)
    return hashlib.sha256(listing.encode("utf-8")).hexdigest(), len(paths)


def device_hint():
    hint = {}
    for key, cmd in (("nvidia_smi", ["nvidia-smi", "--query-gpu=name,driver_version,compute_cap",
                                     "--format=csv,noheader"]),
                     ("rocm_smi", ["rocm-smi", "--showproductname"]),
                     ("cpu_brand", ["sysctl", "-n", "machdep.cpu.brand_string"])):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if r.returncode == 0 and r.stdout.strip():
                hint[key] = r.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return hint


def verify_sources():
    facts = []
    for rel, needle, what in SOURCE_FACTS:
        path = os.path.join(REPO, rel)
        try:
            with open(path, encoding="utf-8") as fh:
                text = fh.read()
        except OSError as exc:
            raise Refusal(f"{what}: cannot read {rel} ({exc}); run from a source tree")
        if needle not in text:
            raise Refusal(f"{what}: {rel} no longer carries {needle!r}; re-verify before trusting a run")
        facts.append({"file": rel, "carries": needle, "what": what})
    if struct.unpack("<I", struct.pack("<f", 1e-5))[0] != EPS_BITS or \
            struct.unpack("<I", EPS.tobytes())[0] != EPS_BITS:
        raise Refusal("float32(1e-5) is not 0x3727C5AC on this host")
    return facts


def import_mojolearn():
    pydir = os.path.join(REPO, "python")
    sys.path.insert(0, pydir)
    prior = os.environ.get("MOJOLEARN_NUMERIC_MODE")
    os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
    import mojolearn
    from mojolearn import _backend, _mamba_impl
    if not os.path.abspath(mojolearn.__file__).startswith(pydir + os.sep):
        raise Refusal(f"imported mojolearn from {mojolearn.__file__}, not this tree's {pydir}; "
                      "the source checks would describe code that is not running")
    consts = {"_M2_D_STATE": 128, "_M2_D_CONV": 4, "_M2_EXPAND": 2, "_M2_HEADDIM": 64,
              "_M2_NGROUPS": 1, "_M2_CHUNK_SIZE": 256}
    bad = {k: getattr(_mamba_impl, k, None) for k, v in consts.items() if getattr(_mamba_impl, k, None) != v}
    if bad:
        raise Refusal(f"_mamba_impl profile constants moved: {bad}")
    return mojolearn, _backend, _mamba_impl, prior


# --------------------------------------------------------------- the run
def find_default(pattern_parts):
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    hits = sorted(glob.glob(os.path.join(hub, *pattern_parts)))
    return hits[-1] if hits else None


def build_blocks(state, Mamba2Block, mode):
    blocks = []
    for i in range(N_LAYER):
        p = f"backbone.layers.{i}."
        w = {
            "block_norm.weight": state[p + "norm.weight"],
            "in_proj.weight": state[p + "mixer.in_proj.weight"],
            "conv1d.weight": state[p + "mixer.conv1d.weight"],
            "conv1d.bias": state[p + "mixer.conv1d.bias"],
            "dt_bias": state[p + "mixer.dt_bias"],
            "A_log": state[p + "mixer.A_log"],
            "D": state[p + "mixer.D"],
            "norm.weight": state[p + "mixer.norm.weight"],
            "out_proj.weight": state[p + "mixer.out_proj.weight"],
        }
        blk = Mamba2Block(w, numeric_mode=mode)
        if (blk.d_model, blk.nheads, blk.conv_dim, blk.dt_limit) != (D_MODEL, NHEADS, CONV_DIM, (0.0, float("inf"))):
            raise Refusal(f"block {i}: d_model/nheads/conv_dim/dt_limit = "
                          f"{(blk.d_model, blk.nheads, blk.conv_dim, blk.dt_limit)}")
        blocks.append(blk)
    return blocks


def run_stack(blocks, x):
    hashes, h = [], x
    for i, blk in enumerate(blocks):
        y = blk.forward(h)
        a = np.asarray(y)
        if a.dtype != F32 or a.shape != h.shape:
            raise Refusal(f"layer {i}: output {a.dtype} {a.shape}, want float32 {h.shape}", EXIT_FAILED)
        if not np.isfinite(a).all():
            raise Refusal(f"layer {i}: output holds a non-finite value", EXIT_FAILED)
        hashes.append(sha256_f32(a))
        h = a
    return h, hashes


def identity_run(args):
    t_start = time.perf_counter()
    result = {"schema": SCHEMA, "profile": PROFILE, "status": "running", "mode": args.mode,
              "harness_sha256": sha256_file(os.path.abspath(__file__))}
    try:
        _identity_body(args, result, t_start)
    except Refusal as exc:
        result["status"] = "mode_refused" if exc.code == EXIT_MODE_REFUSED else (
            "refused" if exc.code == EXIT_REFUSED else "failed")
        result["reason"] = str(exc)
        result["wall_seconds"] = round(time.perf_counter() - t_start, 3)
        write_json(args.out, result)
        print(f"{result['status'].upper()}: {exc}", file=sys.stderr)
        return exc.code
    result["wall_seconds"] = round(time.perf_counter() - t_start, 3)
    write_json(args.out, result)
    print(f"OK: {len(result['steps'])} greedy tokens, logits sha256 of the prompt "
          f"{result['prompt']['logits_all_positions_sha256'][:16]}, wrote {args.out}")
    return EXIT_OK


def _identity_body(args, result, t_start):
    if not MAMBA2_130M_PYTORCH_MODEL_SHA256 or len(MAMBA2_130M_PYTORCH_MODEL_SHA256) != 64:
        raise Refusal("MAMBA2_130M_PYTORCH_MODEL_SHA256 is not filled in; pin the checkpoint first")
    if args.max_new_tokens < 1:
        raise Refusal("--max-new-tokens must be at least 1")
    model_dir = args.model_dir or find_default(("models--state-spaces--mamba2-130m", "snapshots", "*"))
    tok_path = args.tokenizer or find_default(("models--EleutherAI--gpt-neox-20b", "snapshots", "*",
                                               "tokenizer.json"))
    if not model_dir or not tok_path:
        raise Refusal("could not find the model directory or tokenizer.json; pass --model-dir and --tokenizer")
    ckpt, cfg_path = os.path.join(model_dir, "pytorch_model.bin"), os.path.join(model_dir, "config.json")

    result["environment"] = {
        "platform": platform.platform(), "machine": platform.machine(), "system": platform.system(),
        "python": sys.version.split()[0], "numpy": np.__version__, "byteorder": sys.byteorder,
        "device_hint_not_from_mojolearn": device_hint(),
        "thread_env": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    ck_sha = sha256_file(ckpt)
    cfg_sha = sha256_file(cfg_path)
    tok_sha = sha256_file(tok_path)
    result["checkpoint"] = {"repo": MAMBA2_130M_REPO, "revision": MAMBA2_130M_REVISION, "path": ckpt,
                            "sha256": ck_sha, "pinned_sha256": MAMBA2_130M_PYTORCH_MODEL_SHA256,
                            "config_path": cfg_path, "config_sha256": cfg_sha}
    result["tokenizer"] = {"repo": GPT_NEOX_20B_REPO, "revision": GPT_NEOX_20B_REVISION,
                           "path": tok_path, "sha256": tok_sha}
    if ck_sha != MAMBA2_130M_PYTORCH_MODEL_SHA256:
        raise Refusal(f"{ckpt} has sha256 {ck_sha}, the pin is {MAMBA2_130M_PYTORCH_MODEL_SHA256}")
    if cfg_sha != MAMBA2_130M_CONFIG_SHA256:
        raise Refusal(f"{cfg_path} has sha256 {cfg_sha}, the pin is {MAMBA2_130M_CONFIG_SHA256}")
    if tok_sha != GPT_NEOX_20B_TOKENIZER_SHA256:
        raise Refusal(f"{tok_path} has sha256 {tok_sha}, the pin is {GPT_NEOX_20B_TOKENIZER_SHA256}")
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    check_config(cfg)
    result["checkpoint"]["config"] = cfg

    result["verify"] = {"source_facts": verify_sources(),
                        "host_eps_bits": f"0x{EPS_BITS:08X}",
                        "residual": "block 24 output is norm_f input; no host residual add"}
    src_sha, n_src = mojo_source_sha256()
    result["provenance"] = dict(git_info(), mojo_source_sha256=src_sha, mojo_source_files=n_src)

    mojolearn, _backend, _mamba_impl, prior = import_mojolearn()
    mamba_impl_path = os.path.abspath(_mamba_impl.__file__)
    result["mojolearn"] = {"version": mojolearn.__version__, "file": os.path.abspath(mojolearn.__file__),
                           "mamba_impl_sha256": sha256_file(mamba_impl_path),
                           "buffer_py_sha256": sha256_file(os.path.join(os.path.dirname(mamba_impl_path), "_buffer.py")),
                           "prior_MOJOLEARN_NUMERIC_MODE": prior}

    tokenizer = NeoXTokenizer(tok_path)
    if args.prompt_ids:
        ids = [int(t) for t in args.prompt_ids.split(",") if t.strip()]
        ids_source = "--prompt-ids"
    else:
        ids = tokenizer.encode(args.prompt)
        ids_source = "tokenizer (pure-Python byte-level BPE in this file)"
    if not ids or any(t < 0 or t >= VOCAB for t in ids):
        raise Refusal(f"prompt ids {ids} are empty or out of range")
    if len(ids) + args.max_new_tokens - 1 > 4096:
        raise Refusal("sequence too long for a re-prefill harness")
    result["prompt"] = {"text": args.prompt if not args.prompt_ids else None, "ids": ids, "ids_source": ids_source,
                        "decoded": tokenizer.decode(ids)}

    print(f"loading {ckpt}", flush=True)
    state, load_info = load_checkpoint(ckpt)
    check_state(state)
    result["checkpoint"]["load"] = load_info
    E = state["backbone.embedding.weight"]
    norm_f = state["backbone.norm_f.weight"]

    blocks = build_blocks(state, _mamba_impl.Mamba2Block, args.mode)
    binding_info = {}
    try:
        ext = blocks[0]._extension()
    except Exception as exc:  # the identical-only refusal for fast lands here
        code = EXIT_MODE_REFUSED if args.mode != "identical" else EXIT_FAILED
        raise Refusal(f"mode {args.mode!r}: the library refused the Mamba2 binding: "
                      f"{type(exc).__name__}: {exc}", code)
    so_path = os.path.abspath(getattr(ext, "__file__", "") or "")
    if not so_path or not os.path.isfile(so_path):
        raise Refusal(f"cannot locate the loaded _mojolearn_mamba binary ({so_path!r})", EXIT_FAILED)
    so_mtime = os.path.getmtime(so_path)
    stale = [rel for rel in BINDING_SOURCES
             if os.path.getmtime(os.path.join(REPO, rel)) > so_mtime]
    if stale:
        raise Refusal(f"{so_path} is older than {stale}; rebuild it (bash bindings/build_mamba.sh)")
    binding_info.update(path=so_path, sha256=sha256_file(so_path),
                        vendor_used=blocks[0].vendor_used(),
                        numeric_mode_used=blocks[0].numeric_mode_used())
    try:
        binding_info["gpu_arch"] = _backend.gpu_arch()
    except Exception as exc:
        binding_info["gpu_arch"] = f"unavailable: {exc}"
    result["mojolearn"]["binding"] = binding_info
    if args.mode == "identical" and binding_info["numeric_mode_used"] != "identical":
        raise Refusal(f"the binding reports {binding_info['numeric_mode_used']!r}, not identical", EXIT_FAILED)

    census = SubnormalCensus()
    seq, steps, prompt_logits, step_logits = list(ids), [], None, []
    for k in range(args.max_new_tokens):
        t0 = time.perf_counter()
        x = np.ascontiguousarray(E[np.asarray(seq, dtype=np.int64)][None, :, :])
        try:
            hidden, layer_hashes = run_stack(blocks, x)
        except Refusal:
            raise
        except Exception as exc:
            code = EXIT_MODE_REFUSED if args.mode != "identical" else EXIT_FAILED
            raise Refusal(f"Mamba2Block.forward raised {type(exc).__name__}: {exc}", code)
        t_gpu = time.perf_counter()
        normed = host_rmsnorm(hidden[0], norm_f, census)
        if k == 0:
            prompt_logits = np.stack([host_logits(E, normed[p], census) for p in range(len(seq))])
            last = prompt_logits[-1]
        else:
            last = host_logits(E, normed[-1], census)
        tok = greedy(last)
        step_logits.append(last)
        rec = {"step": k, "seq_len": len(seq), "embedding_sha256": sha256_f32(x),
               "layers_sha256": layer_hashes, "final_hidden_sha256": layer_hashes[-1],
               "norm_f_sha256": sha256_f32(normed), "logits_last_sha256": sha256_f32(last),
               "token_id": tok, "token_text": tokenizer.decode([tok]),
               "argmax_logit_hex": float(last[tok]).hex(),
               "gpu_seconds": round(t_gpu - t0, 3), "host_seconds": round(time.perf_counter() - t_gpu, 3)}
        steps.append(rec)
        if k == 0:
            result["prompt"].update(
                embedding_sha256=rec["embedding_sha256"], layers_sha256=layer_hashes,
                final_hidden_sha256=layer_hashes[-1], norm_f_sha256=rec["norm_f_sha256"],
                logits_all_positions_sha256=sha256_f32(prompt_logits),
                logits_last_sha256=rec["logits_last_sha256"])
        print(f"step {k:2d} len {len(seq):3d} token {tok:5d} {rec['token_text']!r} "
              f"logits {rec['logits_last_sha256'][:16]} gpu {rec['gpu_seconds']}s", flush=True)
        seq.append(tok)

    generated = [s["token_id"] for s in steps]
    result["steps"] = steps
    result["generated_ids"] = generated
    result["continuation_text"] = tokenizer.decode(generated)
    result["max_new_tokens"] = args.max_new_tokens
    result["host"] = {"subnormal_counts": dict(census.counts), "subnormal_free": census.total() == 0,
                      "ops": "np.multiply, np.divide, np.sqrt, np.add, np.cumsum(axis=-1)[..., -1], "
                             "np.argmax; no reductions, no BLAS"}
    if args.save_logits:
        path = os.path.splitext(args.out)[0] + "_logits.npz"
        np.savez(path, prompt_logits=prompt_logits, step_logits=np.stack(step_logits),
                 prompt_ids=np.asarray(ids, dtype=np.int64), generated_ids=np.asarray(generated, dtype=np.int64))
        result["saved_logits"] = {"path": path, "sha256": sha256_file(path)}
    if args.torch_reference:
        result["torch_reference"] = torch_reference(state, ids, generated, prompt_logits,
                                                    np.stack(step_logits), tok_path, args.prompt,
                                                    ids_source)
    if not result["host"]["subnormal_free"]:
        result["status"] = "failed"
        raise Refusal(f"subnormal values on the host path {dict(census.counts)}; an FTZ/DAZ flag could "
                      "change those bits, so this run makes no identity claim", EXIT_FAILED)
    result["status"] = "ok"


# ---------------------------------------------------------- torch sanity
def torch_reference(state, prompt_ids, generated, our_prompt_logits, our_step_logits,
                    tok_path, prompt_text, ids_source):
    """HF Mamba2ForCausalLM on the CPU in float32 over the same ids. A SANITY
    check of the model wiring, never an identity claim: torch's arithmetic is
    not this profile's and differences of order 1e-4 are expected."""
    try:
        import torch
        from transformers import Mamba2Config, Mamba2ForCausalLM
    except Exception as exc:
        return {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    rep = {"status": "ok", "claim": "sanity only, not identity",
           "torch": torch.__version__}
    try:
        import transformers
        rep["transformers"] = transformers.__version__
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
        cfg = Mamba2Config(num_heads=NHEADS, head_dim=64, vocab_size=VOCAB_PADDED, hidden_size=D_MODEL,
                           state_size=128, num_hidden_layers=N_LAYER, layer_norm_epsilon=1e-5, expand=2,
                           conv_kernel=4, n_groups=1, use_bias=False, use_conv_bias=True,
                           residual_in_fp32=True, time_step_limit=(0.0, float("inf")), chunk_size=256,
                           tie_word_embeddings=True)
        model = Mamba2ForCausalLM(cfg).to(torch.float32).eval()
        sd = {k.replace("backbone.embedding.weight", "backbone.embeddings.weight"):
              torch.from_numpy(np.array(v, copy=True)) for k, v in state.items()}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            return {"status": "refused", "reason": f"missing {list(missing)} unexpected {list(unexpected)}"}
        seq = list(prompt_ids) + list(generated[:-1])
        with torch.no_grad():
            ref = model(input_ids=torch.tensor([seq], dtype=torch.long), use_cache=False).logits[0]
        ref = ref.to(torch.float32).numpy()
        L = len(prompt_ids)
        ours_p = np.asarray(our_prompt_logits, dtype=np.float64)[:, :VOCAB]
        rep["prompt_max_abs_logit_diff"] = float(np.max(np.abs(ours_p - ref[:L, :VOCAB].astype(np.float64))))
        per_step = []
        for k, tok in enumerate(generated):
            r = ref[L - 1 + k, :VOCAB]
            per_step.append({"step": k, "ours": int(tok), "torch_argmax": int(np.argmax(r)),
                             "max_abs_logit_diff": float(np.max(np.abs(
                                 np.asarray(our_step_logits[k], dtype=np.float64)[:VOCAB] - r.astype(np.float64))))})
        rep["steps"] = per_step
        rep["greedy_agreement"] = sum(s["ours"] == s["torch_argmax"] for s in per_step)
        rep["greedy_steps"] = len(per_step)
        try:
            if ids_source.startswith("--prompt-ids") or not prompt_text:
                raise RuntimeError("the prompt ids were given explicitly, so there is no text to tokenize")
            from tokenizers import Tokenizer
            hf_ids = Tokenizer.from_file(tok_path).encode(prompt_text, add_special_tokens=False).ids
            rep["tokenizer_parity"] = {"hf_ids": hf_ids, "ours": list(prompt_ids),
                                       "equal": hf_ids == list(prompt_ids), "ids_source": ids_source}
        except Exception as exc:
            rep["tokenizer_parity"] = {"status": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return rep


def torch_reference_only(args):
    with open(args.torch_reference_only, encoding="utf-8") as fh:
        res = json.load(fh)
    if res.get("status") != "ok" or "saved_logits" not in res:
        print("the result is not ok or was run without --save-logits", file=sys.stderr)
        return EXIT_REFUSED
    npz = res["saved_logits"]["path"]
    if not os.path.isfile(npz):
        npz = os.path.join(os.path.dirname(os.path.abspath(args.torch_reference_only)), os.path.basename(npz))
    if sha256_file(npz) != res["saved_logits"]["sha256"]:
        print(f"{npz} does not match the sha256 in the result", file=sys.stderr)
        return EXIT_REFUSED
    ckpt = os.path.join(args.model_dir or os.path.dirname(res["checkpoint"]["path"]), "pytorch_model.bin")
    if sha256_file(ckpt) != res["checkpoint"]["sha256"]:
        print(f"{ckpt} is not the checkpoint the result ran", file=sys.stderr)
        return EXIT_REFUSED
    state, _ = load_checkpoint(ckpt)
    check_state(state)
    saved = np.load(npz)
    tok_path = args.tokenizer or res["tokenizer"]["path"]
    rep = torch_reference(state, res["prompt"]["ids"], res["generated_ids"], saved["prompt_logits"],
                          saved["step_logits"], tok_path, res["prompt"].get("text") or "",
                          res["prompt"]["ids_source"])
    out = {"schema": TORCH_SCHEMA, "against": os.path.abspath(args.torch_reference_only),
           "against_sha256": sha256_file(args.torch_reference_only), "torch_reference": rep}
    write_json(args.out, out)
    print(json.dumps({k: rep.get(k) for k in ("status", "prompt_max_abs_logit_diff", "greedy_agreement",
                                              "greedy_steps", "reason")}))
    return EXIT_OK if rep.get("status") == "ok" else EXIT_FAILED


# -------------------------------------------------------------- compare
def compare(path_a, path_b):
    runs = []
    for p in (path_a, path_b):
        with open(p, encoding="utf-8") as fh:
            runs.append(json.load(fh))
    A, B = runs
    for name, r in (("A", A), ("B", B)):
        b = (r.get("mojolearn") or {}).get("binding") or {}
        env = r.get("environment") or {}
        print(f"{name}: status={r.get('status')} mode={r.get('mode')} vendor={b.get('vendor_used')} "
              f"arch={b.get('gpu_arch')} device={env.get('device_hint_not_from_mojolearn')} "
              f"platform={env.get('platform')} commit={(r.get('provenance') or {}).get('commit')}")
    for name, r in (("A", A), ("B", B)):
        if r.get("schema") != SCHEMA or r.get("status") != "ok":
            print(f"NOT COMPARABLE: run {name} has schema {r.get('schema')!r} status {r.get('status')!r} "
                  f"({r.get('reason')})")
            return EXIT_REFUSED
    must_match = (
        ("profile", lambda r: r["profile"]), ("mode", lambda r: r["mode"]),
        ("checkpoint sha256", lambda r: r["checkpoint"]["sha256"]),
        ("prompt ids", lambda r: r["prompt"]["ids"]), ("max_new_tokens", lambda r: r["max_new_tokens"]),
        ("harness sha256", lambda r: r["harness_sha256"]),
        ("mojo source sha256", lambda r: r["provenance"]["mojo_source_sha256"]),
        ("_mamba_impl.py sha256", lambda r: r["mojolearn"]["mamba_impl_sha256"]),
        ("_buffer.py sha256", lambda r: r["mojolearn"]["buffer_py_sha256"]),
    )
    bad = [(what, get(A), get(B)) for what, get in must_match if get(A) != get(B)]
    if A["mode"] != "identical":
        bad.append(("mode must be identical", A["mode"], B["mode"]))
    for name, r in (("A", A), ("B", B)):
        if not r["host"]["subnormal_free"]:
            bad.append((f"run {name} host path not subnormal-free", r["host"]["subnormal_counts"], None))
    if bad:
        for what, a, b in bad:
            print(f"NOT COMPARABLE: {what}: A={a!r} B={b!r}")
        return EXIT_REFUSED
    if A["provenance"].get("commit") != B["provenance"].get("commit"):
        print(f"NOTE: commits differ ({A['provenance'].get('commit')} vs {B['provenance'].get('commit')}); "
              "the Mojo sources, harness and Python layer hash equal, so the arithmetic source is the same")
    va, vb = A["mojolearn"]["binding"]["vendor_used"], B["mojolearn"]["binding"]["vendor_used"]
    print("CROSS-VENDOR comparison: %s vs %s" % (va, vb) if va != vb else
          "SAME VENDOR (%s): a repeat-run check, not a cross-vendor claim" % va)

    all_ok = True

    def line(label, a, b):
        nonlocal all_ok
        same = a == b
        all_ok &= same
        print(f"  {'IDENTICAL' if same else 'DIVERGENT'}  {label}")
        return same

    pa, pb = A["prompt"], B["prompt"]
    print("prompt prefill:")
    line("embedding input", pa["embedding_sha256"], pb["embedding_sha256"])
    first_bad = None
    for i, (ha, hb) in enumerate(zip(pa["layers_sha256"], pb["layers_sha256"])):
        if not line(f"layer {i:2d} output", ha, hb) and first_bad is None:
            first_bad = i
    line("final hidden (norm_f input)", pa["final_hidden_sha256"], pb["final_hidden_sha256"])
    line("norm_f output (host)", pa["norm_f_sha256"], pb["norm_f_sha256"])
    line("logits, every prompt position (host)", pa["logits_all_positions_sha256"], pb["logits_all_positions_sha256"])
    print(f"  first divergent layer: {first_bad if first_bad is not None else 'none'}")
    print("greedy steps:")
    for sa, sb in zip(A["steps"], B["steps"]):
        k = sa["step"]
        lb = next((i for i, (x, y) in enumerate(zip(sa["layers_sha256"], sb["layers_sha256"])) if x != y), None)
        line(f"step {k:2d} layers (first divergent: {lb if lb is not None else 'none'})",
             sa["layers_sha256"], sb["layers_sha256"])
        line(f"step {k:2d} logits", sa["logits_last_sha256"], sb["logits_last_sha256"])
        if not line(f"step {k:2d} token {sa['token_id']} vs {sb['token_id']}", sa["token_id"], sb["token_id"]):
            print("  the sequences differ from here; later steps are not comparable")
            break
    line("continuation text", A["continuation_text"], B["continuation_text"])
    print("RESULT: IDENTICAL" if all_ok else "RESULT: DIVERGENT")
    return EXIT_OK if all_ok else EXIT_DIVERGENT


def write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model-dir", help="directory with pytorch_model.bin and config.json "
                                        "(default: the local HF snapshot)")
    ap.add_argument("--tokenizer", help="gpt-neox-20b tokenizer.json (default: the local HF snapshot)")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--prompt-ids", help="comma-separated ids; bypasses the tokenizer for encoding")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--mode", choices=("identical", "fast"), default="identical")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--save-logits", action="store_true",
                    help="also write <out>_logits.npz (needed by --torch-reference-only)")
    ap.add_argument("--torch-reference", action="store_true",
                    help="after the run, HF Mamba2ForCausalLM on the CPU as a sanity check (needs torch)")
    ap.add_argument("--torch-reference-only", metavar="RESULT_JSON",
                    help="no GPU, no mojolearn: the torch sanity check against a saved run")
    ap.add_argument("--compare", nargs=2, metavar=("A_JSON", "B_JSON"),
                    help="pure comparison of two result files; exit 0 only if everything is identical")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare)
    if args.torch_reference_only:
        return torch_reference_only(args)
    return identity_run(args)


if __name__ == "__main__":
    sys.exit(main())
