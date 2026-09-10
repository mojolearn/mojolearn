# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE OPPONENT ARM for the sequence-model speed lane: torch on CUDA, and
`mamba-ssm`'s fused `selective_scan_cuda` where it is installable.

    python3 tools/speed_torch_seq.py --lane transformer
    python3 tools/speed_torch_seq.py --lane mamba --size smoke
    python3 tools/speed_torch_seq.py --lane attention --dump-dir /tmp/seqdump

The September 5 admission fixes were source-reviewed only by their author.
Main-lane validation is required before publishing their measurements.

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
Read `bench/speed/README.md` before quoting any ratio from this. The
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

THE DETERMINISTIC ARM (DEVIATION 2101; 2026-09-07, the NVIDIA identity-cost grid)
=================================================================================
`--arm torch-deterministic` runs every lane under PyTorch's DOCUMENTED
deterministic configuration and suffixes EVERY arm name this process prints
with `-deterministic` (`torch-gpu-fp32-deterministic`,
`mamba-ssm-cuda-deterministic`, the mamba2 lane's `torch-deterministic` and
`mamba-ssm-deterministic`, ...); the header arm is `torch-deterministic`.
`--arm torch`, the default, is byte-for-byte the process this file was
before the flag existed.

What "documented" means, fetched 2026-09-07:

  * https://pytorch.org/docs/stable/notes/randomness.html (it now redirects
    to https://docs.pytorch.org/docs/2.14/notes/randomness.html), under
    "Avoiding nondeterministic algorithms": "torch.use_deterministic_algorithms()
    lets you configure PyTorch to use deterministic algorithms instead of
    nondeterministic ones where available, and to throw an error if an
    operation is known to be nondeterministic (and without a deterministic
    alternative)." and its warning: "Deterministic operations are often
    slower than nondeterministic operations, so single-run performance may
    decrease for your model." Under "CUDA convolution benchmarking":
    "Disabling the benchmarking feature with `torch.backends.cudnn.benchmark
    = False` causes cuDNN to deterministically select an algorithm, possibly
    at the cost of reduced performance." Under "CUDA convolution
    determinism": "While disabling CUDA convolution benchmarking (discussed
    above) ensures that CUDA selects the same algorithm each time an
    application is run, that algorithm itself may be nondeterministic,
    unless either `torch.use_deterministic_algorithms(True)` or
    `torch.backends.cudnn.deterministic = True` is set."
  * The cuBLAS workspace sentence is NOT on that notes page (2.14); it is on
    the API page of the function, fetched at the torch the 2026-08-28 leg
    installed (2.4.1+cu124):
    https://docs.pytorch.org/docs/2.4/generated/torch.use_deterministic_algorithms.html
    "A handful of CUDA operations are nondeterministic if the CUDA version
    is 10.2 or greater, unless the environment variable
    `CUBLAS_WORKSPACE_CONFIG=:4096:8` or `CUBLAS_WORKSPACE_CONFIG=:16:8` is
    set." and "If one of these environment variable configurations is not
    set, a RuntimeError will be raised from these operations when called
    with CUDA tensors". The 2.14 page of the same function keeps: "When
    enabled, operations will use deterministic algorithms when available,
    and if only nondeterministic algorithms are available they will throw a
    RuntimeError when called."

So the arm is exactly this: `CUBLAS_WORKSPACE_CONFIG=:4096:8` in
`os.environ` BEFORE torch is imported -- `--arm` is read off `sys.argv` at
the top of this module, before any import that could pull torch in, and if
torch is somehow already in `sys.modules` the process re-executes itself
with the variable set rather than run on a runtime initialized without it
-- then `torch.use_deterministic_algorithms(True)`,
`torch.backends.cudnn.deterministic = True`,
`torch.backends.cudnn.benchmark = False`, each READ BACK before the first
arm runs. FP32 matmul precision is untouched: the fp32 AND tf32 sub-arms
run exactly as under the fast arm. A torch op with no deterministic
implementation raises RuntimeError; that is printed as `FSPEED-REFUSED
lane=<L> arm=torch-deterministic reason=<the error's first line>` and the
run continues with the next sub-arm or row (DEVIATION 2113).

DEVIATION 2102: the mamba corpus is loaded from `main`, not at import. The
corpus imports torch at ITS module scope, so a module-scope load here would
have imported torch before the environment variable above could be placed;
and `bench/speed/seq_py_speed_arm.py` (our arm, through the public Python
API, DEVIATION 2100) imports THIS file for the shape parser, the witness and
the Python-side row table and must not drag torch into the process that
holds MAX's runtime. Output under `--arm torch` is unchanged.

THE PYTHON-SIDE ROWS (DEVIATION 2104)
=====================================
The identity-cost grid is LARGE ONLY, in a NARROW and a WIDE variant, on
seed-7 bytes shared by both arms: `narrow.b8_l4096_d512` (B 8, L 4096,
d_model 512) and `wide.b8_l1024_d2048` (B 8, L 1024, d_model 2048). They are
NOT rows of `bench/speed/seq_speed_main.mojo` -- that ladder is untouched
and still parsed -- but of `py_rows()` here, appended to the `transformer`
and `mamba` lanes (only those two; the sub-lanes have no Python ours arm)
and forming the whole `shipped` set of the new `mamba2` and `mamba3` lanes
(whose `smoke` row is the corpus shape `lane.b2_l4_d32`). For the
transformer rows head_dim is 128 with Llama's 4:1 GQA and 3.5x MLP:
d_model 512 -> 4 heads / 1 kv head / intermediate 1792; d_model 2048 -> 16
/ 4 / 7168. Seed 7 is the hash-spec seed for every tensor of these rows,
with the corpus tensor ids and ranges of each family, so our arm and this
one generate the same bytes and print the same witnesses. Bounds are
CHECKED, never shrunk (`check_bounds`): a row outside what the API admits
is refused by name on both sides.

DEVIATION 2105: `hashed_tensor_numpy` is a pure-numpy copy of the hash
spec, because our arm cannot import the corpus (torch) and must still
generate the same bytes. It is CROSS-CHECKED against the corpus primitives
(`gen`) at every start of this process, bitwise, and the transcribed
Mamba-1/2/3 tensor ids, ranges, shapes and profile constants are
cross-checked against the corpus's own functions the same way; a
disagreement kills the process before a number is printed.

THE MAMBA-2 AND MAMBA-3 LANES (DEVIATIONS 2106, 2107, 2108, 2114)
=================================================================
The torch references are written from the corpus's own staged forwards
(`m2_forward`, `m3_forward` in `mamba/corpus/gen_corpus.py`), stage for
stage, in plain torch on the device, with two spellings changed for the
shipped shapes and said so: (2106) the Mamba-2 chunk scan uses einsum
contractions -- `ssd_minimal_discrete`'s spelling, which the corpus also
carries verbatim and reports at roundoff-scale agreement -- instead of HF's
broadcast-and-sum, whose `[B, C, Q, Q, H, P]` intermediate is 68 GB at the
narrow row; (2107) the Mamba-3 angle recurrence is cumsum-then-mod, the
upstream `mamba3_siso_fwd_ref` placement (:252-257), instead of the
corpus's per-token serial mod -- equal in exact arithmetic, a Python loop
of 4096 tiny launches otherwise. Neither reference is a bitwise oracle and
`FSPEED-AGREE` (2114, computed against our arm's dump exactly as the other
lanes do) is a tolerance report, never a gate. The llama cross-check is
skipped by name on these lanes.

(2108) Incumbents. For `mamba2` the strongest incumbent is `mamba_ssm`'s
fused SSD, arm `mamba-ssm`: the same torch prefix (block norm, in_proj,
conv1d -- `causal_conv1d_fn` when that wheel is importable, else
`F.conv1d` -- SiLU, splits), then
`mamba_ssm.ops.triton.ssd_combined.mamba_chunk_scan_combined(x, dt_raw, A,
B, C, chunk_size=256, D=D, z=None, dt_bias=dt_bias, dt_softplus=True)`
exactly as `mamba_ssm/modules/mamba2.py::Mamba2.forward` calls it at
rmsnorm=True, then the gated norm (`layernorm_gated.rmsnorm_fn` when
importable, else torch) and out_proj + residual in torch; every note says
which pieces ran. Triton's `tl.dot` on fp32 operands defaults to TF32
input precision unless a kernel opts out; whether the pinned SSD kernels
opt out was NOT verified here, so that arm's precision is the incumbent's
own and is recorded as such, not corrected. When `mamba_ssm` is not
importable the arm refuses by name, as the mamba lane's does. For `mamba3`
there is NO FP32 incumbent: the pinned `mamba_ssm` ships Mamba-3 Triton
kernels but its surface (`mamba3_siso_combined.py`:390-399, contract
section 1) force-casts Q/K/V/Trap/Angles/Z to bfloat16, which is a
different precision; arm `torch` only, and the note says so.
"""

import argparse
import hashlib
import os
import re
import statistics
import sys
import time

for _thread_env in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_thread_env, '2')


# ---------------------------------------------------------------------------
# DEVIATION 2101: `--arm` is read BEFORE ANY IMPORT THAT COULD PULL TORCH IN,
# because the vendor's deterministic configuration is an environment variable
# the runtime reads when its cuBLAS handle is created and argparse runs far
# too late for that. This is the only place in the file that looks at
# sys.argv by hand; `main`'s argparse sees the same flag and checks that the
# two agree. When this module is IMPORTED (by our arm, DEVIATION 2102) the
# argv is the importer's, which has no --arm, so nothing here fires.
# ---------------------------------------------------------------------------
CUBLAS_WORKSPACE = ":4096:8"


def _arm_from_argv(argv):
    for i, a in enumerate(argv):
        if a == "--arm" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--arm="):
            return a[len("--arm="):]
    return "torch"


ARM = _arm_from_argv(sys.argv[1:])
DETERMINISTIC = ARM == "torch-deterministic"
ARM_SUFFIX = "-deterministic" if DETERMINISTIC else ""

if DETERMINISTIC:
    if "torch" in sys.modules and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != CUBLAS_WORKSPACE:
        # torch is already alive in this interpreter without the variable:
        # a runtime initialized without it cannot be trusted to honor it, so
        # start over with it in place.
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE
        os.execv(sys.executable, [sys.executable] + sys.argv)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE

AGREEMENT_FAILURES = []
INPUT_WITNESSES = {}
AGREE_RTOL = 5e-4
AGREE_ATOL = 1e-5

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
#: `transformer/checks/transformer_fixture.mojo::TID_*`.
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


#: `mamba/corpus/gen_corpus.py`, loaded by `_load_corpus()` FROM `main`,
#: after torch has been imported with the arm's environment in place.
#: DEVIATION 2102 (the docstring): a module-scope load here imported torch
#: as a side effect, which is wrong for the deterministic arm and fatal for
#: the process our own arm runs in. Every function below that reads it runs
#: only after `main` has loaded it.
mamba_corpus = None


def _load_corpus():
    global mamba_corpus
    if mamba_corpus is None:
        try:
            mamba_corpus = _load_module(
                "mojolearn_mamba_corpus",
                os.path.join(REPO, "mamba", "corpus", "gen_corpus.py"))
        except Exception as e:                  # pragma: no cover
            raise SystemExit("speed_torch_seq: cannot import mamba/corpus/gen_corpus.py "
                             "(needed for the hash spec and the reference scan): %s" % e)
    return mamba_corpus


import math                                     # noqa: E402
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


# ---------------------------------------------------------------------------
# DEVIATION 2105: the hash spec in pure numpy, for the process that cannot
# import the corpus. `_check_generator_copy` compares it bitwise with `gen`
# (the corpus-primitive path) every time this process starts.
# ---------------------------------------------------------------------------
_SM_GAMMA = 0x9E3779B97F4A7C15
_SM_M1 = 0xBF58476D1CE4E5B9
_SM_M2 = 0x94D049BB133111EB


def _splitmix64_scalar(z):
    z = (z + _SM_GAMMA) & M64
    z = ((z ^ (z >> 30)) * _SM_M1) & M64
    z = ((z ^ (z >> 27)) * _SM_M2) & M64
    return z ^ (z >> 31)


def _splitmix64_array(z):
    z = z.astype(np.uint64)
    with np.errstate(over="ignore"):
        z = z + np.uint64(_SM_GAMMA)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(_SM_M1)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(_SM_M2)
        return z ^ (z >> np.uint64(31))


def _is_dyadic(v):
    from fractions import Fraction
    den = Fraction(v).denominator
    return den & (den - 1) == 0


def _map_range_numpy(f_unit, lo, hi):
    """`map_range_fast` without the corpus: float64, rounded once, the
    dyadic precondition asserted and the exactness spot-checked on the
    same 256-element sample."""
    from fractions import Fraction
    assert _is_dyadic(lo) and _is_dyadic(hi), (lo, hi)
    span = float(hi) - float(lo)
    v64 = float(lo) + span * f_unit
    flo, fspan = Fraction(lo), Fraction(hi) - Fraction(lo)
    step = max(1, f_unit.size // 256)
    for j in range(0, f_unit.size, step):
        fv = float(f_unit[j])
        assert Fraction(float(v64[j])) == flo + fspan * Fraction(fv), (
            "float64 evaluation is not exact", lo, hi, fv, v64[j])
    return v64.astype(np.float32)


def hashed_tensor_numpy(seed, tid, n, lo, hi, chunk=1 << 22):
    """`gen`, spelled with the local splitmix64 instead of the corpus's.
    Same chunking, same float64 map, same single rounding. DEVIATION 2105."""
    key = _splitmix64_scalar((int(seed) ^ (int(tid) << 32)) & M64)
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        idx = np.arange(s, e, dtype=np.uint64)
        with np.errstate(over="ignore"):
            h = _splitmix64_array(np.uint64(key) + idx)
        f = (h >> np.uint64(40)).astype(np.float64) * (2.0 ** -24)
        out[s:e] = _map_range_numpy(f, lo, hi)
        del idx, h, f
    return out


def _check_generator_copy():
    """Bitwise agreement of `hashed_tensor_numpy` with `gen` on three
    tensors spanning two chunks, a fan-in range and the x range; the
    process dies on the first disagreement. Requires the corpus loaded."""
    for tid, n, lo, hi in ((1, (1 << 22) + 777, -2.0, 2.0),
                           (23, 4097, -fan_in_scale(2048), fan_in_scale(2048)),
                           (44, 33, -7.0, -2.0)):
        a = gen(PY_SEED, tid, n, lo, hi)
        b = hashed_tensor_numpy(PY_SEED, tid, n, lo, hi)
        if a.tobytes() != b.tobytes():
            raise SystemExit("speed_torch_seq: hashed_tensor_numpy disagrees with the "
                             "corpus generator at tid=%d n=%d range=(%r, %r); the "
                             "two arms would not see the same bytes (DEVIATION 2105)"
                             % (tid, n, lo, hi))


# ---------------------------------------------------------------------------
# THE PYTHON-SIDE ROW TABLE (DEVIATION 2104) AND THE PROFILE CONSTANTS,
# TENSOR IDS, RANGES AND SHAPES IT NEEDS, transcribed from
# `mamba/corpus/gen_corpus.py` and `python/mojolearn/_mamba_impl.py` and
# CROSS-CHECKED against the corpus by `_check_transcriptions` (DEVIATION
# 2105). Both arms read this table and nothing else for these rows.
# ---------------------------------------------------------------------------
PY_SEED = 7
TAG_NARROW = "narrow.b8_l4096_d512"
TAG_WIDE = "wide.b8_l1024_d2048"
TAG_M23_SMOKE = "lane.b2_l4_d32"
PY_NARROW = dict(b=8, l=4096, d_model=512)
PY_WIDE = dict(b=8, l=1024, d_model=2048)
FAM_M23 = 2

#: Llama-3-8B's head_dim and GQA ratio, applied to the two d_models; the
#: 3.5x MLP is 14336/4096.
LLAMA_HEAD_DIM = 128
LLAMA_GQA = 4
#: `_transformer_impl._MAX_ABS_POSITION` / modeling_llama.mojo (DEVIATION 812).
MAX_ABS_POSITION = 8192

M1_D_STATE, M1_D_CONV, M1_EXPAND = 16, 4, 2
M1_TENSOR_IDS = {
    "x": 1, "in_proj.weight": 2, "conv1d.weight": 3, "conv1d.bias": 4,
    "x_proj.weight": 5, "dt_proj.weight": 6, "dt_proj.bias": 7, "A_log": 8,
    "D": 9, "out_proj.weight": 10, "norm.weight": 11,
}
#: `Mamba1Block._W_NAMES`, the constructor's key set.
M1_NAMES = ("norm.weight", "in_proj.weight", "conv1d.weight", "conv1d.bias",
            "x_proj.weight", "dt_proj.weight", "dt_proj.bias", "A_log", "D",
            "out_proj.weight")

M2_D_STATE, M2_D_CONV, M2_EXPAND, M2_HEADDIM, M2_NGROUPS, M2_CHUNK = 128, 4, 2, 64, 1, 256
M2_EPS = 1e-5
M2_TENSOR_IDS = {
    "x": 21, "block_norm.weight": 22, "in_proj.weight": 23, "conv1d.weight": 24,
    "conv1d.bias": 25, "dt_bias": 26, "A_log": 27, "D": 28, "norm.weight": 29,
    "out_proj.weight": 30,
}
#: `Mamba2Block._W_NAMES`.
M2_NAMES = ("block_norm.weight", "in_proj.weight", "conv1d.weight", "conv1d.bias",
            "dt_bias", "A_log", "D", "norm.weight", "out_proj.weight")

M3_D_STATE, M3_EXPAND, M3_HEADDIM, M3_NGROUPS, M3_CHUNK, M3_ROPE = 128, 2, 64, 1, 64, 32
M3_EPS = 1e-5
M3_A_FLOOR = 1e-4
M3_TENSOR_IDS = {
    "x": 41, "block_norm.weight": 42, "in_proj.weight": 43, "dt_bias": 44,
    "B_norm.weight": 45, "C_norm.weight": 46, "B_bias": 47, "C_bias": 48,
    "D": 49, "out_proj.weight": 50,
}
#: `Mamba3Block._W_NAMES`.
M3_NAMES = ("block_norm.weight", "in_proj.weight", "dt_bias", "B_norm.weight",
            "C_norm.weight", "B_bias", "C_bias", "D", "out_proj.weight")


def fan_in_scale(fan_in):
    """`gen_corpus.py::fan_in_scale`: 0.5 / 2^ceil(log2(fan_in)/2)."""
    return 0.5 / (2 ** math.ceil(math.log2(fan_in) / 2))


def m1_shapes(dm, b, l):
    di = M1_EXPAND * dm
    r = (dm + 15) // 16
    return {
        "x": (b, l, dm), "in_proj.weight": (2 * di, dm),
        "conv1d.weight": (di, 1, M1_D_CONV), "conv1d.bias": (di,),
        "x_proj.weight": (r + 2 * M1_D_STATE, di), "dt_proj.weight": (di, r),
        "dt_proj.bias": (di,), "A_log": (di, M1_D_STATE), "D": (di,),
        "out_proj.weight": (dm, di), "norm.weight": (dm,),
    }


def m1_ranges(dm):
    di = M1_EXPAND * dm
    s_in, s_x, s_out = fan_in_scale(dm), fan_in_scale(di), fan_in_scale(di)
    return {
        "x": (-2.0, 2.0), "norm.weight": (0.5, 1.5), "in_proj.weight": (-s_in, s_in),
        "conv1d.weight": (-0.5, 0.5), "conv1d.bias": (-0.125, 0.125),
        "x_proj.weight": (-s_x, s_x), "dt_proj.weight": (-1.0, 1.0),
        "dt_proj.bias": (-7.0, -2.0), "A_log": (0.0, 2.75), "D": (0.5, 1.5),
        "out_proj.weight": (-s_out, s_out),
    }


def m2_shapes(dm, b, l):
    di = M2_EXPAND * dm
    H = di // M2_HEADDIM
    CD = di + 2 * M2_NGROUPS * M2_D_STATE
    dip = 2 * di + 2 * M2_NGROUPS * M2_D_STATE + H
    return {
        "x": (b, l, dm), "block_norm.weight": (dm,), "in_proj.weight": (dip, dm),
        "conv1d.weight": (CD, 1, M2_D_CONV), "conv1d.bias": (CD,), "dt_bias": (H,),
        "A_log": (H,), "D": (H,), "norm.weight": (di,), "out_proj.weight": (dm, di),
    }


def m2_ranges(dm):
    di = M2_EXPAND * dm
    s_in, s_out = fan_in_scale(dm), fan_in_scale(di)
    return {
        "x": (-2.0, 2.0), "block_norm.weight": (0.5, 1.5), "in_proj.weight": (-s_in, s_in),
        "conv1d.weight": (-0.5, 0.5), "conv1d.bias": (-0.125, 0.125),
        "dt_bias": (-7.0, -2.0), "A_log": (0.0, 2.75), "D": (0.5, 1.5),
        "norm.weight": (0.5, 1.5), "out_proj.weight": (-s_out, s_out),
    }


def m3_shapes(dm, b, l):
    di = M3_EXPAND * dm
    H = di // M3_HEADDIM
    dip = 2 * di + 2 * M3_NGROUPS * M3_D_STATE + 3 * H + M3_ROPE
    return {
        "x": (b, l, dm), "block_norm.weight": (dm,), "in_proj.weight": (dip, dm),
        "dt_bias": (H,), "B_norm.weight": (M3_D_STATE,), "C_norm.weight": (M3_D_STATE,),
        "B_bias": (H, M3_D_STATE), "C_bias": (H, M3_D_STATE), "D": (H,),
        "out_proj.weight": (dm, di),
    }


def m3_ranges(dm):
    di = M3_EXPAND * dm
    s_in, s_out = fan_in_scale(dm), fan_in_scale(di)
    return {
        "x": (-2.0, 2.0), "block_norm.weight": (0.5, 1.5), "in_proj.weight": (-s_in, s_in),
        "dt_bias": (-7.0, -2.0), "B_norm.weight": (0.5, 1.5), "C_norm.weight": (0.5, 1.5),
        "B_bias": (0.5, 1.5), "C_bias": (0.5, 1.5), "D": (0.5, 1.5),
        "out_proj.weight": (-s_out, s_out),
    }


def llama_spec(row):
    """The nine llama weights as (name, n, lo, hi, shape): the ranges
    `seq_speed_main.mojo::LlamaHostWeights` transcribes from the transformer
    corpus, in the order the witnesses print. `x` and `ctx.x` are (-2, 2)."""
    dm = row["d_model"]
    H, HKV, hd, it = row["n_heads"], row["n_kv"], row["head_dim"], row["intermediate"]
    qw, kw = H * hd, HKV * hd
    s_o = fan_in_scale(qw)
    s_d = fan_in_scale(it)
    return [
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


def py_rows(kind):
    """The Python-side rows for `kind` in {llama, mamba1, mamba2, mamba3}.

    Every row carries the SAME field names the parsed driver rows carry
    (so the llama and mamba runners take either), plus `seed` (7 here;
    the driver rows get `seq_seed`) and `kind`. `i` is -1: these rows
    have no index in the Mojo ladder and `--row` cannot select them."""
    fam = {"llama": FAM_LLAMA, "mamba1": FAM_MAMBA}.get(kind, FAM_M23)

    def base(name, b, l, dm, smoke):
        return dict(i=-1, family=fam, kind=kind, name=name, b=b, l=l, ctx=0,
                    d_model=dm, n_heads=0, n_kv=0, head_dim=0, intermediate=0,
                    smoke=smoke, seed=PY_SEED)

    rows = []
    if kind in ("mamba2", "mamba3"):
        rows.append(base(TAG_M23_SMOKE, 2, 4, 32, 1))
    for tag, s in ((TAG_NARROW, PY_NARROW), (TAG_WIDE, PY_WIDE)):
        r = base(tag, s["b"], s["l"], s["d_model"], 0)
        if kind == "llama":
            nh = s["d_model"] // LLAMA_HEAD_DIM
            r.update(head_dim=LLAMA_HEAD_DIM, n_heads=nh,
                     n_kv=max(1, nh // LLAMA_GQA), intermediate=(s["d_model"] * 7) // 2)
        rows.append(r)
    return rows


def check_bounds(kind, row):
    """None when the row is inside what the public API admits, else the
    refusal text. The rule is CHECK AND REFUSE, never shrink: a shape the
    API cannot take is a finding, and a silently smaller shape would put a
    number under the wrong tag on both sides."""
    b, l, dm = row["b"], row["l"], row["d_model"]
    if b < 1 or l < 1 or dm < 1:
        return "B, L and d_model must be positive, got B=%d L=%d d_model=%d" % (b, l, dm)
    if kind == "llama":
        H, HKV, hd = row["n_heads"], row["n_kv"], row["head_dim"]
        if H < 1 or HKV < 1 or hd < 1:
            return "n_heads, n_kv_heads and head_dim must be positive"
        if dm != H * hd:
            return "d_model must equal n_heads*head_dim (LlamaDims.validate), got %d vs %d*%d" % (dm, H, hd)
        if H % HKV != 0:
            return "n_heads %% n_kv_heads must be 0 (repeat_kv), got %d %% %d" % (H, HKV)
        if hd % 2 != 0:
            return "head_dim must be even (rotate_half), got %d" % hd
        if row["ctx"] + l > MAX_ABS_POSITION:
            return ("absolute positions must stay below %d (DEVIATION 812), got ctx+L=%d"
                    % (MAX_ABS_POSITION, row["ctx"] + l))
    elif kind in ("mamba2", "mamba3"):
        if dm % 32 != 0:
            return ("d_model must be a multiple of 32 so nheads = 2*d_model/64 is whole "
                    "(Mamba%sDims.of); got %d" % (kind[-1], dm))
    return None


def _check_transcriptions():
    """The constants, ids, ranges and shapes above against the corpus's own
    functions, for the d_models the rows use; the process dies on any
    disagreement, the SEQ_SEED_BASE rule of `load_shapes` applied to the
    new table. Requires the corpus loaded."""
    c = mamba_corpus

    def die(what):
        raise SystemExit("speed_torch_seq: transcription of %s disagrees with "
                         "mamba/corpus/gen_corpus.py; fix the copy in this file "
                         "(DEVIATION 2105)" % what)

    if dict(c.TENSOR_IDS) != M1_TENSOR_IDS:
        die("M1_TENSOR_IDS")
    if {k: v for k, v in c.M2_TENSOR_IDS.items() if k in M2_TENSOR_IDS} != M2_TENSOR_IDS:
        die("M2_TENSOR_IDS")
    if {k: v for k, v in c.M3_TENSOR_IDS.items() if k in M3_TENSOR_IDS} != M3_TENSOR_IDS:
        die("M3_TENSOR_IDS")
    if (c.M2_D_STATE, c.M2_D_CONV, c.M2_EXPAND, c.M2_HEADDIM, c.M2_NGROUPS, c.M2_CHUNK, c.M2_EPS) != (
            M2_D_STATE, M2_D_CONV, M2_EXPAND, M2_HEADDIM, M2_NGROUPS, M2_CHUNK, M2_EPS):
        die("the Mamba-2 profile constants")
    if (c.M3_D_STATE, c.M3_EXPAND, c.M3_HEADDIM, c.M3_NGROUPS, c.M3_CHUNK, c.M3_ROPE,
            c.M3_EPS, c.M3_A_FLOOR) != (M3_D_STATE, M3_EXPAND, M3_HEADDIM, M3_NGROUPS,
                                        M3_CHUNK, M3_ROPE, M3_EPS, M3_A_FLOOR):
        die("the Mamba-3 profile constants")
    if (c.D_STATE, c.D_CONV, c.EXPAND, c.EPS) != (M1_D_STATE, M1_D_CONV, M1_EXPAND, MAMBA_EPS):
        die("the Mamba-1 profile constants")
    for dm in (8, 32, 512, 768, 2048):
        if c.fan_in_scale(dm) != fan_in_scale(dm) or c.fan_in_scale(2 * dm) != fan_in_scale(2 * dm):
            die("fan_in_scale")
        di, r = M1_EXPAND * dm, (dm + 15) // 16
        if c.default_ranges(dm, di, r, M1_D_STATE, M1_D_CONV) != m1_ranges(dm):
            die("m1_ranges(%d)" % dm)
        got = {k: tuple(v) for k, v in c.shapes_for(dm, di, r, M1_D_STATE, M1_D_CONV, 8, 16).items()}
        if got != m1_shapes(dm, 8, 16):
            die("m1_shapes(%d)" % dm)
        if dm % 32 == 0:
            r2 = {k: v for k, v in c.m2_default_ranges(dm).items() if k in M2_TENSOR_IDS}
            if r2 != m2_ranges(dm):
                die("m2_ranges(%d)" % dm)
            if {k: tuple(v) for k, v in c.m2_shapes_for(dm, 8, 16).items()} != m2_shapes(dm, 8, 16):
                die("m2_shapes(%d)" % dm)
            r3 = {k: v for k, v in c.m3_default_ranges(dm).items() if k in M3_TENSOR_IDS}
            if r3 != m3_ranges(dm):
                die("m3_ranges(%d)" % dm)
            if {k: tuple(v) for k, v in c.m3_shapes_for(dm, 8, 16).items()} != m3_shapes(dm, 8, 16):
                die("m3_shapes(%d)" % dm)


def fnv1a64(buf, initial=FNV_OFFSET):
    """FNV-1a64 over bytes, in order. `core/identity_trace.mojo`'s function."""
    h = initial
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
    return fnv1a64(a[::stride].tobytes(), h)


def emit_witness(lane, tag, name, arr, samples):
    a = np.ascontiguousarray(arr, dtype=np.float32).reshape(-1)
    INPUT_WITNESSES[(lane, tag, name)] = (str(a.size), hex16(witness_hash(a, samples)))
    print("FSPEED-WEIGHTS lane=%s shape=%s tensor=%s n=%d hash=%s"
          % (lane, tag, name, a.size, hex16(witness_hash(a, samples))))
    if a.nbytes <= 16 * 1024 * 1024:
        print('FSPEED-INPUT-SHA256 lane=%s shape=%s tensor=%s bytes=%d sha256=%s'
              % (lane, tag, name, a.nbytes, hashlib.sha256(a.tobytes()).hexdigest()))


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
    arm = _armname(arm)   # DEVIATION 2101: the suffix travels on every line
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
        sync(torch, _DEVSTR)
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
        print("FSPEED-NOTE lane=%s arm=%s %s"
              % (lane, arm, one_line("output witness shape=%s n=%d hash=%s"
                                     % (tag, a.size, hex16(witness_hash(a, WITNESS_SAMPLES))))))
    return last


def undeterminize(torch):
    """Put torch's global determinism switches back where THIS ARM needs
    them. DEVIATION 1856, and DEVIATION 2101 for the deterministic arm.

    BOTH `gen_corpus.py` files call `torch.use_deterministic_algorithms(True)`
    and `torch.set_num_threads(1)` AT IMPORT TIME. That is right for a corpus
    generator and wrong here twice over: deterministic algorithms select
    slower kernels, and on CUDA a deterministic matmul RAISES unless
    `CUBLAS_WORKSPACE_CONFIG` is set. Under `--arm torch` (the FAST arm on
    both sides) they are turned OFF. Under `--arm torch-deterministic` they
    are turned ON, cudnn included, so that the corpus import can neither arm
    nor disarm anything the flag did not ask for. It is called after EVERY
    import of a corpus, not once at startup, because the transformer corpus
    is imported lazily by the cross-check and would otherwise silently
    re-arm determinism for every timed round after it.
    """
    try:
        torch.use_deterministic_algorithms(DETERMINISTIC)
        if DETERMINISTIC:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        cap = max(1, min(2, int(os.environ.get('MOJOLEARN_CPU_THREADS', '2'))))
        torch.set_num_threads(cap)
    except Exception:
        pass


def deterministic_readback(torch):
    """DEVIATION 2101: the four switches, READ BACK, or a refusal that ends
    the process. A deterministic column whose switches did not take is a
    fast column under the wrong name, which is the one thing this campaign
    cannot survive."""
    got = dict(use_deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
               cudnn_deterministic=bool(torch.backends.cudnn.deterministic),
               cudnn_benchmark=bool(torch.backends.cudnn.benchmark),
               cublas_workspace_config=os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
    want = dict(use_deterministic_algorithms=True, cudnn_deterministic=True,
                cudnn_benchmark=False, cublas_workspace_config=CUBLAS_WORKSPACE)
    if got != want:
        raise SystemExit("speed_torch_seq: REFUSED. --arm torch-deterministic asked for %r "
                         "and read back %r" % (want, got))
    return got


def _armname(arm):
    """The arm as it is PRINTED: suffixed under the deterministic arm, except
    the `agree` pseudo-arm, which names a comparison and not a process."""
    return arm if arm == "agree" else arm + ARM_SUFFIX


def _is_nondeterministic_error(msg):
    m = str(msg).lower()
    return ("deterministic implementation" in m
            or "cublas_workspace_config" in m
            or "use_deterministic_algorithms" in m)


def refuse_exc(lane, arm, tag, e):
    """One sub-arm's exception as a refusal.

    DEVIATION 2113: under the deterministic arm an op with no deterministic
    implementation is printed under the PROCESS arm name (`torch-deterministic`)
    with the torch error's first line as the whole reason, which is the
    line the table reader needs; the sub-arm and the shape go on a note. The
    row then continues. Every other exception keeps the existing spelling."""
    if DETERMINISTIC and _is_nondeterministic_error(e):
        first = str(e).splitlines()[0] if str(e) else type(e).__name__
        note(lane, arm, "no deterministic implementation at shape %s; the refusal "
                        "below is this sub-arm's" % tag)
        refuse(lane, "torch", first)
        return
    refuse(lane, arm, "%s at shape %s: %s" % (type(e).__name__, tag, str(e)[:180]))


def _dump_path(args, lane, tag):
    """Our arm's raw output for `FSPEED-AGREE`. The `mamba` lane also
    accepts a dump written under the leg's `mamba1` spelling (DEVIATION
    2103 in `bench/speed/seq_py_speed_arm.py`) when no `mamba` dump exists."""
    if not args.dump_dir:
        return ""
    p = os.path.join(args.dump_dir, "seq.%s.%s.f32.bin" % (lane, tag))
    if lane == "mamba" and not os.path.exists(p):
        alt = os.path.join(args.dump_dir, "seq.mamba1.%s.f32.bin" % tag)
        if os.path.exists(alt):
            return alt
    return p


def one_line(s):
    """Collapse a message to ONE line.

    `FSPEED-REFUSED ... reason=<one line>` is a one-line contract and a
    torch exception is routinely several, including a whole CUDA backtrace.
    A multi-line reason would put unparseable text into the middle of the
    table."""
    return " ".join(str(s).split())


def refuse(lane, arm, reason):
    print("FSPEED-REFUSED lane=%s arm=%s reason=%s" % (lane, _armname(arm), one_line(reason)))


def note(lane, arm, text):
    print("FSPEED-NOTE lane=%s arm=%s %s" % (lane, _armname(arm), one_line(text)))


# ---------------------------------------------------------------------------
# FSPEED-AGREE against the Mojo dump
# ---------------------------------------------------------------------------
def agree(lane, tag, ours_path, theirs, arm='reference-fp32'):
    """Compare the Mojo driver's dumped output with the eager FP32 arm's.

    A tolerance gate, not bitwise cross-library agreement. What this catches is
    the failure where a speed number was taken for a block that computes
    something else entirely, which is the failure that makes a whole
    benchmark worthless without ever looking wrong.
    """
    if not ours_path or not os.path.exists(ours_path):
        AGREEMENT_FAILURES.append('%s/%s/%s missing Mojo dump' % (lane, tag, arm))
        refuse(lane, "agree", "no mojo dump at %s (set MOJOLEARN_SPEED_DUMP_DIR "
                              "on the mojo side and --dump-dir here)" % ours_path)
        return
    ours = np.fromfile(ours_path, dtype=np.float32)
    th = np.ascontiguousarray(theirs, dtype=np.float32).reshape(-1)
    if ours.size != th.size:
        AGREEMENT_FAILURES.append('%s/%s/%s output size mismatch' % (lane, tag, arm))
        refuse(lane, "agree", "shape %s: mojo dumped %d floats, torch produced %d"
               % (tag, ours.size, th.size))
        return
    d = np.abs(ours.astype(np.float64) - th.astype(np.float64))
    den = np.maximum(np.abs(th.astype(np.float64)), 1e-30)
    print("FSPEED-AGREE lane=%s max_abs_diff=%.6g max_rel_diff=%.6g n=%d"
          % (lane, float(d.max()) if d.size else 0.0,
             float((d / den).max()) if d.size else 0.0, ours.size))
    passed = bool(np.isfinite(ours).all() and np.isfinite(th).all()
                  and np.allclose(ours, th, rtol=AGREE_RTOL, atol=AGREE_ATOL))
    print('FSPEED-AGREE-GATE lane=%s shape=%s arm=%s passed=%s rtol=%g atol=%g'
          % (lane, tag, _armname(arm), str(passed).lower(), AGREE_RTOL, AGREE_ATOL))
    if not passed:
        AGREEMENT_FAILURES.append('%s/%s/%s numerical agreement failed' % (lane, tag, arm))


# ---------------------------------------------------------------------------
# The llama lanes
# ---------------------------------------------------------------------------
def llama_weights(torch, dev, row, seed, samples, lane, tag):
    dm = row["d_model"]
    # The (name, n, lo, hi, shape) table is `llama_spec` so our arm reads the
    # same one (DEVIATION 2104); `fan_in_scale` is the local copy, checked
    # against the corpus's by `_check_transcriptions`.
    spec = llama_spec(row)
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
    seed = row["seed"]   # the driver's `seq_seed(i)`, or 7 on a Python-side row
    samples = consts["witness_samples"]
    bad = check_bounds("llama", row)
    if bad:
        refuse(lane, "torch", "shape %s refused, not shrunk: %s" % (tag, bad))
        return
    W, x, xc = llama_weights(torch, dev, row, seed, samples, lane, tag)
    cfg = dict(n_heads=row["n_heads"], n_kv=row["n_kv"], head_dim=row["head_dim"],
               intermediate=row["intermediate"], d_model=dm, ctx=ctxlen, l=L)

    dumped = _dump_path(args, lane, tag)
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
            refuse_exc(lane, arm, tag, e)

    if eager_out is not None:
        agree(lane, tag, dumped, eager_out.detach().to(torch.float32).contiguous().cpu().numpy())
    else:
        AGREEMENT_FAILURES.append('%s/%s eager FP32 output unavailable' % (lane, tag))
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
    seed = row["seed"]   # the driver's `seq_seed(i)`, or 7 on a Python-side row
    samples = consts["witness_samples"]
    P, x, di, dt_rank = mamba_params(torch, dev, row, seed, samples, lane, tag)
    dumped = _dump_path(args, lane, tag)

    fused = None
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn as fused
    except Exception as e:
        if args.require_fused:
            AGREEMENT_FAILURES.append('Required mamba-ssm CUDA comparator unavailable')
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
            refuse_exc(lane, arm, tag, e)

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
                def call():
                    # The complete block starts with normalization/projection,
                    # convolution and delta/B/C construction in BOTH arms.
                    u, delta, A, Bt, Ct, Dv, gate = mamba_prefix(
                        torch, P, x, di, dt_rank)
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
            fused_out = time_arm(torch, lane, arm, tag, call, args.rounds, args.warmups, pick)
            agree(lane, tag, dumped,
                  fused_out.detach().to(torch.float32).contiguous().cpu().numpy(), arm)
        except Exception as e:
            if args.require_fused:
                AGREEMENT_FAILURES.append('Required mamba-ssm CUDA comparator failed at ' + tag)
            refuse_exc(lane, arm, tag, e)

    if ref_out is not None:
        agree(lane, tag, dumped, ref_out.detach().to(torch.float32).contiguous().cpu().numpy())
    else:
        AGREEMENT_FAILURES.append('%s/%s reference output unavailable' % (lane, tag))
        refuse(lane, "agree", "the reference arm did not produce an output at "
                              "shape %s" % tag)


# ---------------------------------------------------------------------------
# The Mamba-2 and Mamba-3 lanes (DEVIATIONS 2106, 2107, 2108, 2114; the
# docstring). References written from `mamba/corpus/gen_corpus.py::m2_forward`
# and `::m3_forward`, stage for stage; citations are theirs.
# ---------------------------------------------------------------------------
def m23_params(torch, dev, kind, row, seed, samples, lane, tag):
    dm, b, l = row["d_model"], row["b"], row["l"]
    if kind == "mamba2":
        ids, shapes, ranges, names = M2_TENSOR_IDS, m2_shapes(dm, b, l), m2_ranges(dm), M2_NAMES
    else:
        ids, shapes, ranges, names = M3_TENSOR_IDS, m3_shapes(dm, b, l), m3_ranges(dm), M3_NAMES
    P = {}
    for name in names:
        shape = shapes[name]
        n = int(np.prod(shape))
        lo, hi = ranges[name]
        flat = gen(seed, ids[name], n, lo, hi)
        emit_witness(lane, tag, name, flat, samples)
        P[name] = torch.from_numpy(flat.reshape(shape)).to(dev)
    lo, hi = ranges["x"]
    xn = gen(seed, ids["x"], b * l * dm, lo, hi)
    emit_witness(lane, tag, "x", xn, samples)
    x = torch.from_numpy(xn.reshape(b, l, dm)).to(dev)
    return P, x


def _segment_sum(torch, x):
    """HF `modeling_mamba2.py::segment_sum` (:73-90), the corpus's verbatim
    copy respelled without the `F` import: [..., T] -> [..., T, T]."""
    T = x.size(-1)
    xe = x[..., None].expand(*x.size(), T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1)
    seg = torch.cumsum(xe.masked_fill(~mask, 0), dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=0)
    return seg.masked_fill(~mask, -torch.inf)


def _pad_tokens(torch, t, pad):
    """Zero-pad dim 1 (tokens) at the end; the corpus's `m3_pad_tokens`."""
    if pad == 0:
        return t
    return torch.nn.functional.pad(t, [0, 0] * (t.dim() - 2) + [0, pad])


def m2_prefix(torch, P, x, conv_fn=None):
    """`m2_forward` S1-S7 and the splits: block RMSNorm (the mamba_ssm
    `1/sqrt` spelling), in_proj, the z | xBC | dt split, A, causal conv1d
    with bias, SiLU, the x | B | C split. `conv_fn` is `causal_conv1d_fn`
    when the incumbent's conv wheel is importable (the mamba-ssm arm only);
    None means `F.conv1d`, the torch fallback arm of mamba2.py:233."""
    F = torch.nn.functional
    Bsz, L, dm = x.shape
    di = M2_EXPAND * dm
    H = di // M2_HEADDIM
    N, G = M2_D_STATE, M2_NGROUPS
    CD = di + 2 * G * N
    sumsq = x.pow(2).sum(-1)
    rstd = 1 / torch.sqrt(sumsq / dm + M2_EPS)
    hnorm = P["block_norm.weight"] * (x * rstd[..., None])
    zxbcdt = F.linear(hnorm, P["in_proj.weight"])
    z, xBC_raw, dt_raw = torch.split(zxbcdt, [di, CD, H], dim=-1)
    A = -torch.exp(P["A_log"])
    xt = xBC_raw.transpose(1, 2)
    if conv_fn is not None:
        xBC = conv_fn(xt.contiguous(), P["conv1d.weight"].reshape(CD, M2_D_CONV),
                      bias=P["conv1d.bias"], activation="silu").transpose(1, 2)
    else:
        conv = F.conv1d(xt, P["conv1d.weight"], P["conv1d.bias"],
                        padding=M2_D_CONV - 1, groups=CD)[:, :, :L].transpose(1, 2)
        xBC = F.silu(conv)
    xs, Bs, Cs = torch.split(xBC, [di, G * N, G * N], dim=-1)
    Xh = xs.reshape(Bsz, L, H, M2_HEADDIM)
    return z, Xh, Bs.reshape(Bsz, L, G, N), Cs.reshape(Bsz, L, G, N), dt_raw, A


def m2_suffix(torch, P, x, y, z, gated_fn=None):
    """`m2_forward` S21-S22: gated RMSNorm with the gate BEFORE the norm
    (norm_before_gate=False, DEVIATION 787), out_proj, residual. `gated_fn`
    is `layernorm_gated.rmsnorm_fn` on the mamba-ssm arm when importable."""
    F = torch.nn.functional
    Bsz, L, dm = x.shape
    di = M2_EXPAND * dm
    y_flat = y.reshape(Bsz, L, di)
    if gated_fn is not None:
        gout = gated_fn(y_flat, P["norm.weight"], None, z=z, eps=M2_EPS,
                        group_size=None, norm_before_gate=False)
    else:
        gate = y_flat * F.silu(z)
        grstd = 1 / torch.sqrt(gate.pow(2).sum(-1) / di + M2_EPS)
        gout = gate * grstd[..., None] * P["norm.weight"]
    return x + F.linear(gout, P["out_proj.weight"])


def m2_ssd_torch(torch, Xh, dt_raw, A, Bg, Cg, dt_bias, D, dt_limit):
    """`m2_forward` S9-S20 with the contractions spelled as einsums
    (DEVIATION 2106; `ssd_minimal_discrete`'s spelling, the corpus's
    second verbatim reference). Discretize FIRST (S10, DEVIATION 789), pad
    to Q, per-chunk cumsum, L = exp(segsum), G = C.B, Y_diag, decayed chunk
    states, the decay-matrix inter-chunk pass (the reference's spelling;
    the profile's serial pass is DEVIATION 785's and agrees at roundoff),
    Y_off with the contraction over n FIRST and the decay AFTER (HF :330-332),
    then the D residual from the UNDISCRETIZED x, added last (S20)."""
    F = torch.nn.functional
    Bsz, L, H, PP = Xh.shape
    G = Bg.shape[2]
    Q = M2_CHUNK
    dt = torch.clamp(F.softplus(dt_raw + dt_bias), min=dt_limit[0], max=dt_limit[1])
    Xd = Xh * dt[..., None]
    dA = A * dt
    Bh = Bg.repeat_interleave(H // G, dim=2)
    Ch = Cg.repeat_interleave(H // G, dim=2)
    pad = (Q - L % Q) % Q
    C_ = (L + pad) // Q

    def chunks(t):
        t = _pad_tokens(torch, t, pad)
        return t.reshape(Bsz, C_, Q, *t.shape[2:])

    Xc, Bc, Cc = chunks(Xd), chunks(Bh), chunks(Ch)
    dAp = chunks(dA).permute(0, 3, 1, 2)                       # [B, H, C, Q]
    A_cumsum = torch.cumsum(dAp, dim=-1)
    Lmat = torch.exp(_segment_sum(torch, dAp))                 # [B, H, C, Q, Q]
    Gm = torch.einsum("bclhn,bcshn->bclsh", Cc, Bc)
    Mm = Gm * Lmat.permute(0, 2, 3, 4, 1)
    Y_diag = torch.einsum("bclsh,bcshp->bclhp", Mm, Xc)
    decay_states = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
    states = torch.einsum("bcshn,bhcs,bcshp->bchpn", Bc, decay_states, Xc)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    decay_chunk = torch.exp(_segment_sum(torch, F.pad(A_cumsum[:, :, :, -1], (1, 0))))
    new_states = torch.einsum("bhzc,bchpn->bzhpn", decay_chunk, states)
    states_in = new_states[:, :-1]
    Y_off = (torch.einsum("bclhn,bchpn->bclhp", Cc, states_in)
             * torch.exp(A_cumsum).permute(0, 2, 3, 1)[..., None])
    Y = (Y_diag + Y_off).reshape(Bsz, C_ * Q, H, PP)[:, :L]
    return Y + D[..., None] * Xh


def m2_reference(torch, P, x, dt_limit=(0.0, float("inf"))):
    z, Xh, Bg, Cg, dt_raw, A = m2_prefix(torch, P, x)
    y = m2_ssd_torch(torch, Xh, dt_raw, A, Bg, Cg, P["dt_bias"], P["D"], dt_limit)
    return m2_suffix(torch, P, x, y, z)


def m2_fused(torch, P, x, fused, gated_fn, conv_fn, dt_limit=(0.0, float("inf"))):
    """The incumbent: `mamba_chunk_scan_combined` called exactly as
    `Mamba2.forward` calls it at rmsnorm=True (mamba2.py:249-265): raw dt
    with `dt_bias` and `dt_softplus=True` folded inside the kernel, `z=None`
    because the gate is applied in the gated norm after it."""
    z, Xh, Bg, Cg, dt_raw, A = m2_prefix(torch, P, x, conv_fn)
    y = fused(Xh, dt_raw, A, Bg, Cg, chunk_size=M2_CHUNK, D=P["D"], z=None,
              dt_bias=P["dt_bias"], dt_softplus=True, dt_limit=dt_limit)
    return m2_suffix(torch, P, x, y, z, gated_fn)


def m3_reference(torch, dev, P, x):
    """`m3_forward`, stage for stage, on the device: S1-S3 block norm, S4
    in_proj and the 8-way split, S5 heavy-tail A with the A_floor clamp,
    S6 dt (NO clamp), S7 ADT, S8-S9 trapezoid, S21 B/C RMSNorms, S10 angle
    recurrence (cumsum-then-mod here, DEVIATION 2107), S12 biases, S14
    pre-rotation qk.gamma, S13 interleaved-pair rotation on the first R
    pairs, S15 K scaling, the chunked core at Q = 64 with the strict-causal
    mask (S16), the SERIAL inter-chunk pass (S17/S20), S18 diagonal + D in
    one add, S19 raw z gate, out_proj, residual."""
    F = torch.nn.functional
    Bsz, L, dm = x.shape
    di = M3_EXPAND * dm
    H = di // M3_HEADDIM
    G, N, Q, PP, R = M3_NGROUPS, M3_D_STATE, M3_CHUNK, M3_HEADDIM, M3_ROPE
    TWO_PI = 2 * math.pi
    residual = x
    sumsq = x.pow(2).sum(-1)
    rstd = 1 / torch.sqrt(sumsq / dm + M3_EPS)
    hnorm = P["block_norm.weight"] * (x * rstd[..., None])
    proj = F.linear(hnorm, P["in_proj.weight"])
    z, xs, Bs, Cs, dd_dt, dd_A, trap_raw, angle_raw = torch.split(
        proj, [di, di, G * N, G * N, H, H, H, R], dim=-1)
    A = -(dd_A.clamp_min(0) + torch.reciprocal(1 - dd_A.clamp_max(0)))
    A = torch.clamp(A, max=-M3_A_FLOOR)
    dt = F.softplus(dd_dt + P["dt_bias"])
    adt = A * dt
    sig = torch.sigmoid(trap_raw)
    dt_sh = F.pad(dt[:, 1:, :], (0, 0, 0, 1))
    sig_sh = F.pad(sig[:, 1:, :], (0, 0, 0, 1))
    gamma = dt * sig
    scale = gamma + dt_sh * (1 - sig_sh)
    Bn = Bs.reshape(Bsz, L, G, N)
    Cn = Cs.reshape(Bsz, L, G, N)
    Bn = (Bn * (1 / torch.sqrt(Bn.pow(2).mean(-1, keepdim=True) + M3_EPS))) * P["B_norm.weight"]
    Cn = (Cn * (1 / torch.sqrt(Cn.pow(2).mean(-1, keepdim=True) + M3_EPS))) * P["C_norm.weight"]
    a = torch.tanh(angle_raw) * math.pi                       # [B, L, R]
    inc = a.unsqueeze(2) * dt.unsqueeze(-1)                   # [B, L, H, R]
    theta = torch.cumsum(inc, dim=1)                          # DEVIATION 2107
    theta = theta - TWO_PI * torch.floor(theta / TWO_PI)
    q = Cn.repeat_interleave(H // G, dim=2) + P["C_bias"]
    k = Bn.repeat_interleave(H // G, dim=2) + P["B_bias"]
    qkg = (q * k).sum(-1) * gamma                             # [B, L, H]
    cpad = F.pad(torch.cos(theta), (0, N // 2 - R), value=1.0)
    spad = F.pad(torch.sin(theta), (0, N // 2 - R), value=0.0)

    def _rot(t):
        tr = t.reshape(*t.shape[:-1], -1, 2)
        t0, t1 = tr[..., 0], tr[..., 1]
        return torch.stack([t0 * cpad - t1 * spad, t0 * spad + t1 * cpad], dim=-1).reshape(t.shape)

    q_rot = _rot(q)
    k_scaled = _rot(k) * scale.unsqueeze(-1)
    v = xs.reshape(Bsz, L, H, PP)
    pad = (Q - L % Q) % Q
    C_ = (L + pad) // Q
    qc = _pad_tokens(torch, q_rot, pad).reshape(Bsz, C_, Q, H, N)
    kc = _pad_tokens(torch, k_scaled, pad).reshape(Bsz, C_, Q, H, N)
    vc = _pad_tokens(torch, v, pad).reshape(Bsz, C_, Q, H, PP)
    adtq = _pad_tokens(torch, adt, pad).reshape(Bsz, C_, Q, H).permute(0, 3, 1, 2)  # [B, H, C, Q]
    dacs = torch.cumsum(adtq, dim=-1)
    mask_strict = torch.tril(torch.ones(Q, Q, dtype=torch.bool, device=dev), diagonal=-1)
    xrep = adtq.unsqueeze(-1).expand(Bsz, H, C_, Q, Q)
    seg = torch.cumsum(xrep.masked_fill(~mask_strict, 0), dim=-2)
    Lmat = torch.where(mask_strict, torch.exp(seg), seg.new_zeros(()))
    s_qk = torch.einsum("bcthn,bcshn->bhcts", qc, kc)
    Yintra = torch.einsum("bhcts,bcshp->bcthp", s_qk * Lmat, vc)
    h = torch.zeros(Bsz, H, PP, N, dtype=x.dtype, device=dev)
    ys_chunks = []
    for c in range(C_):
        exp_dacs = torch.exp(dacs[:, :, c])                   # [B, H, Q]
        ys = torch.einsum("bthn,bhpn->bthp", qc[:, c], h) * exp_dacs.permute(0, 2, 1)[..., None]
        ys_chunks.append(ys)
        d_last = dacs[:, :, c, -1]
        d_rev = d_last.unsqueeze(-1) - dacs[:, :, c]
        vdec = vc[:, c] * torch.exp(d_rev).permute(0, 2, 1)[..., None]
        incr = torch.einsum("bthp,bthn->bhpn", vdec, kc[:, c])
        h = torch.exp(d_last)[..., None, None] * h + incr
    Ystate = torch.stack(ys_chunks, dim=1).reshape(Bsz, C_ * Q, H, PP)[:, :L]
    Y = Yintra.reshape(Bsz, C_ * Q, H, PP)[:, :L] + Ystate
    skip = Y + (P["D"] + qkg).unsqueeze(-1) * v
    gate = skip * F.silu(z.reshape(Bsz, L, H, PP))
    o = F.linear(gate.reshape(Bsz, L, di), P["out_proj.weight"])
    return residual + o


def run_m23_row(torch, dev, lane, row, args, consts):
    tag = row["name"]
    samples = consts["witness_samples"]
    bad = check_bounds(lane, row)
    if bad:
        refuse(lane, "torch", "shape %s refused, not shrunk: %s" % (tag, bad))
        return
    P, x = m23_params(torch, dev, lane, row, row["seed"], samples, lane, tag)
    dumped = _dump_path(args, lane, tag)

    fused = gated_fn = conv_fn = None
    if lane == "mamba2":
        try:
            from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as fused
        except Exception as e:
            if args.require_fused:
                AGREEMENT_FAILURES.append('Required mamba-ssm SSD comparator unavailable')
            refuse(lane, "mamba-ssm",
                   "mamba_ssm is not importable (%s); the fused SSD incumbent did not "
                   "run on this box and the only opponent is the plain torch reference"
                   % str(e)[:120])
        if fused is not None:
            try:
                from mamba_ssm.ops.triton.layernorm_gated import rmsnorm_fn as gated_fn
            except Exception:
                gated_fn = None
            try:
                from causal_conv1d import causal_conv1d_fn as conv_fn
            except Exception:
                conv_fn = None
    else:
        note(lane, "torch", "NO INCUMBENT KERNEL LIBRARY for Mamba-3 at FP32: the pinned "
                            "mamba_ssm ships Mamba-3 Triton kernels but its surface "
                            "(mamba3_siso_combined.py:390-399) force-casts Q/K/V/Trap/Angles/Z "
                            "to bfloat16, a different precision; the plain torch reference "
                            "is the only opponent (DEVIATION 2108)")

    ref_out = None
    try:
        set_tf32(torch, False)
        if lane == "mamba2":
            call = lambda: m2_reference(torch, P, x)
        else:
            call = lambda: m3_reference(torch, dev, P, x)
        ref_out = time_arm(torch, lane, "torch", tag, call, args.rounds, args.warmups, lambda r: r)
    except Exception as e:
        refuse_exc(lane, "torch", tag, e)

    if fused is not None:
        arm = "mamba-ssm"
        try:
            set_tf32(torch, False)
            note(lane, arm, "composition: torch block norm + in_proj, conv=%s, "
                            "mamba_chunk_scan_combined(chunk_size=256, dt_softplus=True, "
                            "z=None), gated norm=%s, torch out_proj + residual. Triton "
                            "tl.dot on fp32 operands defaults to TF32 input precision "
                            "unless the kernel opts out, which was NOT verified for the "
                            "pinned SSD kernels: this arm's precision is the incumbent's "
                            "own (DEVIATION 2108)"
                 % ("causal_conv1d_fn" if conv_fn is not None else "torch F.conv1d",
                    "layernorm_gated.rmsnorm_fn" if gated_fn is not None else "torch"))
            call = lambda: m2_fused(torch, P, x, fused, gated_fn, conv_fn)
            fused_out = time_arm(torch, lane, arm, tag, call, args.rounds, args.warmups, lambda r: r)
            agree(lane, tag, dumped,
                  fused_out.detach().to(torch.float32).contiguous().cpu().numpy(), arm)
        except Exception as e:
            if args.require_fused:
                AGREEMENT_FAILURES.append('Required mamba-ssm SSD comparator failed at ' + tag)
            refuse_exc(lane, arm, tag, e)

    if ref_out is not None:
        agree(lane, tag, dumped, ref_out.detach().to(torch.float32).contiguous().cpu().numpy())
    else:
        AGREEMENT_FAILURES.append('%s/%s reference output unavailable' % (lane, tag))
        refuse(lane, "agree", "the reference arm did not produce an output at shape %s" % tag)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
LANES = ("transformer", "attention", "mlp", "rmsnorm", "mamba", "selective_scan",
         "mamba2", "mamba3")

#: The lanes that ALSO run the Python-side narrow/wide rows (DEVIATION 2104),
#: and which `py_rows` kind each takes. The sub-lanes have no ours arm for
#: those rows and do not run them.
LARGE_ROW_LANES = {"transformer": "llama", "mamba": "mamba1"}


def main():
    global AGREE_RTOL, AGREE_ATOL
    AGREEMENT_FAILURES.clear()
    INPUT_WITNESSES.clear()
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane", required=True, choices=LANES)
    ap.add_argument("--arm", choices=("torch", "torch-deterministic"), default="torch",
                    help="torch (default; unchanged) or torch-deterministic: "
                         "CUBLAS_WORKSPACE_CONFIG=%s before torch is imported, "
                         "torch.use_deterministic_algorithms(True), cudnn.deterministic, "
                         "cudnn.benchmark=False, every arm name suffixed -deterministic "
                         "(DEVIATION 2101)" % CUBLAS_WORKSPACE)
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
    ap.add_argument('--agree-rtol', type=float, default=5e-4)
    ap.add_argument('--agree-atol', type=float, default=1e-5)
    ap.add_argument('--require-fused', action='store_true',
                    help='Fail unless the Mamba CUDA comparator executes successfully')
    ap.add_argument('--mojo-log', action='append', default=[],
                    help='Mojo output whose input witnesses must match; repeat for both modes')
    args = ap.parse_args()
    AGREE_RTOL, AGREE_ATOL = args.agree_rtol, args.agree_atol
    if args.rounds < 1 or args.warmups < 1 or min(AGREE_RTOL, AGREE_ATOL) < 0:
        raise SystemExit('Positive rounds/warmups and nonnegative tolerances required')
    if not args.mojo_log:
        AGREEMENT_FAILURES.append('No --mojo-log supplied for input witness admission')
    if args.arm != ARM:
        raise SystemExit("speed_torch_seq: --arm read as %r before the imports and %r by "
                         "argparse; the environment was prepared for the former "
                         "(DEVIATION 2101)" % (ARM, args.arm))

    try:
        import torch
    except ImportError:
        raise SystemExit("speed_torch_seq: REFUSED. torch is not importable.")

    _load_corpus()            # DEVIATION 2102: after torch, after the env
    _check_generator_copy()   # DEVIATION 2105: the numpy copy vs the corpus
    _check_transcriptions()   #   and the transcribed ids/ranges/shapes/constants
    undeterminize(torch)   # DEVIATION 1856 / 2101; see the function's docstring
    if DETERMINISTIC:
        deterministic_readback(torch)
    name, build, is_hip, devstr = require_accelerator(torch)
    global _DEVSTR
    _DEVSTR = devstr
    rows, consts = load_shapes()
    for r in rows:
        r["seed"] = seq_seed(r["i"], consts["seed_base"])
        r["kind"] = "mamba1" if r["family"] == FAM_MAMBA else "llama"
    lane = args.lane
    if lane in ("mamba2", "mamba3"):
        fam = FAM_M23
        rows = py_rows(lane)                      # DEVIATION 2104: no driver rows
    else:
        fam = FAM_MAMBA if lane in ("mamba", "selective_scan") else FAM_LLAMA
        if lane in LARGE_ROW_LANES:
            rows = rows + py_rows(LARGE_ROW_LANES[lane])   # DEVIATION 2104
    dev = torch.device(devstr)

    print("FSPEED-HEADER family=seq lane=%s arm=%s mode=FAST device=%s "
          "rounds=%d size=%s" % (lane, _armname("torch"), name.replace(" ", "_"),
                                  args.rounds, args.size))
    if DETERMINISTIC:
        got = deterministic_readback(torch)
        note(lane, "torch", "build=%s torch=%s tf32_switches=%s deterministic=on "
                            "use_deterministic_algorithms=%s cudnn.deterministic=%s "
                            "cudnn.benchmark=%s CUBLAS_WORKSPACE_CONFIG=%s (DEVIATION 2101)"
             % (build, torch.__version__, ",".join(set_tf32(torch, False)) or "none",
                got["use_deterministic_algorithms"], got["cudnn_deterministic"],
                got["cudnn_benchmark"], got["cublas_workspace_config"]))
    else:
        note(lane, "torch", "build=%s torch=%s tf32_switches=%s deterministic=off"
             % (build, torch.__version__, ",".join(set_tf32(torch, False)) or "none"))
    if is_hip:
        note(lane, "torch", "this is a ROCm build; the arm names still say cuda "
                            "because torch's device does. TF32 on CDNA3 is XF32 "
                            "and the same switch reaches it.")

    if args.crosscheck and fam == FAM_LLAMA:
        _crosscheck_llama(torch, dev, rows, consts, lane)
        undeterminize(torch)   # the cross-check imported a corpus; re-arm this arm's setting
    elif args.crosscheck and fam == FAM_M23:
        note(lane, "torch", "crosscheck SKIPPED by name: the llama cross-check has no "
                            "meaning on this lane; the Mamba-%s reference is written from "
                            "the corpus's own staged forward and FSPEED-AGREE against our "
                            "arm's dump is its check (DEVIATION 2114)" % lane[-1])

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
            elif fam == FAM_MAMBA:
                run_mamba_row(torch, dev, lane, row, args, consts)
            else:
                run_m23_row(torch, dev, lane, row, args, consts)
        except Exception as e:
            AGREEMENT_FAILURES.append('%s/%s row raised %s' % (lane, row['name'], type(e).__name__))
            refuse(lane, "torch", "row %d %s: %s: %s"
                   % (row["i"], row["name"], type(e).__name__, str(e)[:180]))
    print("FSPEED-DONE lane=%s arm=%s" % (lane, _armname("torch")))
    for log_path in args.mojo_log:
        expected = {}
        with open(log_path) as handle:
            for line in handle:
                if line.startswith('FSPEED-WEIGHTS '):
                    fields = dict(item.split('=', 1) for item in line.split()[1:] if '=' in item)
                    key = (fields['lane'], fields['shape'], fields['tensor'])
                    expected[key] = (fields['n'], fields['hash'])
        if not expected or any(INPUT_WITNESSES.get(key) != value for key, value in expected.items()):
            AGREEMENT_FAILURES.append('Input witness mismatch or empty Mojo log: ' + log_path)
        else:
            print('FSPEED-INPUT-GATE passed=true tensors=%d path=%s' % (len(expected), log_path))
    if AGREEMENT_FAILURES:
        for failure in AGREEMENT_FAILURES:
            print('FSPEED-ADMISSION-FAILED ' + failure)
        return 2
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
