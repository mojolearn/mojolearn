# SPDX-License-Identifier: Apache-2.0
"""`device=` on the linear-algebra, low-bit, cross-vendor, dense linalg and
training functions: the boundary words only (lane vdf-linalg). The digests of
both sides are verify's; nothing numerical is asserted here."""
import inspect

import numpy as np
import pytest

import mojolearn
from mojolearn import _backend, cross_vendor, linalg, linalg_fast, lowbit, training
from mojolearn import _expansion_decomp as _xd

BAD_WORDS = "device must be 'gpu', 'cpu' or None"


def _a():
    return np.ones((2, 2), dtype=np.float32)


# (public function, a call with tiny inputs that forwards `device`)
CALLS = [
    (linalg.matmul, lambda d: linalg.matmul(_a(), _a(), device=d)),
    (linalg.to_bf16, lambda d: linalg.to_bf16(_a(), device=d)),
    (linalg.from_bf16, lambda d: linalg.from_bf16(np.zeros((2, 2), np.uint16), device=d)),
    (linalg.quantize_int8, lambda d: linalg.quantize_int8(_a(), device=d)),
    (linalg.dequantize_int8, lambda d: linalg.dequantize_int8(np.zeros((2, 2), np.int8), np.zeros(2, np.int32), device=d)),
    (linalg.matmul_bf16, lambda d: linalg.matmul_bf16(_a(), np.zeros((2, 2), np.uint16), device=d)),
    (linalg.matmul_int8, lambda d: linalg.matmul_int8(_a(), _a(), device=d)),
    (linalg.quantize_int15, lambda d: linalg.quantize_int15(_a(), device=d)),
    (linalg.dequantize_int15, lambda d: linalg.dequantize_int15(np.zeros((2, 2), np.int8), np.zeros((2, 2), np.int8),
                                                                np.zeros(2, np.int32), device=d)),
    (linalg.matmul_int15, lambda d: linalg.matmul_int15(_a(), _a(), device=d)),
    (linalg.qr, lambda d: linalg.qr(_a(), device=d)),
    (linalg.eigh, lambda d: linalg.eigh(_a(), device=d)),
    (linalg.svdvals, lambda d: linalg.svdvals(_a(), device=d)),
    (linalg.svd, lambda d: linalg.svd(_a(), device=d)),
    (lowbit.widen_bf16, lambda d: lowbit.widen_bf16(np.zeros((2, 2), np.uint16), device=d)),
    (lowbit.pack_one, lambda d: lowbit.pack_one(_a(), "bfloat16", device=d)),
    (lowbit.materialize_one, lambda d: lowbit.materialize_one(_a(), device=d)),
    (lowbit.pack, lambda d: lowbit.pack({"w": _a()}, "bfloat16", device=d)),
    (lowbit.unpack, lambda d: lowbit.unpack({"w": _a()}, device=d)),
    (_xd.lu_factor, lambda d: _xd.lu_factor(_a(), device=d)),
    (_xd.solve, lambda d: _xd.solve(_a(), _a(), device=d)),
    (_xd.lstsq, lambda d: _xd.lstsq(_a(), _a(), device=d)),
    (_xd.randomized_svd, lambda d: _xd.randomized_svd(_a(), 1, device=d)),
    (_xd.sparse_encode, lambda d: _xd.sparse_encode(_a(), _a(), device=d)),
    (training.clip_grad_norm_, lambda d: training.clip_grad_norm_([_a()], 1.0, device=d)),
    (training.cross_entropy, lambda d: training.cross_entropy(_a(), np.zeros(2, np.int64), device=d)),
    (training.accumulate_grads, lambda d: training.accumulate_grads([[_a()]], [1], device=d)),
    (training.embedding_forward, lambda d: training.embedding_forward(_a(), np.zeros(2, np.int64), device=d)),
    (training.rms_norm_forward, lambda d: training.rms_norm_forward(_a(), np.ones(2, np.float32), 1e-5, device=d)),
    (training.linear_forward, lambda d: training.linear_forward(_a(), _a(), device=d)),
    (training.linear_backward, lambda d: training.linear_backward(_a(), _a(), _a(), device=d)),
    (training.chunked_lm_head_loss, lambda d: training.chunked_lm_head_loss(_a(), _a(), np.zeros(2, np.int64), device=d)),
    (training.residual_dropout, lambda d: training.residual_dropout(_a(), _a(), device=d)),
    (training.residual_dropout_backward, lambda d: training.residual_dropout_backward(_a(), device=d)),
]

HOST_ONLY = [
    (cross_vendor.fold_pair, lambda d: cross_vendor.fold_pair(np.ones(2, np.float32), np.ones(2, np.float32), device=d)),
    (cross_vendor.ordered_fold, lambda d: cross_vendor.ordered_fold([np.ones(2, np.float32)], device=d)),
    (cross_vendor.state_hash, lambda d: cross_vendor.state_hash({}, device=d)),
]

ALL = CALLS + HOST_ONLY + [
    (linalg_fast.matmul_fused, None),
    (training.mlp_inference_sessions, None),
    (training.embedding_backward, None),
    (training.rms_norm_backward, None),
    (cross_vendor.trainer_state_hash, None),
    (_xd.lu_solve, None),
]


@pytest.mark.parametrize("function", [f for f, _ in ALL], ids=lambda f: f.__name__)
def test_signature_has_keyword_only_device(function):
    parameter = inspect.signature(function).parameters["device"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("bad", ["tpu", 0, "GPU"])
@pytest.mark.parametrize("call", [c for _, c in CALLS + HOST_ONLY], ids=[f.__name__ for f, _ in CALLS + HOST_ONLY])
def test_bad_word_is_refused(call, bad):
    with pytest.raises(ValueError, match=BAD_WORDS):
        call(bad)


@pytest.mark.parametrize("call", [c for _, c in HOST_ONLY], ids=[f.__name__ for f, _ in HOST_ONLY])
def test_host_only_refuses_gpu(call):
    with pytest.raises(ValueError, match="has no GPU implementation"):
        call("gpu")


@pytest.mark.parametrize("call", [c for _, c in CALLS], ids=[f.__name__ for f, _ in CALLS])
def test_gpu_on_cpu_only_installation_is_refused(call):
    # Holds only on a CPU-only installation (vendor 'cpu'); on a GPU one the
    # same call runs today's route, which this boundary file does not judge.
    if mojolearn.vendor() != "cpu":
        pytest.skip("CPU-only installation check; this installation has a GPU set")
    with pytest.raises(ValueError, match="needs a GPU build of mojolearn"):
        call("gpu")


def test_matmul_fused_cpu_beside_gpu_is_refused():
    # Holds only on a GPU installation: the Apple fast arm has no CPU image.
    if mojolearn.vendor() == "cpu" or _backend.default_mode() != "fast":
        pytest.skip("GPU fast-mode installation check")
    with pytest.raises(ValueError, match="device='cpu' is not available for mojolearn.linalg_fast.matmul_fused"):
        linalg_fast.matmul_fused(_a(), _a(), device="cpu")
