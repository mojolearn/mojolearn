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
    (cross_vendor.fold_pair, lambda d: cross_vendor.fold_pair(np.ones(2, np.float32), np.ones(2, np.float32), device=d)),
    (cross_vendor.ordered_fold, lambda d: cross_vendor.ordered_fold([np.ones(2, np.float32)], device=d)),
]


ALL = CALLS + [
    (linalg_fast.matmul_fused, None),
    (training.mlp_inference_sessions, None),
    (training.embedding_backward, None),
    (training.rms_norm_backward, None),
    (_xd.lu_solve, None),
]


@pytest.mark.parametrize("function", [f for f, _ in ALL], ids=lambda f: f.__name__)
def test_signature_has_keyword_only_device(function):
    parameter = inspect.signature(function).parameters["device"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.parametrize("bad", ["tpu", 0, "GPU"])
@pytest.mark.parametrize("call", [c for _, c in CALLS], ids=[f.__name__ for f, _ in CALLS])
def test_bad_word_is_refused(call, bad):
    with pytest.raises(ValueError, match=BAD_WORDS):
        call(bad)


@pytest.mark.parametrize("function", [cross_vendor.state_hash, cross_vendor.trainer_state_hash],
                         ids=lambda f: f.__name__)
def test_digests_take_no_device(function):
    assert "device" not in inspect.signature(function).parameters


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


def _device_array_or_skip(a):
    # TrainDeviceArray exists on an identical GPU installation only.
    if mojolearn.vendor() == "cpu":
        pytest.skip("TrainDeviceArray needs a GPU installation")
    return training.train_to_device(a)


# lane vdf-linalg3: TrainDeviceArray inputs (also inside list arguments) with
# device='cpu' are refused in the router's words before any work.
DEVICE_ARRAY_CPU_CALLS = [
    ("clip_grad_norm_", lambda g: training.clip_grad_norm_([g], 1.0, device="cpu")),
    ("accumulate_grads", lambda g: training.accumulate_grads([[g], [g]], None, device="cpu")),
    ("chunked_lm_head_loss", lambda g: training.chunked_lm_head_loss(g, g, np.zeros(2, np.int32), device="cpu")),
    ("mlp_inference_sessions", lambda g: training.mlp_inference_sessions([g], g, g, g, g, device="cpu")),
]


@pytest.mark.parametrize("name,call", DEVICE_ARRAY_CPU_CALLS, ids=[n for n, _ in DEVICE_ARRAY_CPU_CALLS])
def test_device_array_with_device_cpu_is_refused(name, call):
    g = _device_array_or_skip(_a())
    with pytest.raises(ValueError, match="is a TrainDeviceArray, which lives on the GPU"):
        call(g)


def test_device_array_mixed_with_host_is_refused():
    g = _device_array_or_skip(_a())
    with pytest.raises(TypeError, match="device and host arrays are mixed"):
        training.clip_grad_norm_([g, _a()], 1.0)


# lane vdf-linalg4: linalg functions read TrainDeviceArray operands in place;
# with device='cpu' they are refused (they live on the GPU), mixed with host
# arrays they are refused, and `out` is not taken (the result is a new device
# array).
LINALG_DEVICE_ARRAY_CPU_CALLS = [
    ("matmul", lambda g: linalg.matmul(g, g, transpose_b=True, device="cpu")),
    ("to_bf16", lambda g: linalg.to_bf16(g, device="cpu")),
    ("quantize_int8", lambda g: linalg.quantize_int8(g, device="cpu")),
    ("matmul_int8", lambda g: linalg.matmul_int8(g, g, device="cpu")),
]


@pytest.mark.parametrize("name,call", LINALG_DEVICE_ARRAY_CPU_CALLS, ids=[n for n, _ in LINALG_DEVICE_ARRAY_CPU_CALLS])
def test_linalg_device_array_with_device_cpu_is_refused(name, call):
    g = _device_array_or_skip(_a())
    with pytest.raises(ValueError, match="is a TrainDeviceArray, which lives on the GPU"):
        call(g)


def test_linalg_device_array_mixed_with_host_is_refused():
    g = _device_array_or_skip(_a())
    with pytest.raises(TypeError, match="device and host arrays are mixed"):
        linalg.matmul(g, _a(), transpose_b=True)


def test_linalg_device_array_out_is_refused():
    g = _device_array_or_skip(_a())
    with pytest.raises(TypeError, match="out is not taken with device-array operands"):
        linalg.matmul(g, g, transpose_b=True, out=np.zeros((_a().shape[0], _a().shape[0]), np.float32))


def test_linalg_device_array_typed_result_is_refused_by_float_ops():
    g = _device_array_or_skip(_a())
    bits = linalg.to_bf16(g)
    with pytest.raises(TypeError, match="Refused rather than cast"):
        linalg.matmul(bits, g, transpose_b=True)


# lane vdf-linalg5: the int15 trio, from_f16, svdvals / svd(compute_uv=False) / tall qr('r') and lowbit read
# TrainDeviceArray operands in place; with device='cpu' they are refused in the same words.
LINALG5_DEVICE_ARRAY_CPU_CALLS = [
    ("quantize_int15", lambda g: linalg.quantize_int15(g, device="cpu")),
    ("matmul_int15", lambda g: linalg.matmul_int15(g, g, device="cpu")),
    ("svdvals", lambda g: linalg.svdvals(g, device="cpu")),
    ("svd_values", lambda g: linalg.svd(g, compute_uv=False, device="cpu")),
    ("qr_r", lambda g: linalg.qr(g, mode="r", device="cpu")),
    ("lowbit_pack_one", lambda g: lowbit.pack_one(g, "int8", device="cpu")),
]


@pytest.mark.parametrize("name,call", LINALG5_DEVICE_ARRAY_CPU_CALLS, ids=[n for n, _ in LINALG5_DEVICE_ARRAY_CPU_CALLS])
def test_linalg5_device_array_with_device_cpu_is_refused(name, call):
    g = _device_array_or_skip(_a())
    with pytest.raises(ValueError, match="is a TrainDeviceArray, which lives on the GPU"):
        call(g)


def test_linalg5_device_array_refusal_names_the_function_once():
    g = _device_array_or_skip(_a())
    with pytest.raises(ValueError) as caught:
        linalg.svdvals(g, device="cpu")
    assert "mojolearn.linalg.mojolearn" not in str(caught.value)


# lane vdf-linalg6: eigh, qr 'reduced' / 'complete' / 'raw' / wide 'r' and svd with compute_uv read TrainDeviceArray
# operands in place on the x_decomp kit (the vdf-linalg5 refusal "takes host arrays" is gone); with device='cpu'
# they are refused in the same words, and every result is a new device array. Values are checked natively
# (checks/native_tests/linalg_device_kit_gpu.mojo).
LINALG6_DEVICE_ARRAY_CPU_CALLS = [
    ("eigh", lambda g: linalg.eigh(g, device="cpu")),
    ("qr_reduced", lambda g: linalg.qr(g, device="cpu")),
    ("qr_complete", lambda g: linalg.qr(g, mode="complete", device="cpu")),
    ("qr_raw", lambda g: linalg.qr(g, mode="raw", device="cpu")),
    ("svd", lambda g: linalg.svd(g, device="cpu")),
]


@pytest.mark.parametrize("name,call", LINALG6_DEVICE_ARRAY_CPU_CALLS, ids=[n for n, _ in LINALG6_DEVICE_ARRAY_CPU_CALLS])
def test_linalg6_device_array_with_device_cpu_is_refused(name, call):
    g = _device_array_or_skip(_a())
    with pytest.raises(ValueError, match="is a TrainDeviceArray, which lives on the GPU"):
        call(g)


LINALG6_DEVICE_ARRAY_FORMS = [
    ("eigh", lambda g: linalg.eigh(g), 2),
    ("qr_reduced", lambda g: linalg.qr(g), 2),
    ("qr_complete", lambda g: linalg.qr(g, mode="complete"), 2),
    ("qr_raw", lambda g: linalg.qr(g, mode="raw"), 2),
    ("svd", lambda g: linalg.svd(g), 3),
    ("svd_full", lambda g: linalg.svd(g, full_matrices=True), 3),
    ("svd_hermitian", lambda g: linalg.svd(g, hermitian=True), 3),
]


@pytest.mark.parametrize("name,call,parts", LINALG6_DEVICE_ARRAY_FORMS, ids=[n for n, _, _ in LINALG6_DEVICE_ARRAY_FORMS])
def test_linalg6_device_array_results_are_device_arrays(name, call, parts):
    g = _device_array_or_skip(_a())
    result = call(g)
    assert len(result) == parts
    for item in result:
        assert type(item) is type(g)


def test_linalg6_wide_qr_r_device_array_is_a_device_array():
    g = _device_array_or_skip(np.ones((2, 3), dtype=np.float32))
    assert type(linalg.qr(g, mode="r")) is type(g)


def test_linalg5_typed_device_array_is_refused_by_training_ops():
    g = _device_array_or_skip(_a())
    bits = linalg.to_bf16(g)
    with pytest.raises(TypeError, match="typed device array"):
        training.clip_grad_norm_([bits], 1.0)


# lane vdf-linalg7: solve, lstsq, randomized_svd and sparse_encode read TrainDeviceArray operands in place
# (bindings/linalg_decomp_dev_door.mojo); with device='cpu' they are refused in the same words, a host operand beside a
# device one is refused, and every array result is a new device array. Values are checked natively
# (checks/native_tests/linalg_decomp_device_gpu.mojo).
def _spd():
    return (np.eye(3, dtype=np.float32) * 4.0 + 0.5).astype(np.float32)


LINALG7_DEVICE_ARRAY_CPU_CALLS = [
    ("solve", lambda g: mojolearn.solve(g, g, device="cpu")),
    ("lstsq", lambda g: mojolearn.lstsq(g, g, device="cpu")),
    ("randomized_svd", lambda g: mojolearn.randomized_svd(g, 1, device="cpu")),
    ("sparse_encode", lambda g: mojolearn.sparse_encode(g, g, algorithm="threshold", device="cpu")),
]


@pytest.mark.parametrize("name,call", LINALG7_DEVICE_ARRAY_CPU_CALLS, ids=[n for n, _ in LINALG7_DEVICE_ARRAY_CPU_CALLS])
def test_linalg7_device_array_with_device_cpu_is_refused(name, call):
    g = _device_array_or_skip(_spd())
    with pytest.raises(ValueError, match="is a TrainDeviceArray, which lives on the GPU"):
        call(g)


def test_linalg7_device_array_mixed_with_host_is_refused():
    g = _device_array_or_skip(_spd())
    with pytest.raises(TypeError, match="mojolearn.solve: device and host arrays are mixed"):
        mojolearn.solve(g, _spd())
    with pytest.raises(TypeError, match="mojolearn.lstsq: device and host arrays are mixed"):
        mojolearn.lstsq(_spd(), g)


def test_linalg7_device_array_results_are_device_arrays():
    g = _device_array_or_skip(_spd())
    x = mojolearn.solve(g, g)
    assert type(x) is type(g)
    solution, residues, rank, values = mojolearn.lstsq(g, g)
    assert type(solution) is type(g) and type(values) is type(g) and isinstance(rank, int)
    u, s, vt = mojolearn.randomized_svd(g, 1, random_state=0)
    assert type(u) is type(g) and type(s) is type(g) and type(vt) is type(g)
    code = mojolearn.sparse_encode(g, g, algorithm="threshold")
    assert type(code) is type(g)


def test_linalg7_non_finite_device_array_is_refused():
    a = _spd()
    a[1, 1] = np.inf
    g = _device_array_or_skip(a)
    with pytest.raises(ValueError, match="a: input must be finite"):
        mojolearn.solve(g, _device_array_or_skip(_spd()))


# lane vdf-tail-b: lu_factor and lu_solve read TrainDeviceArray operands in place (bindings/linalg_decomp_dev_door.mojo
# `lu_factor_dev_call`, `lu_solve_dev_call`): the pair is two device arrays (float32 lu, int32 piv), lu_solve's X a
# device array; device='cpu' and a host operand beside a device one are refused in words. Values are checked natively
# (checks/native_tests/linalg_decomp_device_gpu.mojo `test_lu_factor_solve_device_operands`).
def test_lu_device_array_with_device_cpu_is_refused():
    g = _device_array_or_skip(_spd())
    with pytest.raises(ValueError, match="is a TrainDeviceArray, which lives on the GPU"):
        mojolearn.lu_factor(g, device="cpu")
    lu, piv = mojolearn.lu_factor(g)
    with pytest.raises(ValueError, match="is a TrainDeviceArray, which lives on the GPU"):
        mojolearn.lu_solve((lu, piv), g, device="cpu")


def test_lu_device_array_results_are_device_arrays():
    g = _device_array_or_skip(_spd())
    lu, piv = mojolearn.lu_factor(g)
    assert type(lu) is type(g) and type(piv) is type(g)
    for trans in (0, 1):
        x = mojolearn.lu_solve((lu, piv), g, trans=trans)
        assert type(x) is type(g)
    with pytest.raises(TypeError, match="mojolearn.lu_solve: device and host arrays are mixed"):
        mojolearn.lu_solve((lu, piv), _spd())
