# SPDX-License-Identifier: Apache-2.0
"""Release-gate coverage of mojolearn.models and captured-program options
changed since 0.8.55 (docs/plans/TEST_COVERAGE_20261021.md): safetensors
file and checkpoint metadata, TensorInfo admission, tokenizer pattern names,
generation stream reads, a parallel model after close and the Huber option
of Program.fit. Real calls first, then poisoned inputs refused by name.
Tiny fixtures, no measurements."""
import array

import numpy as np
import pytest

from mojolearn.models import Checkpoint, SafetensorsFile
from mojolearn.models.safetensors import TensorInfo, write_safetensors
from mojolearn.models.tokenizer import PATTERNS, pattern_name


@pytest.fixture
def checkpoint_dir(tmp_path):
    write_safetensors(tmp_path / "model.safetensors",
                      {"w": ("F32", (2, 3), array.array("f", range(6)).tobytes()),
                       "b": ("BF16", (2,), array.array("H", [0x3F80, 0x4000]).tobytes())})
    return tmp_path


def test_safetensors_info_close_and_reopen_on_read(checkpoint_dir):
    file = SafetensorsFile(checkpoint_dir / "model.safetensors")
    info = file.info("w")
    assert (info.name, info.dtype, info.shape, info.size, info.nbytes) == ("w", "F32", (2, 3), 6, 24)
    assert file.info("b").nbytes == 4 and file.info("b").size == 2
    first = np.asarray(file.read("w")).copy()
    assert file.close() is None and file._mm is None
    assert file.close() is None  # closing twice is a no-op
    np.testing.assert_array_equal(np.asarray(file.read("w")), first)  # a read maps the file again
    file.close()
    with pytest.raises(KeyError, match="safetensors holds no tensor missing"):
        file.info("missing")
    with pytest.raises(ValueError, match="bf16 must be widen or bits"):
        file.read("b", bf16="round")


def test_checkpoint_info_close_and_refusals(checkpoint_dir):
    with Checkpoint.open(checkpoint_dir) as ck:
        info = ck.info("w")
        assert (info.shape, info.nbytes, info.size) == ((2, 3), 24, 6)
        np.testing.assert_array_equal(np.asarray(ck.read("w")), np.arange(6, dtype=np.float32).reshape(2, 3))
        with pytest.raises(KeyError, match="checkpoint holds no tensor missing"):
            ck.info("missing")
    assert ck.close() is None
    with pytest.raises(FileNotFoundError, match="checkpoint path does not exist"):
        Checkpoint.open(checkpoint_dir / "absent")


def test_tensor_info_reports_size_and_nbytes():
    info = TensorInfo("w", "F32", (2, 3), 8, 32, "model.safetensors")
    assert (info.size, info.nbytes, info.shape) == (6, 24, (2, 3))
    scalar = TensorInfo("s", "I64", (), 0, 8, "model.safetensors")
    assert (scalar.size, scalar.nbytes) == (1, 8)


@pytest.mark.parametrize("fields,error,match", [
    ((5, "F32", (2, 3), 0, 24, "f"), TypeError, "name and dtype must be str"),
    (("w", "F32", (-2, 3), 0, 24, "f"), ValueError, "non-negative dimensions"),
    (("w", "F32", (2, 3), 24, 0, "f"), ValueError, r"0 <= begin <= end"),
    (("w", "F32", (2, 3), 0, 20, "f"), ValueError, "bytes differ from dtype/shape"),
])
def test_tensor_info_refuses_what_the_header_parser_refuses(fields, error, match):
    with pytest.raises(error, match=match):
        TensorInfo(*fields)


def test_pattern_name_names_every_known_spelling():
    for name, spellings in PATTERNS.items():
        assert [pattern_name(s) for s in spellings] == [name] * len(spellings)
    for unknown in (r"\w+", "", 5, b"\\s+", None):
        assert pattern_name(unknown) is None


def _stream_model(tmp_path):
    from mojolearn.tests.test_generation_native import load_fixture
    return load_fixture(tmp_path, uniform=True)


def test_generation_stream_read_keyword_info_and_refusals(tmp_path):
    model = _stream_model(tmp_path)
    stream = model.stream(np.array([[33], [34]], dtype="i4"), 4)
    assert np.asarray(stream.read(max_tokens=2)).shape == (2, 2)
    assert stream.info()["generated_tokens"] == 2
    for bad in (0, -1):
        with pytest.raises(ValueError, match="read count must be positive"):
            stream.read(max_tokens=bad)
    for bad in ("2", True, 1.5):
        with pytest.raises(TypeError, match="max_tokens must be an integer"):
            stream.read(max_tokens=bad)
    assert np.asarray(stream.read(np.int64(5))).shape == (2, 2)
    assert stream.info()["done"] is True
    with pytest.raises(ValueError, match="unsupported checkpoint schema"):
        model.resume_generation({"schema": "not-a-stream"})


def test_parallel_causal_lm_parameters_refused_after_close(tmp_path):
    from mojolearn.tests import test_parallel_causal_lm as pc
    root, _, _ = pc.checkpoint(tmp_path)
    split = pc.checked(pc.mod.ParallelCausalLM.load, root, layer_devices=pc.owners(),
                       numeric_profile="fp32_v1")
    try:
        params = pc.checked(split.parameters)
        assert params and all(np.asarray(v).size for v in params.values())
    finally:
        pc.checked(split.close)
    pc.refused(split.parameters, ValueError, "ParallelCausalLM is closed")


def test_program_fit_huber_delta_matches_numpy_and_refuses_bad_delta():
    from mojolearn import ops
    rng = np.random.default_rng(0)
    X = rng.standard_normal((16, 3)).astype(np.float32)
    y = X @ np.array([[1.0], [-2.0], [0.5]], np.float32)
    delta = np.float32(0.5)
    r = np.abs(y)  # the first epoch's residual: the weights start at zero
    expected = np.where(r <= delta, 0.5 * r * r, delta * (r - 0.5 * delta)).mean()
    with ops.Session() as s:
        x, w = s.tensor(X), s.tensor(np.zeros((3, 1), np.float32))
        program = s.capture(x, s.matmul(x, w))
        opt = s.optimizer("sgd", lr=0.05)
        fitted = program.fit(X, targets=y, optimizer=opt, loss="huber", huber_delta=float(delta),
                             epochs=2, batch_size=16, shuffle=False)
        assert fitted is program and len(program.loss_curve_) == 2
        np.testing.assert_allclose(program.loss_curve_[0], expected, rtol=1e-5)
        assert program.loss_curve_[1] < program.loss_curve_[0]
        for bad in (0.0, -1.0, float("nan")):
            with pytest.raises(Exception, match="Huber delta must be positive"):
                program.fit(X, targets=y, optimizer=opt, loss="huber", huber_delta=bad, epochs=1)
