# SPDX-License-Identifier: Apache-2.0
"""`python -m mojolearn verify-model` and the report a save writes beside a
model (python/mojolearn/_verify_model.py, autograd/model_verify.mojo).

(a) The Mojo numbers: SHA-256 equals hashlib's, the digest is FNV-1a64, the
seeded inputs are exact multiples of 2^-23 in [-1, 1) and the same on every
call.
(b) Program.save, Sequential.save_checkpoint and Sequential.export write
<file>.verify.json; verify-model on the same install reproduces it (MATCH,
exit 0), a report round-trips through JSON, and a model with one weight
moved by one ulp DIFFERs against the original's report (exit 1) with the
same description hash and another file hash.
(c) A model saved on the CPU host column verifies on the GPU column and the
reverse: the cross-vendor check a user runs. The NVIDIA and AMD runs are the
same test on those boxes.

Prints MODEL_VERIFY_DIGEST lines to diff across CPU, Metal, NVIDIA and AMD.
Small models (scikit-learn-test sizes)."""
import hashlib
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend, _verify_model
from mojolearn.__main__ import main as cli


def _b():
    return _backend.binding("_mojolearn_x_sequence")


def _has_entries():
    try:
        return hasattr(_b(), "verify_digest")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_entries(),
                                reason="the sequence binding has no verify entries (rebuild it)")

_W1 = np.random.default_rng(1).standard_normal((4, 8), dtype=np.float32)
_W2 = np.random.default_rng(2).standard_normal((8, 3), dtype=np.float32)


def _program(path, w1=_W1):
    from mojolearn import ops
    s = ops.Session()
    x = s.tensor(np.zeros((6, 4), np.float32))
    h = s.linear(x, s.tensor(w1), activation="relu")
    y = s.softmax(s.linear(h, s.tensor(_W2)))
    prog = s.capture(inputs=[x], outputs=[y, h])
    s.reset()
    prog.save(path)
    return s


def _sequential():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((60, 5)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0.5)
    return (ml.Sequential().linear(8).relu().linear(3)
            .compile(loss="cross_entropy", optimizer="adam", lr=0.02)
            .fit(X, y, epochs=2, batch_size=20, seed=5))


def _read(path):
    with open(path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------- (a)

def test_sha256_is_sha256(tmp_path):
    rng = np.random.default_rng(3)
    for n in (0, 3, 55, 56, 63, 64, 65, 1000, 4099):
        data = b"abc" if n == 3 else rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        p = tmp_path / f"f{n}"
        p.write_bytes(data)
        sha, size = _b().verify_file_sha256(str(p))
        assert (str(sha), int(size)) == (hashlib.sha256(data).hexdigest(), n), n


def _fnv(data):
    h = 0xCBF29CE484222325
    for byte in data:
        h = ((h ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def test_digest_is_fnv1a64():
    for a in (np.zeros(0, np.float32), np.arange(5, dtype=np.float32), np.array([1, 2, 3], np.int32)):
        assert str(_b().verify_digest(a.ctypes.data, a.nbytes)) == _fnv(a.tobytes())


def test_seeded_inputs_are_exact_and_repeatable():
    a = np.empty((7, 9), np.float32)
    b = np.empty((7, 9), np.float32)
    c = np.empty((7, 9), np.float32)
    _b().verify_fill(a.ctypes.data, a.size, 0, 0)
    _b().verify_fill(b.ctypes.data, b.size, 0, 0)
    _b().verify_fill(c.ctypes.data, c.size, 0, 1)
    assert a.tobytes() == b.tobytes()
    assert a.tobytes() != c.tobytes()
    assert a.min() >= -1.0 and a.max() < 1.0
    k = (a.astype(np.float64) + 1.0) * 2.0 ** 23
    assert np.all(k == np.round(k))
    print(f"MODEL_VERIFY_DIGEST seeded {_fnv(a.tobytes())}")


# ---------------------------------------------------------------- (b)

def test_program_report_round_trips(tmp_path, capsys):
    path = tmp_path / "mlp.mlprog"
    s = _program(path)
    side = _verify_model.sidecar_path(path)
    assert os.path.isfile(side)
    saved = _read(side)
    assert saved["model"]["kind"] == "program"
    assert [len(saved["inputs"]), len(saved["outputs"])] == [1, 2]
    # the same install reproduces the saved report
    rep = _verify_model.verify(path, expect=saved)
    verdict, rows, notes = _verify_model.compare(rep, saved)
    assert verdict == "MATCH" and not notes, (rows, notes)
    assert [r["verdict"] for r in rows] == ["MATCH", "MATCH"]
    # a written report reads back to the same comparison
    out = tmp_path / "again.json"
    assert cli(["verify-model", str(path), "--out", str(out)]) == 0
    assert "verdict  MATCH" in capsys.readouterr().out
    assert _verify_model.compare(_read(out), saved)[0] == "MATCH"
    assert cli(["verify-model", str(path), "--expect", str(out), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["comparison"]["verdict"] == "MATCH"
    s.close()
    for y in saved["outputs"]:
        print(f"MODEL_VERIFY_DIGEST program_{y['name']} {y['digest']}")


def test_program_differs_on_a_perturbed_weight(tmp_path, capsys):
    a, b = tmp_path / "a.mlprog", tmp_path / "b.mlprog"
    sa = _program(a)
    w = _W1.copy()
    w[1, 2] = np.nextafter(w[1, 2], np.float32(np.inf))
    sb = _program(b, w)
    assert cli(["verify-model", str(b), "--expect", _verify_model.sidecar_path(a)]) == 1
    text = capsys.readouterr().out
    assert "DIFFER" in text and "verdict  DIFFER" in text
    ra, rb = _read(_verify_model.sidecar_path(a)), _read(_verify_model.sidecar_path(b))
    assert ra["model"]["description_hash"] == rb["model"]["description_hash"]
    assert ra["model"]["sha256"] != rb["model"]["sha256"]
    assert ra["inputs"] == rb["inputs"]
    assert ra["outputs"][0]["digest"] != rb["outputs"][0]["digest"]
    sa.close()
    sb.close()


def test_sequential_checkpoint_and_export(tmp_path, capsys):
    m = _sequential()
    ck, ex = tmp_path / "m.ckpt", tmp_path / "m.export"
    m.save_checkpoint(ck)
    m.export(ex)
    rc, re_ = _read(_verify_model.sidecar_path(ck)), _read(_verify_model.sidecar_path(ex))
    assert (rc["model"]["kind"], re_["model"]["kind"]) == ("sequential", "sequential-export")
    # one model: the same description and the same outputs, two files
    assert rc["model"]["description_hash"] == re_["model"]["description_hash"]
    assert rc["outputs"] == re_["outputs"]
    assert rc["inputs"][0]["shape"] == [_verify_model.SEQUENTIAL_ROWS, 5]
    assert cli(["verify-model", str(ck)]) == 0
    assert cli(["verify-model", str(ex)]) == 0
    capsys.readouterr()
    # one parameter moved by one ulp: another file, other outputs
    m.params_[0] = np.nextafter(m.params_[0], np.float32(np.inf))
    moved = tmp_path / "moved.export"
    m.export(moved)
    assert cli(["verify-model", str(moved), "--expect", _verify_model.sidecar_path(ex)]) == 1
    assert "verdict  DIFFER" in capsys.readouterr().out
    print(f"MODEL_VERIFY_DIGEST sequential {rc['outputs'][0]['digest']}")


def test_inputs_from_a_file(tmp_path, capsys):
    path = tmp_path / "mlp.mlprog"
    s = _program(path)
    npz = tmp_path / "x.npz"
    np.savez(npz, x=np.random.default_rng(9).standard_normal((6, 4), dtype=np.float32))
    out = tmp_path / "file.json"
    assert cli(["verify-model", str(path), "--inputs", str(npz), "--no-sidecar", "--out", str(out)]) == 0
    rep = _read(out)
    assert rep["input_source"] == "file" and rep["inputs"][0]["name"] == "x"
    assert cli(["verify-model", str(path), "--inputs", str(npz), "--expect", str(out)]) == 0
    # a report made on file inputs needs the same file
    assert cli(["verify-model", str(path), "--expect", str(out)]) == 2
    capsys.readouterr()
    s.close()


def test_refuses_a_file_that_is_not_a_model(tmp_path, capsys):
    p = tmp_path / "junk.bin"
    p.write_bytes(b"not a model")
    assert cli(["verify-model", str(p)]) in (2, 4)
    assert cli(["verify-model", str(tmp_path / "missing")]) == 2
    capsys.readouterr()


# ---------------------------------------------------------------- (c)

_CHILD = textwrap.dedent('''
    import sys
    import numpy as np
    import mojolearn as ml
    from mojolearn import ops
    from mojolearn.__main__ import main as cli
    mode, path = sys.argv[1], sys.argv[2]
    if mode == "save":
        W1 = np.random.default_rng(1).standard_normal((4, 8), dtype=np.float32)
        W2 = np.random.default_rng(2).standard_normal((8, 3), dtype=np.float32)
        with ops.Session() as s:
            x = s.tensor(np.zeros((6, 4), np.float32))
            y = s.softmax(s.linear(s.linear(x, s.tensor(W1), activation="tanh"), s.tensor(W2)))
            prog = s.capture(inputs=[x], outputs=[y])
            s.reset()
            prog.save(path + ".mlprog")
        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 5)).astype(np.float32)
        yr = (X[:, 0] * 2 - X[:, 1]).astype(np.float32)
        m = (ml.Sequential().linear(8).tanh().linear(1).compile(loss="mse", optimizer="adam", lr=0.02)
             .fit(X, yr, epochs=2, batch_size=20, seed=5))
        m.export(path + ".export")
    else:
        for ext in (".mlprog", ".export"):
            print("RC", ext, cli(["verify-model", path + ext]))
    print("COLUMN", ml.vendor())
''')


def _run(column, mode, path):
    env = dict(os.environ)
    if column == "cpu":
        env["MOJOLEARN_VENDOR"] = "cpu"
    else:
        env.pop("MOJOLEARN_VENDOR", None)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env["PYTHONPATH"] = here + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", _CHILD, mode, str(path)], env=env, capture_output=True,
                       text=True, timeout=900)
    if r.returncode != 0 and column == "cpu" and "No module named" in r.stderr:
        pytest.skip("no CPU host binding in this install")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    return r.stdout


def _gpu_present():
    try:
        return ml.vendor() in ("metal", "cuda", "hip")
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_present(), reason="no GPU binding: nothing to compare the host column with")
@pytest.mark.parametrize("writer, reader", [("cpu", "gpu"), ("gpu", "cpu")])
def test_verify_on_the_other_column(writer, reader, tmp_path):
    path = tmp_path / f"{writer}_to_{reader}"
    _run(writer, "save", path)
    out = _run(reader, "verify", path)
    assert "RC .mlprog 0" in out and "RC .export 0" in out, out[-3000:]
    assert out.count("verdict  MATCH") == 2, out[-3000:]
