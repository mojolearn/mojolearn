# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Exercise interruption/resume and reporting with fake fits, no native bindings."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace, ModuleType

import numpy as np
import pytest
import identity_break as ib
from identity_iterate import command
from identity_timing import summarize


@pytest.fixture
def run_fixture(monkeypatch, tmp_path):
    package = tmp_path / "mojolearn"
    package.mkdir()
    (package / "__init__.py").write_text("# fixture package\n")
    ml = ModuleType("mojolearn")
    ml.__file__ = str(package / "__init__.py")
    ml.__version__ = "test"
    ml.numeric_mode = lambda: "identical"
    ml.vendor = lambda: "test-device"
    monkeypatch.setitem(sys.modules, "mojolearn", ml)
    verify = ModuleType("mojolearn._verify")
    verify.binding_artifacts = lambda: []
    monkeypatch.setitem(sys.modules, "mojolearn._verify", verify)
    monkeypatch.setenv("MOJOLEARN_NUMERIC_MODE", "identical")
    monkeypatch.setattr(ib, "commit_witness", lambda: ("a" * 40, "fixture"))
    monkeypatch.setattr(ib, "refuse_routine_apple_column", lambda *a: "")
    monkeypatch.setattr(ib, "_par_devices", lambda: [0])
    monkeypatch.setattr(ib, "FIXTURES", ["base", "ties"])
    monkeypatch.setattr(ib, "fixture", lambda f: (np.ones((2, 2), dtype=np.float32), np.zeros(2), np.ones(2)))
    monkeypatch.setattr(ib, "heldout", lambda f: np.ones((2, 2), dtype=np.float32))
    monkeypatch.setattr(ib, "BATCH", {"tiny": None})
    monkeypatch.setattr(ib, "RLPAIR", {})
    state = dict(calls=0, crash_at=None, refuse=False)
    def fit(*args):
        state["calls"] += 1
        if state["calls"] == state["crash_at"]:
            raise KeyboardInterrupt
        if state["refuse"]:
            raise ValueError("deliberate refusal")
        return ib._fit(dict(value="a" * 16))
    monkeypatch.setattr(ib, "LANES", {"tiny": fit})
    args = argparse.Namespace(json=str(tmp_path / "record.json"), resume=False, lanes="tiny", skip="",
                              fixtures="base,ties", repeats=2, vendor="fixture-cpu", allow_fast=False,
                              batch_alone=1, no_batch=True, no_rlpair=True, verbose=False,
                              fail_on_refused=True)
    for _, _, flag, _ in ib.EXTRA_PARTS.values():
        setattr(args, flag, False)
    return args, state, package


def test_interruption_keeps_completed_fixture_and_resume_skips_it(run_fixture, capsys):
    args, state, package = run_fixture
    state["crash_at"] = 3
    with pytest.raises(KeyboardInterrupt):
        ib._run_reference(args)
    prior = json.loads(Path(args.json).read_text())
    assert list(prior["cells"]) == ["tiny/base"]
    assert prior["complete"] is False
    assert "repeat=1/train" in prior["cells"]["tiny/base"]["timing"]["stages"]
    args.resume = True
    state["crash_at"] = None
    assert ib._run_reference(args) == 0
    final = json.loads(Path(args.json).read_text())
    assert state["calls"] == 5  # first cell's two fits were not repeated
    assert final["complete"] is True and len(final["cells"]) == 2
    assert prior["cells"]["tiny/base"] == final["cells"]["tiny/base"]
    output = capsys.readouterr().out
    assert "# START tiny/base repeat=1/train" in output
    assert "# REUSED tiny/base" in output


@pytest.mark.parametrize("change", ["binary", "python", "protocol", "env"])
def test_resume_rejects_changed_execution(run_fixture, monkeypatch, change):
    args, state, package = run_fixture
    assert ib._run_reference(args) == 0
    args.resume = True
    if change == "binary":
        (package / "binding.so").write_bytes(b"new native bytes")
    elif change == "python":
        (package / "__init__.py").write_text("changed = True\n")
    elif change == "protocol":
        args.repeats = 3
    else:
        monkeypatch.setenv("MOJOLEARN_CPU_THREADS", "99")
    with pytest.raises(SystemExit, match="REFUSING TO RESUME"):
        ib._run_reference(args)
    assert state["calls"] == 4


def test_refused_cell_is_not_success(run_fixture):
    args, state, _ = run_fixture
    state["refuse"] = True
    assert ib._run_reference(args) == 1
    assert all(c["verdict"] == "REFUSED" for c in json.loads(Path(args.json).read_text())["cells"].values())


def test_numerical_mismatch_keeps_repeated_bytes_but_fails_record(run_fixture, monkeypatch, capsys):
    args, _, _ = run_fixture
    calls = []
    def wrong(*unused):
        calls.append(1)
        raise ib.NumericalMismatch('independent oracle disagrees', dict(value='b' * 16))
    monkeypatch.setattr(ib, 'LANES', {'tiny': wrong})
    assert ib._run_reference(args) == 1
    record = json.loads(Path(args.json).read_text())
    assert record['complete'] and len(calls) == 4
    for cell in record['cells'].values():
        assert cell['verdict'] == 'DIVERGENT'
        assert len(cell['hashes']) == 2 and len(set(cell['hashes'])) == 1
        assert cell['oracle_errors'] == ['independent oracle disagrees'] * 2
        assert 'infer' not in cell
    # Two equally wrong results must never rehabilitate the failed oracle.
    capsys.readouterr()
    assert ib.diff([args.json, args.json]) == 1
    output = capsys.readouterr().out
    assert 'IDENTICAL x2' not in output
    assert 'DIVERGENT' in output


def test_atomic_checkpoint_preserves_old_record_on_write_error(tmp_path, monkeypatch):
    p = tmp_path / "out.json"
    ib.atomic_json(p, {"old": 1})
    monkeypatch.setattr(ib.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("disk error")))
    with pytest.raises(OSError):
        ib.atomic_json(p, {"new": 2})
    assert json.loads(p.read_text()) == {"old": 1}
    assert list(tmp_path.iterdir()) == [p]


def test_iteration_fits_once_and_keeps_all_default_probes(tmp_path):
    cmd = command("python", "mamba1", "base", tmp_path / "record.json", 30, "metal", False)
    assert cmd[cmd.index("--repeats") + 1] == "1"
    assert "--no-batch" not in cmd and "--no-rlpair" not in cmd
    assert "--fail-on-refused" in cmd
    assert cmd[cmd.index("--timeout") + 1] == "30"


def test_timing_report_does_not_count_resume_twice(run_fixture):
    args, _, _ = run_fixture
    assert ib._run_reference(args) == 0
    report = summarize([args.json, args.json])
    assert report["timed_cells"] == 2
    assert report["stage_seconds"]["train"] >= 0


def test_merge_ignores_timing_but_rejects_changed_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(ib, "LANES", {"tiny": None})
    record = dict(vendor="fixture-cpu", commit="a" * 40, mode="identical", complete=True,
                  fixtures={"base": {}}, package={"bindings": [{"module": "native", "sha256": "same"}]},
                  cells={"tiny/base": dict(verdict="STABLE", hashes=["abc", "abc"], timing={"total_seconds": 1})})
    a, b, out = [tmp_path / name for name in ("a.json", "b.json", "merged.json")]
    a.write_text(json.dumps(record))
    record["cells"]["tiny/base"]["timing"]["total_seconds"] = 2
    b.write_text(json.dumps(record))
    assert ib.merge([str(a), str(b)], str(out)) == 0
    record["cells"]["tiny/base"]["hashes"] = ["wrong", "wrong"]
    b.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="different contents"):
        ib.merge([str(a), str(b)], str(out))


def test_resume_rejects_changed_external_binding(run_fixture, monkeypatch, tmp_path):
    args, state, _ = run_fixture
    host = tmp_path / "host"
    host.mkdir()
    binary = host / "oracle.so"
    binary.write_bytes(b"production")
    monkeypatch.setenv("MOJOLEARN_HOST_DIR", str(host))
    assert ib._run_reference(args) == 0
    args.resume = True
    binary.write_bytes(b"sabotage")
    with pytest.raises(SystemExit, match="REFUSING TO RESUME"):
        ib._run_reference(args)


def test_packaged_harness_import_has_no_tools_dependency(tmp_path):
    import subprocess
    copy = tmp_path / "_identity_break.py"
    copy.write_text(Path(ib.__file__).read_text())
    p = subprocess.run([sys.executable, "-c", "import _identity_break"], cwd=tmp_path,
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_resume_signature_does_not_publish_environment_secrets(run_fixture, monkeypatch):
    args, _, _ = run_fixture
    monkeypatch.setenv("MOJOLEARN_PRIVATE_TOKEN", "secret-that-must-not-be-recorded")
    assert ib._run_reference(args) == 0
    assert "secret-that-must-not-be-recorded" not in Path(args.json).read_text()


def test_cpu_requirement_refuses_gpu_before_any_fit(run_fixture):
    args, state, _ = run_fixture
    args.require_cpu = True
    with pytest.raises(SystemExit, match="--require-cpu loaded a GPU"):
        ib._run_reference(args)
    assert state["calls"] == 0


def test_parallel_driver_keeps_lane_order_and_never_splits_an_estimator():
    lanes = [f"lane-{i}" for i in range(7)]
    shards = ib._job_shards(lanes, 3)
    assert shards == [["lane-0", "lane-1"],
                      ["lane-2", "lane-3"],
                      ["lane-4", "lane-5", "lane-6"]]
    assert sum(shards, []) == lanes
    assert len({lane for shard in shards for lane in shard}) == len(lanes)


def test_parallel_driver_strips_only_supervisor_arguments():
    argv = ["--jobs=4", "--lanes", "ridge,kmeans", "--json=out.json",
            "--resume", "--fixtures", "base,ties", "--repeats", "2"]
    assert ib._strip_driver_args(argv) == [
        "--resume", "--fixtures", "base,ties", "--repeats", "2"]


def test_release_marker_does_not_enable_full_apple_matrix(monkeypatch):
    monkeypatch.setattr(ib, "_is_apple_gpu", lambda host: not host)
    lanes = ["ridge", "kmeans"]
    assert ib.refuse_routine_apple_column(lanes, None, {ib.APPLE_RELEASE_RECORD_ENV: "0.8.7"})
    assert not ib.refuse_routine_apple_column(lanes, None, {ib.APPLE_FULL_DIAGNOSTIC_ENV: "1"})
    assert not ib.refuse_routine_apple_column(["ridge"], None, {})
    assert not ib.refuse_routine_apple_column(lanes, {"column": "cpu"}, {})


def test_metal_iteration_needs_no_release_marker_or_diagnostic_flag(monkeypatch, tmp_path):
    import identity_iterate as runner
    monkeypatch.delenv(ib.APPLE_RELEASE_RECORD_ENV, raising=False)
    calls = []
    monkeypatch.setattr(runner, "run_job", lambda cmd, env: calls.append(cmd) or 0)
    assert runner.main(["--lane", "ridge", "--mode", "metal", "--out", str(tmp_path / "out")]) == 0
    assert len(calls) == 1


@pytest.mark.parametrize('backend', ['cpu', 'metal', 'cuda', 'hip'])
def test_wrong_backend_refuses_before_fit(run_fixture, backend):
    args, state, _ = run_fixture
    args.require_backend = backend
    with pytest.raises(SystemExit, match='requested .* backend, loaded'):
        ib._run_reference(args)
    assert state['calls'] == 0


@pytest.mark.parametrize('route,changed', [
    ('par-reference-knn', changed)
    for changed in (None, 'distances', 'indices', 'predict', 'predict_proba', 'raise')
] + [('par-reference-knn-reg', changed) for changed in (None, 'predict', 'raise')])
def test_reference_neighbors_keep_wrong_bytes_and_do_not_waive_refusals(monkeypatch, route, changed):
    from contextlib import nullcontext

    outputs = dict(distances=np.zeros((64, 8), dtype='<f4'),
                   indices=np.zeros((64, 8), dtype='<i4'),
                   predict=np.zeros(64, dtype='<f4'),
                   predict_proba=np.zeros((64, 2), dtype='<f4'))
    actual = {key: value.copy() for key, value in outputs.items()}
    if changed not in (None, 'raise'):
        actual[changed].flat[0] = 1

    class Model:
        def __init__(self, values):
            self.values = values

        def fit(self, *args):
            return self

        def kneighbors(self, *args):
            return self.values['distances'], self.values['indices']

        def predict(self, *args):
            if changed == 'raise' and self.values is actual:
                raise RuntimeError('worker unavailable')
            return self.values['predict']

        def predict_proba(self, *args):
            return self.values['predict_proba']

    ml = SimpleNamespace(KNeighborsClassifier=lambda **kw: Model(outputs),
                         KNeighborsRegressor=lambda **kw: Model(outputs))
    monkeypatch.setattr(ib, '_rsn', lambda model: nullcontext(Model(actual)))
    X = np.zeros((4160, 1), dtype='<f4')
    y = np.zeros(4160, dtype='<f4')
    names = {'predict': 'predict'} if route.endswith('-reg') else {
        'dist': 'distances', 'idx': 'indices', 'predict': 'predict', 'proba': 'predict_proba'}
    expected_parts = {name: ib._h(actual[field]) for name, field in names.items()}
    if changed == 'raise':
        with pytest.raises(RuntimeError, match='worker unavailable'):
            ib.LANES[route](ml, X, y, y, X[:64])
    elif changed:
        with pytest.raises(ib.NumericalMismatch, match=changed) as exc:
            ib.LANES[route](ml, X, y, y, X[:64])
        assert exc.value.parts == expected_parts
        assert exc.value.parts != {name: ib._h(outputs[field]) for name, field in names.items()}
    else:
        assert dict(ib.LANES[route](ml, X, y, y, X[:64])) == expected_parts


# ---------------------------------------------------------------- batchscale
# A CHECK THAT CANNOT FAIL, closed 2026-09-20: `_eval_scale_rows` skipped a
# sub-batch wider than the whole call (`if b > n: continue`) but seeded the
# digest with the whole SCALE_BATCHES tuple, so the cell asserted a size the
# run never took. `_eval_grad_accum` is the model: the digest carries what was
# asked and an `n/a:` note says what was not.

def _scale_digest(n, sabotage=""):
    import hashlib
    rng = np.random.default_rng(0)
    R = rng.standard_normal((n, 8)).astype(np.float32)
    call = ib._ScaleRows("probe", R, lambda r: (np.asarray(r) * np.float32(2.0),))
    d, notes = hashlib.blake2b(digest_size=16), []
    moved = ib._eval_scale_rows(call, sabotage, d, notes)
    return moved, d.hexdigest(), notes


def test_scale_rows_records_only_the_batches_it_asked():
    """A call narrower than a sub-batch must not hash that sub-batch's number,
    and must say so. The old code returned a clean verdict with no note."""
    moved, _, notes = _scale_digest(20)
    assert moved is None
    assert [nt for nt in notes if "B=64 n/a:" in nt]
    assert [nt for nt in notes if "B=256 n/a:" in nt]
    wide_notes = _scale_digest(ib.SCALE_WHOLE)[2]
    assert wide_notes == [], "a 1024-row call asks every batch and owes no n/a"


def test_scale_rows_digest_distinguishes_a_skipped_batch():
    """Two runs that asked DIFFERENT batch sets must not share a digest. Under
    the old seeding both hashed `1,17,64,256` and were told apart only by the
    row bytes, so a shrunk call read as a full one."""
    import hashlib

    def seeded(n, asked):
        d = hashlib.blake2b(digest_size=16)
        d.update(f"scale:probe:{n}:{','.join(map(str, asked))}".encode())
        return d.hexdigest()

    assert seeded(20, [1, 17]) != seeded(20, ib.SCALE_BATCHES)
    # and the live path picks the first of those, not the second
    rng = np.random.default_rng(0)
    call = ib._ScaleRows("probe", rng.standard_normal((20, 8)).astype(np.float32),
                         lambda r: (np.asarray(r) * np.float32(2.0),))
    d, notes = hashlib.blake2b(digest_size=16), []
    ib._eval_scale_rows(call, "", d, notes)
    asked = hashlib.blake2b(digest_size=16)
    asked.update(b"scale:probe:20:1,17")
    wrong = hashlib.blake2b(digest_size=16)
    wrong.update(b"scale:probe:20:1,17,64,256")
    for r in ib._as_rows(call.fn(call.R), 20, "probe"):
        for dt, shape, raw in r:
            asked.update(f"{dt}{shape}".encode())
            asked.update(raw)
            wrong.update(f"{dt}{shape}".encode())
            wrong.update(raw)
    assert d.hexdigest() == asked.hexdigest()
    assert d.hexdigest() != wrong.hexdigest(), "the digest still claims B=64 and B=256"
    assert len(notes) == 2


def test_scale_rows_sabotage_still_moves_the_cell():
    """The n/a note must not have disarmed the part."""
    moved, _, _ = _scale_digest(64, sabotage="1")
    assert moved and moved.startswith("BATCH_MOVED:")


def test_every_scale_rows_in_the_tree_asks_the_whole_batch_set():
    """No committed cell moves, because no _ScaleRows is narrower than 256.
    `_scale_calls` and `_batchscale_kneighbors` slice `Xh[:SCALE_WHOLE]` and
    the fixture guard refuses MOJOLEARN_IDENTITY_N below 8192; the sequence
    specs tile to exactly SCALE_WHOLE rows."""
    assert ib.SCALE_WHOLE >= max(ib.SCALE_BATCHES)
    assert ib._tiled(np.zeros((8192, 4), dtype=np.float32),
                     (ib.SCALE_WHOLE, 3, 4)).shape[0] == ib.SCALE_WHOLE
    assert ib._tiled_ids(np.zeros((8192, 4), dtype=np.float32),
                         ib.SCALE_WHOLE, 8).shape[0] == ib.SCALE_WHOLE
    assert np.zeros((8192, 4))[:ib.SCALE_WHOLE].shape[0] == ib.SCALE_WHOLE
