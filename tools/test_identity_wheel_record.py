# SPDX-License-Identifier: Apache-2.0
"""tools/identity_wheel_record.py: the one identity run on the built wheels. A fake recorder stands in for
record_identity_column.sh (no venv, no device, no lane runs)."""
import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import identity_wheel_record as R  # noqa: E402
import release_identity_wheels as RIW  # noqa: E402

FAKE = r'''#!/bin/bash
# a fake record_identity_column.sh: SELECT_ONLY prints the owed lanes from $FAKE_OWED_<label>, a chunk "records"
label=$1; out=$2
key=FAKE_OWED_$(echo "$label" | tr -c 'a-z0-9\n' '_')
owed=${!key:-}
if [ "${SELECT_ONLY:-0}" = 1 ]; then
  if [ -z "$owed" ]; then echo "REUSED 9 / RECORD 0 $label"; echo "NOTHING-CHANGED $label"; exit 0; fi
  echo "REUSED 7 / RECORD 2 $label"; echo "LANES-SELECTED $owed"; exit 0
fi
[ "$RECORD_FROM_INSTALLED" = 1 ] && [ "$LANE_SELECTION" = auto-reuse-chunk ] || { echo "bad env"; exit 9; }
echo "{}" > "$out/$label.lanes-x.identical.json"
echo "DEVICE ${CUDA_VISIBLE_DEVICES:-${HIP_VISIBLE_DEVICES:-none}} LANES $LANES"
echo "RECORDED $label.lanes-x rc=0 admissible=yes json=$out/$label.lanes-x.identical.json"
'''


@pytest.fixture
def fake(tmp_path, monkeypatch):
    script = tmp_path / "fake_recorder.sh"
    script.write_text(FAKE)
    monkeypatch.setattr(R, "RECORDER", script)
    monkeypatch.setattr(R.Run, "install", lambda self: (pathlib.Path(sys.executable),
                                                        self.out / "installed-wheels.txt"))
    monkeypatch.setattr(R, "gpu_free_gib", lambda vendor: 40.0)
    return monkeypatch


def args(tmp_path, *extra):
    w = tmp_path / "w.whl"
    w.write_bytes(b"w")
    (tmp_path / "out").mkdir(exist_ok=True)
    (tmp_path / "out" / "installed-wheels.txt").write_text("w.whl " + "a" * 64 + "\n")
    return R.parse(["--wheel", str(w), "--sha256", "w.whl=" + "a" * 64, "--out", str(tmp_path / "out"), *extra])


def test_slots_put_each_column_on_its_own_gpu_when_there_are_enough():
    assert R.plan_slots(["nvidia", "ptx"], [0, 1], 2) == {"nvidia": [(0, 0), (0, 1)], "ptx": [(1, 0), (1, 1)]}
    assert R.plan_slots(["nvidia", "ptx"], [0], 1) == {"nvidia": [(0, 0)], "ptx": [(0, 0)]}
    assert R.plan_slots(["nvidia"], [0, 1, 2], 1) == {"nvidia": [(0, 0), (1, 0), (2, 0)]}
    assert R.plan_slots(["cpu"], [], 3) == {"cpu": [(None, 0), (None, 1), (None, 2)]}


def test_workers_use_every_core_bounded_by_memory():
    assert R.auto_workers(0, 4, None) == 1 and R.auto_workers(0, 16, None) == 5
    assert R.auto_workers(1, 32, 80.0) == 6          # cores: 32 // 3 = 10, memory: 80 // 12 = 6
    assert R.auto_workers(2, 16, None) == 1
    assert R.chunks(["a", "b", "c", "d", "e"], 2) == [["a", "c", "e"], ["b", "d"]] and R.chunks([], 3) == []


def test_two_columns_on_two_gpus_record_once_in_parallel(tmp_path, fake, capsys):
    fake.setenv("FAKE_OWED_nvidia_l40s", "knn,ols")
    fake.setenv("FAKE_OWED_nvidia_ptx_l40s_sm80", "ridge")
    a = args(tmp_path, "--columns", "nvidia,ptx", "--label-device", "l40s", "--gpus", "2", "--workers-per-gpu", "1")
    assert R.Run(a).main() == 0
    doc = json.loads((tmp_path / "out" / "identity-wheels.json").read_text())
    assert doc["ok"] and doc["columns"]["nvidia"]["owed"] == 2 and doc["columns"]["ptx"]["owed"] == 1
    assert {c["device"] for c in doc["columns"]["nvidia"]["chunks"]} == {0}
    assert {c["device"] for c in doc["columns"]["ptx"]["chunks"]} == {1}
    out = capsys.readouterr().out
    assert "busy plan: 2 worker(s) of 2" in out and "nvidia-l40s: REUSED 7 / RUN 2" in out
    assert RIW.summary_errors(doc, ["nvidia", "ptx"]) == []


def test_nothing_owed_records_nothing_and_is_ok(tmp_path, fake, capsys):
    a = args(tmp_path, "--columns", "amd", "--label-device", "mi325x", "--gpus", "1")
    assert R.Run(a).main() == 0
    doc = json.loads((tmp_path / "out" / "identity-wheels.json").read_text())
    assert doc["columns"]["amd"]["owed"] == 0 and doc["columns"]["amd"]["chunks"] == []
    assert "amd-mi325x: REUSED 9 / RUN 0" in capsys.readouterr().out
    assert RIW.summary_errors(doc, ["amd"]) == []


def test_a_cpu_shard_records_its_share(tmp_path, fake):
    fake.setenv("FAKE_OWED_cpu", "a,b,c,d,e")
    a = args(tmp_path, "--columns", "cpu", "--shard", "1/2", "--workers-per-gpu", "2")
    assert R.Run(a).main() == 0
    doc = json.loads((tmp_path / "out" / "identity-wheels.json").read_text())
    assert doc["columns"]["cpu"]["owed"] == 2 and sum(c["lanes"] for c in doc["columns"]["cpu"]["chunks"]) == 2


def test_a_wheel_that_is_not_the_release_bytes_is_refused(tmp_path):
    w = tmp_path / "w.whl"
    w.write_bytes(b"other")
    a = R.parse(["--columns", "cpu", "--wheel", str(w), "--sha256", "w.whl=" + "a" * 64, "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit, match="not the release's"):
        R.Run(a).install()


def test_summary_errors_name_what_is_missing():
    assert RIW.summary_errors(None, ["amd"]) == ["no identity-wheels summary"]
    doc = dict(schema="mojolearn.identity-wheels.v1", ok=False,
               columns={"nvidia": dict(rc=1, owed=2, chunks=[dict(worker=0, lanes=1, records=[dict(admissible=False)])])})
    errs = RIW.summary_errors(doc, ["nvidia", "ptx"])
    assert any("columns" in e for e in errs) and any("exited 1" in e for e in errs)
    assert any("recorded 1 of 2" in e for e in errs) and any("inadmissible" in e for e in errs)
