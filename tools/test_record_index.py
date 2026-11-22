"""tools/record_index.py and the index readers in python/mojolearn/_verify_reference.py."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import record_index as RI  # noqa: E402


def tree(tmp_path):
    root = tmp_path / "repo"
    (root / "tools").mkdir(parents=True)
    (root / "python/mojolearn").mkdir(parents=True)
    shutil.copyfile(ROOT / "python/mojolearn/_verify_reference.py", root / "python/mojolearn/_verify_reference.py")
    shutil.copyfile(ROOT / "tools/record_index.py", root / "tools/record_index.py")
    def put(rel, body):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(body))
        return p
    put("verification/records/identity/a/cpu.json", dict(vendor="cpu", commit="a" * 40, cells={}))
    put("verification/records/identity/a/summary.json", dict(note="not a column"))
    put("bench/results/other/nvidia-h100.json", dict(vendor="nvidia-h100", commit="b" * 40, cells={}))
    return root, put


def run(root, *args):
    return subprocess.run([sys.executable, str(root / "tools/record_index.py"), "--root", str(root), *args],
                          capture_output=True, text=True)


def test_generate_takes_what_the_walk_and_glob_took_and_check_ignores_strays(tmp_path):
    root, put = tree(tmp_path)
    r = run(root, "generate")
    assert r.returncode == 0, r.stdout
    assert "rows=1" in r.stdout
    v = RI.vref(str(root))
    assert [rel for rel, _ in v.read_index(str(root), v.RECORD_INDEX)] == ["verification/records/identity/a/cpu.json"]
    assert [rel for rel, _ in v.read_index(str(root), v.CELLS_INDEX)] == ["bench/results/other/nvidia-h100.json"]
    assert len(v.indexed_cells_paths(str(root))) == 2
    assert run(root, "generate").returncode == 1          # one-time
    assert run(root, "check").returncode == 0
    put("verification/records/identity/b/amd.json", dict(vendor="amd-mi300x", commit="c" * 40, cells={}))
    r = run(root, "check")
    assert r.returncode == 3 and "UNLISTED verification/records/identity/b/amd.json" in r.stdout
    assert v.repository_record_paths([str(root / "verification/records/identity")], str(root)) == [
        str(root / "verification/records/identity/a/cpu.json")]
    assert run(root, "add", str(root / "verification/records/identity/b/amd.json")).returncode == 0
    assert run(root, "check").returncode == 0


def test_a_changed_or_missing_listed_file_is_refused(tmp_path):
    root, put = tree(tmp_path)
    assert run(root, "generate").returncode == 0
    put("verification/records/identity/a/cpu.json", dict(vendor="cpu", commit="d" * 40, cells={}))
    r = run(root, "check")
    assert r.returncode == 1 and "listed with sha256" in r.stdout
    (root / "bench/results/other/nvidia-h100.json").unlink()
    r = run(root, "check")
    assert r.returncode == 1 and "listed but missing" in r.stdout
    assert run(root, "retire", "--reason", "test", str(root / "bench/results/other/nvidia-h100.json")).returncode == 0
