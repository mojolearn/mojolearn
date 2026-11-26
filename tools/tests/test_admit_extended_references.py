"""tools/admit_extended_references.py: the single-device verify references, admitted beside the lane tables."""
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import admit_extended_references as X  # noqa: E402

FORMAT = "mojolearn-distributed-program-verify/1"


def program(digest="a" * 64, status="PASS"):
    return dict(format=FORMAT, status=status, cases=[dict(name="analytic_sgd", status="PASS", digest=digest),
                                                      dict(name="uninterrupted_replay", status="PASS", digest="b" * 64)])


def tree(tmp_path):
    t = tmp_path / "tree"
    (t / "python" / "mojolearn").mkdir(parents=True)
    for name in X.MODULES.values():   # the comparisons only: pure functions of two reports
        shutil.copy2(ROOT / "python" / "mojolearn" / (name + ".py"), t / "python" / "mojolearn")
    return t


def capture(root, column, profile, report):
    path = root / "identity" / "extended-references" / column / (profile + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))


def test_equal_columns_admit_the_cpu_column(tmp_path):
    t = tree(tmp_path)
    for column in ("nvidia", "cpu"):
        capture(tmp_path / column, column, "verify-distributed-program", program())
    written = X.admit([tmp_path / "nvidia", tmp_path / "cpu"], tree=t, commit="c" * 40, say=lambda m: None)
    assert written == ["python/mojolearn/verify_reference/extended/verify-distributed-program.json"]
    doc = json.loads((t / written[0]).read_text())
    assert doc["admitted"] and doc["column"] == "cpu" and doc["columns"] == ["cpu", "nvidia"]
    assert doc["report"]["cases"][0]["digest"] == "a" * 64


def test_columns_that_differ_admit_nothing(tmp_path):
    t = tree(tmp_path)
    capture(tmp_path / "cpu", "cpu", "verify-distributed-program", program())
    capture(tmp_path / "amd", "amd", "verify-distributed-program", program("d" * 64))
    with pytest.raises(X.AdmitError, match="differs from the cpu column"):
        X.admit([tmp_path], tree=t, say=lambda m: None)
    assert not (t / X.REFERENCE_DIR).exists()


def test_a_failed_capture_admits_nothing(tmp_path):
    t = tree(tmp_path)
    capture(tmp_path, "cpu", "verify-partitioned-program",
            dict(program(status="FAIL"), format="mojolearn-partitioned-program-verify/1"))
    with pytest.raises(X.AdmitError, match="did not pass its own checks"):
        X.admit([tmp_path], tree=t, say=lambda m: None)
    capture(tmp_path, "nvidia", "verify-graph", dict(cases={"native_inputs": dict(status="failed")}))
    with pytest.raises(X.AdmitError):
        X.admit([tmp_path / "x"] + [tmp_path], tree=t, say=lambda m: None)


def test_the_shipped_placeholders_read_as_no_reference():
    sys.path.insert(0, str(ROOT / "python"))
    import importlib.util
    spec = importlib.util.spec_from_file_location("ext_for_test", ROOT / "python/mojolearn/_verify_extended.py")
    ext = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ext)
    for profile in ext.REFERENCED:
        assert ext.reference(profile, str(ROOT / "python" / "mojolearn")) is None
