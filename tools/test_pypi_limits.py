# SPDX-License-Identifier: Apache-2.0
"""packaging/pypi_limits.json and its loader: the per-project PyPI limits."""
import json
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packaging"))
import pypi_limits  # noqa: E402

MiB = 2 ** 20


def test_committed_limits_are_the_granted_values():
    assert pypi_limits.file_limit("mojolearn-nvidia") == 250 * MiB
    assert pypi_limits.file_limit("mojolearn-amd") == 250 * MiB
    # mojolearn (macOS wheel + Linux meta wheel) stays at 100 MiB until #12773 is granted
    assert pypi_limits.file_limit("mojolearn") == 100 * MiB
    assert pypi_limits.file_limit("mojolearn-core") == 100 * MiB
    assert pypi_limits.file_limit("mojolearn-nvidia-ptx") == 100 * MiB
    assert pypi_limits.project_limit("mojolearn") == 10 * 2 ** 30
    row = pypi_limits.project("mojolearn")
    assert "pypi/support#12773" in [p["issue"] for p in row["pending"]]


def test_names_normalize_like_pypi():
    assert pypi_limits.file_limit("mojolearn_nvidia") == 250 * MiB
    assert pypi_limits.file_limit("Mojolearn.AMD") == 250 * MiB
    assert pypi_limits.wheel_distribution("mojolearn_nvidia_ptx-0.8.65-py3-none-x.whl") == "mojolearn-nvidia-ptx"
    unknown = pypi_limits.project("mojolearn-other")
    assert unknown["configured"] is False and unknown["file_bytes"] == 100 * MiB


def test_check_sizes_flags_file_and_project_overruns(tmp_path):
    rows = {r["distribution"]: r for r in pypi_limits.check_sizes(
        {"mojolearn": [99 * MiB], "mojolearn-nvidia": [240 * MiB], "mojolearn-amd": [251 * MiB]},
        used={"mojolearn": 10 * 2 ** 30 - 50 * MiB})}
    assert rows["mojolearn"]["file_over"] is False and rows["mojolearn"]["project_over"] is True
    assert rows["mojolearn"]["over"] is True
    assert rows["mojolearn-nvidia"]["over"] is False
    assert rows["mojolearn-amd"]["over"] is True
    assert "OVER mojolearn-amd" in pypi_limits.line(rows["mojolearn-amd"])


def test_check_wheels_reads_file_sizes(tmp_path):
    w = tmp_path / "mojolearn_amd-1.0-py3-none-any.whl"
    with zipfile.ZipFile(w, "w") as z:
        z.writestr("a", b"x" * 100)
    (row,) = pypi_limits.check_wheels([w])
    assert row["distribution"] == "mojolearn-amd" and row["biggest_bytes"] == w.stat().st_size


def test_a_config_with_the_wrong_schema_is_refused(tmp_path):
    bad = tmp_path / "limits.json"
    bad.write_text(json.dumps({"schema": "other", "defaults": {}, "projects": {}}))
    with pytest.raises(ValueError):
        pypi_limits.load(bad)
