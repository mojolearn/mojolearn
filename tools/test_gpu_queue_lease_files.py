# SPDX-License-Identifier: Apache-2.0
"""tools/gpu_queue_box.sh writes its lease files whole and reads them validated (lane fix-pod-hold, 2026-11-07).

The 0.8.66 release's NVIDIA pod was deleted under a live hold: an `nvq hold` start truncated lease.last_busy
(`date +%s > file`) while the dispatcher read it back, the empty value made the idle deadline 0 + IDLE_SEC
(lease.deadline=3600 on the pod's archived queue dir), and the pod was judged idle since 1970. Nothing here
reaches a box: the script is read, and its two file helpers run in a local bash."""
import pathlib
import re
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "gpu_queue_box.sh"


def code_lines():
    return [line for line in SCRIPT.read_text().splitlines() if not line.lstrip().startswith("#")]


def test_every_lease_file_write_is_whole_and_every_read_is_validated():
    lines = code_lines()
    # a shell redirection into a lease file truncates it first; only putf (temp, then rename) writes them
    assert [l for l in lines if re.search(r">\s*\$Q/lease\.(last_busy|deadline)\b", l)] == []
    assert [l for l in lines if re.search(r"\bcat\s+\$Q/lease\.(last_busy|deadline)\b", l)] == []
    assert any("putf $Q/lease.deadline" in l for l in lines)
    assert sum("mark_busy" in l for l in lines) >= 4          # submit, hold start and end, dispatcher start
    # a busy pass computes the deadline from its own clock, never from a re-read of the file
    assert any(re.search(r"putf \$Q/lease\.last_busy \$t; lb=\$t", l) for l in lines)


def helpers():
    text = SCRIPT.read_text()
    out = [m.group(0) for name in ("putf", "numf")
           for m in [re.search(rf"^{name}\(\) \{{.*\}};?.*$", text, re.M)]]
    assert len(out) == 2
    return "\n".join(out)


def sh(tmp_path, body):
    got = subprocess.run(["bash", "-c", f"set -uo pipefail\n{helpers()}\n{body}"], cwd=tmp_path,
                         capture_output=True, text=True, timeout=30)
    assert got.returncode == 0, got.stderr
    return got.stdout


def test_a_torn_missing_or_garbage_lease_value_reads_as_the_fallback(tmp_path):
    (tmp_path / "empty").write_text("")
    (tmp_path / "torn").write_text("17941")              # a number is taken as written
    (tmp_path / "junk").write_text("17941x\n")
    out = sh(tmp_path, "for f in empty missing junk torn; do echo \"$f=$(numf $f 99)\"; done")
    assert out.split() == ["empty=99", "missing=99", "junk=99", "torn=17941"]


def test_putf_replaces_the_file_whole_and_leaves_no_temp(tmp_path):
    (tmp_path / "lease.last_busy").write_text("1\n")
    out = sh(tmp_path, "putf lease.last_busy 1794102810; putf lease.last_busy 1794102811; numf lease.last_busy 0")
    assert out == "1794102811"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["lease.last_busy"]
