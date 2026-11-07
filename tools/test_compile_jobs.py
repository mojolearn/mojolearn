# SPDX-License-Identifier: Apache-2.0
"""bindings/compile_jobs.sh: the one `-j` rule of every binding build (plan N456). A caller on a rented Linux host
that does not export MOJOLEARN_COMPILE_JOBS gets every core it may use (cgroup quota respected), never a fixed 1 or
2; macOS keeps 2; the laptop's 1 comes from tools/compile_slot.sh only. Fake `uname` / `nproc` and a fake cgroup tree
stand in for the host; nothing is compiled."""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "bindings" / "compile_jobs.sh"
SOURCE_LINE = '. "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)/compile_jobs.sh"'


def resolve(tmp_path, system="Linux", cores="64", cgroup=None, jobs=None):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name, out in (("uname", system), ("nproc", cores)):
        p = bin_dir / name
        p.write_text("#!/bin/sh\necho %s\n" % out)
        p.chmod(0o755)
    root = tmp_path / "cgroup"
    root.mkdir(exist_ok=True)
    for rel, text in (cgroup or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text + "\n")
    env = {"PATH": "%s:/usr/bin:/bin" % bin_dir, "MLCJ_CGROUP_ROOT": str(root)}
    if jobs is not None:
        env["MOJOLEARN_COMPILE_JOBS"] = jobs
    r = subprocess.run(["/bin/sh", "-c", 'set -eu; . "%s"; echo "$MOJOLEARN_COMPILE_JOBS_RESOLVED"' % HELPER],
                       env=env, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr


def test_an_explicit_count_is_used_as_is(tmp_path):
    assert resolve(tmp_path, jobs="3")[:2] == (0, "3")
    assert resolve(tmp_path, system="Darwin", jobs="1")[:2] == (0, "1")


@pytest.mark.parametrize("bad", ["0", "-1", "two", "2 3", "08"])
def test_a_malformed_count_stops_the_build(tmp_path, bad):
    rc, _, err = resolve(tmp_path, jobs=bad)
    assert rc == 2 and "MOJOLEARN_COMPILE_JOBS" in err


def test_unset_on_linux_is_every_core(tmp_path):
    assert resolve(tmp_path, cores="64")[:2] == (0, "64")
    assert resolve(tmp_path, cores="64", jobs="")[:2] == (0, "64")


def test_unset_on_macos_is_two(tmp_path):
    assert resolve(tmp_path, system="Darwin", cores="12")[:2] == (0, "2")


@pytest.mark.parametrize("cgroup, want", [
    ({"cpu.max": "400000 100000"}, "4"),              # cgroup v2 quota of 4 cores
    ({"cpu.max": "250000 100000"}, "3"),              # a partial core rounds up
    ({"cpu.max": "max 100000"}, "64"),                # no quota
    ({"cpu.max": "9000000 100000"}, "64"),            # a quota above nproc keeps nproc
    ({"cpu/cpu.cfs_quota_us": "800000", "cpu/cpu.cfs_period_us": "100000"}, "8"),   # cgroup v1
    ({"cpu/cpu.cfs_quota_us": "-1", "cpu/cpu.cfs_period_us": "100000"}, "64"),      # v1, no quota
])
def test_a_container_quota_caps_the_cores(tmp_path, cgroup, want):
    assert resolve(tmp_path, cores="64", cgroup=cgroup)[:2] == (0, want)


def build_scripts():
    return sorted(p for p in (ROOT / "bindings").glob("*.sh")
                  if any("mojo build" in l and not l.lstrip().startswith("#") for l in p.read_text().splitlines()))


def test_no_build_script_keeps_a_fixed_default():
    offenders = [p.name for p in sorted((ROOT / "bindings").glob("*.sh"))
                 if re.search(r"MOJOLEARN_COMPILE_JOBS:-[0-9]", p.read_text())]
    offenders += ["tools/build_native_adapters.sh"] * bool(
        re.search(r"MOJOLEARN_COMPILE_JOBS:-[0-9]", (ROOT / "tools" / "build_native_adapters.sh").read_text()))
    assert offenders == []


def test_every_compile_line_takes_the_resolved_count():
    scripts = build_scripts()
    assert len(scripts) >= 30
    for path in scripts:
        lines = path.read_text().splitlines()
        assert lines[2].startswith(SOURCE_LINE), path.name + ": compile_jobs.sh not sourced on line 3"
        joined = path.read_text().replace("\\\n", " ")
        compile_lines = [l for l in joined.splitlines() if "build -j" in l and not l.lstrip().startswith("#")]
        assert compile_lines, path.name
        for line in compile_lines:
            assert re.search(r'build -j "\$(MOJOLEARN_COMPILE_JOBS_RESOLVED|jobs)"', line), (path.name, line)


def test_the_laptop_rule_lives_in_the_compile_slot():
    text = (ROOT / "tools" / "compile_slot.sh").read_text()
    assert re.search(r'uname -s\)" = Darwin \] && \[ -z "\$\{MOJOLEARN_COMPILE_JOBS:-\}" \]', text)
    assert "export MOJOLEARN_COMPILE_JOBS=1" in text
    # the export comes before the slot loop runs the command
    assert text.index("export MOJOLEARN_COMPILE_JOBS=1") < text.index('nice -n 19 "$@"')


def test_the_helper_is_a_keyed_build_input():
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    import binding_stamps
    import development_build
    assert "bindings/compile_jobs.sh" in binding_stamps.MERGED_INPUTS
    assert "bindings/compile_jobs.sh" in development_build.BUILD_INPUTS
