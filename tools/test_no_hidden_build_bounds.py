# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE SOURCE FOR BUILD-PATH TIME LIMITS (2026-09-25).

0.8.19 rebuilt every binding cold and hit three different hardcoded 2400 s
build bounds (tools/release061_remote_build.sh, tools/do_release061_leg.sh,
tools/gemm_remote_leg.sh). Raising one left the next to kill the build.

This test reads every build-path script for a numeric time bound long enough
to bound a compile (at least FLOOR seconds) and fails unless that number is
either gone (the script reads it from tools/release_limits.sh) or listed in
ALLOWED below with a reason. An allow-listed number that mirrors a limit
names it, and the test holds the two EQUAL, so it cannot drift either.

    python3 -m pytest -q tools/test_no_hidden_build_bounds.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import release_limits  # noqa: E402

#: The release build path: every script between "build the sets" and the
#: remote `mojo build`, local and remote.
BUILD_PATH = (
    "tools/release061_remote_build.sh",
    "tools/do_release061_leg.sh",
    "tools/hotaisle_release_leg.sh",
    "tools/gemm_remote_leg.sh",
    "tools/release_linux_build.sh",
    "tools/release_linux_cpu_box.sh",
    "tools/release_ubuntu22_build.sh",
    "packaging/linux/build_sets.sh",
)

#: Shorter bounds are operation timeouts (an ssh, an apt, a fetch), not a
#: compile bound; 20 minutes is shorter than any cold binding set builds.
FLOOR = 1200

_NAME = r"[A-Za-z_]*(?:SECONDS|TIMEOUT|CAP|BOUND|BUDGET|DEADLINE|LIMIT|LEASE|MINUTES)[A-Za-z_]*"
PATTERNS = (
    re.compile(r"\btimeout\s+(?:-k\s+\d+\s+|-s\s+\w+\s+)*(\d+)\b"),        # timeout [-k N] 2400
    re.compile(r":-(\d+)\}"),                                                # ${X:-2400}
    re.compile(r"-(?:gt|ge|lt|le|eq)\s+\"?(\d+)\b"),                          # [ $x -gt 2400 ]
    re.compile(r"(?:<=|>=|<|>)\s*(\d+)\b"),                                  # (( x <= 2400 ))
    re.compile(r"\b" + _NAME + r"=\"?(\d+)\b"),                              # WORK_SECONDS=2400
)

#: (script, number, reason[, limit name the number must equal]).
ALLOWED = (
    ("tools/release061_remote_build.sh", 6000,
     "the MOJOLEARN_RELEASE_BUILD_SECONDS range guard. This script is a LINUX_BUILDERS identity "
     "input (tools/release_reuse.py): editing it would change every Linux binding's identity and "
     "force a full rebuild, so the literal stays and is held equal to the limit",
     "RELEASE_BUILD_SECONDS_MAX"),
    ("tools/gemm_remote_leg.sh", 2880,
     "SEGMENT_CAP_MINUTES: the 48 h segment-lease ceiling in MINUTES, a rental bound, not a build bound"),
    ("tools/gemm_remote_leg.sh", 2400,
     "P9_DIAG_TIMEOUT (the phase 9 diagnostic) and the HIGGS download in the speed body; neither "
     "is on the release build path, whose cap is RUNPOD_RELEASE_BUILD_CAP"),
    ("tools/gemm_remote_leg.sh", 1500,
     "MOJOLEARN_SPEED_LGBM_BUILD: the speed leg's LightGBM opponent build, not a mojolearn binding"),
    ("tools/gemm_remote_leg.sh", 1200,
     "the YEAR dataset download in the speed body, not a build"),
    ("tools/gemm_remote_leg.sh", 2700,
     "byte-LM resume campaign profile 6: reuses a retained binding, builds nothing"),
    ("tools/gemm_remote_leg.sh", 3000,
     "byte-LM training campaign profiles: model steps, not the release build"),
)


def hits(text):
    """{number: [line numbers]} for every numeric bound >= FLOOR outside a comment."""
    out = {}
    for n, line in enumerate(text.splitlines(), 1):
        code = line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
        if not code.strip():
            continue
        for pat in PATTERNS:
            for m in pat.finditer(code):
                v = int(m.group(1))
                if v >= FLOOR:
                    out.setdefault(v, set()).add(n)
    return {k: sorted(v) for k, v in out.items()}


def allowed_for(script):
    return {row[1]: row for row in ALLOWED if row[0] == script}


def test_every_build_path_bound_is_sourced_or_allow_listed():
    bad = []
    for rel in BUILD_PATH:
        path = ROOT / rel
        if not path.is_file():
            continue
        allow = allowed_for(rel)
        for value, lines in hits(path.read_text(errors="replace")).items():
            if value not in allow:
                bad.append(f"{rel}:{','.join(map(str, lines))}: {value} s is a hardcoded bound; "
                           f"put it in tools/release_limits.sh and read it from there, or allow-list "
                           f"it in {Path(__file__).name} with a reason")
    assert not bad, "\n".join(bad)


def test_allow_list_rows_are_live_and_reasoned():
    """An allow-list row whose number is gone is a stale excuse; delete it."""
    for row in ALLOWED:
        rel, value, reason = row[:3]
        assert len(reason) > 20, row
        found = hits((ROOT / rel).read_text(errors="replace"))
        assert value in found, f"{rel}: allow-listed {value} no longer appears; drop the row"


def test_mirrored_literals_equal_their_limit():
    for row in ALLOWED:
        if len(row) == 4:
            assert release_limits.get(row[3]) == row[1], (
                f"{row[0]} holds {row[1]} but tools/release_limits.sh {row[3]}="
                f"{release_limits.get(row[3])}; change both together")


def test_the_limits_file_is_sourced_by_the_scripts_it_governs():
    for rel, name in (("tools/do_release061_leg.sh", "DO_RELEASE_BUILD_CAP"),
                      ("tools/do_release061_leg.sh", "DO_RELEASE_DEADMAN_SECONDS"),
                      ("tools/release_linux_cpu_box.sh", "CPU_BOX_RELEASE_BUILD_SECONDS"),
                      ("tools/hotaisle_release_leg.sh", "HOTAISLE_RELEASE_BUILD_CAP"),
                      ("tools/gemm_remote_leg.sh", "RUNPOD_RELEASE_BUILD_CAP")):
        text = (ROOT / rel).read_text()
        assert "tools/release_limits.sh\"" in text and "\n. " in text, rel
        assert name in text, (rel, name)
        assert name in release_limits.LIMITS, name


def test_limits_values():
    """The values moved here unchanged from the scripts they came from."""
    lim = release_limits.LIMITS
    assert lim["RELEASE_BUILD_SECONDS_MIN"] == 120
    assert lim["RELEASE_BUILD_SECONDS_MAX"] == 6000
    assert lim["DO_RELEASE_BUILD_CAP"] == 5700
    assert lim["RUNPOD_RELEASE_BUILD_CAP"] == 6000
    assert lim["HOTAISLE_RELEASE_BUILD_CAP"] == 2400
    assert lim["DO_RELEASE_DEADMAN_SECONDS"] == 7200
    assert lim["CPU_BOX_RELEASE_BUILD_SECONDS"] == 2400
    assert lim["CROSS_COMPILE_SECONDS"] == 600
    # no remote cap may exceed what the remote build script accepts
    for name in ("DO_RELEASE_BUILD_CAP", "RUNPOD_RELEASE_BUILD_CAP", "HOTAISLE_RELEASE_BUILD_CAP",
                 "CPU_BOX_RELEASE_BUILD_SECONDS"):
        assert lim["RELEASE_BUILD_SECONDS_MIN"] <= lim[name] <= lim["RELEASE_BUILD_SECONDS_MAX"], name


def test_the_detector_sees_the_bug_it_exists_for():
    """Perturb first: the three 0.8.19 spellings must be caught."""
    for line in ('[ "$WORK_SECONDS" -gt 2400 ] && WORK_SECONDS=2400',
                 'MOJOLEARN_RELEASE_BUILD_SECONDS=2400 MOJOLEARN_BUILD_JOBS="$J" \\',
                 'DEADMAN_SECONDS="${DEADMAN_SECONDS:-3600}"',
                 'BUILD_CAP="${MOJOLEARN_RELEASE_BUILD_CAP:-5700}"',
                 '((seconds >= 120 && seconds <= 2400)) || exit 2',
                 'timeout -k 20 2400 bash tools/release061_remote_build.sh'):
        assert hits(line), line
    assert not hits("# a comment about 2400 s")
    assert not hits("timeout -k 10 60 apt-get update")


def test_the_limits_file_is_plain_assignments(tmp_path):
    bad = tmp_path / "limits.sh"
    bad.write_text("X=1\nif true; then Y=2; fi\n")
    try:
        release_limits.load(bad)
    except ValueError:
        return
    raise AssertionError("release_limits.load accepted shell logic")
