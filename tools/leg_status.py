# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""WHY A BOARD LOOKED FULL WHEN THE LEG HAD FALLEN OVER, AND WHAT FIXES IT.

    python3 tools/leg_status.py <run-dir> [--out STATUS.md] [--json STATUS.json]

On 2026-08-28 nine separate harness defects produced boards "full of NO
OPPONENT or plausible rows rather than errors"
(`bench/results/BOARD_2026-08-28_three-vendor.md` part 3): a gemm arm whose
IDENTICAL build had not compiled since 2026-08-25, a `git rev-parse` that
killed the NVIDIA matrix, iforest cards written to a scratch path, a missing
`einops`, a torch arm refusing every non-CUDA box, and an AMD leg with no
dataset download step that quietly ran sixty timed rounds on a synthetic
fallback fixture. Not one of them made a table say anything was wrong.

THE MECHANISM IS THE SAME EVERY TIME AND IT IS NOT SUBTLE. A speed board is
built by `tools/fast_speed_table.py`, which parses the `FSPEED*` lines an arm
prints. An arm that dies before printing anything contributes NO LINES, and
the absence of lines renders exactly like the absence of an opponent. The
table cannot tell "cuML is not installed on an Apple box, correctly and by
design" from "our arm segfaulted" from "the box killed it at the budget",
because in all three cases what reaches the parser is nothing at all.

**The evidence to separate them already exists and was simply never read.**
`tools/local_speed_run.sh` writes `arm_exit <log>=<rc>` for every arm it runs
and `build_exit <name>=<rc>` for every binary it builds, both into `leg.txt`,
and has done so all along. No board has ever opened that file. This file is
that read.

WHAT THIS PRODUCES
==================
One row per arm, each in exactly one CLASS, each class in exactly one
SEVERITY, and a process exit code taken from the worst severity present:

    severity  exit  meaning for a reader of the board
    ok          0   the row is what it says it is
    warn        3   the row exists but is weaker than it looks
    infra       3   the BOX failed, not the code -- see below
    defect      1   our side broke; the board is not evidence of anything
    unknown     1   we cannot say, which is treated as `defect`

THE POINT OF THE INFRA CLASS, AND ITS ONE DANGER
=================================================
Andrew's ask, 2026-08-30: name the reason an arm failed -- a RunPod lease
expiring, an AMD readiness timeout, a per-arm budget -- so that a red cell is
not read as a broken library. That is right, and the repository has already
paid for the lesson twice: the 2026-08-27 "outage" was a NEGATIVE RUNPOD
BALANCE, and the standing rule from 2026-08-29 is that an infra failure never
retracts a green card (`RunPod/AMD failure is NOT invalidation`).

**The danger is the obvious one: a classifier that can call a failure
"infra" is a machine for explaining bugs away.** So the rule in this file,
which is not negotiable and is why the table below is a closed list of
literal signatures rather than a heuristic:

    A FAILURE IS INFRA ONLY IF THE LOG SAYS SO IN WORDS THE BOX WROTE.
    EVERYTHING ELSE IS OURS. An unrecognised failure classifies as
    `unknown`, which carries the SAME exit code as `defect`, never the
    same one as `infra`.

`infra` therefore never lowers the alarm below where an unclassified failure
would have put it -- it only says which of two already-loud things happened.
The one place it does change a verdict is the exit code, and it moves it from
1 to 3, not to 0. There is no path through this file that turns a failure
into a pass.

A REFUSAL BY `arm=ours` IS NOT A REFUSAL
========================================
The single most valuable rule here, and it was found by pointing this file at
`fast_speed/do-2026-08-28_140119-amd-forest`, whose
`forest.gbdt-symmetric.r1000000.log` exits 0 and contains THREE refusals of
three different kinds:

    arm=catboost-cpu  GPU-PATH-ONLY ...                 <- our own policy
    arm=catboost-gpu  CUDA error 35: CUDA driver
                      version is insufficient ...        <- the box
    arm=ours          Exception during warm-up:
                      half-byte histograms on a
                      64-lane column (DEVIATION 1910)    <- OUR DEFECT

All three render identically on today's board: one row each in "Refused arms,
kept rather than dropped". The third is our learner failing to run on AMD at
all, and it is the reason six gbdt cells on the three-vendor board say
`unrun`. It has to be louder than the first two, so:

    A refusal printed by `arm=ours` is a DEFECT unless its reason is a
    refusal this library MEANT to make -- an unported option named by name,
    or IDENTICAL declining a shape it cannot pin. "Exception during warm-up"
    is not that.

Classification is therefore per ARM inside a log, not per log file. One
process can hold an ok row, an infra row and a defect row at once, and
collapsing them to a per-file verdict would lose exactly the one that matters.

WHAT THIS DELIBERATELY DOES NOT DO
===================================
It does not read timings, does not compute ratios and does not judge a
number. `fast_speed_table.py` owns the measurement; this file owns whether
the measurement happened. Keeping them apart is what lets this one be
developed on a laptop against a directory of old logs, which is how it was
written.
"""

import argparse
import json
import os
import re
import sys


# --------------------------------------------------------------------------
# THE CLASSES. Ordered worst-first for reporting; severity drives the exit.
# --------------------------------------------------------------------------

SEVERITY_EXIT = {"ok": 0, "warn": 3, "infra": 3, "defect": 1, "unknown": 1}

#: HOW BAD, which is NOT the same order as the exit code and must not be
#: derived from it. `infra` exits 3 and `defect` exits 1 because a caller
#: wants to branch on WHICH failure, not on which is worse; ranking by the
#: exit number made a leg with six of our own arms dead report its verdict as
#: INFRA, which is precisely the misreading this file exists to prevent.
SEVERITY_RANK = {"ok": 0, "warn": 1, "infra": 2, "defect": 3, "unknown": 4}

#: severity, and the one-line reading a board carries beside the row.
CLASSES = {
    "BUILD_FAILED": (
        "defect",
        "the binary did not compile, so the arm never ran and the lane is"
        " absent from the board rather than failing on it"),
    "NOT_RUN": (
        "defect",
        "the runner skipped this arm because its binary was missing"),
    "OURS_RAISED": (
        "defect",
        "our arm exited non-zero after raising by name"),
    "CRASHED": (
        "unknown",
        "non-zero exit with nothing in the log this file can classify;"
        " treated as ours until someone reads it"),
    "NO_OUTPUT": (
        "unknown",
        "the arm exited 0 and printed no FSPEED line at all, which is not a"
        " refusal and not a result"),
    "EMPTY_LEG": (
        "unknown",
        "this run directory yielded no arms at all, so the reader is looking"
        " at a leg that either never ran or is laid out in a shape this file"
        " cannot read. Either way it is not an empty set of problems."),
    "TIMEOUT": (
        "infra",
        "killed by the per-arm budget or by timeout(1) before it finished"),
    "OOM_KILLED": (
        "infra",
        "killed by the kernel or refused by an allocator"),
    "INFRA": (
        "infra",
        "the box, the lease, the driver or the network failed in words the"
        " box itself wrote"),
    "PARTIAL_BUDGET": (
        "warn",
        "timed rounds exist but the arm was cut off, so its median stands on"
        " fewer samples than the run asked for"),
    "REFUSED_POLICY": (
        "ok",
        "refused by our own rule, most often GPU-PATH-ONLY on a vendor box"),
    "REFUSED_PEER": (
        "ok",
        "the opponent does not exist on this box and said so by name"),
    "OK": ("ok", "ran to completion and produced timed rounds"),
}

REPORT_ORDER = [
    "BUILD_FAILED", "NOT_RUN", "OURS_RAISED", "CRASHED", "NO_OUTPUT",
    "EMPTY_LEG", "TIMEOUT", "OOM_KILLED", "INFRA", "PARTIAL_BUDGET",
    "REFUSED_POLICY", "REFUSED_PEER", "OK",
]


# --------------------------------------------------------------------------
# THE SIGNATURE TABLE. Every entry is a string a MACHINE printed, copied from
# a log in `bench/results/`. Nothing here is a guess about what a failure
# might look like; an entry is added when a real leg produced it, and the
# board that produced it is named.
# --------------------------------------------------------------------------

#: (regex, class, human reason). First match wins, so order is priority.
SIGNATURES = [
    # ---- the box killed us -------------------------------------------
    (r"KILLED BY THE PER-ARM BUDGET", "TIMEOUT",
     "per-arm budget (tools/local_speed_run.sh watchdog)"),
    (r"\bterminated\b.*\btimeout\b|\btimeout: sending signal\b", "TIMEOUT",
     "coreutils timeout(1) on the rented payload"),
    (r"Killed\s*$|signal 9|SIGKILL", "OOM_KILLED",
     "SIGKILL, which on these boxes is the OOM killer"),
    (r"std::bad_alloc|CUDA out of memory|HIP out of memory"
     r"|MemoryError|hipErrorOutOfMemory|CUDA_ERROR_OUT_OF_MEMORY",
     "OOM_KILLED", "the allocator refused"),

    # ---- the box itself was wrong ------------------------------------
    # Every one of these was measured. The catboost-gpu and lightgbm-cuda
    # lines are verbatim from `fast_speed/2026-08-28-AMD-forest-higgs.md`,
    # where they are the CORRECT behaviour of a CUDA library on an MI325X
    # and must never read as a defect in this repository.
    (r"CUDA driver version is insufficient", "INFRA",
     "a CUDA library on a box with no usable CUDA driver"),
    (r"no CUDA-capable device|CUDA driver init|nvidia-smi.*failed"
     r"|couldn't communicate with the NVIDIA driver", "INFRA",
     "the NVIDIA driver did not answer"),
    (r"hipErrorNoDevice|No HIP devices|rocm-smi.*not found", "INFRA",
     "the ROCm stack did not answer"),
    (r"Connection reset by peer|Temporary failure in name resolution"
     r"|Could not resolve host|Connection timed out|SSL.*handshake",
     "INFRA", "the network dropped mid-leg"),
    (r"No space left on device|Disk quota exceeded", "INFRA",
     "the box ran out of disk"),
    # `402 ` USED TO BE AN ALTERNATIVE HERE AND IT WAS A BUG, kept as the
    # reason this table is prose-only. It matched `ms=1332.402 ` in a timing
    # line and classified a healthy Apple arm with eighteen timed rounds as
    # "the rented account could not pay". A signature that can match a NUMBER
    # is a signature that can explain away a real defect, which is the one
    # thing this file must never do. Every entry is a PHRASE.
    (r"insufficient funds|insufficient balance|payment required"
     r"|negative balance|HTTP 402", "INFRA",
     "the rented account could not pay, which is the 2026-08-27 outage"),
    (r"pod (terminated|stopped|expired)|instance was preempted"
     r"|lease expired", "INFRA", "the lease ended under the leg"),

    # ---- our side broke ----------------------------------------------
    # Ordered AFTER the box signatures on purpose: a mojo traceback that
    # follows a driver failure is a consequence, not the cause.
    (r"^Error:.*mojolearn|Unhandled exception|mojo: error:", "OURS_RAISED",
     "our binary raised"),
    (r"error: .*\.mojo:\d+", "OURS_RAISED", "a Mojo compile error"),
]

#: Refusals are the arms that were RIGHT to stop. `reason=` is free text by
#: design (the arm explains itself in its own words), so this table reads it
#: rather than constraining it, and an unmatched refusal is still a refusal:
#: the arm reached the point of printing FSPEED-REFUSED, which a crash does
#: not do. It classifies as REFUSED_PEER, the milder of the two ok classes.
POLICY_REFUSAL = re.compile(
    r"GPU-PATH-ONLY|CPU arm refused|by policy|not a legal peer"
    r"|refuses the CPU arm", re.I)

#: A refusal reason that means the arm BROKE rather than declined. Matched
#: only for `arm=ours`, because an opponent that throws on this box is a fact
#: about the box and ours throwing is a fact about us. Every string here is
#: from a real log; "Exception during warm-up" is the AMD gbdt case in the
#: module docstring.
OURS_BROKE = re.compile(
    r"Exception during|Error during|Traceback|raised at|segmentation fault"
    r"|assertion failed|panic|failed to build|did not build", re.I)

#: ...and the refusals `arm=ours` is DESIGNED to make. This library refuses
#: by name on purpose in two places -- an unported option (`glm/NOT_IMPLEMENTED.tsv`
#: and every `check()` that names its option) and IDENTICAL declining a shape
#: whose summation order it cannot pin (`gemm_tn`'s row 27 refusal). Those
#: are the port working, not failing, and they must not be reported as
#: defects or the class stops meaning anything.
OURS_BY_DESIGN = re.compile(
    r"NOT PORTED|is not ported|refuses .*by name|UNPORTED"
    r"|NUMERIC_IDENTICAL refuses|IDENTITY_PATHS row"
    r"|no algorithm with this id", re.I)


#: Logs the RUNNER writes about itself rather than about an arm. These are
#: transcripts, not measurements, and counting them as arms invents defects.
RUNNER_OWNED = re.compile(
    r"^(build[._]|console\.|speed_run\.|pixi_install\.|bootstrap\.|leg\.)")

HAS_HEADER = re.compile(r"^FSPEED-HEADER\s", re.M)

ARM_EXIT = re.compile(r"^arm_exit\s+(\S+)=(-?\d+)\s*$")
BUILD_EXIT = re.compile(r"^build_exit\s+(\S+)=(-?\d+)\s*$")
EXPECT_ARM = re.compile(r"^expect_arm\s+(\S+)(?:\s+build=(\S+))?\s*$")
FSPEED_ROUND = re.compile(r"^FSPEED\s")
FSPEED_REFUSED = re.compile(r"^FSPEED-REFUSED\s")


def _tail(path, n_bytes=200000):
    """The last of a log, which is where a failure says why.

    Bounded because a crashed arm can leave a very large log and this file
    is meant to run on a laptop against a directory of them."""
    try:
        size = os.path.getsize(path)
        with open(path, "r", errors="replace") as fh:
            if size > n_bytes:
                fh.seek(size - n_bytes)
            return fh.read()
    except OSError:
        return ""


def _read_leg(run_dir):
    """`arm_exit`, `build_exit` and `expect_arm` out of leg.txt.

    `expect_arm` is written by the runner from 2026-08-30 and is what makes
    a NEVER-RAN arm visible. Legs recorded before that date have none, and
    this file falls back to "what has a log or an arm_exit line", which is
    exactly the blind spot `expect_arm` closes -- so an old run is reported
    honestly as having no expectation list rather than as complete."""
    arm_exit, build_exit, expected = {}, {}, {}
    # A leg fetched off a droplet nests its record at `<out>/run/leg.txt`,
    # so this walks rather than joins, and reads EVERY leg.txt it finds: a
    # resumed leg can hold more than one. Reading only `<out>/leg.txt` made
    # twelve arms that had run and reported five timed rounds each classify
    # as never-run, because their exit codes were in a file this function
    # was not looking at.
    paths = []
    for dirpath, _, names in os.walk(run_dir):
        if "leg.txt" in names:
            paths.append(os.path.join(dirpath, "leg.txt"))
    if not paths:
        return arm_exit, build_exit, expected, False
    for path in sorted(paths):
      with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            m = ARM_EXIT.match(line)
            if m:
                arm_exit[m.group(1)] = int(m.group(2))
                continue
            m = BUILD_EXIT.match(line)
            if m:
                build_exit[m.group(1)] = int(m.group(2))
                continue
            m = EXPECT_ARM.match(line)
            if m:
                expected[m.group(1)] = m.group(2)
    return arm_exit, build_exit, expected, bool(expected)


def _log_facts(index, log_name):
    """Did this arm print rounds, print a refusal, and what does its tail say.

    `index` maps a log's basename to its full path and is built by walking
    the run directory, because the leg layouts differ: `local_speed_run.sh`
    writes `<out>/logs/`, while `do_speed_leg.sh` fetches a droplet into
    `<out>/run/logs/`. Joining a fixed relative path found the first and
    silently found NOTHING in the second, which this file then reported as a
    clean leg -- the exact defect class it exists to catch, committed inside
    the fix for it. It is recorded here rather than quietly corrected."""
    path = index.get(log_name)
    if path is None or not os.path.exists(path):
        return None
    rounds = 0
    refusals = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if FSPEED_ROUND.match(line):
                rounds += 1
            elif FSPEED_REFUSED.match(line):
                refusals.append(parse_kv(line.split(" ", 1)[1]
                                         if " " in line else ""))
    return {"path": path, "rounds": rounds, "refusals": refusals,
            "tail": _tail(path)}


def parse_kv(rest):
    """`k=v` pairs, values allowed to contain spaces up to the next ` k=`.

    The same reader `fast_speed_table.py` uses, deliberately: two parsers
    that disagree about what an arm said would be a third way for a board to
    be wrong about itself."""
    out = {}
    for m in re.finditer(r"(\w+)=(.*?)(?=\s+\w+=|$)", rest.strip()):
        out[m.group(1)] = m.group(2).strip()
    return out


def classify_refusal(kv):
    """One FSPEED-REFUSED line -> (class, reason). See the module docstring's
    "A refusal by arm=ours is not a refusal"."""
    arm = kv.get("arm", "?")
    reason = kv.get("reason", "")
    lane = kv.get("lane", "?")
    where = "%s/%s" % (lane, arm)

    for pattern, cls, why in SIGNATURES:
        if re.search(pattern, reason, re.M):
            # An OPPONENT hitting a box signature is the box; OURS hitting
            # one is still ours to explain, so only the opponent gets to be
            # infra here.
            if arm != "ours" and CLASSES[cls][0] == "infra":
                return (cls, "%s: %s" % (where, why))
            break

    if arm == "ours":
        if OURS_BY_DESIGN.search(reason):
            return ("REFUSED_POLICY",
                    "%s: refused by name, which is the port working: %s"
                    % (where, reason[:200]))
        if OURS_BROKE.search(reason):
            return ("OURS_RAISED",
                    "%s: OUR ARM DID NOT RUN: %s" % (where, reason[:240]))
        return ("OURS_RAISED",
                "%s: our arm refused for a reason this file does not"
                " recognise as designed, which is ours until read: %s"
                % (where, reason[:200]))

    if POLICY_REFUSAL.search(reason):
        return ("REFUSED_POLICY", "%s: %s" % (where, reason[:200]))
    return ("REFUSED_PEER", "%s: %s" % (where, reason[:200]))


def classify(arm, rc, facts, build_rc):
    """One arm, one class, one reason. See the module docstring for the rule
    that an unrecognised failure is OURS.

    `rc` is None when the runner never reached this arm at all."""
    if build_rc is not None and build_rc != 0:
        return ("BUILD_FAILED",
                "build_exit=%d for the binary this arm runs" % build_rc)

    if facts is None:
        if rc is None:
            return ("NOT_RUN",
                    "no log and no arm_exit line: the runner skipped it,"
                    " which `builtok` does silently when a build failed")
        return ("CRASHED",
                "arm_exit=%d and no log file, so the process died before it"
                " could open one" % rc)

    tail = facts["tail"]
    rounds = facts["rounds"]
    refusals = facts["refusals"]

    # A signature in the log beats an exit code, because the exit code of a
    # killed process says how it died and the log says why.
    hit = None
    for pattern, cls, reason in SIGNATURES:
        if re.search(pattern, tail, re.M):
            hit = (cls, reason)
            break

    if hit is not None:
        cls, reason = hit
        if cls in ("TIMEOUT", "OOM_KILLED") and rounds > 0:
            return ("PARTIAL_BUDGET",
                    "%s after %d timed round(s)" % (reason, rounds))
        return (cls, reason)

    # No signature. Now the exit code decides, and the unclassified branch
    # is deliberately the loud one.
    if rc is None:
        # TIMED ROUNDS ARE POSITIVE EVIDENCE THE ARM RAN, and they outrank a
        # missing bookkeeping line. Letting `rc is None` win here reported
        # twelve arms with five rounds each as never-run, which is a lie in
        # the opposite direction from the one this file exists to stop, and
        # a status tool that cries wolf gets skipped like any other.
        if rounds > 0:
            return ("OK", "%d timed round(s), though leg.txt carries no"
                          " arm_exit line for it" % rounds)
        if refusals:
            return ("REFUSED_PEER",
                    "%d refusal(s) reported below, one row each; leg.txt"
                    " carries no arm_exit line" % len(refusals))
        return ("NOT_RUN", "no timed round, no refusal and no arm_exit line:"
                           " the runner skipped it, which `builtok` does"
                           " silently when a build failed")
    if rc != 0:
        if rounds > 0:
            return ("PARTIAL_BUDGET",
                    "arm_exit=%d after %d timed round(s), cause not named in"
                    " the log" % (rc, rounds))
        return ("CRASHED",
                "arm_exit=%d with no signature this file recognises. UNTIL"
                " SOMEONE READS THE LOG THIS IS OURS, not the box." % rc)

    if rounds > 0:
        return ("OK", "%d timed round(s)" % rounds)
    if refusals:
        # The process itself is fine; every refusal inside it gets its own
        # row from `survey`, which is where the arm=ours case is caught.
        return ("REFUSED_PEER",
                "no timed round in this process; %d refusal(s) reported"
                " below, one row each" % len(refusals))
    return ("NO_OUTPUT",
            "exit 0, zero FSPEED lines, zero refusals. An arm that neither"
            " measured nor refused has not reported anything, and a board"
            " built from it is showing an absence as a coverage gap.")


def survey(run_dir):
    arm_exit, build_exit, expected, has_expect = _read_leg(run_dir)

    # Index every log under the run dir by basename, at any depth. See
    # `_log_facts` for why this is a walk and not a join.
    index = {}
    for dirpath, _, names in os.walk(run_dir):
        for n in names:
            if n.endswith(".log") and not RUNNER_OWNED.match(n):
                index.setdefault(n, os.path.join(dirpath, n))

    # The union of what was expected, what exited, and what left a log. Any
    # one of the three alone has a blind spot; the union has none.
    #
    # A log the runner did not record is only an ARM if it carries an
    # `FSPEED-HEADER`. Without that test `console.log` -- which holds the
    # last three lines of every arm, so it matches on FSPEED alone -- and the
    # per-binary build logs all became arms, and a clean Apple leg reported
    # one spurious defect. An arm declares itself with a header; a transcript
    # does not.
    names = set(arm_exit) | set(expected)
    for n, path in index.items():
        if n in names:
            continue
        if HAS_HEADER.search(_tail(path, 400000)):
            names.add(n)

    rows = []
    for arm in sorted(names):
        facts = _log_facts(index, arm)
        build_rc = None
        build_name = expected.get(arm)
        if build_name:
            build_rc = build_exit.get(build_name)
        cls, reason = classify(arm, arm_exit.get(arm), facts, build_rc)
        rows.append({
            "arm": arm,
            "class": cls,
            "severity": CLASSES[cls][0],
            "reason": reason,
            "exit": arm_exit.get(arm),
            "rounds": facts["rounds"] if facts else 0,
        })
        # Every refusal inside the process is its own row. One log can hold
        # a policy refusal, a box failure and our own arm dying.
        for kv in (facts["refusals"] if facts else []):
            rcls, rreason = classify_refusal(kv)
            rows.append({
                "arm": "%s :: %s/%s" % (arm, kv.get("lane", "?"),
                                        kv.get("arm", "?")),
                "class": rcls,
                "severity": CLASSES[rcls][0],
                "reason": rreason,
                "exit": arm_exit.get(arm),
                "rounds": 0,
            })

    if not rows:
        rows.append({
            "arm": "(none found)", "class": "EMPTY_LEG",
            "severity": CLASSES["EMPTY_LEG"][0],
            "reason": "no leg.txt arm_exit lines and no *.log under %s"
                      % os.path.abspath(run_dir),
            "exit": None, "rounds": 0,
        })

    # Builds are reported in their own right. A failed build with no arm
    # attached to it is still the reason a lane is missing from the board.
    builds = [{"name": k, "exit": v,
               "severity": "ok" if v == 0 else "defect"}
              for k, v in sorted(build_exit.items())]

    worst = "ok"
    for r in rows:
        if SEVERITY_RANK[r["severity"]] > SEVERITY_RANK[worst]:
            worst = r["severity"]
    for b in builds:
        if SEVERITY_RANK[b["severity"]] > SEVERITY_RANK[worst]:
            worst = b["severity"]

    return {"run_dir": os.path.abspath(run_dir), "rows": rows,
            "builds": builds, "worst": worst,
            "expectation_list": has_expect}


def render(rep):
    out = []
    w = out.append
    counts = {}
    for r in rep["rows"]:
        counts[r["class"]] = counts.get(r["class"], 0) + 1

    w("## Leg status: did every arm actually run")
    w("")
    w("Read this BEFORE the ratio tables. Every arm the leg was supposed to")
    w("run appears here in exactly one class, and an arm that produced no")
    w("`FSPEED` line is separated into the reason it produced none. A board")
    w("whose status is not `ok` is not evidence of coverage.")
    w("")
    if not rep["expectation_list"]:
        w("**This leg wrote no `expect_arm` lines**, so the arms below are")
        w("only those that left a log or an exit code. An arm the runner")
        w("skipped entirely cannot appear. Legs recorded from 2026-08-30")
        w("carry the expectation list and do not have this hole.")
        w("")
    w("verdict: **%s**" % rep["worst"].upper())
    w("")
    w("| class | n | severity | what it means |")
    w("|---|---|---|---|")
    for cls in REPORT_ORDER:
        if cls in counts:
            sev, meaning = CLASSES[cls]
            w("| `%s` | %d | %s | %s |" % (cls, counts[cls], sev, meaning))
    w("")
    non_ok = [r for r in rep["rows"] if r["severity"] != "ok"]
    if non_ok:
        w("### Every arm that is not `ok`, with the reason it is not")
        w("")
        w("| arm | class | severity | exit | rounds | reason |")
        w("|---|---|---|---|---|---|")
        for r in non_ok:
            w("| `%s` | %s | %s | %s | %d | %s |"
              % (r["arm"], r["class"], r["severity"],
                 "-" if r["exit"] is None else r["exit"], r["rounds"],
                 r["reason"].replace("|", "/")))
        w("")
    bad_builds = [b for b in rep["builds"] if b["exit"] != 0]
    if bad_builds:
        w("### Builds that failed")
        w("")
        w("A build failure removes a lane from the board without failing a")
        w("row on it. This is the 2026-08-25 gemm_nt_gram case.")
        w("")
        w("| binary | build_exit |")
        w("|---|---|")
        for b in bad_builds:
            w("| `%s` | %d |" % (b["name"], b["exit"]))
        w("")
    return "\n".join(out) + "\n"


#: Lines a HEALTHY arm prints. No signature may match any of them.
#:
#: This exists because one did. The payment signature carried a bare `402 `
#: and matched `ms=1332.402` in a timing line, so a clean Apple arm with
#: eighteen timed rounds was reported as a billing failure. A classifier that
#: can invent an infra excuse out of a benign number is worse than no
#: classifier, so the benign corpus is checked on every run: it costs
#: microseconds and it is the only thing standing between this table and the
#: failure mode its own docstring warns about.
BENIGN = [
    "FSPEED lane=transformer arm=ours shape=llama8b.prefill.t512 round=3"
    " ms=1332.402 hash=0x4021abcd",
    "FSPEED-HEADER family=classical lane=kmeans arm=ours mode=FAST"
    " device=Apple_M4 rounds=3 size=shipped",
    "FSPEED-ACC lane=rf arm=ours metric=logloss value=0.6224",
    "FSPEED-WARMUP lane=gemm arm=ours shape=gram.32x32x1M ms=137.0",
    "FSPEED lane=knn arm=cuml shape=400000x32 round=1 ms=9.555 hash=0x137",
    "FSPEED-NOTE lane=ols arm=ours devices=gpu (auto)",
    "FSPEED lane=ivf arm=ours shape=512x8 round=2 ms=7.143 hash=0x9137",
]


def self_test():
    """Assert no signature fires on a healthy line. Raises on failure."""
    bad = []
    for line in BENIGN:
        for pattern, cls, why in SIGNATURES:
            if re.search(pattern, line, re.M):
                bad.append((cls, pattern, line))
    for line in BENIGN:
        if OURS_BROKE.search(line):
            bad.append(("OURS_BROKE", OURS_BROKE.pattern, line))
    if bad:
        msg = "\n".join(
            "  %s matched /%s/ on a HEALTHY line: %s" % b for b in bad)
        raise AssertionError(
            "leg_status: a failure signature fires on ordinary arm output."
            " Tighten it to a phrase.\n" + msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--out", help="write the markdown section here")
    ap.add_argument("--json", dest="json_out", help="write the report as JSON")
    ap.add_argument("--quiet", action="store_true",
                    help="exit code only, print nothing")
    args = ap.parse_args()

    self_test()

    if not os.path.isdir(args.run_dir):
        print("leg_status: no such run dir: %s" % args.run_dir,
              file=sys.stderr)
        return 1

    rep = survey(args.run_dir)
    text = render(rep)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rep, fh, indent=2, sort_keys=True)
    if not args.quiet and not args.out:
        sys.stdout.write(text)
    elif not args.quiet:
        print("wrote %s -- verdict %s" % (args.out, rep["worst"].upper()))
    return SEVERITY_EXIT[rep["worst"]]


if __name__ == "__main__":
    raise SystemExit(main())
