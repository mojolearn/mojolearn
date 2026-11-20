#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/paper_compare.py -- rerun ONLY the measurements of the mlsys paper on a newer published release and
compare (Andrew, 2026-11-08: "rerun JUST the measurements we have in the mlsys on our 0.8.66 to see how we compare
and if we have corrected regressions").

The paper's boards (0.8.55, NVIDIA L40S and AMD MI300X; the merged per-vendor boards under
~/mojolearn-evidence/mlsys27-evidence/measurements/paper-tables/, READ ONLY: that repo is under review and is never
written) hold every race the paper reports: the classical and tree lanes with a GPU-library opponent and the neural
board. This tool:

  recipes  --vendor V [--board PAPER_BOARD] [--out JSON]
        extracts the exact recipe set of one vendor's paper board: one recipe per OUR arm cell (fit, and each
        inference batch), with its race id, family, lane, dataset, shape, numeric mode, the explicit parameters the
        board read back (the paper copy is anonymized; the library name is restored), and the 0.8.55 numbers
        (median_ms, the kernel span where the board derived one, quality, status) plus every paired opponent's
        stored time and ratio, exactly as the paper's audit recorded them. Opponents are NEVER rerun (Andrew: opponents
        are scored once); the compare reuses these numbers.
        Default out: bench/results/paper_compare/recipes-<vendor>.json.
  ids      --recipes JSON --planner tools/bench_board.py --out-dir DIR
        ON THE BOX: the recipe races the current board plans (bench_board.plan_races with the recipe file's
        bench_board_args), written as race-ids-full.txt and race-ids-smoke.txt (the smoke's rows and shape), the
        board arguments as board-args.txt (one per line) and the races the board no longer plans as
        not-planned.json. A race the board does not plan is recorded, never a refusal of the whole run.
  ids-all  --vendor V --planner tools/bench_board.py --out-dir DIR
        ON THE BOX, the full-board screen: every race the board plans on V (every family and lane, identical, our
        arms), in the same files as ids plus races.json (family/lane/dataset per race, for measure_owed data-keys).
  pass-ids --full-ids FILE --smoke-json SMOKE.json... --out FILE
        the full-size ids of the planned races whose smoke passed (smoke-failed.json beside --out lists the rest;
        each failed race and its reason is printed as a PAPER-PASS-IDS-FAILED line).
  left     --ids FILE --board BOARD.json
        ON THE BOX: how many of the ids are not yet done in the board (the disk gate of a resumed board).
  compare  --recipes JSON --board BOARD.json... [--prior-board VERSION=BOARD.json ...] [--reference-compare JSON]
           [--stored-baseline-board BOARD.json] [--verdict-prior NAME] [--best-ever [FILE]] [--out-md MD]
           [--out-json JSON] [--summary TXT]
        the new board against OUR prior wheels re-timed in the same job on the same box, per recipe: ms of each,
        ratios, quality, the flags below, and the stored columns (the paper record, a stored compare, the stored
        opponents, the best ever) as reference only. Prints a one-screen summary with both arms' host probes.
        A row that is not a race (tools/race_fairness.py lane_class: the call-overhead internal check
        algos/internal-call-overhead-lr-at, Andrew 2026-11-17) is never a recipe or a best-ever row: it has its own
        "internal checks" section, new against our prior versions re-timed on this box only.

SAME BOX ONLY (AGENTS.md "Times are compared only on the same box", Andrew, 2026-11-14; the 2026-11-14 compare of
80a7f8aca3 on L40S host 9204fff9f8f7 against stored numbers from host 1e08d01c3d48 raised 56 false REGRESSION flags
on code that had not changed). Every board carries the box's host probe (tools/ops/host_probe.py through
bench_board --host-probe: box.host_probe and each race's host.host_id). A time flag is derived only from two cells
whose boards carry the same host id. A --prior-board from another host REFUSES the compare (exit non-zero); a board
without a probe gives no verdict (NOT-COMPARED). SAME SYSTEM SOFTWARE TOO (2026-11-17): the GPU uuid, GPU driver,
kernel release and boot id (SOFTWARE_FIELDS) must each be one value on both sides and the same; otherwise (a pod
restart or a driver update between the arms, or a field not recorded) every row is NOT-COMPARED and the summary's
first line says why. The stored paper record (old_ms in the recipes: ref_ms here), a
stored compare JSON (--reference-compare: ref_prior_ms), the paper's opponent cells and the best-ever file are
shown with their host and never make a flag.
STORED BASELINE (Andrew, 2026-11-17, AGENTS.md part 8: "0.8.55 is not timed again"): --stored-baseline-board takes
the 2026-11-17 L40S board of 0.8.55 as the baseline of a job that times our new wheel only. The verdict flags are
derived from it; the summary's and the report's first line says the baseline is stored and names its host, GPU,
driver, kernel and date beside this job's; a different host or driver is a WARNING line, not the refusal above (which
stays for the two-arm modes). BOX-STATE: a row whose in-race upload probe (the harness's separate upload,
upload_bytes_separate / upload_ms_separate of the race, else the median of its dataset) differs from the stored run's
by more than 15% has its REGRESSION / FASTER / SAME replaced by BOX-STATE: the ratio is shown, not counted.
VERDICT PRIOR (lane fx-same-pod-prior; Andrew, 2026-11-19: a speed verdict compares two wheels timed on the SAME host in
the SAME job; the stored 0.8.55 board is a screen only): --verdict-prior NAME makes the --prior-board NAME (one of OUR
OWN EARLIER BUILT WHEELS timed beside the candidate in this job, paper_compare_job.sh PC_PRIOR_WHEELS) the baseline:
REGRESSION / FASTER / SAME / FIXED / BROKEN come from that pair only, under the same-box refusal above, and a
--stored-baseline-board is then a screen column (ratio shown, no flag, no BOX-STATE).

FLAGS (per recipe; a recipe can carry several)
  REGRESSION     new / baseline median_ms > 1.10 (slower)     FASTER   new / baseline < 0.90
  SAME           0.90 <= ratio <= 1.10
                 (baseline = the --prior-board of the paper's version, 0.8.55, re-timed on this box)
  FIXED          the baseline cell has no time (refused, failed) and the new board has one
  BROKEN         the baseline has a time and the new board has none (its status is shown)
  STILL-FAILING  neither has a time
  BASELINE-NOT-RUN  the baseline board has no cell for the race (its smoke failed, it does not plan it, the job did
                 not reach it): NOT-RUN for that column, never a regression
  NOT-COMPARED   no on-box baseline (no --prior-board of the paper's version, no host probe on either side, or the
                 arms ran under different system software: driver, kernel or boot id)
  NOT-RUN        no new cell: the board no longer plans the race, its smoke failed, or the job did not reach it
  QUALITY-DROP   tools/board_quality.compare(new, paper record) FAILs on a one-sided score or a reference bound:
                 worse than 0.8.55 beyond the board's declared margin (the quality-parity policy)
  QUALITY-CHANGE the same policy FAILs on a symmetric metric (no better side)
  QUALITY-MISSING the 0.8.55 record holds a judged quality metric the new cell lacks (quality_missing lists them,
                 quality_unavailable the cell's own reason); the row cannot be judged on it. Shown, never a gate
  NOW-WINS / NOW-LOSES  vs the paper's fastest stored opponent: only when the stored opponents ran on this box's
                 host (never, for the paper's boards; the ratio is shown)
  PARAMS-DIFF    the explicit parameters our arm read back differ from the paper's (a float that rounds to the same
                 float32 is the same setting: the arms run FP32)
  HW-DIFF        the new board's GPU is not the paper's GPU
  HARNESS-CLOCK  the new cell and its baseline cell were timed under different harness-clock revisions (the cell's
                 harness_clock; lane fx-harness 2026-11-18: train-test-split, resample, the model_selection and a few
                 more lanes stopped copying, casting and reducing their outputs inside the clock, so a stored 0.8.55
                 cell holds harness work the new one does not). FASTER and SAME are replaced by it (shown, never
                 counted as a gain); a REGRESSION stays (less harness work and still slower is real)
  HOST-STALL     the new cell's scored calls spent more than half their wall in system time (host_usage, getrusage:
                 page faults through direct reclaim or THP compaction, lane fx-harness 2026-11-18); a REGRESSION is
                 replaced by it (shown, not counted); the worker's NUMA node is shown beside it
  NUMA-MIXED     the new cell's worker and its baseline cell's worker ran on different NUMA nodes and neither was
                 bound (worker_numa.node_now, placement node none; lane fx-cagra-taxi 2026-11-19: algos/cagra fit is
                 5-7 s slower on the node that is not the GPU's, on byte-identical code, and a same-pod pair job read
                 one such cell as a 2.54x regression). REGRESSION and FASTER are replaced by it (shown, not counted)
  SCORING-MIXED  the new time and an on-box column it is divided by were taken under different small-row rules
                 (tools/bench_board_small_rows.py: one is the median of 20 calls, the other one call, because their
                 first scored calls fell on either side of 10 ms); scoring_mixed names the columns and, when both
                 cells recorded per-call times, the two FIRST scored calls (the same quantity on both arms). The
                 ratio is kept and shown; REGRESSION / FASTER of the baseline column and REGRESSION-VS-PRIOR /
                 FASTER-VS-PRIOR of the prior column are replaced by it (scoring_mixed_replaced; shown, not
                 counted): a median of 20 calls against one call is biased, because the first scored call carries
                 the first-touch page faults (lane fx-metrics-ranking 2026-11-19: metric-coverage-error istella,
                 byte-identical path and hash, prior median of 20 calls 9.83 ms with calls of 8.06..15.97 ms, new
                 one call 16.27 ms, read as a 1.65x regression)
With more --prior-board versions (the default job times one, 0.8.55: Andrew, 2026-11-16; an explicit extra prior or
last release gets one each), each other version gets a column (ms, new / prior); the last one is the prior release:
  REGRESSION-VS-PRIOR / FASTER-VS-PRIOR   new / prior median ms above 1.10 / below 0.90
  RECOVERED      the prior was > 1.10 x the on-box baseline and the new board is SAME or FASTER vs that baseline
  STILL-REGRESSED the prior was > 1.10 x the on-box baseline and the new board still is
With --best-ever [FILE] (default bench/results/board_best_ever/<vendor>.json, tools/board_best_ever.py mine): a
"best ever (NV)" / "(AMD)" reference column, our best time ever on that race key, with its date, release and host,
and new / best-ever. Its time flags need the same host id on the best-ever cell (records mined before 2026-11-14
carry none, so they are shown and never flagged) and the same system state (SOFTWARE_FIELDS; mined records carry
none yet, so these flags are shown only once the miner keeps it):
  BEST-EVER-REGRESSION  new / best ever > 1.10 with the same explicit parameters and the same output hash
  BEST-EVER-REGRESSION-UNCHECKED  the same, but the params or the hash of one side were not recorded
  BEST-EVER-SUSPECT  new / best ever > 1.10 but the best ever is below 5% of the row's median time
  PARAMS-DIFFER  the best-ever cell's explicit parameters differ from the new cell's on a shared key (not a time).
                 A different output hash alone is not this flag: bits may change between versions (AGENTS.md); it is
                 shown as "vs best ever: hash" in the note
  BEST-EVER-HW-DIFF  the best ever was measured on another device than the new board's
Every our-arm cell of the new boards outside the recipes (a full-board screen, tools/ops/paper_compare_job.sh
board-all) gets an extra row in its own table with the on-box prior columns (REGRESSION / FASTER vs the baseline,
REGRESSION-VS-PRIOR) and the best-ever reference.
A board made from candidate wheels (bench_board --candidate-wheel; box.mojolearn.wheel.source = candidate) is
labelled "candidate <sha12> (<version>)" everywhere, with its source commit and wheel sha256s, never as a release.

The new numbers come from tools/ops/paper_compare_job.sh, which runs these recipes through tools/bench_board.py
(our arms only) for the candidate and for each prior wheel from PyPI (PC_PRIOR_VERSIONS) on the same box in the same
job. Tests: tools/test_paper_compare.py, tools/test_paper_compare_same_box.py.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import re
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import board_quality as BQ  # noqa: E402
import board_best_ever as BE  # noqa: E402
import race_fairness as RF  # noqa: E402
import bench_board_small_rows as SMALL  # noqa: E402

SCHEMA = "mojolearn-paper-compare-recipes/1"
COMPARE_SCHEMA = "mojolearn-paper-compare/1"
PAPER_VERSION = "0.8.55"
PAPER_ROOT = os.path.expanduser("~/mojolearn-evidence/mlsys27-evidence/measurements/paper-tables")
OUT_DIR = os.path.join(REPO, "bench", "results", "paper_compare")
SLOWER = 1.10        # new / old above this: REGRESSION (brief, 2026-11-08)
FASTER = 0.90        # new / old below this: FASTER
VENDORS = ("nvidia", "amd")
# The paper copy is anonymized (the supplement names no library). Restore the name in what we compare.
ANON = (("ANONLIB", "MOJOLEARN"), ("Anonlib", "Mojolearn"), ("anonlib", "mojolearn"))
# Statuses with no usable time. "not-scored" keeps its time: the scorer failed, the clock did not.
NO_TIME = re.compile(r"^(REFUSED|FAILED|UNKNOWN|ERROR|TIMEOUT|SKIPPED|CRASHED)", re.I)
SMOKE_SHAPE = "small"


# ---------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------

def write_atomic(path, text):
    """Write a results file whole or not at all: a temp file beside it, fsync, then rename over the target.

    ATOMIC-WRITE FIX (2026-11-14, L40S pod nvc1): the owed report died on a truncated JSON written while the disk
    was full (JSONDecodeError line 1 col 87). A reader now sees the old file or the complete new one; a failed write
    (ENOSPC) raises here and leaves the target untouched."""
    path = os.path.abspath(str(path))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = "%s.tmp-%d" % (path, os.getpid())
    try:
        with open(tmp, "w") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def json_text(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + "\n"


def load_json(path):
    path = str(path)  # callers pass str or Path (measure_candidate stage crashed on a PosixPath, 2026-11-13)
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        return json.load(fh)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def deanon(value):
    """The value with the anonymized library name restored, recursively (keys too)."""
    if isinstance(value, str):
        for a, b in ANON:
            value = value.replace(a, b)
        return value
    if isinstance(value, list):
        return [deanon(v) for v in value]
    if isinstance(value, dict):
        return {deanon(k): deanon(v) for k, v in value.items()}
    return value


def number(value):
    return BQ.number(value)


def timed(status, ms):
    return number(ms) and not NO_TIME.match(str(status or ""))


def short_status(status):
    s = str(status if status is not None else "missing")
    return s if len(s) <= 300 else s[:297] + "..."


def recipe_key(race_id, phase, arm, batch):
    return " ".join(x for x in (race_id, phase, arm, batch or "") if x)


def scored_quality(quality):
    """Only the metrics the quality-parity policy declares (the ones the compare judges)."""
    q = quality or {}
    keep = set(BQ.ABSOLUTE) | set(BQ.RELATIVE) | set(BQ.REFERENCE) | set(BQ.AGREEMENT)
    return {k: v for k, v in q.items() if k in keep}


#: a cell's explicit reason for a quality metric it could not record (bench_board_neural, bench_board_infer)
UNAVAILABLE = "quality_unavailable"


def shown_quality(quality):
    """QUALITY-MISSING FIX (2026-11-13, candidate ba08dac006 L40S): what the compare SHOWS of a cell's quality: the
    judged metrics, every other finite number the scorer recorded (the own-host reference distances of the forward
    and decode lanes, program-run's max_abs_err_vs_fp64) and the cell's quality_unavailable reason. Before, only the
    judged metrics were kept, so those rows read new_quality={} although the board held their quality."""
    q = quality or {}
    out = {k: v for k, v in q.items() if number(v) and not isinstance(v, bool)}
    out.update(scored_quality(q))
    if q.get(UNAVAILABLE):
        out[UNAVAILABLE] = str(q[UNAVAILABLE])[:600]
    return out


def quality_missing(old_quality, new_quality):
    """The judged metrics the old record holds and the new cell does not (QUALITY-MISSING; shown, never a gate)."""
    return sorted(set(scored_quality(old_quality)) - set(scored_quality(new_quality)))


def kernel_ms(cell):
    span = ((cell.get("comparability") or {}).get("span") or {})
    v = span.get("fit_minus_upload_ms")
    return v if number(v) else None


def gpu_name(board):
    gpu = ((board.get("box") or {}).get("gpu") or {})
    return gpu.get("name")


# ---------------------------------------------------------------------------------------------------------------
# the box a time was taken on (AGENTS.md "Times are compared only on the same box", Andrew, 2026-11-14)
# ---------------------------------------------------------------------------------------------------------------

def board_probe(board):
    """The tools/ops/host_probe.py document a board recorded (bench_board --host-probe), or None."""
    p = (board.get("box") or {}).get("host_probe")
    return p if isinstance(p, dict) else None


def board_host_ids(board):
    """Every host id a board carries: its box probe's and each race's host.host_id (bench_board race_host)."""
    ids = set()
    p = board_probe(board)
    if p and p.get("host_id"):
        ids.add(p["host_id"])
    for race in (board.get("races") or {}).values():
        h = (race.get("host") or {}).get("host_id") if isinstance(race, dict) else None
        if h:
            ids.add(h)
    return ids


def host_label(board):
    """The host a stored board ran on, for a reference column: its probe's host id, else its hostname."""
    p = board_probe(board)
    if p and p.get("host_id"):
        return p["host_id"]
    return ((board.get("box") or {}).get("host") or {}).get("hostname")


def candidate_host(cand_boards):
    """(host id, None) of the candidate boards, or (None, why) without a probe. Two hosts refuse (SystemExit)."""
    cand = set().union(*[board_host_ids(b) for b in cand_boards]) if cand_boards else set()
    if len(cand) > 1:
        raise SystemExit("paper_compare: REFUSED: the candidate boards ran on %d hosts (%s); times are compared only "
                         "on the same box" % (len(cand), ", ".join(sorted(cand))))
    return (next(iter(cand)), None) if cand else (None, "the candidate board carries no host probe")


def same_box(cand_boards, other_boards, what):
    """(host id, None) when the candidate boards and OTHER_BOARDS carry one host id; (None, why) when either side
    has none (no probe: the column is shown and makes no flag). Two host ids refuse (SystemExit): a compare across
    hosts is never a verdict (2026-11-14: 56 false REGRESSION flags from two L40S hosts)."""
    host, why = candidate_host(cand_boards)
    other = set().union(*[board_host_ids(b) for b in other_boards]) if other_boards else set()
    if host and other and other != {host}:
        raise SystemExit("paper_compare: REFUSED: %s ran on host %s, the candidate on %s; times are compared only on "
                         "the same box (AGENTS.md)" % (what, ", ".join(sorted(other)), host))
    if not host:
        return None, why
    if not other:
        return None, "%s carries no host probe" % what
    return host, None


# SAME SYSTEM SOFTWARE (fb-harness, 2026-11-17): the full-board compare of d95f805cce gave verdicts for two arms on the
# same host id and GPU whose pod had restarted between them (candidate on NVIDIA driver 595.91.07 and kernel
# 6.8.0-147, 0.8.55 on 595.112.07 and 6.8.0-149). A verdict now needs the GPU uuid, the GPU driver, the kernel release
# and the boot id (/proc/sys/kernel/random/boot_id) to be one value on each side and the same on both; a field that
# is not recorded is no proof, so it gives no verdict either.
SOFTWARE_FIELDS = tuple(zip(BE.SOFTWARE_FIELDS, ("GPU uuid", "GPU driver", "kernel", "boot id")))


def board_software(board):
    """{(gpu uuid, driver, kernel, boot id): [race ids]} of one board: every distinct system state its races ran
    under (a race's host record wins over the board default field by field). A board with no race gives its
    default alone."""
    base = BE.software_default(board)
    out = {}
    for rid, race in (board.get("races") or {}).items():
        out.setdefault(tuple(BE.race_software(race, base)), []).append(rid)
    if not out:
        out[tuple(base[f] for f, _ in SOFTWARE_FIELDS)] = []
    return out


def software_mismatch(cand_boards, other_board, what):
    """None when the candidate boards and OTHER_BOARD ran under one and the same system state (SOFTWARE_FIELDS), else
    why not, in words: the first line of the summary."""
    cand = {}
    for b in cand_boards:
        for k, v in board_software(b).items():
            cand.setdefault(k, []).extend(v)
    other = board_software(other_board)
    why = []
    for side, states in (("the candidate board", cand), (what, other)):
        if len(states) > 1:
            parts = []
            for i, (f, label) in enumerate(SOFTWARE_FIELDS):
                vals = sorted({str(k[i]) for k in states})
                if len(vals) > 1:
                    parts.append("%s %s" % (label, " / ".join(vals)))
            why.append("%s spans %d system states (a restart or an update between its races: %s)"
                       % (side, len(states), "; ".join(parts)))
    if len(cand) == 1 and len(other) == 1:
        c, o = next(iter(cand)), next(iter(other))
        for i, (f, label) in enumerate(SOFTWARE_FIELDS):
            if c[i] and o[i] and c[i] != o[i]:
                why.append("%s %s (candidate) vs %s (%s)" % (label, c[i], o[i], what))
    missing = []
    for side, states in (("the candidate board", cand), (what, other)):
        for i, (f, label) in enumerate(SOFTWARE_FIELDS):
            if any(not k[i] for k in states):
                missing.append("%s not recorded on %s" % (label, side))
    if missing:
        why.append("; ".join(missing) + " (a board made before 2026-11-17 has no boot id)")
    if not why:
        return None
    return "the arms did not run under the same system software: " + "; ".join(why)


def probe_line(probe):
    """One line of a host probe for the summary (both arms' probes are printed beside the verdict)."""
    if not probe:
        return "no host probe"
    pcie = probe.get("pcie") or {}
    link = ("gen%s x%s%s" % (pcie.get("gen"), pcie.get("width"), "" if pcie.get("read_during_copy") else " (idle)")
            if pcie.get("gen") is not None else "pcie ?")
    g = lambda k: "-" if probe.get(k) is None else probe.get(k)  # noqa: E731
    return ("host %s; cpu %s; numa %s; cores %s (cgroup %s); memcpy %s GB/s; h2d pageable %s / pinned %s GB/s; %s; "
            "gpu %s driver %s; kernel %s; boot %s" % (
                g("host_id"), g("cpu_model"), g("numa_nodes"), g("affinity_cores") if probe.get(
                    "affinity_cores") is not None else g("nproc"), g("cgroup_cores"), g("memcpy_gbps"),
                g("h2d_pageable_gbps"), g("h2d_pinned_gbps"), link, (probe.get("gpu") or {}).get("name"),
                (probe.get("gpu") or {}).get("driver"), g("kernel"), g("boot_id")))


def parse_prior_board(text):
    """'<version>=<board.json>' (compare --prior-board) -> (version, path)."""
    v, sep, path = text.partition("=")
    if not sep or not v or not path:
        raise SystemExit("paper_compare: --prior-board wants <version>=<board.json>, not %r" % text)
    return v, path


# ---------------------------------------------------------------------------------------------------------------
# recipes
# ---------------------------------------------------------------------------------------------------------------

def _opponents_for(race, phase, mode, arm, batch):
    """The paper's stored opponent timings paired with this arm and batch (the audit's preferred clocks)."""
    audit = (race.get("audit") or {}).get("%s/%s" % (mode, phase)) or {}
    cells = race.get("cells" if phase == "fit" else "infer_cells") or []
    out = []
    for opp, od in sorted((audit.get("opponents") or {}).items()):
        if (od.get("ours_arm") or "ours") != arm:
            continue
        tcs = od.get("timing_comparisons") or ([od["timing_comparison"]] if od.get("timing_comparison") else [])
        for tc in tcs:
            if (tc.get("batch") or None) != (batch or None):
                continue
            ocell = next((c for c in cells if c.get("arm") == opp and (c.get("batch") or None) == (batch or None)),
                         {})
            out.append({
                "arm": opp,
                "opponent_ms": tc.get("opponent_ms") if number(tc.get("opponent_ms")) else None,
                "ours_ms": tc.get("ours_ms") if number(tc.get("ours_ms")) else None,
                "clock": tc.get("clock"),
                "label": tc.get("label"),
                "ratio": tc.get("value") if number(tc.get("value")) else None,
                "verdict": od.get("verdict"),
                "opponent_status": short_status(ocell.get("status")) if ocell else None,
            })
    return out


def extract_recipes(board, vendor, source_path=None, version=PAPER_VERSION):
    """The recipe document of one paper board (see the module docstring)."""
    races = board.get("races") or {}
    recipes, race_ids = [], []
    families, lanes, datasets = [], [], []
    for rid in sorted(races):
        race = races[rid]
        ours = race.get("our_arms") or {}
        if not ours:
            continue
        race_ids.append(rid)
        fam, lane, ds = race.get("family"), race.get("lane"), race.get("dataset")
        for lst, v in ((families, fam), (lanes, lane)):
            if v not in lst:
                lst.append(v)
        if fam != "neural" and ds not in datasets:
            datasets.append(ds)
        for phase, cells in (("fit", race.get("cells") or []), ("infer", race.get("infer_cells") or [])):
            for cell in cells:
                arm = cell.get("arm")
                if arm not in ours:
                    continue
                batch = cell.get("batch")
                mode = ours[arm]
                params = deanon((cell.get("params") or {}).get("params"))
                impl = deanon((cell.get("params") or {}).get("implementation"))
                recipes.append({
                    "key": recipe_key(rid, phase, arm, batch),
                    "race_id": rid, "family": fam, "lane": lane, "dataset": ds,
                    "rows": race.get("rows"), "shape": cell.get("shape"),
                    "neural_shape": cell.get("neural_shape"),
                    "phase": phase, "arm": arm, "mode": mode,
                    "batch": batch, "batch_rows": cell.get("batch_rows"),
                    "implementation": impl, "params": params,
                    "call": deanon(cell.get("call")),
                    "params_check": race.get("params_check"),
                    "v0855": {
                        "status": short_status(cell.get("status")),
                        "median_ms": cell.get("median_ms") if number(cell.get("median_ms")) else None,
                        "times_ms": cell.get("times_ms"),
                        "kernel_ms": kernel_ms(cell) if phase == "fit" else None,
                        "quality": cell.get("quality") or {},
                        "hash": cell.get("hash"),
                        "device_name": cell.get("device_name"),
                        "library_version": cell.get("library_version"),
                    },
                    "opponents": _opponents_for(race, phase, mode, arm, batch),
                })
    order = ("trees", "classical", "classical2", "neural", "algos")
    families.sort(key=lambda f: order.index(f) if f in order else len(order))
    doc = {
        "schema": SCHEMA,
        "vendor": vendor,
        "paper_version": version,
        "source": {
            "board": source_path,
            "sha256": sha256_file(source_path) if source_path and os.path.isfile(source_path) else None,
            "gpu": gpu_name(board),
            # the stored column's host (reference only: never a flag against a time from another host)
            "host": host_label(board),
            "created": board.get("created"), "updated": board.get("updated"),
            "note": "the paper's merged board (anonymized copy; library name restored in params and calls)",
        },
        # what tools/ops/paper_compare_job.sh passes to tools/bench_board.py: our arms only (no opponent is
        # raced: the default of the board without --with-opponents), IDENTICAL, one round, GPU opponents only
        # (keeps exactly the own pairing arms of GPU opponents, which is what the paper raced)
        "bench_board_args": {
            "vendor": vendor, "families": families, "lanes": lanes, "datasets": datasets,
            "modes": "identical", "rounds": 1, "gpu_opponents_only": True,
        },
        "thresholds": {"regression_ratio_above": SLOWER, "faster_ratio_below": FASTER,
                       "quality_policy": BQ.POLICY},
        "race_ids": race_ids,
        "counts": {"races": len(race_ids), "recipes": len(recipes),
                   "fit": sum(r["phase"] == "fit" for r in recipes),
                   "infer": sum(r["phase"] == "infer" for r in recipes),
                   "by_family": {f: sum(r["family"] == f for r in recipes) for f in families}},
        "recipes": recipes,
    }
    return doc


def cmd_recipes(args):
    board_path = args.board or os.path.join(PAPER_ROOT, "board-%s.json.gz" % args.vendor)
    doc = extract_recipes(load_json(board_path), args.vendor, board_path, args.version)
    out = args.out or os.path.join(OUT_DIR, "recipes-%s.json" % args.vendor)
    write_atomic(out, json_text(doc))
    c = doc["counts"]
    print("PAPER-RECIPES vendor=%s races=%d recipes=%d fit=%d infer=%d gpu=%s out=%s"
          % (args.vendor, c["races"], c["recipes"], c["fit"], c["infer"], doc["source"]["gpu"], out))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# ids (on the box)
# ---------------------------------------------------------------------------------------------------------------

def load_planner(path):
    spec = importlib.util.spec_from_file_location("paper_compare_planner", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


def board_args(bargs):
    """tools/bench_board.py arguments for the recipe set, one list."""
    out = ["--vendor", bargs["vendor"], "--families", ",".join(bargs["families"]),
           "--lanes", ",".join(bargs["lanes"]), "--modes", bargs.get("modes") or "identical"]
    if bargs.get("datasets"):
        out += ["--datasets", ",".join(bargs["datasets"])]
    if bargs.get("gpu_opponents_only"):
        out.append("--gpu-opponents-only")
    return out


def smoke_id(rid, smoke_rows):
    rid = re.sub(r"/rows=full$", "/rows=%d" % int(smoke_rows), rid)
    return re.sub(r"/shape=full$", "/shape=" + SMOKE_SHAPE, rid)


def plan_ids(doc, planner):
    """(full ids planned, smoke ids planned, {race id: why not planned})."""
    b = doc["bench_board_args"]
    vendor = b["vendor"]
    modes = planner.modes_for(vendor, b.get("modes"))
    families = tuple(b["families"])
    lanes = tuple(b["lanes"])
    datasets = tuple(b.get("datasets") or ())
    known = set()
    for f in families:
        known |= set(planner.family_lanes(f))
    lanes_ok = tuple(l for l in lanes if l in known)

    def plan(rows, shape):
        races = planner.plan_races(vendor, modes, families, lanes_ok, datasets or planner.DATASETS, rows, shape)
        if b.get("gpu_opponents_only"):
            races = planner.gpu_opponents_only(races, vendor)
        return {r["id"]: r for r in races}

    full = plan(None, "full")
    smoke_rows = getattr(planner, "SMOKE_ROWS", 2000)
    smoke = plan(int(smoke_rows), SMOKE_SHAPE)
    wanted_arms = {}
    for r in doc["recipes"]:
        wanted_arms.setdefault(r["race_id"], set()).add(r["arm"])
    planned_full, planned_smoke, missing = [], [], {}
    for rid in doc["race_ids"]:
        race = full.get(rid)
        if race is None:
            lane = rid.split("/")[1] if "/" in rid else rid
            missing[rid] = ("lane %s is no longer a board lane" % lane if lane not in known
                            else "the board no longer plans this race on %s" % vendor)
            continue
        sid = smoke_id(rid, smoke_rows)
        if sid not in smoke:
            missing[rid] = "the board plans no smoke race %s" % sid
            continue
        lost = sorted(wanted_arms.get(rid, set()) - set(race.get("our_arms") or {}))
        if lost:
            # recorded per arm by the compare (NOT-RUN); the race still runs for the arms it plans
            missing[rid + " arms"] = "our arm(s) %s not planned" % ",".join(lost)
        planned_full.append(rid)
        planned_smoke.append(sid)
    return planned_full, planned_smoke, missing


def cmd_ids(args):
    doc = load_json(args.recipes)
    planner = load_planner(args.planner)
    full, smoke, missing = plan_ids(doc, planner)
    os.makedirs(args.out_dir, exist_ok=True)
    write_atomic(os.path.join(args.out_dir, "race-ids-full.txt"), "".join(x + "\n" for x in full))
    write_atomic(os.path.join(args.out_dir, "race-ids-smoke.txt"), "".join(x + "\n" for x in smoke))
    write_atomic(os.path.join(args.out_dir, "board-args.txt"),
                 "".join(x + "\n" for x in board_args(doc["bench_board_args"])))
    write_atomic(os.path.join(args.out_dir, "not-planned.json"), json_text(missing))
    print("PAPER-IDS vendor=%s recipe_races=%d planned=%d not_planned=%d"
          % (doc["vendor"], len(doc["race_ids"]), len(full), len([k for k in missing if not k.endswith(" arms")])))
    return 0


def plan_all(planner, vendor):
    """The full-board screen (tools/ops/paper_compare_job.sh board-all): (full ids, smoke ids, {id: why not},
    board args, races) of EVERY race the board plans on the vendor (every family and lane, the board's datasets,
    identical, our arms; no --gpu-opponents-only: every own configuration is screened)."""
    modes = planner.modes_for(vendor, "identical")
    families = tuple(planner.FAMILIES)

    def plan(rows, shape):
        return {r["id"]: r for r in planner.plan_races(vendor, modes, families, None, planner.DATASETS, rows, shape)}

    full = plan(None, "full")
    smoke_rows = int(getattr(planner, "SMOKE_ROWS", 2000))
    smoke = plan(smoke_rows, SMOKE_SHAPE)
    full_ids, smoke_ids, missing = [], [], {}
    for rid in sorted(full):
        sid = smoke_id(rid, smoke_rows)
        if sid not in smoke:
            missing[rid] = "the board plans no smoke race %s" % sid
            continue
        full_ids.append(rid)
        smoke_ids.append(sid)
    bargs = ["--vendor", vendor, "--families", ",".join(families), "--modes", "identical",
             "--datasets", ",".join(planner.DATASETS)]
    races = [{"race_id": rid, "family": full[rid].get("family"), "lane": full[rid].get("lane"),
              "dataset": full[rid].get("dataset")} for rid in full_ids]
    return full_ids, smoke_ids, missing, bargs, races


def cmd_ids_all(args):
    planner = load_planner(args.planner)
    full, smoke, missing, bargs, races = plan_all(planner, args.vendor)
    os.makedirs(args.out_dir, exist_ok=True)
    for name, lines in (("race-ids-full.txt", full), ("race-ids-smoke.txt", smoke), ("board-args.txt", bargs)):
        write_atomic(os.path.join(args.out_dir, name), "".join(x + "\n" for x in lines))
    write_atomic(os.path.join(args.out_dir, "not-planned.json"), json_text(missing))
    # the shape tools/measure_owed.py data-keys reads as --paper-recipes (family, lane, dataset per race)
    write_atomic(os.path.join(args.out_dir, "races.json"), json_text({"recipes": races}))
    print("PAPER-IDS-ALL vendor=%s planned=%d no_smoke=%d" % (args.vendor, len(full), len(missing)))
    return 0


def race_smoke_key(rid):
    """bench_board.smoke_key of a race id: family/lane/dataset, the key smoke.json files its races under and the
    board's smoke gate reads (the same race at smoke and at full size)."""
    return "/".join(rid.split("/")[:3])


def passed_full_ids(full_ids, smoke_docs):
    """(full ids whose smoke passed, {full id: why not}). Matched on the smoke key (family/lane/dataset), as the
    board's own smoke gate does: the 0.8.66 NVIDIA run lost 15 of 62 races to "no smoke result" when this
    matched the smoke id turned back into a full id (taxi classical lanes, int8 gemm, decode lanes, ...)."""
    passed, seen = set(), {}
    for doc in smoke_docs:
        for key, v in (doc.get("races") or {}).items():
            k = race_smoke_key(v.get("id") or key) if "/" in (v.get("id") or key) else key
            seen[k] = v
            if v.get("pass") is True:
                passed.add(k)
    keep, failed = [], {}
    for rid in full_ids:
        k = race_smoke_key(rid)
        if k in passed:
            keep.append(rid)
        elif k in seen:
            v = seen[k]
            fails = v.get("failures") or []
            failed[rid] = short_status("smoke failed: " + "; ".join(
                "%s: %s" % (f.get("arm"), f.get("reason")) for f in fails if isinstance(f, dict)) if fails
                else "smoke failed")
        else:
            failed[rid] = "no smoke result"
    return keep, failed


def cmd_pass_ids(args):
    with open(args.full_ids) as fh:
        full = [x.strip() for x in fh if x.strip()]
    docs = []
    for p in args.smoke_json:
        try:
            docs.append(load_json(p))
        except (OSError, ValueError) as exc:
            print("PAPER-PASS-IDS unreadable smoke %s: %s" % (p, exc), file=sys.stderr)
    keep, failed = passed_full_ids(full, docs)
    write_atomic(args.out, "".join(x + "\n" for x in keep))
    write_atomic(os.path.join(os.path.dirname(os.path.abspath(args.out)), "smoke-failed.json"), json_text(failed))
    # each failed race with its reason, before the count line (the job prints them to job.log): the d95f805cce
    # full board lost 23 0.8.55 races at smoke and only the count reached the Mac (fb-harness, 2026-11-17)
    for rid in sorted(failed):
        print("PAPER-PASS-IDS-FAILED %s: %s" % (rid, str(failed[rid])[:300]))
    print("PAPER-PASS-IDS planned=%d smoke_passed=%d smoke_failed=%d" % (len(full), len(keep), len(failed)))
    return 0


def ids_left(ids, board):
    """The ids not yet done in BOARD (a board.json document or None): what a resumed timed board still runs."""
    races = (board or {}).get("races") or {}
    return [x for x in ids if (races.get(x) or {}).get("status") != "done"]


def cmd_left(args):
    """(box) the number of races of --ids not yet done in --board: the disk gate counts only these (fb-harness,
    2026-11-17: a resumed full board was gated on all 650 races and refused with 23 GB free)."""
    with open(args.ids) as fh:
        ids = [x.strip() for x in fh if x.strip()]
    board = None
    if os.path.isfile(args.board):
        try:
            board = load_json(args.board)
        except (OSError, ValueError):
            board = None
    print(len(ids_left(ids, board)))
    return 0


# ---------------------------------------------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------------------------------------------

def merge_boards(boards):
    races, gpus, versions = {}, [], []
    for b in boards:
        g = gpu_name(b)
        if g and g not in gpus:
            gpus.append(g)
        for rid, race in (b.get("races") or {}).items():
            races[rid] = race
            for c in race.get("cells") or []:
                v = c.get("library_version")
                if str(c.get("arm", "")).startswith("ours") and v and v not in versions:
                    versions.append(v)
    return races, gpus, versions


def find_cell(race, phase, arm, batch):
    cells = race.get("cells" if phase == "fit" else "infer_cells") or []
    return next((c for c in cells if c.get("arm") == arm and (c.get("batch") or None) == (batch or None)), None)


#: lane fix-gbdt-prebinned-infer (2026-11-13; PARAMS-DIFF on trees/gbdt-depthwise/istella ours-prebinned, every
#: cell): the pre-binned `input_binning` record gained "; <n> training rows" on 2026-10-30 (FAIRNESS FIX,
#: tools/speed_gbdt_arm.py PREBIN_ROWS, read by the LightGBM lossguide mapping) after the 0.8.55 paper record was
#: taken. The borders, the border count and the bins sha256 (which hashes both binned blocks, so it fixes the row
#: count too) are unchanged, so the clause is spelling only for this comparison; any other change still differs.
_PREBIN_ROWS_CLAUSE = re.compile(r"; \d+ training rows(?=;)")


def _param_value(key, value):
    if key == "input_binning" and isinstance(value, str):
        return _PREBIN_ROWS_CLAUSE.sub("", value)
    return value


def _same_fp32(a, b):
    """Whether two read-back values are the same setting for an FP32 arm: equal, or floats (and lists of them) that
    round to the same float32."""
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_fp32(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b or struct.pack("<f", a) == struct.pack("<f", b)
    return a == b


def params_diff(old, new):
    # FAIRNESS FIX (2026-11-16): neural/sequential-fit and sequential-predict were flagged PARAMS-DIFF
    # ['learning_rate'] on every arm of the d95f805cce L40S job. Cause: the harness, not the product. The paper record
    # (0.8.55 board) read back learning_rate 0.001; on 2026-10-27 (02cedb9b1c, FAIRNESS FIX 2026-10-13 audit)
    # tools/bench_board_neural.py FIT_ADAM["lr"] became float.fromhex("0x1.0624de0000000p-10") = float32(1e-3),
    # passed to every fit arm, so every arm now reads back 0.0010000000474974513 (the on-box 0.8.55 arm and the
    # candidate agree). The product default is unchanged and the lane passes the value explicitly. Both values are
    # the same float32, which is what the FP32 arms run, so a float (or a list of floats) that rounds to the same
    # float32 is the same setting here; any other difference still flags.
    if old is None or new is None:
        return []
    keys = sorted(set(old) | set(new))
    return [k for k in keys if not _same_fp32(_param_value(k, old.get(k)), _param_value(k, new.get(k)))]


def same_clock_ms(opp, cell, phase):
    """Our new time on the opponent comparison's clock: whole = median_ms, kernel = the derived kernel span."""
    if cell is None:
        return None
    if opp.get("clock") == "kernel":
        return kernel_ms(cell) if phase == "fit" else None
    if opp.get("clock") == "whole":
        return cell.get("median_ms") if number(cell.get("median_ms")) else None
    return None       # MIXED clocks: the paper shows no ratio; neither do we


def compare_one(rec, races, not_run_why, hw_diff, base=None, same_host_opponents=False):
    """One recipe row. BASE: the on-box baseline (the paper's version re-timed in the same job on the same box):
    {"races": {...}, "label": v}, or None. Every time flag (REGRESSION, FASTER, SAME, FIXED, BROKEN,
    STILL-FAILING) comes from BASE only; the stored paper record is the reference column (ref_ms, with its host) and
    never makes a flag (AGENTS.md "Times are compared only on the same box", 2026-11-14). Without BASE the row is
    NOT-COMPARED. Quality and parameters are still judged against the paper record (they are not times).
    SAME_HOST_OPPONENTS: the stored opponents were timed on this box (NOW-WINS / NOW-LOSES may then be derived)."""
    old = rec["v0855"]
    race = races.get(rec["race_id"])
    cell = find_cell(race, rec["phase"], rec["arm"], rec["batch"]) if race else None
    ref_t = timed(old["status"], old["median_ms"])
    row = {"key": rec["key"], "race_id": rec["race_id"], "family": rec["family"], "lane": rec["lane"],
           "dataset": rec["dataset"], "phase": rec["phase"], "arm": rec["arm"], "batch": rec["batch"],
           "ref_ms": old["median_ms"] if ref_t else None, "ref_status": old["status"],
           "old_ms": None, "old_status": "no on-box baseline" if base is None else "not run on this box",
           "old_quality": shown_quality(old["quality"]), "flags": []}
    if base is not None:
        brace = (base.get("races") or {}).get(rec["race_id"])
        bcell = find_cell(brace, rec["phase"], rec["arm"], rec["batch"]) if brace else None
        if bcell is not None:
            row["old_status"] = short_status(bcell.get("status"))
            if timed(row["old_status"], bcell.get("median_ms")):
                row["old_ms"] = bcell.get("median_ms")
                row.update(calls_fields("old", bcell))
    if cell is None:
        row.update(new_ms=None, new_status="missing", ratio=None, new_quality={},
                   why=not_run_why.get(rec["race_id"]) or ("race not in the new board" if race is None
                                                          else "no %s cell for arm %s" % (rec["phase"], rec["arm"])))
        row["flags"].append("NOT-RUN")
        row["opponents"] = [dict(o, ratio_new=None) for o in rec["opponents"]]
        return row
    new_status = short_status(cell.get("status"))
    new_t = timed(new_status, cell.get("median_ms"))
    row.update(new_ms=cell.get("median_ms") if new_t else None, new_status=new_status,
               new_quality=shown_quality(cell.get("quality")))
    if new_t:
        row.update(calls_fields("new", cell))
        row["new_first_ms"] = first_call_ms(cell)
    missing = quality_missing(old["quality"], cell.get("quality")) if new_t else []   # untimed: BROKEN says it
    if missing:
        # QUALITY-MISSING (2026-11-13): the 0.8.55 record holds a judged metric the new cell lacks; the row cannot be
        # judged on it. Shown with the cell's own reason when it gave one; never a gate.
        row["flags"].append("QUALITY-MISSING")
        row["quality_missing"] = missing
        row["quality_unavailable"] = (str((cell.get("quality") or {}).get(UNAVAILABLE) or "")[:600]
                                      or "the new cell recorded no reason (its scorer did not record it)")
    row["ratio"] = None
    row["ratio_vs_ref"] = row["new_ms"] / row["ref_ms"] if new_t and row["ref_ms"] else None   # shown, never a flag
    old_t = row["old_ms"] is not None
    if base is None:
        row["flags"].append("NOT-COMPARED")
    elif old_t and new_t:
        row["ratio"] = row["new_ms"] / row["old_ms"] if row["old_ms"] else None
        r = row["ratio"]
        row["flags"].append("REGRESSION" if r is not None and r > SLOWER else
                            "FASTER" if r is not None and r < FASTER else "SAME")
        flag_scoring_mixed(row, "baseline", row.get("old_scoring"), replaces=SCORING_MIXED_REPLACES,
                           other_first_ms=first_call_ms(bcell))
    elif row["old_status"] == "not run on this box":
        # the prior wheel has no cell for this row (its smoke failed, it refused at plan, the job did not reach it):
        # NOT-RUN for that column, never a regression (brief same-box-compare, 2026-11-14)
        row["flags"].append("BASELINE-NOT-RUN")
    else:
        row["flags"].append("FIXED" if new_t else "BROKEN" if old_t else "STILL-FAILING")
    # quality: the new release judged against 0.8.55 under the board's own policy (new = "ours")
    q = BQ.compare({"arm": rec["arm"], "lane": rec["lane"], "quality": cell.get("quality") or {},
                    "status": cell.get("status")},
                   {"arm": rec["arm"], "lane": rec["lane"], "quality": old["quality"] or {},
                    "status": old["status"]})
    row["quality_verdict"] = q["verdict"]
    bad = [c for c in q["checks"] if c.get("verdict") == "FAIL"]
    row["quality_failures"] = ["%s %s->%s (%s, margin %s)" % (c["metric"], c.get("opponent"), c.get("ours"),
                                                            c.get("comparison"), c.get("margin")) for c in bad]
    if any(str(c.get("comparison", "")).startswith(("one-sided", "both-")) for c in bad) or q["failures"]:
        row["flags"].append("QUALITY-DROP")
    elif bad:
        row["flags"].append("QUALITY-CHANGE")
    # opponents: the paper's stored numbers (their host is the paper board's), our new time on the same clock. The
    # ratio is shown; NOW-WINS / NOW-LOSES only when the stored opponent ran on this box (never across hosts).
    opps = []
    for o in rec["opponents"]:
        mine = same_clock_ms(o, cell, rec["phase"]) if new_t else None
        ratio_new = mine / o["opponent_ms"] if number(mine) and number(o.get("opponent_ms")) and o["opponent_ms"] \
            else None
        opps.append(dict(o, ours_ms_new=mine, ratio_new=ratio_new))
    row["opponents"] = opps
    best = [o for o in opps if number(o.get("opponent_ms")) and number(o.get("ratio"))]
    if best:
        b = min(best, key=lambda o: o["opponent_ms"])
        row["best_opponent"] = b
        if number(b.get("ratio_new")) and same_host_opponents:
            if b["ratio"] > 1.0 >= b["ratio_new"]:
                row["flags"].append("NOW-WINS")
            elif b["ratio"] <= 1.0 < b["ratio_new"]:
                row["flags"].append("NOW-LOSES")
    new_params = (cell.get("params") or {}).get("params")
    diff = params_diff(rec["params"], new_params)
    row["params_diff"] = diff
    if diff:
        row["flags"].append("PARAMS-DIFF")
    if hw_diff:
        row["flags"].append("HW-DIFF")
    return row


def board_candidate(boards):
    """The candidate identity of the new boards (bench_board --candidate-wheel: box.mojolearn.wheel carries the
    candidate manifest), or None for a published release. Two different candidates in one compare refuse."""
    found = {}
    for b in boards:
        wheel = (((b.get("box") or {}).get("mojolearn") or {}).get("wheel") or {})
        man = wheel.get("candidate") if wheel.get("source") == "candidate" else None
        if not man:
            continue
        ident = {"source_commit": man.get("source_commit"), "version": man.get("version"),
                 "vendor": man.get("vendor"),
                 "wheels": {f.get("distribution"): {"file": f.get("file"), "sha256": f.get("sha256")}
                            for f in man.get("files") or []}}
        found[json.dumps(ident, sort_keys=True)] = ident
    if len(found) > 1:
        raise SystemExit("paper_compare: the boards hold %d different candidates; compare one at a time"
                         % len(found))
    return next(iter(found.values()), None)


def new_label(candidate, versions):
    """'candidate <sha12> (<version>)' for a candidate build, so it is never read as a release."""
    if candidate:
        return "candidate %s (%s)" % ((candidate.get("source_commit") or "?")[:12], candidate.get("version"))
    return ",".join(versions) or "?"


def attach_prior_board(row, version, races, flagged):
    """One on-box prior column (a prior release re-timed in the same job on the same box): row["priors"][version]
    = {ms, status, ratio new/prior}. FLAGGED (the last release, PC_PRIOR_VERSION, on a verified same box) adds:
      REGRESSION-VS-PRIOR new / prior > 1.10      FASTER-VS-PRIOR new / prior < 0.90
      RECOVERED           the prior was > 1.10 x the on-box baseline; new is SAME or FASTER vs that baseline
      STILL-REGRESSED     the prior was > 1.10 x the on-box baseline and new still is
    A prior without a cell (its smoke failed, it refused, the job did not reach it) is NOT-RUN in its column."""
    c = find_cell((races or {}).get(row["race_id"]), row["phase"], row["arm"], row["batch"]) \
        if (races or {}).get(row["race_id"]) else None
    status = short_status(c.get("status")) if c else "NOT-RUN on this box"
    ms = c.get("median_ms") if c and timed(status, c.get("median_ms")) else None
    r = (row["new_ms"] / ms) if number(row.get("new_ms")) and ms else None
    row.setdefault("priors", {})[version] = dict({"ms": ms, "status": status, "ratio": r},
                                                 **(calls_fields("prior", c) if ms is not None else {}))
    mixed = r is not None and scoring_differs(row, row["priors"][version].get("prior_scoring"))
    if r is not None:
        flag_scoring_mixed(row, version, row["priors"][version].get("prior_scoring"),
                           other_first_ms=first_call_ms(c))
    if not flagged:
        return row
    row["prior_ms"], row["prior_status"], row["ratio_vs_prior"] = ms, status, r
    if r is not None:
        verdict = "REGRESSION-VS-PRIOR" if r > SLOWER else "FASTER-VS-PRIOR" if r < FASTER else None
        if verdict and mixed:
            # SCORING-MIXED replaces the prior column's verdict (a median of calls against one call)
            row.setdefault("scoring_mixed_replaced", []).append(verdict)
        elif verdict:
            row["flags"].append(verdict)
    if ms and number(row.get("old_ms")) and ms / row["old_ms"] > SLOWER:
        if "SAME" in row["flags"] or "FASTER" in row["flags"]:
            row["flags"].append("RECOVERED")
        elif "REGRESSION" in row["flags"]:
            row["flags"].append("STILL-REGRESSED")
    return row


def attach_reference(row, ref):
    """A stored compare's new column (another run, another host): shown as reference only, never a flag."""
    p = ref.get(row["key"]) if ref else None
    row["ref_prior_ms"] = p.get("new_ms") if p and number(p.get("new_ms")) else None
    row["ref_prior_status"] = p.get("new_status") if p else "not in the reference compare"
    return row


VENDOR_TAG = {"nvidia": "NV", "amd": "AMD"}


def new_cell_records(boards, vendor):
    """{race key: our identical cell record} of the new boards (tools/board_best_ever.cell_records; later boards
    win, as in merge_boards)."""
    out = {}
    for b in boards:
        for rec in BE.cell_records(b):
            if rec["vendor"] == vendor:
                out[rec["key"]] = rec
    return out


def attach_best_ever(row, new_rec, best_rows, host=None, software=None):
    """The best-ever column of one row (module docstring). NEW_REC: the new cell's record or None. HOST: the new
    board's host id; SOFTWARE: its one system state (SOFTWARE_FIELDS values, a list) or None. The best ever is a stored
    time of another run: its ratio is shown, and the time flags (BEST-EVER-REGRESSION, -UNCHECKED, -SUSPECT) are
    derived only when it carries the same host id (2026-11-14) and the same system state (2026-11-17)."""
    b = (best_rows or {}).get(row["key"])
    row.update(best_ever_ms=None, best_ever_date=None, best_ever_release=None, best_ever_commit=None,
               best_ever_device=None, best_ever_hash=None, best_ever_host=None, ratio_vs_best_ever=None,
               best_ever_params_differ=[], best_ever_hash_differs=False)
    if not b:
        return row
    device = (new_rec or {}).get("device")
    best = (b.get("best_by_device") or {}).get(device) if device else None
    best = best or b["best"]
    timeline = b.get("timeline") or []
    suspect = (b["best"].get("suspect") if best is b["best"] else None) \
        or BE.suspect_reason(best, [t for t in timeline if t.get("ms") != best.get("ms")] + [best])
    row["best_ever_suspect"] = suspect
    row.update(best_ever_ms=best["ms"], best_ever_date=best.get("date"), best_ever_release=best.get("release"),
               best_ever_commit=best.get("commit"), best_ever_device=best.get("device"),
               best_ever_hash=best.get("hash"), best_ever_host=best.get("host_id"))
    if not new_rec or not number(row.get("new_ms")):
        return row
    pd = BE.params_differ(best.get("params"), new_rec.get("params"))
    hd = bool(best.get("hash") and new_rec.get("hash") and best["hash"] != new_rec["hash"])
    row["best_ever_params_differ"], row["best_ever_hash_differs"] = pd, hd
    ratio = row["new_ms"] / best["ms"] if best["ms"] else None
    row["ratio_vs_best_ever"] = ratio
    # PARAMS-DIFFER FIX (fb-harness, 2026-11-17): the flag was raised when only the output hash differed. The best ever
    # is an older version's cell, and bits may change between versions (AGENTS.md "Bits may change between
    # versions"): 241 of the 272 hash-only rows of the d95f805cce full board had the candidate's hash EQUAL to 0.8.55's
    # on the same box (e.g. classical/ols/taxi fit: f396.. both, best ever 0.8.25 89843bf5..) with params_check
    # MATCHED. The flag now means what it says, a shared parameter with another value; the hash difference is still
    # shown (best_ever_hash_differs, "vs best ever: hash" in the note) and no longer hides a time flag.
    if pd:
        row["flags"].append("PARAMS-DIFFER")
    elif (ratio is not None and ratio > SLOWER and host and best.get("host_id") == host
          and software is not None and best.get("software") == software):
        known = (isinstance(best.get("params"), dict) and isinstance(new_rec.get("params"), dict)
                 and best.get("hash") and new_rec.get("hash"))
        if suspect:
            row["flags"].append("BEST-EVER-SUSPECT")
        else:
            row["flags"].append("BEST-EVER-REGRESSION" if known else "BEST-EVER-REGRESSION-UNCHECKED")
    if device and best.get("device") and best["device"] != device:
        row["flags"].append("BEST-EVER-HW-DIFF")
    return row


def extra_row(rec):
    """A row for a new-board cell outside the recipes (the full-board screen): new time and best ever only."""
    return {"key": rec["key"], "race_id": rec["race_id"], "family": rec.get("family"), "lane": rec.get("lane"),
            "dataset": rec.get("dataset"), "phase": rec["phase"], "arm": rec["arm"], "batch": rec.get("batch"),
            "new_ms": rec["ms"], "new_hash": rec.get("hash"), "new_device": rec.get("device"), "flags": [],
            **calls_fields("new", rec)}


# ---------------------------------------------------------------------------------------------------------------
# the STORED baseline (Andrew, 2026-11-17, AGENTS.md part 8 "0.8.55 is not timed again"): the 0.8.55 times of the
# 2026-11-17 L40S full board are the campaign's baseline; a job times our new wheel only and is compared against
# that stored board. The same-box and same-software refusals of the two-arm modes become WARNING lines here (the
# direction overrides them for this campaign), and the first line of the summary says the baseline is stored.
# ---------------------------------------------------------------------------------------------------------------
#: a row whose in-race upload probe differs from the stored run's by more than this fraction is BOX-STATE: the box's
#: host-memory path moved (lane fb3-drift, 2026-11-17: istella X upload 85 ms in three runs, 113..126 ms in the slow
#: one, and single-call rows moved by the same milliseconds); its ratio is shown and is not a verdict
BOX_STATE_TOLERANCE = 0.15
VERDICT_FLAGS = ("REGRESSION", "FASTER", "SAME")


def board_side(board):
    """{host, gpu, drivers, kernels, date} of one board, for the stored-baseline line: every distinct value its
    races ran under (SOFTWARE_FIELDS), its host ids, its GPU name and the date its probe was taken (else created)."""
    states = board_software(board)
    probe = board_probe(board) or {}
    vals = lambda i: sorted({str(k[i]) for k in states if k[i]})  # noqa: E731
    return {"host": ", ".join(sorted(board_host_ids(board))) or host_label(board) or "not recorded",
            "gpu": (probe.get("gpu") or {}).get("name") or gpu_name(board) or "?",
            "drivers": vals(1), "kernels": vals(2),
            "date": str(probe.get("taken") or board.get("created") or "?")[:10]}


def side_text(s):
    return "host %s, GPU %s, driver %s, kernel %s, date %s" % (
        s["host"], s["gpu"], " / ".join(s["drivers"]) or "?", " / ".join(s["kernels"]) or "?", s["date"])


def stored_baseline_basis(version, stored, boards):
    """(basis line, [WARNING lines]) of a compare against a stored baseline board: the basis names both sides; a
    WARNING says when the host or the GPU driver differ (the direction of 2026-11-17 makes these warnings, not the
    refusal of the two-arm modes)."""
    st = board_side(stored)
    new = [board_side(b) for b in boards]
    nh = sorted({s["host"] for s in new})
    nd = sorted({d for s in new for d in s["drivers"]})
    cur = {"host": ", ".join(nh) or "not recorded", "gpu": ", ".join(sorted({s["gpu"] for s in new})) or "?",
           "drivers": nd, "kernels": sorted({k for s in new for k in s["kernels"]}),
           "date": ", ".join(sorted({s["date"] for s in new})) or "?"}
    basis = ("STORED BASELINE, not timed in this job: %s stored (%s); this job: %s"
             % (version, side_text(st), side_text(cur)))
    warn = []
    if set(nh) != {st["host"]}:
        warn.append("WARNING: the stored %s baseline ran on host %s, this job on %s; times of two hosts are compared "
                    "under the 2026-11-17 direction (0.8.55 is not timed again)" % (version, st["host"], cur["host"]))
    if nd and st["drivers"] and set(nd) != set(st["drivers"]):
        warn.append("WARNING: GPU driver %s (stored %s) vs %s (this job)"
                    % (" / ".join(st["drivers"]), version, " / ".join(nd)))
    return basis, warn


#: the dataset fallback reads only uploads this large (bandwidth-bound, not latency-bound), so a targeted job's few
#: races and the stored board's many give comparable medians
LARGE_UPLOAD_BYTES = 16 * 1000 * 1000


def upload_probe(cell):
    """(ms, GB/s, bytes) of a cell's in-race separate upload probe (the harness's own, outside the clock:
    comparability.span upload_ms_separate over upload_bytes_separate), or None."""
    span = ((cell or {}).get("comparability") or {}).get("span") or {}
    ms, nb = span.get("upload_ms_separate"), span.get("upload_bytes_separate")
    if number(ms) and ms > 0 and number(nb) and nb > 0:
        return ms, nb / ms / 1e6, nb
    return None


def race_probe(race):
    """The upload probe of a race: the first of our cells that carries one (the fit cell), or None."""
    for c in (race or {}).get("cells") or []:
        if str(c.get("arm", "")).startswith("ours"):
            p = upload_probe(c)
            if p:
                return p
    return None


def dataset_probes(races):
    """{dataset: median GB/s} of the race probes in RACES of at least LARGE_UPLOAD_BYTES: the fallback for a race
    that has none of its own."""
    by = {}
    for r in (races or {}).values():
        p = race_probe(r) if isinstance(r, dict) else None
        if p and p[2] >= LARGE_UPLOAD_BYTES:
            by.setdefault(r.get("dataset"), []).append(p[1])
    return {d: sorted(v)[len(v) // 2] for d, v in by.items()}


def attach_box_state(row, new_races, stored_races, new_ds, stored_ds):
    """The in-race upload probe of the row's race beside the stored run's for the same race (else the median of the
    same dataset on each side), and BOX-STATE when they differ by more than BOX_STATE_TOLERANCE: the verdict flag
    (REGRESSION / FASTER / SAME) is then replaced by BOX-STATE and the ratio is shown, not counted."""
    rid = row["race_id"]
    nr, sr = (new_races or {}).get(rid), (stored_races or {}).get(rid)
    ds = row.get("dataset") or (nr or {}).get("dataset") or (sr or {}).get("dataset")
    n, s = race_probe(nr), race_probe(sr)
    ng = n[1] if n else new_ds.get(ds)
    sg = s[1] if s else stored_ds.get(ds)
    if ng is None or sg is None:
        row["upload_probe"] = None
        return row
    diff = ng / sg - 1.0
    row["upload_probe"] = {"new_ms": n[0] if n else None, "new_gbps": ng, "stored_ms": s[0] if s else None,
                           "stored_gbps": sg, "diff": diff,
                           "source": "race" if n and s else "dataset %s median" % ds}
    if abs(diff) > BOX_STATE_TOLERANCE and any(f in row["flags"] for f in VERDICT_FLAGS):
        row["flags"] = [f for f in row["flags"] if f not in VERDICT_FLAGS] + ["BOX-STATE"]
    return row


HARNESS_CLOCK_REPLACES = ("FASTER", "SAME")
NUMA_MIXED_REPLACES = ("REGRESSION", "FASTER")


def unbound_node(cell):
    """The NUMA node an UNBOUND worker of CELL was on when it recorded itself, else None (no record, bound by the
    conductor, or node unknown)."""
    wn = (cell or {}).get("worker_numa")
    if not isinstance(wn, dict):
        return None
    # a placement asked but not applied (the worker's affinity outside the planned cores; tools/numa_place.py
    # worker_node "bound": False, lane fx-numa-bind-smoke 2026-11-19) is unbound too
    if (wn.get("placement") or {}).get("node") is not None and wn.get("bound") is not False:
        return None
    node = wn.get("node_now")
    return node if isinstance(node, int) and not isinstance(node, bool) else None


def attach_cell_marks(row, new_races, base_races):
    """Lane fx-harness (2026-11-18): what the new cell recorded about its host and its clock, beside the row.
    worker_numa (the node the worker ran on and the placement asked), host_usage (the scored calls' system time and
    faults), and the flags HOST-STALL (system time above half the wall: a REGRESSION becomes HOST-STALL) and
    HARNESS-CLOCK (the baseline cell was timed under another harness-clock revision: FASTER / SAME become
    HARNESS-CLOCK, a REGRESSION stays)."""
    rid = row.get("race_id")
    nr, br = (new_races or {}).get(rid), (base_races or {}).get(rid)
    nc = find_cell(nr, row["phase"], row["arm"], row.get("batch")) if nr else None
    if nc is None:
        return row
    bc = find_cell(br, row["phase"], row["arm"], row.get("batch")) if br else None
    wn = nc.get("worker_numa")
    if isinstance(wn, dict):
        pl = wn.get("placement") or {}
        row["worker_numa"] = {"node_now": wn.get("node_now"), "cpu_nodes": wn.get("cpu_nodes"),
                              "mems_allowed": wn.get("mems_allowed"), "bound_node": pl.get("node"),
                              "mem": pl.get("mem"), "why": pl.get("why")}
    use = nc.get("host_usage")
    if isinstance(use, dict):
        row["host_usage"] = {k: use.get(k) for k in ("stime_fraction", "stime_ms", "wall_ms", "minflt", "majflt",
                                                     "calls")}
    if nc.get("host_stall"):
        had = "REGRESSION" in row["flags"]
        row["flags"] = [f for f in row["flags"] if f != "REGRESSION"] + ["HOST-STALL"]
        row["host_stall_replaced"] = "REGRESSION" if had else None
    if bc is not None and (nc.get("harness_clock") or None) != (bc.get("harness_clock") or None):
        row["harness_clock"] = {"new": nc.get("harness_clock"), "baseline": bc.get("harness_clock")}
        replaced = [f for f in row["flags"] if f in HARNESS_CLOCK_REPLACES]
        row["flags"] = [f for f in row["flags"] if f not in HARNESS_CLOCK_REPLACES] + ["HARNESS-CLOCK"]
        row["harness_clock"]["replaced"] = replaced or None
    new_node, base_node = unbound_node(nc), unbound_node(bc)
    if new_node is not None and base_node is not None and new_node != base_node:
        replaced = [f for f in row["flags"] if f in NUMA_MIXED_REPLACES]
        row["flags"] = [f for f in row["flags"] if f not in NUMA_MIXED_REPLACES] + ["NUMA-MIXED"]
        row["numa_mixed"] = {"new": new_node, "baseline": base_node, "replaced": replaced or None}
    return row


def opponent_rescore_rows(races):
    """H2 of the 2026-11-18 generality review (lane fx-harness-honest): every race of the new boards whose own cell
    carries a harness_clock that changed every arm's clock or settings key (tools/race_fairness.py
    opponent_rescore_owed). Its stored opponent cells are BROKEN against it until the opponent job re-scores them
    (Andrew set opponents aside: nothing here runs one); the compare lists them once so the gap is not lost.
    Our arm's time and the HARNESS-CLOCK mark of these rows are shown as for any row. -> [{race_id, harness_clock}]"""
    out = {}
    for rid, race in sorted((races or {}).items()):
        for c in (race.get("cells") or []) + (race.get("infer_cells") or []):
            hc = RF.opponent_rescore_owed(c)
            if hc:
                out.setdefault(rid, hc)
    return [{"race_id": rid, "harness_clock": hc} for rid, hc in sorted(out.items())]


def marked_rows(res, flag):
    """Every recipe, extra and internal row carrying FLAG."""
    return [r for k in ("rows", "extra_rows", "internal_rows") for r in res.get(k) or [] if flag in r["flags"]]


def probe_text(p):
    if not p:
        return "-"
    return "%s GB/s%s vs stored %s GB/s%s (%+.0f%%, %s)" % (
        "%.2f" % p["new_gbps"], " (%s ms)" % fmt_ms(p["new_ms"]) if p.get("new_ms") is not None else "",
        "%.2f" % p["stored_gbps"], " (%s ms)" % fmt_ms(p["stored_ms"]) if p.get("stored_ms") is not None else "",
        100 * p["diff"], p["source"])


def compare(doc, boards, not_run_why=None, prior_boards=None, reference=None, reference_label=None,
            best_ever=None, stored_baseline=None, verdict_prior=None):
    """PRIOR_BOARDS: [(version, board doc)] of OUR prior wheels re-timed in the same job on the same box
    (tools/ops/paper_compare_job.sh board-prior-<version>/): the one whose version is the paper's is the baseline
    (old_ms, the REGRESSION / FASTER / FIXED / BROKEN flags), the last other one the flagged prior column
    (REGRESSION-VS-PRIOR, RECOVERED, STILL-REGRESSED). A prior board from another host refuses (SystemExit).
    REFERENCE: {recipe key: row} of a stored compare JSON (another run): a reference column, never a flag.
    BEST_EVER: a tools/board_best_ever.py document of the compare's vendor (the best-ever column), or None.
    STORED_BASELINE: a board of the paper's version from another job (the 2026-11-17 direction: 0.8.55 is not timed
    again): it is the baseline in place of an on-box prior board; host and driver differences are WARNING lines, not
    refusals, and each row's upload probe is checked against it (BOX-STATE).
    VERDICT_PRIOR: the name of one PRIOR_BOARDS entry that is the baseline instead of the paper's version: one of OUR
    OWN EARLIER BUILT WHEELS timed in the same job on the same box (paper_compare_job.sh PC_PRIOR_WHEELS; lane
    fx-same-pod-prior, Andrew 2026-11-19: a speed verdict compares two wheels timed on the same host in the same job).
    REGRESSION / FASTER / SAME then come from that pair only, under the same same-box refusal; a STORED_BASELINE given
    with it is a screen column (its ratio shown, never a flag, no BOX-STATE)."""
    races, gpus, versions = merge_boards(boards)
    candidate = board_candidate(boards)
    paper_gpu = doc["source"].get("gpu")
    hw_diff = bool(gpus) and paper_gpu is not None and any(g != paper_gpu for g in gpus)
    host, why_no_host = candidate_host(boards)
    priors = []
    stored_basis, stored_warn = None, []
    if stored_baseline is not None:
        if any(v == doc["paper_version"] for v, _ in prior_boards or []):
            raise SystemExit("paper_compare: a stored baseline and an on-box %s board in one compare; give one"
                             % doc["paper_version"])
        stored_basis, stored_warn = stored_baseline_basis(doc["paper_version"], stored_baseline, boards)
        priors.append({"version": doc["paper_version"], "races": stored_baseline.get("races") or {},
                       "host_id": board_side(stored_baseline)["host"], "why_not": None, "stored": True,
                       "probe": board_probe(stored_baseline), "versions": merge_boards([stored_baseline])[2]})
    for version, pb in prior_boards or []:
        h, why = same_box(boards, [pb], "the %s prior board" % version)    # refuses two hosts
        if h:
            # one host id is not enough: the same GPU, driver, kernel and boot (SOFTWARE_FIELDS), else reference only
            sw = software_mismatch(boards, pb, "the %s board" % version)
            if sw:
                h, why = None, sw
        priors.append({"version": version, "races": pb.get("races") or {}, "host_id": h, "why_not": why,
                       "probe": board_probe(pb), "versions": merge_boards([pb])[2]})
    if verdict_prior is not None:
        base = next((p for p in priors if p["version"] == verdict_prior and not p.get("stored")), None)
        if base is None:
            raise SystemExit("paper_compare: --verdict-prior %s names no --prior-board (%s)"
                             % (verdict_prior, ", ".join(v for v, _ in prior_boards or []) or "none given"))
        for p in priors:
            if p.get("stored"):
                # the stored board is a screen beside the on-box pair: shown, never verified, never a flag
                p["host_id"], p["why_not"] = None, "stored board of another job: a screen column, never a flag"
        stored_basis, stored_warn = None, []
    else:
        base = next((p for p in priors if p["version"] == doc["paper_version"]), None)
    stored_is_base = base is not None and bool(base.get("stored"))
    others = [p for p in priors if p is not base]
    live = [p for p in others if not p.get("stored")]
    flagged = live[-1] if live else (others[-1] if others and not verdict_prior else None)
    verified = lambda p: p is not None and p["host_id"] is not None  # noqa: E731
    stored_host = doc["source"].get("host")
    same_host_opps = bool(host and stored_host and stored_host == host)
    rows = [compare_one(rec, races, not_run_why or {}, hw_diff,
                        {"races": base["races"]} if verified(base) else None, same_host_opps)
            for rec in doc["recipes"]]
    for p in others:
        rows = [attach_prior_board(r, p["version"], p["races"], p is flagged and verified(p)) for r in rows]
    if reference is not None:
        rows = [attach_reference(r, reference) for r in rows]
    extra, extra_counts, be_label = [], {}, None
    # CALL OVERHEAD, INTERNAL CHECK (Andrew, 2026-11-17; tools/race_fairness.py rule 1c): a lane that is not a race
    # (the lr_at call-overhead row) is never a recipe, an extra row or a best-ever row. It gets its own section, new
    # against OUR prior versions re-timed on this box in this job only (the same REGRESSION / FASTER rule as a
    # recipe, when the prior's host and system software are verified); never a stored or best-ever column.
    recipe_keys = {r["key"] for r in doc["recipes"]}
    all_recs = new_cell_records(boards, doc["vendor"])
    internal = [attach_extra_priors(extra_row(rec), base, others, flagged, verified)
                for k, rec in sorted(all_recs.items()) if RF.lane_class(rec.get("lane")) and k not in recipe_keys]
    box_state = None
    if stored_is_base:
        sds, nds = dataset_probes(base["races"]), dataset_probes(races)
        box_state = lambda r: attach_box_state(r, races, base["races"], nds, sds)  # noqa: E731
        rows = [box_state(r) for r in rows]
        internal = [box_state(r) for r in internal]
    if base is not None:
        # HOST-STALL and HARNESS-CLOCK (lane fx-harness, 2026-11-18): from the new cell's own record, against the
        # baseline cell (on-box or stored)
        marks = lambda r: attach_cell_marks(r, races, base["races"])  # noqa: E731
        rows = [marks(r) for r in rows]
        internal = [marks(r) for r in internal]
    internal_counts = {}
    for r in internal:
        for f in r["flags"]:
            internal_counts[f] = internal_counts.get(f, 0) + 1
    if best_ever is not None:
        if best_ever.get("vendor") != doc["vendor"]:
            raise SystemExit("paper_compare: the best-ever file is %s, the recipes %s"
                             % (best_ever.get("vendor"), doc["vendor"]))
        be_label = "best ever (%s)" % VENDOR_TAG.get(doc["vendor"], doc["vendor"])
        new_recs = {k: rec for k, rec in all_recs.items() if not RF.lane_class(rec.get("lane"))}
        best_rows = best_ever.get("rows") or {}
        states = {}
        for b in boards:
            states.update(board_software(b))
        sw = list(next(iter(states))) if len(states) == 1 and all(next(iter(states))) else None
        rows = [attach_best_ever(r, new_recs.get(r["key"]), best_rows, host, sw) for r in rows]
        extra = [attach_best_ever(extra_row(rec), rec, best_rows, host, sw)
                 for k, rec in sorted(new_recs.items()) if k not in recipe_keys]
        if others or base:
            extra = [attach_extra_priors(r, base, others, flagged, verified) for r in extra]
        if box_state is not None:
            extra = [box_state(r) for r in extra]
        if base is not None:
            extra = [attach_cell_marks(r, races, base["races"]) for r in extra]
        for r in extra:
            for f in r["flags"]:
                extra_counts[f] = extra_counts.get(f, 0) + 1
    counts = {}
    for r in rows:
        for f in r["flags"]:
            counts[f] = counts.get(f, 0) + 1
    if stored_basis:
        basis = stored_basis
    elif verified(base) and verdict_prior is not None:
        basis = "same box %s: our wheel %s timed in this job beside the candidate%s" % (
            host, base["version"], "; the stored %s board is a screen only" % doc["paper_version"]
            if stored_baseline is not None else "")
    elif verified(base):
        basis = "same box %s: %s re-timed in this job" % (host, base["version"])
    elif base is not None:
        basis = "no verdict: %s" % base["why_not"]
    else:
        basis = "no verdict: no on-box %s board (--prior-board %s=...)" % (doc["paper_version"], doc["paper_version"])
    return {"schema": COMPARE_SCHEMA, "vendor": doc["vendor"], "paper_version": doc["paper_version"],
            "new_versions": versions, "new_label": new_label(candidate, versions), "candidate": candidate,
            "host_id": host, "host_probe": next((board_probe(b) for b in boards if board_probe(b)), None),
            "host_missing": why_no_host, "verdict_basis": basis,
            "baseline_stored": stored_is_base, "baseline_warnings": stored_warn,
            "box_state_tolerance": BOX_STATE_TOLERANCE if stored_is_base else None,
            "baseline_version": base["version"] if base else None, "verdict_prior": verdict_prior,
            "screen_stored": stored_baseline is not None and not stored_is_base,
            "baseline_label": ("%s (%s)" % (base["version"], "stored" if base.get("stored") else "this box"))
                              if base else None,
            "prior_label": flagged["version"] if flagged else None,
            "prior_columns": [{"version": p["version"], "host_id": p["host_id"], "probe": p["probe"],
                               "why_not": p["why_not"], "board_versions": p["versions"],
                               "stored": bool(p.get("stored"))} for p in priors],
            "reference_label": reference_label if reference is not None else None,
            "stored_host": stored_host,
            "paper_gpu": paper_gpu, "new_gpus": gpus, "hw_diff": hw_diff,
            "thresholds": doc.get("thresholds"), "recipes": len(rows), "flag_counts": counts, "rows": rows,
            "best_ever_label": be_label, "best_ever_source": (best_ever or {}).get("mined_at_head"),
            "extra_rows": extra, "extra_flag_counts": extra_counts,
            "internal_label": RF.CALL_OVERHEAD, "internal_rows": internal, "internal_flag_counts": internal_counts,
            "opponent_rescore_owed": opponent_rescore_rows(races),
            "progress": [dict(board_progress(boards), arm="candidate")]
                        + [dict(board_progress([pb]), arm=v) for v, pb in prior_boards or []]}


def box_state_rows(res):
    """Every recipe, extra and internal row flagged BOX-STATE."""
    return [r for k in ("rows", "extra_rows", "internal_rows") for r in res.get(k) or [] if "BOX-STATE" in r["flags"]]


def board_progress(boards):
    """{done, total, boot_ids}: the races of the boards' own plans that finished (status done), of how many; and every
    boot id their races recorded (two: the job resumed after a restart). A board without a plan counts its races."""
    plan, races, boots = [], {}, []
    for b in boards:
        plan += [x for x in (b.get("plan") or []) if isinstance(x, str)]
        races.update(b.get("races") or {})
        for x in (b.get("boot_ids") or []):
            if x not in boots:
                boots.append(x)
        default = BE.software_default(b)
        for r in (b.get("races") or {}).values():
            x = BE.race_software(r, default)[3]          # the race's boot id, else the board's probe's
            if x and x not in boots:
                boots.append(x)
    want = sorted(set(plan)) or sorted(races)
    done = sum(1 for rid in want if (races.get(rid) or {}).get("status") == "done")
    return {"done": done, "total": len(want), "boot_ids": boots}


def attach_extra_priors(row, base, others, flagged, verified):
    """The on-box prior columns of a full-board row outside the recipes: new / <version> per prior, and REGRESSION /
    FASTER vs the on-box baseline and the flagged prior when their host is verified (the same rule as a recipe)."""
    for p in ([base] if base else []) + others:
        attach_prior_board(row, p["version"], p["races"], False)
    # SCORING-MIXED replaces a column's REGRESSION / FASTER (flag_scoring_mixed; lane fx-metrics-ranking 2026-11-19)
    if verified(base):
        p = row["priors"][base["version"]]
        r = p["ratio"]
        if r is not None:
            verdict = "REGRESSION" if r > SLOWER else "FASTER" if r < FASTER else "SAME"
            if verdict != "SAME" and scoring_differs(row, p.get("prior_scoring")):
                row.setdefault("scoring_mixed_replaced", []).append(verdict)
            else:
                row["flags"].append(verdict)
    if verified(flagged):
        p = row["priors"][flagged["version"]]
        r = p["ratio"]
        verdict = None if r is None else ("REGRESSION-VS-PRIOR" if r > SLOWER else
                                          "FASTER-VS-PRIOR" if r < FASTER else None)
        if verdict and scoring_differs(row, p.get("prior_scoring")):
            row.setdefault("scoring_mixed_replaced", []).append(verdict)
        elif verdict:
            row["flags"].append(verdict)
    return row


def load_reference(path):
    """(rows by key, label) of a stored compare JSON (bench/results/paper_compare/compare-<vendor>.json): a reference
    column only. The label names the run and its host ("host not recorded" for a compare made before 2026-11-14)."""
    doc = load_json(path)
    if doc.get("schema") != COMPARE_SCHEMA:
        raise SystemExit("paper_compare: %s is not a %s document" % (path, COMPARE_SCHEMA))
    label = doc.get("new_label") or ",".join(doc.get("new_versions") or []) or "reference"
    return ({r["key"]: r for r in doc.get("rows") or []},
            "%s stored (host %s)" % (label, doc.get("host_id") or "not recorded"))


def fmt_ms(v):
    if not number(v):
        return "-"
    return "%.0f" % v if v >= 100 else "%.2f" % v if v >= 1 else "%.3f" % v


def calls_fields(prefix, cell):
    """SMALL ROWS (Andrew, 2026-11-15): {<prefix>_scoring} (the rule the time was taken under, SMALL.scoring_of:
    one call for every cell before 2026-11-15 and opponents) and, for a cell whose time is the median of more than
    one consecutive call, {<prefix>_calls, <prefix>_min_ms, <prefix>_max_ms}."""
    out = {prefix + "_scoring": SMALL.scoring_of(cell)}
    calls = (cell or {}).get("calls")
    if not isinstance(calls, int) or calls <= 1:
        return out
    return dict(out, **{prefix + "_calls": calls, prefix + "_min_ms": cell.get("min_ms"),
                        prefix + "_max_ms": cell.get("max_ms")})


#: the baseline column's verdicts SCORING-MIXED replaces (the prior column's are replaced in attach_prior_board)
SCORING_MIXED_REPLACES = ("REGRESSION", "FASTER")


def first_call_ms(cell):
    """The wall time of a cell's FIRST scored call (host_usage.per_call, tools/numa_place.py), or None when the cell
    recorded no per-call times. For a one-call cell it is its time; for a median of calls it is the call the
    small-row rule decided on, which is the same quantity."""
    use = (cell or {}).get("host_usage")
    per = use.get("per_call") if isinstance(use, dict) else None
    if isinstance(per, list) and per and isinstance(per[0], dict) and number(per[0].get("wall_ms")):
        return per[0]["wall_ms"]
    return None


def scoring_differs(row, other_scoring):
    """Whether the new time and another column's time were taken under different small-row rules."""
    new = row.get("new_scoring")
    return new is not None and other_scoring is not None and new != other_scoring


def flag_scoring_mixed(row, column, other_scoring, replaces=(), other_first_ms=None):
    """SCORING-MIXED (harness-small-row, 2026-11-16): the new time and the time of `column` it is divided by were
    taken under different small-row rules (one a median of calls, the other one call: their first scored calls
    fell on either side of 10 ms). The ratio is still shown; the flag says it compares a median with one call.
    Lane fx-metrics-ranking (2026-11-19): the verdicts in `replaces` are removed from the row's flags and recorded
    in scoring_mixed_replaced (the first scored call carries the first-touch page faults, so the arm that fell on
    one call reads slower than the arm that fell on the median, on unchanged code), and the note carries both arms'
    first scored calls when both cells recorded them (`other_first_ms`, row["new_first_ms"]): one sample each,
    shown, never a flag."""
    if not scoring_differs(row, other_scoring):
        return row
    note = "%s: new %s, %s %s" % (column, row.get("new_scoring"), column, other_scoring)
    first = row.get("new_first_ms")
    if number(first) and number(other_first_ms) and other_first_ms:
        note += "; first scored calls: new %s ms, %s %s ms (%.2f)" % (fmt_ms(first), column, fmt_ms(other_first_ms),
                                                                    first / other_first_ms)
    row.setdefault("scoring_mixed", []).append(note)
    if "SCORING-MIXED" not in row["flags"]:
        row["flags"].append("SCORING-MIXED")
    replaced = [f for f in row["flags"] if f in replaces]
    if replaced:
        row["flags"] = [f for f in row["flags"] if f not in replaces]
        row.setdefault("scoring_mixed_replaced", []).extend(replaced)
    return row


def fmt_ms_calls(r, prefix):
    """A time with its calls and spread beside it: '0.76 (median of 20 calls, 0.7..0.95 ms)'."""
    return fmt_ms(r.get(prefix + "_ms")) + SMALL.text({"calls": r.get(prefix + "_calls"),
                                                       "min_ms": r.get(prefix + "_min_ms"),
                                                       "max_ms": r.get(prefix + "_max_ms")})


def fmt_prior_ms(r, version):
    """One on-box prior column's time, with its calls and spread beside it when it was a median of calls."""
    p = (r.get("priors") or {}).get(version) or {}
    return fmt_ms(p.get("ms")) + SMALL.text({"calls": p.get("prior_calls"), "min_ms": p.get("prior_min_ms"),
                                             "max_ms": p.get("prior_max_ms")})


def fmt_ratio(v):
    return "%.2f" % v if number(v) else "-"


def fmt_q(q):
    if not q:
        return "-"
    judged = scored_quality(q)
    keys = sorted(judged) + sorted(k for k in q if k not in judged and number(q[k]) and not isinstance(q[k], bool))
    text = " ".join("%s=%.4g" % (k, q[k]) for k in keys[:3])
    if q.get(UNAVAILABLE):
        text = (text + " " if text else "") + "unavailable"
    return text or "-"


def other_prior_versions(res):
    """The on-box prior columns besides the baseline, in order."""
    base = (res.get("baseline_label") or "").split(" ")[0]
    return [p["version"] for p in res.get("prior_columns") or [] if p["version"] != base]


def render_md(res):
    v, new = res["vendor"], res.get("new_label") or ", ".join(res["new_versions"]) or "?"
    pv = other_prior_versions(res)
    base = res.get("baseline_label") or "%s (no on-box board)" % res["paper_version"]
    stored = {p["version"] for p in res.get("prior_columns") or [] if p.get("stored")}
    lines = ["# Paper measurements: %s%s against %s on %s" % (
        base, "".join(", %s (%s)" % (x, "stored, screen" if x in stored else "this box") for x in pv), new, v), ""]
    if res.get("baseline_stored"):
        # the FIRST line says the baseline is stored, with both sides' host, GPU, driver, kernel and date
        lines = ["**%s**" % res.get("verdict_basis"), ""] + ["**%s**" % w for w in res.get("baseline_warnings") or []] \
            + lines
    cand = res.get("candidate")
    if cand:
        lines.append("**Candidate build, not a release**: source %s, version %s; wheels %s." % (
            cand.get("source_commit"), cand.get("version"),
            ", ".join("%s sha256 %s" % (w.get("file"), w.get("sha256")) for _, w in sorted(cand["wheels"].items()))))
        lines.append("")
    for p in res.get("progress") or []:
        lines.append("- progress %s: %d/%d races done%s" % (p["arm"], p["done"], p["total"],
                                                            "; boot ids " + ", ".join(p["boot_ids"])
                                                            if p.get("boot_ids") else ""))
    lines.append("")
    lines.append("**Verdict basis**: %s. Times are compared only on the same box (AGENTS.md); every stored column "
                 "below is reference only and makes no flag." % res.get("verdict_basis"))
    lines.append("")
    lines.append("- candidate box: %s" % probe_line(res.get("host_probe")))
    for p in res.get("prior_columns") or []:
        lines.append("- %s box: %s%s" % (p["version"], probe_line(p.get("probe")),
                                         " (%s)" % p["why_not"] if p.get("why_not") else ""))
    lines.append("- stored %s column: the paper board, host %s" % (res["paper_version"],
                                                                   res.get("stored_host") or "not recorded"))
    if res.get("reference_label"):
        lines.append("- stored reference column: %s" % res["reference_label"])
    lines.append("")
    lines.append("Paper GPU: %s. New GPU: %s.%s" % (res["paper_gpu"], ", ".join(res["new_gpus"]) or "?",
                 " **Different hardware: timings are not comparable.**" if res["hw_diff"] else ""))
    t = res.get("thresholds") or {}
    lines.append("Ratio = new / %s median ms%s; REGRESSION above %s, FASTER below %s; quality judged by %s "
                 "against the paper record. Opponents are the paper's stored cells (never rerun, host %s); 'vs opp' is "
                 "ours over the paper's fastest paired opponent on that comparison's clock, shown, never a flag across "
                 "hosts." % (base, " (the stored board; BOX-STATE: the upload probe moved more than %d%%, ratio "
                             "not counted)" % round(100 * BOX_STATE_TOLERANCE) if res.get("baseline_stored")
                             else " on this box", t.get("regression_ratio_above"), t.get("faster_ratio_below"),
                             t.get("quality_policy"), res.get("stored_host") or "not recorded"))
    lines.append("")
    lines.append("Flags: " + ", ".join("%s %d" % kv for kv in sorted(res["flag_counts"].items())))
    lines.append("")
    pcols = "".join("%s ms | new/%s | " % (x, x) for x in pv)
    bl = res.get("best_ever_label")
    bcols = "%s ms | best-ever date, release | new/best | " % bl if bl else ""
    rl = res.get("reference_label")
    rcols = "reference ms | " if rl else ""
    lines.append("| recipe | %s ms | %snew ms | ratio | %sflags | stored %s ms | %squality %s | quality new | q | "
                 "best opponent (ms) | vs opp %s | vs opp new | note |" % (base, pcols, bcols, res["paper_version"],
                                                                         rcols, res["paper_version"],
                                                                         res["paper_version"]))
    lines.append("|---|---:|%s---:|---:|%s---|---:|%s---|---|---|---|---:|---:|---|" % (
        "---:|---:|" * len(pv), "---:|---|---:|" if bl else "", "---:|" if rl else ""))
    for r in res["rows"]:
        pvals = "".join("%s | %s | " % (fmt_prior_ms(r, x),
                                        fmt_ratio(((r.get("priors") or {}).get(x) or {}).get("ratio"))) for x in pv)
        bvals = best_ever_cells(r) if bl else ""
        rvals = "%s | " % fmt_ms(r.get("ref_prior_ms")) if rl else ""
        b = r.get("best_opponent") or {}
        note = []
        if r.get("why"):
            note.append(r["why"])
        if r.get("new_status") not in (None, "ok", "missing") and r.get("new_ms") is None:
            note.append("new: " + r["new_status"][:120])
        if r.get("old_ms") is None and r.get("old_status") != "ok":
            note.append("this box %s: %s" % (res["paper_version"], str(r["old_status"])[:120]))
        for x in pv:
            st = ((r.get("priors") or {}).get(x) or {}).get("status")
            if st and st != "ok" and ((r.get("priors") or {}).get(x) or {}).get("ms") is None:
                note.append("this box %s: %s" % (x, st[:120]))
        if r.get("ref_ms") is None and r.get("ref_status") != "ok":
            note.append("stored: " + str(r["ref_status"])[:120])
        if r.get("quality_failures"):
            note.append("quality: " + "; ".join(r["quality_failures"])[:200])
        if r.get("quality_missing"):
            note.append("quality missing: %s (%s)" % (",".join(r["quality_missing"]),
                                                      str(r.get("quality_unavailable") or "")[:160]))
        if r.get("params_diff"):
            note.append("params: " + ",".join(r["params_diff"][:6]))
        if r.get("best_ever_params_differ") or r.get("best_ever_hash_differs"):
            note.append("vs best ever: " + ",".join(r.get("best_ever_params_differ", [])[:6] +
                                                    (["hash"] if r.get("best_ever_hash_differs") else [])))
        if r.get("upload_probe"):
            note.append("upload probe " + probe_text(r["upload_probe"]))
        lines.append("| %s | %s | %s%s | %s | %s%s | %s | %s%s | %s | %s | %s | %s | %s | %s |" % (
            r["key"], fmt_ms_calls(r, "old"), pvals, fmt_ms_calls(r, "new"), fmt_ratio(r.get("ratio")), bvals,
            " ".join(r["flags"]), fmt_ms(r.get("ref_ms")), rvals, fmt_q(r.get("old_quality")),
            fmt_q(r.get("new_quality")), r.get("quality_verdict", "-"),
            "%s (%s)" % (b.get("arm"), fmt_ms(b.get("opponent_ms"))) if b else "-",
            fmt_ratio(b.get("ratio")), fmt_ratio(b.get("ratio_new")), "; ".join(note).replace("|", "/")))
    lines.append("")
    if bl:
        ex = res.get("extra_rows") or []
        allp = [p["version"] for p in res.get("prior_columns") or []]
        lines.append("## Rows outside the paper's recipes: %d (new against %s)" % (
            len(ex), ", ".join(["%s (this box)" % x for x in allp] + [bl + " (reference)"])))
        lines.append("")
        if res.get("extra_flag_counts"):
            lines.append("Flags: " + ", ".join("%s %d" % kv for kv in sorted(res["extra_flag_counts"].items())))
            lines.append("")
        xcols = "".join("%s ms | new/%s | " % (x, x) for x in allp)
        lines.append("| race | new ms | %s%s ms | best-ever date, release | new/best | flags | note |" % (xcols, bl))
        lines.append("|---|---:|%s---:|---|---:|---|---|" % ("---:|---:|" * len(allp)))
        for r in ex:
            note = ""
            if r.get("best_ever_params_differ") or r.get("best_ever_hash_differs"):
                note = "vs best ever: " + ",".join(r.get("best_ever_params_differ", [])[:6] +
                                                   (["hash"] if r.get("best_ever_hash_differs") else []))
            elif r.get("best_ever_ms") is None:
                note = "no earlier measurement"
            if r.get("upload_probe"):
                note = (note + "; " if note else "") + "upload probe " + probe_text(r["upload_probe"])
            xvals = "".join("%s | %s | " % (fmt_prior_ms(r, x),
                                            fmt_ratio(((r.get("priors") or {}).get(x) or {}).get("ratio")))
                            for x in allp)
            lines.append("| %s | %s | %s%s%s | %s |" % (r["key"], fmt_ms_calls(r, "new"), xvals, best_ever_cells(r),
                                                         " ".join(r["flags"]), note.replace("|", "/")))
        lines.append("")
    lines += render_internal_md(res)
    rl = rescore_lines(res)
    if rl:
        lines += ["## Waiting for an opponent re-score", ""] + [rl[0], ""] + ["- `%s`" % l.strip() for l in rl[1:]] + [""]
    return "\n".join(lines)


def render_internal_md(res):
    """The internal checks section (call overhead, not a race): new against our prior versions on this box only."""
    rows = res.get("internal_rows") or []
    if not rows:
        return []
    allp = [p["version"] for p in res.get("prior_columns") or []]
    lines = ["## Internal checks, %s: %d (not a race; new against our own prior on this box only)" % (
        res.get("internal_label") or "call overhead", len(rows)), ""]
    if res.get("internal_flag_counts"):
        lines += ["Flags: " + ", ".join("%s %d" % kv for kv in sorted(res["internal_flag_counts"].items())), ""]
    xcols = "".join("%s ms | new/%s | " % (x, x) for x in allp)
    lines.append("| row | new ms | %sflags |" % xcols)
    lines.append("|---|---:|%s---|" % ("---:|---:|" * len(allp)))
    for r in rows:
        xvals = "".join("%s | %s | " % (fmt_prior_ms(r, x), fmt_ratio(((r.get("priors") or {}).get(x) or {}).get("ratio")))
                        for x in allp)
        lines.append("| %s | %s | %s%s |" % (r["key"], fmt_ms_calls(r, "new"), xvals, " ".join(r["flags"]) or "-"))
    lines.append("")
    return lines


def best_ever_cells(r):
    """'best ms | date, release | new/best | ' of one row."""
    when = ", ".join(str(x) for x in (r.get("best_ever_date"), r.get("best_ever_release")) if x) or "-"
    return "%s | %s | %s | " % (fmt_ms(r.get("best_ever_ms")), when.replace("|", "/"),
                                fmt_ratio(r.get("ratio_vs_best_ever")))


def rescore_lines(res):
    """The one list of races waiting for an opponent re-score (opponent_rescore_rows), for the summary and the md."""
    owed = res.get("opponent_rescore_owed") or []
    if not owed:
        return []
    return (["waiting for an opponent re-score (%d races: the harness clock or settings key changed on all arms; "
             "their stored opponent cells are BROKEN until the opponent job scores them again; our times are shown "
             "above):" % len(owed)]
            + ["  %s (%s)" % (o["race_id"], o["harness_clock"]) for o in owed])


def render_summary(res, top=8):
    rows = res["rows"]
    label = res.get("new_label") or ",".join(res["new_versions"]) or "?"
    # the FIRST line is the verdict basis, in words (fb-harness, 2026-11-17): a compare without a verdict says why
    # before anything else, and every row below it is reference only
    basis = str(res.get("verdict_basis"))
    if res.get("baseline_stored"):
        out = [basis] + list(res.get("baseline_warnings") or [])
    else:
        out = [("NO VERDICT, every row is reference only: " + basis[len("no verdict: "):])
               if basis.startswith("no verdict: ") else "verdict: " + basis]
    out.append("PAPER-COMPARE vendor=%s %s->%s%s recipes=%d gpu %s -> %s%s" % (
        res["vendor"], res["paper_version"], (res["prior_label"] + "->") if res.get("prior_label") else "",
        label.replace(" ", "_"), res["recipes"],
        res["paper_gpu"], ",".join(res["new_gpus"]) or "?", " HW-DIFF" if res["hw_diff"] else ""))
    # done/total per arm (a job over its cap resumes; a board cut short is a partial column) and its boot ids
    if res.get("progress"):
        out.append("progress: " + "; ".join("%s %d/%d races done%s" % (
            p["arm"], p["done"], p["total"],
            " (boot ids %s: resumed after a restart)" % ", ".join(p["boot_ids"]) if len(p.get("boot_ids") or []) > 1
            else "") for p in res["progress"]))
    # the verdict is printed with BOTH arms' probes beside it: a verdict without the probe is not a verdict (AGENTS.md)
    out.append("  probe candidate: " + probe_line(res.get("host_probe")))
    for p in res.get("prior_columns") or []:
        out.append("  probe %s: %s%s" % (p["version"], probe_line(p.get("probe")),
                                         " [%s]" % p["why_not"] if p.get("why_not") else ""))
    out.append("  stored %s (reference, no flags): host %s%s" % (
        res["paper_version"], res.get("stored_host") or "not recorded",
        "; %s" % res["reference_label"] if res.get("reference_label") else ""))
    out.append("flags: " + " ".join("%s=%d" % kv for kv in sorted(res["flag_counts"].items())))
    if res.get("baseline_stored"):
        bs = box_state_rows(res)
        checked = sum(1 for k in ("rows", "extra_rows", "internal_rows") for r in res.get(k) or []
                      if r.get("upload_probe"))
        out.append("BOX-STATE %d rows (the in-race upload probe differs from the stored run's by more than %d%%; "
                   "ratio shown, not counted as REGRESSION or FASTER); %d rows had a probe on both sides" % (
                       len(bs), round(100 * (res.get("box_state_tolerance") or BOX_STATE_TOLERANCE)), checked))
        out += ["  %s %s; probe %s" % (fmt_ratio(r.get("ratio") if r.get("ratio") is not None else
                                                 ((r.get("priors") or {}).get(res["paper_version"]) or {}).get("ratio")),
                                       r["key"], probe_text(r.get("upload_probe"))) for r in bs[:top]]
    for flag, what in (("HARNESS-CLOCK", "timed under another harness-clock revision than the baseline; FASTER and "
                                         "SAME not counted"),
                       ("HOST-STALL", "system time above half the wall in the scored calls; not counted as "
                                      "REGRESSION"),
                       ("NUMA-MIXED", "the two workers ran unbound on different NUMA nodes; not counted as "
                                      "REGRESSION or FASTER")):
        marked = marked_rows(res, flag)
        if marked:
            out.append("%s %d rows (%s)" % (flag, len(marked), what))
            out += ["  %s %s%s%s" % (fmt_ratio(r.get("ratio")), r["key"],
                                     "; sys %.0f%% of wall, %s minor faults" % (
                                         100 * (r["host_usage"].get("stime_fraction") or 0),
                                         r["host_usage"].get("minflt")) if r.get("host_usage") else "",
                                     "; worker node %s" % (r.get("worker_numa") or {}).get("node_now")
                                     if r.get("worker_numa") else "") for r in marked[:top]]
    timed_rows = [r for r in rows if number(r.get("ratio")) and "BOX-STATE" not in r["flags"]
                  and "HARNESS-CLOCK" not in r["flags"] and "HOST-STALL" not in r["flags"]
                  and "NUMA-MIXED" not in r["flags"]]
    if timed_rows:
        rs = sorted(r["ratio"] for r in timed_rows)
        geo = 1.0
        for x in rs:
            geo *= x
        out.append("timed both=%d median ratio=%.2f geomean=%.2f" % (len(rs), rs[len(rs) // 2],
                                                                      geo ** (1.0 / len(rs))))
    worst = sorted([r for r in timed_rows if "REGRESSION" in r["flags"]], key=lambda r: -r["ratio"])[:top]
    if worst:
        out.append("worst regressions:")
        out += ["  %.2fx %s (%s -> %s ms)" % (r["ratio"], r["key"], fmt_ms_calls(r, "old"), fmt_ms_calls(r, "new"))
                for r in worst]
    best = sorted([r for r in timed_rows if "FASTER" in r["flags"]], key=lambda r: r["ratio"])[:top]
    if best:
        out.append("biggest gains:")
        out += ["  %.2fx %s (%s -> %s ms)" % (r["ratio"], r["key"], fmt_ms_calls(r, "old"), fmt_ms_calls(r, "new"))
                for r in best]
    if res.get("prior_label"):
        worse = sorted([r for r in rows if "REGRESSION-VS-PRIOR" in r["flags"]], key=lambda r: -r["ratio_vs_prior"])
        if worse:
            out.append("worse than %s:" % res["prior_label"])
            out += ["  %.2fx %s (%s -> %s ms)" % (r["ratio_vs_prior"], r["key"], fmt_ms(r["prior_ms"]),
                                                  fmt_ms(r["new_ms"])) for r in worse[:top]]
    if res.get("best_ever_label"):
        ex = res.get("extra_rows") or []
        out.append("%s: extra rows outside the recipes=%d; extra flags: %s" % (
            res["best_ever_label"], len(ex),
            " ".join("%s=%d" % kv for kv in sorted((res.get("extra_flag_counts") or {}).items())) or "-"))
        for label, pool in (("paper rows", rows), ("extra rows", ex)):
            hit = sorted([r for r in pool if "BEST-EVER-REGRESSION" in r["flags"]],
                         key=lambda r: -r["ratio_vs_best_ever"])
            if hit:
                out.append("worse than best ever, same params and hash (%s, %d):" % (label, len(hit)))
                out += ["  %.2fx %s (best %s ms %s %s -> %s ms)" % (
                    r["ratio_vs_best_ever"], r["key"], fmt_ms(r["best_ever_ms"]), r.get("best_ever_release"),
                    r.get("best_ever_date"), fmt_ms(r["new_ms"])) for r in hit[:top]]
    for flag in ("RECOVERED", "STILL-REGRESSED", "FIXED", "BROKEN", "BASELINE-NOT-RUN", "NOT-COMPARED",
                 "QUALITY-DROP", "QUALITY-CHANGE",
                 "QUALITY-MISSING", "SCORING-MIXED", "NOW-WINS",
                 "NOW-LOSES", "PARAMS-DIFF", "NOT-RUN", "BEST-EVER-REGRESSION-UNCHECKED", "PARAMS-DIFFER"):
        hit = [r for r in rows if flag in r["flags"]]
        if hit:
            out.append("%s (%d): %s%s" % (flag, len(hit), ", ".join(r["key"] for r in hit[:6]),
                                          " ..." if len(hit) > 6 else ""))
    out += rescore_lines(res)
    internal = res.get("internal_rows") or []
    if internal:
        out.append("internal checks (%s; not a race, our prior on this box only): %d; flags: %s" % (
            res.get("internal_label") or "call overhead", len(internal),
            " ".join("%s=%d" % kv for kv in sorted((res.get("internal_flag_counts") or {}).items())) or "-"))
        for r in internal:
            pri = "; ".join("%s %s ms new/%s %s" % (v, fmt_prior_ms(r, v), v, fmt_ratio(x.get("ratio")))
                            for v, x in sorted((r.get("priors") or {}).items()))
            out.append("  %s new %s ms%s%s" % (r["key"], fmt_ms_calls(r, "new"), ("; " + pri) if pri else "",
                                                (" " + " ".join(r["flags"])) if r["flags"] else ""))
    return "\n".join(out)


def cmd_compare(args):
    doc = load_json(args.recipes)
    boards = []
    for p in args.board:
        try:
            boards.append(load_json(p))
        except (OSError, ValueError) as exc:
            print("PAPER-COMPARE unreadable board %s: %s" % (p, exc), file=sys.stderr)
    why = {}
    for p in args.not_run or []:
        try:
            why.update({k: v for k, v in load_json(p).items() if isinstance(v, str)})
        except (OSError, ValueError):
            pass
    priors = []
    for spec in args.prior_board or []:
        version, path = parse_prior_board(spec)
        try:
            priors.append((version, load_json(path)))
        except (OSError, ValueError) as exc:
            # a prior that never ran (its install or every smoke failed) is a missing column, not a crash
            print("PAPER-COMPARE unreadable prior board %s: %s (column NOT-RUN)" % (path, exc), file=sys.stderr)
            priors.append((version, {"races": {}}))
    ref, rlabel = (None, None)
    if args.reference_compare:
        ref, rlabel = load_reference(args.reference_compare)
    best = None
    if args.best_ever:
        best = BE.load(os.path.join(REPO, "bench", "results", "board_best_ever", "%s.json" % doc["vendor"])
                       if args.best_ever == "auto" else args.best_ever)
    stored = None
    if args.stored_baseline_board:
        stored = load_json(args.stored_baseline_board)     # unreadable: the compare fails (no silent NOT-RUN column)
    res = compare(doc, boards, why, priors, ref, rlabel, best, stored, args.verdict_prior)
    v = doc["vendor"]
    md = args.out_md or os.path.join(OUT_DIR, "COMPARE-%s.md" % v)
    js = args.out_json or os.path.join(os.path.dirname(os.path.abspath(md)), "compare-%s.json" % v)
    summary = render_summary(res)
    for path, text in ((md, render_md(res)), (js, json.dumps(res, indent=1, sort_keys=True) + "\n"),
                       (args.summary or os.path.join(os.path.dirname(os.path.abspath(md)), "SUMMARY-%s.txt" % v),
                        summary + "\n")):
        write_atomic(path, text)
    print(summary)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="paper_compare", description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recipes", help="extract the paper's recipe set of one vendor")
    r.add_argument("--vendor", required=True, choices=VENDORS)
    r.add_argument("--board", default=None, help="the paper board (default the paper-tables board of the vendor)")
    r.add_argument("--version", default=PAPER_VERSION)
    r.add_argument("--out", default=None)
    i = sub.add_parser("ids", help="(box) the recipe races the current board plans")
    i.add_argument("--recipes", required=True)
    i.add_argument("--planner", default=os.path.join(HERE, "bench_board.py"))
    i.add_argument("--out-dir", required=True)
    a = sub.add_parser("ids-all", help="(box) every race the board plans: the full-board screen")
    a.add_argument("--vendor", required=True, choices=VENDORS)
    a.add_argument("--planner", default=os.path.join(HERE, "bench_board.py"))
    a.add_argument("--out-dir", required=True)
    s = sub.add_parser("pass-ids", help="(box) the planned races whose smoke passed")
    s.add_argument("--full-ids", required=True)
    s.add_argument("--smoke-json", action="append", default=[])
    s.add_argument("--out", required=True)
    lf = sub.add_parser("left", help="(box) how many of --ids are not yet done in --board (the disk gate's count)")
    lf.add_argument("--ids", required=True)
    lf.add_argument("--board", required=True, help="board.json (absent: every id is left)")
    c = sub.add_parser("compare", help="the paper's numbers against a new board")
    c.add_argument("--recipes", required=True)
    c.add_argument("--board", action="append", default=[], required=True)
    c.add_argument("--not-run", action="append", default=[],
                   help="json {race id: why} (not-planned.json, smoke-failed.json) for NOT-RUN rows")
    c.add_argument("--prior-board", action="append", default=[], metavar="VERSION=BOARD.json",
                   help="OUR prior wheel of VERSION re-timed in the same job on the same box (board-prior-<v>/"
                        "board.json); repeat. The paper's version is the baseline (REGRESSION / FASTER / FIXED / "
                        "BROKEN), the last other one the prior column (REGRESSION-VS-PRIOR, RECOVERED, "
                        "STILL-REGRESSED). A board from another host refuses")
    c.add_argument("--reference-compare", default=None,
                   help="a stored compare JSON of another run (bench/results/paper_compare/compare-<vendor>.json): "
                        "a reference column with its host, never a flag")
    c.add_argument("--best-ever", nargs="?", const="auto", default=None, metavar="FILE",
                   help="the best-ever column (tools/board_best_ever.py mine; no FILE: "
                        "bench/results/board_best_ever/<vendor>.json): BEST-EVER-REGRESSION, PARAMS-DIFFER, and a "
                        "row for every new cell outside the recipes")
    c.add_argument("--stored-baseline-board", default=None, metavar="BOARD.json",
                   help="a STORED board of the paper's version from another job (the 2026-11-17 direction: 0.8.55 is "
                        "not timed again): the baseline in place of an on-box --prior-board; host and driver "
                        "differences are WARNING lines, and a row whose in-race upload probe moved more than 15%% "
                        "from the stored run's is BOX-STATE")
    c.add_argument("--verdict-prior", default=None, metavar="NAME",
                   help="the --prior-board NAME that is the verdict baseline in place of the paper's version: one of "
                        "our own earlier built wheels timed in this job on this box (paper_compare_job.sh "
                        "PC_PRIOR_WHEELS); REGRESSION / FASTER / SAME come from that pair only, and a "
                        "--stored-baseline-board is then a screen column (shown, never a flag)")
    c.add_argument("--out-md", default=None)
    c.add_argument("--out-json", default=None)
    c.add_argument("--summary", default=None)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return {"recipes": cmd_recipes, "ids": cmd_ids, "ids-all": cmd_ids_all, "pass-ids": cmd_pass_ids,
            "left": cmd_left, "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
