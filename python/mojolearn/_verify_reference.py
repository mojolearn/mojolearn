# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The reference hash table `python -m mojolearn verify --all` compares against.

WHAT IT IS. One small JSON shipped in the wheel at
`mojolearn/verify_reference/table.json`. For every identity_break cell
(`<lane>/<fixture>`) and every part (`train`, `infer`, `model`, `batch`,
`stepfull`) it holds the hash the committed records agree on, and which
record each device class (apple, nvidia, amd, cpu) got it from, with that
record's directory and commit. No hash in it is typed: `build_table` reads the JSON columns
`tools/identity_break.py --json` wrote under `bench/results/identity_break/`.

WHICH RECORDS COUNT. A column is admitted only if it is identical mode, names
a commit, ran the default fixture size on one device, is not a sabotage run
(by its JSON flags and by its file name), is not a partial or an unfixed
before-picture, did not declare itself a PARTIAL COLUMN (`partial_column`,
written by a harness run that was told to leave parts out, 2026-09-20), and
its fixture hashes equal the ones this harness generates now. A column whose fixtures differ hashed different input bytes, so its
cells say nothing about the current lanes. The same holds per lane for the
harness's `LANE_REVISIONS` (a lane whose input changed, 2026-09-15): a column
whose revision of that lane is not the harness's contributes no cell there,
so the cell reads OWED until a record at the new revision is committed.

WHICH HASH WINS. Lanes change: a fix moves a hash and only some vendors are
re-recorded before the next release record. So per cell, part and device
class the NEWEST admitted column wins (commit time, then record directory,
then file name). The cell's reference is the newest of those. A class whose
newest value differs from it at an OLDER commit is kept as `superseded`
(the record moved on and that class was not re-recorded); a class that
differs at the SAME commit is a disagreement inside the record itself, and
the cell has no reference (`conflict`), so a user box reads OWED there and
is never blamed for it.

FORMAT (`mojolearn.verify-reference.v1`)

    records  [{dir, file, vendor, class, commit, commit_time}]
    absent_parts {part: {reason: count}} over the admitted columns
    fixtures {fixture: {X, y_clf, y_reg}}, heldout {fixture: {X}}
    cells    {"<lane>/<fixture>": {part: {"ref": value | null,
                                           "cols": {class: record index
                                                    | [record index, value]},
                                           "conflict": true (only when set)}}}

`ref` is a 16-hex hash or an `n/a:<reason>` string. A `cols` entry that is a
bare index agrees with `ref`; a pair is a superseded value.

This module imports nothing from the package at module level, so the table
can be built and read on a machine with no binding built.
"""
import hashlib
import copy
import json
import os
import re
import subprocess

FORMAT = "mojolearn.verify-reference.v1"
TABLE_DIR = "verify_reference"
TABLE_NAME = "table.json"
#: THE PARTS A USER CAN CHECK. `stepfull` joined the four on 2026-09-16
#: (lane/expose-stepfull). It asserts that a sequence decoded ONE TOKEN AT A
#: TIME with a carried state is, at every position, the bits the same model
#: answers when the whole sequence runs as ONE fresh-state forward pass. That
#: is the property incremental decoding rests on and the place bitwise
#: determinism usually breaks, and it was proved on four columns for all
#: eight decode lanes before it was exposed here. Adding a part to this tuple
#: is not free: no row of an EXISTING table carries a value for it, so the
#: change only lands together with a regenerated table.
PARTS = ("train", "infer", "model", "batch", "stepfull")
OPTIONAL_PARTS = ("batchgrad", "batchscale", "ragged", "rlpair")
CLASSES = ("apple", "nvidia", "amd", "cpu")

#: what `mojolearn.vendor()` reads back, as a device class of the table
VENDOR_CLASS = {"metal": "apple", "cuda": "nvidia", "hip": "amd", "cpu": "cpu"}

_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
#: Tokens of runs that are not evidence of the claim, matched against the
#: WHOLE PATH because each marks a whole DIRECTORY of such runs:
#: `2026-09-14_kmeans-sqrt-fix/unfixed/` holds columns taken with the bug still
#: present, and admitting those would feed known-wrong hashes into the table.
_EXCLUDED_PATH_TOKENS = ("partial", "probe", "unfixed", "post-merge-smoke")
#: `sabotage` is matched against the FILE NAME ALONE (2026-09-16,
#: lane/sabotage-evidence). A negative control is recorded BESIDE the clean
#: column it is a control for, in one record directory, and naming that
#: directory after what it records is the obvious thing to do. Matching the
#: whole path therefore refused clean columns for their neighbor's sin, and
#: refused them SILENTLY: two committed ones,
#: `2026-09-15_ties-sabotage/x86-runpod/cpu-x86.json` and
#: `2026-09-15_metrics-sabotage-coverage/cpu-prod.json`, were being discarded
#: with no error to read. A sabotage build also declares itself in its own
#: metadata, `host.families[*].sabotage` and the `*_sabotage` flags checked
#: below, which is the stronger test and still refuses it. Measured over the
#: 495 committed columns: 220 admitted before, 222 after, NOTHING newly
#: refused, and no column carrying a sabotage signal admitted
#: (`tests/test_verify_reference_admit.py`).
_EXCLUDED_BASENAME_TOKENS = ("sabotage",)
#: the whole vocabulary, for readers and for anything that wants to report it
_EXCLUDED_NAME_TOKENS = _EXCLUDED_PATH_TOKENS + _EXCLUDED_BASENAME_TOKENS
#: n/a values that describe the RUN, not the estimator
_SKIPPED_NA = ("n/a:skipped", "n/a:UNDECLARED")


class TableError(Exception):
    """The table is missing, unreadable or not this format. Maps to exit 5."""


def table_path(pkg_dir=None):
    pkg_dir = pkg_dir or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(pkg_dir, TABLE_DIR, TABLE_NAME)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------

def load_table(path=None):
    """The parsed table. Raises TableError naming what is wrong."""
    path = path or table_path()
    if not os.path.isfile(path):
        raise TableError(f"no reference table at {path}; this install ships none")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            table = json.load(fh)
    except (OSError, ValueError) as exc:
        raise TableError(f"the reference table at {path} is unreadable: {exc}")
    if not isinstance(table, dict) or table.get("format") != FORMAT:
        raise TableError(f"the reference table at {path} is not {FORMAT} "
                         f"(format {table.get('format') if isinstance(table, dict) else type(table).__name__!r})")
    for key in ("records", "cells", "fixtures", "heldout"):
        if key not in table:
            raise TableError(f"the reference table at {path} has no {key!r}")
    return table


def entry(table, lane, fixture, part, device_class=None):
    """The table entry of one cell part, or None when no record has it."""
    cell = table["cells"].get(f"{lane}/{fixture}")
    ent = None if cell is None else cell.get(part)
    if not ent or part != "model" or device_class not in ("cpu", "apple", "nvidia", "amd"):
        return ent
    # These CPU lanes load the GPU's saved file; they do not write a model.
    # A newer CPU N/A record must not erase a GPU's model-byte reference.
    # Keep the two roles separate, without treating disagreeing GPU hashes
    # as interchangeable or granting an unrecorded CPU exemption.
    from .host_surface import GBDT_CTR_MODEL_LANES
    if lane not in GBDT_CTR_MODEL_LANES:
        return ent
    values = {cls: (column, ent.get("ref")) if isinstance(column, int)
              else tuple(column) for cls, column in ent.get("cols", {}).items()}
    cpu = values.get("cpu")
    cpu_na = "n/a:gpu-saved-file (a CPU column loads the GPU column's model and writes none)"
    if not cpu or cpu[1] != cpu_na:
        return ent
    if device_class == "cpu":
        return dict(ref=cpu_na, cols={"cpu": cpu[0]})
    gpu = {cls: value for cls, value in values.items() if cls in ("apple", "nvidia", "amd")}
    hashes = {value for _, value in gpu.values()}
    if not gpu or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{16}", value)
                      for value in hashes):
        return ent
    if len(hashes) != 1:
        return dict(ref=None, conflict=True, cols={cls: list(value) for cls, value in gpu.items()})
    return dict(ref=next(iter(hashes)), cols={cls: index for cls, (index, _) in gpu.items()})


def columns_of(table, ent):
    """{class: {record, commit, agrees, value}} for one entry, for the report."""
    out = {}
    if not ent:
        return out
    for cls, ref in sorted(ent.get("cols", {}).items()):
        idx, value = (ref, None) if isinstance(ref, int) else (ref[0], ref[1])
        rec = table["records"][idx]
        out[cls] = dict(record=f"{rec['dir']}/{rec['file']}", vendor=rec["vendor"],
                        commit=rec["commit"], agrees=value is None,
                        **({} if value is None else dict(value=value)))
    return out


# --------------------------------------------------------------------------
# judging one cell part
# --------------------------------------------------------------------------

IDENTICAL = "IDENTICAL"
DIVERGENT = "DIVERGENT"
OWED = "OWED"
REFUSED = "REFUSED"
NA = "N/A"
STATES = (IDENTICAL, DIVERGENT, OWED, REFUSED, NA)


def judge(value, ent, error=None):
    """(state, detail) for one part of one cell on this box.

    `value` is what this box produced after its repeats: a 16-hex hash, an
    `n/a:<reason>` string, `MOVED` (the box disagreed with itself), a
    `BATCH_MOVED:...` or `RELOAD-MOVED` verdict, or None where the lane or the
    probe raised (then `error` is the sentence it raised with)."""
    if value is None:
        return REFUSED, error or "raised"
    if not isinstance(value, str):
        return REFUSED, "invalid computed value: expected a digest or explicit verdict"
    if value == "MOVED":
        return DIVERGENT, "this box gave two different hashes for the same fit (MOVED)"
    if value.startswith("BATCH_MOVED"):
        return DIVERGENT, "batch invariance failed on this box: " + value[:300]
    if value.startswith("RLPAIR_MOVED"):
        return DIVERGENT, "sampler/replay or continuous batching failed on this box: " + value[:300]
    if value.startswith("RELOAD-MOVED"):
        return DIVERGENT, "the saved model predicts differently from the model in memory"
    if error:
        return REFUSED, error
    if not re.fullmatch(r"[0-9a-f]{16}", value) and not (value.startswith("n/a:") and len(value) > 4):
        return REFUSED, "invalid computed value: expected a 16-hex digest or n/a reason"
    if ent is None:
        if value.startswith("n/a"):
            return NA, value
        return OWED, "no committed record carries this cell part yet"
    if ent.get("conflict"):
        return OWED, "the committed columns disagree with each other at one commit; no reference"
    ref = ent.get("ref")
    if value.startswith("n/a") and isinstance(ref, str) and ref.startswith("n/a"):
        return NA, value
    if value.startswith("n/a") != (isinstance(ref, str) and ref.startswith("n/a")):
        # A hash against an n/a, either way round, is the probe changing
        # between the record and this harness (a lane that gained a batch
        # declaration or a held-out probe), not arithmetic; the harness's
        # own --diff reads the pair ONE-COLUMN, never DIVERGENT.
        return OWED, f"the record carries {ref} and this harness {value}; the part changed after the record"
    if value == ref:
        return IDENTICAL, ""
    return DIVERGENT, f"this box {value}, reference {ref}"


# --------------------------------------------------------------------------
# building (maintainer path)
# --------------------------------------------------------------------------

def device_class(vendor, filename):
    """apple, nvidia, amd or cpu for a column, from its vendor label and its
    file name (one 2026-09-14 Apple column recorded the vendor `arm64`)."""
    v = (vendor or "").lower()
    base = os.path.basename(filename).lower()
    if v.startswith("cpu") or base.startswith("cpu"):
        return "cpu"
    for cls in ("apple", "nvidia", "amd"):
        # METAL IS APPLE'S GPU API AND NOTHING ELSE'S (2026-09-19). The
        # `arm64` escape below required "apple" in the FILE NAME, so a column
        # named `metal.new-base.json` recording `vendor: "arm64"` -- which is
        # what a Metal round writes when the GPU label does not reach the
        # recorder -- classified as None and counted for NOTHING. Two admitted
        # arima-exog cells sat unread in
        # bench/results/identity_break/2026-09-15_arima-exog/metal/ for four
        # days because of it. A file cannot be named `metal` and be any other
        # vendor.
        if (cls in v or cls in base
                or (cls == "apple" and v == "arm64"
                    and ("apple" in base or "metal" in base))):
            return cls
    return None


def _commit_time(root, commit, cache):
    if commit in cache:
        return cache[commit]
    t = 0
    try:
        out = subprocess.run(["git", "-C", root, "log", "-1", "--format=%ct", commit],
                             capture_output=True, text=True, timeout=20)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            t = int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    cache[commit] = t
    return t


def admit(j, path, par_axis=False):
    """None when the column is admissible, else the reason it is not.

    `par_axis=True` admits a column recorded on MORE THAN ONE DEVICE, and is
    only ever correct for a `par-*` multi-device driver lane.

    WHY THIS MODE HAD TO EXIST (2026-09-19). A `par-*` lane's whole claim is
    that a TWO-DEVICE column hashes equal to the one-device column cell for
    cell. The default rule below refuses `par_devices != "0"`, which is right
    for every ORDINARY lane -- a two-device answer must never be read as the
    one-device reference. But applied to the drivers it produced a rule that
    could not be satisfied from either side:

      * a two-device column, the only run that can STATE their claim, was
        refused here and so was invisible to `gpu_coverage`;
      * and on ONE device every one of them is DEGENERATE -- measured,
        `lane_applicability.degenerate('apple-metal')` holds all 13 -- because
        comparing a one-device run against a one-device run is comparing a run
        to itself, and it passes whatever the code does.

    Stateable only on two devices, admissible only on one. No run on any
    hardware could ever discharge them, while `docs/VERIFICATION_MATRIX.md`
    listed them as thirteen gaps to go close. The evidence existed the whole
    time: bench/results/identity_break/2026-09-19_par-lane-amd-class/ carries
    117 cells at par_devices='0' and 117 at par_devices='0,1', off one build
    on one box at one commit, with IDENTICAL hashes on all 117.

    THE DEFAULT IS UNCHANGED AND STAYS STRICT. A caller must ask for this
    mode, and must only credit `par-*` lanes from the column it admits.
    """
    low = path.lower()
    # This particular incident directory was explicitly quarantined by its
    # contemporaneous README. Stable repetitions inside a faulty-device run
    # do not make its cells admissible. Keep the raw records for diagnosis;
    # do not infer a general exception for Apple or disagreeing columns.
    normalized = "/" + low.replace("\\", "/").lstrip("/")
    if "/2026-09-15_gp-sample-y/metal-transient/" in normalized:
        return "quarantined Metal incident (2026-09-15_gp-sample-y/README.md)"
    base = os.path.basename(low)
    if any(tok in low for tok in _EXCLUDED_PATH_TOKENS) or any(
            tok in base for tok in _EXCLUDED_BASENAME_TOKENS):
        return "sabotage, partial, probe, unfixed or smoke run (by name)"
    if not isinstance(j, dict) or not isinstance(j.get("cells"), dict):
        return "not an identity_break column"
    if j.get("complete") is False:
        return "incomplete identity_break checkpoint"
    # COMPLETE AND WHOLE ARE DIFFERENT QUESTIONS (2026-09-20,
    # lane/full-part-set-by-default). `complete` is a RUN-LEVEL flag: it says
    # the process reached the end, and says NOTHING about which parts the
    # column carries. From 2026-09-15 to 2026-09-20 four of the nine parts
    # were opt-in, and a four-part column and a nine-part column were both
    # `complete: true` and both admitted here, identically. Measured over the
    # committed corpus: 245 of the 559 admissible columns carry four parts,
    # 151 five, and only 15 all nine; 310 of 559 carry no `stepfull` at all,
    # though `stepfull` is in the mandatory PARTS tuple above.
    #
    # THE HARNESS NOW SAYS SO ABOUT ITSELF and this REFUSES it, rather than
    # admitting it with a marker. A partial column is by construction a local
    # loop: the caller typed `--partial-column` and named what to leave out,
    # so there is no case where one should feed the shipped table. Admitting
    # it with a marker would leave every downstream reader to remember to
    # check the marker, which is the same shape of defect one layer up.
    #
    # THIS REFUSES NOTHING THAT EXISTS. `partial_column` is written only by
    # runs after 2026-09-20; every committed column lacks the key, reads
    # falsey, and is admitted exactly as before. The corpus is not
    # retroactively invalidated -- which is also why a partial column cannot
    # be recognised by counting parts: an honest pre-2026-09-20 record and a
    # deliberately narrowed one carry the same four.
    if j.get("partial_column"):
        left_out = j.get("parts_omitted") or []
        return ("partial column: the run left out "
                + (", ".join(str(p) for p in left_out) if left_out else "parts it did not name"))
    if j.get("mode") != "identical":
        return f"mode {j.get('mode')!r}"
    if not _COMMIT.match(str(j.get("commit") or "")):
        return "no commit"
    if j.get("batch_sabotage") or j.get("rlpair_sabotage"):
        return "sabotage flag set"
    for key, val in j.items():
        if key.endswith("_sabotage") and val:
            return f"{key} set"
    pkg = j.get("package") or {}
    if pkg.get("fixture_n") or pkg.get("wide"):
        return "non-default fixture size or wide mode"
    if not par_axis and str(pkg.get("par_devices") or "0") != "0":
        return f"par_devices {pkg.get('par_devices')}"
    # A COLUMN PRODUCED DURING SOMEONE ELSE'S GPU RUN IS NOT EVIDENCE
    # (2026-09-19). Concurrent Metal on one M4 returns NaN, constant and zero
    # output that still hashes STABLY -- a wrong answer with no sabotage in
    # it, which is why four sharded processes produced 1049 phantom DIVERGENT
    # parts this morning and the same lanes run solo came back VERIFIED.
    # `identity_break` records `gpu_slot` and `mac_slot.py` marks its own
    # children, so the dangerous state names itself and is refused here.
    # ABSENCE IS FINE: every column recorded before today lacks the field,
    # and refusing those would discard the whole existing corpus over a
    # question they were never asked.
    if str(j.get("gpu_slot") or "").startswith("HELD BY ANOTHER RUN"):
        return "concurrent GPU work: " + str(j.get("gpu_slot"))
    host = j.get("host") or {}
    for fam in (host.get("families") or {}).values():
        if isinstance(fam, dict) and fam.get("sabotage"):
            return "a host binding reads back sabotage"
    if not par_axis and str(j.get("vendor", "")).endswith("-two"):
        return "two-device part"
    if j.get("heldout_seed") not in (None, 1):
        return f"heldout_seed {j.get('heldout_seed')}"
    return None


def _part_value(cell, part, min_repeats=1):
    """The one value a column carries for a part, or None when it carries no
    usable one (moved, refused, reload-moved, batch-moved, skipped).

    Historical provenance readers may accept one sample. New table admission
    explicitly requires two; reading an old table does not silently rewrite it.
    """
    if not isinstance(cell, dict):
        return None
    verdict = cell.get("verdict" if part == "train" else f"{part}_verdict")
    values = cell.get("hashes" if part == "train" else part)
    if not isinstance(values, list) or len(values) < min_repeats:
        return None
    value = values[0]
    if not isinstance(value, str) or any(v != value for v in values):
        return None
    if verdict == "STABLE" and re.fullmatch(r"[0-9a-f]{16}", value):
        return value
    if (part != "train" and verdict == "N/A" and value.startswith("n/a:")
            and len(value) > 4 and not value.startswith(_SKIPPED_NA)):
        return value
    return None


def build_table(record_paths, harness, repo_root, lanes=None, log=None, parts=None):
    """The table dict from the column JSONs at `record_paths`. `harness` is
    the imported identity_break module: its fixtures decide which columns
    hashed the current input bytes, and its LANES decide which cells are
    kept. `repo_root` is the checkout whose git history dates the commits."""
    parts = tuple(PARTS if parts is None else parts)
    unknown = set(parts) - set(PARTS + OPTIONAL_PARTS)
    if unknown:
        raise ValueError(f"unknown reference parts: {sorted(unknown)}")
    log = log or (lambda s: None)
    lanes = set(lanes if lanes is not None else harness.LANES)
    want_fix = {f: dict(X=harness._h(X), y_clf=harness._h(yc), y_reg=harness._h(yr))
                for f, (X, yc, yr) in ((f, harness.fixture(f)) for f in harness.FIXTURES)}
    want_held = {f: dict(X=harness._h(harness.heldout(f))) for f in harness.FIXTURES}
    records, cache = [], {}
    best = {}                      # (cell, part, class) -> (key, record idx, value)
    absent_total = {}              # part -> {reason: count}, over every admitted column
    for path in sorted(record_paths):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                j = json.load(fh)
        except (OSError, ValueError) as exc:
            log(f"skip {path}: unreadable ({exc})")
            continue
        why = admit(j, path)
        if why:
            log(f"skip {path}: {why}")
            continue
        cls = device_class(j.get("vendor"), path)
        if cls is None:
            log(f"skip {path}: no device class for vendor {j.get('vendor')!r}")
            continue
        fx = j.get("fixtures") or {}
        bad = [f for f, h in fx.items() if want_fix.get(f) != h]
        held = j.get("heldout") or {}
        bad += [f"heldout {f}" for f, h in held.items() if want_held.get(f) != h]
        if bad:
            log(f"skip {path}: fixture bytes differ from this harness ({', '.join(bad[:4])})")
            continue
        rel_dir = os.path.relpath(os.path.dirname(path), os.path.join(repo_root, "bench", "results", "identity_break"))
        commit = j["commit"]
        rec = dict(dir=rel_dir, file=os.path.basename(path), vendor=j.get("vendor"), **{"class": cls},
                   commit=commit, commit_time=_commit_time(repo_root, commit, cache))
        idx = len(records)
        records.append(rec)
        key = (rec["commit_time"], rec["dir"], rec["file"])
        kept = 0
        # AN ABSENT PART WAS INVISIBLE HERE (2026-09-20). Every rejection
        # below used to be a bare `continue`: no log line, no flag, no count.
        # The ONLY signal a reader got was a smaller N in `use <path>: ... N
        # cell parts`, a number with no denominator to compare it against, so
        # a column missing four of its nine parts and a whole one produced
        # the same shape of line and nobody could tell which was which
        # without reopening the JSON. `asked` is the denominator and `absent`
        # says why each slot is not in `kept`.
        asked = 0
        absent = {}          # part -> {reason: count}

        def _absent(part, why):
            absent.setdefault(part, {})
            absent[part][why] = absent[part].get(why, 0) + 1

        revs = getattr(harness, "LANE_REVISIONS", {})
        have_revs = j.get("lane_revisions") or {}
        for cell_key, cell in j["cells"].items():
            lane, _, fixture = cell_key.partition("/")
            if lane not in lanes or fixture not in want_fix or not isinstance(cell, dict):
                continue
            # Absence is not agreement: a cell must carry its own input witness.
            if fx.get(fixture) != want_fix[fixture]:
                continue
            if lane in revs and have_revs.get(lane) != revs[lane]:
                continue
            for part in parts:
                asked += 1
                if part != "train" and held.get(fixture) != want_held[fixture]:
                    _absent(part, "held-out bytes differ")
                    continue
                if part in OPTIONAL_PARTS:
                    expected = (harness._rlpair_protocol() if part == "rlpair" else
                                harness._part_protocol(part, harness.BATCH_ALONE))
                    # A different split/length protocol is a different claim.
                    if j.get(f"{part}_protocol") != expected:
                        _absent(part, "not run" if j.get(f"{part}_protocol") is None
                                else "protocol differs")
                        continue
                value = _part_value(cell, part, min_repeats=2)
                if value is None:
                    # NOT RUN and RAN BUT UNUSABLE are different facts. The
                    # first is a hole in the column; the second is a moved,
                    # refused, skipped or single-repeat cell that the column
                    # did collect. Reading both as "absent" is how a
                    # four-part column passed for a nine-part one.
                    ran = ("verdict" if part == "train" else f"{part}_verdict") in cell
                    _absent(part, "not usable (moved, refused, skipped or one repeat)"
                            if ran else "not run")
                    continue
                if not value.startswith("n/a:") and part in ("batch", "stepfull"):
                    expected = (dict(alone=harness.BATCH_ALONE, split=list(harness.BATCH_SPLIT) + ["n"],
                                     prefix="1,7,full-1", enabled=True) if part == "batch" else
                                harness._part_protocol(part, harness.BATCH_ALONE))
                    if j.get(f"{part}_protocol") != expected:
                        _absent(part, "not run" if j.get(f"{part}_protocol") is None
                                else "protocol differs")
                        continue
                slot = (cell_key, part, cls)
                if slot not in best or key > best[slot][0]:
                    best[slot] = (key, idx, value)
                kept += 1
        line = f"use  {path}: class {cls}, commit {commit[:9]}, {kept} of {asked} cell parts"
        if absent:
            line += "; absent " + ", ".join(
                f"{part} x{sum(why.values())} ({'; '.join(sorted(why))})"
                for part, why in sorted(absent.items()))
        log(line)
        for part, why in absent.items():
            for reason, n in why.items():
                absent_total.setdefault(part, {})
                absent_total[part][reason] = absent_total[part].get(reason, 0) + n
    # THE TOTAL, NOT JUST THE PER-COLUMN LINES. One column short of a part
    # reads as a detail; every column short of the same part is the defect.
    for part in sorted(absent_total):
        why = absent_total[part]
        log(f"absent {part}: {sum(why.values())} cell parts over {len(records)} admitted columns ("
            + ", ".join(f"{reason} x{n}" for reason, n in sorted(why.items())) + ")")
    # only records a winning value points at are kept, renumbered
    cells = {}
    grouped = {}
    for (cell_key, part, cls), won in best.items():
        grouped.setdefault((cell_key, part), {})[cls] = won
    used = sorted({won[1] for g in grouped.values() for won in g.values()})
    renum = {old: new for new, old in enumerate(used)}
    for (cell_key, part), by_cls in sorted(grouped.items()):
        newest_cls = max(by_cls, key=lambda c: by_cls[c][0])
        n_key, _, ref = by_cls[newest_cls]
        ent = dict(ref=ref, cols={})
        for cls in sorted(by_cls):
            k, idx, value = by_cls[cls]
            if value == ref:
                ent["cols"][cls] = renum[idx]
            elif k[0] == n_key[0] and records[idx]["commit"] == records[by_cls[newest_cls][1]]["commit"]:
                ent["conflict"] = True
                ent["cols"][cls] = [renum[idx], value]
            else:
                ent["cols"][cls] = [renum[idx], value]
        if ent.get("conflict"):
            ent["ref"] = None
            ent["cols"][newest_cls] = [renum[by_cls[newest_cls][1]], ref]
        cells.setdefault(cell_key, {})[part] = ent
    return dict(
        format=FORMAT,
        generated_by="python -m mojolearn verify --all --emit-reference",
        harness_sha256=sha256_file(harness.__file__),
        #: THE LANE REVISIONS THIS TABLE WAS GENERATED AGAINST (2026-09-16,
        #: lane/identity-fixtures-light). A lane whose fixture moves gets a new
        #: LANE_REVISIONS entry in the harness; recording the revisions here is
        #: what lets `stale_reference_lanes` below detect, mechanically, that a
        #: reference predates the input it is supposed to describe. A table
        #: generated before this key existed carries none, which reads as
        #: "unknown" and therefore stale for any lane that has a revision.
        admission_policy=dict(min_repeats=2, input_witness_required=True, property_protocol_required=True),
        #: WHAT THE ADMITTED COLUMNS DID NOT CARRY (2026-09-20). The per-part
        #: counts `build_table` logged, kept in the table so a reader of the
        #: FILE, not only of the build log, can see that (say) every stepfull
        #: slot of this table came up "not run". A part missing here is a part
        #: no record collected, which is a different fact from a part that
        #: disagreed, and neither one is visible in the cell counts.
        absent_parts={p: dict(sorted(w.items())) for p, w in sorted(absent_total.items())},
        lane_revisions=dict(getattr(harness, "LANE_REVISIONS", {}) or {}),
        fixtures=want_fix, heldout=want_held,
        records=[records[i] for i in used],
        cells=cells,
    )


def stale_reference_lanes(table, harness):
    """Lanes whose FIXTURE has moved but whose REFERENCE has not, sorted.

    A reference hash describes one exact input. When a lane's fixture is
    shrunk or otherwise changed, the harness bumps its `LANE_REVISIONS`
    entry, and every hash recorded at the old input stops describing
    anything this harness can produce. Comparing against it would fail for a
    reason that has nothing to do with the user's machine, which is the worst
    failure this tool has: it looks exactly like the identity claim being
    false.

    A lane is stale when the table still carries cells for it and the table's
    recorded revision is not the harness's. A table generated before
    `lane_revisions` was recorded carries no revision at all, which is not
    evidence that it is current, so it counts as stale. A lane the table has
    no cells for is not stale; there is nothing to compare against.
    """
    want = dict(getattr(harness, "LANE_REVISIONS", {}) or {})
    if not want:
        return []
    have = table.get("lane_revisions") or {}
    with_cells = {k.partition("/")[0] for k in table.get("cells", {})}
    return sorted(lane for lane, rev in want.items()
                  if lane in with_cells and have.get(lane) != rev)


def merge_reference_lanes(base, candidate, lanes):
    """Admit selected complete lanes without rewriting unrelated evidence.

    The candidate must come from build_table's strict admission. Keep the
    base's global policy and harness witness: updating a few lanes does not
    qualify its legacy cells. Record the new policy per selected lane instead.
    """
    lanes = set(lanes)
    if not lanes:
        raise TableError("scoped admission requires at least one lane")
    policy = candidate.get("admission_policy", {})
    if (policy.get("min_repeats", 0) < 2 or not policy.get("input_witness_required")
            or not policy.get("property_protocol_required")):
        raise TableError("scoped admission requires a strict generated candidate")
    for field in ("format", "fixtures", "heldout"):
        if base.get(field) != candidate.get(field):
            raise TableError(f"scoped admission cannot change {field}")
    selected = {}
    for lane in sorted(lanes):
        for fixture in base["fixtures"]:
            key = f"{lane}/{fixture}"
            cell = candidate["cells"].get(key, {})
            required = {"train", "infer", "model", "batch"} | set(base["cells"].get(key, {}))
            if required - set(cell):
                raise TableError(f"{key}: missing parts {sorted(required - set(cell))}")
            if any(e.get("ref") is None or e.get("conflict") for e in cell.values()):
                raise TableError(f"{key}: missing or conflicted reference")
            selected[key] = cell
    result = copy.deepcopy(base)
    indices = {}
    for key, cell in selected.items():
        cell = copy.deepcopy(cell)
        for entry in cell.values():
            for cls, value in entry["cols"].items():
                idx = value if isinstance(value, int) else value[0]
                if idx not in indices:
                    indices[idx] = len(result["records"])
                    result["records"].append(copy.deepcopy(candidate["records"][idx]))
                entry["cols"][cls] = indices[idx] if isinstance(value, int) else [indices[idx], value[1]]
        result["cells"][key] = cell
    for lane in lanes:
        revisions = result.setdefault("lane_revisions", {})
        if lane in candidate.get("lane_revisions", {}):
            revisions[lane] = candidate["lane_revisions"][lane]
        else:
            revisions.pop(lane, None)
        result.setdefault("lane_admission", {})[lane] = dict(
            policy=copy.deepcopy(policy), harness_sha256=candidate["harness_sha256"])
    return result


def write_table(table, path):
    """Compact JSON, sorted keys, one trailing newline, so a regenerated
    table diffs as a small change."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(table, fh, sort_keys=True, separators=(",", ":"))
        fh.write("\n")


def summary(table):
    """Counts for the maintainer's log and the docs: cell parts with a
    reference, by how many device classes back it, and conflicts."""
    by_n = {}
    conflicts = 0
    parts = 0
    for cell in table["cells"].values():
        for ent in cell.values():
            parts += 1
            if ent.get("conflict"):
                conflicts += 1
                continue
            n = sum(1 for v in ent["cols"].values() if isinstance(v, int))
            by_n[n] = by_n.get(n, 0) + 1
    return dict(cells=len(table["cells"]), parts=parts, conflicts=conflicts,
                classes_agreeing=dict(sorted(by_n.items())), records=len(table["records"]))
