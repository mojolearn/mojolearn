# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The reference hash table `python -m mojolearn verify --all` compares against.

WHAT IT IS. One small JSON shipped in the wheel at
`mojolearn/verify_reference/table.json`. For every identity_break cell
(`<lane>/<fixture>`) and every part (`train`, `infer`, `model`, `batch`) it
holds the hash the committed records agree on, and which record each device
class (apple, nvidia, amd, cpu) got it from, with that record's directory and
commit. No hash in it is typed: `build_table` reads the JSON columns
`tools/identity_break.py --json` wrote under `bench/results/identity_break/`.

WHICH RECORDS COUNT. A column is admitted only if it is identical mode, names
a commit, ran the default fixture size on one device, is not a sabotage run
(by its JSON flags and by its file name), is not a partial or an unfixed
before-picture, and its fixture hashes equal the ones this harness generates
now. A column whose fixtures differ hashed different input bytes, so its
cells say nothing about the current lanes.

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
import json
import os
import re
import subprocess

FORMAT = "mojolearn.verify-reference.v1"
TABLE_DIR = "verify_reference"
TABLE_NAME = "table.json"
PARTS = ("train", "infer", "model", "batch")
CLASSES = ("apple", "nvidia", "amd", "cpu")

#: what `mojolearn.vendor()` reads back, as a device class of the table
VENDOR_CLASS = {"metal": "apple", "cuda": "nvidia", "hip": "amd", "cpu": "cpu"}

_COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
#: file and directory name tokens of runs that are not evidence of the claim
_EXCLUDED_NAME_TOKENS = ("sabotage", "partial", "probe", "unfixed", "post-merge-smoke")
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


def entry(table, lane, fixture, part):
    """The table entry of one cell part, or None when no record has it."""
    cell = table["cells"].get(f"{lane}/{fixture}")
    return None if cell is None else cell.get(part)


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
    if value == "MOVED":
        return DIVERGENT, "this box gave two different hashes for the same fit (MOVED)"
    if value.startswith("BATCH_MOVED"):
        return DIVERGENT, "batch invariance failed on this box: " + value[:300]
    if value.startswith("RELOAD-MOVED"):
        return DIVERGENT, "the saved model predicts differently from the model in memory"
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
        if cls in v or cls in base or (cls == "apple" and v == "arm64" and "apple" in base):
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


def admit(j, path):
    """None when the column is admissible, else the reason it is not."""
    low = path.lower()
    if any(tok in low for tok in _EXCLUDED_NAME_TOKENS):
        return "sabotage, partial, probe, unfixed or smoke run (by name)"
    if not isinstance(j, dict) or not isinstance(j.get("cells"), dict):
        return "not an identity_break column"
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
    if str(pkg.get("par_devices") or "0") != "0":
        return f"par_devices {pkg.get('par_devices')}"
    host = j.get("host") or {}
    for fam in (host.get("families") or {}).values():
        if isinstance(fam, dict) and fam.get("sabotage"):
            return "a host binding reads back sabotage"
    if str(j.get("vendor", "")).endswith("-two"):
        return "two-device part"
    if j.get("heldout_seed") not in (None, 1):
        return f"heldout_seed {j.get('heldout_seed')}"
    return None


def _part_value(cell, part):
    """The one value a column carries for a part, or None when it carries no
    usable one (moved, refused, reload-moved, batch-moved, skipped)."""
    if part == "train":
        if cell.get("verdict") != "STABLE" or not cell.get("hashes"):
            return None
        return cell["hashes"][0]
    verdict = cell.get(f"{part}_verdict")
    values = cell.get(part)
    if verdict not in ("STABLE", "N/A") or not values or values[0] is None:
        return None
    v = values[0]
    if not isinstance(v, str) or v.startswith(_SKIPPED_NA):
        return None
    return v


def build_table(record_paths, harness, repo_root, lanes=None, log=None):
    """The table dict from the column JSONs at `record_paths`. `harness` is
    the imported identity_break module: its fixtures decide which columns
    hashed the current input bytes, and its LANES decide which cells are
    kept. `repo_root` is the checkout whose git history dates the commits."""
    log = log or (lambda s: None)
    lanes = set(lanes if lanes is not None else harness.LANES)
    want_fix = {f: dict(X=harness._h(X), y_clf=harness._h(yc), y_reg=harness._h(yr))
                for f, (X, yc, yr) in ((f, harness.fixture(f)) for f in harness.FIXTURES)}
    want_held = {f: dict(X=harness._h(harness.heldout(f))) for f in harness.FIXTURES}
    records, cache = [], {}
    best = {}                      # (cell, part, class) -> (key, record idx, value)
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
        for cell_key, cell in j["cells"].items():
            lane, _, fixture = cell_key.partition("/")
            if lane not in lanes or fixture not in want_fix or not isinstance(cell, dict):
                continue
            for part in PARTS:
                value = _part_value(cell, part)
                if value is None:
                    continue
                slot = (cell_key, part, cls)
                if slot not in best or key > best[slot][0]:
                    best[slot] = (key, idx, value)
                kept += 1
        log(f"use  {path}: class {cls}, commit {commit[:9]}, {kept} cell parts")
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
        fixtures=want_fix, heldout=want_held,
        records=[records[i] for i in used],
        cells=cells,
    )


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
