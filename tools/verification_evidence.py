#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE IN-REPO VERIFICATION EVIDENCE: `verification/latest/<column>/`, always overwritten (Andrew, 2026-11-26).

Every identity record run, and the proving and GPU smoke runs of the same commit, leave ONE evidence set per
column (cpu, nvidia, amd, apple) in the repo. git history is the history: no dated copies are kept in the tree.

    stage    ON THE BOX, at the end of a job: read the job's results dir and write one evidence PART
             (`<out>/MANIFEST.json` and the files it lists) under `<results>/evidence/<column>/<part>/`. Parts:
               record   tools/identity_wheel_record.py: identity-wheels.json, every column record JSON the
                        recorder wrote (the reference tables admission builds from), per-chunk record logs
               verify   tools/ops/identity_record_job.sh: the whole `verify --all` JSON report, the single-device
                        verify commands' reports split out of it (extended/<profile>.json), the self-test summary
               proving  tools/ops/proving_job.sh: pass/fail/error counts and the failed test names per pytest step
               smoke    tools/ops/gpu_smoke_job.sh (and the CPU wheel smoke): SUMMARY.txt, the failed lanes and
                        tests from each pool's results.tsv
             plus, in every part, the job's logs (each capped at LOG_CAP_BYTES, its tail kept; the full log's R2
             location named in the MANIFEST) and any sabotage summary (*sabotage*.json).
    install  ON THE MAC, after the results come back: find every staged part under the given directories and
             write them into `verification/latest/<column>/<part>/`. A part of ANOTHER commit than the column's
             MANIFEST replaces the column's directory WHOLE (delete, then write: one vendor's install never touches
             another column); a part of the same commit replaces that part's directory only, so the record, the
             proving run and the smoke of one commit sit side by side. `--commit` commits the column directories
             (git add -A verification/latest/<column>) with a message naming the parts and the commit.

Where it runs automatically: the release (tools/release_identity_wheels.py identity-admit) installs and commits the
evidence of every fetched column on `ref/<commit12>` beside the admission; `tools/box_persist.py fetch --extract
--evidence` installs and commits a fetched job's parts. By hand, after any fetch:

    python3 tools/verification_evidence.py install --from <fetched dir> [--from ...] --commit

A human sees it in git at `verification/latest/<column>/MANIFEST.json` (`git log -- verification/latest`).
Tests: tools/tests/test_verification_evidence.py (no box, no network).
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LATEST = Path("verification") / "latest"
SCHEMA = "mojolearn.verification-evidence.v1"
COLUMNS = ("cpu", "nvidia", "amd", "apple")
PARTS = ("record", "verify", "proving", "smoke")
#: each copied log keeps its last LOG_CAP_BYTES bytes (64 KiB); the MANIFEST names the full log in R2
LOG_CAP_BYTES = 64 * 1024
#: pytest steps of tools/ops/proving_job.sh whose logs carry the proving counts
PROVING_STEPS = ("proving", "proving_gpu")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _git_head(root):
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_json(path):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_json(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")


def _cap_log(src, dst):
    """Copy the last LOG_CAP_BYTES of SRC to DST; returns (full bytes, kept bytes)."""
    size = os.path.getsize(src)
    with open(src, "rb") as fh:
        fh.seek(max(0, size - LOG_CAP_BYTES))
        tail = fh.read()
    dst.parent.mkdir(parents=True, exist_ok=True)
    head = b"" if size <= LOG_CAP_BYTES else (
        b"[verification_evidence: the last %d of %d bytes; the full log is in R2, see MANIFEST.json]\n"
        % (len(tail), size))
    dst.write_bytes(head + tail)
    return size, len(tail)


def _busy(path):
    """'mean X, max Y' of a busy sampler file (tools/ops/box_cores_lib.sh busy_sampler: one number a line)."""
    try:
        vals = [float(line.split()[0]) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError, IndexError):
        return None
    return dict(mean=round(sum(vals) / len(vals), 1), max=max(vals), samples=len(vals)) if vals else None


def _steps(results):
    """The job's steps from <results>/steps.tsv (name, rc, seconds, start, end, busy), as the jobs write it."""
    rows = []
    path = Path(results) / "steps.tsv"
    if path.is_file():
        for line in path.read_text().splitlines():
            f = line.split("\t")
            if len(f) >= 3 and f[0] != "name":
                row = dict(name=f[0], rc=int(f[1]) if f[1].lstrip("-").isdigit() else f[1],
                           seconds=int(f[2]) if f[2].isdigit() else f[2])
                for k, v in zip(("start", "end", "busy"), f[3:6]):
                    row[k] = v
                rows.append(row)
    return rows


def _wheels(results):
    """[{file, sha256}] from the recorder's installed-wheels.txt, or the wheels named in a verify report."""
    for path in sorted(Path(results).rglob("installed-wheels.txt")):
        rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
        return [dict(file=r[0], sha256=r[1]) for r in rows if len(r) == 2]
    return []


def _driver():
    for cmd in (["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                ["rocm-smi", "--showdriverversion"]):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            continue
        if p.returncode == 0 and p.stdout.strip():
            return p.stdout.strip().splitlines()[:4]
    return None


def _r2():
    tag = os.environ.get("MOJOLEARN_PERSIST_TAG")
    return dict(tag=tag, fetch="python3 tools/box_persist.py fetch --tag %s --extract" % tag) if tag else None


# ------------------------------------------------------------------ the parts

def _proving_summary(log_text):
    """Counts and failed test names of one pytest -q log."""
    counts = {}
    tail = [line for line in log_text.splitlines() if re.search(r"\b(passed|failed|error)", line)]
    for line in tail[-3:]:
        for n, what in re.findall(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed|deselected)", line):
            counts[what.rstrip("s") if what.startswith("error") else what] = int(n)
    failed = sorted(set(re.findall(r"^(?:FAILED|ERROR) (\S+)", log_text, re.M)))
    return dict(counts=counts, failed=failed)


def _smoke_summary(smoke_dir):
    smoke_dir = Path(smoke_dir)
    summary = (smoke_dir / "SUMMARY.txt").read_text(errors="replace") if (smoke_dir / "SUMMARY.txt").is_file() else ""
    steps = {}
    for line in summary.splitlines():
        m = re.match(r"^(\S+) rc=(\d+)", line)
        if m:
            steps[m.group(1)] = int(m.group(2))
    failed = {}
    for tsv in sorted(smoke_dir.glob("*/results.tsv")):
        pool = tsv.parent.name
        for line in tsv.read_text(errors="replace").splitlines():
            f = line.split("\t")
            # tools/ops/wheel_verify_smoke.py: status `ok` passed; anything else (failed, crash, timeout, env) is listed
            if len(f) >= 3 and f[0] != "unit" and f[2] != "ok":
                failed.setdefault(pool, []).append(dict(unit=f[0], rc=f[1], status=f[2]))
    return summary, dict(steps=steps, failed=failed, failed_count=sum(len(v) for v in failed.values()))


def stage(results, out, column, part, commit=None, root=ROOT):
    """Write one evidence part from a job's RESULTS dir into OUT (replaced whole). Returns the MANIFEST."""
    if column not in COLUMNS:
        raise SystemExit("verification_evidence: unknown column %s (%s)" % (column, ", ".join(COLUMNS)))
    if part not in PARTS and not re.fullmatch(r"(record|verify|proving|smoke)(-[a-z0-9-]+)?", part):
        raise SystemExit("verification_evidence: unknown part %s (%s, optionally -<suffix>)" % (part, ", ".join(PARTS)))
    results, out = Path(results).resolve(), Path(out).resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    files, logs = [], []
    kind = part.split("-", 1)[0]
    manifest = dict(schema=SCHEMA, column=column, part=part, staged=_now(), commit=commit or _git_head(root),
                    wheels=_wheels(results), driver=_driver(), r2=_r2(), steps=_steps(results),
                    log_cap_bytes=LOG_CAP_BYTES)

    def keep_json(src, name):
        dst = out / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        files.append(name)

    report = _read_json(results / "verify-all.json")
    if report is not None:
        device = report.get("device") or {}
        manifest.update(version=device.get("mojolearn_version"), device=device,
                        verify=dict(verdict=report.get("verdict"), exit=report.get("exit"),
                                    detail=report.get("detail"), elapsed_s=report.get("elapsed_s"),
                                    profiles=report.get("profiles"), extended=report.get("extended_summary"),
                                    cell_workers=(report.get("execution") or {}).get("cell_workers")))
        manifest["commit"] = manifest["commit"] or device.get("commit")
        if kind == "verify":
            keep_json(results / "verify-all.json", "verify-all.json")
            for name, block in (report.get("extended") or {}).items():
                _write_json(out / "extended" / (name + ".json"), block)
                files.append("extended/%s.json" % name)
            if report.get("self_test") is not None:
                _write_json(out / "self-test.json", report["self_test"])
                files.append("self-test.json")
    if kind == "record":
        summary = _read_json(results / "identity-wheels.json")
        if summary is not None:
            keep_json(results / "identity-wheels.json", "identity-wheels.json")
            manifest["record"] = dict(ok=summary.get("ok"), failures=summary.get("failures"),
                                      busy_plan=dict(workers_per_device=summary.get("workers_per_device"),
                                                     nproc=summary.get("nproc"), gpus=summary.get("gpus")),
                                      box=summary.get("box"))
            for col in (summary.get("columns") or {}).values():
                for chunk in col.get("chunks", []):
                    for rec in chunk.get("records", []):
                        src = Path(rec.get("json") or "")
                        if not src.is_absolute():
                            src = results / src
                        if src.is_file():
                            keep_json(src, "tables/%s/w%s-%s" % (col.get("label"), chunk.get("worker"), src.name))
        for path in sorted(results.glob("*/w*/record.log")):
            logs.append((path, Path("logs") / path.parent.parent.name / (path.parent.name + "-record.log")))
    if kind == "proving":
        proving = {}
        for step in PROVING_STEPS:
            log = results / "logs" / (step + ".log")
            if log.is_file():
                proving[step] = _proving_summary(log.read_text(errors="replace"))
        manifest["proving"] = proving
        _write_json(out / "proving-summary.json", proving)
        files.append("proving-summary.json")
    if kind == "smoke":
        smoke_dir = results / "smoke" if (results / "smoke").is_dir() else results
        text, smoke = _smoke_summary(smoke_dir)
        if text:
            (out / "SUMMARY.txt").write_text(text)
            files.append("SUMMARY.txt")
        manifest["smoke"] = smoke
        _write_json(out / "smoke-failures.json", smoke)
        files.append("smoke-failures.json")
        for path in sorted(smoke_dir.glob("crashes/*.txt")):
            logs.append((path, Path("crashes") / path.name))
    for path in sorted(results.glob("*sabotage*.json")) + sorted(results.glob("logs/*sabotage*.json")):
        keep_json(path, "sabotage/" + path.name)
    for path in sorted(results.glob("logs/*.log")):
        logs.append((path, Path("logs") / path.name))
    busy = {}
    for path in sorted(results.glob("logs/*.busy")):
        b = _busy(path)
        if b:
            busy[path.stem] = b
    manifest["busy_cores"] = busy
    capped = []
    for src, rel in logs:
        if rel.suffix == ".log":
            # verification/ takes no new *.log (.gitignore, tools/hooks/check_tree_additions.sh): a capped excerpt is
            # saved as <name>.log.txt; the full log is in R2
            rel = rel.with_name(rel.name + ".txt")
        size, kept = _cap_log(src, out / rel)
        capped.append(dict(file=str(rel), bytes=size, kept=kept,
                           full="r2:%s/%s" % ((manifest["r2"] or {}).get("tag") or "<no persist tag>",
                                              src.relative_to(results))))
    manifest["logs"] = capped
    manifest["files"] = sorted(files)
    if manifest["steps"]:
        manifest["start"] = manifest["steps"][0].get("start")
        manifest["end"] = manifest["steps"][-1].get("end")
    _write_json(out / "MANIFEST.json", manifest)
    return manifest


# ------------------------------------------------------------------ install on the Mac

def staged_parts(dirs):
    """[(column, part, dir)] of every staged part (<...>/evidence/<column>/<part>/MANIFEST.json) under DIRS."""
    found = []
    for d in dirs:
        for m in sorted(Path(d).rglob("MANIFEST.json")):
            doc = _read_json(m)
            if not doc or doc.get("schema") != SCHEMA or "part" not in doc:   # a part, not a column MANIFEST
                continue
            found.append((doc["column"], doc["part"], m.parent, doc))
    return found


def install(dirs, tree=ROOT, commit=False, say=print):
    """Write every staged part under DIRS into TREE/verification/latest/<column>/<part>/; returns the columns."""
    tree = Path(tree)
    by_column = {}
    for column, part, src, doc in staged_parts(dirs):
        by_column.setdefault(column, []).append((part, src, doc))
    for column, parts in sorted(by_column.items()):
        dest = tree / LATEST / column
        commits = {doc.get("commit") for _, _, doc in parts}
        if len(commits) > 1:
            raise SystemExit("verification_evidence: the %s parts name different commits %s; install one commit "
                             "at a time" % (column, sorted(c or "?" for c in commits)))
        new_commit = commits.pop()
        old = _read_json(dest / "MANIFEST.json") or {}
        if dest.exists() and old.get("commit") != new_commit:
            shutil.rmtree(dest)          # another commit: the column is replaced whole
            say("evidence %s: replaced whole (was %s, now %s)" % (column, (old.get("commit") or "?")[:12],
                                                                 (new_commit or "?")[:12]))
        kept = {p: d for p, d in (old.get("parts") or {}).items()} if dest.exists() else {}
        for part, src, doc in parts:
            target = dest / part
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(src, target)
            kept[part] = dict(staged=doc.get("staged"), version=doc.get("version"),
                              verdict=(doc.get("verify") or {}).get("verdict"),
                              steps=[dict(name=s.get("name"), rc=s.get("rc"), seconds=s.get("seconds"),
                                          busy=s.get("busy")) for s in doc.get("steps", [])],
                              r2=doc.get("r2"))
            say("evidence %s/%s <- %s" % (column, part, src))
        version = next((d.get("version") for _, _, d in parts if d.get("version")), old.get("version"))
        _write_json(dest / "MANIFEST.json", dict(schema=SCHEMA, column=column, commit=new_commit, version=version,
                                                 installed=_now(), parts=kept))
    if commit and by_column:
        paths = [str(LATEST / c) for c in sorted(by_column)]
        subprocess.run(["git", "-C", str(tree), "add", "-A", "--", *paths], check=True)
        if subprocess.run(["git", "-C", str(tree), "diff", "--cached", "--quiet", "--", *paths]).returncode:
            names = ", ".join("%s (%s)" % (c, "+".join(sorted(p for p, _, _ in v))) for c, v in sorted(by_column.items()))
            first = next(iter(by_column.values()))[0][2].get("commit") or "?"
            subprocess.run(["git", "-C", str(tree), "commit", "-q", "-m",
                            "Verification evidence of %s: %s (tools/verification_evidence.py install)\n\n"
                            "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" % (first[:12], names),
                            "--", *paths], check=True)
            say("evidence committed: %s" % names)
        else:
            say("evidence unchanged: nothing to commit")
    return sorted(by_column)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage", help="on the box: one evidence part from a job's results dir")
    s.add_argument("--results", required=True)
    s.add_argument("--column", required=True, choices=COLUMNS)
    s.add_argument("--part", required=True, help="record, verify, proving or smoke (optionally -<suffix>)")
    s.add_argument("--out", help="default <results>/evidence/<column>/<part>")
    s.add_argument("--commit", help="default: git HEAD of this tree (a box tree is at the build's commit)")
    i = sub.add_parser("install", help="on the Mac: staged parts into verification/latest/<column>/")
    i.add_argument("--from", dest="dirs", action="append", required=True)
    i.add_argument("--tree", default=str(ROOT))
    i.add_argument("--commit", action="store_true", help="git add and commit the column directories")
    a = ap.parse_args(argv)
    if a.cmd == "stage":
        out = a.out or os.path.join(a.results, "evidence", a.column, a.part)
        m = stage(a.results, out, a.column, a.part, a.commit)
        print("EVIDENCE staged %s/%s -> %s (%d file(s), %d log(s))" % (a.column, a.part, out, len(m["files"]),
                                                                       len(m["logs"])))
        return 0
    columns = install(a.dirs, a.tree, a.commit)
    if not columns:
        print("EVIDENCE nothing staged under %s" % ", ".join(a.dirs))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
