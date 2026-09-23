#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE NVIDIA AND AMD COLUMNS OF A RELEASE, on the Mac side
(lane/release-gpu-columns, 2026-09-22). `pixi run release` calls these; run
by hand only as the fallback docs/RELEASE_CHECKLIST.md section 4 describes.

    python3 tools/release_gpu_columns.py select  <dir>           cuda.selection.json, hip.selection.json
    python3 tools/release_gpu_columns.py place   <leg column dir> <cuda|hip>
    python3 tools/release_gpu_columns.py compare [--no-supplement] [--json OUT]

WHICH LANES. Exactly the Apple pass's rule, per backend: the lanes whose
sources changed since the last pass on that backend that FINISHED (its
records live under ~/mojolearn-evidence/release-check/<commit>/<backend>/,
where `place` puts every column that comes home), else since the newest v*
tag, widening to every lane when a changed path cannot be attributed, and
leaving out BY NAME every lane that cannot state its proposition on one GPU
(tools/lane_applicability.py; the multi-device `par-*` drivers and the CPU
host-route lanes). `select` asks `tools/verify_lanes.py --gpu-pass <b>
--write-selection` for that answer on this Mac, where the records are; the
build leg ships the file and runs exactly those lanes on the set it built.

WHAT IS COMPARED. `compare` diffs every column of the commit together with
`tools/identity_break.py --diff`, scoped to the lanes the NVIDIA and AMD
columns carry: the CPU column (the reference; the CPU pass covers the union
of every backend's selection, and any lane it lacks is run here first, on the
CPU, as a supplement), the Apple column when it exists, the CPU column each
GPU box recorded of itself when there is one, and the NVIDIA and AMD columns.
A part that only GPUs hash (a saved GPU model, which a CPU column loads and
does not write) is held to the other GPU columns. Every DIVERGENT cell fails
the release, printed as lane/fixture part with each column's hash. A part no
second column carries is UNCOMPARED and is printed by name, not hidden.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
BACKENDS = {"cuda": "NVIDIA", "hip": "AMD"}
FIXTURES = "base,denormal,odd"
#: diff verdicts that fail a release. A cell only one column hashed is
#: UNCOMPARED (printed, not failed); FIXTURE and HELD-OUT MISMATCH fail.
BAD_VERDICTS = ("DIVERGENT", "MOVED", "RELOAD-MOVED", "BATCH_MOVED", "RLPAIR_MOVED")


def check_dir(commit):
    base = os.environ.get("MOJOLEARN_RELEASE_CHECK_DIR") or os.path.expanduser("~/mojolearn-evidence/release-check")
    return Path(base) / commit[:12]


def python_cmd():
    """The interpreter that can import the harness (numpy): the test
    environment when pixi is here, else this one."""
    test_py = ROOT / ".pixi" / "envs" / "test" / "bin" / "python"
    return [str(test_py)] if test_py.exists() else [sys.executable]


def selection_cmd(backend, out):
    return python_cmd() + [str(TOOLS / "verify_lanes.py"), "--gpu-pass", backend, "--write-selection", str(out)]


def write_selection(backend, out, run=subprocess.run):
    """Write <out> for `backend`; return the selection record."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res = run(selection_cmd(backend, out), cwd=ROOT)
    code = res if isinstance(res, int) else res.returncode
    if code != 0:
        raise RuntimeError(f"verify_lanes --gpu-pass {backend} --write-selection exited {code}")
    return json.loads(Path(out).read_text())


def selection_ok(path, commit, backend):
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return False
    return d.get("commit") == commit and d.get("backend") == backend and isinstance(d.get("lanes"), list)


def _record_ok(d, commit):
    try:
        manifest = json.loads((d / "manifest.json").read_text())
        summary = json.loads((d / "run-summary.json").read_text())
    except (OSError, ValueError):
        return False
    return (manifest.get("commit") == commit and summary.get("complete") is True
            and not summary.get("validation_failures") and (d / "column.json").is_file())


def place(column_dir, backend, commit):
    """Copy a column a leg fetched (<column_dir>/<backend>/ and, when present,
    <column_dir>/cpu-box/) into this commit's release-check directory, where
    the next pass on `backend` finds it as its anchor. Returns the placed
    directory, or raises naming what is missing."""
    src = Path(column_dir) / backend
    if not _record_ok(src, commit):
        raise RuntimeError(f"no complete {BACKENDS[backend]} column of {commit[:12]} in {src} "
                           "(manifest, run-summary complete, column.json)")
    placed = {}
    for name, dest_name in ((backend, backend), ("cpu-box", f"cpu-box-{backend}")):
        s = Path(column_dir) / name
        if not s.is_dir() or (name == "cpu-box" and not _record_ok(s, commit)):
            continue
        dest = check_dir(commit) / dest_name
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(s, dest, ignore=shutil.ignore_patterns("gpu-package", "cpu-package", "*.tmp"),
                        symlinks=True)
        placed[name] = dest
    return placed[backend]


def _load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def _cells(path):
    d = _load(path)
    return d.get("cells", {}) if isinstance(d, dict) else {}


def parse_diff(text):
    """(bad rows, uncompared rows, mismatches) from an `identity_break --diff`
    transcript. A row is (lane/fixture, part, verdict, {column: shown})."""
    bad, uncompared, mismatch = [], [], []
    header = None
    for line in text.splitlines():
        if line.startswith(("FIXTURE MISMATCH", "HELD-OUT MISMATCH")):
            mismatch.append(line)
            continue
        if line.startswith("REQUIRE FAIL"):
            uncompared.append(line)
            continue
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells[0] == "lane/fixture":
            header = cells
            continue
        if header is None or set(cells[0]) <= {"-"}:
            continue
        if header[1] == "verdict":
            key, part, verdict, shown = cells[0], "train", cells[1], cells[2:]
            names = header[2:]
        else:
            key, part, verdict, shown = cells[0], cells[1], cells[2], cells[3:]
            names = header[3:]
        if verdict.split()[0] in BAD_VERDICTS:
            bad.append((key, part, verdict, dict(zip(names, shown))))
    return bad, uncompared, mismatch


def supplement_cpu(commit, lanes, run=subprocess.run):
    """Run the CPU column for `lanes` on this Mac (the reference the GPU
    columns lack), into release-check/<commit>/cpu-extra-<n>/. Returns the
    column path. The CPU pass normally covers the union already; this is the
    safety net for a selection that moved between the two."""
    import hashlib
    out = check_dir(commit) / ("cpu-extra-" + hashlib.sha256(",".join(sorted(lanes)).encode()).hexdigest()[:10])
    col = out / "column.json"
    if _record_ok(out, commit):
        return col
    slots = max(1, int(os.environ.get("MAC_SLOTS", "5")) - 1)
    cmd = python_cmd() + [str(TOOLS / "verify_lanes.py"), "--lanes", ",".join(sorted(lanes)), "--backend", "cpu",
                          "--fixtures", FIXTURES, "--repeats", "1", "--shards", str(slots), "--jobs", str(slots),
                          "--budget", "3600", "--timeout", "3600", "--wait-timeout", "3600", "--out", str(out)]
    if (out / "manifest.json").exists():
        cmd.append("--resume")
    print(f"# the CPU column lacks {len(lanes)} lane(s) the GPU columns carry; running them on the CPU here: "
          f"{','.join(sorted(lanes))}", flush=True)
    res = run(cmd, cwd=ROOT)
    code = res if isinstance(res, int) else res.returncode
    if code != 0 or not col.is_file():
        raise RuntimeError(f"the supplementary CPU column exited {code}; {out}")
    return col


def compare(commit, backends=tuple(BACKENDS), supplement=True, run=subprocess.run, out=print):
    """Diff the NVIDIA and AMD columns of `commit` against the CPU column
    (and every other column of the commit). Returns a verdict dict:
    {ok, backends: {b: dict(lanes, cells, status)}, divergent: [...],
    uncompared: [...], mismatch: [...], transcript}."""
    cdir = check_dir(commit)
    gpu = {b: cdir / b / "column.json" for b in backends}
    missing = [b for b, p in gpu.items() if not _record_ok(p.parent, commit)]
    verdict = dict(commit=commit, ok=False, backends={}, divergent=[], uncompared=[], mismatch=[],
                   missing=missing, supplement=None)
    if missing:
        out(f"# FAIL: no complete column for {', '.join(BACKENDS[b] for b in missing)} at {commit[:12]} "
            f"(looked in {cdir})")
        return verdict
    lanes = set()
    for b, p in gpu.items():
        cells = _cells(p)
        blanes = sorted({k.split("/")[0] for k in cells})
        verdict["backends"][b] = dict(lanes=len(blanes), cells=len(cells), column=str(p))
        lanes |= set(blanes)
        out(f"# {BACKENDS[b]} ({b}) column: {len(blanes)} lane(s), {len(cells)} cell(s)")
    if not lanes:
        verdict["ok"] = True
        out("# nothing to compare: the NVIDIA and AMD columns carry no lane")
        return verdict
    cpu = cdir / "cpu" / "column.json"
    if not _record_ok(cpu.parent, commit):
        out(f"# FAIL: no complete CPU column at {commit[:12]} ({cpu.parent}); run `pixi run -e test release-check`")
        verdict["missing"] = ["cpu"]
        return verdict
    sys.path.insert(0, str(TOOLS))
    import lane_applicability
    no_cpu_route = lane_applicability.degenerate("cpu-host")
    cpu_cells = _cells(cpu)
    fixtures = FIXTURES.split(",")
    lack = sorted(n for n in lanes if n not in no_cpu_route
                  and any(f"{n}/{f}" not in cpu_cells for f in fixtures))
    refs = [cpu]
    if lack:
        if not supplement:
            out(f"# FAIL: the CPU column lacks {len(lack)} lane(s) the GPU columns carry: {','.join(lack)}")
            verdict["missing"] = ["cpu:" + n for n in lack]
            return verdict
        refs.append(supplement_cpu(commit, lack, run=run))
        verdict["supplement"] = lack
    others = [p for p in [cdir / "metal" / "column.json"] + [cdir / f"cpu-box-{b}" / "column.json" for b in backends]
              if _record_ok(p.parent, commit)]
    columns = refs + others + list(gpu.values())
    gpu_only = sorted(n for n in lanes if n in no_cpu_route)
    if gpu_only:
        out(f"# {len(gpu_only)} lane(s) have no CPU route, so the GPU columns are held to each other: "
            f"{','.join(gpu_only)}")
    cmd = python_cmd() + [str(TOOLS / "identity_break.py"), "--diff", *map(str, columns),
                          "--lanes", ",".join(sorted(lanes)), "--require-columns", "2"]
    out("# $ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    transcript = res.stdout + res.stderr
    (cdir / "gpu-columns.diff.txt").write_text(transcript)
    bad, uncompared, mismatch = parse_diff(transcript)
    verdict.update(divergent=[dict(cell=k, part=p, verdict=v, hashes=h) for k, p, v, h in bad],
                   uncompared=uncompared, mismatch=mismatch, transcript=str(cdir / "gpu-columns.diff.txt"),
                   diff_exit=res.returncode, columns=[str(c) for c in columns])
    for k, p, v, h in bad:
        out(f"# {v}: {k} part={p} " + " ".join(f"{n}={s}" for n, s in h.items()))
    for line in mismatch:
        out(f"# {line}")
    for line in uncompared:
        out("# UNCOMPARED " + line[len("REQUIRE FAIL "):])
    summary = [ln for ln in transcript.splitlines() if ln.startswith("summary")]
    for ln in summary[:3]:
        out(f"# {ln}")
    verdict["ok"] = not bad and not mismatch and "summary:" in transcript
    if not bad and not mismatch and "summary:" not in transcript:
        out("# FAIL: the diff printed no summary; its transcript is " + str(cdir / "gpu-columns.diff.txt"))
    out(f"# NVIDIA/AMD against CPU: {'IDENTICAL' if verdict['ok'] else 'FAILED'} "
        f"({len(bad)} divergent cell part(s), {len(uncompared)} uncompared, {len(mismatch)} input mismatch(es))")
    (cdir / "gpu-columns.verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    return verdict


def verdict_ok(commit):
    d = _load(check_dir(commit) / "gpu-columns.verdict.json")
    return bool(d and d.get("ok") and d.get("commit") == commit)


def _head():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("dir")
    p = sub.add_parser("place")
    p.add_argument("column_dir")
    p.add_argument("backend", choices=tuple(BACKENDS))
    c = sub.add_parser("compare")
    c.add_argument("--no-supplement", action="store_true")
    c.add_argument("--backends", default=",".join(BACKENDS))
    for q in (s, p, c):
        q.add_argument("--commit", default="")
    args = ap.parse_args(argv)
    commit = args.commit or _head()
    if args.cmd == "select":
        for b in BACKENDS:
            rec = write_selection(b, Path(args.dir) / f"{b}.selection.json")
            print(f"# {BACKENDS[b]}: {rec['summary']}")
        return 0
    if args.cmd == "place":
        print(place(args.column_dir, args.backend, commit))
        return 0
    v = compare(commit, tuple(args.backends.split(",")), supplement=not args.no_supplement)
    return 0 if v["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
