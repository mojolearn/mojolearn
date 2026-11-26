# SPDX-License-Identifier: Apache-2.0
"""`python -m mojolearn verify --show-paths [LANE] [--from PATH]`: what each side did, from a saved record.

`verify --all` already counts, per lane and per side, what the runtime did while the lane's cells ran (the
activity record, `what_ran`: device kernel launches by site, bytes and calls to and from the device, host routines,
host thread splits and tasks). It saves that record with each side's output hash for every lane (`save`): beside
its JSON (`<json-out stem>.what-ran.json`) and, always, at `last_path()` (the user cache), and the identity record
keeps it in verification/latest/<column>/verify/what-ran.json. This command reads that file, or a whole `verify
--all` JSON report, and RUNS NOTHING:

    verify --show-paths            every lane: CPU and GPU counts side by side and each side's output hash
    verify --show-paths LANE       that lane's CPU steps (host routines, splits) and GPU steps (launches, copies)
                                   side by side in full, then both hashes; a DIVERGENT lane shows the two hashes
    verify --show-paths --from F   a record made on another machine

On a CPU-only install the GPU side was not run, and it says so. Verification tooling: nothing here is product code.
"""
import json
import os
import sys

FORMAT = "mojolearn.verify-what-ran.v1"
SIDES = ("cpu", "gpu")
COUNTS = ("launches", "upload_bytes", "upload_calls", "download_bytes", "download_calls", "host_splits", "host_tasks")
EXIT_OK, EXIT_USAGE, EXIT_NO_RECORD = 0, 2, 5


def last_path():
    """Where `verify --all` always leaves its latest record: the user cache, never the working directory."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "mojolearn", "verify-what-ran.json")


def beside(json_out):
    stem = json_out[:-5] if str(json_out).endswith(".json") else str(json_out)
    return stem + ".what-ran.json"


def _hashes(cells):
    """{lane: {side: dict(part, fixture, hash, state)}}: each side's `infer` hash (else `train`), base fixture
    first; a DIVERGENT row wins, so a broken lane shows the hashes that differ."""
    out = {}
    rank = lambda r: (r.get("state") != "DIVERGENT", r.get("part") != "infer", r.get("fixture") != "base")
    for r in cells or ():
        if r.get("part") not in ("infer", "train") or r.get("side") not in SIDES:
            continue
        lane, side = r["lane"], r["side"]
        have = out.setdefault(lane, {}).get(side)
        row = dict(part=r["part"], fixture=r["fixture"], hash=r.get("value"), state=r.get("state"),
                   reference=r.get("reference"))
        if have is None or rank(row) < rank(have):
            out[lane][side] = row
    return out


def document(report):
    """The saved record: the device, each side's activity per lane, and each side's hash per lane."""
    what = ((report.get("what_ran") or {}).get("sides")) or {}
    sides = (report.get("sides") or {})
    return dict(format=FORMAT, device=report.get("device"), gpu_available=sides.get("gpu_available"),
                lanes=report.get("lanes"), what_ran={side: (what.get(side) or {}).get("lanes") or {} for side in SIDES},
                hashes=_hashes(report.get("cells")), paths_sample=report.get("paths_sample"),
                verdict=report.get("verdict"))


def save(report, json_out=None):
    """Write the record beside the JSON (when there is one) and at `last_path()`; returns the paths written."""
    doc, written = document(report), []
    for path in ([beside(json_out)] if json_out else []) + [last_path()]:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, sort_keys=True)
                fh.write("\n")
            written.append(path)
        except OSError:
            continue                    # a read-only cache never costs the run
    return written


def load(path):
    """A saved record, or a whole `verify --all` report (converted)."""
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    if doc.get("format") == FORMAT:
        return doc
    if "what_ran" in doc and "cells" in doc:
        return document(doc)
    raise ValueError(f"{path} is neither a what-ran record ({FORMAT}) nor a verify --all report")


def _size(n):
    for unit, scale in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if n >= scale:
            return f"{n / scale:.1f}{unit}"
    return f"{n}B"


def _cpu_only(doc):
    return not doc["what_ran"].get("gpu") and ((doc.get("device") or {}).get("device_class") == "cpu"
                                               or doc.get("gpu_available") is False)


def _side_cell(a):
    if a is None:
        return "not run"
    return (f"L {a.get('launches', 0):>6,} up {_size(a.get('upload_bytes', 0)):>8}/{a.get('upload_calls', 0):<4} "
            f"down {_size(a.get('download_bytes', 0)):>8}/{a.get('download_calls', 0):<4} "
            f"split {a.get('host_splits', 0):>5,}/{a.get('host_tasks', 0):<6,}")


def table_lines(doc):
    lanes = sorted(set(doc["what_ran"].get("cpu") or {}) | set(doc["what_ran"].get("gpu") or {}) | set(doc["hashes"]))
    d = doc.get("device") or {}
    lines = [f"# what each side did, per lane (counted by the runtime in the run; {d.get('vendor')} "
             f"{d.get('device')}, cpu {d.get('cpu_model')})",
             "# L launches; up/down bytes/calls to and from the device; split host thread splits/tasks"]
    if _cpu_only(doc):
        lines.append("# GPU side: not run here (CPU-only install)")
    w = max([4] + [len(lane) for lane in lanes])
    lines.append(f"{'lane':<{w}}  {'CPU':<62}  {'GPU':<62}  hash cpu / gpu")
    for lane in lanes:
        a = {side: (doc["what_ran"].get(side) or {}).get(lane) for side in SIDES}
        h = doc["hashes"].get(lane) or {}
        hc, hg = (h.get("cpu") or {}).get("hash"), (h.get("gpu") or {}).get("hash")
        state = "DIVERGENT " if any((h.get(s) or {}).get("state") == "DIVERGENT" for s in SIDES) else ""
        lines.append(f"{lane:<{w}}  {_side_cell(a['cpu']):<62}  {_side_cell(a['gpu']):<62}  "
                     f"{state}{str(hc)[:16]} / {str(hg)[:16] if hg is not None else 'not run'}")
    lines.append(f"{len(lanes)} lane(s). One lane in full: python -m mojolearn verify --show-paths LANE")
    return lines


def lane_lines(doc, lane):
    acts = {side: (doc["what_ran"].get(side) or {}).get(lane) for side in SIDES}
    hashes = doc["hashes"].get(lane) or {}
    if not any(acts.values()) and not hashes:
        raise KeyError(lane)
    cpu, gpu = acts["cpu"] or {}, acts["gpu"] or {}
    left = [f"host routines ({sum(c for _, c in cpu.get('host_names') or [])} calls):"]
    left += [f"  {name} x{count:,}" for name, count in cpu.get("host_names") or []]
    left += [f"host thread splits {cpu.get('host_splits', 0):,}, tasks {cpu.get('host_tasks', 0):,}, "
             f"widest {cpu.get('host_tasks_widest', 0):,}",
             f"device launches {cpu.get('launches', 0):,}"]
    if gpu or not _cpu_only(doc):
        right = [f"device launches ({gpu.get('launches', 0):,}):"]
        right += [f"  {name} x{count:,}" for name, count in gpu.get("launch_names") or []]
        right += [f"to the device {_size(gpu.get('upload_bytes', 0))} in {gpu.get('upload_calls', 0):,} call(s)",
                  f"back {_size(gpu.get('download_bytes', 0))} in {gpu.get('download_calls', 0):,} call(s)",
                  f"host thread splits {gpu.get('host_splits', 0):,}"]
    else:
        right = ["not run here (CPU-only install)"]
    w = max(len(x) for x in left + ["CPU side"]) + 2
    lines = [f"# {lane}: what each side did (counted by the runtime in the run)", f"{'CPU side':<{w}}GPU side"]
    for i in range(max(len(left), len(right))):
        lines.append(f"{(left[i] if i < len(left) else ''):<{w}}{right[i] if i < len(right) else ''}")
    hc, hg = hashes.get("cpu") or {}, hashes.get("gpu") or {}
    part = hc.get("part") or hg.get("part")
    lines.append(f"hash cpu ({part}, {hc.get('fixture') or hg.get('fixture')}): {hc.get('hash')}")
    lines.append(f"hash gpu ({part}, {hg.get('fixture') or hc.get('fixture')}): "
                 f"{hg.get('hash') if hg else 'not run here (CPU-only install)'}")
    if "DIVERGENT" in (hc.get("state"), hg.get("state")):
        lines.append(f"DIVERGENT: cpu {hc.get('hash')} ({hc.get('state')}) vs gpu {hg.get('hash')} "
                     f"({hg.get('state')}); reference {hc.get('reference') or hg.get('reference')}")
    elif hc.get("hash") is not None and hc.get("hash") == hg.get("hash"):
        lines.append("same bits on both sides")
    return lines


def cmd_show_paths(args):
    path = getattr(args, "paths_from", None) or last_path()
    try:
        doc = load(path)
    except FileNotFoundError:
        print(f"verify --show-paths: no saved record at {path}; run `python -m mojolearn verify --all` first, "
              "or pass --from a what-ran record or a verify --all JSON report", file=sys.stderr)
        return EXIT_NO_RECORD
    except (OSError, ValueError) as exc:
        print(f"verify --show-paths: {exc}", file=sys.stderr)
        return EXIT_USAGE
    lane = args.show_paths or None
    if lane is None:
        print("\n".join(table_lines(doc)))
        return EXIT_OK
    try:
        print("\n".join(lane_lines(doc, lane)))
    except KeyError:
        print(f"verify --show-paths: the record has no lane {lane!r}", file=sys.stderr)
        return EXIT_USAGE
    return EXIT_OK
