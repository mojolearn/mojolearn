#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Drive a whole run: every segment of every route, in dependency order, on rented boxes.

    python3 tools/lm_run_driver.py plan   --spec run.json            # print the segments and their dependencies
    python3 tools/lm_run_driver.py run    --spec run.json --out DIR  # render, rent, wait, verify, next (resumable)
    python3 tools/lm_run_driver.py status --spec run.json --out DIR

THE SPEC (`run.json`):
    {"run": "runs/t2/2026-09-22", "recipe": "path/to/recipe.json", "recipe_key": "runs/t2/2026-09-22/recipe.json",
     "tokens_stage": "corpus/fineweb-edu-10BT/tokens/mojolearn-bpe-fineweb-edu-50257-v1",
     "lease_minutes": 120, "dollar_cap": 10, "amd_providers": ["do", "hotaisle", "runpod"],
     "wheel": "0.8.15",              # run from this published wheel; omit to build from source on each box
     "nvidia_gpus": "NVIDIA H100 80GB HBM3|NVIDIA H200|...", "nvidia_devices": "0,1", "amd_devices": "0",
     "routes": {"A": [{"segment": "1", "vendor": "nvidia", "steps": 10},
                      {"segment": "2", "vendor": "amd", "steps": 10},
                      {"segment": "3", "vendor": "live", "steps": 10, "first": "nvidia", "shards": [44, 20]},
                      ...],
                "B": [...],
                "C": [{"segment": "1", "vendor": "nvidia", "steps": 10,
                       "nvidia_devices": "0", "nvidia_gpus": "NVIDIA H100 80GB HBM3"}]}}

A segment entry may carry its own "nvidia_devices" and "nvidia_gpus"; they
win over the spec-wide values for that segment only (its rendered
`--devices`, the GPU types its rental walks and the GPU count it rents: one
device is one GPU, "0,1" is two). On a live segment they apply to the NVIDIA
side. A route may have fewer segments than route A: route C above replays
A's segment 1 on ONE H100 where A ran it on two, held to A's chain and
boundary, so a pass is evidence that the device fold is exact. Nothing
depends on a route other than A, and a route other than A and B starts only
when no ready A or B segment wants its vendor class (once started it holds
that class until it lands, as every segment does).

THE ORDER. Route A's segment k starts from route A's checkpoint k-1. Route
B's segment k starts from ROUTE A's checkpoint k-1 (the pipeline of the plan,
section 4) and is held to route A's chain for segment k (`--expect-chain`),
so it can run as soon as A's segment k-1 has landed, one segment behind A.
A route named in the spec's `"own_chain": ["B"]` instead starts segment k
from ITS OWN checkpoint k-1 (a separate run from the shared seed), still held
to route A's chain for segment k, so it runs once its own k-1 and A's k landed.
Every segment after the first does the arrival replay from the previous
segment's boundary-minus-two checkpoint on its own hardware first.

WHAT LANDING MEANS. The runner fetched the box's results, the segment's
verdict is PASS, its manifest.tsv (name, bytes, sha256 of every artifact) is
in the results, and for route B its boundary checkpoint's sha256 equals
route A's. The ledger (`DIR/ledger.json`) records every landed segment with
its checkpoint hashes; a rerun of `run` skips landed segments, so a failed
rental is retried by running the driver again. A segment that lands with a
FAIL verdict stops the driver (the plan's halt rule): nothing after a
disputed boundary is kept.

A HUNG SEGMENT RESUMES. A one-box segment that brings no segment.json
home (a box that hung and was stopped, a lease that ran out) is read for
what it DID land: the last checkpoint that its manifest.tsv pins, its log
says was uploaded, its checkpoints.sha256 (when the box wrote one) agrees
with, and that is in R2 at the manifest's size, with at least one chain
line after it. The ledger's `partial` entry records that step
(`resume_from`), the checkpoints landed so far, the attempt's chain (saved
under `legs/<route>-<segment>/`) and the attempt's small files
(`legs/<route>-<segment>/attempt-<n>/`); the attempt's results directory
is moved aside as `previous-attempt-<n>-*`. The next start of the segment
runs only the remaining steps from the segment's OWN checkpoint in R2, with
the same boundary. It is held to the hung attempt's chain (route A: the
chain is uploaded as `<run>/A/<segment>/partial.chain.jsonl`; the resumed
box re-derives every step the hung box wrote after the checkpoint and must
agree on each, and the checkpoint's state must equal the chain's line at
its step) or to route A's chain as before (every other route). The arrival
replay is not repeated; the first attempt's verdict stands. Landing takes
the union: both attempts' checkpoints, the steps summed, the resumed run's
verdict, and the joined chain uploaded as `chain.union.jsonl`, which every
later segment is then held to or replays against. `"resume": false` on a
segment (or on the spec) restarts it from its start instead. A segment
with no such checkpoint restarts from its start, as before. Live segments
are never resumed.

RENTALS. One-box segments go through tools/gemm_remote_leg.sh (nvidia, a
GPU-type walk) or, for amd, the providers named in `amd_providers`, walked in
order: `do` is tools/do_extra_leg.sh (DigitalOcean), `hotaisle` is
tools/hotaisle_leg.sh, anything else is gemm_remote_leg.sh amd (RunPod).
Live segments go through tools/lm_live_leg.sh. The runners own the
dead-men, the fetch and the verified delete. The driver never talks to a
cloud API itself; its only network use is R2 by presigned URLs minted by
tools/dataset_store.sh (a one-byte read to see a checkpoint is there, a PUT
of a chain).

AN AMD SEGMENT'S PROVIDER. A segment entry may name `"provider": "hotaisle"`
(or "do", "runpod"); it then rents from that provider only. Without it the
segment walks `amd_providers`. The vendor class is per box source: an amd
segment holds one class per provider it may use ("amd:do", "amd:hotaisle"),
so under `--parallel 2` a DigitalOcean AMD segment and a Hot Aisle AMD
segment run at once, while a segment that walks both providers holds both.

HOT AISLE SPEC KEYS (all optional):
    "hotaisle_devices": "0,1"          # the body's --devices on the 2x MI300X VM; `amd_devices` is
                                       # spec-wide (sized for the DigitalOcean box), so the segment
                                       # body is rendered again for hotaisle with these
    "hotaisle_extra_minutes": 30       # added to the segment lease: provisioning, the image pull,
                                       # the wheel install and the fetch
    "hotaisle_dollar_cap": 250         # the cap for a hotaisle lease (default the segment's cap); the
                                       # leg prices the offering live and refuses above it
    "hotaisle_stock_wait_minutes": 5   # how long the leg waits for stock or a slot before it says
                                       # busy and the driver walks on
The leg rents the 2x MI300X VM with ONE body on both GPUs
(`hotaisle_leg.sh --spec 2gpu --one-body`). A hotaisle lease over its cap is
skipped like a busy provider, never raised. A RESUMED segment on Hot Aisle is
rendered again the same way: from its own checkpoint with the remaining steps
and the partial chain as its expect chain, on `hotaisle_devices`.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
REPO = Path(__file__).resolve().parents[1]


def _log(out, msg):
    line = "[%s driver] %s" % (time.strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(Path(out) / "driver.log", "a") as fh:
        fh.write(line + "\n")


def load_spec(path):
    spec = json.loads(Path(path).read_text())
    for route, segs in spec["routes"].items():
        for i, s in enumerate(segs):
            s.setdefault("index", i + 1)
    return spec


def segment_plan(spec):
    """Every (route, segment) with its global step range, dependencies and
    the checkpoints it starts from and replays from."""
    plan = []
    for route, segs in spec["routes"].items():
        at = 0
        for s in segs:
            first, last = at, at + int(s["steps"])
            prev = segs[s["index"] - 2] if s["index"] > 1 else None
            a_first = spec["routes"]["A"][0]["segment"]
            # the seed: route A's first segment draws it on its box; every other
            # route's first segment starts from THAT file, never a seed of its own
            seed_from_a = route != "A" and prev is None
            # a route in "own_chain" (spec) chains from ITS OWN previous
            # segment, independent of route A's checkpoints; it is still held
            # to route A's chain for the same segment, so both runs must end
            # at the same bits. Every other route hangs off route A's
            # checkpoints (the pipeline of the plan, section 4; route C).
            src = route if (route == "A" or route in spec.get("own_chain", [])) else "A"
            entry = dict(route=route, segment=s["segment"], index=s["index"], vendor=s["vendor"], steps=int(s["steps"]),
                         hold=s.get("hold") or None,
                         first=first, last=last, boundary=last, live=s.get("first"), shards=s.get("shards"),
                         lease_minutes=s.get("lease_minutes"), dollar_cap=s.get("dollar_cap"),
                         nvidia_devices=s.get("nvidia_devices"), nvidia_gpus=s.get("nvidia_gpus"),
                         provider=s.get("provider"), hotaisle_dollar_cap=s.get("hotaisle_dollar_cap"),
                         from_route=(src if prev else "A") if (prev or seed_from_a) else None,
                         from_segment=(prev["segment"] if prev else a_first) if (prev or seed_from_a) else None,
                         from_ckpt=("ckpt_%08d.blm" % first) if (prev or seed_from_a) else "init",
                         replay_ckpt=("ckpt_%08d.blm" % (first - 2)) if prev else None,
                         expect=(f"A/{s['segment']}/chain.jsonl" if route != "A" else None),
                         depends=[(src, prev["segment"])] if prev else ([("A", a_first)] if seed_from_a else []))
            if route != "A" and ("A", s["segment"]) not in entry["depends"]:
                entry["depends"].append(("A", s["segment"]))  # B's chain is held to A's; A's segment must exist
            # "after": ["A/3"] on a segment: it also waits for those segments to
            # land. One AMD box on the account means route B's AMD segments
            # would otherwise hold it for days while route A's live segment
            # waits for its AMD worker; route A's path comes first.
            for dep in s.get("after", []):
                r, _, sg = str(dep).partition("/")
                if (r, sg) not in entry["depends"]:
                    entry["depends"].append((r, sg))
            plan.append(entry)
            at = last
    return plan


def cmd_plan(args):
    spec = load_spec(args.spec)
    for e in segment_plan(spec):
        print("%s/%s %-7s steps %d..%d from %s%s%s%s%s%s" % (
            e["route"], e["segment"], e["vendor"], e["first"], e["last"],
            e["from_ckpt"] if e["from_ckpt"] == "init" else "%s/%s/%s" % (e["from_route"], e["from_segment"], e["from_ckpt"]),
            " replay " + e["replay_ckpt"] if e["replay_ckpt"] else "",
            " expect " + e["expect"] if e["expect"] else "",
            "  devices " + e["nvidia_devices"] if e.get("nvidia_devices") else "",
            "  gpus " + e["nvidia_gpus"] if e.get("nvidia_gpus") else "",
            "  provider " + e["provider"] if e.get("provider") else ""))
        extra = [d for d in e["depends"] if d not in {(e["from_route"], e["from_segment"]), ("A", e["segment"])}]
        if extra:
            print("      after %s" % ", ".join("%s/%s" % d for d in extra))
        if e.get("hold"):
            print("      HELD: %s" % e["hold"])
    return 0


# ---------------------------------------------------------------- the ledger

class Ledger:
    def __init__(self, out):
        self.path = Path(out) / "ledger.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"landed": {}}

    def key(self, e):
        return "%s/%s" % (e["route"], e["segment"])

    def landed(self, e):
        return self.data["landed"].get(self.key(e))

    def land(self, e, record):
        self.data["landed"][self.key(e)] = record
        self._save()

    def partial(self, e):
        """What a segment that did not land left behind (see record_partial)."""
        return self.data.get("partial", {}).get(self.key(e))

    def set_partial(self, e, record):
        self.data.setdefault("partial", {})[self.key(e)] = record
        self._save()

    def _save(self):
        self.path.write_text(json.dumps(self.data, indent=1) + "\n")


def _manifest(results_dir):
    """name -> (bytes, sha256) from a fetched segment's manifest.tsv."""
    rows = {}
    for p in Path(results_dir).rglob("manifest.tsv"):
        if p.parent.name == "segment":
            for line in p.read_text().splitlines():
                if line.strip():
                    name, size, digest = line.split("\t")
                    rows[name] = (int(size), digest)
    return rows


def _this_attempt(results_dir, name):
    """Every `name` under `results_dir` that belongs to THIS attempt: a path
    under a `previous-*` directory is an earlier attempt's, moved aside by
    tools/lm_live_leg.sh, and is never read as this one's."""
    for p in sorted(Path(results_dir).rglob(name)):
        if not any(part.startswith("previous-") for part in p.relative_to(results_dir).parts):
            yield p


def _segment_dir(results_dir):
    """The `segment/` directory whose segment.json lands the segment. A live
    segment brings two home (the coordinator's, which holds the checkpoints
    and the arrival replay beside it, and the worker's); the coordinator's is
    the one. T2 segment 3 once landed from the worker's (no checkpoints) with
    the arrival verdict of a previous attempt."""
    found = []
    for p in _this_attempt(results_dir, "segment.json"):
        if p.parent.name == "segment":
            seg = json.loads(p.read_text())
            live = seg.get("live") or {}
            if live.get("role") in (None, "coordinator"):
                found.append((seg.get("utc_start") or "", p.parent, seg))
    if not found:
        return None, None
    found.sort(key=lambda t: (t[0], str(t[1])))
    return found[-1][1], found[-1][2]


def _segment_json(results_dir):
    return _segment_dir(results_dir)[1]


def _worker_verdicts(results_dir, since):
    """label -> verdict of every live worker's segment.json that ended after
    `since` (the coordinator's start). A worker directory an earlier attempt
    left in place (its box fetched, the directory not moved aside) ended
    before this coordinator started, and is not this attempt's."""
    out = {}
    for p in _this_attempt(results_dir, "segment.json"):
        if p.parent.name == "segment":
            seg = json.loads(p.read_text())
            if (seg.get("live") or {}).get("role") == "worker" and (seg.get("utc_end") or "") >= (since or ""):
                out[seg.get("label") or str(p.parent.parent)] = seg.get("verdict")
    return out


def _status(results_dir):
    seg_dir, _ = _segment_dir(results_dir)
    if seg_dir is not None and (seg_dir.parent / "status.txt").exists():
        return (seg_dir.parent / "status.txt").read_text()
    for p in _this_attempt(results_dir, "status.txt"):
        return p.read_text()
    return ""


# ---------------------------------------------------------------- resuming a segment that did not land

ATTEMPT_FILE_CAP = 8_000_000  # bytes; an attempt's files above this stay where they are (checkpoints live in R2)


def _ckpt(step):
    return "ckpt_%08d.blm" % step


def _step_of(name):
    return int(name.split("_")[1].split(".")[0])


def r2_bytes(key, seconds=900):
    """The size of an R2 object by one byte of a presigned GET (a presigned
    URL refuses HEAD), or None when it cannot be read. The URL is minted on
    this machine by tools/dataset_store.sh, as every body's are."""
    from lm_segment_leg import presign_get
    try:
        url = presign_get(key, seconds)
    except subprocess.CalledProcessError:
        return None
    r = subprocess.run(["curl", "-fsS", "--max-time", "60", "-r", "0-0", "-D", "-", "-o", os.devnull, url],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    for line in (r.stdout or "").splitlines():
        if line.lower().startswith("content-range:"):
            return int(line.rsplit("/", 1)[1])
    return None


def r2_put(key, path, seconds=900):
    """Upload one small file to R2 by a presigned PUT; True when curl says so."""
    from lm_segment_leg import presign_put
    try:
        url = presign_put(key, seconds)
    except subprocess.CalledProcessError:
        return False
    return subprocess.run(["curl", "-fsS", "--retry", "3", "-T", str(path), url],
                          capture_output=True, text=True).returncode == 0


def _chain_lines(path, linked=True):
    """[(step, line)] of a chain a box wrote, in order, byte-exact. Reading
    stops at the first line that does not parse (a line cut by the kill),
    does not follow its predecessor's step, or (linked) does not carry its
    predecessor's sha256 in `prev`, as tools/lm_segment.py writes them."""
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for text in p.read_text(errors="replace").splitlines():
        if not text.strip():
            continue
        try:
            row = json.loads(text)
            step = int(row["step"])
        except (ValueError, KeyError, TypeError):
            break
        if out and (step != out[-1][0] + 1 or
                    (linked and row.get("prev") != hashlib.sha256(out[-1][1].encode()).hexdigest())):
            break
        out.append((step, text))
    return out


def _write_chain(path, lines):
    Path(path).write_text("".join(text + "\n" for _, text in lines))


def _resume_allowed(spec, e):
    segs = spec["routes"].get(e["route"]) or []
    s = segs[e["index"] - 1] if 0 < e.get("index", 0) <= len(segs) else {}
    return bool(s.get("resume", spec.get("resume", True)))


def resume_point(spec, e, ledger, out=None):
    """The ledger's partial entry when this segment resumes from it, else None."""
    part = ledger.partial(e)
    if not part or e["vendor"] == "live":
        return None
    if not _resume_allowed(spec, e):
        if out is not None:
            _log(out, "%s/%s: \"resume\": false; restarting from %s although %s landed"
                 % (e["route"], e["segment"], e["from_ckpt"], _ckpt(part["resume_from"])))
        return None
    if not (e["first"] < part["resume_from"] < e["last"]):
        return None
    if out is not None:
        _log(out, "%s/%s: RESUMING from its own %s (%d of %d steps landed), %d steps left"
             % (e["route"], e["segment"], _ckpt(part["resume_from"]), part["resume_from"] - e["first"], e["steps"],
                e["last"] - part["resume_from"]))
    return part


def _attempt_box(results):
    """The box directory (holding segment/) of THIS attempt's chain, or None."""
    for p in _this_attempt(results, "chain.jsonl"):
        if p.parent.name == "segment":
            return p.parent.parent
    return None


def record_partial(spec, e, results, out, ledger):
    """A one-box segment brought no segment.json home: record what it landed.

    A checkpoint counts as landed when the segment's manifest.tsv pins it,
    its log.txt says `uploaded <name>`, the box's checkpoints.sha256 (when
    the body lived to write it) carries the same sha256, R2 holds an object
    of the manifest's size under `<run>/<route>/<segment>/<name>` (unless
    the spec says `"resume_check_r2": false`), and the chain has at least
    one line after it (the resumed box must re-derive a step the hung box
    wrote). The next box fetches it by that sha256, so a wrong object is
    refused on the box before it trains. Returns the partial entry or None."""
    key = "%s/%s" % (e["route"], e["segment"])
    if e["vendor"] == "live":
        _log(out, "%s: a live segment is not resumed; it restarts from %s" % (key, e["from_ckpt"]))
        return None
    box = _attempt_box(results)
    prev = ledger.partial(e)
    start = prev["resume_from"] if prev else e["first"]
    if box is None:
        _log(out, "%s: no chain came home; nothing to resume from" % key)
        return None
    lines = _chain_lines(box / "segment" / "chain.jsonl")
    if not lines or lines[0][0] != start + 1:
        _log(out, "%s: the fetched chain does not start at step %d; it is not this attempt's, nothing recorded" % (key, start + 1))
        return None
    last_line = lines[-1][0]
    pinned = {}
    mf = box / "segment" / "manifest.tsv"
    if mf.exists():
        for row in mf.read_text().splitlines():
            parts = row.split("\t")
            if len(parts) == 3 and parts[0].startswith("ckpt_"):
                pinned[parts[0]] = (int(parts[1]), parts[2])
    log_txt = box / "segment" / "log.txt"
    uploaded = set(re.findall(r"uploaded (ckpt_\d{8}\.blm) in", log_txt.read_text(errors="replace"))) if log_txt.exists() else set()
    on_box = None
    if (box / "checkpoints.sha256").exists():
        on_box = {}
        for row in (box / "checkpoints.sha256").read_text().splitlines():
            if row.strip():
                digest, path = row.split(None, 1)
                on_box[Path(path.strip()).name] = digest
    landed, refused = {}, {}
    for name in sorted(pinned):
        step = _step_of(name)
        if not (start < step < e["last"] and step < last_line):
            continue
        size, digest = pinned[name]
        if name not in uploaded:
            refused[name] = "the log has no upload line"
        elif on_box is not None and on_box.get(name) != digest:
            refused[name] = "checkpoints.sha256 says %s, the manifest %s" % (on_box.get(name), digest)
        elif spec.get("resume_check_r2", True) and r2_bytes("%s/%s/%s" % (spec["run"], key, name)) != size:
            refused[name] = "R2 has no object of %d bytes under its key" % size
        else:
            landed[name] = digest
    for name, why in refused.items():
        _log(out, "%s: %s does not count as landed: %s" % (key, name, why))
    arrival = prev["arrival"] if prev else None
    if not prev and e["replay_ckpt"]:
        aj = box / "arrival" / "segment.json"
        arrival = json.loads(aj.read_text()).get("verdict") if aj.exists() else None
        if arrival != "PASS":
            _log(out, "%s: the arrival replay did not pass on this attempt (%s); nothing to resume" % (key, arrival))
            return None
    seed = {}
    if e["from_ckpt"] == "init" and not prev and (box / "seed.sha256").exists() \
            and "upload seed exit=0" in ((box / "status.txt").read_text() if (box / "status.txt").exists() else ""):
        # the first segment drew and uploaded the seed; later routes start from it
        seed[_ckpt(0)] = (box / "seed.sha256").read_text().split()[0]
    seg_root = Path(out) / "legs" / ("%s-%s" % (e["route"], e["segment"]))
    # numbered past every attempt directory already there (an attempt that
    # landed nothing keeps its files but has no ledger entry)
    n = 1
    while (seg_root / ("attempt-%d" % n)).exists():
        n += 1
    dest = seg_root / ("attempt-%d" % n)
    # the attempt's small files in the box's layout (lm_file_evidence reads
    # them so), then the runner's top-level files where the box has none
    for p in sorted(box.rglob("*")):
        rel = p.relative_to(box)
        if p.is_file() and rel.parts[0] != "in" and p.stat().st_size <= ATTEMPT_FILE_CAP:
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dest / rel)
    for p in sorted(Path(results).iterdir()):
        if p.is_file() and not (dest / p.name).exists() and p.stat().st_size <= ATTEMPT_FILE_CAP:
            shutil.copyfile(p, dest / p.name)
    attempt = dict(n=n, dir=str(dest), first_step=start, last_chain_step=last_line, checkpoints=landed,
                   refused=refused, utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    # the results directory is moved aside so the next rental fetches into a
    # clean one; _this_attempt never reads a `previous-*` directory
    if Path(results).resolve() != seg_root.resolve() and Path(results).exists():
        aside = Path(results).parent / ("previous-attempt-%d-%s" % (n, Path(results).name))
        k = 1
        while aside.exists():
            k += 1
            aside = Path(results).parent / ("previous-attempt-%d.%d-%s" % (n, k, Path(results).name))
        Path(results).rename(aside)
        attempt["results"] = str(aside)
    if not landed:
        if prev:
            prev["attempts"].append(attempt)
            ledger.set_partial(e, prev)
            _log(out, "%s: attempt %d landed no new checkpoint; the next start resumes from %s again"
                 % (key, n, _ckpt(prev["resume_from"])))
            return prev
        _log(out, "%s: no checkpoint past step %d landed; the next start restarts from %s (attempt files in %s)"
             % (key, start, e["from_ckpt"], dest))
        return None
    resume_from = max(_step_of(name) for name in landed)
    earlier = [l for l in _chain_lines(prev["chain"], linked=False) if l[0] <= start] if prev else []
    chain = seg_root / "partial.chain.jsonl"
    _write_chain(chain, earlier + lines)
    record = dict(resume_from=resume_from, checkpoints=dict(prev["checkpoints"] if prev else {}, **seed, **landed),
                  chain=str(chain), chain_steps=[(earlier or lines)[0][0], last_line], arrival=arrival,
                  attempts=(prev["attempts"] if prev else []) + [attempt])
    ledger.set_partial(e, record)
    _log(out, "%s: PARTIAL: checkpoints %s landed, chain to step %d saved (%s); the next start resumes from %s with %d steps"
         % (key, sorted(landed), last_line, chain, _ckpt(resume_from), e["last"] - resume_from))
    return record


# ---------------------------------------------------------------- rendering and renting

NVIDIA_GPUS = "NVIDIA H100 80GB HBM3|NVIDIA H200|NVIDIA H100 PCIe|NVIDIA H100 NVL"


def _nvidia_devices(spec, e):
    """The segment's own NVIDIA devices, else the run's."""
    return e.get("nvidia_devices") or spec.get("nvidia_devices", "0")


def _nvidia_gpus(spec, e):
    """The segment's own NVIDIA GPU types (a | list), else the run's."""
    return e.get("nvidia_gpus") or spec.get("nvidia_gpus", NVIDIA_GPUS)


def _chain_key(run, ledger, route, segment):
    """The R2 key of a landed segment's whole chain: its box's chain.jsonl,
    or for a resumed segment the joined chain the landing uploaded. A resumed
    segment whose joined chain is not in R2 is refused by name: holding a
    later segment to the resumed box's part alone would compare fewer steps."""
    rec = ledger.landed(dict(route=route, segment=segment)) or {}
    if rec.get("resumed_from") is not None:
        if not rec.get("chain_key"):
            raise SystemExit("%s/%s was resumed and its joined chain is not in R2; run `reland --segment %s/%s`"
                             % (route, segment, route, segment))
        return rec["chain_key"]
    return "%s/%s/%s/chain.jsonl" % (run, route, segment)


def render(spec, e, out, ledger, role=None, suffix=""):
    """One body for a segment (or one side of a live segment). A one-box
    segment with a `partial` ledger entry (and resuming allowed) runs only
    its remaining steps, from its own last landed checkpoint. `suffix` names a
    second rendering of the same segment for another box (Hot Aisle's
    devices); it resumes exactly as the first, and neither logs the resume
    nor uploads the partial chain again."""
    run = spec["run"]
    arm = e["vendor"] if role is None else role
    mode = "one" if role is None else ("live-coordinator" if role == e["live"] else "live-worker")
    devices = _nvidia_devices(spec, e) if arm == "nvidia" else spec.get("amd_devices", "0")
    label = "%s-%s-%s" % (arm, e["route"], e["segment"])
    resume = resume_point(spec, e, ledger, None if suffix else out) if role is None else None
    from_ckpt, steps = e["from_ckpt"], e["steps"]
    if resume:
        from_ckpt = _ckpt(resume["resume_from"])
        steps = e["last"] - resume["resume_from"]
    cmd = [sys.executable, str(REPO / "tools" / "lm_segment_leg.py"), "render", "--run", run, "--recipe", spec["recipe"],
           "--recipe-key", spec["recipe_key"], "--arm", arm, "--mode", mode, "--devices", devices,
           "--tokens-key", spec["tokens_stage"],
           *(["--wheel", spec["wheel"]] if spec.get("wheel") else []),
           "--route", e["route"], "--segment", e["segment"], "--label", label, "--steps", str(steps),
           "--from", from_ckpt, "--seconds", str(spec.get("url_seconds", 8 * 3600))]
    if mode != "live-worker":
        cmd += ["--boundary", str(e["boundary"])]
    if resume:
        # the segment's OWN checkpoint, by the sha256 its box pinned; the
        # arrival replay passed on the first attempt and is not repeated:
        # the resumed box is held to the chain instead
        cmd += ["--from-sha", resume["checkpoints"][from_ckpt],
                "--from-key", "%s/%s/%s/%s" % (run, e["route"], e["segment"], from_ckpt)]
        if e["route"] == "A":
            key = "%s/%s/%s/partial.chain.jsonl" % (run, e["route"], e["segment"])
            # a second rendering (suffix) follows a first whose upload succeeded
            if not suffix and not r2_put(key, resume["chain"]):
                raise SystemExit("%s/%s: the partial chain could not be uploaded to %s; not resuming blind" % (e["route"], e["segment"], key))
            cmd += ["--expect-key", key]
        else:
            cmd += ["--expect-key", _chain_key(run, ledger, "A", e["segment"])]
    elif e["from_ckpt"] != "init":
        src = ledger.landed(dict(route=e["from_route"], segment=e["from_segment"]))
        if not src:
            raise SystemExit("%s/%s has not landed" % (e["from_route"], e["from_segment"]))
        cmd += ["--from-sha", src["checkpoints"][e["from_ckpt"]], "--from-key", "%s/%s/%s/%s" % (run, e["from_route"], e["from_segment"], e["from_ckpt"])]
        # a route's first segment seeded from route A's file has no previous
        # segment of its own and so no arrival replay
        if mode != "live-worker" and e["replay_ckpt"]:
            cmd += ["--replay", e["replay_ckpt"], "--replay-sha", src["checkpoints"][e["replay_ckpt"]],
                    "--replay-key", "%s/%s/%s/%s" % (run, e["from_route"], e["from_segment"], e["replay_ckpt"]),
                    "--replay-chain-key", _chain_key(run, ledger, e["from_route"], e["from_segment"])]
    if e["expect"] and not resume:
        route, _, rest = e["expect"].partition("/")
        cmd += ["--expect-key", _chain_key(run, ledger, route, rest.partition("/")[0])]
    if role is not None:
        n_first, n_second = e["shards"]
        K = n_first + n_second
        block = "0:%d" % n_first if role == e["live"] else "%d:%d" % (n_first, K)
        cmd += ["--live-shards", block, "--live-workers", "2", "--live-port", str(spec.get("live_port", 7777))]
    body = Path(out) / "bodies" / ("%s-%s%s%s.sh" % (e["route"], e["segment"], "-" + arm if role else "", suffix))
    body.parent.mkdir(parents=True, exist_ok=True)
    cmd += ["--out", str(body)]
    subprocess.run(cmd, check=True, cwd=REPO)
    return body


def rent_one(spec, e, body, out, rerender=None):
    """Rent one box for a one-box segment; returns the results directory and the runner's exit code."""
    res = Path(out) / "legs" / ("%s-%s" % (e["route"], e["segment"]))
    res.mkdir(parents=True, exist_ok=True)
    # the body fetches the token parts itself (parallel, timed); the runners stage nothing
    env = dict(os.environ, MOJOLEARN_GEMM_LEG_EXTRA=str(body), MOJOLEARN_STAGE_KEYS=spec.get("runner_stage_keys", ""),
               MOJOLEARN_GEMM_LEG_OUT=str(res / "leg"),
               # a recorded Apple card: the NVIDIA runner would otherwise take one on this Mac and refuse under the Metal lock
               MOJOLEARN_GEMM_LEG_LOCAL_CARD=spec.get("apple_card", os.path.expanduser("~/mojolearn-evidence/vendor-class-gaps-sep19/apple.card")))
    # a segment's own lease and cap win over the run's (an AMD segment on an
    # 8-GPU box needs both bigger); the driver never rents without a cap
    minutes = int(e.get("lease_minutes") or spec.get("lease_minutes", 120))
    cap = str(e.get("dollar_cap") or spec.get("dollar_cap", 10))
    lease = ["--segment-lease", str(minutes), "--dollar-cap", cap] if minutes > 60 else ["--minutes", str(minutes)]
    if e["vendor"] == "nvidia":
        rc = 3
        for gpu in _nvidia_gpus(spec, e).split("|"):
            slug = "".join(c for c in gpu.replace(" ", "_") if c.isalnum() or c == "_")
            env["MOJOLEARN_GEMM_LEG_OUT"] = str(res / ("leg-" + slug))
            if e.get("nvidia_devices"):
                # the segment names its devices: rent exactly that many GPUs,
                # whatever the run's devices or the caller's environment say
                env["MOJOLEARN_GEMM_LEG_GPU_COUNT"] = str(len(e["nvidia_devices"].split(",")))
            elif spec.get("nvidia_devices", "0").count(",") == 1:
                env["MOJOLEARN_GEMM_LEG_GPU_COUNT"] = "2"
            with open(res / ("leg-%s.log" % slug), "w") as log:
                rc = subprocess.run(["sh", "tools/gemm_remote_leg.sh", "nvidia", "--rent", "--allow-concurrent", *lease, "--gpu", gpu],
                                    cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            cr = res / ("leg-" + slug) / "create_response.json"
            if cr.exists() and "no instances currently available" in cr.read_text():
                _log(out, "nvidia: no capacity for %s" % gpu)
                continue
            return res / ("leg-" + slug), rc
        return res, rc
    env["MOJOLEARN_GPU_ARCHS"] = "gfx942"
    providers = _amd_providers(spec, e)
    rc = 3
    # AMD capacity is the choke point (one DigitalOcean GPU droplet per account,
    # RunPod MI300X and Hot Aisle often without stock): walk the providers, and
    # when every one is busy, wait and walk again, for up to amd_wait_minutes.
    deadline = time.monotonic() + 60 * int(spec.get("amd_wait_minutes", 180))
    attempt = 0
    while True:
        attempt += 1
        result = _rent_amd_once(spec, e, res, env, lease, providers, out, attempt, rerender=rerender)
        if result is not None:
            return result
        if time.monotonic() > deadline:
            _log(out, "amd: no provider had capacity within amd_wait_minutes; giving up on this attempt")
            return res, rc
        _log(out, "amd: every provider busy or without capacity; waiting 10 minutes (attempt %d)" % attempt)
        time.sleep(600)


HOTAISLE_DEFAULT_DEVICES = "0,1"

# What an AMD runner prints when the provider is busy, without capacity or
# short of balance: the driver walks to the next provider (and waits and
# walks again) instead of halting. do_extra_leg.sh prints "ONE GPU droplet at
# a time" and "REFUSING to create"; hotaisle_leg.sh prints "no stock", "no
# slot freed", "REFUSED: balance" (below the floor or the whole lease) and
# "create REFUSED by the API" (the team's VM limit, balance or stock at the
# create).
AMD_BUSY_PHRASES = ("ONE GPU droplet at a time", "no instances currently available", "REFUSING to create",
                    "quantity=0", "no slot freed", "REFUSED: balance", "create REFUSED by the API")
AMD_BUSY_PHRASES_LOWER = ("no stock", "starved")


def _amd_busy(text):
    low = text.lower()
    return any(p in text for p in AMD_BUSY_PHRASES) or any(p in low for p in AMD_BUSY_PHRASES_LOWER)


def _amd_providers(spec, e):
    """The providers an amd segment rents from: its own `provider`, else the run's walk."""
    if e.get("provider"):
        return [e["provider"]]
    return list(spec.get("amd_providers") or [spec.get("amd_provider", "do")])


def _hotaisle_lease(spec, e):
    """The hotaisle lease: the segment's minutes plus hotaisle_extra_minutes (provisioning,
    the pull, the wheel, the fetch), capped by hotaisle_dollar_cap or the segment's own cap.
    The 2x MI300X offering's minimum reservation is 60 minutes, so a shorter lease is 60."""
    minutes = int(e.get("lease_minutes") or spec.get("lease_minutes", 120)) + int(spec.get("hotaisle_extra_minutes", 30))
    cap = str(e.get("hotaisle_dollar_cap") or spec.get("hotaisle_dollar_cap") or e.get("dollar_cap") or spec.get("dollar_cap", 10))
    return ["--segment-lease", str(minutes), "--dollar-cap", cap] if minutes > 60 else ["--minutes", "60"]


def _rent_amd_once(spec, e, res, env, lease, providers, out, attempt, rerender=None):
    rc = 3
    for provider in providers:
        tag = "%s-%d" % (provider, attempt)
        env["MOJOLEARN_GEMM_LEG_OUT"] = str(res / ("leg-" + tag))
        with open(res / ("leg-%s.log" % tag), "w") as log:
            if provider == "do":
                extra = []
                if spec.get("amd_size"):
                    extra += ["--size", spec["amd_size"]]
                if spec.get("amd_region"):
                    extra += ["--region", spec["amd_region"]]
                rc = subprocess.run(["bash", "tools/do_extra_leg.sh", "amd", *lease, *extra], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            elif provider == "hotaisle":
                # the 2x MI300X VM, ONE body on both GPUs, for the whole segment: the body is
                # rendered again with hotaisle_devices (amd_devices is spec-wide)
                henv = dict(env)
                devices = spec.get("hotaisle_devices", HOTAISLE_DEFAULT_DEVICES)
                if rerender is not None and devices != spec.get("amd_devices", "0"):
                    henv["MOJOLEARN_GEMM_LEG_EXTRA"] = str(rerender(devices))
                wait = str(spec.get("hotaisle_stock_wait_minutes", 5))
                henv.update(MOJOLEARN_HOTAISLE_SPEC="2gpu", MOJOLEARN_HOTAISLE_GPU_ONLY="1",
                            MOJOLEARN_HOTAISLE_LANE="lm-%s-%s" % (e["route"], e["segment"]),
                            MOJOLEARN_HOTAISLE_STOCK_WAIT_MINUTES=wait, MOJOLEARN_HOTAISLE_SLOT_WAIT_MINUTES=wait)
                rc = subprocess.run(["bash", "tools/hotaisle_leg.sh", "amd", "--rent", "--skip-gates", "--spec", "2gpu",
                                     "--one-body", *_hotaisle_lease(spec, e)],
                                    cwd=REPO, env=henv, stdout=log, stderr=subprocess.STDOUT).returncode
            else:
                rc = subprocess.run(["sh", "tools/gemm_remote_leg.sh", "amd", "--rent", "--allow-concurrent", *lease], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        text = (res / ("leg-%s.log" % tag)).read_text()
        if rc != 0 and _amd_busy(text):
            _log(out, "amd: %s is busy or without capacity (attempt %d)" % (provider, attempt))
            continue
        if rc != 0 and provider == "hotaisle" and "above the --dollar-cap" in text:
            _log(out, "amd: hotaisle's lease is over its cap (hotaisle_dollar_cap); skipping hotaisle (attempt %d)" % attempt)
            continue
        return res / ("leg-" + tag), rc
    return None


def rent_live(spec, e, nv_body, amd_body, out):
    res = Path(out) / "legs" / ("%s-%s-live" % (e["route"], e["segment"]))
    res.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MOJOLEARN_LIVE_STAGE_KEYS=spec.get("runner_stage_keys", ""),
               MOJOLEARN_LIVE_NVIDIA_GPUS=_nvidia_gpus(spec, e))
    if e.get("nvidia_devices"):
        # the coordinator's NVIDIA pod follows the segment's devices; the AMD box is untouched
        env["MOJOLEARN_LIVE_NVIDIA_GPU_COUNT"] = str(len(e["nvidia_devices"].split(",")))
    with open(res / "live.log", "w") as log:
        # a live segment's own lease and cap win over the run's, as a one-box
        # segment's do (route B's live segment 2 is 1,000 steps, 29 hours)
        rc = subprocess.run(["bash", "tools/lm_live_leg.sh", str(res), str(nv_body), str(amd_body), "--minutes",
                             str(e.get("lease_minutes") or spec.get("lease_minutes", 120)),
                             "--dollar-cap", str(e.get("dollar_cap") or spec.get("dollar_cap", 10)),
                             "--amd", spec.get("amd_provider", "do")], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return res, rc


# ---------------------------------------------------------------- landing

def land(spec, e, results, out, ledger):
    """Read the fetched results; PASS lands the segment with its checkpoint hashes."""
    seg_dir, seg = _segment_dir(results)
    status = _status(results)
    if seg is None:
        _log(out, "%s/%s: no segment.json came home; status:\n%s" % (e["route"], e["segment"], status[-800:]))
        record_partial(spec, e, results, out, ledger)
        return None
    if e["vendor"] != "live" and seg.get("first_step") is not None and int(seg["first_step"]) != e["first"]:
        return _land_resumed(spec, e, results, out, ledger, seg_dir, seg)
    verdict = seg.get("verdict")
    ckpts = {c["file"]: c["sha256"] for c in seg.get("checkpoints", [])}
    if e["from_ckpt"] == "init":
        for p in Path(results).rglob("seed.sha256"):
            digest, _ = p.read_text().split()
            ckpts["ckpt_00000000.blm"] = digest
    record = dict(verdict=verdict, checkpoints=ckpts, steps_completed=seg.get("steps_completed"),
                  disagreements=seg.get("disagreements"), results=str(results), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    arrival = None
    arrival_json = seg_dir.parent / "arrival" / "segment.json"
    if arrival_json.exists():
        arrival = json.loads(arrival_json.read_text()).get("verdict")
    record["arrival"] = arrival
    workers = _worker_verdicts(results, seg.get("utc_start"))
    if workers:
        record["workers"] = workers
        if any(v != "PASS" for v in workers.values()):
            verdict = record["verdict"] = "FAIL-WORKER"
    if verdict != "PASS" or (e["replay_ckpt"] and arrival != "PASS"):
        _log(out, "%s/%s: verdict %s, arrival %s; THE RUN HALTS HERE (%s)" % (e["route"], e["segment"], verdict, arrival, seg.get("disagreements")))
        ledger.land(e, record)
        return False
    if e["route"] != "A":
        a = ledger.landed(dict(route="A", segment=e["segment"]))
        if a:
            boundary = "ckpt_%08d.blm" % e["last"]
            if a["checkpoints"].get(boundary) != ckpts.get(boundary):
                _log(out, "%s/%s: boundary checkpoint %s differs from route A's; THE RUN HALTS HERE" % (e["route"], e["segment"], boundary))
                record["verdict"] = "FAIL-BOUNDARY"
                ledger.land(e, record)
                return False
    ledger.land(e, record)
    _log(out, "%s/%s LANDED: %d steps, checkpoints %s" % (e["route"], e["segment"], seg.get("steps_completed") or 0, sorted(ckpts)))
    return True


def _land_resumed(spec, e, results, out, ledger, seg_dir, seg):
    """Land a segment whose box started from its own partial checkpoint: the
    union of every attempt. The resumed box's verdict decides; its loaded
    checkpoint's state must equal the chain's line at that step (lm_segment
    compared it, `from_state`), the arrival verdict is the first attempt's,
    and the joined chain (the earlier attempts' lines up to the checkpoint,
    then the resumed box's) goes to R2 as chain.union.jsonl for every later
    segment to be held to."""
    key = "%s/%s" % (e["route"], e["segment"])
    part = ledger.partial(e)
    first = int(seg["first_step"])
    record = dict(verdict=seg.get("verdict"), steps_completed=None, disagreements=seg.get("disagreements"),
                  results=str(results), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), resumed_from=first)
    if not part or part["resume_from"] != first:
        record.update(verdict="FAIL-RESUME", checkpoints={c["file"]: c["sha256"] for c in seg.get("checkpoints", [])})
        _log(out, "%s: the box started at step %d and the ledger has no partial entry at that step; THE RUN HALTS HERE" % (key, first))
        ledger.land(e, record)
        return False
    ckpts = dict(part["checkpoints"])
    ckpts.update({c["file"]: c["sha256"] for c in seg.get("checkpoints", [])})
    record.update(checkpoints=ckpts, arrival=part.get("arrival"), attempts=part["attempts"],
                  partial_chain=part["chain"],
                  steps_completed=(first - e["first"]) + int(seg.get("steps_completed") or 0))
    verdict = record["verdict"]
    fs = seg.get("from_state") or {}
    if verdict == "PASS" and not fs.get("agrees"):
        verdict = record["verdict"] = "FAIL-RESUME"
        record["disagreements"] = (record["disagreements"] or []) + [dict(
            reason="the resumed checkpoint's state was not shown equal to the chain's line at step %d" % first, from_state=fs)]
    if verdict != "PASS" or (e["replay_ckpt"] and record["arrival"] != "PASS"):
        _log(out, "%s: resumed from %d, verdict %s, arrival %s; THE RUN HALTS HERE (%s)"
             % (key, first, verdict, record["arrival"], record["disagreements"]))
        ledger.land(e, record)
        return False
    if e["route"] != "A":
        a = ledger.landed(dict(route="A", segment=e["segment"]))
        if a:
            boundary = _ckpt(e["last"])
            if a["checkpoints"].get(boundary) != ckpts.get(boundary):
                _log(out, "%s: boundary checkpoint %s differs from route A's; THE RUN HALTS HERE" % (key, boundary))
                record["verdict"] = "FAIL-BOUNDARY"
                ledger.land(e, record)
                return False
    earlier = [l for l in _chain_lines(part["chain"], linked=False) if l[0] <= first]
    resumed = _chain_lines(seg_dir / "chain.jsonl")
    joined = earlier + resumed
    steps = [s for s, _ in joined]
    last = seg.get("last_completed") or e["last"]
    if steps != list(range(e["first"] + 1, last + 1)):
        record["verdict"] = "FAIL-RESUME"
        record["disagreements"] = [dict(reason="the joined chain does not run step by step from %d to %d" % (e["first"] + 1, last),
                                        earlier=[earlier[0][0], earlier[-1][0]] if earlier else None,
                                        resumed=[resumed[0][0], resumed[-1][0]] if resumed else None)]
        _log(out, "%s: %s; THE RUN HALTS HERE" % (key, record["disagreements"][0]["reason"]))
        ledger.land(e, record)
        return False
    union = Path(out) / "legs" / ("%s-%s" % (e["route"], e["segment"])) / "chain.union.jsonl"
    union.parent.mkdir(parents=True, exist_ok=True)
    _write_chain(union, joined)
    record["chain"] = str(union)
    record["chain_sha256"] = hashlib.sha256(union.read_bytes()).hexdigest()
    chain_key = "%s/%s/chain.union.jsonl" % (spec["run"], key)
    if r2_put(chain_key, union):
        record["chain_key"] = chain_key
    else:
        _log(out, "%s: the joined chain did not upload to %s; later segments refuse until `reland --segment %s`" % (key, chain_key, key))
    ledger.land(e, record)
    _log(out, "%s LANDED (resumed from %s): %d steps, checkpoints %s" % (key, _ckpt(first), record["steps_completed"], sorted(ckpts)))
    return True


def _vendor_class(e, spec=None):
    """The classes a segment holds while it runs; two segments that share one never
    run at once. An amd segment holds one class per box source it may rent from
    ("amd:do", "amd:hotaisle"), so AMD segments on different providers run together."""
    if e["vendor"] == "live":
        return {"live"}
    if e["vendor"] == "amd":
        return {"amd:%s" % p for p in _amd_providers(spec or {}, e)}
    return {e["vendor"]}


def _ready_order(e):
    """Start order among ready segments: routes A and B first, smallest global
    step first; a further route (a replay such as route C) only takes a vendor
    class that no ready A or B segment wants."""
    return (e["route"] not in ("A", "B"), e["last"], e["route"])


def _start(spec, e, out, ledger):
    """Render and rent one segment; returns (results, rc) or raises."""
    if e["vendor"] == "live":
        nv_body = render(spec, e, out, ledger, role="nvidia")
        amd_body = render(spec, e, out, ledger, role="amd")
        return rent_live(spec, e, nv_body, amd_body, out)
    body = render(spec, e, out, ledger)
    # a provider whose box has another device count renders the body again with its devices
    rerender = lambda devices: render(dict(spec, amd_devices=devices), e, out, ledger, suffix="-hotaisle")  # noqa: E731
    return rent_one(spec, e, body, out, rerender=rerender)


def _plan_shape(plan):
    return [(e["route"], e["segment"], e["vendor"], e["steps"], e["first"], e["last"], tuple(e["depends"]), e.get("hold"))
            for e in plan]


def fresh_spec(path, plan):
    """The spec file read again when a segment starts, so that its rental
    parameters (wheel, amd_size, amd_devices, amd_providers, nvidia_gpus,
    caps) are the ones on disk now, not the ones at launch: a wheel with a
    fix, or the AMD size once a provider's limit lands, reach the next
    segment without a restart. The PLAN may not change under a run: a spec
    whose routes, vendors or step counts differ from the launch plan is
    refused by name and the segment is not started."""
    spec = load_spec(path)
    if _plan_shape(segment_plan(spec)) != _plan_shape(plan):
        raise RuntimeError("%s: the routes, vendors or step counts changed since the driver started; "
                           "restart the driver for a new plan" % path)
    return spec


def cmd_run(args):
    import threading
    spec = load_spec(args.spec)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out)
    lock = threading.Lock()
    plan = segment_plan(spec)
    parallel = int(getattr(args, "parallel", 1) or 1)
    _log(out, "run %s: %d segments over %d routes, up to %d at once" % (spec["run"], len(plan), len(spec["routes"]), parallel))

    def is_passed(route, segment):
        return (ledger.landed(dict(route=route, segment=segment)) or {}).get("verdict") == "PASS"

    running = {}   # key -> (thread, entry, result holder)
    halted = {"flag": False}

    def worker(e, holder):
        try:
            now = fresh_spec(args.spec, plan)
            changed = sorted(k for k in set(now) | set(spec) if k != "routes" and now.get(k) != spec.get(k))
            if changed:
                _log(out, "%s/%s: spec re-read, changed since launch: %s" % (
                    e["route"], e["segment"], ", ".join("%s=%s" % (k, json.dumps(now.get(k))) for k in changed)))
            holder["result"] = _start(now, e, out, ledger)
        except BaseException as exc:  # noqa: BLE001
            holder["error"] = exc

    while True:
        # collect finished segments
        for key in list(running):
            th, e, holder = running[key]
            if th.is_alive():
                continue
            del running[key]
            if "error" in holder:
                _log(out, "%s/%s: the rental could not be started: %s" % (e["route"], e["segment"], holder["error"]))
                halted["flag"] = True
                continue
            results, rc = holder["result"]
            _log(out, "%s/%s runner exit %s; results %s" % (e["route"], e["segment"], rc, results))
            with lock:
                ok = land(spec, e, results, out, ledger)
            if ok is False:
                halted["flag"] = True
            elif ok is None:
                part = resume_point(spec, e, ledger)
                _log(out, "%s/%s did not land (no results); run the driver again to %s" % (
                    e["route"], e["segment"], "resume it from %s" % _ckpt(part["resume_from"]) if part else "retry"))
                halted["flag"] = True
        pending = [e for e in plan if not is_passed(e["route"], e["segment"]) and "%s/%s" % (e["route"], e["segment"]) not in running]
        if not pending and not running:
            _log(out, "every segment landed")
            return 0
        if halted["flag"]:
            if running:
                time.sleep(30)
                continue
            _log(out, "halted: a segment failed to land; see the ledger, fix, and run the driver again")
            return 1
        # start what is ready: route A first (it is the critical path; route B
        # hangs off A's checkpoints and can never overtake it), then smallest
        # global step; never two of one vendor class at once. A ready live
        # segment of route A needs the driver idle, so no other route's
        # segment starts ahead of it. Routes beyond A and B (a route C replay)
        # come after both, by _ready_order.
        # a held segment ("hold": "<reason>" in the spec) is never ready: it
        # waits for a person to remove the hold and relaunch (a plan change)
        ready = sorted([e for e in pending if not e.get("hold") and all(is_passed(r, sg) for r, sg in e["depends"])],
                       key=lambda e: (e["route"] != "A", _ready_order(e)))
        live_a_ready = any(e["route"] == "A" and e["vendor"] == "live" for e in ready)
        busy = set().union(*(_vendor_class(e, spec) for _, e, _ in running.values())) if running else set()
        for e in ready:
            if len(running) >= parallel:
                break
            if live_a_ready and e["route"] != "A":
                continue
            cls = _vendor_class(e, spec)
            if cls & busy or ("live" in busy) or ("live" in cls and busy):
                continue
            key = "%s/%s" % (e["route"], e["segment"])
            _log(out, "starting %s on %s (%d steps from %s)" % (key, e["vendor"], e["steps"], e["from_ckpt"]))
            holder = {}
            th = threading.Thread(target=worker, args=(e, holder), name=key, daemon=True)
            th.start()
            running[key] = (th, e, holder)
            busy |= cls
        if not running:
            held = ["%s/%s" % (e["route"], e["segment"]) for e in pending if e.get("hold")]
            if held:
                _log(out, "nothing is ready and nothing is running; held: %s (remove the hold from the spec and run the driver again)" % ", ".join(held))
            else:
                _log(out, "nothing is ready and nothing is running: a dependency failed; see the ledger")
            return 1
        time.sleep(30)


def cmd_status(args):
    spec = load_spec(args.spec)
    ledger = Ledger(args.out)
    for e in segment_plan(spec):
        rec = ledger.landed(e) or {}
        print("%s/%s %-7s %s %s%s" % (e["route"], e["segment"], e["vendor"], rec.get("verdict", "pending"), rec.get("utc", ""),
                                      "  (resumed from step %d)" % rec["resumed_from"] if rec.get("resumed_from") is not None else ""))
        part = ledger.partial(e)
        if part and rec.get("verdict") != "PASS":
            print("      partial: resume from step %d%s; checkpoints landed %s; chain to step %d (%s); %d attempt(s), last in %s"
                  % (part["resume_from"], "" if _resume_allowed(spec, e) else " (\"resume\": false: restarts from its start)",
                     ", ".join(sorted(part["checkpoints"])), part["chain_steps"][1], part["chain"],
                     len(part["attempts"]), part["attempts"][-1]["dir"]))
    return 0


def cmd_reland(args):
    """Read a landed segment's fetched results again and rewrite its ledger
    record; nothing is rented. For a record written by a landing that read
    the wrong file."""
    spec = load_spec(args.spec)
    route, _, segment = args.segment.partition("/")
    ledger = Ledger(args.out)
    for e in segment_plan(spec):
        if e["route"] == route and e["segment"] == segment:
            was = ledger.landed(e)
            if not was or not was.get("results"):
                raise SystemExit("%s has no landed record with results to read" % args.segment)
            ok = land(spec, e, Path(was["results"]), args.out, ledger)
            now = ledger.landed(e)
            print(json.dumps(dict(before={k: was.get(k) for k in ("verdict", "arrival", "checkpoints")},
                                  after={k: now.get(k) for k in ("verdict", "arrival", "checkpoints", "workers")}), indent=1))
            return 0 if ok else 1
    raise SystemExit("%s is not in the spec" % args.segment)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "status", "reland"):
        p = sub.add_parser(name)
        p.add_argument("--spec", required=True)
        if name != "plan":
            p.add_argument("--out", required=True)
        if name == "run":
            p.add_argument("--parallel", type=int, default=1, help="segments at once (on different vendor classes; a live segment runs alone)")
        if name == "reland":
            p.add_argument("--segment", required=True, help="ROUTE/SEGMENT whose fetched results are read again (no rental)")
    args = ap.parse_args(argv)
    return dict(plan=cmd_plan, run=cmd_run, status=cmd_status, reland=cmd_reland)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
