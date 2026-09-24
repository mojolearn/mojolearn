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

RENTALS. One-box segments go through tools/gemm_remote_leg.sh (nvidia, a
GPU-type walk) or, for amd, the providers named in `amd_providers`, walked in
order: `do` is tools/do_extra_leg.sh (DigitalOcean), `hotaisle` is
tools/hotaisle_leg.sh, anything else is gemm_remote_leg.sh amd (RunPod).
Live segments go through tools/lm_live_leg.sh. The runners own the
dead-men, the fetch and the verified delete. The driver never talks to a
cloud API itself.

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
skipped like a busy provider, never raised.
"""
import argparse
import json
import os
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
            entry = dict(route=route, segment=s["segment"], index=s["index"], vendor=s["vendor"], steps=int(s["steps"]),
                         first=first, last=last, boundary=last, live=s.get("first"), shards=s.get("shards"),
                         lease_minutes=s.get("lease_minutes"), dollar_cap=s.get("dollar_cap"),
                         nvidia_devices=s.get("nvidia_devices"), nvidia_gpus=s.get("nvidia_gpus"),
                         provider=s.get("provider"), hotaisle_dollar_cap=s.get("hotaisle_dollar_cap"),
                         from_route=("A" if route != "A" else route) if (prev or seed_from_a) else None,
                         from_segment=(prev["segment"] if prev else a_first) if (prev or seed_from_a) else None,
                         from_ckpt=("ckpt_%08d.blm" % first) if (prev or seed_from_a) else "init",
                         replay_ckpt=("ckpt_%08d.blm" % (first - 2)) if prev else None,
                         expect=(f"A/{s['segment']}/chain.jsonl" if route != "A" else None),
                         depends=[("A" if route != "A" else route, prev["segment"])] if prev else ([("A", a_first)] if seed_from_a else []))
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


# ---------------------------------------------------------------- rendering and renting

NVIDIA_GPUS = "NVIDIA H100 80GB HBM3|NVIDIA H200|NVIDIA H100 PCIe|NVIDIA H100 NVL"


def _nvidia_devices(spec, e):
    """The segment's own NVIDIA devices, else the run's."""
    return e.get("nvidia_devices") or spec.get("nvidia_devices", "0")


def _nvidia_gpus(spec, e):
    """The segment's own NVIDIA GPU types (a | list), else the run's."""
    return e.get("nvidia_gpus") or spec.get("nvidia_gpus", NVIDIA_GPUS)


def render(spec, e, out, ledger, role=None, suffix=""):
    """One body for a segment (or one side of a live segment)."""
    run = spec["run"]
    arm = e["vendor"] if role is None else role
    mode = "one" if role is None else ("live-coordinator" if role == e["live"] else "live-worker")
    devices = _nvidia_devices(spec, e) if arm == "nvidia" else spec.get("amd_devices", "0")
    label = "%s-%s-%s" % (arm, e["route"], e["segment"])
    cmd = [sys.executable, str(REPO / "tools" / "lm_segment_leg.py"), "render", "--run", run, "--recipe", spec["recipe"],
           "--recipe-key", spec["recipe_key"], "--arm", arm, "--mode", mode, "--devices", devices,
           "--tokens-key", spec["tokens_stage"],
           *(["--wheel", spec["wheel"]] if spec.get("wheel") else []),
           "--route", e["route"], "--segment", e["segment"], "--label", label, "--steps", str(e["steps"]),
           "--from", e["from_ckpt"], "--seconds", str(spec.get("url_seconds", 8 * 3600))]
    if mode != "live-worker":
        cmd += ["--boundary", str(e["boundary"])]
    if e["from_ckpt"] != "init":
        src = ledger.landed(dict(route=e["from_route"], segment=e["from_segment"]))
        if not src:
            raise SystemExit("%s/%s has not landed" % (e["from_route"], e["from_segment"]))
        cmd += ["--from-sha", src["checkpoints"][e["from_ckpt"]], "--from-key", "%s/%s/%s/%s" % (run, e["from_route"], e["from_segment"], e["from_ckpt"])]
        # a route's first segment seeded from route A's file has no previous
        # segment of its own and so no arrival replay
        if mode != "live-worker" and e["replay_ckpt"]:
            cmd += ["--replay", e["replay_ckpt"], "--replay-sha", src["checkpoints"][e["replay_ckpt"]],
                    "--replay-key", "%s/%s/%s/%s" % (run, e["from_route"], e["from_segment"], e["replay_ckpt"]),
                    "--replay-chain-key", "%s/%s/%s/chain.jsonl" % (run, e["from_route"], e["from_segment"])]
    if e["expect"]:
        cmd += ["--expect-key", "%s/%s" % (run, e["expect"])]
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
        return None
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
    return [(e["route"], e["segment"], e["vendor"], e["steps"], e["first"], e["last"], tuple(e["depends"])) for e in plan]


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
                _log(out, "%s/%s did not land (no results); run the driver again to retry" % (e["route"], e["segment"]))
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
        ready = sorted([e for e in pending if all(is_passed(r, sg) for r, sg in e["depends"])],
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
            _log(out, "nothing is ready and nothing is running: a dependency failed; see the ledger")
            return 1
        time.sleep(30)


def cmd_status(args):
    spec = load_spec(args.spec)
    ledger = Ledger(args.out)
    for e in segment_plan(spec):
        rec = ledger.landed(e) or {}
        print("%s/%s %-7s %s %s" % (e["route"], e["segment"], e["vendor"], rec.get("verdict", "pending"), rec.get("utc", "")))
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
