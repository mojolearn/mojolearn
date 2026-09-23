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
                "B": [...]}}

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
GPU-type walk) or tools/do_extra_leg.sh (amd, DigitalOcean; `amd_provider`
runpod uses gemm_remote_leg.sh amd). Live segments go through
tools/lm_live_leg.sh. The runners own the dead-men, the fetch and the
verified delete. The driver never talks to a cloud API itself.
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
            entry = dict(route=route, segment=s["segment"], index=s["index"], vendor=s["vendor"], steps=int(s["steps"]),
                         first=first, last=last, boundary=last, live=s.get("first"), shards=s.get("shards"),
                         lease_minutes=s.get("lease_minutes"), dollar_cap=s.get("dollar_cap"),
                         from_route=("A" if route != "A" else route) if prev else None,
                         from_segment=prev["segment"] if prev else None,
                         from_ckpt=("ckpt_%08d.blm" % first) if prev else "init",
                         replay_ckpt=("ckpt_%08d.blm" % (first - 2)) if prev else None,
                         expect=(f"A/{s['segment']}/chain.jsonl" if route != "A" else None),
                         depends=[("A" if route != "A" else route, prev["segment"])] if prev else [])
            if route != "A":
                entry["depends"].append(("A", s["segment"]))  # B's chain is held to A's; A's segment must exist
            plan.append(entry)
            at = last
    return plan


def cmd_plan(args):
    spec = load_spec(args.spec)
    for e in segment_plan(spec):
        print("%s/%s %-7s steps %d..%d from %s%s%s" % (
            e["route"], e["segment"], e["vendor"], e["first"], e["last"],
            e["from_ckpt"] if e["from_ckpt"] == "init" else "%s/%s/%s" % (e["from_route"], e["from_segment"], e["from_ckpt"]),
            " replay " + e["replay_ckpt"] if e["replay_ckpt"] else "",
            " expect " + e["expect"] if e["expect"] else ""))
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


def _segment_json(results_dir):
    for p in Path(results_dir).rglob("segment.json"):
        if p.parent.name == "segment":
            return json.loads(p.read_text())
    return None


def _status(results_dir):
    for p in Path(results_dir).rglob("status.txt"):
        return p.read_text()
    return ""


# ---------------------------------------------------------------- rendering and renting

def render(spec, e, out, ledger, role=None):
    """One body for a segment (or one side of a live segment)."""
    run = spec["run"]
    arm = e["vendor"] if role is None else role
    mode = "one" if role is None else ("live-coordinator" if role == e["live"] else "live-worker")
    devices = spec.get("nvidia_devices", "0") if arm == "nvidia" else spec.get("amd_devices", "0")
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
        if mode != "live-worker":
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
    body = Path(out) / "bodies" / ("%s-%s%s.sh" % (e["route"], e["segment"], "-" + arm if role else ""))
    body.parent.mkdir(parents=True, exist_ok=True)
    cmd += ["--out", str(body)]
    subprocess.run(cmd, check=True, cwd=REPO)
    return body


def rent_one(spec, e, body, out):
    """Rent one box for a one-box segment; returns the results directory and the runner's exit code."""
    res = Path(out) / "legs" / ("%s-%s" % (e["route"], e["segment"]))
    res.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MOJOLEARN_GEMM_LEG_EXTRA=str(body), MOJOLEARN_STAGE_KEYS=spec["tokens_stage"],
               MOJOLEARN_GEMM_LEG_OUT=str(res / "leg"))
    # a segment's own lease and cap win over the run's (an AMD segment on an
    # 8-GPU box needs both bigger); the driver never rents without a cap
    minutes = int(e.get("lease_minutes") or spec.get("lease_minutes", 120))
    cap = str(e.get("dollar_cap") or spec.get("dollar_cap", 10))
    lease = ["--segment-lease", str(minutes), "--dollar-cap", cap] if minutes > 60 else ["--minutes", str(minutes)]
    if e["vendor"] == "nvidia":
        rc = 3
        for gpu in spec.get("nvidia_gpus", "NVIDIA H100 80GB HBM3|NVIDIA H200|NVIDIA H100 PCIe|NVIDIA H100 NVL").split("|"):
            slug = "".join(c for c in gpu.replace(" ", "_") if c.isalnum() or c == "_")
            env["MOJOLEARN_GEMM_LEG_OUT"] = str(res / ("leg-" + slug))
            if spec.get("nvidia_devices", "0").count(",") == 1:
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
    providers = spec.get("amd_providers") or [spec.get("amd_provider", "do")]
    rc = 3
    # AMD capacity is the choke point (one DigitalOcean GPU droplet per account,
    # RunPod MI300X and Hot Aisle often without stock): walk the providers, and
    # when every one is busy, wait and walk again, for up to amd_wait_minutes.
    deadline = time.monotonic() + 60 * int(spec.get("amd_wait_minutes", 180))
    attempt = 0
    while True:
        attempt += 1
        result = _rent_amd_once(spec, e, res, env, lease, providers, out, attempt)
        if result is not None:
            return result
        if time.monotonic() > deadline:
            _log(out, "amd: no provider had capacity within amd_wait_minutes; giving up on this attempt")
            return res, rc
        _log(out, "amd: every provider busy or without capacity; waiting 10 minutes (attempt %d)" % attempt)
        time.sleep(600)


def _rent_amd_once(spec, e, res, env, lease, providers, out, attempt):
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
                # one hour is Hot Aisle's maximum and minimum; the body fetches the token parts itself
                rc = subprocess.run(["bash", "tools/hotaisle_leg.sh", "amd", "--rent", "--skip-gates", "--minutes", "60"],
                                    cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            else:
                rc = subprocess.run(["sh", "tools/gemm_remote_leg.sh", "amd", "--rent", "--allow-concurrent", *lease], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        text = (res / ("leg-%s.log" % tag)).read_text()
        busy = ("ONE GPU droplet at a time" in text or "no instances currently available" in text
                or "REFUSING to create" in text or "quantity=0" in text or "no stock" in text.lower()
                or "starved" in text.lower())
        if rc != 0 and busy:
            _log(out, "amd: %s is busy or without capacity (attempt %d)" % (provider, attempt))
            continue
        return res / ("leg-" + tag), rc
    return None


def rent_live(spec, e, nv_body, amd_body, out):
    res = Path(out) / "legs" / ("%s-%s-live" % (e["route"], e["segment"]))
    res.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MOJOLEARN_LIVE_STAGE_KEYS=spec["tokens_stage"],
               MOJOLEARN_LIVE_NVIDIA_GPUS=spec.get("nvidia_gpus", "NVIDIA H100 80GB HBM3|NVIDIA H200|NVIDIA H100 PCIe|NVIDIA H100 NVL"))
    with open(res / "live.log", "w") as log:
        rc = subprocess.run(["bash", "tools/lm_live_leg.sh", str(res), str(nv_body), str(amd_body), "--minutes",
                             str(spec.get("lease_minutes", 120)), "--dollar-cap", str(spec.get("dollar_cap", 10)),
                             "--amd", spec.get("amd_provider", "do")], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return res, rc


# ---------------------------------------------------------------- landing

def land(spec, e, results, out, ledger):
    """Read the fetched results; PASS lands the segment with its checkpoint hashes."""
    seg = _segment_json(results)
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
    for p in Path(results).rglob("segment.json"):
        if p.parent.name == "arrival":
            arrival = json.loads(p.read_text()).get("verdict")
    record["arrival"] = arrival
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


def cmd_run(args):
    spec = load_spec(args.spec)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(out)
    plan = segment_plan(spec)
    _log(out, "run %s: %d segments over %d routes" % (spec["run"], len(plan), len(spec["routes"])))
    pending = [e for e in plan if not (ledger.landed(e) or {}).get("verdict") == "PASS"]
    while pending:
        ready = [e for e in pending if all((ledger.landed(dict(route=r, segment=s)) or {}).get("verdict") == "PASS" for r, s in e["depends"])]
        if not ready:
            _log(out, "nothing is ready and nothing is running: a dependency failed; see the ledger")
            return 1
        e = ready[0]  # one segment at a time in this driver; concurrency across routes is a later refinement
        _log(out, "starting %s/%s on %s (%d steps from %s)" % (e["route"], e["segment"], e["vendor"], e["steps"], e["from_ckpt"]))
        try:
            if e["vendor"] == "live":
                first = e["live"]; second = "amd" if first == "nvidia" else "nvidia"
                nv_body = render(spec, e, out, ledger, role="nvidia")
                amd_body = render(spec, e, out, ledger, role="amd")
                results, rc = rent_live(spec, e, nv_body if first == "nvidia" else nv_body, amd_body, out)
            else:
                body = render(spec, e, out, ledger)
                results, rc = rent_one(spec, e, body, out)
        except subprocess.CalledProcessError as exc:
            _log(out, "render failed for %s/%s: %s" % (e["route"], e["segment"], exc))
            return 1
        _log(out, "%s/%s runner exit %s; results %s" % (e["route"], e["segment"], rc, results))
        ok = land(spec, e, results, out, ledger)
        if ok is False:
            return 1
        if ok is None:
            _log(out, "%s/%s did not land (no results); run the driver again to retry" % (e["route"], e["segment"]))
            return 2
        pending = [x for x in pending if not (x["route"] == e["route"] and x["segment"] == e["segment"])]
    _log(out, "every segment landed")
    return 0


def cmd_status(args):
    spec = load_spec(args.spec)
    ledger = Ledger(args.out)
    for e in segment_plan(spec):
        rec = ledger.landed(e) or {}
        print("%s/%s %-7s %s %s" % (e["route"], e["segment"], e["vendor"], rec.get("verdict", "pending"), rec.get("utc", "")))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "status"):
        p = sub.add_parser(name)
        p.add_argument("--spec", required=True)
        if name != "plan":
            p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return dict(plan=cmd_plan, run=cmd_run, status=cmd_status)[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
