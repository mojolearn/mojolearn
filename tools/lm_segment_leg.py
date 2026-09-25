#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Render one segment's leg body: mint its URLs on the Mac, fill the placeholders.

    python3 tools/lm_segment_leg.py render --run runs/gpt3-small/2026-09-25 --recipe recipe.json \\
        --arm nvidia --mode one --devices 0,1 --route A --segment 1 --label nvidia-h100 \\
        --from ckpt_00000000.blm --from-sha SHA --steps 1000 --boundary 1000 \\
        [--expect-chain A/1/chain.jsonl] [--replay ckpt_00000998.blm --replay-sha SHA --replay-chain A/1/chain.jsonl] \\
        [--live-shards 0:44 --live-workers 2 --live-port 7777] --out body.sh

Everything a box needs reaches it as a presigned URL baked into the body:
the recipe (GET), the checkpoint it starts from (GET), the chain it is held
to (GET), the arrival replay's checkpoint and chain (GET), and one PUT per
artifact the segment will write (from `tools/lm_segment.py keys`). The Mac
mints them with tools/dataset_store.sh, so no credential leaves it, and the
run's objects live under `<run>/<route>/<segment>/<file>` in R2.

`--from` and `--replay` are file names under the run prefix of the route
and segment named by `--from-route --from-segment` (default: this route's
previous segment for the replay, route A's previous segment for a route B
start). The sha256 of every fetched checkpoint is required so the box
verifies before it trains.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parents[1]


def _store(*args):
    return subprocess.run(["sh", str(REPO / "tools" / "dataset_store.sh"), *args], capture_output=True, text=True,
                          check=True, cwd=REPO).stdout.strip()


def presign_get(key, seconds):
    return _store("presign", key, str(seconds))


def presign_put(key, seconds):
    return _store("presign-put", key, str(seconds))


def tokens_urls(group, seconds):
    """Presigned GETs for every pinned object under a token-stream group key
    (the parts and the manifest), from bench/results/dataset_store/manifest.tsv."""
    out = {}
    for line in (REPO / "bench" / "results" / "dataset_store" / "manifest.tsv").read_text().splitlines():
        key = line.split("\t")[0]
        if key.startswith(group.rstrip("/") + "/"):
            out[key.rsplit("/", 1)[1]] = presign_get(key, seconds)
    if not out:
        raise SystemExit("no pinned objects under %s" % group)
    return out


def render(args):
    from lm_segment import expected_keys, load_recipe
    recipe = load_recipe(args.recipe)
    recipe_sha = hashlib.sha256(Path(args.recipe).read_bytes()).hexdigest()
    run = args.run.rstrip("/")
    here = f"{run}/{args.route}/{args.segment}"
    from_route = args.from_route or args.route
    from_seg = args.from_segment
    init = args.from_ckpt == "init"
    from_key = "" if init else (args.from_key or (f"{run}/{from_route}/{from_seg}/{args.from_ckpt}" if from_seg else f"{run}/{args.from_ckpt}"))
    from_step = 0 if init else int(args.from_ckpt.split("_")[1].split(".")[0])
    if not init and not args.from_sha:
        raise SystemExit("--from-sha is required unless --from init")
    keys = expected_keys(recipe, from_step, args.steps, args.boundary) if args.mode != "live-worker" else \
        ["chain.jsonl", "manifest.tsv", "segment.json"]
    uploads = {k: presign_put(f"{here}/{k}", args.seconds) for k in keys}
    body = (REPO / "tools" / "lm_segment_body.sh").read_text()
    subst = {
        "@ARM@": args.arm, "@MODE@": args.mode, "@DEVICES@": args.devices,
        "@ROUTE@": args.route, "@SEGMENT@": args.segment, "@LABEL@": args.label,
        "@STEPS@": str(args.steps), "@BOUNDARY@": "" if args.boundary is None else str(args.boundary),
        "@TOKENS_GROUP_PARTS@": "",
        "@FROM_NAME@": args.from_ckpt, "@FROM_URL@": "" if init else presign_get(from_key, args.seconds), "@FROM_SHA@": args.from_sha or "",
        "@SEED_URL@": presign_put(f"{here}/ckpt_00000000.blm", args.seconds) if init else "",
        "@EXPECT_URL@": presign_get(args.expect_key or f"{run}/{args.expect_chain}", args.seconds) if (args.expect_chain or args.expect_key) else "",
        "@REPLAY_NAME@": args.replay or "",
        "@REPLAY_URL@": presign_get(args.replay_key or f"{run}/{args.replay_route or from_route}/{args.replay_segment or from_seg}/{args.replay}", args.seconds) if args.replay else "",
        "@REPLAY_SHA@": args.replay_sha or "",
        "@REPLAY_CHAIN_URL@": presign_get(args.replay_chain_key or f"{run}/{args.replay_chain}", args.seconds) if (args.replay_chain or args.replay_chain_key) else "",
        "@UPLOADS@": json.dumps(uploads),
        "@LIVE_SHARDS@": args.live_shards or "", "@LIVE_WORKERS@": str(args.live_workers), "@LIVE_PORT@": str(args.live_port),
        "@RECIPE_URL@": presign_get(args.recipe_key or f"{run}/recipe.json", args.seconds), "@RECIPE_SHA@": recipe_sha,
        "@TOKENS_URLS@": json.dumps(tokens_urls(args.tokens_key, args.seconds)) if args.tokens_key else "",
        "@WHEEL@": args.wheel or "",
    }
    if args.replay and not args.replay_sha:
        raise SystemExit("--replay needs --replay-sha")
    for k, v in subst.items():
        body = body.replace(k, v)
    left = [l for l in body.splitlines() if "@" in l.split("#")[0] and "@" in l.split("#")[0].split("'")[-1]]
    Path(args.out).write_text(body)
    subprocess.run(["sh", "-n", args.out], check=True)
    ledger = dict(run=run, route=args.route, segment=args.segment, arm=args.arm, mode=args.mode, from_key=from_key,
                  from_sha=args.from_sha, steps=args.steps, boundary=args.boundary, expect_chain=args.expect_chain,
                  replay=args.replay, uploads=sorted(uploads), recipe_sha256=recipe_sha, body=str(args.out))
    Path(args.out).with_suffix(".ledger.json").write_text(json.dumps(ledger, indent=1) + "\n")
    print("rendered %s: %s/%s %s %s, from %s, %d steps, %d uploads%s" % (
        args.out, args.route, args.segment, args.arm, args.mode, from_key, args.steps, len(uploads),
        ", expect " + args.expect_chain if args.expect_chain else ""))
    return 0


DEFAULT_CONTROLS = ("none:none zero-moments:zero-moments k63:shards=63 swap-5-40:swap=5,40 swap-0-1:swap=0,1 "
                    "split:split ulp:ulp=63,auto")


def render_controls(args):
    """The negative-controls body (tools/lm_controls_body.sh): GETs only, the
    expected chain's lines for the checkpoint's step and the --steps steps
    after it embedded (two by default). `--arm amd --devices 0,1 --controls ""
    --steps 3` is a rehearsal on an AMD box: the positive replay alone."""
    from lm_segment import load_recipe
    load_recipe(args.recipe)
    recipe_sha = hashlib.sha256(Path(args.recipe).read_bytes()).hexdigest()
    name = args.from_key.rsplit("/", 1)[1]
    step = int(name.split("_")[1].split(".")[0])
    rows = {}
    for line in Path(args.expect_chain).read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["step"])] = line.strip()
    held = list(range(step, step + args.steps + 1))
    missing = [s for s in held if s not in rows]
    if missing:
        raise SystemExit("the expected chain has no line for step(s) %s" % missing)
    for flag in args.controls.split():
        if ":" not in flag:
            raise SystemExit("--controls entries are NAME:FLAG, got %r" % flag)
    subst = {
        "@ARM@": args.arm, "@DEVICES@": args.devices, "@STEPS@": str(args.steps),
        "@FROM_NAME@": name, "@FROM_STEP@": str(step), "@RECIPE_SHA@": recipe_sha, "@WHEEL@": args.wheel,
        "@CONTROLS@": args.controls,
        "@TOKENS_URLS@": json.dumps(tokens_urls(args.tokens_key, args.seconds)),
        "@RECIPE_URL@": presign_get(args.recipe_key, args.seconds),
        "@CKPT_URLS@": json.dumps({name: presign_get(args.from_key, args.seconds)}),
        "@EXPECT_LINES@": "\n".join(rows[s] for s in held),
    }
    body = (REPO / "tools" / "lm_controls_body.sh").read_text()
    for k, v in subst.items():
        body = body.replace(k, v)
    Path(args.out).write_text(body)
    subprocess.run(["sh", "-n", args.out], check=True)
    print("rendered %s: controls from %s (step %d), expected lines %d..%d, arm %s, devices %s, wheel %s, controls: %s"
          % (args.out, args.from_key, step, step, step + args.steps, args.arm, args.devices, args.wheel, args.controls or "none (positive only)"))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("controls", help="the negative-controls body (plan section 6 item 7)")
    c.add_argument("--recipe", required=True, help="the local recipe.json (pinned by its sha256 on the box)")
    c.add_argument("--recipe-key", required=True)
    c.add_argument("--from-key", required=True, help="the checkpoint's full R2 key (read only)")
    c.add_argument("--tokens-key", required=True)
    c.add_argument("--expect-chain", required=True, help="a local chain holding the checkpoint's step and the two after it")
    c.add_argument("--wheel", required=True)
    c.add_argument("--controls", default=DEFAULT_CONTROLS, help="space-separated NAME:FLAG (FLAG zero-moments or a --control value)")
    c.add_argument("--seconds", type=int, default=6 * 3600)
    c.add_argument("--arm", choices=("nvidia", "amd"), default="nvidia")
    c.add_argument("--devices", default="0", help="the devices every run uses, e.g. 0,1 for both GPUs of a 2-GPU box")
    c.add_argument("--steps", type=int, default=2, help="optimizer steps per run, held to the expected chain's lines")
    c.add_argument("--out", required=True)
    r = sub.add_parser("render")
    r.add_argument("--run", required=True, help="the run's R2 prefix, e.g. runs/gpt3-small/2026-09-25")
    r.add_argument("--recipe", required=True, help="the local recipe.json (its R2 copy is --recipe-key)")
    r.add_argument("--recipe-key", default=None, help="R2 key of the recipe (default <run>/recipe.json)")
    r.add_argument("--arm", choices=("nvidia", "amd"), required=True)
    r.add_argument("--mode", choices=("one", "live-coordinator", "live-worker"), default="one")
    r.add_argument("--devices", default="0")
    r.add_argument("--route", required=True)
    r.add_argument("--segment", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--from", dest="from_ckpt", required=True, help="checkpoint file name, e.g. ckpt_00001000.blm, or `init` to draw the seed on the box")
    r.add_argument("--from-sha", default=None, help="required unless --from init")
    r.add_argument("--from-key", default=None, help="the checkpoint's full R2 key (overrides --from-route/--from-segment)")
    r.add_argument("--expect-key", default=None, help="the expected chain's full R2 key")
    r.add_argument("--replay-key", default=None, help="the replay checkpoint's full R2 key")
    r.add_argument("--replay-chain-key", default=None, help="the replay chain's full R2 key")
    r.add_argument("--from-route", default=None)
    r.add_argument("--from-segment", default=None, help="the segment whose directory holds --from ('' for the run root)")
    r.add_argument("--steps", type=int, required=True)
    r.add_argument("--boundary", type=int, default=None)
    r.add_argument("--expect-chain", default=None, help="R2 path under the run, e.g. A/1/chain.jsonl")
    r.add_argument("--replay", default=None, help="boundary-minus-two checkpoint file name for the arrival replay")
    r.add_argument("--replay-sha", default=None)
    r.add_argument("--replay-route", default=None)
    r.add_argument("--replay-segment", default=None)
    r.add_argument("--replay-chain", default=None, help="R2 path under the run of the sender's chain")
    r.add_argument("--live-shards", default=None)
    r.add_argument("--live-workers", type=int, default=2)
    r.add_argument("--live-port", type=int, default=7777)
    r.add_argument("--seconds", type=int, default=6 * 3600, help="URL lifetime")
    r.add_argument("--wheel", default=None, help="run from this PUBLISHED mojolearn version (pip) instead of building on the box")
    r.add_argument("--tokens-key", default=None, help="token-stream group key: bake presigned GETs for its parts into the body (for a runner that stages nothing)")
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return render_controls(args) if args.cmd == "controls" else render(args)


if __name__ == "__main__":
    sys.exit(main())
