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


def render(args):
    from lm_segment import expected_keys, load_recipe
    recipe = load_recipe(args.recipe)
    recipe_sha = hashlib.sha256(Path(args.recipe).read_bytes()).hexdigest()
    run = args.run.rstrip("/")
    here = f"{run}/{args.route}/{args.segment}"
    from_route = args.from_route or args.route
    from_seg = args.from_segment
    from_key = args.from_key or (f"{run}/{from_route}/{from_seg}/{args.from_ckpt}" if from_seg else f"{run}/{args.from_ckpt}")
    from_step = int(args.from_ckpt.split("_")[1].split(".")[0])
    keys = expected_keys(recipe, from_step, args.steps, args.boundary) if args.mode != "live-worker" else \
        ["chain.jsonl", "manifest.tsv", "segment.json"]
    uploads = {k: presign_put(f"{here}/{k}", args.seconds) for k in keys}
    body = (REPO / "tools" / "lm_segment_body.sh").read_text()
    subst = {
        "@ARM@": args.arm, "@MODE@": args.mode, "@DEVICES@": args.devices,
        "@ROUTE@": args.route, "@SEGMENT@": args.segment, "@LABEL@": args.label,
        "@STEPS@": str(args.steps), "@BOUNDARY@": "" if args.boundary is None else str(args.boundary),
        "@TOKENS_GROUP_PARTS@": "",
        "@FROM_NAME@": args.from_ckpt, "@FROM_URL@": presign_get(from_key, args.seconds), "@FROM_SHA@": args.from_sha,
        "@EXPECT_URL@": presign_get(args.expect_key or f"{run}/{args.expect_chain}", args.seconds) if (args.expect_chain or args.expect_key) else "",
        "@REPLAY_NAME@": args.replay or "",
        "@REPLAY_URL@": presign_get(args.replay_key or f"{run}/{args.replay_route or from_route}/{args.replay_segment or from_seg}/{args.replay}", args.seconds) if args.replay else "",
        "@REPLAY_SHA@": args.replay_sha or "",
        "@REPLAY_CHAIN_URL@": presign_get(args.replay_chain_key or f"{run}/{args.replay_chain}", args.seconds) if (args.replay_chain or args.replay_chain_key) else "",
        "@UPLOADS@": json.dumps(uploads),
        "@LIVE_SHARDS@": args.live_shards or "", "@LIVE_WORKERS@": str(args.live_workers), "@LIVE_PORT@": str(args.live_port),
        "@RECIPE_URL@": presign_get(args.recipe_key or f"{run}/recipe.json", args.seconds), "@RECIPE_SHA@": recipe_sha,
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
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
    r.add_argument("--from", dest="from_ckpt", required=True, help="checkpoint file name, e.g. ckpt_00001000.blm")
    r.add_argument("--from-sha", required=True)
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
    r.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return render(args)


if __name__ == "__main__":
    sys.exit(main())
