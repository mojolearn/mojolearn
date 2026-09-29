#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""lane/lowbit-blocks: a whole model's full logits under a numeric profile.

    python3 tools/lowbit_blocks/model_logits.py --model /root/models/SmolLM2-360M \
        --profile fixed15_v1 --device auto --out <dir> [--phases identity,decode,batch,time]

`fixed15_v1`'s row has `inference: True` since lane/lowbit-default
(2026-09-29), so this script no longer opens it in its own process.

PHASES (each prints `RESULT <name> ...` lines and writes JSON to --out):
  identity  B=2 rows of L=64 fixed token ids (splitmix64, seed 0x6c6f7762):
            sha256 of the full logits bytes (B x L x vocab float32), and of
            each row's. The table across boxes is built from these lines.
  decode    row 0 alone: prefill of p tokens then one `step` per token, p in
            {1, 7, L-1}; every position's logits against the full prefill's
            row, bit for bit.
  batch     each row alone at B=1 against its rows in the B=2 call.
  time      prefill at 512 tokens (B=1) and decode per token (32 steps after
            a 512-token prefill): one untimed run, then 5 timed, medians.
The profile's head is `mojolearn.linalg.matmul_int15`; the blocks are
`TransformerBlock(numeric_profile=...)`.
"""
import argparse
import hashlib
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

L_ID = 64
B_ID = 2
SEED = 0x6C6F7762


def splitmix(z):
    z = (z + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def token_ids(b, l, vocab, seed=SEED):
    return [[splitmix(seed + r * 1000003 + t) % vocab for t in range(l)] for r in range(b)]


def raw(a):
    from mojolearn._bufcheck import flat_view
    return bytes(flat_view(a, "f").cast("B"))


def sha(bs):
    return hashlib.sha256(bs).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--profile", default="fp32_v1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--phases", default="identity,decode,batch")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", default=os.uname().nodename)
    ap.add_argument("--l-id", type=int, default=L_ID)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mojolearn
    from mojolearn.models import CausalLM
    from mojolearn._array import Array

    t0 = time.perf_counter()
    lm = CausalLM.load(args.model, numeric_profile=args.profile, device=args.device)
    load_s = time.perf_counter() - t0
    v = lm.vocab_size
    try:
        vendor = mojolearn._backend.vendor()
    except Exception as e:  # noqa: BLE001
        vendor = f"? {e}"
    rec = {"box": args.box, "profile": lm.numeric_profile, "device": lm.device, "vendor": vendor,
           "model": os.path.basename(args.model.rstrip("/")), "load_s": load_s,
           "commit": os.environ.get("MOJOLEARN_COMMIT", "")}
    print(f"model {rec['model']} profile {rec['profile']} device {lm.device} vendor {vendor} load {load_s:.1f} s",
          flush=True)
    phases = args.phases.split(",")
    L = args.l_id
    rows = token_ids(B_ID, L, v)
    ids = Array.from_list(rows, "<i4")

    full = None
    if any(p in phases for p in ("identity", "decode", "batch")):
        full = raw(lm.forward(ids))
        rowlen = L * v * 4
        rec["identity"] = {"B": B_ID, "L": L, "vocab": v, "sha256": sha(full),
                           "rows": [sha(full[r * rowlen:(r + 1) * rowlen]) for r in range(B_ID)]}
        print(f"RESULT identity box={args.box} profile={lm.numeric_profile} B={B_ID} L={L} "
              f"logits_sha256={rec['identity']['sha256']} row0={rec['identity']['rows'][0][:16]} "
              f"row1={rec['identity']['rows'][1][:16]}", flush=True)

    if "batch" in phases:
        rowlen = L * v * 4
        out = []
        for r in range(B_ID):
            alone = raw(lm.forward(Array.from_list([rows[r]], "<i4")))
            eq = alone == full[r * rowlen:(r + 1) * rowlen]
            out.append(eq)
            print(f"RESULT batch row {r} alone vs in B={B_ID}: {'EQUAL' if eq else 'DIFFERS'}", flush=True)
        rec["batch"] = out

    if "decode" in phases:
        rowlen = L * v * 4
        ref = full[0:rowlen]
        res = {}
        for p in (1, 7, L - 1):
            st = lm.allocate_state(1, L)
            got = [raw(lm.forward(Array.from_list([rows[0][:p]], "<i4"), st))]
            for t in range(p, L):
                got.append(raw(lm.step(Array.from_list([[rows[0][t]]], "<i4"), st)))
            cat = b"".join(got)
            bad = [t for t in range(L) if cat[t * v * 4:(t + 1) * v * 4] != ref[t * v * 4:(t + 1) * v * 4]]
            res[str(p)] = {"equal": not bad, "positions_differing": len(bad), "first": bad[:1]}
            print(f"RESULT decode prefix {p}: {L - p} decode steps; {len(bad)} of {L} positions differ from "
                  f"the full prefill{'  EQUAL' if not bad else '  first at ' + str(bad[0])}", flush=True)
        rec["decode"] = res

    if "time" in phases:
        # EVERY TIMED OUTPUT IS CHECKED (brief rule 10, a Metal launch cut
        # short leaves cells unwritten and says nothing): the sha256 of every
        # timed prefill's logits and of every timed decode run's logits are
        # recorded, must agree across the runs, and are compared across boxes.
        pre = Array.from_list(token_ids(1, 512, v, SEED + 7), "<i4")
        pre_sha = []

        # The hash is taken AFTER the clock stops: hashing 100 MB of logits
        # inside the interval added a fixed cost to every timed prefill
        # (found 2026-09-29 in nvc3-0044, fixed here).
        pre_sha.append(sha(raw(lm.forward(pre))))
        pts = []
        for _ in range(5):
            t = time.perf_counter()
            out = lm.forward(pre)
            pts.append(time.perf_counter() - t)
            pre_sha.append(sha(raw(out)))
            del out
        pm = statistics.median(pts)
        steps = 32
        nxt = token_ids(1, steps, v, SEED + 11)[0]
        dec_sha = []

        def decode_run():
            st = lm.allocate_state(1, 512 + steps)
            lm.forward(pre, st)
            outs = []
            t = time.perf_counter()
            for k in range(steps):
                outs.append(lm.step(Array.from_list([[nxt[k]]], "<i4"), st))
            dt = (time.perf_counter() - t) / steps
            dec_sha.append(sha(b"".join(raw(o) for o in outs)))
            return dt
        decode_run()
        dts = [decode_run() for _ in range(5)]
        dm = statistics.median(dts)
        same = len(set(pre_sha)) == 1 and len(set(dec_sha)) == 1
        rec["time"] = {"prefill512_s": pm, "prefill512_runs": pts, "decode_per_token_s": dm, "decode_runs": dts,
                       "prefill512_sha256": pre_sha, "decode32_sha256": dec_sha, "outputs_agree": same}
        print(f"RESULT time box={args.box} profile={lm.numeric_profile} prefill512 {pm * 1000:.1f} ms "
              f"decode {dm * 1000:.2f} ms/token  outputs {'AGREE' if same else 'DISAGREE'} "
              f"prefill512_sha256={pre_sha[0][:16]} decode32_sha256={dec_sha[0][:16]}", flush=True)

    path = os.path.join(args.out, f"model_{args.box}_{lm.numeric_profile}_{lm.device}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
