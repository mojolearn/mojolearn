#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Check the restored FP32 default on a real model (SmolLM2-360M).

    python3 tools/lowbit_default/default_gate.py --model /root/models/SmolLM2-360M \
        --phases hash,generate --out <dir>

hash      `CausalLM.load(path)` with NO keyword must print the fp32_v1
          hash (833c9a8947bdd619...), equal to explicit fp32_v1;
          explicit fixed15_v1 must retain d37c2ea81d13743a...;
          and the two explicit FP32 selectors,
          `set_numeric_profile("fp32_v1")` in this process and
          `MOJOLEARN_NUMERIC_PROFILE=fp32_v1` in a child, the fp32_v1 hash
          with no keyword. Same ids as tools/lowbit_blocks/model_logits.py
          (B=2, L=64, splitmix64 seed 0x6c6f7762). Exit 1 on any mismatch.
generate  what a user who passes nothing gets from `generate` against what
          fp32_v1 gave them (the maintainer's condition before the merge):
          a 512-token prompt at B=1, N new tokens, (a) the default model
          (fp32_v1) and (b) a model loaded with
          numeric_profile="fp32_v1" (its resident session where the binding
          has one; which route ran is recorded). One untimed call each, then
          the two alternate, R rounds; medians. Every timed call's ids are
          hashed after the clock and must agree within each arm.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))
sys.path.insert(0, os.path.join(HERE, "..", "lowbit_blocks"))

WANT_FIXED15 = "d37c2ea81d13743a"
WANT_FP32 = "833c9a8947bdd619"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--phases", default="hash,generate")
    ap.add_argument("--out", required=True)
    ap.add_argument("--box", default=os.uname().nodename)
    ap.add_argument("--new", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--child-hash", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from model_logits import token_ids, raw, sha, SEED
    from mojolearn.models import CausalLM
    from mojolearn._array import Array
    import mojolearn

    def ident(lm):
        rows = token_ids(2, 64, lm.vocab_size)
        return sha(raw(lm.forward(Array.from_list(rows, "<i4"))))

    if args.child_hash:  # the environment hatch, in a fresh process
        lm = CausalLM.load(args.model)
        print(f"CHILD {lm.numeric_profile} {ident(lm)}", flush=True)
        return 0

    rec = {"box": args.box, "commit": os.environ.get("MOJOLEARN_COMMIT", ""),
           "vendor": mojolearn._backend.vendor(), "env_profile": os.environ.get("MOJOLEARN_NUMERIC_PROFILE")}
    bad = []
    phases = args.phases.split(",")
    lm = CausalLM.load(args.model)
    ref = CausalLM.load(args.model, numeric_profile="fp32_v1")
    print(f"default model reports numeric_profile={lm.numeric_profile}; named model {ref.numeric_profile}; "
          f"device {lm.device}", flush=True)

    if "hash" in phases:
        h = {"default": ident(lm), "fp32_v1": ident(ref)}
        opted = CausalLM.load(args.model, numeric_profile="fixed15_v1")
        h["fixed15_v1"] = ident(opted)
        del opted
        prev = mojolearn.set_numeric_profile("fp32_v1")
        try:
            hatch = CausalLM.load(args.model)
            h["set_numeric_profile_fp32"] = ident(hatch)
            h["set_numeric_profile_fp32_reports"] = hatch.numeric_profile
            del hatch
        finally:
            mojolearn.set_numeric_profile(prev)
        env = dict(os.environ, MOJOLEARN_NUMERIC_PROFILE="fp32_v1")
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--model", args.model, "--out", args.out,
                            "--child-hash"], env=env, capture_output=True, text=True)
        line = [x for x in r.stdout.splitlines() if x.startswith("CHILD ")]
        h["env_fp32"] = line[-1].split()[2] if line else "CHILD FAILED: " + r.stderr[-300:]
        h["env_fp32_reports"] = line[-1].split()[1] if line else None
        checks = {
            "default is fp32_v1": lm.numeric_profile == "fp32_v1",
            "default equals explicit fp32_v1": h["default"] == h["fp32_v1"],
            "opt-in retains fixed15 hash": h["fixed15_v1"].startswith(WANT_FIXED15),
            "fp32_v1 hash 833c9a8947bdd619": h["fp32_v1"].startswith(WANT_FP32),
            "set_numeric_profile(fp32_v1) == fp32_v1 bits": h["set_numeric_profile_fp32"] == h["fp32_v1"]
            and h["set_numeric_profile_fp32_reports"] == "fp32_v1",
            "MOJOLEARN_NUMERIC_PROFILE=fp32_v1 == fp32_v1 bits": h["env_fp32"] == h["fp32_v1"]
            and h["env_fp32_reports"] == "fp32_v1",
        }
        for k, v in h.items():
            print(f"RESULT hash {k} {v}", flush=True)
        for k, ok in checks.items():
            print(f"RESULT check {k}: {'HELD' if ok else 'BROKEN'}", flush=True)
            if not ok:
                bad.append(k)
        rec["hash"] = h
        rec["hash_checks"] = checks

    if "generate" in phases:
        v = lm.vocab_size
        prompt = Array.from_list(token_ids(1, 512, v, SEED + 7), "<i4")
        routes = {}

        def spy(model, name):
            inner = model._generate_resident

            def wrapped(*a, **k):
                out = inner(*a, **k)
                routes.setdefault(name, set()).add("resident" if out is not None else "per-layer")
                return out
            model._generate_resident = wrapped
        spy(lm, "default")
        spy(ref, "fp32_v1")
        times = {"default": [], "fp32_v1": []}
        hashes = {"default": [], "fp32_v1": []}

        def one(name, model, timed=True):
            t = time.perf_counter()
            out = model.generate(prompt, args.new)
            dt = time.perf_counter() - t
            hashes[name].append(hashlib.sha256(out.tobytes()).hexdigest())
            if timed:
                times[name].append(dt)
        one("default", lm, False)
        one("fp32_v1", ref, False)
        for _ in range(args.rounds):
            one("default", lm)
            one("fp32_v1", ref)
        g = {}
        for name in times:
            med = statistics.median(times[name])
            agree = len(set(hashes[name])) == 1
            g[name] = {"median_s": med, "runs_s": times[name], "route": sorted(routes.get(name, ())),
                       "ids_sha256": hashes[name][0], "ids_agree": agree}
            print(f"RESULT generate box={args.box} arm={name} route={'+'.join(g[name]['route'])} "
                  f"prompt=512 new={args.new} median {med * 1000:.1f} ms ({med * 1000 / args.new:.2f} ms/token) "
                  f"runs {' '.join(f'{x * 1000:.1f}' for x in times[name])} ids {'AGREE' if agree else 'DISAGREE'} "
                  f"{hashes[name][0][:16]}", flush=True)
            if not agree:
                bad.append(f"generate {name} ids disagree")
        ratio = g["default"]["median_s"] / g["fp32_v1"]["median_s"]
        print(f"RESULT generate box={args.box} default/fp32_v1 = {ratio:.3f} "
              f"({'default QUICKER' if ratio < 1 else 'default SLOWER'})", flush=True)
        g["ratio_default_over_fp32"] = ratio
        rec["generate"] = g

    path = os.path.join(args.out, f"default_gate_{args.box}.json")
    with open(path, "w") as f:
        json.dump(rec, f, indent=1, default=list)
    print(f"wrote {path}")
    print("DEFAULT GATE " + ("GREEN" if not bad else "RED: " + "; ".join(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
