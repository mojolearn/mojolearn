#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The training table from the run records of `byte_lm_train.py`.

Lane lane/lowbit-quality, 2026-09-29. Reads JSON, runs no model.

THE NOISE FLOOR is the sample standard deviation, over the baseline's
seeds, of the validation loss at the equal-step point. It is measured
before any arm is read and printed beside every arm.

AN ARM'S CHANGE is the mean over its seeds of (arm loss minus the baseline
loss OF THE SAME SEED) at the equal-step point, as a relative change of
perplexity `exp(mean) - 1`.

THE VERDICT against 1 percent:
  MISS          a run went non-finite, or the change is 1 percent or more
                and outside the noise floor
  PASS          the change is under 1 percent and the noise floor is under
                1 percent; "inside the noise" is said when it is
  UNDERPOWERED  otherwise: the seeds cannot separate this arm from 1 percent
"""
import argparse
import glob
import json
import math
import os
import statistics
import sys


def at(run, step):
    for t in run["trace"]:
        if t["step"] == step:
            return t["val_loss"]
    return None


def smoothed(run, step, n=5):
    every = run["eval_every"]
    vals = [at(run, step - i * every) for i in range(n)]
    return None if any(v is None for v in vals) else sum(vals) / n


def first_reach(run, target):
    for t in run["trace"]:
        if t["step"] > 0 and t["val_loss"] <= target:
            return t["step"]
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.runs, "run_*.json")))]
    if not runs:
        print("no run records under", args.runs)
        return 1
    n_eq, n_end = runs[0]["equal_steps"], runs[0]["steps"]
    base = {r["seed"]: r for r in runs if r["arm"] == "a"}
    complete = {s: r for s, r in base.items() if r["nonfinite_at_step"] is None and at(r, n_eq) is not None}
    if len(complete) < 3:
        print("REFUSED: fewer than three finite baseline seeds; no arm is read")
        return 1
    b_eq = {s: at(r, n_eq) for s, r in complete.items()}
    b_sm = {s: smoothed(r, n_eq) for s, r in complete.items()}
    floor = statistics.stdev(b_eq.values())
    floor_sm = statistics.stdev(b_sm.values())
    table = dict(
        schema="mojolearn.lowbit_quality.training_table.v1", commit=runs[0]["commit"],
        model_profile=runs[0]["model_profile"], parameters=runs[0]["parameters"], shape=runs[0]["shape"],
        optimizer=runs[0]["optimizer"], data=runs[0]["data"], equal_steps=n_eq, steps=n_end,
        eval_every=runs[0]["eval_every"], threshold=dict(rel_ppl_change_max=0.01),
        baseline=dict(
            seeds=sorted(complete), val_loss_at_equal_steps=b_eq,
            mean=statistics.mean(b_eq.values()), noise_floor_nats=floor, noise_floor_rel_ppl=math.expm1(floor),
            range_nats=max(b_eq.values()) - min(b_eq.values()),
            smoothed_mean=statistics.mean(b_sm.values()), smoothed_noise_floor_nats=floor_sm,
            val_loss_at_end={s: at(r, n_end) for s, r in complete.items()},
            first_step_at_or_below_own_final={s: first_reach(r, b_eq[s]) for s, r in complete.items()},
            initial_val_loss={s: at(r, 0) for s, r in complete.items()}),
        arms=[])
    groups = {}
    for r in runs:
        if r["arm"] != "a":
            groups.setdefault((r["attention_products"], r["arm"], r["mode"]), []).append(r)
    for (attn, arm, mode), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        rs = [r for r in rs if r["seed"] in complete]
        bad = [r["seed"] for r in rs if r["nonfinite_at_step"] is not None or at(r, n_eq) is None
               or not math.isfinite(at(r, n_eq))]
        good = [r for r in rs if r["seed"] not in bad]
        row = dict(arm=arm, profile=rs[0]["profile_name"], mode=mode, attention_products=attn,
                   seeds=sorted(r["seed"] for r in rs), nonfinite_seeds=sorted(bad),
                   nonfinite_at_step={r["seed"]: r["nonfinite_at_step"] for r in rs if r["seed"] in bad},
                   noise_floor_nats=floor, noise_floor_rel_ppl=math.expm1(floor))
        if good:
            d = [at(r, n_eq) - b_eq[r["seed"]] for r in good]
            ds = [smoothed(r, n_eq) - b_sm[r["seed"]] for r in good]
            mean = statistics.mean(d)
            se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            reach = [first_reach(r, b_eq[r["seed"]]) for r in good]
            reached = sorted(x for x in reach if x is not None)
            cos = [v["cosine"] for r in good for v in r["gradient_against_fp32"].values()]
            fp = [r["fp32_forward_val_loss"].get(str(n_eq)) for r in good]
            row.update(
                val_loss_at_equal_steps={r["seed"]: at(r, n_eq) for r in good},
                delta_nats_per_seed={r["seed"]: x for r, x in zip(good, d)},
                delta_nats_mean=mean, delta_nats_se=se, rel_ppl_change=math.expm1(mean),
                smoothed_delta_nats_mean=statistics.mean(ds), smoothed_rel_ppl_change=math.expm1(statistics.mean(ds)),
                inside_noise=abs(mean) <= floor,
                val_loss_at_end={r["seed"]: at(r, n_end) for r in good},
                steps_to_baseline_final={r["seed"]: x for r, x in zip(good, reach)},
                steps_to_baseline_final_median=(statistics.median(reached) if len(reached) == len(reach) else None),
                seeds_not_reaching=len(reach) - len(reached), steps_run=n_end,
                fp32_forward_delta_nats_mean=statistics.mean(
                    f - at(r, n_eq) for r, f in zip(good, fp) if f is not None) if all(f is not None for f in fp) else None,
                gradient_cosine_against_fp32_mean=statistics.mean(cos) if cos else None,
                gradient_cosine_against_fp32_min=min(cos) if cos else None)
        if bad:
            row["verdict"] = "MISS"
            row["reason"] = "non-finite loss in seed(s) %s" % bad
        elif len(good) < 3:
            row["verdict"] = "UNDERPOWERED"
            row["reason"] = "fewer than three seeds"
        elif row["rel_ppl_change"] >= 0.01 and not row["inside_noise"]:
            row["verdict"] = "MISS"
        elif row["rel_ppl_change"] < 0.01 and math.expm1(floor) < 0.01:
            row["verdict"] = "PASS"
        else:
            row["verdict"] = "UNDERPOWERED"
            row["reason"] = "the seed noise does not separate this arm from 1 percent"
        table["arms"].append(row)
    with open(os.path.join(args.out, "training.json"), "w") as fh:
        json.dump(table, fh, indent=1)
    b = table["baseline"]
    lines = ["baseline fp32.v1: seeds %s, val loss at step %d mean %.5f, noise floor %.5f nats (%.3f%% of perplexity), range %.5f"
             % (b["seeds"], n_eq, b["mean"], floor, 100 * math.expm1(floor), b["range_nats"]), "",
             "| profile | products | mode | seeds | change at step %d | noise floor | inside noise | steps to baseline final | verdict |" % n_eq,
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table["arms"]:
        if "rel_ppl_change" in r:
            change = "%+.3f%% (%+.5f nats, se %.5f)" % (100 * r["rel_ppl_change"], r["delta_nats_mean"], r["delta_nats_se"])
            reach = ("median %s" % r["steps_to_baseline_final_median"]) if r["seeds_not_reaching"] == 0 else \
                "%d of %d seeds not within %d" % (r["seeds_not_reaching"], len(r["delta_nats_per_seed"]), n_end)
            inside = "yes" if r["inside_noise"] else "no"
        else:
            change, reach, inside = "non-finite", "-", "-"
        lines.append("| %s | %s | %s | %d | %s | %.3f%% | %s | %s | %s |" % (
            r["profile"], "projections + attention" if r["attention_products"] else "projections",
            "forward" if r["mode"] == "fwd" else "forward + backward", len(r["seeds"]), change,
            100 * r["noise_floor_rel_ppl"], inside, reach, r["verdict"]))
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, "training.md"), "w") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
