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

THE INTERVAL of an arm's change is the mean of its per-seed paired
differences plus and minus t(0.975, seeds - 1) standard errors of that mean
(Student's t, because the seeds are few), mapped through exp(x) - 1. The
seeds are the unit that is resampled. It bounds the run-to-run error AT THIS
SHAPE, ON THIS CORPUS, and says nothing about another shape or another text.

THE VERDICT against 1 percent:
  MISS          a run went non-finite, or the change is 1 percent or more
                and the lower end of its interval is above zero
  PASS          the change AND the upper end of its interval are both
                under 1 percent
  UNDERPOWERED  otherwise: the seeds cannot separate this arm from 1
                percent, and no conclusion is drawn

THE ZERO-CODE RECORD. At three steps of every run one diagnostic backward
pass records, for every product, the fraction of entries of each backward
operand whose code is 0 under each width (8, 10, 12 and 15 bits), in the
orientation its GEMM has. `training_zero_codes.md` prints the baseline's,
at the last step before the equal-step point, averaged over its seeds.
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


T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
FINALIST = {"int15-both+attn": "F1", "F2-int15proj-int8attn": "F2"}
WIDTHS = ("bf16", "int8", "int10", "int12", "int15")
DROPPED_NOTE = "dropped 2026-09-29, Andrew (not offered by the flag)"


def is_dropped(row_or_run):
    spec = row_or_run.get("spec") or {}
    kinds = [spec.get("weight_kind", ""), spec.get("activation_kind", "")]
    kinds += [k for pair in (spec.get("overrides") or {}).values() for k in pair]
    return any(k.startswith("int8") for k in kinds)
OPERANDS = (("Gt_rows_along_m", "G^T, the weight-gradient GEMM's left operand (rows over the tokens)"),
            ("At_rows_along_m", "A^T, the weight-gradient GEMM's right operand (rows over the tokens)"),
            ("G_rows_along_n", "G, the input-gradient GEMM's left operand (one row per token)"),
            ("Bt_rows_along_n", "B^T, the input-gradient GEMM's right operand"))


def zero_code_tables(runs, step):
    """Mean over the given runs of the zero-code fractions at `step`."""
    acc = {}
    for r in runs:
        rec = r.get("gradient_zero_codes", {}).get(str(step))
        if not rec:
            continue
        for product, ops in rec.items():
            for op, _ in OPERANDS:
                cell = acc.setdefault((product, op), dict(n=0, exact=0.0, **{w: 0.0 for w in WIDTHS},
                                                          **{w + "_nz": 0.0 for w in WIDTHS}))
                cell["n"] += 1
                cell["exact"] += ops[op]["exactly_zero_fraction"]
                for w in WIDTHS:
                    if w not in ops[op]:  # a record that did not code this operand under this width
                        cell[w] = cell[w + "_nz"] = float("nan")
                        continue
                    cell[w] += ops[op][w]["zero_code_fraction"]
                    cell[w + "_nz"] += ops[op][w]["zero_code_fraction_of_nonzero"]
    out = {}
    for (product, op), c in acc.items():
        n = c.pop("n")
        out.setdefault(product, {})[op] = {k: v / n for k, v in c.items()}
    return out


def zero_code_markdown(table, title):
    lines = [title, ""]
    widths = [w for w in WIDTHS
              if any(c.get(w) == c.get(w) for ops in table.values() for c in ops.values())]  # not NaN
    for op, what in OPERANDS:
        lines += ["### " + what, "",
                  "Fraction of the operand's NONZERO entries whose code is 0 (in brackets: of all entries).", "",
                  "| product | exactly zero as float32 | " + " | ".join(w for w in widths) + " |",
                  "|---|---|" + "---|" * len(widths)]
        for product in table:
            c = table[product].get(op)
            if c is None:
                continue
            lines.append("| %s | %.4f | %s |" % (product, c["exact"], " | ".join(
                "%.4f (%.4f)" % (c[w + "_nz"], c[w]) for w in widths)))
        lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="training", help="the basename of the files written")
    ap.add_argument("--only", default=None,
                    help="read only these arms besides the baseline: arm:mode:proj|attn, comma separated")
    args = ap.parse_args(argv)
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.runs, "run_*.json")))]
    if args.only:
        keep = set(tuple(x.split(":")) for x in args.only.split(","))
        runs = [r for r in runs if r["arm"] == "a"
                or (r["arm"], r["mode"], "attn" if r["attention_products"] else "proj") in keep]
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
                   spec=rs[0]["spec"], dropped=DROPPED_NOTE if is_dropped(rs[0]) else None,
                   seeds=sorted(r["seed"] for r in rs), nonfinite_seeds=sorted(bad),
                   nonfinite_at_step={r["seed"]: r["nonfinite_at_step"] for r in rs if r["seed"] in bad},
                   noise_floor_nats=floor, noise_floor_rel_ppl=math.expm1(floor))
        if good:
            d = [at(r, n_eq) - b_eq[r["seed"]] for r in good]
            ds = [smoothed(r, n_eq) - b_sm[r["seed"]] for r in good]
            mean = statistics.mean(d)
            se = statistics.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            half = T975.get(len(d) - 1, 1.96) * se if len(d) > 1 else float("nan")
            row.update(rel_ppl_change_lo=math.expm1(mean - half), rel_ppl_change_hi=math.expm1(mean + half),
                       interval="mean of the per-seed paired differences +- t(0.975, %d) standard errors, "
                                "through exp(x) - 1; the seeds are resampled" % (len(d) - 1))
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
        row["finalist"] = FINALIST.get(row["profile"])
        if bad:
            row["verdict"] = "MISS"
            row["reason"] = "non-finite loss in seed(s) %s" % bad
        elif len(good) < 3:
            row["verdict"] = "UNDERPOWERED"
            row["reason"] = "fewer than three seeds"
        elif row["rel_ppl_change"] < 0.01 and row["rel_ppl_change_hi"] < 0.01:
            row["verdict"] = "PASS"
        elif row["rel_ppl_change"] >= 0.01 and row["rel_ppl_change_lo"] > 0:
            row["verdict"] = "MISS"
        else:
            row["verdict"] = "UNDERPOWERED"
            row["reason"] = "the seeds do not separate this arm from 1 percent"
        if row["dropped"]:
            row["verdict"] += "; " + DROPPED_NOTE
        table["arms"].append(row)
    with open(os.path.join(args.out, args.name + ".json"), "w") as fh:
        json.dump(table, fh, indent=1)
    b = table["baseline"]
    lines = ["baseline fp32.v1: seeds %s, val loss at step %d mean %.5f, noise floor %.5f nats (%.3f%% of perplexity), range %.5f"
             % (b["seeds"], n_eq, b["mean"], floor, 100 * math.expm1(floor), b["range_nats"]), "",
             "The change is the mean over the seeds of (arm minus the baseline of the same seed) at step %d, as a "
             "relative change of validation perplexity. The interval is that mean plus and minus t(0.975, seeds - 1) "
             "standard errors; the seeds are resampled; it bounds the run-to-run error at this shape on this corpus "
             "and says nothing about another shape or text." % n_eq, "",
             "| profile | products | mode | seeds | change at step %d | interval | noise floor | inside noise | steps to baseline final | gradient cosine against fp32 (min) | verdict |" % n_eq,
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table["arms"]:
        if "rel_ppl_change" in r:
            change = "%+.3f%% (%+.5f nats)" % (100 * r["rel_ppl_change"], r["delta_nats_mean"])
            interval = "%+.3f%% to %+.3f%%" % (100 * r["rel_ppl_change_lo"], 100 * r["rel_ppl_change_hi"])
            reach = ("median %s" % r["steps_to_baseline_final_median"]) if r["seeds_not_reaching"] == 0 else \
                "%d of %d seeds not within %d" % (r["seeds_not_reaching"], len(r["delta_nats_per_seed"]), n_end)
            inside = "yes" if r["inside_noise"] else "no"
            cos = "%.6f" % r["gradient_cosine_against_fp32_min"] if r["gradient_cosine_against_fp32_min"] is not None else "-"
        else:
            change, interval, reach, inside, cos = "non-finite", "-", "-", "-", "-"
        name = r["profile"] + (" (%s)" % r["finalist"] if r.get("finalist") else "")
        lines.append("| %s | %s | %s | %d | %s | %s | %.3f%% | %s | %s | %s | %s |" % (
            name, "projections + attention" if r["attention_products"] else "projections",
            "forward" if r["mode"] == "fwd" else "forward + backward", len(r["seeds"]), change, interval,
            100 * r["noise_floor_rel_ppl"], inside, reach, cos, r["verdict"]))
    zc = zero_code_tables(list(complete.values()), n_eq - 1)
    table["baseline_gradient_zero_codes"] = dict(step=n_eq - 1, seeds=sorted(complete), products=zc)
    with open(os.path.join(args.out, args.name + ".json"), "w") as fh:
        json.dump(table, fh, indent=1)
    text_zc = zero_code_markdown(zc, "## Zero codes of the backward operands: fp32.v1 baseline, step %d, mean over seeds %s"
                                 % (n_eq - 1, sorted(complete)))
    table["arm_gradient_zero_codes"] = {}
    for (attn, arm, mode), rs in sorted(groups.items(), key=lambda kv: (not kv[0][0], kv[0][1], kv[0][2])):
        if arm != "e":
            continue
        rs = [r for r in rs if r["nonfinite_at_step"] is None]
        z = zero_code_tables(rs, n_eq - 1)
        if not z:
            continue
        name = rs[0]["profile_name"]
        table["arm_gradient_zero_codes"][name + "." + mode] = dict(step=n_eq - 1, seeds=sorted(r["seed"] for r in rs), products=z)
        text_zc += "\n" + zero_code_markdown(
            z, "## Zero codes of the backward operands: %s, %s, step %d, mean over seeds %s"
            % (name, "forward and backward" if mode == "fwdbwd" else "forward only", n_eq - 1,
               sorted(r["seed"] for r in rs)))
    with open(os.path.join(args.out, args.name + ".json"), "w") as fh:
        json.dump(table, fh, indent=1)
    with open(os.path.join(args.out, args.name + "_zero_codes.md"), "w") as fh:
        fh.write(text_zc)
    text = "\n".join(lines) + "\n"
    with open(os.path.join(args.out, args.name + ".md"), "w") as fh:
        fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
