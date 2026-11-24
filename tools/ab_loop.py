#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""tools/ab_loop.py -- A/B LOOP MODE (lane ab-loop-mode, 2026-11-24; AGENTS.md "Candidate evaluation": an A/B arm is
decided on loop time per iteration, under several conditions, never one).

THE TABLE (data): tools/ab_loop_caps.tsv says per family and lane whether a capped iteration count represents the fit
(valid), the PUBLIC parameter that caps it, the default cap, the parameters pinned so the executed count equals the
cap (early stopping off) and why. tools/ab_loop_conditions.tsv says which datasets are wide and which narrow. Nothing
here branches on a lane name: every lane-specific fact is a table row.

THE PLAN (tools/bench_board.py --loop-cap): bench_board puts one JSON plan per race into the driver's environment
(ENV); the driver applies it to OUR arms only:

    plan = ab_loop.driver_plan(family, lane)            # None: loop mode off (every board run but an A/B unit)
    params = ab_loop.capped_params(plan, params)        # the cap and the pins, equal on both arms (both are ours)
    ... construct, then the warm-up call (the same call at the cap) and the scored calls, as today ...
    loops = [ab_loop.read_fit_loop(model) for each scored call, read AFTER the call's clock has closed]
    cell["loop_mode"] = ab_loop.cell_record(plan, model.get_params(), loops, warm=ab_loop.read_fit_loop(...))

The product reports the loop region through the fitted attribute `fit_loop_` = (iterations, milliseconds): the clock
starts before the first iteration and stops after ONE device synchronization at the loop's end (core/fit_loop_clock.mojo).

THE VERDICT (tools/ops/targeted_rows.py summary, per owed A/B unit): arm_verdict below. A capped cell's number is the
loop time per executed iteration; the fixed cost (whole operation minus loop) is recorded beside it and flagged when
the arm adds to it, never deciding. A cell whose lane is not valid for the cap (or whose product does not report the
loop) keeps the whole-operation time and says so. The arm is PROMOTE when it covers a wide and a narrow dataset (two
sizes for the neural shapes) among those planned for it, is faster in aggregate (geometric mean of its cell ratios
below 1 - NOISE) and slower on no cell beyond the noise floor; SPLIT when it wins some conditions and loses one (fix
the losing condition or delete; never promoted as built); NOT-FASTER otherwise; INCOMPLETE while a planned width has no
compared cell. A PROMOTE arm whose output hash differs from the base's at the cap gets FULL-RUN-OWED: one full run of
both arms (loop mode off for that arm), for the final quality. Equal hashes: nothing more is run.

Tests: tools/test_ab_loop.py (pure Python, laptop).
"""
import json
import math
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
CAPS_TSV = os.path.join(HERE, "ab_loop_caps.tsv")
CONDITIONS_TSV = os.path.join(HERE, "ab_loop_conditions.tsv")
#: the driver environment key bench_board sets per race (a JSON plan; absent: loop mode off)
ENV = "MOJOLEARN_BENCH_LOOP_CAP"
#: the product's fitted attribute: (executed iterations, loop milliseconds)
ATTRIBUTE = "fit_loop_"
SCHEMA = "mojolearn-ab-loop/1"
#: THE NOISE FLOOR of a per-iteration ratio: 5 %, the same band tools/ops/targeted_rows.py direction() already treats
#: as "against" a one-way reading (ONE_WAY_AGAINST). A cell slower than 1 + NOISE is a loss; the aggregate must be
#: below 1 - NOISE to be a win. PC_AB_LOOP_NOISE (the job) overrides it for one run, recorded in the verdict.
NOISE = 0.05
_cache = {}


def _rows(path):
    with open(path) as fh:
        lines = [ln.rstrip("\n") for ln in fh if ln.strip() and not ln.startswith("#")]
    head = lines[0].split("\t")
    return [dict(zip(head, ln.split("\t"))) for ln in lines[1:]]


def load_caps(path=CAPS_TSV):
    """{(family, lane): row} of the cap table; row: valid (bool), param, cap (int), pin (dict), unit, reason."""
    key = ("caps", path)
    if key not in _cache:
        out = {}
        for r in _rows(path):
            if (r["family"], r["lane"]) in out:
                raise ValueError("ab_loop_caps.tsv: %s/%s listed twice" % (r["family"], r["lane"]))
            if r["valid"] not in ("yes", "no"):
                raise ValueError("ab_loop_caps.tsv: %s/%s valid must be yes or no" % (r["family"], r["lane"]))
            row = dict(valid=r["valid"] == "yes", param=None if r["cap_param"] == "-" else r["cap_param"],
                       cap=int(r["cap"]), pin=json.loads(r["pin"] or "{}"), unit=r["unit"], reason=r["reason"])
            if row["valid"] and (not row["param"] or row["cap"] < 1):
                raise ValueError("ab_loop_caps.tsv: %s/%s is valid without a cap parameter and cap" % (
                    r["family"], r["lane"]))
            out[(r["family"], r["lane"])] = row
        _cache[key] = out
    return _cache[key]


def load_conditions(path=CONDITIONS_TSV):
    """{condition: width} (wide, narrow or size)."""
    key = ("cond", path)
    if key not in _cache:
        _cache[key] = {r["condition"]: r["width"] for r in _rows(path)}
    return _cache[key]


def lookup(family, lane, caps=None):
    """The table row of (family, lane): the lane's own, else the family's '*' row, else an invalid row."""
    caps = load_caps() if caps is None else caps
    row = caps.get((family, lane)) or caps.get((family, "*"))
    if row is None:
        return dict(valid=False, param=None, cap=0, pin={}, unit="-",
                    reason="family %s is not in tools/ab_loop_caps.tsv" % family)
    return row


def race_plan(family, lane, cap=None, caps=None):
    """The JSON-able plan bench_board hands one race's driver: mode 'capped' (with the parameter, the cap, the pins)
    for a valid lane, else mode 'whole' with the table's reason. CAP overrides the table's default for valid rows."""
    row = lookup(family, lane, caps)
    plan = dict(schema=SCHEMA, family=family, lane=lane, unit=row["unit"], reason=row["reason"])
    if not row["valid"]:
        return dict(plan, mode="whole", param=None, k=None, pin={})
    return dict(plan, mode="capped", param=row["param"], k=int(cap) if cap else row["cap"], pin=dict(row["pin"]))


def driver_plan(family=None, lane=None, environ=None):
    """The plan of this driver process (ENV), or None (loop mode off). FAMILY/LANE, when given, must match the
    plan's: a plan for another race is refused (ValueError), never applied."""
    raw = (os.environ if environ is None else environ).get(ENV)
    if not raw:
        return None
    plan = json.loads(raw)
    for k, v in (("family", family), ("lane", lane)):
        if v is not None and plan.get(k) != v:
            raise ValueError("%s is for %s=%s, this driver runs %s" % (ENV, k, plan.get(k), v))
    return plan


#: a cap_param (or pin name) written `fit:<name>` is a keyword of the estimator's public fit() (Sequential.fit(epochs=K)),
#: not a constructor parameter: capped_params leaves it out, fit_kwargs gives it, and readback cannot read it from
#: get_params, so the executed iterations equal to the cap are its read-back (cell_record)
FIT_PREFIX = "fit:"


def _wanted(plan):
    return dict(plan.get("pin") or {}, **{plan["param"]: plan["k"]})


def capped_params(plan, params):
    """PARAMS (the constructor's) with the cap and the pins set (capped plans only), fit() keywords left out; a copy,
    never PARAMS itself."""
    out = dict(params or {})
    if plan and plan.get("mode") == "capped":
        out.update({k: v for k, v in _wanted(plan).items() if not k.startswith(FIT_PREFIX)})
    return out


def fit_kwargs(plan, kwargs=None):
    """KWARGS (the fit() call's) with the plan's `fit:` cap and pins set (capped plans only); a copy."""
    out = dict(kwargs or {})
    if plan and plan.get("mode") == "capped":
        out.update({k[len(FIT_PREFIX):]: v for k, v in _wanted(plan).items() if k.startswith(FIT_PREFIX)})
    return out


def readback(plan, effective):
    """{effective: {param: value}, ok, mismatch: [param]} of the capped parameters as the fitted estimator reports
    them (get_params()). A capped cell whose read-back differs from the plan is not compared (ok False)."""
    if not plan or plan.get("mode") != "capped":
        return dict(effective={}, ok=True, mismatch=[])
    want = {k: v for k, v in _wanted(plan).items() if not k.startswith(FIT_PREFIX)}
    eff = {k: (effective or {}).get(k, "<absent>") for k in want}
    bad = sorted(k for k, v in want.items() if not _same(eff[k], v))
    return dict(effective=eff, ok=not bad, mismatch=bad)


def _same(a, b):
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a is b or a == b
    try:
        return float(a) == float(b)
    except (TypeError, ValueError):
        return a == b


def read_fit_loop(model):
    """{status: recorded, iterations, loop_ms} from the model's fit_loop_, read after the clock has closed; or
    {status: unavailable, reason}. Never estimated."""
    try:
        value = getattr(model, ATTRIBUTE)
        it, ms = value
        it, ms = int(it), float(ms)
        if it < 0 or not math.isfinite(ms) or ms < 0:
            raise ValueError("invalid fit_loop_")
        return dict(status="recorded", iterations=it, loop_ms=ms)
    except (AttributeError, TypeError, ValueError, OverflowError) as e:
        return dict(status="unavailable", reason="the product does not report the loop region (%s: %s)" % (
            type(e).__name__, str(e)[:120]))


def cell_record(plan, effective, loops, warm=None):
    """cell["loop_mode"]: the plan's mode, the read-back, the scored calls' loop records (LOOPS), their median loop
    time and the executed iterations, and the mode the cell is decided in ('capped' only when the lane is valid, the
    read-back matches and every scored call reported its loop with the same executed count; otherwise 'whole' with
    why)."""
    rb = readback(plan, effective)
    rec = dict(schema=SCHEMA, planned=(plan or {}).get("mode", "off"), reason=(plan or {}).get("reason"),
               param=(plan or {}).get("param"), k=(plan or {}).get("k"), pin=(plan or {}).get("pin") or {},
               unit=(plan or {}).get("unit"), readback=rb, calls=list(loops or []), warmup=warm)
    got = [x for x in rec["calls"] if isinstance(x, dict) and x.get("status") == "recorded"]
    if got:
        rec["iterations"] = got[0]["iterations"]
        rec["loop_ms"] = statistics.median(x["loop_ms"] for x in got)
    if not plan or plan.get("mode") != "capped":
        rec["mode"] = "whole"
        rec["why"] = "lane not valid for a cap: %s" % rec["reason"] if plan else "loop mode off"
    elif not rb["ok"]:
        rec["mode"], rec["why"] = "whole", "read-back differs from the cap: %s" % ",".join(rb["mismatch"])
    elif not got or len(got) != len(rec["calls"]):
        bad = next((x for x in rec["calls"] if not (isinstance(x, dict) and x.get("status") == "recorded")), {})
        rec["mode"], rec["why"] = "whole", (bad or {}).get("reason") or "no scored call reported its loop"
    elif len({x["iterations"] for x in got}) != 1 or not got[0]["iterations"]:
        rec["mode"], rec["why"] = "whole", "executed iterations differ between scored calls or are zero"
    elif plan["param"].startswith(FIT_PREFIX) and got[0]["iterations"] != plan["k"]:
        # a fit() keyword cap has no get_params read-back: the executed count is its read-back
        rec["mode"], rec["why"] = "whole", "fit() cap %s=%s, executed %s" % (plan["param"], plan["k"],
                                                                           got[0]["iterations"])
    else:
        rec["mode"], rec["why"] = "capped", None
    return rec


# ---------------------------------------------------------------------------------------------- the verdict (box)

def cell_view(cell):
    """{mode, whole_ms, loop_ms, iterations, per_iteration_ms, fixed_ms, why, hash, quality, executed_ne_cap} of one
    board cell (whole_ms: the scored whole operation, cell median_ms)."""
    lm = (cell or {}).get("loop_mode") or {}
    whole = (cell or {}).get("median_ms")
    out = dict(mode=lm.get("mode") or "whole", why=lm.get("why") or ("loop mode off" if not lm else None),
               whole_ms=whole, loop_ms=lm.get("loop_ms"), iterations=lm.get("iterations"), k=lm.get("k"),
               hash=(cell or {}).get("hash"), quality=(cell or {}).get("quality") or {})
    if out["mode"] == "capped" and out["iterations"]:
        out["per_iteration_ms"] = out["loop_ms"] / out["iterations"]
        out["fixed_ms"] = whole - out["loop_ms"] if isinstance(whole, (int, float)) else None
        out["executed_ne_cap"] = bool(out["k"]) and out["iterations"] != out["k"]
    else:
        out["per_iteration_ms"] = out["fixed_ms"] = None
        out["executed_ne_cap"] = False
    return out


def pair(base_cell, arm_cell, noise=NOISE):
    """One condition's comparison: {basis: loop|whole, ratio, base, arm, why, fixed_up, same_hash, ...}.
    basis loop: both cells capped on the SAME executed iterations (the loop per iteration decides); basis whole: either
    cell not capped (the whole operation decides, the cell says why); ratio None: one side has no time."""
    b, a = cell_view(base_cell), cell_view(arm_cell)
    out = dict(base=b, arm=a, same_hash=bool(b["hash"]) and b["hash"] == a["hash"], fixed_up=False)
    if b["mode"] == a["mode"] == "capped" and b["iterations"] == a["iterations"]:
        out["basis"] = "loop"
        out["ratio"] = a["per_iteration_ms"] / b["per_iteration_ms"] if b["per_iteration_ms"] else None
        out["why"] = None
        if b["fixed_ms"] is not None and a["fixed_ms"] is not None and b["whole_ms"]:
            # an arm adding to the fixed cost beyond the noise floor of the whole operation: flagged, never deciding
            out["fixed_up"] = a["fixed_ms"] - b["fixed_ms"] > noise * b["whole_ms"]
    else:
        out["basis"] = "whole"
        bw, aw = b["whole_ms"], a["whole_ms"]
        out["ratio"] = aw / bw if isinstance(bw, (int, float)) and isinstance(aw, (int, float)) and bw > 0 else None
        if b["mode"] == a["mode"] == "capped":
            out["why"] = "executed iterations differ (base %s, arm %s)" % (b["iterations"], a["iterations"])
        else:
            out["why"] = a["why"] if a["mode"] != "capped" else b["why"]
    return out


def condition_of(race_id):
    """The condition of a board race id: its dataset ('family/lane/dataset/rows=R'), or its shape for a neural race
    ('neural/lane/dataset/shape=S')."""
    parts = race_id.split("/")
    if parts[0] == "neural" and len(parts) > 3 and parts[3].startswith("shape="):
        return parts[3][len("shape="):]
    return parts[2] if len(parts) > 2 else race_id


def unit_verdict(cell_pairs, planned_ids, noise=NOISE, conditions=None):
    """arm_verdict over one A/B unit. CELL_PAIRS: [(race id, base cell, arm cell)] of every cell key both boards hold;
    PLANNED_IDS: the unit's race ids. Returns (verdict, [(race id, pair)])."""
    pairs = [(rid, pair(b, a, noise)) for rid, b, a in cell_pairs]
    v = arm_verdict([(condition_of(rid), p) for rid, p in pairs], [condition_of(r) for r in planned_ids], noise,
                    conditions)
    v["full_run_races"] = sorted({rid for rid, p in pairs if p["basis"] == "loop" and not p["same_hash"]}) \
        if v["full_run"] else []
    return v, pairs


def width_of(dataset, conditions=None):
    return (load_conditions() if conditions is None else conditions).get(dataset)


def arm_verdict(pairs, planned_conditions, noise=NOISE, conditions=None):
    """The arm's verdict over its compared cells. PAIRS: [(condition, pair())], PLANNED_CONDITIONS: the datasets (or
    neural shapes) of every race planned for the arm. Returns {verdict, aggregate, cells, losses, wins, missing, why,
    full_run (bool), hash_differs [conditions], fixed_up [conditions], noise}."""
    conditions = load_conditions() if conditions is None else conditions
    rated = [(c, p) for c, p in pairs if p.get("ratio")]
    planned = {width_of(c, conditions) for c in planned_conditions} - {None}
    seen = {width_of(c, conditions) for c, _ in rated} - {None}
    missing = []
    if "size" in planned:
        sizes_planned = {c for c in planned_conditions if width_of(c, conditions) == "size"}
        sizes_seen = {c for c, _ in rated if width_of(c, conditions) == "size"}
        if len(sizes_seen) < min(2, len(sizes_planned)):
            missing.append("sizes (%d of %d planned compared)" % (len(sizes_seen), min(2, len(sizes_planned))))
    missing += sorted(w for w in planned & {"wide", "narrow"} if w not in seen)
    one_width = bool(planned & {"wide", "narrow"}) and len(planned & {"wide", "narrow"}) < 2
    ratios = [p["ratio"] for _, p in rated]
    agg = math.exp(sum(math.log(r) for r in ratios) / len(ratios)) if ratios else None
    losses = sorted({c for c, p in rated if p["ratio"] > 1.0 + noise})
    wins = sorted({c for c, p in rated if p["ratio"] < 1.0 - noise})
    out = dict(aggregate=agg, cells=len(rated), losses=losses, wins=wins, missing=missing, noise=noise,
               one_width=one_width, full_run=False,
               hash_differs=sorted({c for c, p in rated if p["basis"] == "loop" and not p["same_hash"]}),
               fixed_up=sorted({c for c, p in rated if p.get("fixed_up")}),
               whole_basis=sorted({c for c, p in rated if p["basis"] == "whole"}),
               untimed=sorted({c for c, p in pairs if not p.get("ratio")}))
    if not ratios or missing:
        out["verdict"] = "INCOMPLETE"
        out["why"] = "no compared cell" if not ratios else "no compared cell on: " + ", ".join(missing)
    elif agg < 1.0 - noise and not losses:
        out["verdict"] = "PROMOTE"
        out["why"] = "faster in aggregate, slower on none beyond the noise floor"
        out["full_run"] = bool(out["hash_differs"])
    elif wins and losses:
        out["verdict"] = "SPLIT"
        out["why"] = "wins %s, loses %s: fix the losing condition or delete; not promoted as built" % (
            ",".join(wins), ",".join(losses))
    else:
        out["verdict"] = "NOT-FASTER"
        out["why"] = "slower beyond the noise floor on %s" % ",".join(losses) if losses else \
            "not faster in aggregate beyond the noise floor"
    if one_width and out["verdict"] != "INCOMPLETE":
        out["why"] += "; only one width planned (add the other if its estimators support it)"
    return out


def verdict_line(arm, rows_tag, v):
    """One AB-LOOP line of the job's summary."""
    return ("AB-LOOP %s rows=%s %s aggregate %s cells=%d noise=%.2f%s%s%s%s: %s" % (
        arm, rows_tag, v["verdict"], "-" if v["aggregate"] is None else "%.3fx" % v["aggregate"], v["cells"],
        v["noise"], (" FULL-RUN-OWED (output differs from base at the cap: %s)" % ",".join(v["hash_differs"]))
        if v["full_run"] else "",
        (" FIXED-COST-UP %s" % ",".join(v["fixed_up"])) if v["fixed_up"] else "",
        (" whole-operation cells %s" % ",".join(v["whole_basis"])) if v["whole_basis"] else "",
        (" untimed %s" % ",".join(v["untimed"])) if v["untimed"] else "", v["why"]))


def pair_text(p):
    """' | loop/it base X (N it) -> arm Y ratio R[ fixed +F ms FIXED-COST-UP][ hash differs]' or the whole basis."""
    b, a = p["base"], p["arm"]
    if p["basis"] == "loop":
        return " | loop/it base %.4g ms (%d it) -> arm %.4g ms ratio %s; fixed base %s -> arm %s ms%s%s" % (
            b["per_iteration_ms"], b["iterations"], a["per_iteration_ms"],
            "-" if p["ratio"] is None else "%.3fx" % p["ratio"], _ms(b["fixed_ms"]), _ms(a["fixed_ms"]),
            " FIXED-COST-UP" if p["fixed_up"] else "", "" if p["same_hash"] else " hash differs at the cap")
    return " | whole operation (%s)" % (p.get("why") or "not capped")


def _ms(v):
    return "-" if v is None else "%.4g" % v
