#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""RACE FAIRNESS, applied by code (AGENTS.md "Race fairness: what the measurements are for", Andrew 2026-10-13).

The boards measure the cost of bitwise identity; every result is shown, wins and losses alike, and only a broken
comparison is hidden. This module is the one place the rule runs. It marks every (mode, phase) row of every race
of a board.json, and tools/main_board_ingest.py calls it on every rebuild. The paper (mlsys) and mojolearn.org
read the marks and omit rows with rerace "x" and opponent arms in race["opponents_excluded"].

    python3 tools/race_fairness.py bench/results/bench_board/main-nvidia-l40s/board.json ...   # apply, write
    python3 tools/race_fairness.py --check <board.json>...    # print what would change, write nothing
    python3 tools/race_fairness.py --review <board.json>...   # list warning signs on rows still shown

Publication requires sufficient evidence (Andrew, 2026-10-13, later clarification).
Existing supported audits remain usable; no blanket rerun is required. Unknown,
unresolved or unfair pairings are excluded individually. Each row carries the
admitted opponent arms, and is hidden only when none remain. Full measurement
fingerprints invalidate a decision when its supporting evidence changes.
Nothing here edits a measurement.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

TIE_BAND = (0.95, 1.05)

# 1a. Code our harness wrote rather than a library's own implementation: never an opponent (Andrew, 2026-10-13).
# lane -> predicate on the arm name. (OLS torch-gpu is torch.linalg.lstsq, a library routine, and stays.)
_TORCH = lambda arm: str(arm).startswith("torch")
HARNESS_ARMS = {
    "kmeans": lambda a: a == "torch-gpu",          # k-means from torch ops (index_add_)
    "knn": lambda a: a == "torch-gpu",             # cdist + topk
    "pca": lambda a: a == "torch-gpu",             # covariance + torch.linalg.eigh assembled by the harness
    "ols": lambda a: a == "torch-gpu-eigh",        # normal equations by eigh, harness-written
}
for _lane in ("lm-forward", "lm-train-step", "lm-infer", "lm-host-train-step", "transformer-forward",
              "transformer-infer", "mamba1-forward", "mamba2-forward", "mamba3-forward", "mamba1-infer",
              "mamba2-infer", "mamba3-infer", "samba-forward", "samba-train-step", "samba-infer", "moe",
              "resnet-block"):
    HARNESS_ARMS[_lane] = _TORCH                   # harness twins and pure-PyTorch references (the stored cells)

# Existing evidence-backed host finding is retained. The former automatic
# lane/arm/shape timing heuristic was insufficient to establish equal work.
SLOW_HOSTS = {
    "cc560ebdaf91": "L40S host about 1.75x slower than host 24a11adce16e on 36 matched torch cells (RNN/GRU/LSTM, same "
                    "shape) and 5x on matched cuML AutoARIMA (fairness audit 2026-10-13)",
}


def phase_cells(race, phase):
    return (race.get("cells") if phase == "fit" else race.get("infer_cells")) or []


def completed(cell):
    ms = cell.get("median_ms")
    return cell.get("status") == "ok" and isinstance(ms, (int, float)) and ms > 0


def our_arm(mode):
    return "ours-fast" if mode == "fast" else "ours"


def n_iter(cell):
    q = (cell or {}).get("quality") or {}
    for k in ("n_iter", "n_iter_", "iterations", "n_steps"):
        if isinstance(q.get(k), (int, float)) and not isinstance(q.get(k), bool):
            return q[k]
    return None


def cell_host(cell):
    host = ((cell.get("copied_from") or {}).get("box")) or ""
    return host or (cell.get("hardware") or "")


def legacy_pairing(race, mode, phase):
    """Fingerprint of what one (mode, phase) row measures: our arm's and every opponent's status and median."""
    rows = sorted((str(c.get("arm")), str(c.get("status")),
                   round(c["median_ms"], 6) if isinstance(c.get("median_ms"), (int, float)) else None)
                  for c in phase_cells(race, phase)
                  if c.get("arm") == our_arm(mode) or c.get("mode") == "opponent")
    return hashlib.sha256(json.dumps(rows).encode()).hexdigest()[:16]


def pairing(race, mode, phase):
    """Bind approval to actual evidence, never just unchanged median times."""
    cells = sorted((c for c in phase_cells(race, phase)
                    if c.get("arm") == our_arm(mode) or c.get("mode") == "opponent"),
                   key=lambda c: str(c.get("arm")))
    # Computed display ratios are not observations and can change on rebuild.
    ignored = {"ratio_ours_identical_over", "ratio_ours_fast_over", "ratio_ours_identical_clock",
               "ratio_ours_fast_clock", "clock"}
    basis = dict(cells=[{k: v for k, v in c.items() if k not in ignored} for c in cells],
                 params=race.get("params"), params_check=race.get("params_check"),
                 lane=race.get("lane"), dataset=race.get("dataset"),
                 lane_config=race.get("lane_config"))
    return "v2:" + hashlib.sha256(json.dumps(basis, sort_keys=True, default=str).encode()).hexdigest()


def slow_hosts(board):
    """Only recorded host findings; timing ratios alone do not establish cause.

    FAIRNESS FIX (2026-10-13 audit): matching lane/arm/shape did not verify
    matching settings, versions or resource allocation. Do not automatically
    turn that heuristic into a blanket exclusion of an entire host.
    """
    return dict(SLOW_HOSTS)


def exclusions(race, slow=None):
    """{arm: reason} for the opponent arms rules 1a and 1b exclude from this race."""
    out = {}
    pred = HARNESS_ARMS.get(race.get("lane"))
    slow = SLOW_HOSTS if slow is None else slow
    for c in phase_cells(race, "fit") + phase_cells(race, "infer"):
        arm = c.get("arm")
        if c.get("mode") != "opponent" or arm in out:
            continue
        if pred and pred(arm):
            implementation = (c.get("params") or {}).get("implementation") or {}
            module = implementation.get("module", "")
            # New real-library replacements retain the actual class read-back.
            # Historical harness arms keep their original exclusion.
            if not module.startswith(("transformers.", "mamba_ssm.", "hipvs.", "cuml.", "sklearn.")):
                out[arm] = "code our harness wrote, not a library's own implementation"
                continue
        for host, why in slow.items():
            if host in cell_host(c):
                out[arm] = "measured on %s, a slow box: %s" % (host, why)
    return out


def exclude(race, today, slow=None):
    """Merge rule 1 into race["opponents_excluded"] (arms a human audit excluded stay)."""
    found = exclusions(race, slow)
    if not found:
        return
    ex = race.setdefault("opponents_excluded", {"arms": [], "reason": "", "date": today})
    why = ex.setdefault("why", {})
    for arm, reason in sorted(found.items()):
        if arm not in ex["arms"]:
            ex["arms"] = sorted(ex["arms"] + [arm])
        why.setdefault(arm, reason)
    if not ex.get("reason"):
        ex["reason"] = "; ".join("%s: %s" % kv for kv in sorted(why.items()))


def exclusion_basis(race):
    cells = [c for phase in ("fit", "infer") for c in phase_cells(race, phase) if c.get("mode") == "opponent"]
    return hashlib.sha256(json.dumps(cells, sort_keys=True, default=str).encode()).hexdigest()


def _parameters(race, cell):
    record = cell.get("params") or ((race.get("params") or {}).get("arms") or {}).get(cell.get("arm"))
    if not record:
        fit = next((c for c in phase_cells(race, "fit") if c.get("arm") == cell.get("arm")), {})
        record = fit.get("params")
    return (record or {}).get("params") or {}


def _pair_problems(race, ours, opp):
    bad, missing = [], []
    if not completed(ours):
        missing.append("our measurement did not complete")
    for c in (ours, opp):
        q = c.get("quality") or {}
        if q.get("error") or q.get("finite") is False or q.get("host_reference_passed") is False:
            bad.append("recorded quality failure: " + str(c.get("arm")))
    a, b = n_iter(ours), n_iter(opp)
    if a is not None and b is not None and a != b:
        bad.append("iterations differ: ours %s, %s %s" % (a, opp.get("arm"), b))
    po, pt = _parameters(race, ours), _parameters(race, opp)
    # New paired runs must witness executed iterations, not infer them from
    # matching caps/tolerances. Historical accepted evidence is not rerun.
    if any(c.get("measurement") for c in (ours, opp)) and any(
            k in po or k in pt for k in ("max_iter", "maxiter", "n_iter", "n_iters")):
        actual = []
        for cell in (ours, opp):
            receipts = cell.get("iteration_receipts") or []
            if receipts and len(receipts) == cell.get("rounds") and all(
                    r and r.get("status") == "recorded" and r.get("values") for r in receipts):
                actual.append([r["values"] for r in receipts])
            elif n_iter(cell) is not None:
                actual.append(n_iter(cell))
            else:
                actual.append(None)
                missing.append("actual executed iterations not recorded: " + str(cell.get("arm")))
        if all(v is not None for v in actual) and actual[0] != actual[1]:
            bad.append("executed iteration receipts differ: %r vs %r" % tuple(actual))
    exceptions = list((race.get("params") or {}).get("exceptions") or [])
    exceptions.extend((opp.get("measurement") or {}).get("parameter_exceptions") or [])
    allowed = {e.get("param") for e in exceptions if e.get("arm") == opp.get("arm") and e.get("reason")}
    for key in sorted(set(po) & set(pt) - {"seed"}):
        if key in allowed:
            continue
        if po[key] is None or pt[key] is None:
            missing.append("unresolved setting: " + key)
        elif po[key] != pt[key]:
            bad.append("setting %s differs: %r vs %r" % (key, po[key], pt[key]))
    for cell in (ours, opp):
        m = cell.get("measurement")
        if m:
            if m.get("completeness") != "complete":
                missing.extend(m.get("missing") or ["incomplete measurement record"])
            if any((m.get(k) or {}).get("status") != "verified" for k in ("source", "artifacts")):
                missing.append("measurement source/logs not verified in R2")
    mo, mt = ours.get("measurement") or {}, opp.get("measurement") or {}
    if mo and mt:
        if (mo.get("dataset") or {}).get("identity") != (mt.get("dataset") or {}).get("identity"):
            bad.append("different dataset identity")
        if mo.get("shape") and mt.get("shape") and mo["shape"] != mt["shape"]:
            bad.append("different measured input shape")
    return bad, missing, bool(po and pt and (set(po) & set(pt) - {"seed"}))


def fairness_rule(race, mode, phase, today, prior=None):
    """The entry for a row no audit covers, or None when there is no row (no our cell, or no completed opponent that
    is not excluded)."""
    cells = phase_cells(race, phase)
    our_cells = [c for c in cells if c.get("arm") == our_arm(mode)]
    ours = next(iter(our_cells), None)
    excluded = set((race.get("opponents_excluded") or {}).get("arms") or [])
    opps = [c for c in cells if c.get("mode") == "opponent" and completed(c) and c.get("arm") not in excluded]
    if ours is None:
        return None
    judged, admitted = {}, []
    def batch(c):
        return c.get("batch"), c.get("batch_rows")
    for arm in sorted({c["arm"] for c in opps}):
        group = [c for c in opps if c["arm"] == arm]
        opp = next((c for c in group if batch(c) == batch(ours)), group[0])
        arm = opp["arm"]
        bad, missing, readback = [], [], True
        for candidate in group:
            own = next((c for c in our_cells if batch(c) == batch(candidate)), None)
            if own is None:
                missing.append("no matching inference batch")
                continue
            problems, gaps, checked = _pair_problems(race, own, candidate)
            bad.extend(problems); missing.extend(gaps); readback = readback and checked
        if any(not any(batch(c) == batch(o) for c in group) for o in our_cells):
            missing.append("opponent lacks a matching batch")
        # Historical hand verdicts judged rows, not an explicit per-arm map.
        # A problem in our result (or an unscoped old finding) cannot be cured
        # by silently choosing a different opponent. A later audit can scope it.
        hand = prior and not prior.get("rule") and (prior.get("opponent") == arm or
            (prior.get("verdict") in ("UNFAIR", "SUSPECT", "INSUFFICIENT") and prior.get("scope") != "opponent"))
        old_pair = ((prior or {}).get("opponents") or {}).get(arm)
        if old_pair and old_pair.get("audit"):
            hand = old_pair
        else:
            hand = prior if hand else None
        if bad:
            verdict, reason = "UNFAIR", "; ".join(bad)
        elif missing:
            verdict, reason = "INSUFFICIENT", "; ".join(sorted(set(missing)))
        elif hand and hand.get("verdict") in ("UNFAIR", "SUSPECT", "INSUFFICIENT"):
            verdict, reason = hand["verdict"], hand.get("reason", "audit unresolved")
        elif hand and hand.get("verdict") in ("FAIR", "TIE", "MATCHED") and hand.get("evidence"):
            verdict, reason = hand["verdict"], hand.get("reason", "retained supported audit")
        elif readback and (race.get("params_check") == "MATCHED" or
                           (ours.get("measurement") and opp.get("measurement"))):
            verdict, reason = "MATCHED", "recorded effective settings compared on both arms"
        else:
            verdict, reason = "INSUFFICIENT", "no supported audit or matched effective settings for this pairing"
        row = dict(verdict=verdict, reason=reason, rerace="" if verdict in ("FAIR", "TIE", "MATCHED") else "x")
        if hand:
            row.update({k: hand[k] for k in ("audit", "evidence", "date") if k in hand})
        judged[arm] = row
        if not row["rerace"]:
            admitted.append(opp)
    best = min([c for c in admitted if c.get("device") == "gpu"] or admitted,
               key=lambda c: c["median_ms"]) if admitted else None
    ratio = ours["median_ms"] / best["median_ms"] if completed(ours) and best else None
    entry = {"rule": "race_fairness.fairness_rule (2026-10-13)", "date": today, "opponent": best.get("arm") if best else None,
             "ours_over_theirs": round(ratio, 3) if ratio is not None else None,
             "pairing": pairing(race, mode, phase), "opponents": judged,
             "admitted_opponents": [c["arm"] for c in admitted]}
    if best is None:
        entry.update(verdict="UNFAIR" if any(e["verdict"] == "UNFAIR" for e in judged.values()) else "INSUFFICIENT",
                     rerace="x", reason="no opponent with sufficient valid comparison evidence")
    else:
        accepted = judged[best["arm"]]
        entry.update(verdict=accepted["verdict"], rerace="", reason=accepted["reason"])
        if ratio is not None and TIE_BAND[0] <= ratio <= TIE_BAND[1]:
            entry["verdict"] = "TIE"
    return entry


def apply(board, prev=None, today=None, upgrade_legacy=False):
    """Mark every row of every race of `board` (rules 1-3). `prev`: the previous version of the board, whose audit
    entries and exclusions carry over while unchanged (None: the board's own). Returns the races marked "x"."""
    today = today or datetime.date.today().isoformat()
    prev = board if prev is None else prev
    modes = (board.get("config") or {}).get("modes") or ["identical"]
    slow = slow_hosts(board)
    marked = 0
    for section in ("races", "extra_races"):
        for rid, rr in ((board.get(section) or {}).items()):
            prev_rr = (((prev or {}).get("races") or {}).get(rid)
                       or ((prev or {}).get("extra_races") or {}).get(rid) or {})
            old = prev_rr.get("audit") or {}
            old_ex = rr.get("opponents_excluded") or prev_rr.get("opponents_excluded")
            basis = exclusion_basis(rr)
            if old_ex and (old_ex.get("basis") == basis or (upgrade_legacy and not old_ex.get("basis"))):
                rr["opponents_excluded"] = json.loads(json.dumps(old_ex))
            elif old_ex:
                rr.pop("opponents_excluded", None)
            exclude(rr, today, slow)
            if rr.get("opponents_excluded"):
                rr["opponents_excluded"]["basis"] = basis
            audit = {}
            for mode in modes:
                for phase in ("fit", "infer"):
                    key = "%s/%s" % (mode, phase)
                    entry = old.get(key)
                    keep = entry and entry.get("pairing") == pairing(rr, mode, phase)
                    if upgrade_legacy and entry and entry.get("pairing") == legacy_pairing(rr, mode, phase):
                        keep = True
                        history = rr.setdefault("audit_history", [])
                        archived = dict(entry, row=key)
                        if archived not in history:
                            history.append(archived)
                    prior = entry if keep else None
                    if entry and not keep:
                        history = rr.setdefault("audit_history", [])
                        if entry not in history:
                            history.append(entry)
                    entry = fairness_rule(rr, mode, phase, today, prior=prior)
                    if entry:
                        audit[key] = entry
            if audit:
                rr["audit"] = audit
            else:
                rr.pop("audit", None)
            if any(e.get("rerace") == "x" for e in audit.values()):
                rr["rerace"] = "x"
                marked += 1
            else:
                rr.pop("rerace", None)
    return marked


def review(board):
    """Warning signs on rows still shown, for a human to judge: (lane, dataset, key, signs)."""
    out = []
    for section in ("races", "extra_races"):
        for rr in ((board.get(section) or {}).values()):
            excluded = set((rr.get("opponents_excluded") or {}).get("arms") or [])
            for key, e in sorted((rr.get("audit") or {}).items()):
                if e.get("rerace") == "x":
                    continue
                mode, phase = key.split("/")
                cells = phase_cells(rr, phase)
                ours = next((c for c in cells if c.get("arm") == our_arm(mode)), None)
                opp = next((c for c in cells if c.get("arm") == e.get("opponent")), None)
                if ours is None or opp is None or opp.get("arm") in excluded:
                    continue
                signs = []
                r = e.get("ours_over_theirs")
                if isinstance(r, (int, float)) and (r < 0.2 or r > 20):
                    signs.append("extreme ratio %.3g" % r)
                qo, qt = ours.get("quality") or {}, opp.get("quality") or {}
                for k in ("accuracy", "r2", "auc", "logloss", "inertia", "explained_variance_ratio_sum", "mean_log_likelihood"):
                    a, b = qo.get(k), qt.get(k)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and max(abs(a), abs(b)) > 0:
                        if abs(a - b) / max(abs(a), abs(b)) > 0.05:
                            signs.append("quality %s %.4g vs %.4g" % (k, a, b))
                        break
                times = ours.get("times_ms") or []
                med = ours.get("median_ms")
                if times and isinstance(med, (int, float)) and all(abs(t - med) > 1e-6 * max(1, med) for t in times):
                    signs.append("our median is not among our timed rounds")
                if e.get("verdict") == "UNCHECKED":
                    signs.append("no audit")
                if signs:
                    out.append((rr.get("lane"), rr.get("dataset"), key, e.get("verdict"), signs))
    return out


def _load(path):
    raw = open(path).read()
    board = json.loads(raw)
    second = raw.split("\n")[1] if "\n" in raw else ""
    fmt = dict(indent=len(second) - len(second.lstrip()) or None, tail="\n" if raw.endswith("\n") else "",
               sort=json.dumps(board, indent=len(second) - len(second.lstrip()) or None, sort_keys=True,
                               default=str) + ("\n" if raw.endswith("\n") else "") == raw)
    return board, raw, fmt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("boards", nargs="+")
    ap.add_argument("--check", action="store_true", help="print what would change; write nothing")
    ap.add_argument("--review", action="store_true", help="list warning signs on rows still shown")
    ap.add_argument("--date", default=datetime.date.today().isoformat())
    ap.add_argument("--upgrade-legacy-audits", action="store_true",
                    help="explicitly bind existing supported hand audits to the current full evidence (no measurements)")
    args = ap.parse_args(argv)
    for path in args.boards:
        board, raw, fmt = _load(path)
        if args.review:
            rows = review(board)
            print("%s: %d shown rows with warning signs" % (path, len(rows)))
            for lane, ds, key, verdict, signs in rows:
                print("  %-22s %-14s %-15s %-9s %s" % (lane, ds, key, verdict, "; ".join(signs)))
            continue
        before = json.loads(raw)
        marked = apply(board, None, args.date, upgrade_legacy=args.upgrade_legacy_audits)
        changed = []
        for section in ("races", "extra_races"):
            for rid, rr in (board.get(section) or {}).items():
                old = (before.get(section) or {}).get(rid) or {}
                for key, e in (rr.get("audit") or {}).items():
                    o = (old.get("audit") or {}).get(key) or {}
                    if (o.get("verdict"), o.get("rerace")) != (e.get("verdict"), e.get("rerace")):
                        changed.append((rr.get("lane"), rr.get("dataset"), key, o.get("verdict"), e.get("verdict")))
                if (old.get("opponents_excluded") or {}).get("arms") != (rr.get("opponents_excluded") or {}).get("arms"):
                    changed.append((rr.get("lane"), rr.get("dataset"), "excluded arms",
                                    (old.get("opponents_excluded") or {}).get("arms"),
                                    (rr.get("opponents_excluded") or {}).get("arms")))
        print("%s: %d races marked x, %d changes" % (path, marked, len(changed)))
        for ch in changed[:60]:
            print("  %-22s %-14s %-15s %s -> %s" % ch)
        if not args.check:
            text = json.dumps(board, indent=fmt["indent"], sort_keys=fmt["sort"], default=str) + fmt["tail"]
            with open(path + ".tmp", "w") as fh:
                fh.write(text)
            os.replace(path + ".tmp", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
