#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which public names the measurement board covers (static; no fit, no GPU, no driver run).

    python3 tools/measurement_coverage.py --out coverage.tsv [--boards DIR ...] [--summary]

For every public name of tools/public_runtime_inventory.py (the 418-name runtime inventory) it answers, from
the repository and the stored boards only:

  recipe     the board lanes (bench_board.plan_races on NVIDIA and AMD, identical) that call the name, with
             the phases the lane clocks: fit (the lane's own clock) and infer (predict / transform / search
             on held-out rows, where the lane has one)
  smoke      whether every one of those recipes is in the gate smoke's plan
             (gpu_gap_campaign.selected_plan(vendor, None, smoke=True, cpu_fallback=True), the selection
             tools/gate_smoke_box.py smoke runs)
  opponent   the planned opponent arms per vendor (nv:..., amd:...), "none" for an ours-only lane
  params_state  the stored main boards' board_pairing.params_state counts for those lanes'
             opponent cells (MATCHED / unchecked / refused), or "not raced" when no stored race exists
  action     covered | race (a planned line never raced with a parameter check) | re-race (stored pairs
             are legacy copies or refused: the gate's measure stage re-races them) | add-recipe | n/a: <why>

A name is n/a only for a reason in NOT_MEASURED (a pure getter or container, a refusal path, multi-GPU,
a CPU-only host binding the board never times, an I/O or metadata object). Everything else with no lane is
add-recipe. tools/test_measurement_coverage.py keeps both tables honest.
"""
import argparse
import collections
import fnmatch
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

VENDORS = ("nvidia", "amd")
MAIN_BOARDS = {"nvidia": "bench/results/bench_board/main-nvidia-l40s/board.json",
               "amd": "bench/results/bench_board/main-amd-mi325x/board.json"}

#: The public classes the trees / classical / classical2 drivers construct (the algos lanes name theirs in
#: bench_board_algos.LANES[lane]["ours"], the neural lanes in bench_board_neural.LANE_TEXT). Read from the
#: drivers: tools/speed_gbdt_arm.py and bench/speed/forest_speed_arm.py (trees),
#: tools/classical_two_datasets.py (classical), tools/bench_board_more.py (classical2).
DRIVER_CLASSES = {
    "trees/gbdt-symmetric": ("GradientBoosting", "GradientBoostingRegressor", "GradientBoostingClassifier"),
    "trees/gbdt-symmetric-1000": ("GradientBoosting",),
    "trees/gbdt-depthwise": ("GradientBoosting",),
    "trees/gbdt-lossguide": ("GradientBoosting",),
    "trees/gbdt-rank-yetirank": ("GradientBoosting",),
    "trees/gbdt-rank-pairlogit": ("GradientBoosting",),
    "trees/gbdt-multiclass": ("GradientBoosting", "GradientBoostingClassifier"),
    "trees/gbdt-categorical": ("GradientBoosting",),
    "trees/gbdt-ordered": ("GradientBoosting",),
    "trees/rf": ("RandomForest", "RandomForestClassifier", "RandomForestRegressor"),
    "trees/et": ("ExtraTrees", "ExtraTreesClassifier", "ExtraTreesRegressor"),
    "trees/iforest": ("IsolationForest",),
    "classical/kmeans": ("KMeans",),
    "classical/pca": ("PCA",),
    "classical/ols": ("LinearRegression",),
    "classical/knn": ("NearestNeighbors",),
    "classical/kde": ("KernelDensity",),
    "classical/svc": ("SVC",),
    "classical/dbscan": ("DBSCAN",),
    "classical/hdbscan": ("HDBSCAN",),
    "classical2/umap": ("UMAP",),
    "classical2/spectral-embedding": ("SpectralEmbedding",),
    "classical2/gmm": ("GaussianMixture",),
    "classical2/logreg": ("LogisticRegression",),
    "classical2/linearsvc": ("LinearSVC",),
    "classical2/ridge": ("Ridge",),
    "classical2/lasso": ("Lasso",),
    "classical2/elasticnet": ("ElasticNet",),
    "classical2/linearsvr": ("LinearSVR",),
    "classical2/tsvd": ("TruncatedSVD",),
    "classical2/knn-clf": ("KNeighborsClassifier",),
    "classical2/knn-reg": ("KNeighborsRegressor",),
    "classical2/spectral": ("SpectralClustering",),
    "classical2/agglomerative": ("AgglomerativeClustering",),
    "classical2/gpr": ("GaussianProcessRegressor", "RBF", "ConstantKernel", "WhiteKernel"),
    "classical2/gpc": ("GaussianProcessClassifier", "RBF", "ConstantKernel"),
    "classical2/svr": ("SVR",),
    "classical2/kernel-ridge": ("KernelRidge",),
    "classical2/nystroem": ("Nystroem",),
    "classical2/rbf-sampler": ("RBFSampler",),
    "classical2/arima": ("ARIMA",),
    "classical2/ets": ("ExponentialSmoothing",),
    "classical2/ivf": ("IVFIndex",),
}

#: Neural lanes whose LANE_TEXT names no constructor call (`mojolearn.X(`).
NEURAL_EXTRA = {
    "neural/sequential-predict": ("Sequential",),
    "neural/program-run": ("ops.Session", "ops.Program"),
    "neural/ops-session-step": ("ops.Session",),
}

#: Public names whose compute a lane times through another public entry (the lane, and why). The algos
#: driver's own list of such names is bench_board_algos.NOT_RACED; this table holds them by canonical name.
INDIRECT = {
    "mojolearn.DampedETS": ("algos/damped-ets", "ETS(model='AAN', damped=True) by construction, the model the "
                                                "damped-ets lane races (bench_board_algos.NOT_RACED)"),
    "mojolearn.sparse_encode": ("algos/sparse-coder", "the encoder SparseCoder.transform runs, raced as "
                                                      "algos/sparse-coder (bench_board_algos.NOT_RACED)"),
    "mojolearn.lu_solve": ("algos/lu-factor", "the lane times lu_factor(A) then lu_solve(., B)"),
    "mojolearn.manifold.spectral_embedding": ("classical2/spectral-embedding",
                                              "the function form of SpectralEmbedding (cuML's "
                                              "manifold.spectral_embedding), the class the lane races"),
    "mojolearn.OrderedRMSE": ("trees/gbdt-ordered", "a fixed GradientBoosting configuration (loss RMSE, "
                                                    "ordered boosting) of the gbdt-ordered lane's estimator"),
}

#: Public names the board does not time, each with its reason (fnmatch patterns over the canonical name).
#: Andrew 2026-10-22: every exposed API is measured "if appropriate"; these are the ones where it is not.
NOT_MEASURED = (
    # multi-GPU: excluded from the board by the standing rule (Andrew 2026-09-29, COVERAGE.md)
    ("mojolearn.Parallel*", "multi-GPU, excluded from the single-GPU board (Andrew 2026-09-29, COVERAGE.md)"),
    ("mojolearn.Pooled*", "multi-GPU model pool, excluded from the single-GPU board (COVERAGE.md)"),
    ("mojolearn.Offloaded*", "multi-GPU/offload trainer, excluded from the single-GPU board (COVERAGE.md)"),
    ("mojolearn.cross_vendor.*", "multi-GPU cross-vendor coordination, excluded from the single-GPU board"),
    ("mojolearn.multi_gpu.*", "multi-GPU, excluded from the single-GPU board (COVERAGE.md)"),
    ("mojolearn.parallel_*", "multi-GPU (explicit devices / worker processes), excluded from the single-GPU "
                             "board (COVERAGE.md)"),
    ("mojolearn.models.Parallel*", "multi-GPU model, excluded from the single-GPU board (COVERAGE.md)"),
    # CPU host bindings: the CPU is never timed (CLAUDE.md "Never time a CPU or host route")
    ("mojolearn.*Inference", "CPU host binding (inference); the board never times a CPU route (CLAUDE.md)"),
    ("mojolearn.LanguageModelHostTrainer", "CPU host binding; the board never times a CPU route (CLAUDE.md)"),
    ("mojolearn.Host*", "CPU host binding; the board never times a CPU route (CLAUDE.md)"),
    ("mojolearn.host_*", "CPU host binding; the board never times a CPU route (CLAUDE.md)"),
    # no peer library implements it
    ("mojolearn.resample.monte_carlo_integrate", "no pinned library has a plain seeded Monte Carlo integrator "
                                                 "(scipy.integrate.qmc_quad is quasi-Monte Carlo); "
                                                 "bench_board_algos.NOT_RACED"),
    ("mojolearn.ExperimentalTwoLevelFeatureFreq", "experimental one-tree depth-two helper; no peer library "
                                                  "implements it"),
    # getters, factories and metadata: no compute of their own
    ("mojolearn.model_selection.check_cv", "returns a splitter object; no compute"),
    ("mojolearn.model_selection.get_scorer", "returns a scorer object; its compute is the metric lanes'"),
    ("mojolearn.model_selection.get_scorer_names", "returns names; no compute"),
    ("mojolearn.model_selection.make_scorer", "returns a scorer object; its compute is the metric lanes'"),
    ("mojolearn.model_selection.split_descriptor", "returns a split description; no compute"),
    ("mojolearn.training.accumulation_is_aligned", "a boolean check of two sizes; no compute"),
    ("mojolearn.training.numeric_mode_used", "getter; no compute"),
    ("mojolearn.training.vendor_used", "getter; no compute"),
    ("mojolearn.ensemble.catboost_auto_learning_rate", "a scalar formula of the row count; no array compute"),
    ("mojolearn.models.plan_for", "returns an import plan record; no compute"),
    ("mojolearn.import_model", "one-time model import (parse and weight load, disk-bound); the imported "
                               "program's compute is ops.Program, timed by neural/program-run"),
    # containers, metadata, I/O and RNG streams: no compute to time against a peer
    ("mojolearn.Array", "array container; its compute is timed through every lane that takes one"),
    ("mojolearn.Generator", "RNG stream object; correctness only (seeded draws), no timed peer"),
    ("mojolearn.Checkpoint", "checkpoint I/O object; disk-bound, no GPU compute to compare"),
    ("mojolearn.SafetensorsFile", "file reader; disk-bound, no GPU compute to compare"),
    ("mojolearn.HFConfig", "configuration record; no compute"),
    ("mojolearn.TensorInfo", "metadata record; no compute"),
    ("mojolearn.ModelPlan", "import plan record; no compute"),
    ("mojolearn.CausalLMState", "state record of CausalLM; timed through neural/causal-lm-generate"),
    ("mojolearn.models.CausalLMState", "state record of CausalLM; timed through neural/causal-lm-generate"),
    ("mojolearn.GenerationStream", "iterator over CausalLM.generate; timed through neural/causal-lm-generate"),
    ("mojolearn.models.GenerationStream", "iterator over CausalLM.generate; timed through neural/causal-lm-generate"),
    ("mojolearn.*Kernel", "kernel specification object (gaussian_process); timed through classical2/gpr and gpc"),
    ("mojolearn.config_context", "configuration context manager; no compute"),
    ("mojolearn.get_config", "configuration getter; no compute"),
    ("mojolearn.set_config", "configuration setter; no compute"),
    ("mojolearn.show_versions", "diagnostic printout; no compute"),
    ("mojolearn.device_info", "device query; no compute"),
    ("mojolearn.*Error", "exception class; no compute"),
    ("mojolearn.*Warning", "warning class; no compute"),
    ("mojolearn.LearningRateTable", "precomputed schedule record; no compute"),
)


def not_measured(name):
    for pattern, why in NOT_MEASURED:
        if fnmatch.fnmatchcase(name, pattern):
            return why
    return None


def _bb():
    import bench_board as B
    return B


def lane_tokens(family, lane):
    """The public-name tokens (as the drivers spell them, without `mojolearn.`) a lane calls."""
    B = _bb()
    key = "%s/%s" % (family, lane)
    if family == "algos":
        spec = B.ALGOS.LANES[lane]
        # a SelectKBest lane calls its score function inside the timed fit (bench_board_algos select-* lanes)
        return tuple(spec["ours"]) + ((spec["score_func"],) if spec.get("score_func") else ())
    if family == "neural":
        text = (B.NEURAL.LANE_TEXT.get(lane) or ("",))[0]
        found = re.findall(r"mojolearn\.([A-Za-z_][\w.]*?)\(", text)
        found = [re.sub(r"\.(load|from_pretrained|decode_session)$", "", t) for t in found]
        return tuple(dict.fromkeys(found + list(NEURAL_EXTRA.get(key, ()))))
    return DRIVER_CLASSES.get(key, ())


class Names:
    """Public alias -> inventory canonical name (aliases from public_runtime_inventory.public_names)."""

    def __init__(self, canonicals, groups):
        self.canonicals = set(canonicals)
        self.full, self.tail = {}, collections.defaultdict(set)
        for aliases in groups:
            members = set(aliases)
            target = next((c for c in sorted(members) if c in self.canonicals), None)
            if target is None:
                continue
            for a in members:
                self.full[a] = target
                self.tail[a.rsplit(".", 1)[-1]].add(target)
        for c in self.canonicals:
            self.full.setdefault(c, c)
            self.tail[c.rsplit(".", 1)[-1]].add(c)

    def resolve(self, token):
        """The canonical name of a lane token, or None (an unknown or ambiguous short name)."""
        hit = self.full.get("mojolearn." + token) or self.full.get(token)
        if hit:
            return hit
        cands = self.tail.get(token.rsplit(".", 1)[-1]) or set()
        return next(iter(cands)) if len(cands) == 1 else None


def inventory(root=REPO, tsv=None):
    """({canonical: {"family", "methods"}}, Names)."""
    import public_runtime_inventory as P
    rows = {}
    if tsv:
        for rec in P.read_tsv(Path(tsv)):
            r = rows.setdefault(rec["class"], {"family": rec["family"], "methods": []})
            r["methods"].append(rec["method"])
    else:
        for rec in P.inventory_rows(Path(root)):
            r = rows.setdefault(rec.public, {"family": rec.family, "methods": []})
            r["methods"].append(rec.method)
    groups = [[c, *aliases] for c, _origin, aliases in P.public_names(Path(root))]
    return rows, Names(rows, groups)


def plans():
    """{vendor: {race id: race}} of the board plan and {vendor: set(race ids)} of the gate smoke plan."""
    B = _bb()
    import gpu_gap_campaign as G
    plan = {v: {r["id"]: r for r in B.plan_races(v, ["identical"])} for v in VENDORS}
    smoke = {v: {r["id"] for r in G.selected_plan(v, None, smoke=True, cpu_fallback=True)} for v in VENDORS}
    return plan, smoke


def stored_pairs(boards=None, root=REPO):
    """{vendor: {"family/lane": Counter(MATCHED|unchecked|refused)}} over the stored main boards."""
    B = _bb()
    out = {}
    for vendor, rel in (boards or MAIN_BOARDS).items():
        path = Path(root, rel)
        counts = collections.defaultdict(collections.Counter)
        if path.is_file():
            doc = json.loads(path.read_text())
            rep = B.pair_report(doc)
            for state in ("unchecked", "refused"):
                for row in rep[state]:
                    counts[lane_key(row["race"])][state] += 1
            for rid, rr in (doc.get("races") or {}).items():
                for c in rr.get("cells") or []:
                    if (B.PAIR.is_opponent(c) and c.get("median_ms") and c.get("status") in B.TIMED_STATUSES
                            and B.PAIR.params_state(rr, c)[0] == B.PAIR.PAIR_MATCHED):
                        counts[lane_key(rid)]["MATCHED"] += 1
                counts[lane_key(rid)]["raced"] += 1
        out[vendor] = counts
    return out


def refused_by_name(keys):
    """'arm: reason; ...' of the opponents the algos lanes `keys` refuse by name (bench_board_algos LANES
    `refused`, printed by not_planned), or ''."""
    B = _bb()
    out = []
    for k in keys:
        fam, lane = k.split("/", 1)
        if fam == "algos":
            out += ["%s: %s" % (a, str(why)[:160]) for a, why in sorted(B.ALGOS.LANES[lane].get("refused", {}).items())]
    return "; ".join(out)


def lane_key(race_id):
    return "/".join(str(race_id).split("/")[:2])


def lane_map(plan):
    """{"family/lane": {"vendors": {vendor: [race ids]}, "tokens", "opponents": {vendor: [...]}, "infer"}}."""
    B = _bb()
    lanes = {}
    for vendor, races in plan.items():
        for rid, r in races.items():
            key = "%s/%s" % (r["family"], r["lane"])
            e = lanes.setdefault(key, {"vendors": {}, "opponents": {}, "tokens": lane_tokens(r["family"], r["lane"]),
                                       "infer": _has_infer(B, r)})
            e["vendors"].setdefault(vendor, []).append(rid)
            e["opponents"][vendor] = sorted(set(e["opponents"].get(vendor, [])) | set(r.get("opponents") or []))
    return lanes


def _has_infer(B, race):
    if race["family"] == "algos":
        return bool(B.ALGOS.has_infer(race["lane"]))
    try:
        return bool(B.INFER.plan_cells(dict(race)))
    except Exception:  # noqa: BLE001 - a planner without inference cells for this family
        return False


def coverage(root=REPO, inventory_tsv=None, boards=None):
    rows, names = inventory(root, inventory_tsv)
    plan, smoke = plans()
    lanes = lane_map(plan)
    pairs = stored_pairs(boards, root)
    by_name = collections.defaultdict(list)
    unresolved = {}
    for key, e in sorted(lanes.items()):
        for token in e["tokens"]:
            canonical = names.resolve(token)
            if canonical is None:
                unresolved.setdefault(key, []).append(token)
                continue
            by_name[canonical].append(key)
    out = []
    for name in sorted(rows):
        r = rows[name]
        keys = sorted(set(by_name.get(name, [])))
        if not keys and name in INDIRECT and INDIRECT[name][0] in lanes:
            lane, why = INDIRECT[name]
            out.append(dict(name=name, family=r["family"], recipe="%s[indirect]" % lane,
                            smoke="yes" if all(rid in smoke[v] for v, rids in lanes[lane]["vendors"].items()
                                               for rid in rids) else "NO",
                            opponent="-", params_state="see " + lane, action="covered (indirect: %s)" % why))
            continue
        if not keys:
            why = not_measured(name)
            out.append(dict(name=name, family=r["family"], recipe="-", smoke="-", opponent="-",
                            params_state="-", action=("n/a: " + why) if why else "add-recipe"))
            continue
        recipe = ",".join("%s[%s]" % (k, "fit+infer" if lanes[k]["infer"] else "fit") for k in keys)
        in_smoke = all(rid in smoke[v] for k in keys for v, rids in lanes[k]["vendors"].items() for rid in rids)
        opp = ";".join("%s:%s" % (v[:2] if v == "nvidia" else v, "|".join(
            sorted({a for k in keys for a in lanes[k]["opponents"].get(v, [])})) or "none") for v in VENDORS)
        counts = collections.Counter()
        raced = 0
        for v in VENDORS:
            for k in keys:
                c = pairs.get(v, {}).get(k) or {}
                raced += c.get("raced", 0)
                for s in ("MATCHED", "unchecked", "refused"):
                    counts[s] += c.get(s, 0)
        has_opp = any(lanes[k]["opponents"].get(v) for k in keys for v in VENDORS)
        if not has_opp:
            refused = refused_by_name(keys)
            if refused:
                state = "refused-by-name"
                action = "covered (ours alone; opponent refused by name, listed in not_planned: %s)" % refused
            else:
                state, action = "no-opponent", "covered (ours alone: no peer library implements it on these vendors)"
        elif not raced:
            state, action = "not raced", "race"
        else:
            state = "MATCHED=%d unchecked=%d refused=%d" % (counts["MATCHED"], counts["unchecked"], counts["refused"])
            action = "covered" if counts["MATCHED"] and not (counts["unchecked"] or counts["refused"]) else "re-race"
        if not in_smoke:
            action += "; smoke-missing"
        out.append(dict(name=name, family=r["family"], recipe=recipe, smoke="yes" if in_smoke else "NO",
                        opponent=opp, params_state=state, action=action))
    return out, unresolved


COLUMNS = ("name", "family", "recipe", "smoke", "opponent", "params_state", "action")


def to_tsv(rows):
    lines = ["\t".join(COLUMNS)]
    lines += ["\t".join(str(r[c]) for c in COLUMNS) for r in rows]
    return "\n".join(lines) + "\n"


def summary(rows):
    c = collections.Counter(r["action"].split(":")[0].split(";")[0].split(" (")[0] for r in rows)
    return dict(names=len(rows), actions=dict(sorted(c.items())),
                smoke_missing=sum(r["smoke"] == "NO" for r in rows))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", type=Path)
    p.add_argument("--inventory-tsv", help="a saved public_runtime_inventory.py --out TSV (faster than the walk)")
    p.add_argument("--summary", action="store_true")
    args = p.parse_args(argv)
    rows, unresolved = coverage(REPO, args.inventory_tsv)
    if args.out:
        args.out.write_text(to_tsv(rows))
    s = summary(rows)
    s["unresolved_lane_tokens"] = unresolved
    print(json.dumps(s, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
