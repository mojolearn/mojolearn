# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which of our arms an opponent arm is compared with.

smoke-fixes r13 (Andrew, 2026-10-14). A race normally has one own arm per numeric mode
(`ours`, and `ours-ab` / `ours-fast` for FAST), and every opponent is compared with it. Some
lanes race more than one explicit own configuration, each matched to the opponent library whose
algorithm it reproduces (OUR_PAIRINGS below): on trees/gbdt-lossguide
`ours` (min_split_gain None, no gain check, CatBoost Lossguide's growth) is paired with CatBoost
only and `ours-gain0` (min_split_gain 0.0, a strictly positive gain, XGBoost gamma 0) with XGBoost
only. Such an own cell carries `pairs_with`, the opponent arm-name globs it is compared with; an
opponent no own cell lists has no paired own cell and gets no ratio. A race whose
own cells carry no `pairs_with` keeps the old rule (the first own cell of the mode).

The board's ratios (bench_board.add_ratios), the clock ratios (board_clock_audit), the fairness
rows (race_fairness) and the parameter check (bench_board_params.check) all choose through here.
"""

import fnmatch

PAIRS_FIELD = "pairs_with"

#: lane -> {own arm: its pairing}. Every own arm of such a lane is listed (`ours` too), and each
#: `settings` key is passed to GradientBoosting explicitly and read back (a None included).
#: `opponents` are opponent arm-name globs (fnmatch).
#: The driver (bench/speed/forest_speed_arm.py) builds every listed arm, the board
#: (tools/bench_board.py our_arms) plans a cell for each, and every comparison pairs by this table.
#: LightGBM is in no list: the driver refuses every boosted LightGBM arm by name
#: (tools/speed_gbdt_arm.py lightgbm_arms), so it never has a cell to pair.
#: `prebinned` True: that own arm and its opponents take the same pre-binned input (decision 1
#: below; tools/speed_gbdt_arm.py prebinned_data), the other arms the raw rows.
#: Andrew 2026-10-14 (DECISIONS.md decision 1), the bin borders against XGBoost on gbdt-depthwise and
#: gbdt-lossguide: ours builds GreedyLogSum borders over a 200,000-row sample (ensemble.py:1076,
#: 1180, 1190; CatBoost's rule), XGBoost 3.2.0 hist a weighted quantile sketch over every row
#: (src/common/hist_util.cu); neither can be set to the other's cut points. So both arms of the
#: XGBoost pairing are given the same pre-binned matrix (every feature value replaced by its
#: GreedyLogSum bin id, once per dataset), where every distinct bin id is its own bin on both: ours
#: 254 borders over every row (border_build_max_samples 0), XGBoost max_bin 255. Every split
#: candidate is then the same on both arms. The CatBoost pairings keep the raw input.
OUR_PAIRINGS = {
    "gbdt-depthwise": {
        "ours": {"opponents": ("catboost-*",), "settings": {},
                 "why": "raw input, ours' and CatBoost's own GreedyLogSum borders"},
        "ours-prebinned": {"opponents": ("xgboost-*",), "settings": {"border_build_max_samples": 0},
                           "prebinned": True,
                           "why": "pre-binned input on both arms (GreedyLogSum bin ids): the same "
                                  "split candidates as XGBoost hist at max_bin 255"},
    },
    # Andrew 2026-10-14 (~/mojolearn-evidence/smoke-fixes/DECISIONS.md, decision 2): the split-gain
    # floor. XGBoost 3.2.0 gamma 0 splits only on a strictly positive gain
    # (src/tree/gpu_hist/expand_entry.cuh:34) and has no setting to drop the check; CatBoost 1.2.10
    # Lossguide splits the best leaf with no gain check at all
    # (catboost/cuda/methods/greedy_subsets_searcher/greedy_search_helper.cpp:297-309, 321-328) and
    # has no setting to add one. Ours can run either: min_split_gain 0.0 requires a positive gain
    # (python/mojolearn/ensemble.py:1059,1297), None is no gain check (ensemble.py:1187). So two
    # explicit own configurations race, each against the library it matches, never the other.
    "gbdt-lossguide": {
        # catboost-cpu is not listed: its CPU learner keeps a gain >= 1e-9 check
        # (greedy_tensor_search.cpp) and it is refused by name on this lane anyway (it has no
        # NewtonL2 score, tools/speed_gbdt_arm.py _catboost_cpu_refusal).
        "ours": {"opponents": ("catboost-gpu",), "settings": {"min_split_gain": None},
                 "why": "min_split_gain None: no gain check, CatBoost Lossguide's growth "
                        "(greedy_search_helper.cpp:297-309, 321-328)"},
        # decision 1 too: the XGBoost pairing takes the pre-binned input on both arms
        "ours-gain0": {"opponents": ("xgboost-*",),
                       "settings": {"min_split_gain": 0.0, "border_build_max_samples": 0},
                       "prebinned": True,
                       "why": "min_split_gain 0.0: a strictly positive gain, XGBoost gamma 0 "
                              "(src/tree/gpu_hist/expand_entry.cuh:34); pre-binned input on both arms"},
    },
}


def lane_pairing(lane):
    """{own arm: [opponent arm globs]} of a paired lane, or None."""
    table = OUR_PAIRINGS.get(lane)
    if not table:
        return None
    return {arm: list(p["opponents"]) for arm, p in table.items()}


def extra_own_arms(lane):
    """The own arms a paired lane races beside `ours`, in table order."""
    return [arm for arm in (OUR_PAIRINGS.get(lane) or {}) if arm != "ours"]


def pairs(cell, arm):
    """True when own `cell` is compared with opponent `arm`."""
    listed = cell.get(PAIRS_FIELD)
    return listed is None or any(fnmatch.fnmatch(str(arm), g) for g in listed)


def paired_ours(candidates, opponent):
    """The own cell (of `candidates`, already one mode) an `opponent` cell is compared with, or None."""
    candidates = [c for c in candidates if c is not None]
    tagged = [c for c in candidates if c.get(PAIRS_FIELD) is not None]
    if not tagged:
        return candidates[0] if candidates else None
    return next((c for c in tagged if pairs(c, opponent.get("arm"))), None)


def reference_for(arm, pairing, default="ours"):
    """The own arm an opponent `arm` is checked against, from a race's pairing {own arm: [arm globs]};
    `default` when the race has no pairing or no own arm lists it."""
    for own, globs in (pairing or {}).items():
        if any(fnmatch.fnmatch(str(arm), g) for g in (globs or ())):
            return own
    return default


def prebinned_arm(lane, arm):
    """True when `arm` (own or opponent) of `lane` races on the pre-binned input."""
    for own, p in (OUR_PAIRINGS.get(lane) or {}).items():
        if not p.get("prebinned"):
            continue
        if arm == own or any(fnmatch.fnmatch(str(arm), g) for g in p["opponents"]):
            return True
    return False
