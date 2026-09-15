"""Compare our gbdt-pair-logit values with the CatBoost CPU reference's.

A correctness reference, not a bitwise one: the reference runs on its CPU in
float64 with its own leaf arithmetic. This prints the exact differences and
decides nothing; no threshold is applied and nothing is tuned.

Our PairLogit `loss_curve_[k]` is the learn loss after tree k + 1,
`-sum w * (diff - log(1 + e^diff)) / sum w` over the pairs, which is the
reference's PairLogit metric (`-sum w log(e^a_w / (e^a_w + e^a_l)) / sum w`,
metric.cpp:2542-2552) at iteration k.

    python compare_pair_logit_reference.py ours.json catboost.json
"""
import json
import math
import sys


def main():
    ours = json.load(open(sys.argv[1]))
    ref = json.load(open(sys.argv[2]))
    print(f"CatBoost {ref.get('catboost')} CPU reference vs our Metal fit")
    for kind, o in ours["fixtures"].items():
        r = ref["fixtures"].get(kind)
        if r is None:
            print(f"{kind}: no reference entry")
            continue
        per_it = r["per_iteration"]
        qkey = next(k for k in per_it if k.startswith("PairLogit"))
        nkey = next(k for k in per_it if k.startswith("NDCG"))
        dkey = next(k for k in per_it if k.startswith("DCG"))
        ref_sq = list(per_it[qkey])
        our_curve = o["loss_curve"]
        n = min(len(ref_sq), len(our_curve))
        diffs = [our_curve[k] - ref_sq[k] for k in range(n)]
        worst = max(range(n), key=lambda k: abs(diffs[k]))
        print(f"\n{kind}: {o['n_rows']} rows, {o['n_groups']} queries (reference {r['n_groups']})")
        print(f"  learn loss, ours vs their PairLogit metric, {n} iterations:")
        print(f"    first  ours {our_curve[0]:.9g}  reference {ref_sq[0]:.9g}  diff {diffs[0]:+.3e}")
        print(f"    last   ours {our_curve[n - 1]:.9g}  reference {ref_sq[n - 1]:.9g}  diff {diffs[n - 1]:+.3e}")
        rel = abs(diffs[worst]) / max(abs(ref_sq[worst]), 1e-300)
        print(f"    largest |diff| at iteration {worst}: {diffs[worst]:+.3e} (relative {rel:.3e})")
        fn = per_it[nkey][-1]
        print(f"  final NDCG       ours {o['final_ndcg']:.9g}  reference {fn:.9g}  diff {o['final_ndcg'] - fn:+.3e}")
        fd = per_it[dkey][-1]
        print(f"  final DCG        ours {o['final_dcg']:.9g}  reference {fd:.9g}  diff {o['final_dcg'] - fd:+.3e}")
        ro = r.get("final_raw_first8", [])
        oo = o.get("final_raw_first8", [])
        if ro and oo:
            ds = [a - b for a, b in zip(oo, ro)]
            md = max(abs(d) for d in ds)
            print(f"  first 8 raw predictions, largest |diff| {md:.3e}, "
                  f"diff spread (max - min) {max(ds) - min(ds):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
