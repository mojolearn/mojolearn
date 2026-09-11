# Lane linear-cluster-istella: the flip verdict, by hand from the race JSONs.
# ENGINEERING_RULES.md section 9: geometric mean of after/before over the two
# datasets below 1, and quality not worse on EITHER. tools/flip_verdict.py
# reads FSPEED logs, which this harness does not write.
import json
import math
import os
import statistics
import sys

out = sys.argv[1] if len(sys.argv) > 1 else "/root/lane/out"
QUAL = {"ols": ("r2", "higher"), "pca": ("explained_variance_ratio_sum", "higher"),
        "kmeans": ("inertia", "lower"), "dbscan": ("n_clusters", "same"),
        "hdbscan": ("n_clusters", "same")}
rows = {}
for lane in ("ols", "kmeans", "pca", "dbscan", "hdbscan"):
    for ds in ("taxi", "istella"):
        p = os.path.join(out, "%s-%s.json" % (lane, ds))
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        arms = {}
        for arm, a in r["arms"].items():
            ms = a.get("ms") or []
            if a.get("status") == "ok" and ms:
                arms[arm] = {"median": statistics.median(ms), "min": min(ms), "max": max(ms),
                             "digests": sorted(set(a["digests"])),
                             "quality": r.get("quality", {}).get(arm, {})}
            else:
                arms[arm] = {"status": a.get("status"), "error": str(a.get("error"))[:200]}
        rows[(lane, ds)] = arms
        line = ["%s %s" % (lane, ds)]
        for arm in sorted(arms):
            a = arms[arm]
            if "median" in a:
                line.append("%s=%.1f" % (arm, a["median"]))
        print(" ".join(line), flush=True)

print()
for lane in ("ols", "kmeans", "pca"):
    ratios = []
    for ds in ("taxi", "istella"):
        a = rows.get((lane, ds), {})
        if "ours" in a and "ours-base" in a and "median" in a["ours"] and "median" in a["ours-base"]:
            ratio = a["ours"]["median"] / a["ours-base"]["median"]
            ratios.append((ds, ratio))
            same = a["ours"]["digests"] == a["ours-base"]["digests"]
            key, direction = QUAL[lane]
            q_a = a["ours"]["quality"].get(key)
            q_b = a["ours-base"]["quality"].get(key)
            cu = a.get("cuml-gpu", {})
            print("%-7s %-8s after/before=%.4f after=%.1f before=%.1f cuml=%s ours/cuml=%s "
                  "digests_equal=%s quality after=%s before=%s"
                  % (lane, ds, ratio, a["ours"]["median"], a["ours-base"]["median"],
                     ("%.2f" % cu["median"]) if "median" in cu else "-",
                     ("%.2f" % (a["ours"]["median"] / cu["median"])) if "median" in cu else "-",
                     same, q_a, q_b))
    if len(ratios) == 2:
        g = math.exp(sum(math.log(r) for _, r in ratios) / 2)
        print("%-7s GEOMEAN over taxi and istella = %.4f -> %s"
              % (lane, g, "FLIP (below 1)" if g < 1.0 else "NO FLIP"))
    else:
        print("%-7s only %d dataset(s): no verdict" % (lane, len(ratios)))
    print()

for lane in ("dbscan", "hdbscan"):
    for ds in ("taxi", "istella"):
        a = rows.get((lane, ds), {})
        for arm in sorted(a):
            e = a[arm]
            if "median" in e:
                print("%s %s %s median_ms=%.1f quality=%s"
                      % (lane, ds, arm, e["median"], json.dumps(e["quality"], sort_keys=True)))
            else:
                print("%s %s %s %s" % (lane, ds, arm, e))
