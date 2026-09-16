import json, sys
from math import lgamma, exp, log

def logC(n, k):
    return lgamma(n+1) - lgamma(k+1) - lgamma(n-k+1)

def fisher_right(a, b, c, d):
    """One-sided Fisher exact: P(control moves >= a | margins), 2x2
       [[a moved_control, b clean_control],[c moved_repair, d clean_repair]]."""
    n = a+b+c+d
    r1, c1 = a+b, a+c
    lo, hi = max(0, c1-b), min(r1, c1)
    tot = 0.0
    num = 0.0
    for x in range(lo, hi+1):
        p = exp(logC(r1, x) + logC(n-r1, c1-x) - logC(n, c1))
        tot += p
        if x >= a:
            num += p
    return num/tot

def binom_zero(n, p):
    return (1-p)**n

d = json.load(open(sys.argv[1]))
t = d["totals"]
print("commit", d["commit"], "token", d["token"])
for k in sorted(t):
    v = t[k]
    print("  %-26s moved %4d / %5d comparisons  (%5d fits, %3d rounds, distinct-key sum %d, errors %d)"
          % (k, v["moved"], v["comparisons"], v["fits"], v["rounds"], v["distinct_keys"], len(v["errors"])))
for cfg in ("cols16", "cols10"):
    s = t.get(cfg+"/stock_prerepair"); r = t.get(cfg+"/claimfix")
    if not s or not r: continue
    a, b = s["moved"], s["comparisons"]-s["moved"]
    c, dd = r["moved"], r["comparisons"]-r["moved"]
    print("\n%s  stock %d/%d = %.3f%%   repaired %d/%d = %.3f%%"
          % (cfg, a, s["comparisons"], 100*a/s["comparisons"], c, r["comparisons"], 100*c/r["comparisons"]))
    print("   Fisher exact one-sided p = %.3g" % fisher_right(a, b, c, dd))
    if s["comparisons"]:
        rate = a/s["comparisons"]
        if rate > 0:
            print("   P(%d moves in %d at the control's own rate %.4f) = %.3g"
                  % (c, r["comparisons"], rate, binom_zero(r["comparisons"], rate)))
