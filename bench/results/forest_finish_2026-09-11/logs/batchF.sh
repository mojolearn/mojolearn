#!/bin/sh
# Lane forest-finish, batch F: DEVIATION 2663's OTHER two cells.
#
# The batch width is set in extratrees/estimator.mojo::resolve, which the
# ExtraTrees REGRESSOR takes as well, so the switch reaches four (lane,
# dataset) cells and section 9 decides it over all four. Batch C times the two
# classification cells (max_features='sqrt': 4 and 14 columns). This one times
# the regression cells (max_features=1.0: taxireg 11 columns, istellareg ALL
# 220), which cost several times a classification fit.
#
# INERT BY DEFAULT. It waits for /root/trees_out/GO_BATCH_F, which the lane
# creates only if batch C's classification A/B puts a flip on the table. No
# flip on the table, no reason to spend the pod on these.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
while [ ! -f $OUT/GO_BATCH_F ]; do sleep 20; done
echo "batchF start $(date -u +%T)"

# taxireg is cheap (11 columns): two passes in rotated order, 3 rounds.
for set in ctl bw16k bw32k; do MOJOLEARN_SPEED_TAG=p1 $AB speed $set et taxireg 1000000 3 ours; done
for set in bw32k bw16k ctl; do MOJOLEARN_SPEED_TAG=p2 $AB speed $set et taxireg 1000000 3 ours; done
echo "F1_TAXIREG_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
: > $OUT/phase.F1_TAXIREG_DONE

# istellareg samples every one of the 220 columns per node: one pass, 2 rounds.
for set in ctl bw16k bw32k; do MOJOLEARN_SPEED_TAG=p1 $AB speed $set et istellareg 1000000 2 ours; done
echo "F2_ISTELLAREG_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
: > $OUT/phase.F2_ISTELLAREG_DONE

# The medians, printed here so the regression cells are read the same way the
# classification ones are (flip_verdict keys on the two dataset NAMES, and
# these two are a different pair).
python3 - "$OUT/speed" >> $OUT/verdicts.txt 2>&1 <<'PY'
import glob, os, re, statistics, sys
base = sys.argv[1]
rows = {}
for p in sorted(glob.glob(os.path.join(base, "*.et.*reg.*.ours.*.log"))):
    for line in open(p, errors="replace"):
        m = re.match(r"^FSPEED lane=et arm=ours shape=(\S+) round=\d+ ms=([\d.]+) hash=(\S+)", line)
        if m:
            st = os.path.basename(p).split(".")[0]
            key = (m.group(1).split("-")[0], st)
            rows.setdefault(key, {"ms": [], "h": set()})
            rows[key]["ms"].append(float(m.group(2)))
            rows[key]["h"].add(m.group(3))
print("=== DEVIATION 2663 regression cells (ours-only medians)")
for (ds, st), v in sorted(rows.items()):
    print("REGCELL dataset=%-11s set=%-6s median_ms=%9.1f n=%d hashes=%s"
          % (ds, st, statistics.median(sorted(v["ms"])), len(v["ms"]), ",".join(sorted(v["h"]))))
for ds in sorted({d for d, _ in rows}):
    ctl = rows.get((ds, "ctl"))
    if not ctl:
        continue
    for st in ("bw16k", "bw32k"):
        cur = rows.get((ds, st))
        if cur:
            r = statistics.median(sorted(cur["ms"])) / statistics.median(sorted(ctl["ms"]))
            same = "SAME BITS" if cur["h"] == ctl["h"] else "HASH MOVED"
            print("REGRATIO dataset=%-11s %s/ctl=%.3f %s" % (ds, st, r, same))
PY
echo "F3_MEDIANS_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
: > $OUT/phase.F3_MEDIANS_DONE
echo "batchF end $(date -u +%T)"
