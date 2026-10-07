#!/bin/bash
# lq CMD body for MOJOLEARN_GBDT_NS_PREDICT_BITVEC (lane/trees-predict-ideas, 2026-10-07).
# Runs in the branch tree after the bindings are built (switch OFF). Set
# NSBV_ON_PKG to an ON-build package dir to skip the rebuild below. Copies the
# OFF package, rebuilds the gbdt binding with the switch ON (compile check),
# copies that package, then runs bench/speed/gbdt_ns_bitvec_check.py once per
# arm in separate processes and prints one NSBV-CMP line per case
# (SAME / DIFFER digests, OFF and ON predict ms).
set -u
O=${NSBV_OUT:-$PWD/nsbv-out}; rm -rf "$O"; mkdir -p "$O"
export MOJOLEARN_NUMERIC_MODE=identical
rm -rf "$O/pkg-off" && mkdir -p "$O/pkg-off" && cp -r python/mojolearn "$O/pkg-off/"
if [ -n "${NSBV_ON_PKG:-}" ]; then   # orchestrator-built ON package dir (holds mojolearn/)
  mkdir -p "$O/pkg-on" && cp -r "$NSBV_ON_PKG/mojolearn" "$O/pkg-on/"
else
  MOJOLEARN_EXTRA_DEFINES="-D MOJOLEARN_GBDT_NS_PREDICT_BITVEC" bash bindings/build_gbdt.sh > "$O/build-on.log" 2>&1
  rc=$?; echo "NSBV build-on rc=$rc"
  [ $rc -ne 0 ] && { grep -m 3 -B 2 -A 8 'error' "$O/build-on.log" | cut -c1-240; exit $rc; }
  mkdir -p "$O/pkg-on" && cp -r python/mojolearn "$O/pkg-on/"
fi
for arm in off on; do
  PYTHONPATH="$O/pkg-$arm" pixi run python bench/speed/gbdt_ns_bitvec_check.py --arm $arm > "$O/run-$arm.log" 2>&1
  echo "NSBV run-$arm rc=$?"
  grep -E '^NSBV ' "$O/run-$arm.log" | cut -c1-260
done
pixi run python - "$O" <<'EOF'
import re, sys
o = sys.argv[1]
r = {}
for arm in ("off", "on"):
    for line in open(f"{o}/run-{arm}.log"):
        m = re.match(r"NSBV arm=\w+ case=(\S+) predict_ms=(\S+) digest=(\w+)", line)
        if m:
            r.setdefault(m.group(1), {})[arm] = (m.group(2), m.group(3))
for case, v in r.items():
    if "off" in v and "on" in v:
        st = "SAME" if v["off"][1] == v["on"][1] else "DIFFER"
        print(f"NSBV-CMP case={case} {st} off_ms={v['off'][0]} on_ms={v['on'][0]} digest_off={v['off'][1]} digest_on={v['on'][1]}")
    else:
        print(f"NSBV-CMP case={case} INCOMPLETE {v}")
EOF
