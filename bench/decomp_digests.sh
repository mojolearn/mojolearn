#!/bin/sh
# bench/decomp_digests.sh -- the decomp lane's part hashes on this box's GPU
# and on its CPU (lane/decomp-apple): the identity harness fits every
# x-decomp lane once per column and prints one line per (lane, fixture,
# part, column) with its hash, so a before and an after commit compare
# line for line (the speed changes must leave every hash where it was).
# env: LANES (comma list; default every x-decomp lane), COLUMNS (default
# "gpu cpu"), OUT (default /tmp/decomp_digests).
set -u
cd "$(dirname "$0")/.."
LANES=${LANES:-x-decomp-als,x-decomp-dict-learning,x-decomp-factor-analysis,x-decomp-fastica,x-decomp-grp,x-decomp-ipca,x-decomp-lda,x-decomp-lstsq-rsvd,x-decomp-lu,x-decomp-manifold,x-decomp-nmf,x-decomp-pca-randomized,x-decomp-pls,x-decomp-robust-cov,x-decomp-sparse-pca,x-decomp-spectral-rbf,x-decomp-srp,x-decomp-umap-options}
OUT=${OUT:-/tmp/decomp_digests}
mkdir -p "$OUT"
case "$(uname)" in Darwin) gpu=metal ;; *) if command -v rocm-smi >/dev/null 2>&1; then gpu=hip; else gpu=cuda; fi ;; esac
for col in ${COLUMNS:-gpu cpu}; do
    if [ "$col" = cpu ]; then flags="--require-cpu --require-backend cpu"; else flags="--require-backend $gpu"; fi
    # shellcheck disable=SC2086
    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python pixi run -e default python -u tools/identity_break.py \
        --lanes "$LANES" --repeats 1 --fail-on-refused --json "$OUT/$col.json" $flags > "$OUT/$col.log" 2>&1
    echo "column $col exit $?"
    tail -3 "$OUT/$col.log"
done
pixi run -e default python - "$OUT" <<'PYEOF'
import json, sys, hashlib, os
out = sys.argv[1]
for col in ("gpu", "cpu"):
    p = os.path.join(out, f"{col}.json")
    if not os.path.exists(p):
        continue
    cells = json.load(open(p))["cells"]
    rows = []
    for k in sorted(cells):
        c = cells[k]
        parts = (c.get("parts") or [{}])[0] or {}
        for name in sorted(parts):
            rows.append((f"{k} part {name}", str(parts[name])))
        for key in sorted(c):
            v = c[key]
            if key != "parts" and isinstance(v, list) and v and all(isinstance(x, str) for x in v):
                rows.append((f"{k} {key}", ",".join(v)))
    h = hashlib.sha256("\n".join(f"{a} {b}" for a, b in rows).encode()).hexdigest()[:16]
    print(f"DIGESTS {col}: {len(rows)} hashes, combined {h}")
    for a, b in rows:
        print(f"  {col} {a} {b}")
PYEOF
