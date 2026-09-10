#!/usr/bin/env bash
set -euo pipefail
cd /root/embedding-sort
out=/root/jobs/embedding-plan-sort
grep -q 'sabotage= SORT_NEGATIVE_CONTROL' "$out/negative.log"
grep -q 'PLAN_SORT edge metadata mismatch' "$out/negative.log"
printf '%s\n' 'PASS: production plan/geometry, edge cases, and device negative control (portable grep verification after image lacked rg)' > "$out/verdict.txt"
sha256sum tools/check_embedding_plan_sort.sh >> "$out/context.txt"
/root/.pixi/bin/pixi run --manifest-path /root/embedding-sort/pixi.toml mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . embedding/checks/embedding_sort_shipped_check.mojo -o "$out/shipped-check" > "$out/shipped-build.log" 2>&1
"$out/shipped-check" > "$out/shipped.log" 2>&1
MOJOLEARN_EMB_CHECK_CLAUSE_F=1 MOJOLEARN_EMB_DEVICE_REFUSAL_GAP_ACK=1 MOJOLEARN_IDENTITY_TRACE="$out/refusal-audit.card" "$out/embedding-check" > "$out/refusal-audit.log" 2>&1
