#!/usr/bin/env bash
set -euo pipefail
cd /root/embedding-sort
out=/root/jobs/embedding-plan-sort
mkdir -p "$out"
{
  date -u
  uname -a
  nvidia-smi --query-gpu=name,uuid,driver_version,memory.total --format=csv,noheader
  /root/.pixi/bin/pixi run --manifest-path /root/embedding-sort/pixi.toml mojo --version
  printf '%s\n' 'Executable source: 0250d63b (production registry316f4c5f)'
  sha256sum embedding/checks/embedding_identical.mojo embedding/checks/embedding_sort.mojo embedding/checks/embedding_check.mojo embedding/checks/embedding_sort_check.mojo embedding/checks/embedding_sort_shipped_check.mojo tools/check_embedding_plan_sort.sh
} > "$out/context.txt" 2>&1
MOJOLEARN_IDENTITY_TRACE="$out/embedding.identical.card" /root/.pixi/bin/pixi run --manifest-path /root/embedding-sort/pixi.toml bash tools/check_embedding_plan_sort.sh "$out" > "$out/driver.log" 2>&1
/root/.pixi/bin/pixi run --manifest-path /root/embedding-sort/pixi.toml mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . embedding/checks/embedding_sort_shipped_check.mojo -o "$out/shipped-check" > "$out/shipped-build.log" 2>&1
"$out/shipped-check" > "$out/shipped.log" 2>&1
MOJOLEARN_EMB_CHECK_CLAUSE_F=1 MOJOLEARN_EMB_DEVICE_REFUSAL_GAP_ACK=1 MOJOLEARN_IDENTITY_TRACE="$out/refusal-audit.card" "$out/embedding-check" > "$out/refusal-audit.log" 2>&1
