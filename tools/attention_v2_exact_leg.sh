#!/bin/sh
set -eu
mkdir -p /root/gemm_leg_out/attention_v2
OUT=/root/gemm_leg_out/attention_v2
pixi run python -m unittest tools.tests.test_attention_v2_oracle tools.tests.test_attention_v2_backward_oracle -v > "$OUT/python.log" 2>&1
for rep in 1 2; do
  pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/attention_v2_forward_check.mojo > "$OUT/forward_$rep.log" 2>&1
  pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/attention_v2_backward_check.mojo > "$OUT/backward_$rep.log" 2>&1
done
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/attention_v2_forward_bench.mojo > "$OUT/forward_bench.log" 2>&1
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/attention_v2_backward_bench.mojo > "$OUT/backward_bench.log" 2>&1
sha256sum "$OUT"/*.log > "$OUT/logs.sha256"
cat "$OUT"/forward_1.log "$OUT"/backward_1.log "$OUT"/forward_bench.log "$OUT"/backward_bench.log
