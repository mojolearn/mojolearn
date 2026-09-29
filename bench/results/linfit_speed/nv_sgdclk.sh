#!/bin/bash
# lane/linfit-speed: the clock64-instrumented SGD copy (/root/mojolearn-sgdclk,
# built from the dev tree plus instr.py; NOT a lane source): cycles per row
# by segment, first epoch, synthetic n x d.
cd /root/mojolearn-sgdclk
export MOJOLEARN_NUMERIC_MODE=identical
for d in 11 32 220; do .pixi/envs/default/bin/python tools/linfit_speed.py --lanes sgd-reg --datasets s --synthetic 200000,$d --max-iter 1 --no-warm --out /tmp/clk$d.json 2>&1 | grep "SGDCLK\|fit_s"; done
