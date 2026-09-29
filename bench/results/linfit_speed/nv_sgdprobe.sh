#!/bin/bash
# lane/linfit-speed: where one SGD row's time goes (synthetic n x d, 3 epochs):
# d sweep, then the board setting with one piece switched off at a time.
set -uo pipefail
T=${T:-/root/mojolearn-linfit-speed}
cd $T
export MOJOLEARN_NUMERIC_MODE=identical
PY=".pixi/envs/default/bin/python"
O=bench/results/linfit_speed/sgdprobe-$(date -u +%Y%m%dT%H%M%SZ); mkdir -p $O
N=${N:-300000}
r() { local tag=$1; shift; $PY tools/linfit_speed.py --lanes sgd-reg --datasets s --max-iter 3 --out $O/$tag.json "$@" 2>/dev/null | python3 -c "
import sys, json
for l in sys.stdin:
    if l.startswith('{'):
        d = json.loads(l); print('%-28s d=%-4d us/row=%.3f' % ('$tag', d['shape'][1], d['fit_s'] / (d['shape'][0] * d['n_iter']) * 1e6))"; }
for d in 1 2 11 32 33 64 96 128 220 256; do r d$d --synthetic $N,$d; done
r base11 --synthetic $N,11
r nopen11 --synthetic $N,11 --set penalty=None
r noint11 --synthetic $N,11 --set fit_intercept=False
r noshuf11 --synthetic $N,11 --set shuffle=False
r hinge11 --lanes sgd-clf --synthetic $N,11
r base220 --synthetic $N,220
r nopen220 --synthetic $N,220 --set penalty=None
