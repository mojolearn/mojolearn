#!/bin/bash
set -uo pipefail
cd ${T:-/root/mojolearn-linfit-speed}
pixi run mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . bench/results/linfit_speed/chain_bench.mojo 2>&1 | grep -v "warning\|__add__\|~~\|^\s*^\|declared here\|Imported from\|unsafe_offset\|^\s*$" | tail -30
