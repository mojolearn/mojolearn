#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/transformer-admission/repo
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 bash bindings/build_transformer.sh
