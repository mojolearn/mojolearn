#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/transformer-admission/repo
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_rope_admission_probe.mojo -o /root/transformer-admission/rope_probe
