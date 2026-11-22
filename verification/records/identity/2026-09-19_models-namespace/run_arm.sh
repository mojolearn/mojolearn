#!/bin/bash
# usage: run_arm.sh <host_dir> <out.json>; one core, nice 19.
#
# <host_dir> holds one binding per family. The clean arm points at a full
# prebuilt set; a sabotage arm points at a directory that links every clean
# binding EXCEPT the family under test and holds a fresh build of that one:
#
#   MOJOLEARN_HOST_OUTDIR=<fresh dir> MOJOLEARN_BUILD_JOBS=1 \
#     MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1 -D <own arm>=1" \
#     nice -n 19 sh bindings/build_<family>_host.sh
#
# PYTHONPATH names a copy of python/mojolearn with the GPU .so removed, which
# is what makes _backend select the CPU-only set on a Mac that has them: the
# lanes pass CausalLM's public default device="auto", so a box with a GPU
# binding present would run Metal and this record is a CPU column.
set -u
cd "$(git rev-parse --show-toplevel)"
HD=$1; OUT=$2
env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_HOST_DIR="$HD" \
    MOJOLEARN_HOST_ALLOW_SABOTAGE=1 OMP_NUM_THREADS=1 MOJOLEARN_NUM_THREADS=1 \
    PYTHONPATH="${MOJOLEARN_CPU_ONLY_TREE:?a python/mojolearn copy with no *.so}" \
  nice -n 19 python3 tools/identity_break.py \
    --lanes hf-checkpoint,hf-tokenizer,hf-causal-lm --repeats 2 --require-cpu --json "$OUT"
