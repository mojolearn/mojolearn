#!/bin/bash
# lane/sabotage-sweep, shard e: the two single-device lanes the matrix reads as
# "none". Neither has a host family, because NEITHER HAS A BUILD: both run an
# independent Python implementation (mojolearn/_bpe_trainer.py and
# model_selection._default_folds). Their negative control is therefore an
# IMPLEMENTATION env switch in the library, not the harness and not a compiled
# define, and this pair is what it looks like when it is watched failing.
set -u
O="$LEG_OUT"
LANES='bpe-trainer,cross-val-folds'
st() { n=$1; shift; "$@"; rc=$?; printf '%s\t%s\n' "$n" "$rc" >> "$O/step_exit_codes.txt"; return 0; }
echo "shard e nproc=$(nproc) $(uname -m) $(python3 --version 2>&1)" > "$O/box.txt"
sha256sum python/mojolearn/host/*.so > "$O/so_sha256.txt" 2>&1
st prod python3 tools/identity_break.py --lanes "$LANES" --fixtures base,ties --repeats 2 --json "$O/cpu-x86.json" > "$O/prod.log" 2>&1
st sab env MOJOLEARN_BPE_TRAINER_SABOTAGE=1 MOJOLEARN_FOLD_ORDER_SABOTAGE=1 MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
   python3 tools/identity_break.py --lanes "$LANES" --fixtures base,ties --repeats 2 --json "$O/cpu-x86.sabotage.json" > "$O/sab.log" 2>&1
# The arms must differ in the library, not in the probe: with the switches OFF
# a second clean run must reproduce the first byte for byte.
st prod2 python3 tools/identity_break.py --lanes "$LANES" --fixtures base,ties --repeats 2 --json "$O/cpu-x86.replay.json" > "$O/prod2.log" 2>&1
st diff python3 tools/identity_break.py --diff "$O/cpu-x86.json" "$O/cpu-x86.sabotage.json" > "$O/diff.clean-vs-sabotage.txt" 2>&1
st diff2 python3 tools/identity_break.py --diff "$O/cpu-x86.json" "$O/cpu-x86.replay.json" > "$O/diff.clean-vs-replay.txt" 2>&1
