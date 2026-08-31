#!/bin/sh
# Build the RNG oracle and regenerate pcg_reference.txt IN THIS DIRECTORY.
#
# No CUDA toolkit is required: main.cpp copies RAFT's PCGenerator and cuML's
# fnv1a32 with the device decorations defined away, and the arithmetic is
# plain C++.
#
# -ffp-contract=off is deliberate and load-bearing. RAFT's uniform-float
# expression is `(res * (end - start)) + start`, which clang at its default
# `-ffp-contract=on` DOES contract into a single FMA -- measured: dropping this
# flag changes the `uf` lines of pcg_reference.txt. The Mojo side is written
# not to contract, so the reference must not either. See DEVIATION 142 in
# ../../checks/pcg_rng.mojo.
#
# The compiled binary is a build artifact and is deleted again; only
# pcg_reference.txt is meant to be kept.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
bin=$(mktemp -t rng_oracle)
trap 'rm -f "$bin"' EXIT INT TERM
/usr/bin/clang++ -std=c++17 -O2 -ffp-contract=off -Wall -o "$bin" main.cpp
"$bin"
wc -l pcg_reference.txt
