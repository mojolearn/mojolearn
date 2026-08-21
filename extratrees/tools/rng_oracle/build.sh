#!/bin/sh
# Build the RNG oracle and regenerate pcg_reference.txt IN THIS DIRECTORY.
#
# No CUDA toolkit is required: main.cpp copies RAFT's PCGenerator and cuML's
# fnv1a32 with the device decorations defined away, and the arithmetic is
# plain C++.
#
# -ffp-contract=off is deliberate and load-bearing. RAFT's uniform-float
# expression is `(res * (end - start)) + start`, which clang is otherwise free
# to contract into a single FMA; the Mojo side is written to NOT contract, so
# the reference must not either or the two disagree by one rounding of the
# product. See DEVIATION 142 in ../../mojo_only/pcg_rng.mojo.
set -eu
here=$(cd "$(dirname "$0")" && pwd)
cd "$here"
/usr/bin/clang++ -std=c++17 -O2 -ffp-contract=off -Wall -o rng_oracle main.cpp
./rng_oracle
wc -l pcg_reference.txt
