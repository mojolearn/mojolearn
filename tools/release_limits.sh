# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# THE ONE SOURCE FOR BUILD-PATH TIME LIMITS (seconds), 2026-09-25.
#
# WHY. 0.8.19 rebuilt every binding cold and ran into THREE different
# hardcoded 2400 s bounds, one in each of tools/release061_remote_build.sh,
# tools/do_release061_leg.sh and tools/gemm_remote_leg.sh; raising one left
# the others to kill the build (exit 124). Every build-path bound now lives
# here, and tools/test_no_hidden_build_bounds.py fails on a numeric bound in a
# build-path script that neither reads this file nor is allow-listed there
# with a reason.
#
# Sourced by the shell scripts (`. "$ROOT/tools/release_limits.sh"`) and read
# by Python through tools/release_limits.py, which parses THIS file, so there
# is exactly one place a value is written. Plain NAME=integer lines only.
# A caller's environment variable of the same name overrides nothing here;
# each script keeps its own MOJOLEARN_* override knob and falls back to these.

# The largest MOJOLEARN_RELEASE_BUILD_SECONDS tools/release061_remote_build.sh
# accepts (it refuses anything outside MIN..MAX). Raised from 2400 for 0.8.19,
# when a release reusing nothing rebuilt all ~240 bindings cold.
RELEASE_BUILD_SECONDS_MIN=120
RELEASE_BUILD_SECONDS_MAX=6000

# The build bound a DigitalOcean AMD leg (tools/do_release061_leg.sh) hands
# the remote build: the lease less the fetch reserve, capped here
# (MOJOLEARN_RELEASE_BUILD_CAP overrides). Raised from 2400 for 0.8.19.
DO_RELEASE_BUILD_CAP=5700

# The DigitalOcean leg's dead-man (DEADMAN_SECONDS overrides): the droplet is
# destroyed at this age whatever the build is doing, so it bounds the build
# above DO_RELEASE_BUILD_CAP plus the fetch reserve. Raised from 3600 for 0.8.19.
DO_RELEASE_DEADMAN_SECONDS=7200

# The build bound the RunPod NVIDIA release body in tools/gemm_remote_leg.sh
# hands the remote build: the remaining work window, capped here. Raised from
# 2400 for 0.8.19.
RUNPOD_RELEASE_BUILD_CAP=6000

# The build bound of the Hot Aisle AMD leg (tools/hotaisle_release_leg.sh).
# Its default lease is 60 minutes, so the lease, not this cap, is what binds
# it today. Raised from 2400 to RELEASE_BUILD_SECONDS_MAX with the leg's
# bound following the lease (lane/build-parallelism, 2026-09-25).
HOTAISLE_RELEASE_BUILD_CAP=6000

# The CPU build route (tools/release_linux_cpu_box.sh, on the rented CPU box):
# the MOJOLEARN_RELEASE_BUILD_SECONDS it hands each of the three sets. NOT
# raised with the others for 0.8.19 and still 2400, the bound the GPU legs ran
# past on a cold rebuild; moved here unchanged so raising it is one line.
CPU_BOX_RELEASE_BUILD_SECONDS=2400

# tools/cross_compile_check.py: the per-binding compile bound on the Mac.
# IDENTICAL svm cross-compiles for sm_89 in about 3 minutes at -j 1; a binding
# that takes ten is a hang to look at before a rental pays for it.
CROSS_COMPILE_SECONDS=600
