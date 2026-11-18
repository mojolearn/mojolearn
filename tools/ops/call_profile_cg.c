// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Andrew Hendel. Part of mojolearn
//
// tools/ops/call_profile_cg.c -- the callgrind window of tools/ops/call_profile_job.sh's `callgrind` pass (diagnostic
// tooling, never product). Built on the pod against valgrind's own header (`cc -O2 -shared -fPIC -I<valgrind include>
// -o callprof_cg.so call_profile_cg.c`) and loaded with ctypes by tools/bench_board_small_rows.py, which calls
// callprof_cg_open() at the profile window's open (call 2) and callprof_cg_close() at its close (call n). The race
// runs under `valgrind --tool=callgrind --instr-atstart=no --collect-atstart=no`: nothing is instrumented before the
// window (start-up, data load, warmup and the first call run at valgrind's core speed), and the window's dump holds
// exactly the instructions of calls 2..n. Outside valgrind every request is a no-op.
#include <valgrind/callgrind.h>

int callprof_cg_running(void) { return RUNNING_ON_VALGRIND; }

void callprof_cg_open(void) {
    CALLGRIND_START_INSTRUMENTATION;
    CALLGRIND_ZERO_STATS;
    CALLGRIND_TOGGLE_COLLECT;
}

void callprof_cg_close(void) {
    CALLGRIND_TOGGLE_COLLECT;
    CALLGRIND_DUMP_STATS_AT("callprof-window");
    CALLGRIND_STOP_INSTRUMENTATION;
}
