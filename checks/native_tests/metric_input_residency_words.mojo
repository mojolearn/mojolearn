# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""THE RESIDENCY WORDS OF A METRIC PROGRAM'S INPUTS (lane vdf-xm-wire2; bindings/arena_prog_run.mojo,
bindings/metric_store_refusal.mojo), in a process with no Python.

1. the temporaries mask: bit i names input i; inputs past `METRIC_TEMP_BITS` are never temporaries (kept as host
   inputs, as before); an empty mask names none;
2. the sources word: `metric_source_word(input, argument)` round-trips through `metric_input_source` for every
   input below `METRIC_SOURCE_INPUTS` and argument 0..14; out-of-range inputs or arguments leave the word 0 and the
   input unmarked (-1); words for different inputs combine without touching each other;
3. the refusal words of a GPU metric image whose x_prep store cannot be reached name the family, the tier, the
   vendor and the table's own cause, as a RuntimeError.

FAIL-PROOF EDITS (each must make the named case fail): drop the `i < METRIC_TEMP_BITS` bound (case 0, input 64
shifts past the word); return `UInt64(argument) << ...` without the +1 (case 1, argument 0 reads unmarked); drop
the cause from the refusal (case 2).
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import check_cases, require
from bindings.arena_prog_run import (
    METRIC_TEMP_BITS, METRIC_SOURCE_INPUTS, metric_input_temp, metric_source_word, metric_input_source,
)
from bindings.metric_store_refusal import metric_store_refusal

comptime CASES = 3


def run_case(which: Int) raises:
    if which == 0:
        var temps: UInt64 = 0
        for i in range(METRIC_TEMP_BITS + 2):  # small-loop(fixture)
            require(not metric_input_temp(temps, i), "an empty mask names no temporary")
        temps = (UInt64(1) << 1) | (UInt64(1) << 3) | (UInt64(1) << 63)
        require(metric_input_temp(temps, 1) and metric_input_temp(temps, 3) and metric_input_temp(temps, 63),
                "the marked inputs are temporaries")
        require(not metric_input_temp(temps, 0) and not metric_input_temp(temps, 2), "unmarked inputs are kept")
        require(not metric_input_temp(~UInt64(0), METRIC_TEMP_BITS), "an input past the mask is never a temporary")
    elif which == 1:
        for i in range(METRIC_SOURCE_INPUTS):  # small-loop(fixture)
            for a in range(15):  # small-loop(fixture)
                var w = metric_source_word(i, a)
                require(metric_input_source(w, i) == a, "input " + String(i) + " names argument " + String(a))
                if i > 0:
                    require(metric_input_source(w, i - 1) == -1, "a neighbour stays unmarked")
        require(metric_source_word(METRIC_SOURCE_INPUTS, 0) == 0, "an input past the word stays unmarked")
        require(metric_source_word(0, 15) == 0 and metric_source_word(0, -1) == 0, "an argument past 14 stays unmarked")
        require(metric_input_source(UInt64(0), METRIC_SOURCE_INPUTS + 3) == -1, "inputs past the word are unmarked")
        var reg = metric_source_word(0, 0) | metric_source_word(1, 1) | metric_source_word(2, 2)
        require(metric_input_source(reg, 0) == 0 and metric_input_source(reg, 1) == 1
                and metric_input_source(reg, 2) == 2 and metric_input_source(reg, 3) == -1,
                "the regression sources word: y_true, y_pred, weights")
    else:
        var got = metric_store_refusal("x_metrics", 1, 2, "native entry table: provider is not published: X")
        require(got == "RuntimeError: x_metrics: the identical GPU image (vendor 2) keeps device-resident inputs in "
                + "the x_prep store, but the x_prep image of the same tier and vendor is not connected or does not "
                + "match (native entry table: provider is not published: X); install mojolearn binaries of one build",
                "refusal words: " + got)
        require(metric_store_refusal("metrics", 0, 3, "c").startswith("RuntimeError: metrics: the fast GPU image"),
                "the fast tier is named")


def test_metric_input_residency_words() raises:
    check_cases("metric_input_residency_words", CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
