# SPDX-License-Identifier: Apache-2.0
"""Typed rules and placements of the parallel forecasting owner (RUN): the
span, step, index and mode refusals of the four prediction drivers and the
two fits, and the block copies that cut a shard's series out of the batch
and place a shard's result at its series offset.
`bindings/parallel_forecast_native_door.mojo` unpacks the Python values into
these arguments, runs the shards on the process pool and packs the result;
no Python object reaches this file. The copies are one transfer per shard
block; the shards themselves run one per device in the workers."""
from core.mode_answer import MODE_FAMILY_TSA
from bindings.parallel_plan_run import plan_mode_admitted
from bindings.parallel_shard_common_run import shard_block_copy

comptime FORECAST_F32 = 4
comptime FORECAST_I32 = 4


def forecast_arima_mode_admit(present: Bool, exact: Bool, word: String) raises:
    """The parallel ARIMA drivers' identical-only admission of the model's
    own word: absent, `identical`, or `fast` where the identical set answers
    it for TSA (core/mode_answer.mojo), so a model fitted under an answered
    `fast` shards as its own fit ran."""
    if not present:
        return
    if not exact or not plan_mode_admitted(word, MODE_FAMILY_TSA):
        raise Error("ValueError: parallel ARIMA requires IDENTICAL numeric mode")


def forecast_process_mode_admit(word: String, refusal: String) raises:
    """The Holt-Winters drivers read the process default mode."""
    if word != "identical":
        raise Error("ValueError: " + refusal)


def forecast_exog_admit(columns: Int, given: Bool) raises:
    if columns != 0 or given:
        raise Error("NotImplementedError: parallel ARIMA prediction does not shard exogenous regressors")


def forecast_span_admit(start: Int, end: Int, refusal: String) raises:
    if start < 0 or end <= start:
        raise Error("ValueError: " + refusal)


def forecast_positive_admit(value: Int, name: String) raises:
    if value < 1:
        raise Error("ValueError: " + name + " must be positive")


def forecast_out_of_sample_admit(start: Int, observations: Int) raises:
    if start < observations:
        raise Error("NotImplementedError: in-sample Holt-Winters prediction uses host arithmetic")


def forecast_index_admit(index: Int, batch: Int) raises:
    if not (0 <= index and index < batch):
        raise Error("IndexError: series index outside fitted batch")


def forecast_batch_admit(batch: Int) raises:
    if batch < 1:
        raise Error("ValueError: at least one series is required")


def forecast_length_admit(returned: Int, expected: Int) raises:
    if returned != expected:
        raise Error("ValueError: Holt-Winters worker returned a different series length")


def forecast_envelope_admit(fields: Int) raises:
    if fields != 3:
        raise Error("parallel forecasting: invalid worker request envelope")


def forecast_rows_copy(dst: Int, src: Int, lo: Int, hi: Int, width: Int):
    """Rows [lo, hi) of a C-ordered (rows, width) float32 block at `src` to
    `dst`: every shard starts at its own series."""
    shard_block_copy(dst, src + lo * width * FORECAST_F32, (hi - lo) * width * FORECAST_F32)


def forecast_rows_place(dst: Int, src: Int, lo: Int, hi: Int, width: Int):
    """A shard's (hi - lo, width) float32 block at `src` to rows [lo, hi) of
    the batch block at `dst`."""
    shard_block_copy(dst + lo * width * FORECAST_F32, src, (hi - lo) * width * FORECAST_F32)


def forecast_words_place(dst: Int, src: Int, row: Int, batch: Int, lo: Int, count: Int):
    """`count` 4-byte words (float32 statistics or int32 flags) of a shard at
    `src` to series [lo, lo + count) of row `row` of a (rows, batch) block."""
    shard_block_copy(dst + (row * batch + lo) * FORECAST_F32, src, count * FORECAST_F32)
