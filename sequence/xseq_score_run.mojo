# SPDX-License-Identifier: Apache-2.0
"""RUN half of the sequence-expansion regressors' `score`
(`bindings/xseq_score_native.mojo` is the UNPACK/PACK boundary; lane
doorsplit-cnn-xseq, 2026-11-11).

Typed values only: the metrics image's R^2 table key, the shapes and
addresses of the admitted float32 arrays, the sample count. Here: the
vector shape, finiteness, length and sample-weight refusals (one min/max
scan serves both weight refusals), the compiled-mode check, and the one R^2
entry call. The boundary admits the arrays, releases the GIL around
`xseq_run_r2` (where the door released it before the split) and boxes the
Float64 it returns.
"""
from core.native_descriptor_call import native_descriptor_call
from core.metrics_entry_table import MetricsEntryKey, metrics_r2_entry
from core.xseq_marshal import xseq_min_max_f32
from core.host_finite_scan import host_all_finite_f32
from sequence.xseq_neural_run import XseqRunF32P, xseq_run_shape_text


def xseq_run_vector_rows(shape: List[Int], name: String) raises -> Int:
    """`_vector_shape`'s refusals of a 1-D (or (n, 1)) target; its rows."""
    var rank = len(shape)
    var column = rank == 2 and shape[1] == 1
    if rank != 1 and not column:
        raise Error("ValueError: mojolearn metrics: " + name + " must be 1-D, got shape " + xseq_run_shape_text(shape))
    var n = shape[0]
    if n == 0:
        raise Error("ValueError: mojolearn metrics: " + name + " is empty")
    return n


def xseq_run_vector_finite(address: Int, n: Int, name: String) raises:
    """The finiteness refusal of a metrics vector."""
    if not host_all_finite_f32(XseqRunF32P(unsafe_from_address=address), n):
        raise Error("ValueError: mojolearn metrics: " + name + " contains NaN or infinity; refused "
                    + "here (metrics/README.md HAND-OFF ask 3) rather than letting it "
                    + "reach a kernel whose answer for it is a washed sentinel")


def xseq_run_lengths_check(n: Int, m: Int) raises:
    if m != n:
        raise Error("ValueError: mojolearn metrics: y_true has " + String(n) + " entries and y_pred has "
                    + String(m) + "; they must be the same length")


def xseq_run_weights_check(shape: List[Int], n: Int, address: Int) raises:
    """`_sample_weight_f32(sample_weight, n, "r2_score")`'s refusals on the
    admitted float32 weights."""
    if len(shape) != 1:
        raise Error("ValueError: mojolearn r2_score: sample_weight must be 1-D, got shape " + xseq_run_shape_text(shape))
    if shape[0] != n:
        raise Error("ValueError: mojolearn r2_score: sample_weight has " + String(shape[0])
                    + " entries for " + String(n) + " samples")
    var p = XseqRunF32P(unsafe_from_address=address)
    if not host_all_finite_f32(p, n):
        raise Error("ValueError: mojolearn r2_score: sample_weight must have finite nonnegative entries")
    # one min/max scan serves both refusals (it was run twice)
    var bounds = xseq_min_max_f32(p, n)
    if bounds[0] < Float32(0.0):
        raise Error("ValueError: mojolearn r2_score: sample_weight must have finite nonnegative entries")
    if not bounds[1] > Float32(0.0):
        raise Error("ValueError: mojolearn r2_score: sample_weight must have positive total weight")


def xseq_run_metrics_mode(requested: String, actual: Int) raises:
    """The compiled-mode word the bridge read at import against the mode
    asked for."""
    var name = String("fast") if actual == 0 else String("identical")
    if name != requested:
        raise Error("RuntimeError: mojolearn metrics: requested " + requested + ", binary reports "
                    + name + "; rebuild it")


def xseq_run_r2(key: MetricsEntryKey, operation: Int, var addresses: List[Int], n: Int) raises -> Float64:
    """One call of the metrics image's R^2 entry (core/metrics_entry_table.mojo
    layouts); the value lands in a native word. The caller keeps the
    addressed arrays alive and has released the GIL."""
    var entry = metrics_r2_entry(key)
    var value = List[Float64](length=1, fill=0.0)
    addresses.append(Int(value.unsafe_ptr()))
    var words: List[Int] = [n]
    _ = native_descriptor_call(entry, operation, addresses, words, List[Float64]())
    var result = value[0]
    _ = value^
    return result
