# SPDX-License-Identifier: Apache-2.0
"""The safetensors reader's half-precision widening, moved from
`test_models_loader.py::test_safetensors_reader_dtypes_round_trip_and_widen_exactly`
(lane tests-to-mojo-models, 2026-11-16).

`SafetensorsFile.read` and `Checkpoint.read` widen an F16 tensor, and a BF16
tensor read with `bf16="widen"`, through
`core/causal_checkpoint_impl.mojo::checkpoint_widen_halves`
(bindings/causal_checkpoint_door.mojo `_safetensors_read`), on every install:
it is host code even beside a GPU. The Python test compared twelve halves
with Python's `struct` value oracle and twelve bfloat16 words with `<< 16`;
this check runs every one of the 65,536 bit patterns through the entry, so
those twenty-four are among them:
  * F16: the value oracle of `byte_lm_models_loader_f16.half_matches`
    (finite values and infinities by their float32 bits, a NaN by its sign
    and NaN-ness), the oracle the Python test used;
  * BF16: the float32 bits are the word shifted left by 16, exactly.
The 128 chunks (64 per dtype) are independent and run across host cores.
The file reading, dtype admission, returned Array types, the `bits` path and
the refusals stay in the Python test.
"""
from std.memory import bitcast
from core.host_parallel import host_parallelize
from core.causal_checkpoint_impl import checkpoint_widen_halves
from checks.native_tests.byte_lm_models_loader_f16 import half_matches
from checks.native_tests.runner import Check, Column, Report


comptime CKW_HALVES = 65536
comptime CKW_CHUNKS = 64
comptime CKW_CHUNK = CKW_HALVES // CKW_CHUNKS


def ckw_first_failure(chunk: Int, hp: Int, wp: Int) -> Int:
    """Widen chunk `chunk` (F16 below CKW_CHUNKS, BF16 above) into its own
    slice and return its first failing bit pattern, or -1."""
    var bf16 = chunk >= CKW_CHUNKS
    var first = (chunk - CKW_CHUNKS if bf16 else chunk) * CKW_CHUNK
    var source = MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=hp)
    var dest = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=wp)
    var offset = chunk * CKW_CHUNK
    checkpoint_widen_halves(source + offset, dest + offset, CKW_CHUNK, String("BF16") if bf16 else String("F16"))
    for i in range(CKW_CHUNK):  # driver-loop: over one chunk's bit patterns
        var h = first + i
        var got = dest[offset + i]
        if bf16:
            if bitcast[DType.uint32](got) != UInt32(h) << 16:
                return h
        elif not half_matches(h, got):
            return h
    return -1


def _run(mut report: Report) raises:
    # Source: every pattern twice, the F16 half then the BF16 half.
    var halves = List[UInt16](length=2 * CKW_HALVES, fill=UInt16(0))
    for h in range(CKW_HALVES):  # driver-loop: over the fixture's 65,536 bit patterns
        halves[h] = UInt16(h)
        halves[CKW_HALVES + h] = UInt16(h)
    var widened = List[Float32](length=2 * CKW_HALVES, fill=Float32(0))
    var failures = List[Int](length=2 * CKW_CHUNKS, fill=-1)
    var fp = failures.unsafe_ptr()
    var hp = Int(halves.unsafe_ptr())
    var wp = Int(widened.unsafe_ptr())
    def task(chunk: Int) {imm fp, imm hp, imm wp}:
        fp.unsafe_store(chunk, ckw_first_failure(chunk, hp, wp))
    host_parallelize(task, 2 * CKW_CHUNKS)
    _ = halves  # read through its address by every task above
    _ = widened
    var cases = report.cases(2 * CKW_CHUNKS)
    for chunk in range(2 * CKW_CHUNKS):  # driver-loop: over the chunk verdicts
        if failures[chunk] >= 0:
            cases.fail_message(chunk, (String("BF16") if chunk >= CKW_CHUNKS else String("F16"))
                               + " chunk " + String(chunk) + " mismatch at bit pattern " + String(failures[chunk]))
    _ = failures


# The widening is host code on a CPU and on a GPU install alike, so both
# columns run the same host-parallel case set.
def run_cases(column: Column, mut report: Report):
    try:
        _run(report)
    except error:
        report.abort(error)


def register() -> Check:
    return Check("models_checkpoint_widen", cpu=True, gpu=True, serial="", run=run_cases)
