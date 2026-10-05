# SPDX-License-Identifier: Apache-2.0
"""Raw typed transport and safe lifecycle gate; authorized GPU boxes only.

Every bank transfers 257 raw sentinels twice, changing their order on reuse.
Float sentinels include negative zero, subnormal and quiet NaN payloads.
No arithmetic kernel uses the optional FP16/BFloat16/FP64 storage types.
"""
from std.memory import bitcast
from core.identical_callpath import IdenticalCallSession
from experiments.identical_callpath.minmax import ResidentIdenticalMinMax


def check_f32(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_f32(257)
    var sentinels: List[Scalar[DType.uint32]] = [0x80000000, 0x1, 0x7fc01234]
    var values = List[Scalar[DType.float32]](length=257, fill=0)
    var output = List[Scalar[DType.float32]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.float32](sentinels[(i + turn) % 3])
        session.stage_f32(slot, values)
        session.begin()
        session.upload_f32(slot)
        session.readback_f32(slot)
        session.finish()
        session.collect_f32(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint32](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: f32")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_f64(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_f64(257)
    var sentinels: List[Scalar[DType.uint64]] = [0x8000000000000000, 0x1, 0x7ff8123456789abc]
    var values = List[Scalar[DType.float64]](length=257, fill=0)
    var output = List[Scalar[DType.float64]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.float64](sentinels[(i + turn) % 3])
        session.stage_f64(slot, values)
        session.begin()
        session.upload_f64(slot)
        session.readback_f64(slot)
        session.finish()
        session.collect_f64(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint64](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: f64")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_f16(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_f16(257)
    var sentinels: List[Scalar[DType.uint16]] = [0x8000, 0x1, 0x7e55]
    var values = List[Scalar[DType.float16]](length=257, fill=0)
    var output = List[Scalar[DType.float16]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.float16](sentinels[(i + turn) % 3])
        session.stage_f16(slot, values)
        session.begin()
        session.upload_f16(slot)
        session.readback_f16(slot)
        session.finish()
        session.collect_f16(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint16](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: f16")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_bf16(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_bf16(257)
    var sentinels: List[Scalar[DType.uint16]] = [0x8000, 0x1, 0x7fc5]
    var values = List[Scalar[DType.bfloat16]](length=257, fill=0)
    var output = List[Scalar[DType.bfloat16]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.bfloat16](sentinels[(i + turn) % 3])
        session.stage_bf16(slot, values)
        session.begin()
        session.upload_bf16(slot)
        session.readback_bf16(slot)
        session.finish()
        session.collect_bf16(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint16](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: bf16")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_i8(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_i8(257)
    var sentinels: List[Scalar[DType.uint8]] = [0x80, 0xff, 0x1]
    var values = List[Scalar[DType.int8]](length=257, fill=0)
    var output = List[Scalar[DType.int8]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.int8](sentinels[(i + turn) % 3])
        session.stage_i8(slot, values)
        session.begin()
        session.upload_i8(slot)
        session.readback_i8(slot)
        session.finish()
        session.collect_i8(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint8](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: i8")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_u8(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_u8(257)
    var sentinels: List[Scalar[DType.uint8]] = [0x80, 0xff, 0x1]
    var values = List[Scalar[DType.uint8]](length=257, fill=0)
    var output = List[Scalar[DType.uint8]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.uint8](sentinels[(i + turn) % 3])
        session.stage_u8(slot, values)
        session.begin()
        session.upload_u8(slot)
        session.readback_u8(slot)
        session.finish()
        session.collect_u8(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint8](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: u8")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_i16(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_i16(257)
    var sentinels: List[Scalar[DType.uint16]] = [0x8000, 0xffff, 0x1]
    var values = List[Scalar[DType.int16]](length=257, fill=0)
    var output = List[Scalar[DType.int16]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.int16](sentinels[(i + turn) % 3])
        session.stage_i16(slot, values)
        session.begin()
        session.upload_i16(slot)
        session.readback_i16(slot)
        session.finish()
        session.collect_i16(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint16](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: i16")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_u16(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_u16(257)
    var sentinels: List[Scalar[DType.uint16]] = [0x8000, 0xffff, 0x1]
    var values = List[Scalar[DType.uint16]](length=257, fill=0)
    var output = List[Scalar[DType.uint16]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.uint16](sentinels[(i + turn) % 3])
        session.stage_u16(slot, values)
        session.begin()
        session.upload_u16(slot)
        session.readback_u16(slot)
        session.finish()
        session.collect_u16(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint16](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: u16")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_i32(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_i32(257)
    var sentinels: List[Scalar[DType.uint32]] = [0x80000000, 0xffffffff, 0x1]
    var values = List[Scalar[DType.int32]](length=257, fill=0)
    var output = List[Scalar[DType.int32]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.int32](sentinels[(i + turn) % 3])
        session.stage_i32(slot, values)
        session.begin()
        session.upload_i32(slot)
        session.readback_i32(slot)
        session.finish()
        session.collect_i32(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint32](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: i32")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_u32(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_u32(257)
    var sentinels: List[Scalar[DType.uint32]] = [0x80000000, 0xffffffff, 0x1]
    var values = List[Scalar[DType.uint32]](length=257, fill=0)
    var output = List[Scalar[DType.uint32]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.uint32](sentinels[(i + turn) % 3])
        session.stage_u32(slot, values)
        session.begin()
        session.upload_u32(slot)
        session.readback_u32(slot)
        session.finish()
        session.collect_u32(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint32](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: u32")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_i64(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_i64(257)
    var sentinels: List[Scalar[DType.uint64]] = [0x8000000000000000, 0xffffffffffffffff, 0x1]
    var values = List[Scalar[DType.int64]](length=257, fill=0)
    var output = List[Scalar[DType.int64]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.int64](sentinels[(i + turn) % 3])
        session.stage_i64(slot, values)
        session.begin()
        session.upload_i64(slot)
        session.readback_i64(slot)
        session.finish()
        session.collect_i64(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint64](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: i64")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def check_u64(mut session: IdenticalCallSession, mut digest: UInt64) raises:
    var slot = session.reserve_u64(257)
    var sentinels: List[Scalar[DType.uint64]] = [0x8000000000000000, 0xffffffffffffffff, 0x1]
    var values = List[Scalar[DType.uint64]](length=257, fill=0)
    var output = List[Scalar[DType.uint64]](length=257, fill=0)
    for turn in range(2):
        for i in range(257):
            values[i] = bitcast[DType.uint64](sentinels[(i + turn) % 3])
        session.stage_u64(slot, values)
        session.begin()
        session.upload_u64(slot)
        session.readback_u64(slot)
        session.finish()
        session.collect_u64(slot, output)
        for i in range(257):
            var bits = bitcast[DType.uint64](output[i])
            if bits != sentinels[(i + turn) % 3]:
                raise Error("typed raw transfer failed: u64")
            digest = (digest ^ UInt64(bits)) * UInt64(1099511628211)


def lifecycle() raises -> Int:
    var checks = 0
    var session = IdenticalCallSession()
    var slot = session.reserve_u32(1)
    var value: List[UInt32] = [17]
    var wrong = List[UInt32]()
    var refused = False
    try:
        session.stage_u32(slot, wrong)
    except:
        refused = True
    if not refused:
        raise Error("wrong staging shape accepted")
    checks += 1
    session.stage_u32(slot, value)
    session.begin()
    refused = False
    try:
        session.begin()
    except:
        refused = True
    if not refused:
        raise Error("overlapping batch accepted")
    checks += 1
    session.upload_u32(slot)
    session.readback_u32(slot)
    session.finish()
    refused = False
    try:
        session.collect_u32(slot, wrong)
    except:
        refused = True
    if not refused:
        raise Error("wrong result shape accepted")
    checks += 1
    var zero_slot = session.reserve_u32(0)
    session.stage_u32(zero_slot, wrong)
    session.begin()
    session.upload_u32(zero_slot)
    session.readback_u32(zero_slot)
    session.finish()
    session.collect_u32(zero_slot, wrong)
    checks += 1
    session.begin()
    session.readback_u32(slot)
    refused = False
    try:
        session.readback_u32(slot)
    except:
        refused = True
    session.abort()
    if not refused:
        raise Error("duplicate readback accepted")
    checks += 1
    refused = False
    try:
        session.begin()
    except:
        refused = True
    if not refused:
        raise Error("poisoned session reused")
    checks += 1
    var invalid = IdenticalCallSession()
    refused = False
    try:
        _ = invalid.reserve_u32(-1)
    except:
        refused = True
    if not refused:
        raise Error("negative extent accepted")
    checks += 1
    refused = False
    try:
        invalid.begin()
    except:
        refused = True
    if not refused:
        raise Error("failed reservation did not poison session")
    checks += 1
    var model: List[Float32] = [1.0]
    var offset: List[Float32] = [0.0]
    var scaler = ResidentIdenticalMinMax(1, 1, 1, model, offset)
    var inputs = List[List[Float32]]()
    var outputs = List[List[Float32]]()
    for i in range(2):
        inputs.append(model.copy())
        outputs.append(model.copy())
    refused = False
    try:
        scaler.transform_batch_into(inputs, outputs, 0, 0, 0.0, 1.0)
    except:
        refused = True
    if not refused or outputs[0][0] != 1.0 or outputs[1][0] != 1.0:
        raise Error("capacity refusal changed caller results")
    checks += 1
    return checks


def main() raises:
    var session = IdenticalCallSession()
    var digest = UInt64(14695981039346656037)
    check_f32(session, digest)
    check_f64(session, digest)
    check_f16(session, digest)
    check_bf16(session, digest)
    check_i8(session, digest)
    check_u8(session, digest)
    check_i16(session, digest)
    check_u16(session, digest)
    check_i32(session, digest)
    check_u32(session, digest)
    check_i64(session, digest)
    check_u64(session, digest)
    var checks = lifecycle()
    if checks != 9:
        raise Error("incomplete lifecycle coverage")
    print("CALLPATH_TRANSPORT status=PASS banks=12 roundtrips=24 cells_per_transfer=257 lifecycle_checks=", checks, " digest=", digest)
