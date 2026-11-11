# SPDX-License-Identifier: Apache-2.0
"""Concurrent native context publication, staging and device-pool ownership.

Run in a fresh process before any x_decomp/x_neighbors operation. Each of
five phases uses one independent owner per allocated CPU worker. Worker
regions record actual overlap; serial-only scheduling cannot claim a pass.
The bounded arrays identify every owner and position; this is not timing or
an identity campaign. Existing device operations and copy routes are used.
"""
from std.atomic import Atomic
from std.os import getenv
from std.sys.info import num_logical_cores
from max.gpu.host import DeviceBuffer
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.kernel_matrix import TARGET_COLUMN, COLUMN_CPU, COLUMN_APPLE
from core.host_parallel import host_parallelize
from core.device_zero import enqueue_fill
from core.device_pool import pool_take, pool_give, _DevPool, DEVICE_POOL_LAYOUT
from core.shared_slot import SharedSlot
from x_decomp.cells import F32Ptr, OP_SCALE
from x_decomp.device import xd_ctx, DevExec, XD_STAGE_MIN
from x_decomp.resident import pool_alloc, pool_free, pool_live_count, _ptr, dev_upload_run, dev_download_run, dev_ew_run
from x_neighbors.device_ops import xn_ctx, op_unary
from x_neighbors.items import U_IDENTITY

comptime CONCURRENCY_POOL = "MojolearnNativeConcurrencyDevicePool20261110V2"
comptime CONCURRENCY_GUARD = Float32(-77777)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def worker_count() raises -> Int:
    # The orchestrator supplies the actual affinity/cgroup allocation on a
    # rented host. The local default uses every logical CPU, without a cap.
    var configured = String(getenv("MOJOLEARN_NATIVE_TEST_WORKERS"))
    var count = num_logical_cores() if configured == "" else Int(atol(configured))
    require(count >= 2, "native concurrency coverage requires at least two allocated workers")
    return count


def parallel_phase[Func: def(Int) raises -> None](ref body: Func, workers: Int,
        name: String, require_overlap: Bool = True) raises:
    var failures = List[Int](length=workers, fill=0)
    var depths = List[Int](length=workers, fill=0)
    var active = List[Int32](length=1, fill=0)
    var fp = failures.unsafe_ptr()
    var dp = depths.unsafe_ptr()
    var ap = active.unsafe_ptr()
    def task(unit: Int) {imm body, imm fp, imm dp, imm ap, imm name}:
        var depth = Atomic.fetch_add(ap, Int32(1)) + 1
        dp.unsafe_store(unit, Int(depth))
        try:
            body(unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL native_device_concurrency", name, "owner", unit, String(error))
        finally:
            _ = Atomic.fetch_add(ap, Int32(-1))
    host_parallelize(task, workers)
    var failed = 0
    var peak = 0
    for unit in range(workers):  # Joined diagnostic reduction, not device work.
        failed += failures[unit]
        peak = max(peak, depths[unit])
    require(active[0] == 0, name + " concurrent regions did not join")
    _ = active^
    _ = failures^
    _ = depths^
    print("native_device_concurrency", name, "workers", workers, "overlapping_call_regions", peak)
    require(failed == 0, name + " failed " + String(failed) + " owners")
    require(not require_overlap or peak >= 2, name + " observed no concurrent callers")


def word(owner: Int, index: Int) -> Float32:
    return Float32((owner + 1) * 512 + index % 127)


def source_values(owner: Int, count: Int) -> List[Float32]:
    var result = List[Float32](unsafe_uninit_length=count)
    # All CPU workers already own independent arrays; each initializes its
    # own cells, with no serial driver over owners or second nested pool.
    for index in range(count):
        result[index] = word(owner, index)
    return result^


def verify(dst: List[Float32], owner: Int, count: Int, scale: Float32) raises:
    require(dst[0] == CONCURRENCY_GUARD and dst[count + 1] == CONCURRENCY_GUARD,
            "concurrent operation crossed caller destination extent")
    var first_bad = -1
    for index in range(count):  # Independent owners execute across all workers.
        if dst[index + 1] != scale * word(owner, index) and first_bad < 0:
            first_bad = index
    require(first_bad < 0, "concurrent operation mixed owner " + String(owner)
            + " or source position " + String(first_bad))


def decomp_owner(unit: Int, count: Int) raises:
    var source = source_values(unit, count)
    var scalar: List[Float32] = [0]
    var dst = List[Float32](length=count + 2, fill=CONCURRENCY_GUARD)
    var xp = F32Ptr(unsafe_from_address=Int(source.unsafe_ptr()))
    var sp = F32Ptr(unsafe_from_address=Int(scalar.unsafe_ptr()))
    var dp = F32Ptr(unsafe_from_address=Int(dst.unsafe_ptr()) + 4)
    DevExec.ew(OP_SCALE, xp, sp, 1, 3, sp, 1, 3, dp, count, 1, 2)
    verify(dst, unit, count, 2)
    var first_bad = -1
    for index in range(count):
        if source[index] != word(unit, index) and first_bad < 0:
            first_bad = index
    require(first_bad < 0, "concurrent decomposition modified input position " + String(first_bad))
    _ = source^
    _ = scalar^
    _ = dst^


def first_contexts(workers: Int, neighbors: Bool) raises:
    var keys = List[Int](length=workers, fill=0)
    var kp = keys.unsafe_ptr()
    def owner(unit: Int) raises {imm kp, imm neighbors}:
        if neighbors:
            var ctx = xn_ctx()
            kp.unsafe_store(unit, Int(ctx._handle.value()))
            var source = source_values(unit, 19)
            var dst = List[Float32](length=21, fill=CONCURRENCY_GUARD)
            op_unary(Int(source.unsafe_ptr()), Int(dst.unsafe_ptr()) + 4, 19, U_IDENTITY, 2, 0)
            verify(dst, unit, 19, 2)
            for index in range(19):
                require(source[index] == word(unit, index), "concurrent neighbor modified its input")
            _ = source^
            _ = dst^
            _ = ctx^
        else:
            var ctx = xd_ctx()
            kp.unsafe_store(unit, Int(ctx._handle.value()))
            decomp_owner(unit, 17)
            _ = ctx^
    parallel_phase(owner, workers, "neighbors first context" if neighbors else "decomp first context")
    require(keys[0] != 0, "native context handle must be published")
    for key in keys:
        require(key == keys[0], "concurrent first users received different context owners")


def staged_downloads(workers: Int) raises:
    def owner(unit: Int) raises:
        # Derived from the unchanged staging threshold, with neighboring
        # extents. Every Apple download must use the shared pinned stage.
        decomp_owner(unit, XD_STAGE_MIN + 1 + 17 * (unit % 3))
    parallel_phase(owner, workers, "distinct staged download owners")
    comptime if TARGET_COLUMN == COLUMN_APPLE:
        print("native_device_concurrency Apple pinned staging route exercised")
    else:
        print("native_device_concurrency raw download route exercised; Apple staging remains untested on this target")


def distinct(values: List[Int], name: String) raises:
    var count = len(values)
    var vp = values.unsafe_ptr()
    def owner(unit: Int) raises {imm vp, imm count, imm name}:
        require(vp[unit] >= 0, name + " missing owner")
        for other in range(unit):
            require(vp[unit] != vp[other], name + " simultaneously held owners alias")
    parallel_phase(owner, count, name, False)
    _ = values


def resident_pool(workers: Int, wave: Int) raises:
    var baseline = pool_live_count()
    var ids = List[Int](length=workers, fill=-1)
    var pointers = List[Int](length=workers, fill=-1)
    var ip = ids.unsafe_ptr()
    var pp = pointers.unsafe_ptr()
    try:
        def allocate(unit: Int) raises {imm ip, imm pp}:
            var count = 257 + 256 * (unit % 3)
            var id = pool_alloc(count)
            ip.unsafe_store(unit, id)
            pp.unsafe_store(unit, Int(_ptr(id, count)))
        parallel_phase(allocate, workers, "resident append/pop", False)
        distinct(ids, "resident live IDs")
        distinct(pointers, "resident live buffers")
        require(pool_live_count() == baseline + workers, "resident live count after allocation")
        def owner(unit: Int) raises {imm ip, imm pp, imm wave, imm workers}:
            var count = 257 + 256 * (unit % 3)
            var id = ip[unit]
            var source = source_values(unit + wave * workers, count)
            var dst = List[Float32](length=count + 2, fill=CONCURRENCY_GUARD)
            var scratch = -1
            try:
                _ = dev_upload_run(id, F32Ptr(unsafe_from_address=Int(source.unsafe_ptr())), count)
                source[0] = -999  # Upload completed; resident data owns its bytes.
                scratch = pool_alloc(1025 + unit)
                _ = pool_live_count()  # Read metadata while other owners append/pop.
                require(Int(_ptr(id, count)) == pp[unit], "resident live pointer changed during metadata growth")
                _ = dev_ew_run(id, id, id, scratch, OP_SCALE, count, 1, 1, 3, 1, 3, 2.0)
                _ = dev_download_run(scratch, F32Ptr(unsafe_from_address=Int(dst.unsafe_ptr()) + 4), count)
                verify(dst, unit + wave * workers, count, 2)
                _ = dev_download_run(id, F32Ptr(unsafe_from_address=Int(dst.unsafe_ptr()) + 4), count)
                verify(dst, unit + wave * workers, count, 1)
            finally:
                if scratch >= 0:
                    pool_free(scratch)
                _ = source^
                _ = dst^
        parallel_phase(owner, workers, "resident growth/read/download")
    finally:
        def release(unit: Int) raises {imm ip}:
            if ip[unit] >= 0:
                pool_free(ip[unit])
                ip.unsafe_store(unit, -1)
        parallel_phase(release, workers, "resident return/reuse", False)
        _ = pointers^
        _ = ids^
    require(pool_live_count() == baseline, "resident fixture leaked a live ID")


def isolated_pool(workers: Int) raises:
    var slots = List[Optional[DeviceBuffer[DType.float32]]](length=workers,
        fill=Optional[DeviceBuffer[DType.float32]]())
    var addresses = List[Int](length=workers, fill=-1)
    var op = slots.unsafe_ptr()
    var ap = addresses.unsafe_ptr()
    # Each dependent wave checks a different ownership transition: initial
    # allocation, same-size idle reuse, and changed exact-size membership.
    for wave in range(3):
        try:
            def take(unit: Int) raises {imm op, imm ap, imm wave}:
                var count = 17 + 256 * ((unit + wave) % 3)
                var ctx = xd_ctx()
                var buf = pool_take[CONCURRENCY_POOL](ctx, count)
                require(len(buf) == count, "isolated pool returned wrong exact extent")
                ap.unsafe_store(unit, Int(buf.unsafe_ptr()))
                op[unit] = Optional[DeviceBuffer[DType.float32]](buf^)
                _ = ctx^
            parallel_phase(take, workers, "isolated exact-size take", False)
            distinct(addresses, "isolated held buffers")
            def owner(unit: Int) raises {imm op, imm wave}:
                var ctx = xd_ctx()
                var buf = op[unit].value().copy()
                var count = len(buf)
                var value = Float32(1000 * (wave + 1) + unit)
                var dst = List[Float32](length=count + 2, fill=CONCURRENCY_GUARD)
                enqueue_fill[DType.float32](ctx, buf, value)
                ctx.enqueue_copy(dst_ptr=F32Ptr(unsafe_from_address=Int(dst.unsafe_ptr()) + 4), src_buf=buf)
                ctx.synchronize()
                require(dst[0] == CONCURRENCY_GUARD and dst[count + 1] == CONCURRENCY_GUARD,
                        "isolated pool download crossed destination extent")
                for index in range(count):
                    require(dst[index + 1] == value, "isolated pool mixed simultaneously held owners")
                _ = dst^
                _ = buf^
                _ = ctx^
            parallel_phase(owner, workers, "isolated live fill/download")
        finally:
            def give(unit: Int) raises {imm op}:
                if op[unit]:
                    var buf = op[unit].value().copy()
                    op[unit] = Optional[DeviceBuffer[DType.float32]]()
                    pool_give[CONCURRENCY_POOL](buf^)
            parallel_phase(give, workers, "isolated concurrent give", False)
        # No concurrent readers/writers remain: inspect the isolated fixture
        # slot only, never an algorithm's pool or a borrowed growing List.
        comptime SLOT = SharedSlot[_DevPool, CONCURRENCY_POOL, DEVICE_POOL_LAYOUT]
        var pool = SLOT.get_or_create_ptr()
        var bytes = 0
        for index in range(len(pool[].bufs)):
            bytes += len(pool[].bufs[index]) * 4
        require(pool[].bytes == bytes, "isolated pool idle byte accounting lost an update")
        require(len(pool[].bufs) >= workers, "isolated pool lost returned owned buffers")
    _ = slots^
    _ = addresses^


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL and TARGET_COLUMN != COLUMN_CPU
    var workers = worker_count()
    print("native_device_concurrency fresh process required; allocated workers", workers)
    first_contexts(workers, False)
    first_contexts(workers, True)
    staged_downloads(workers)
    resident_pool(workers, 0)
    resident_pool(workers, 1)  # Same classes exercise reuse; fresh owner values.
    isolated_pool(workers)
    print("PASS native_device_concurrency: 5 native ownership/concurrency phases; workers", workers)
