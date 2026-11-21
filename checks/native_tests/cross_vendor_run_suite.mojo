# SPDX-License-Identifier: Apache-2.0
"""Native state contracts of the typed cross-vendor run
(bindings/cross_vendor_native_run.mojo): the `losses` header field, the
batch feed and the rate table, the coordinator's admission of a group and its
refusal words, and the recorded float words and digests.

Host bookkeeping only: no socket is opened (the group's descriptors are -1,
so a refusal's notices fail and are dropped, as for a closed peer), no
trainer session and no device. Four independent fixtures, run across the
host's cores.
"""
from core.host_parallel import host_parallelize
from core.sha256 import sha256_hex_addr
from core.transport_header import TransportHeader, transport_header_int
from bindings.cross_vendor_native_run import (
    CROSS_PROTOCOL, CROSS_SCHEME_V1, CROSS_SCHEME_SLICED, CROSS_DEFECT_NONE, CROSS_DEFECT_NAME,
    CrossFeed, CrossRates, CrossCoordinatorState,
    cross_header_put_floats, cross_header_floats, cross_feed_address, cross_rate,
    cross_coordinator_clear, cross_coordinator_admit, cross_state_hash, cross_float_word, cross_scheme_name,
)

comptime CROSS_VENDOR_RUN_FIXTURES = 4


def require(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


def refused(text: String, wanted: String) raises:
    require(text.find(wanted) >= 0, "expected refusal '" + wanted + "', got '" + text + "'")


def group(mut c: CrossCoordinatorState, workers: Int, shards: Int, chained: Bool):
    c.workers = workers
    c.shards = shards
    c.chained = chained
    c.max_bytes = 1024
    c.n_total = 4
    c.completed = 0
    cross_coordinator_clear(c)


def peer(mut c: CrossCoordinatorState, name: String, ids: List[Int], chained: Bool):
    c.fds.append(-1)
    c.timeouts.append(Float64(-1))
    c.names.append(name)
    c.protocols.append(String(CROSS_PROTOCOL))
    c.defects.append(CROSS_DEFECT_NONE)
    c.same.append(7)
    c.peer_chained.append(chained)
    for id in ids:  # metadata: one peer's declared shard ids
        c.ids.append(id)
    c.starts.append(len(c.ids))


def admit_refusal(mut c: CrossCoordinatorState) -> String:
    try:
        cross_coordinator_admit(c)
    except problem:
        return String(problem)
    return String("")


def fixture(unit: Int) raises:
    if unit == 0:
        # The losses field reads back as the same float64 bits.
        var values: List[Float64] = [2.0, 0.1, 5.545177459716797, 1e-30, 123456789012345680.0, 0.0]
        var header = TransportHeader()
        header.put_int("step", 3)
        cross_header_put_floats(header, "losses", values)
        var text = header.encode(0)
        var address = Int(text.unsafe_ptr())
        var back = cross_header_floats(address, len(text), "losses")
        require(len(back) == len(values), "losses count")
        for i in range(len(values)):  # metadata: fixture values
            require(back[i] == values[i], "losses value " + String(i))
        require(transport_header_int(address, len(text), "step") == 3, "step beside losses")
        var missing = String("")
        try:
            _ = cross_header_floats(address, len(text), "step")
        except problem:
            missing = String(problem)
        refused(missing, "is not a list of numbers")
        _ = len(text)
        var bad = TransportHeader()
        var huge = Float64(1e308)
        var infinite: List[Float64] = [1.0, huge * 10.0]
        var words = String("")
        try:
            cross_header_put_floats(bad, "losses", infinite)
        except problem:
            words = String(problem)
        refused(words, "JSON numbers must be finite")
    elif unit == 1:
        var feed = CrossFeed()
        feed.address = 4096
        feed.steps = 3
        feed.shards = 2
        feed.batch = 5
        feed.width = 7
        feed.start = 10
        require(cross_feed_address(feed, 10, 0) == 4096, "feed first block")
        require(cross_feed_address(feed, 11, 1) == 4096 + (1 * 2 + 1) * 5 * 7 * 4, "feed block address")
        var probes: List[Int] = [9, 13]
        for probe in probes:  # metadata: steps outside the table
            var words = String("")
            try:
                _ = cross_feed_address(feed, probe, 0)
            except problem:
                words = String(problem)
            refused(words, "outside its retained data")
        var rates = CrossRates()
        rates.values.append(Float32(0.0625))
        rates.values.append(Float32(0.03125))
        rates.start = 4
        require(cross_rate(rates, 5) == Float32(0.03125), "rate value")
        var late = String("")
        try:
            _ = cross_rate(rates, 6)
        except problem:
            late = String(problem)
        refused(late, "outside its retained schedule")
    elif unit == 2:
        var c = CrossCoordinatorState()
        # Gathered: any split that owns every shard once, in connection order.
        group(c, 3, 3, False)
        peer(c, "a", List[Int](length=1, fill=2), False)
        peer(c, "b", List[Int](length=1, fill=0), False)
        peer(c, "c", List[Int](length=1, fill=1), False)
        require(admit_refusal(c).byte_length() == 0, "gathered split admitted")
        require(c.order[0] == 0 and c.order[1] == 1 and c.order[2] == 2, "gathered order is connection order")
        require(len(c.total) == 4, "total sized by n_total")
        # Chained: blocks by first shard.
        group(c, 2, 3, True)
        peer(c, "late", List[Int](length=1, fill=2), True)
        var block: List[Int] = [0, 1]
        peer(c, "early", block, True)
        require(admit_refusal(c).byte_length() == 0, "chained split admitted")
        require(c.order[0] == 1 and c.order[1] == 0, "chained order is by first shard")
        # Refusals, in the order the declarations are read.
        group(c, 2, 2, False)
        peer(c, "a", List[Int](length=1, fill=0), False)
        refused(admit_refusal(c), "worker count differs from the admitted group")
        peer(c, "a", List[Int](length=1, fill=1), False)
        refused(admit_refusal(c), "workers require distinct string names")
        group(c, 2, 2, False)
        peer(c, "a", List[Int](length=1, fill=0), False)
        peer(c, "b", List[Int](length=1, fill=0), False)
        refused(admit_refusal(c), "workers must own shards 0..1 exactly once")
        group(c, 1, 1, False)
        peer(c, "a", List[Int](length=1, fill=0), False)
        c.defects[0] = CROSS_DEFECT_NAME
        refused(admit_refusal(c), "workers require distinct string names")
        group(c, 1, 1, False)
        peer(c, "a", List[Int](length=1, fill=0), False)
        c.protocols[0] = String("other")
        refused(admit_refusal(c), "protocol mismatch: a")
        group(c, 2, 2, False)
        peer(c, "a", List[Int](length=1, fill=0), False)
        peer(c, "b", List[Int](length=1, fill=1), False)
        c.same[1] = 5
        refused(admit_refusal(c), "workers start from different completed")
        group(c, 1, 1, False)
        peer(c, "a", List[Int](length=1, fill=0), False)
        c.n_total = 1000
        refused(admit_refusal(c), "gradient exceeds the coordinator byte bound")
        group(c, 2, 3, True)
        var gap: List[Int] = [0, 2]
        peer(c, "a", gap, True)
        peer(c, "b", List[Int](length=1, fill=1), True)
        refused(admit_refusal(c), "chained fold: worker must own a contiguous block")
        group(c, 1, 1, True)
        peer(c, "a", List[Int](length=1, fill=0), False)
        refused(admit_refusal(c), "chained fold: every worker must start with chained=True")
    else:
        require(cross_float_word(1.0) == "3f800000", "float word of 1.0")
        require(cross_float_word(0.0625) == "3d800000", "float word of 0.0625")
        require(cross_scheme_name(CROSS_SCHEME_V1) == "sha256.v1", "scheme v1 name")
        require(cross_scheme_name(CROSS_SCHEME_SLICED) == "sliced-sha256-8.v2", "scheme v2 name")
        var cells = List[Float32](length=6, fill=Float32(1.5))
        var flags = List[Int32](length=2, fill=Int32(1))
        var base = Int(cells.unsafe_ptr())
        var addresses: List[Int] = [base, base + 8, base + 16, Int(flags.unsafe_ptr())]
        var first = cross_state_hash(addresses, 2, 2)
        require(first.byte_length() == 64, "state hash is 64 hex digits")
        require(first == cross_state_hash(addresses, 2, 2), "state hash is a function of the bytes")
        flags[1] = Int32(0)
        require(first != cross_state_hash(addresses, 2, 2), "state hash covers the flags")
        require(sha256_hex_addr(base, 8) == sha256_hex_addr(base + 8, 8), "equal spans hash equally")
        _ = len(cells)
        _ = len(flags)


def run_checks() raises:
    var failures = List[Int](length=CROSS_VENDOR_RUN_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def run(unit: Int) {imm fp}:
        try:
            fixture(unit)
        except problem:
            fp[unit] = 1
            print("FAIL cross_vendor_run fixture", unit, String(problem))
    host_parallelize(run, CROSS_VENDOR_RUN_FIXTURES)
    var failed = 0
    for value in failures:  # metadata: fixture summaries
        failed += value
    require(failed == 0, "native cross vendor run failed " + String(failed) + " fixtures")
    print("PASS cross_vendor_run:", CROSS_VENDOR_RUN_FIXTURES, "native state fixtures")
