# SPDX-License-Identifier: Apache-2.0
# Per-executor diagnostics. Disabled by default; clocks and allocation maps
# are touched only during an explicit profile. Timings are inclusive host
# call durations, not device-event timings and not performance evidence.
from std.time import perf_counter_ns


struct ExecutionProfile(Copyable, Movable):
    var enabled: Bool
    var device: Bool
    var addresses: Dict[Int, Int]
    var allocations: Int
    var allocated_bytes: Int
    var frees: Int
    var freed_bytes: Int
    var live_bytes: Int
    var retained_bytes: Int
    var peak_retained_bytes: Int
    var retired_bytes: Int
    var staging_bytes: Int
    var peak_staging_bytes: Int
    var staging_acquires: Int
    var staging_acquired_bytes: Int
    var upload_calls: Int
    var upload_bytes: Int
    var download_calls: Int
    var download_bytes: Int
    var device_copy_calls: Int
    var device_copy_bytes: Int
    var host_copy_calls: Int
    var host_copy_bytes: Int
    var upload_host_ns: Int
    var download_host_ns: Int
    var device_copy_host_ns: Int
    var sync_calls: Int
    var sync_ns: Int
    var kernels: Int
    var fill_calls: Int
    var fill_bytes: Int
    var opaque_calls: Int
    var ops: List[Int]
    var calls: List[Int]
    var elements: List[Int]
    var enqueue_ns: List[Int]

    def __init__(out self, device: Bool = False):
        self.enabled = False
        self.device = device
        self.addresses = Dict[Int, Int]()
        self.allocations = 0
        self.allocated_bytes = 0
        self.frees = 0
        self.freed_bytes = 0
        self.live_bytes = 0
        self.retained_bytes = 0
        self.peak_retained_bytes = 0
        self.retired_bytes = 0
        self.staging_bytes = 0
        self.peak_staging_bytes = 0
        self.staging_acquires = 0
        self.staging_acquired_bytes = 0
        self.upload_calls = 0
        self.upload_bytes = 0
        self.download_calls = 0
        self.download_bytes = 0
        self.device_copy_calls = 0
        self.device_copy_bytes = 0
        self.host_copy_calls = 0
        self.host_copy_bytes = 0
        self.upload_host_ns = 0
        self.download_host_ns = 0
        self.device_copy_host_ns = 0
        self.sync_calls = 0
        self.sync_ns = 0
        self.kernels = 0
        self.fill_calls = 0
        self.fill_bytes = 0
        self.opaque_calls = 0
        self.ops = List[Int]()
        self.calls = List[Int]()
        self.elements = List[Int]()
        self.enqueue_ns = List[Int]()

    def clock(self) -> Int:
        return Int(perf_counter_ns()) if self.enabled else 0

    def allocation(mut self, address: Int, size: Int, existing: Bool = False):
        if not self.enabled:
            return
        self.addresses[address] = size
        self.live_bytes += size
        self.retained_bytes += size
        self.peak_retained_bytes = max(self.peak_retained_bytes, self.retained_bytes)
        if not existing:
            self.allocations += 1
            self.allocated_bytes += size

    def release(mut self, address: Int) raises:
        if not self.enabled or address not in self.addresses:
            return
        var size = self.addresses[address]
        _ = self.addresses.pop(address)
        self.frees += 1
        self.freed_bytes += size
        self.live_bytes -= size
        if self.device:
            self.retired_bytes += size
        else:
            self.retained_bytes -= size

    def staging(mut self, size: Int, existing: Bool = False):
        if not self.enabled:
            return
        self.staging_bytes += size
        self.peak_staging_bytes = max(self.peak_staging_bytes, self.staging_bytes)
        if not existing and size > 0:
            self.staging_acquires += 1
            self.staging_acquired_bytes += size

    def copied(mut self, direction: Int, size: Int):
        if not self.enabled:
            return
        if direction == 0:
            self.upload_calls += 1
            self.upload_bytes += size
        elif direction == 1:
            self.download_calls += 1
            self.download_bytes += size
        elif direction == 2:
            self.device_copy_calls += 1
            self.device_copy_bytes += size
        else:
            self.host_copy_calls += 1
            self.host_copy_bytes += size

    def copy_timed(mut self, direction: Int, started: Int):
        if not self.enabled:
            return
        var elapsed = Int(perf_counter_ns()) - started
        if direction == 0:
            self.upload_host_ns += elapsed
        elif direction == 1:
            self.download_host_ns += elapsed
        else:
            self.device_copy_host_ns += elapsed

    def waited(mut self, started: Int):
        if self.enabled:
            self.sync_calls += 1
            self.sync_ns += Int(perf_counter_ns()) - started

    def drained(mut self):
        if self.enabled:
            self.retained_bytes -= self.retired_bytes
            self.retired_bytes = 0

    def kernel(mut self):
        if self.enabled:
            self.kernels += 1

    def fill(mut self, size: Int):
        if self.enabled:
            self.fill_calls += 1
            self.fill_bytes += size

    def opaque(mut self):
        if self.enabled:
            self.opaque_calls += 1

    def launched(mut self, op: Int, n: Int, started: Int):
        if not self.enabled or n <= 0:
            return
        var elapsed = Int(perf_counter_ns()) - started
        var i = 0
        while i < len(self.ops) and self.ops[i] != op:  # driver-loop: over diagnostic operation codes
            i += 1
        if i == len(self.ops):
            self.ops.append(op)
            self.calls.append(0)
            self.elements.append(0)
            self.enqueue_ns.append(0)
        self.calls[i] += 1
        self.elements[i] += n
        self.enqueue_ns[i] += elapsed
