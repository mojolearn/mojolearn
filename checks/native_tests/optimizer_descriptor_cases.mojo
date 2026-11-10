# SPDX-License-Identifier: Apache-2.0
"""Real optimizer descriptor contracts, native owners and independent literals.

Every malformed call reaches the exported C entry. The custom sink copies
transient error text before return; no Python bridge or mock entry is involved.
Independent CPU fixtures are parallel. Only one fixture opens optimizer state;
other math calls explicitly use caller-owned moments.
"""
from core.native_descriptor_abi import NativeDescriptorCall
from core.native_descriptor_call import native_descriptor_call, native_descriptor_call_spans
from core.native_call_output import NativeCallOutput
from training.optimizer_native_api import OptimizerNativeDriver, OPT_NATIVE_COUNT
from training.optimizer_public_common import PublicStep, OptHyper

comptime DESCRIPTOR_COMMON_FIXTURES = 24
comptime GUARD = Float32(913.25)


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


struct ErrorProbe(Movable):
    var result: Int
    var calls: Int
    var bytes: List[UInt8]

    def __init__(out self):
        self.result = 72391
        self.calls = 0
        self.bytes = List[UInt8]()


def capture_error(context: Int, address: Int, count: Int) abi("C") -> NoneType:
    var probe = MutPointer[ErrorProbe, MutUntrackedOrigin](unsafe_from_address=context)
    probe[].calls += 1
    var source = Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address)
    # Serial ABI sink: the supplied message expires when this callback returns.
    for index in range(count):
        probe[].bytes.append(source[index])
    return None


def malformed(entry: NativeDescriptorCall, fixture_id: Int) raises:
    var probes = List[ErrorProbe]()
    probes.append(ErrorProbe())
    var addresses: List[Int] = [0]
    var integers: List[Int] = [0, 0, 0, 0]
    var floats: List[Float64] = [0]
    var op = fixture_id
    var aa = Int(addresses.unsafe_ptr())
    var pp = Int(integers.unsafe_ptr())
    var na = 1 if op <= 3 else 0
    var np = 4 if op == 4 or op == 5 else (1 if op == 6 or op == 7 or op == 10 else 0)
    var nf = 0
    var result = Int(MutPointer(to=probes[0].result))
    var expected = String("native optimizer: request span counts differ from the operation")
    if fixture_id < 11:
        # All eleven operations reject a surplus address before dereferencing it.
        na += 1
    elif fixture_id == 11 or fixture_id == 12:
        op = -1 if fixture_id == 11 else OPT_NATIVE_COUNT
        expected = "native optimizer: unknown operation or missing result storage"
    elif fixture_id == 13:
        op = 8
        na = 0
        np = 0
        result = 0
        expected = "native optimizer: unknown operation or missing result storage"
    elif fixture_id == 14:
        op = 0
        na = 1
        np = 0
        aa = 0
        expected = "native optimizer: missing request span"
    elif fixture_id == 15:
        op = 4
        na = 0
        np = 4
        pp = 0
        expected = "native optimizer: missing request span"
    elif fixture_id == 16:
        op = 1
        na = 1
        np = 0
        expected = "native optimizer: missing typed request"
    else:
        op = 9
        na = 0
        np = 0
        nf = 1
    var status = entry(op, aa, na, pp, np, Int(floats.unsafe_ptr()), nf, result,
        Int(probes.unsafe_ptr()), capture_error)
    require(status != 0, "malformed descriptor succeeded")
    require(probes[0].result == 72391, "failed descriptor changed result storage")
    require(probes[0].calls == 1, "error sink was not called exactly once")
    require(len(probes[0].bytes) == len(expected.as_bytes()), "error text was truncated or duplicated")
    probes[0].bytes.append(0)
    var copied = String(unsafe_from_utf8_ptr=probes[0].bytes.unsafe_ptr())
    require(copied == expected, "error sink did not preserve complete refusal text: " + copied)
    _ = probes
    _ = addresses
    _ = integers
    _ = floats


struct FourCells(Movable):
    var data: List[Float32]

    def __init__(out self, a: Float32 = 0, b: Float32 = 0, c: Float32 = 0, d: Float32 = 0):
        self.data = [GUARD, a, b, c, d, GUARD]

    def address(self) -> Int:
        return Int(self.data.unsafe_ptr()) + 4

    def expect(self, a: Float32, b: Float32, c: Float32, d: Float32) raises:
        require(self.data[0] == GUARD and self.data[5] == GUARD, "descriptor wrote past buffer bounds")
        require(self.data[1] == a and self.data[2] == b and self.data[3] == c and self.data[4] == d,
            "descriptor output differs from independent expected cells")

    def fill(mut self, value: Float32):
        self.data[1] = value
        self.data[2] = value
        self.data[3] = value
        self.data[4] = value


struct StepOwners(Movable):
    var parameters: FourCells
    var gradients: FourCells
    var first: FourCells
    var second: FourCells
    var offsets: List[Int32]
    var initialized: List[Int32]

    def __init__(out self):
        self.parameters = FourCells(1, -2, 0.5, -0.25)
        self.gradients = FourCells(1, 2, -1, -2)
        self.first = FourCells()
        self.second = FourCells()
        self.offsets = [0, 4]
        self.initialized = [0]

    def request(self, handle: Int = -1, resident: Int = 0) -> PublicStep:
        var req = PublicStep()
        req.j = 1
        req.hyper = OptHyper(0, 0, False, 0.125, 0.9, 0.999, 1e-8, 0, 0.5, 0, 0)
        req.n_total = 4
        req.offsets = Int(self.offsets.unsafe_ptr())
        req.init = Int(self.initialized.unsafe_ptr())
        req.p_addr = [self.parameters.address()]
        req.p_bytes = [16]
        req.g_addr = [self.gradients.address()]
        req.g_bytes = [16]
        req.m_addr = self.first.address()
        req.v_addr = self.second.address()
        req.handle = handle
        req.resident_wanted = resident
        return req^


def metadata_and_scratch(driver: OptimizerNativeDriver) raises:
    var outputs = List[NativeCallOutput]()
    outputs.append(NativeCallOutput())
    outputs[0].result = 771
    outputs[0].error.append(88)
    var expected = String("native optimizer: unknown operation or missing result storage")
    var refused = False
    try:
        _ = native_descriptor_call_spans(driver.entry, 11, 0, 0, 0, 0, 0, 0, outputs[0])
    except error:
        refused = True
        require(String(error) == expected, "wrapper changed descriptor refusal text")
    require(refused and outputs[0].result == 0, "wrapper did not reset writable error scratch")
    require(len(outputs[0].error) == len(expected.as_bytes()) + 1, "wrapper retained stale error bytes")
    var host = native_descriptor_call_spans(driver.entry, 8, 0, 0, 0, 0, 0, 0, outputs[0])
    require(host == Int(driver.host) and outputs[0].result == host and len(outputs[0].error) == 0,
        "successful call retained previous error/result storage")
    require(native_descriptor_call(driver.entry, 9, [], [], []) == 1, "identical numeric mode metadata differs")
    _ = outputs


def step_roundtrip(driver: OptimizerNativeDriver) raises:
    var owners = StepOwners()
    var request = owners.request()
    var result = driver.step(request^)
    # First SGD momentum buffer is g; p -= (1/8) g, exactly representable.
    owners.parameters.expect(0.875, -2.25, 0.625, 0)
    owners.gradients.expect(1, 2, -1, -2)
    owners.first.expect(1, 2, -1, -2)
    owners.second.expect(0, 0, 0, 0)
    require(owners.offsets[0] == 0 and owners.offsets[1] == 4 and owners.initialized[0] == 1,
        "step registry/init metadata differs")
    require(result.t == 1 and result.handle == -1 and not result.on_device and not result.clip_ran,
        "step result lost counter/ownership flags")
    require(result.total_norm == 0 and result.coef == 0 and result.lr == 0.125 and result.packed == 0 and not result.g_dev,
        "step result lost scalar/output flags")
    _ = owners


def clip_roundtrip(driver: OptimizerNativeDriver) raises:
    var cells = FourCells(3, 4, 0, 0)
    var values = driver.clip([cells.address()], [16], 10)
    require(values[0] == 5 and values[1] == 1, "3-4-5 norm/clamped coefficient differs")
    cells.expect(3, 4, 0, 0)
    values = driver.clip([cells.address()], [16], 2.5)
    # Contract epsilon is 1e-6 rounded to f32. Independent real formula;
    # bound covers the denominator and division's two f32 roundings.
    var expected = Float64(2.5) / (Float64(5) + Float64(Float32(1e-6)))
    require(values[0] == 5 and abs(Float64(values[1]) - expected) <= 1e-7,
        "clipped norm/coefficient differs from independent equation")
    require(abs(Float64(cells.data[1]) - 3 * expected) <= 4e-7 and
        abs(Float64(cells.data[2]) - 4 * expected) <= 4e-7,
        "clipping did not scatter the independently expected values")
    require(cells.data[0] == GUARD and cells.data[5] == GUARD and cells.data[3] == 0 and cells.data[4] == 0,
        "clipping changed zero cells or buffer guards")
    _ = cells


def accumulate_roundtrip(driver: OptimizerNativeDriver) raises:
    var first = FourCells(1, -2, 0.5, 4)
    var second = FourCells(3, 1, -0.5, -2)
    var target = FourCells(17, 17, 17, 17)
    var count = driver.accumulate([first.address(), second.address()], [16], [target.address()], 2, 1024)
    require(count == 4, "accumulate result count differs")
    target.expect(4, -1, 0, 2)
    first.expect(1, -2, 0.5, 4)
    second.expect(3, 1, -0.5, -2)
    _ = first
    _ = second
    _ = target


def state_lifecycle(driver: OptimizerNativeDriver) raises:
    var first = FourCells(1, -2, 0.5, 4)
    var second = FourCells(3, 1, -0.5, -2)
    var m = FourCells(17, 17, 17, 17)
    var v = FourCells(17, 17, 17, 17)
    var handle = driver.open(4)
    try:
        require(driver.download(handle, m.address(), v.address(), 4) == 4, "new state download count differs")
        m.expect(0, 0, 0, 0)
        v.expect(0, 0, 0, 0)
        require(driver.upload(handle, first.address(), second.address(), 4) == 4, "state upload count differs")
        first.fill(81)
        second.fill(82)
        require(driver.download(handle, m.address(), v.address(), 4) == 4, "state download count differs")
        m.expect(1, -2, 0.5, 4)
        v.expect(3, 1, -0.5, -2)
        var refused = False
        try:
            _ = driver.download(handle, m.address(), v.address(), 3)
        except error:
            refused = True
            require(String(error) == "optimizer_resident: handle " + String(handle) + " holds 4 floats, the call names 3",
                "state count refusal changed across descriptor")
        require(refused, "wrong state count was accepted")
        m.expect(1, -2, 0.5, 4)
        v.expect(3, 1, -0.5, -2)
        # Zero addresses explicitly skip one moment, preserving the other.
        require(driver.upload(handle, first.address(), 0, 4) == 4, "partial moment upload count differs")
        _ = driver.download(handle, m.address(), v.address(), 4)
        m.expect(81, 81, 81, 81)
        v.expect(3, 1, -0.5, -2)
    finally:
        _ = driver.close(handle)
    var refused = False
    try:
        _ = driver.download(handle, m.address(), v.address(), 4)
    except error:
        refused = True
        require(String(error) == "optimizer_resident: handle " + String(handle) + " is not open", "closed owner error text differs")
    require(refused, "closed state owner remained usable")
    m.expect(81, 81, 81, 81)
    v.expect(3, 1, -0.5, -2)
    refused = False
    try:
        _ = driver.close(handle)
    except:
        refused = True
    require(refused, "double close was accepted")
    handle = driver.open(4)
    try:
        _ = driver.download(handle, m.address(), v.address(), 4)
        m.expect(0, 0, 0, 0)
        v.expect(0, 0, 0, 0)
    finally:
        _ = driver.close(handle)
    first.expect(81, 81, 81, 81)
    second.expect(82, 82, 82, 82)
    _ = first
    _ = second
    _ = m
    _ = v


def host_io_refusals(driver: OptimizerNativeDriver) raises:
    var owners = StepOwners()
    var req = owners.request()
    var queried = driver.io(req^, 0, 0)
    require(queried[0] == -1 and queried[1] == 0, "host query falsely reports device ownership")
    var refused = False
    req = owners.request()
    try:
        _ = driver.io(req^, 1, 0)
    except error:
        refused = True
        require(String(error) == "RuntimeError: mojolearn optimizer: this build has no device-resident parameter entries (a CPU-only install)",
            "host IO error text differs across descriptor")
    require(refused, "host accepted device IO")
    refused = False
    try:
        _ = driver.free_device(0)
    except error:
        refused = True
        require(String(error) == "RuntimeError: optimizer: a CPU-only install has no device tensor allocation", "host free error text differs")
    require(refused, "host accepted device allocation release")
    owners.parameters.expect(1, -2, 0.5, -0.25)
    owners.gradients.expect(1, 2, -1, -2)
    _ = owners


def run_descriptor_fixture(driver: OptimizerNativeDriver, fixture_id: Int) raises:
    if fixture_id < 18:
        malformed(driver.entry, fixture_id)
    elif fixture_id == 18:
        metadata_and_scratch(driver)
    elif fixture_id == 19:
        step_roundtrip(driver)
    elif fixture_id == 20:
        clip_roundtrip(driver)
    elif fixture_id == 21:
        accumulate_roundtrip(driver)
    elif fixture_id == 22:
        state_lifecycle(driver)
    else:
        # Invalid allocation must refuse before opening shared pool state.
        # Check the complete error after it has crossed the native ABI.
        var refused = False
        try:
            _ = native_descriptor_call(driver.entry, 6, [], [0], [])
        except error:
            refused = True
            var expected = String("optimizer_resident_open: n_total must be >= 1, got 0")
            require(String(error) == expected, "open refusal text differs")
        require(refused, "zero-sized native owner was accepted")
