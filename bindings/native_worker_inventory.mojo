# SPDX-License-Identifier: Apache-2.0
# Driver placement metadata, not evidence that a model executed GPU kernels.
# Public CUDA Driver API and HIP Runtime API; never ctypes, subprocess,
# Python audit hooks, or a GPU driver call on the CPU/Metal routes.
# https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DEVICE.html
# https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.2/doxygen/html/hip__runtime__api_8h_source.html
from std.ffi import OwnedDLHandle, c_int, c_uint, external_call
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_setitem
from bindings.native_transport import TRANSPORT_BOUNDARY


def _inventory_status(status: c_int, operation: String) raises:
    if status != 0:
        raise Error("worker inventory: " + operation + " failed with driver status " + String(status))


def _inventory_dict(value: PythonObject) raises:
    var bi = TRANSPORT_BOUNDARY.get_or_create_ptr()[].bi
    if py_type(value) is not bi.dict:
        raise Error("worker inventory: metadata requires exact dictionaries")
    var keys = py_call(bi.list, value)
    for i in range(len(keys)):  # driver-loop: refuse hooks before schema key lookup
        if py_type(keys[i]) is not bi.str:
            raise Error("worker inventory: metadata keys require exact strings")


def _inventory_text(record: PythonObject, key: String) raises -> String:
    var value = py_get(record, key)
    if py_type(value) is not TRANSPORT_BOUNDARY.get_or_create_ptr()[].bi.str:
        raise Error("worker inventory: missing native string " + key)
    return String(value)


def _inventory_integer(record: PythonObject, key: String) raises -> Int:
    var value = py_get(record, key)
    if py_type(value) is not TRANSPORT_BOUNDARY.get_or_create_ptr()[].bi.int:
        raise Error("worker inventory: missing native integer " + key)
    return Int(py=value)


def _inventory_utf8(pointer: MutPointer[Int8, MutUntrackedOrigin], maximum: Int) raises -> PythonObject:
    var bi = TRANSPORT_BOUNDARY.get_or_create_ptr()[].bi
    var bytes = bi.bytearray()
    for i in range(maximum):  # small-loop(the bounded driver name byte record)
        if pointer[i] == 0:
            break
        _ = py_call(bytes.append, Int(UInt8(pointer[i])))
    return py_call(bytes.decode, "utf-8", "replace")


def native_worker_inventory(vendor: String, environ: PythonObject) raises -> PythonObject:
    var bi = TRANSPORT_BOUNDARY.get_or_create_ptr()[].bi
    if vendor != "cuda" and vendor != "hip" and vendor != "cpu" and vendor != "metal":
        raise Error("worker inventory: unsupported vendor")
    var result = bi.dict()
    py_setitem(result, "vendor", vendor)
    py_setitem(result, "pid", Int(external_call["getpid", c_int]()))
    py_setitem(result, "ppid", Int(external_call["getppid", c_int]()))
    if vendor == "cpu" or vendor == "metal":
        py_setitem(result, "kind", "worker-process-identity" if vendor == "cpu" else "single-metal-worker")
        return result
    if py_type(environ) is not bi.dict:
        raise Error("worker inventory: visibility requires the connected native environment dictionary")
    # The connected os.environ._data dictionary has native bytes keys/values.
    # Admit before lookups so a foreign equal/hash protocol cannot execute.
    var keys = py_call(bi.list, environ)
    for i in range(len(keys)):  # driver-loop: admit native environment keys before visibility lookup
        if py_type(keys[i]) is not bi.bytes:
            raise Error("worker inventory: environment keys require exact bytes")
    var visibility = bi.dict()
    for key in [String("CUDA_VISIBLE_DEVICES"), String("HIP_VISIBLE_DEVICES"), String("ROCR_VISIBLE_DEVICES")]:  # small-loop(the three visibility variable names)
        var raw_key = py_call(PythonObject(key).encode, "ascii")
        var value = py_get(environ, raw_key)
        if value is not None:
            if py_type(value) is not bi.bytes:
                raise Error("worker inventory: environment values require exact bytes")
            py_setitem(visibility, key, py_call(value.decode, "utf-8", "surrogateescape"))
    var library = String("libcuda.so.1") if vendor == "cuda" else String("libamdhip64.so")
    var driver = OwnedDLHandle(library, flags=2)  # RTLD_NOW, local symbols
    var cuda = vendor == "cuda"
    var init_name = String("cuInit") if cuda else String("hipInit")
    _inventory_status(driver.get_function[c_int](init_name)(c_uint(0)), init_name)
    var count = Array[c_int, 1](c_int(0))
    var count_name = String("cuDeviceGetCount") if cuda else String("hipGetDeviceCount")
    _inventory_status(driver.get_function[c_int](count_name)(count.unsafe_ptr()), count_name)
    if count[0] < 1:
        raise Error("worker inventory: GPU driver returned no visible device")
    var uuid_name = String("hipDeviceGetUuid")
    if cuda:
        uuid_name = String("cuDeviceGetUuid_v2") if driver.check_symbol("cuDeviceGetUuid_v2") else String("cuDeviceGetUuid")
    var devices = bi.list()
    for ordinal in range(Int(count[0])):  # driver-loop: one record per visible driver ordinal
        var device = Array[c_int, 1](c_int(ordinal))
        if cuda:
            _inventory_status(driver.get_function[c_int]("cuDeviceGet")(device.unsafe_ptr(), c_int(ordinal)), "cuDeviceGet")
        var uuid = Array[UInt8, 16](uninitialized=True)
        _inventory_status(driver.get_function[c_int](uuid_name)(uuid.unsafe_ptr(), device[0]), uuid_name)
        var uuid_bytes = py_call(bi.bytearray, 16)
        var any_byte = False
        for j in range(16):  # small-loop(the driver's 16 UUID bytes)
            any_byte = any_byte or uuid[j] != 0
            py_setitem(uuid_bytes, j, Int(uuid[j]))
        if not any_byte:
            raise Error("worker inventory: GPU driver returned an empty UUID")
        var name = Array[Int8, 256](uninitialized=True)
        var bus = Array[Int8, 64](uninitialized=True)
        # Initialize sentinels too: do not scan unbounded memory if a driver
        # returns malformed nonterminated metadata.
        for j in range(256):  # small-loop(the fixed driver name output buffer)
            name[j] = 0
        for j in range(64):  # small-loop(the fixed PCI address output buffer)
            bus[j] = 0
        var name_call = String("cuDeviceGetName") if cuda else String("hipDeviceGetName")
        var bus_call = String("cuDeviceGetPCIBusId") if cuda else String("hipDeviceGetPCIBusId")
        _inventory_status(driver.get_function[c_int](name_call)(name.unsafe_ptr(), c_int(256), device[0]), name_call)
        _inventory_status(driver.get_function[c_int](bus_call)(bus.unsafe_ptr(), c_int(64), device[0]), bus_call)
        name[255] = 0
        bus[63] = 0
        var pci = py_call(PythonObject(String(unsafe_from_utf8_ptr=bus.unsafe_ptr())).lower)
        if len(pci) == 0:
            raise Error("worker inventory: GPU driver returned an empty PCI bus ID")
        var row = bi.dict()
        py_setitem(row, "ordinal", ordinal)
        py_setitem(row, "uuid", py_call(uuid_bytes.hex))
        py_setitem(row, "pci_bus_id", pci)
        # Driver names are UTF-8 in the existing inventory contract.
        py_setitem(row, "name", _inventory_utf8(MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=Int(name.unsafe_ptr())), 256))
        _ = py_call(devices.append, row)
    py_setitem(result, "kind", "visible-device-inventory")
    py_setitem(result, "driver_library", library)
    py_setitem(result, "devices", devices)
    py_setitem(result, "visibility", visibility)
    _ = driver^
    return result


def _inventory_hex(text: String, count_min: Int, count_max: Int) -> Bool:
    if text.byte_length() < count_min or text.byte_length() > count_max:
        return False
    var bytes = text.as_bytes()
    for i in range(len(bytes)):  # small-loop(the bounded UUID or PCI hexadecimal field)
        var byte = bytes[i]
        if not (UInt8(48) <= byte <= UInt8(57)) and not (UInt8(97) <= byte <= UInt8(102)):
            return False
    return True


def _inventory_pci(text: String) raises -> Bool:
    var parts = py_call(PythonObject(text).split, ":")
    if len(parts) != 3 or not _inventory_hex(String(parts[0]), 4, 8) or not _inventory_hex(String(parts[1]), 2, 2):
        return False
    var tail = py_call(parts[2].split, ".")
    return len(tail) == 2 and _inventory_hex(String(tail[0]), 2, 2) and len(tail[1]) == 1 and String(tail[1]) >= "0" and String(tail[1]) <= "7"


def native_require_worker_inventory(records: PythonObject, vendor: String, expected_pids: PythonObject, parent_pid: Int) raises:
    var bi = TRANSPORT_BOUNDARY.get_or_create_ptr()[].bi
    if vendor != "cpu" and vendor != "metal" and vendor != "cuda" and vendor != "hip":
        raise Error("worker inventory: unsupported vendor")
    if (py_type(records) is not bi.list and py_type(records) is not bi.tuple) or (py_type(expected_pids) is not bi.list and py_type(expected_pids) is not bi.tuple):
        raise Error("worker inventory: records and authenticated PIDs require exact lists or tuples")
    if len(records) < 1 or len(records) != len(expected_pids) or parent_pid < 1:
        raise Error("worker inventory: count differs from authenticated worker count")
    if vendor == "metal" and len(records) != 1:
        raise Error("worker inventory: Metal supports one local worker")
    var pids = bi.set()
    var uuids = bi.set()
    var buses = bi.set()
    for i in range(len(records)):  # driver-loop: bind each ordered reply to its authenticated process
        var record = records[i]
        _inventory_dict(record)
        var kind = _inventory_text(record, "kind")
        if _inventory_text(record, "vendor") != vendor:
            raise Error("worker inventory: wrong vendor")
        var pid = _inventory_integer(record, "pid")
        if py_type(expected_pids[i]) is not bi.int or pid < 1 or pid != Int(py=expected_pids[i]):
            raise Error("worker inventory: PID differs from its authenticated worker")
        if _inventory_integer(record, "ppid") != parent_pid:
            raise Error("worker inventory: process was not started by this coordinator")
        if PythonObject(pid) in pids:
            raise Error("worker inventory: repeated process")
        _ = py_call(pids.add, pid)
        if vendor == "cpu" or vendor == "metal":
            if kind != ("worker-process-identity" if vendor == "cpu" else "single-metal-worker"):
                raise Error("worker inventory: wrong process route")
            continue
        if kind != "visible-device-inventory":
            raise Error("worker inventory: GPU placement requires a driver inventory")
        var devices = py_get(record, "devices")
        if py_type(devices) is not bi.list and py_type(devices) is not bi.tuple:
            raise Error("worker inventory: GPU inventory requires a native device list")
        if len(devices) != 1:
            raise Error("worker inventory: each worker must see exactly one GPU")
        var device = devices[0]
        _inventory_dict(device)
        if _inventory_integer(device, "ordinal") != 0:
            raise Error("worker inventory: GPU must have local ordinal zero")
        var uuid = _inventory_text(device, "uuid")
        var bus = _inventory_text(device, "pci_bus_id")
        if not _inventory_hex(uuid, 32, 32) or uuid == "00000000000000000000000000000000" or not _inventory_pci(bus):
            raise Error("worker inventory: missing UUID or PCI identity")
        if PythonObject(uuid) in uuids or PythonObject(bus) in buses:
            raise Error("worker inventory: repeated physical GPU")
        _ = py_call(uuids.add, uuid)
        _ = py_call(buses.add, bus)
