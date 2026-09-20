# SPDX-License-Identifier: Apache-2.0
"""Driver device inventory for verifier/worker placement checks.

This records visible hardware identity, not evidence that a model ran kernels.
CUDA signatures: https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__DEVICE.html
HIP signatures: https://rocm.docs.amd.com/projects/HIP/en/docs-6.4.2/doxygen/html/hip__runtime__api_8h_source.html
"""
import ctypes as C
from ctypes.util import find_library
import os
import re


def _library(vendor):
    return C.CDLL('libcuda.so.1' if vendor == 'cuda' else
                  (find_library('amdhip64') or 'libamdhip64.so'))


def visible_gpu_inventory(vendor):
    """Query this worker's driver ordinals after its visibility mask is set.

    PCI identity also rejects two MIG instances on the same PCI device when
    the caller requires distinct physical GPUs. No device context, stream,
    allocation, model computation or performance claim is recorded here.
    """
    if vendor not in ('cuda', 'hip'):
        raise NotImplementedError('GPU inventory requires CUDA or HIP')
    lib = _library(vendor)

    def invoke(name, argtypes, *args):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = argtypes, C.c_int
        status = fn(*args)
        if status != 0:
            raise RuntimeError(f'{name} failed with driver status {status}')

    cuda = vendor == 'cuda'
    invoke('cuInit' if cuda else 'hipInit', [C.c_uint], 0)
    count = C.c_int()
    invoke('cuDeviceGetCount' if cuda else 'hipGetDeviceCount', [C.POINTER(C.c_int)], C.byref(count))
    if count.value < 1:
        raise RuntimeError('GPU inventory returned no visible device')
    uuid_type = C.c_ubyte * 16
    uuid_name = ('cuDeviceGetUuid_v2' if hasattr(lib, 'cuDeviceGetUuid_v2') else 'cuDeviceGetUuid') if cuda else 'hipDeviceGetUuid'
    devices = []
    for ordinal in range(count.value):
        device = C.c_int(ordinal)
        if cuda:
            invoke('cuDeviceGet', [C.POINTER(C.c_int), C.c_int], C.byref(device), ordinal)
        uuid = uuid_type()
        invoke(uuid_name, [C.POINTER(uuid_type), C.c_int], C.byref(uuid), device.value)
        if not any(uuid):
            raise RuntimeError('GPU driver returned an empty UUID')
        name, bus = C.create_string_buffer(256), C.create_string_buffer(64)
        invoke('cuDeviceGetName' if cuda else 'hipDeviceGetName',
               [C.POINTER(C.c_char), C.c_int, C.c_int], name, len(name), device.value)
        invoke('cuDeviceGetPCIBusId' if cuda else 'hipDeviceGetPCIBusId',
               [C.POINTER(C.c_char), C.c_int, C.c_int], bus, len(bus), device.value)
        pci = bus.value.decode('ascii').lower()
        if not pci:
            raise RuntimeError('GPU driver returned an empty PCI bus ID')
        devices.append(dict(ordinal=ordinal, uuid=bytes(uuid).hex(), pci_bus_id=pci,
                            name=name.value.decode('utf-8', errors='replace')))
    return dict(kind='visible-device-inventory', vendor=vendor, pid=os.getpid(),
                driver_library=lib._name, devices=devices,
                visibility={key: os.environ[key] for key in
                            ('CUDA_VISIBLE_DEVICES', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES')
                            if key in os.environ})


def worker_process_inventory():
    """This worker's PROCESS identity, and nothing else.

    THE CPU ROUTE'S PLACEMENT RECORD (lane/cpu-routes-gpu-only-four,
    2026-09-20), and it is NOT a device inventory and must never be read as
    one. `visible_gpu_inventory` above asks a vendor driver which physical
    GPUs this process can see; there is no such question on a CPU-only
    install, where `DevicePool` gives a worker no visibility mask at all and
    a "device index" means one worker process. So this records the one fact
    that IS true there -- which OS process answered -- under its own `kind`,
    so a reader who mistakes the two has to ignore the word `process` in
    every field. It admits that the driver's folds were dispatched to
    separate processes. It admits NOTHING about hardware, isolation,
    residency or throughput, and a column carrying it owes the two-device
    GPU column exactly as before."""
    return dict(kind='worker-process-identity', vendor='cpu', pid=os.getpid(),
                ppid=os.getppid())


def require_distinct_processes(records, count):
    """Require one distinct worker PROCESS per requested index.

    The CPU counterpart of `require_distinct_workers`, deliberately a
    SEPARATE function rather than a vendor branch inside it: the GPU check
    demands a UUID, a PCI bus id and a local ordinal zero, and softening any
    of those to let a CPU record through would have weakened the only place
    that refuses two MIG instances on one card. Nothing here is a device
    claim; see `worker_process_inventory`."""
    if type(count) is not int or count < 1:
        raise RuntimeError('worker placement requires a positive worker count')
    if len(records) != count:
        raise RuntimeError('worker inventory count differs from requested indices')
    pids = set()
    for record in records:
        if record.get('kind') != 'worker-process-identity' or record.get('vendor') != 'cpu':
            raise RuntimeError('worker inventory has the wrong kind or vendor')
        pid, ppid = record.get('pid'), record.get('ppid')
        if type(pid) is not int or pid < 1 or type(ppid) is not int or ppid < 1:
            raise RuntimeError('worker inventory lacks process identity')
        if ppid != os.getpid():
            raise RuntimeError('worker inventory came from a process this driver did not start')
        if pid in pids:
            raise RuntimeError('workers resolve to a repeated process')
        pids.add(pid)


def require_distinct_workers(records, vendor, count):
    """Require one visible physical device per worker, with no repeated GPU.

    This admits placement only. It cannot certify numerical work or throughput.
    """
    if vendor not in ('cuda', 'hip') or type(count) is not int or count < 1:
        raise RuntimeError('GPU placement requires CUDA/HIP and a positive worker count')
    if len(records) != count:
        raise RuntimeError('GPU worker inventory count differs from requested devices')
    uuids, buses, pids = set(), set(), set()
    for record in records:
        if record.get('kind') != 'visible-device-inventory' or record.get('vendor') != vendor:
            raise RuntimeError('GPU worker inventory has the wrong kind or vendor')
        devices = record.get('devices', [])
        if (len(devices) != 1 or type(devices[0].get('ordinal')) is not int
                or devices[0]['ordinal'] != 0):
            raise RuntimeError('each fold worker must see exactly one GPU at local ordinal zero')
        device, pid = devices[0], record.get('pid')
        uuid, bus = device.get('uuid'), device.get('pci_bus_id')
        if (not isinstance(uuid, str) or not re.fullmatch('[0-9a-f]{32}', uuid)
                or not int(uuid, 16) or not isinstance(bus, str)
                or not re.fullmatch(r'[0-9a-f]{4,8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', bus)
                or type(pid) is not int or pid < 1):
            raise RuntimeError('GPU worker inventory lacks UUID, PCI identity or process identity')
        if uuid in uuids or bus in buses or pid in pids:
            raise RuntimeError('GPU workers resolve to a repeated physical device or process')
        uuids.add(uuid)
        buses.add(bus)
        pids.add(pid)
