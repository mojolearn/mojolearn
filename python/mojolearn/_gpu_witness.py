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
import sys


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


#: How long a child may take to answer the driver's device count. Loading a
#: driver library on a cold box is seconds, not minutes.
INVENTORY_TIMEOUT_S = 120


def unmasked_device_count(vendor, timeout=INVENTORY_TIMEOUT_S):
    """How many GPUs this box shows a process that sets no visibility mask.

    THERE WAS NO SUCH DOOR BEFORE THIS (2026-09-20, lane/par-verify-and-
    queries-nn). `visible_gpu_inventory` is the closest thing the package has
    and every caller runs it INSIDE a worker whose mask has already been set
    to one device, so every existing answer is `1` by construction and none of
    them answers "does this box have two". A verifier that wants two devices
    and cannot count them is a verifier that reports a clean pass over a
    one-device run.

    IT COUNTS IN A CHILD, on purpose. The caller is usually about to fit on
    this box's GPU through the Mojo runtime, and driver initialization in the
    parent is state the parent did not ask for. A child process pays it and
    exits. The child prints the inventory as JSON on stdout; anything else,
    including a non-zero exit or a timeout, is reported as the refusal it is
    rather than guessed at.
    """
    import json
    import subprocess
    if vendor not in ('cuda', 'hip'):
        raise RuntimeError('a device count requires CUDA or HIP; this install reads ' + repr(vendor))
    child = subprocess.run([sys.executable, '-m', 'mojolearn._gpu_witness', vendor],
                           capture_output=True, text=True, timeout=timeout)
    if child.returncode != 0:
        detail = (child.stderr or child.stdout or '').strip().splitlines()
        raise RuntimeError('the GPU driver could not be asked how many devices this box has: '
                           + (detail[-1] if detail else f'exit {child.returncode}'))
    try:
        answer = json.loads(child.stdout)
        return int(answer['count'])
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError('the device count child did not answer a count: ' + repr(exc)) from None


def require_device_count(vendor, count, timeout=INVENTORY_TIMEOUT_S):
    """Refuse BY NAME unless this box shows at least `count` GPUs.

    This is the guard that keeps a two-device column honest before it starts,
    rather than after it has produced hashes nobody can place.
    """
    if type(count) is not int or count < 1:
        raise RuntimeError('a device requirement must be a positive integer')
    have = unmasked_device_count(vendor, timeout=timeout)
    if have < count:
        raise RuntimeError(f'this box shows {have} {vendor} device(s) and the requested column '
                           f'needs {count}. A column recorded here would be a one-device column '
                           'wearing a two-device name, which is the failure the count exists to '
                           'stop. Run it on a box with the devices, or name fewer.')
    return have


def main(argv=None):
    """`python -m mojolearn._gpu_witness <vendor>`: the unmasked inventory as
    JSON. Exists so `unmasked_device_count` can pay the driver init in a child
    and leave the parent's process state alone."""
    import json
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0] not in ('cuda', 'hip'):
        print('usage: python -m mojolearn._gpu_witness {cuda|hip}', file=sys.stderr)
        return 2
    inventory = visible_gpu_inventory(argv[0])
    print(json.dumps(dict(count=len(inventory['devices']), inventory=inventory)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
