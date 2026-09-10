"""Query static CUDA driver occupancy for an offline cubin; launch no kernel."""
import ctypes as c
import hashlib
import json
from pathlib import Path
import sys

path, entry = sys.argv[1:]
cuda = c.CDLL('libcuda.so.1')


def call(name, types, *args):
    fn = getattr(cuda, name)
    fn.argtypes, fn.restype = types, c.c_int
    status = fn(*args)
    if status:
        raise RuntimeError(f'{name}: CUDA status {status}')


P = c.c_void_p
I = c.c_int
PI = c.POINTER(I)
PP = c.POINTER(P)
call('cuInit', [c.c_uint], 0)
device, driver = I(), I()
call('cuDeviceGet', [PI, I], c.byref(device), 0)
call('cuDriverGetVersion', [PI], c.byref(driver))
context, module, function = P(), P(), P()
call('cuDevicePrimaryCtxRetain', [PP, I], c.byref(context), device)
try:
    call('cuCtxSetCurrent', [P], context)
    call('cuModuleLoad', [PP, c.c_char_p], c.byref(module), path.encode())
    try:
        call('cuModuleGetFunction', [PP, P, c.c_char_p], c.byref(function), module, entry.encode())
        attrs = {}
        for key, attr in [('max_threads_per_block', 0), ('shared_bytes', 1), ('local_bytes_per_thread', 3), ('registers_per_thread', 4)]:
            value = I()
            call('cuFuncGetAttribute', [PI, I, P], c.byref(value), attr, function)
            attrs[key] = value.value
        maximum, warp, blocks = I(), I(), I()
        call('cuDeviceGetAttribute', [PI, I, I], c.byref(maximum), 39, device)
        call('cuDeviceGetAttribute', [PI, I, I], c.byref(warp), 10, device)
        call('cuOccupancyMaxActiveBlocksPerMultiprocessor', [PI, P, I, c.c_size_t], c.byref(blocks), function, 256, 0)
        print(json.dumps(dict(cubin_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            entry=entry, driver_api_version=driver.value, function_attributes=attrs,
            block_threads=256, dynamic_shared_bytes=0, max_threads_per_sm=maximum.value,
            warp_threads=warp.value, max_active_blocks_per_sm=blocks.value,
            theoretical_thread_occupancy=blocks.value * 256 / maximum.value,
            scope='Static driver occupancy of offline cubin. No kernel launched; achieved occupancy and runtime JIT equivalence are not established.'), indent=2))
    finally:
        call('cuModuleUnload', [P], module)
finally:
    call('cuDevicePrimaryCtxRelease_v2', [I], device)
