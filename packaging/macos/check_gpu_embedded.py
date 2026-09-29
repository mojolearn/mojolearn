#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Refuse a wheel whose extension contains no compiled GPU kernels.

THIS CHECK EXISTS BECAUSE A GREEN BUILD SHIPPED A BROKEN WHEEL.
mojolearn 0.1.0a2 was built on a GitHub-hosted macOS runner, passed every gate
in the release workflow -- portable-baseline build, ISA disassembly, minos/tag
agreement, twine metadata, install and import on python 3.10 through 3.14 --
was published to TestPyPI, installed cleanly from the index on a real M4, and
then failed on the first fit with

    Failed to create Metal function: core_row_norms_row_norm_kernel...

The function was not in the binary. Compare the same commit built two ways:

                                 local (M4)     hosted runner
    __TEXT/__const               692,543 B          49,039 B
    AIR / metallib markers              98                 0
    kernel-like symbols                 21                 7

The hosted runner compiles the HOST half and silently emits no Metal shader
code, because it has no GPU it recognizes. `mojo build` exits 0. Nothing
downstream noticed, and nothing could: every other gate examines the host
binary or the package metadata, and the wheel imports perfectly because
importing does not touch the device.

WHY THIS AND NOT "RUN A FIT". Running a fit is the better check and is not
available where it is needed: the machine that can be tricked into producing
this wheel is precisely the machine that cannot run one. This check is static,
so it works on the runner that has the problem.

WHAT IT DOES NOT PROVE. That the kernels are correct, or complete, or that any
particular one is present. It proves only that GPU code was emitted at all,
which is the difference between the two builds above. Correctness is
`packaging/macos/verify_wheel.sh` with no flag, on real Apple silicon.
"""

import struct
import sys
from pathlib import Path

# One policy for explicit dispatch-only bindings on all device platforms.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "linux"))
from device_glue import DEVICE_GLUE


def metallib_kernels(data):
    """Read bounded MTLB function records and nonempty LLVM payloads.

    String counts are not kernel counts: x_linear has two genuine kernels
    and only four AIR strings. An empty compiler placeholder has no named
    compute-function record backed by bitcode and cannot pass this check.
    This proves emitted code exists, not that every required kernel works.
    """
    kernels = []
    start = 0
    while (start := data.find(b"MTLB", start)) >= 0:
        offset = start
        start += 4
        if offset + 88 > len(data):
            continue
        total = struct.unpack_from("<Q", data, offset + 16)[0]
        if total < 88 or total > len(data) - offset:
            continue
        blob = data[offset:offset + total]
        function_offset, function_size = struct.unpack_from("<QQ", blob, 24)
        code_offset, code_size = struct.unpack_from("<QQ", blob, 72)
        if not (88 <= function_offset and function_size >= 4
                and function_offset + 4 + function_size <= total
                and 88 <= code_offset and code_size > 24
                and code_offset + code_size <= total):
            continue
        code = blob[code_offset:code_offset + code_size]
        # LLVM's bitcode wrapper gives the real payload's offset and size.
        if code[:4] != b"\xde\xc0\x17\x0b":
            continue
        payload_offset, payload_size = struct.unpack_from("<II", code, 8)
        if not (payload_offset >= 20 and payload_size > 4
                and payload_offset + payload_size <= len(code)
                and code[payload_offset:payload_offset + 4] == b"BC\xc0\xde"):
            continue
        count = struct.unpack_from("<I", blob, function_offset)[0]
        cursor = function_offset + 4
        end = cursor + function_size
        names = []
        if not count or count > function_size // 8:
            continue
        for _ in range(count):
            if cursor + 4 > end:
                break
            length = struct.unpack_from("<I", blob, cursor)[0]
            record_end = cursor + length
            if length < 8 or record_end > end:
                break
            cursor += 4
            fields = {}
            terminated = False
            while cursor + 4 <= record_end:
                tag = blob[cursor:cursor + 4]
                cursor += 4
                if tag == b"ENDT":
                    terminated = True
                    break
                if cursor + 2 > record_end:
                    break
                size = struct.unpack_from("<H", blob, cursor)[0]
                cursor += 2
                if cursor + size > record_end:
                    break
                fields[tag] = blob[cursor:cursor + size]
                cursor += size
            name = fields.get(b"NAME", b"")
            if (not terminated or cursor != record_end or not name.endswith(b"\0")
                    or len(name) < 2 or fields.get(b"TYPE") != b"\x02"):
                break
            names.append(name[:-1].decode("utf-8", errors="replace"))
        else:
            if cursor == end:
                kernels.extend(names)
    return kernels


def identity(path):
    path = Path(path)
    tier = path.parent.name if path.parent.name in ("identical", "deterministic") else "fast"
    return tier, path.stem


def main(paths):
    files = [Path(p) for p in paths]
    if not files:
        print("FAIL no device bindings supplied", file=sys.stderr)
        return 1
    indexed = {}
    for path in files:
        key = identity(path)
        if key in indexed:
            print("FAIL duplicate tier/binding: " + str(key), file=sys.stderr)
            return 1
        indexed[key] = metallib_kernels(path.read_bytes())
    failed = False
    for path in files:
        key = identity(path)
        names = indexed[key]
        if names:
            print(f"  {path}: {len(names)} named Metal compute kernels with LLVM bitcode")
            continue
        delegates = DEVICE_GLUE.get(key)
        if delegates and all(indexed.get((key[0], name)) for name in delegates):
            print(f"  {path}: explicit device-free glue; same-tier compiled delegates {', '.join(delegates)}")
            continue
        failed = True
        reason = "missing compiled same-tier delegates" if delegates else "no validated Metal compute library"
        print(f"FAIL {path}: {reason}", file=sys.stderr)
    return int(failed)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: check_gpu_embedded.py <extension.so> [<extension.so> ...]")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
