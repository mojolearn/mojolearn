#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Derive a wheel's macOS floor from shipped native and Metal metadata.

Packaging inspection only. Apple tools read exact existing bytes; no compiler,
load-command edits, GPU execution or binary rewriting occurs here.
"""
import argparse
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import tempfile
from pathlib import Path
import zipfile

# Pinned Mojo1.0 supports macOS15+. Native dylib minos alone does not describe
# the runtime/GPU contract: https://mojolang.static.modular.com/1.0.0/docs/requirements/
SUPPORTED_RUNTIME_FLOOR = "15.0"


def version(text):
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,2}", text):
        raise ValueError("invalid macOS version: " + repr(text))
    words = tuple(map(int, text.split(".")))
    return words + (0,) * (3 - len(words))


def version_text(words):
    return ".".join(map(str, words if words[2] else words[:2]))


def macho_floor(output):
    floors = []
    for block in re.split(r"\nLoad command [0-9]+\n", output):
        if re.search(r"\bcmd LC_BUILD_VERSION\b", block):
            platform = re.search(r"\n\s*platform\s+(\S+)", block)
            minimum = re.search(r"\n\s*minos\s+(\S+)", block)
            if not platform or platform[1] not in ("1", "MACOS", "macos") or not minimum:
                raise ValueError("native payload has unsupported/missing macOS build metadata")
            floors.append(version(minimum[1]))
        elif re.search(r"\bcmd LC_VERSION_MIN_MACOSX\b", block):
            minimum = re.search(r"\n\s*version\s+(\S+)", block)
            if not minimum:
                raise ValueError("native payload has no macOS minimum version")
            floors.append(version(minimum[1]))
    if not floors:
        raise ValueError("native payload has no macOS deployment load command")
    return max(floors)


def metal_floor(output):
    if not re.search(rb"(?m)^Platform: MACOS\r?$", output):
        raise ValueError("embedded Metal library is not a macOS payload")
    values = []
    for name in (b"PlatformMajor", b"PlatformMinor", b"PlatformUpdate"):
        # Apple metal-objdump prints major numerically; its current minor and
        # update fields use one-byte characters (including NUL). Accept both
        # this observed tool output and ordinary numeric output, fail closed
        # for anything else. SDK version is deliberately not consulted.
        prefix = rb"(?m)^" + name + rb": "
        number = re.search(prefix + rb"([0-9]+)\r?\n", output)
        octet = re.search(prefix + rb"(.)\n", output, re.DOTALL)
        if number:
            values.append(int(number[1]))
        elif octet:
            values.append(octet[1][0])
        else:
            raise ValueError("Metal tool omitted " + name.decode())
    if values[0] == 0:
        raise ValueError("Metal library has no deployment major version")
    return tuple(values)


def metallibs(data):
    # Reuse the existing bounded metallib/function/LLVM-payload validator.
    path = Path(__file__).with_name("check_gpu_embedded.py")
    spec = importlib.util.spec_from_file_location("_macos_compat_gpu", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    offset = 0
    while (offset := data.find(b"MTLB", offset)) >= 0:
        start = offset
        offset += 4
        if start + 88 > len(data):
            continue
        size = struct.unpack_from("<Q", data, start + 16)[0]
        if not 88 <= size <= len(data) - start:
            continue
        blob = data[start:start + size]
        if not module.metallib_kernels(blob):
            raise ValueError("unclassified embedded Metal payload; cannot establish deployment floor")
        yield start, blob


def inspect_package(package, requested=None):
    package = Path(package)
    paths = sorted(p for p in package.rglob("*") if p.is_file() and p.suffix in (".so", ".dylib"))
    if not paths:
        raise ValueError("macOS wheel requires staged native payloads before tagging")
    floor = version(SUPPORTED_RUNTIME_FLOOR)
    records, metal_headers = [], {}
    with tempfile.TemporaryDirectory(prefix="mojolearn-metal-metadata-") as temp:
        for path in paths:
            data = path.read_bytes()
            load_commands = subprocess.check_output(["otool", "-l", str(path)], text=True)
            native = macho_floor(load_commands)
            floor = max(floor, native)
            payloads = []
            for offset, blob in metallibs(data):
                # Only byte-identical full libraries share inspection. The
                # deployment version comes from Apple's tool, not guessed
                # header offsets or a nearby target/SDK string.
                key = hashlib.sha256(blob).hexdigest()
                if key not in metal_headers:
                    (Path(temp) / (key + ".metallib")).write_bytes(blob)
                    metal_headers[key] = None
                payloads.append(dict(offset=offset, size=len(blob), sha256=key))
            records.append(dict(path=path.relative_to(package).as_posix(), sha256=hashlib.sha256(data).hexdigest(),
                                macho_minimum=version_text(native), metal_payloads=payloads))
        keys = list(metal_headers)
        # Apple accepts multiple object files. Bounded batches keep argv small
        # while avoiding a tool launch for each of thousands of GPU libraries.
        for start in range(0, len(keys), 128):
            batch = keys[start:start + 128]
            output = subprocess.check_output(["xcrun", "metal-objdump", "--metallib", "--all-headers",
                                              *(key + ".metallib" for key in batch)], cwd=temp)
            sections = re.split(rb"(?m)^([0-9a-f]{64})\.metallib:\r?\n", output)
            if len(sections) != 1 + 2 * len(batch) or sections[0].strip():
                raise ValueError("Metal tool did not identify every inspected payload")
            seen = set()
            for index in range(1, len(sections), 2):
                key = sections[index].decode()
                if key not in batch or key in seen:
                    raise ValueError("Metal tool returned a missing/duplicate payload identity")
                seen.add(key)
                body = sections[index + 1]
                minimum = metal_floor(body)
                metal_headers[key] = (minimum, body.decode("utf-8"))
                floor = max(floor, minimum)
        for record in records:
            for payload in record["metal_payloads"]:
                payload["minimum"] = version_text(metal_headers[payload["sha256"]][0])
    required = floor
    if requested is not None:
        override = version(requested)
        if override < required:
            raise ValueError("requested macOS floor " + requested + " is below required native/Metal/runtime floor " + version_text(required))
        floor = override
    # Wheel macOS compatibility tags represent major/minor. A patch minimum
    # cannot be rounded down silently into a broader installability promise.
    if floor[2]:
        raise ValueError("macOS patch deployment minimum cannot be represented by the wheel tag: " + version_text(floor))
    return dict(required_floor=version_text(required), selected_floor=version_text(floor),
                tag="macosx_" + str(floor[0]) + "_" + str(floor[1]) + "_arm64",
                supported_runtime_floor=SUPPORTED_RUNTIME_FLOOR, native_files=records,
                metal_headers={key: output for key, (_, output) in metal_headers.items()})


def check_wheel(wheel):
    wheel = Path(wheel)
    with tempfile.TemporaryDirectory(prefix="mojolearn-wheel-macos-") as temp, zipfile.ZipFile(wheel) as archive:
        root = Path(temp)
        metadata = [n for n in archive.namelist() if n.endswith(".dist-info/WHEEL")]
        if len(metadata) != 1:
            raise ValueError("wheel must contain one WHEEL metadata member")
        dist_info = metadata[0].removesuffix("WHEEL")
        for member in ("mojolearn/CITATION.cff", dist_info + "licenses/LICENSE", dist_info + "licenses/NOTICE"):
            if member not in archive.namelist():
                raise ValueError("wheel missing required attribution member: " + member)
        for name in archive.namelist():
            if not name.startswith("mojolearn/") or not name.endswith((".so", ".dylib")):
                continue
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError("invalid wheel native member path")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.read(name))
        filename_tag = wheel.name[:-4].rsplit("-", 1)[1]
        match = re.fullmatch(r"macosx_([0-9]+)_([0-9]+)_arm64", filename_tag)
        if not match:
            raise ValueError("unsupported macOS wheel tag: " + filename_tag)
        report = inspect_package(root / "mojolearn", match[1] + "." + match[2])
        tags = re.findall(r"(?m)^Tag: (.+)$", archive.read(metadata[0]).decode())
        if not tags or any(t.rsplit("-", 1)[-1] != filename_tag for t in tags):
            raise ValueError("wheel filename and WHEEL platform tags disagree")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--package", type=Path)
    inputs.add_argument("--wheel", type=Path)
    parser.add_argument("--requested-floor")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    report = check_wheel(args.wheel) if args.wheel else inspect_package(args.package, args.requested_floor)
    if args.receipt:
        args.receipt.write_text(json.dumps(report, indent=2) + "\n")
    print(report["tag"])


if __name__ == "__main__":
    main()
