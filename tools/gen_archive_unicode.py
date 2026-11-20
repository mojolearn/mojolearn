# SPDX-License-Identifier: Apache-2.0
"""Generate native archive Unicode names/digits before compilation, never at runtime.

The existing tokenizer uses the same generated-Mojo-constant mechanism. Download
independent version inputs concurrently, verify pinned bytes, then merge them
deterministically. The generated module contains no Python or runtime data loader.
Hangul/CJK ranges remain algorithmic; no 100,000-name expansion is generated.
The UCD files have the Unicode data license; see docs/licenses/UNICODE-LICENSE.txt.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import subprocess


VERSIONS = (
    (10, "13.0.0", "bdbffbbfc8ad4d3a6d01b5891510458f3d36f7170422af4ea2bed3211a73e8bb", "985128e8c8b2540233331d9b27d897d2c23bd5bcf1d05da960befce6efc59542"),
    (11, "14.0.0", "36018e68657fdcb3485f636630ffe8c8532e01c977703d2803f5b89d6c5feafb", "14b3b677d33f95c51423dce6eef4a6a28b4b160451ecedee4b91edb6745cf4a3"),
    (12, "15.0.0", "806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73", "3e39509e8fae3e5d50ba73759d0b97194501d14a9c63107a6372a46b38be18e8"),
    (13, "15.1.0", "2fc713e6a31a87c4850a37fe2caffa4218180fadb5de86b43a143ddb4581fb86", "fbf0e640bab36e165c4da5b6a98bdd963fcb4f923b5097f26f6f7f18b9678698"),
    (14, "16.0.0", "ff58e5823bd095166564a006e47d111130813dcf8bf234ef79fa51a870edb48f", "9953f0fcebf5ea8091c5c581e4df0e43f20d2533c84ccca7987a9bb819a896a8"),
)


def load_file(cache: Path, version: str, name: str, sha: str) -> str:
    path = cache / version / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".part-{os.getpid()}")
        url = f"https://www.unicode.org/Public/{version}/ucd/{name}"
        try:
            subprocess.run(["curl", "-fsSL", "--retry", "2", "--max-time", "120", url, "-o", str(temporary)], check=True)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != sha:
                raise ValueError(f"Unicode source hash mismatch: {url}")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError(f"Unicode cached source hash mismatch: {path}")
    return raw.decode("utf-8")


def load_version(cache: Path, spec: tuple[int, str, str, str]):
    minor, version, data_sha, alias_sha = spec
    data = load_file(cache, version, "UnicodeData.txt", data_sha)
    aliases = load_file(cache, version, "NameAliases.txt", alias_sha)
    names: dict[str, int] = {}
    digits: dict[int, int] = {}
    ranges: list[tuple[str, int, int]] = []
    start = None
    for line in data.splitlines():
        fields = line.split(";")
        point, name = int(fields[0], 16), fields[1]
        if name.startswith("<CJK Ideograph") or name.startswith("<Tangut Ideograph"):
            family = "cjk" if name.startswith("<CJK") else "tangut"
            if name.endswith(", First>"):
                start = point
            elif name.endswith(", Last>"):
                if start is None:
                    raise ValueError("CJK range end without start")
                ranges.append((family, start, point))
                start = None
        elif not name.startswith("<"):
            names[name] = point
        if fields[7]:
            digits[point] = int(fields[6]) if fields[6] else -2
    if start is not None:
        raise ValueError("unterminated CJK range")
    for line in aliases.splitlines():
        value = line.partition("#")[0].strip()
        if value:
            point, name, _kind = value.split(";")
            names[name] = int(point, 16)
    return minor, names, digits, ranges


def generate(cache: Path) -> tuple[str, int, int]:
    names: dict[str, tuple[int, int]] = {}
    digits: dict[int, tuple[int, int]] = {}
    ranges: dict[tuple[str, int, int], int] = {}
    # Each independent UCD version is one task. These five bounded inputs fit
    # comfortably in memory; never cap a rented machine below its allocation.
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as pool:
        versions = list(pool.map(lambda spec: load_version(cache, spec), VERSIONS))
    for minor, version_names, version_digits, version_ranges in versions:
        bit = 1 << (minor - 10)
        for name, point in version_names.items():
            previous, mask = names.get(name, (point, 0))
            if previous != point or not name.isascii() or any(c in name for c in '"\\\n;'):
                raise ValueError(f"incompatible Unicode name record: {name!r}")
            names[name] = point, mask | bit
        for point, digit in version_digits.items():
            previous, mask = digits.get(point, (digit, 0))
            if previous != digit:
                raise ValueError(f"incompatible Unicode digit record: {point:x}")
            digits[point] = digit, mask | bit
        for bounds in version_ranges:
            ranges[bounds] = ranges.get(bounds, 0) | bit
    records, offsets = [], []
    at = 0
    for name, (point, mask) in sorted(names.items()):
        offsets.append(f"{at:08X}")
        record = f"{name};{point:06X};{mask:02X}\n"
        records.append(record)
        at += len(record)
    offsets.append(f"{at:08X}")
    digit_text = "".join(f"{point:06X}{(digit if digit >= 0 else 15):X}{mask:02X}" for point, (digit, mask) in sorted(digits.items()))
    range_text = "".join(f"{begin:06X}{end:06X}{mask:02X}" for (family, begin, end), mask in sorted(ranges.items()) if family == "cjk")
    tangut_text = "".join(f"{begin:06X}{end:06X}{mask:02X}" for (family, begin, end), mask in sorted(ranges.items()) if family == "tangut")
    text = render_payload(len(names), "".join(records), "".join(offsets), digit_text, range_text, tangut_text)
    return text, len(names), len(digits)


def render_payload(name_count: int, names: str, offsets: str, digits: str, cjk: str, tangut: str) -> str:
    """Deterministic rendering also used to verify retained generated bytes.

    Build evidence checks the payload against its tracked native digest pin
    and then calls this renderer from the exact tracked generator revision.
    That verification needs neither network access nor an interpreter UCD.
    """
    table_sha = hashlib.sha256((names + offsets + digits + cjk + tangut).encode("ascii")).hexdigest()
    return (
        '# SPDX-License-Identifier: Unicode-3.0\n'
        '# Generated by tools/gen_archive_unicode.py from pinned official UCD bytes.\n'
        '# Unicode data license: docs/licenses/UNICODE-LICENSE.txt. Do not edit.\n'
        f'comptime ARCHIVE_UNICODE_TABLE_SHA256 = "{table_sha}"\n'
        f'comptime ARCHIVE_UNICODE_NAME_COUNT = {name_count}\n'
        f'comptime ARCHIVE_UNICODE_NAME_OFFSETS = "{offsets}"\n'
        f'comptime ARCHIVE_UNICODE_NAMES = """{names}"""\n'
        f'comptime ARCHIVE_UNICODE_DIGITS = "{digits}"\n'
        f'comptime ARCHIVE_UNICODE_CJK_RANGES = "{cjk}"\n'
        f'comptime ARCHIVE_UNICODE_TANGUT_RANGES = "{tangut}"\n'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path.home() / ".cache/mojolearn-archive-unicode")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "core/archive_unicode_names_generated.mojo")
    args = parser.parse_args()
    text, names, digits = generate(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.output.exists() or args.output.read_text() != text:
        temporary = args.output.with_suffix(f".part-{os.getpid()}")
        temporary.write_text(text)
        temporary.replace(args.output)
    print(f"generated archive Unicode data: {names} names/aliases, {digits} digits, {len(text)} source bytes; {args.output}")


if __name__ == "__main__":
    main()
