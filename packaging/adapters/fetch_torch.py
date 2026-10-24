#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch the pinned PyTorch wheel of one vendor and extract what a C++ extension builds against.

    python3 packaging/adapters/fetch_torch.py --vendor cuda|hip|metal --dest DIR [--cache DIR]

The release builds the Torch registration (adapters/torch/registration.cpp)
against the exact wheel its users install (packaging/adapters/pins.json): PyPI's
default torch for NVIDIA and macOS, PyTorch's ROCm index for AMD. A Torch C++
extension is bound to one Torch minor series, one backend and one C++ ABI, so
all three are read from the wheel itself and recorded:

    DIR/torch/include, DIR/torch/share/cmake, DIR/torch/version.py and the
    link libraries (libtorch, libtorch_cpu, libc10) of DIR/torch/lib
    DIR/torch-build.json   version, series, backend and its version, C++ ABI
                           and where it was read (cmake, or symbols:<library>),
                           wheel file, URL and sha256, the include/library dirs

The download is verified against the sha256 the index serves and, when
pins.json names one, against that. Nothing is imported from the wheel and
nothing is installed: the wheel is a zip and only those members are read.

Build glue only (stdlib); it never runs in the library.
"""
import argparse
import ast
import hashlib
import html.parser
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
PINS = HERE / "pins.json"
SCHEMA = "mojolearn.adapter-torch-build.v1"
WHEEL = re.compile(r"^torch-(?P<version>[^-]+)-(?P<py>cp3[0-9]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$")
#: The libraries the registration links (-ltorch -ltorch_cpu -lc10). macOS
#: extracts every dylib of torch/lib: ld64 reads a linked dylib's re-exports.
LINUX_LINK = ("libtorch.so", "libtorch_cpu.so", "libc10.so")


def load_pins(path=PINS):
    return json.loads(pathlib.Path(path).read_text())


def system_of(vendor):
    return "macos" if vendor == "metal" else "linux"


class _Links(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)


def index_files(page, base_url):
    """[(file name, URL without fragment, sha256 or None)] of a PEP 503 project page."""
    parser = _Links()
    parser.feed(page)
    out = []
    for href in parser.links:
        url = urllib.parse.urljoin(base_url, href)
        clean, _, fragment = url.partition("#")
        name = urllib.parse.unquote(clean.rsplit("/", 1)[-1])
        out.append((name, clean, fragment[len("sha256="):] if fragment.startswith("sha256=") else None))
    return out


def platform_matches(plat, system):
    if system == "linux":
        return bool(re.fullmatch(r"(manylinux_[0-9]+_[0-9]+|manylinux[0-9]+|linux)_x86_64", plat))
    return bool(re.fullmatch(r"macosx_[0-9]+_[0-9]+_arm64", plat))


def choose_wheel(files, version, system, python_tag=None):
    """The one wheel of `version` for `system`: the build Python's tag when the
    index has it, else the highest CPython tag; the headers and libraries a C++
    extension uses do not differ between them. Ties break on the file name."""
    rows = []
    for name, url, digest in files:
        m = WHEEL.match(name)
        if m and m["version"] == version and platform_matches(m["plat"], system):
            rows.append((int(m["py"][3:]), m["py"], name, url, digest))
    if not rows:
        raise SystemExit(f"fetch_torch: the index has no torch=={version} wheel for {system}")
    same = [r for r in rows if r[1] == python_tag]
    pick = sorted(same or rows, key=lambda r: (-r[0], r[2]))[0]
    return dict(python_tag=pick[1], wheel=pick[2], url=pick[3], index_sha256=pick[4])


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url, target, expected):
    """`url` to `target` unless `target` already has the `expected` sha256."""
    target = pathlib.Path(target)
    if target.is_file() and expected and sha256_file(target) == expected:
        return expected
    part = target.with_suffix(target.suffix + ".part")
    h = hashlib.sha256()
    with urllib.request.urlopen(url, timeout=120) as response, open(part, "wb") as f:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            h.update(chunk)
            f.write(chunk)
    digest = h.hexdigest()
    if expected and digest != expected:
        part.unlink()
        raise SystemExit(f"fetch_torch: {url} has sha256 {digest}, expected {expected}")
    part.replace(target)
    return digest


def wanted(member, system):
    if member.startswith(("torch/include/", "torch/share/cmake/")) or member == "torch/version.py":
        return True
    if system == "linux":
        return member in tuple("torch/lib/" + name for name in LINUX_LINK)
    return member.startswith("torch/lib/") and member.endswith(".dylib")


def extract(wheel, dest, system):
    dest = pathlib.Path(dest)
    count = 0
    with zipfile.ZipFile(wheel) as z:
        for info in z.infolist():
            name = info.filename
            if info.is_dir() or not wanted(name, system):
                continue
            path = pathlib.PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"fetch_torch: unsafe wheel member {name}")
            target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
            count += 1
    return count


def read_version_py(text):
    """{'__version__', 'cuda', 'hip'} from torch/version.py, read as a syntax tree."""
    out = {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
            name, value = node.target.id, node.value
        else:
            continue
        if name in ("__version__", "cuda", "hip") and isinstance(value, ast.Constant):
            out[name] = value.value
    return out


def read_cxx11_abi(cmake_dir):
    """The _GLIBCXX_USE_CXX11_ABI value Torch's own CMake config hands extensions, or None."""
    for path in sorted(pathlib.Path(cmake_dir).rglob("*.cmake")):
        m = re.search(r"-D_GLIBCXX_USE_CXX11_ABI=([01])", path.read_text(errors="replace"))
        if m:
            return int(m.group(1))
    return None


#: The libraries whose exported symbols name the C++ ABI, in the order asked.
ABI_LIBRARIES = ("libtorch_cpu.so", "libc10.so")
#: std::__cxx11 in a mangled name (std::string is St7__cxx1112basic_string...).
NEW_ABI_MANGLING = (b"St7__cxx11",)
#: The pre-C++11 std::string (Ss) members and the unabbreviated basic_string.
OLD_ABI_MANGLING = (b"_ZNSs", b"_ZNKSs", b"_ZNSt12basic_string")


def _exported_symbols(library):
    """The defined dynamic symbols of an ELF library as bytes (nm -D
    --defined-only), or the whole file when nm is missing or cannot read it."""
    nm = shutil.which("nm")
    if nm:
        done = subprocess.run([nm, "-D", "--defined-only", str(library)], capture_output=True)
        if done.returncode == 0 and done.stdout.strip():
            return done.stdout
    return pathlib.Path(library).read_bytes()


def abi_from_symbols(lib_dir):
    """(abi, library) from the wheel's own binaries, or (None, None).

    PyTorch >= 2.9 builds every manylinux_2_28 wheel with the C++11 ABI and no
    longer writes -D_GLIBCXX_USE_CXX11_ABI into torch/share/cmake, so the ABI is
    read from the mangled names libtorch_cpu (or libc10) exports: any
    std::__cxx11 name means ABI 1; only old-ABI std::string names means ABI 0."""
    for name in ABI_LIBRARIES:
        library = pathlib.Path(lib_dir) / name
        if not library.is_file():
            continue
        symbols = _exported_symbols(library)
        if any(m in symbols for m in NEW_ABI_MANGLING):
            return 1, name
        if any(m in symbols for m in OLD_ABI_MANGLING):
            return 0, name
    return None, None


def detect_cxx11_abi(dest):
    """(abi, source) of an extracted Linux tree: Torch's CMake config first,
    then the exported symbols of its libraries. Nothing else decides: a tree
    neither names refuses with both facts."""
    dest = pathlib.Path(dest)
    abi = read_cxx11_abi(dest / "torch/share/cmake")
    if abi is not None:
        return abi, "cmake"
    abi, library = abi_from_symbols(dest / "torch/lib")
    if abi is not None:
        return abi, f"symbols:{library}"
    raise SystemExit("fetch_torch: the C++ ABI is undecided: torch/share/cmake names no _GLIBCXX_USE_CXX11_ABI, "
                     f"and the exported symbols of torch/lib/{{{','.join(ABI_LIBRARIES)}}} carry neither "
                     "std::__cxx11 nor old-ABI std::string mangling")


def backend_of(info):
    if info.get("hip"):
        return "rocm", info["hip"]
    if info.get("cuda"):
        return "cuda", info["cuda"]
    return None, None


def public_version(version):
    """'2.13.0' from '2.13.0+cu130': the PEP 440 public part, before the local label."""
    return str(version).split("+", 1)[0]


def version_matches(have, pin):
    """Whether a wheel's own torch version satisfies the pinned one. A pin
    without a local label ('2.13.0') names the public release: PyPI's wheel
    reports '2.13.0+cu130' from torch/version.py though its file name says
    2.13.0, so any local label is accepted. A pin with a label
    ('2.13.0+rocm7.2') names one build and must match exactly."""
    if have is None:
        return False
    if "+" in str(pin):
        return str(have) == str(pin)
    return public_version(have) == str(pin)


def describe(dest, pin, system, chosen, digest):
    """The torch-build.json record of an extracted tree; refuses a wheel that
    is not the pinned version or backend."""
    dest = pathlib.Path(dest)
    info = read_version_py((dest / "torch/version.py").read_text())
    if not version_matches(info.get("__version__"), pin["version"]):
        rule = "exactly" if "+" in pin["version"] else "public version"
        raise SystemExit(f"fetch_torch: the wheel says torch {info.get('__version__')}, pins.json {pin['version']} "
                         f"(compared {rule})")
    backend, backend_version = backend_of(info)
    if pin["backend"] == "mps":
        if system != "macos" or backend is not None:
            raise SystemExit(f"fetch_torch: the macOS pin needs a CPU/MPS wheel, got backend {backend}")
        backend = "mps"
    elif backend != pin["backend"]:
        raise SystemExit(f"fetch_torch: the wheel is a {backend} build, pins.json asks for {pin['backend']}")
    abi, abi_source = detect_cxx11_abi(dest) if system == "linux" else (None, None)
    return dict(schema=SCHEMA, version=info["__version__"], version_pin=pin["version"], series=".".join(public_version(info["__version__"]).split(".")[:2]),
                backend=backend, backend_version=backend_version, cxx11_abi=abi, cxx11_abi_source=abi_source,
                wheel=chosen["wheel"], url=chosen["url"], wheel_sha256=digest, index_url=pin["index_url"],
                python_tag=chosen["python_tag"], include_dir="torch/include",
                api_include_dir="torch/include/torch/csrc/api/include", library_dir="torch/lib")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", required=True, choices=("cuda", "hip", "metal"))
    ap.add_argument("--dest", required=True, type=pathlib.Path)
    ap.add_argument("--cache", type=pathlib.Path, default=pathlib.Path.home() / ".cache/mojolearn-adapters")
    ap.add_argument("--pins", type=pathlib.Path, default=PINS)
    ap.add_argument("--python-tag", default=f"cp3{sys.version_info.minor}")
    a = ap.parse_args(argv)
    pins = load_pins(a.pins)
    pin = pins["torch"]["build"][a.vendor]
    system = system_of(a.vendor)
    project = pin["index_url"].rstrip("/") + "/torch/"
    with urllib.request.urlopen(project, timeout=120) as response:
        page = response.read().decode("utf-8", "replace")
    chosen = choose_wheel(index_files(page, project), pin["version"], system, a.python_tag)
    expected = pin.get("wheel_sha256") or chosen["index_sha256"]
    if pin.get("wheel_sha256") and chosen["index_sha256"] and chosen["index_sha256"] != pin["wheel_sha256"]:
        raise SystemExit(f"fetch_torch: the index serves {chosen['wheel']} with sha256 {chosen['index_sha256']}, "
                         f"pins.json pins {pin['wheel_sha256']}")
    if not expected:
        raise SystemExit(f"fetch_torch: {project} serves no sha256 for {chosen['wheel']} and pins.json pins none")
    a.cache.mkdir(parents=True, exist_ok=True)
    wheel = a.cache / chosen["wheel"]
    print(f"fetch_torch: {chosen['wheel']} from {chosen['url']}")
    digest = download(chosen["url"], wheel, expected)
    if a.dest.exists():
        shutil.rmtree(a.dest)
    a.dest.mkdir(parents=True)
    count = extract(wheel, a.dest, system)
    record = describe(a.dest, pin, system, chosen, digest)
    (a.dest / "torch-build.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"fetch_torch: {count} members extracted; torch {record['version']} {record['backend']} "
          f"{record['backend_version'] or ''} cxx11_abi={record['cxx11_abi']} ({record['cxx11_abi_source']}) sha256 {digest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
