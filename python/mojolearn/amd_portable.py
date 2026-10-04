"""AMD portable payload: LLVM bitcode compiled for the installed GPU (PROTOTYPE).

docs/AMD_PORTABLE_PATH.md has the design and the evidence. In short: the AMD
vendor wheel carries native gfx942 code objects only. A portable payload adds,
for each binding, the optimized AMDGPU LLVM IR of every kernel the binding
embeds (as bitcode an old LLVM can read) plus the native binding itself. On a
detected AMD GPU with no native set, this module compiles every kernel for that
GPU through AMD COMGR (libamd_comgr, shipped by every ROCm runtime install),
replaces each embedded code object of a copy of the binding with the new one,
and returns the directory of patched copies. The Mojo runtime then loads those
copies unchanged (measured on gfx942: bench/results/amd-portable-20261004).

Never a CPU substitution: every failure raises `AmdPortableError`, and the
caller turns it into the import refusal.

Python here is glue only (manifest checks, file IO, ELF byte surgery, one
ctypes call per COMGR action); no numerical work happens in Python.

Not done yet, so the route stays a prototype:
  * an embedded object that grows past its old span is refused; moving it and
    repointing the host `lea` (packaging/linux/cubin_contract.py does this for
    CUDA) is owed;
  * no release marker binds the payload yet, and IDENTICAL has no admission
    scheme on AMD, so IDENTICAL always refuses on this route.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import struct
from pathlib import Path

MANIFEST = "AMD_PORTABLE.json"
SCHEMA = "mojolearn.amd-portable.v1"
CODE_FORMAT = "amdgcn-llvm-bc"
#: Installed location, beside the vendor's native `hip/<gfx>` sets.
PAYLOAD_DIR = ("hip_portable", "cdna-w64")
#: The one family the gfx942-emitted IR is specialized for: wave64 CDNA.
#: Mojo resolves WARP_SIZE, MFMA choices and other target branches when it
#: emits the IR, so the IR cannot serve wave32 RDNA parts (gfx10/11/12).
FAMILY_TARGETS = ("gfx90a", "gfx942", "gfx950")
CACHE_ENV = "MOJOLEARN_AMD_PORTABLE_CACHE"


class AmdPortableError(ValueError):
    """The portable AMD payload cannot serve this device."""


# --------------------------------------------------------------- manifest

def validate_manifest(doc, actual):
    """Raise unless `doc` is a well-formed portable manifest whose file table
    equals `actual` ({relative path: sha256}) exactly."""
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise AmdPortableError("not an AMD portable manifest")
    if doc.get("code_format") != CODE_FORMAT:
        raise AmdPortableError(f"unknown code format {doc.get('code_format')!r}")
    if doc.get("identical_qualified") is not False:
        raise AmdPortableError("a portable manifest must not claim IDENTICAL qualification")
    files = doc.get("files")
    if not isinstance(files, dict) or not files:
        raise AmdPortableError("manifest lists no files")
    if files != actual:
        missing = sorted(set(files) - set(actual))  # glue: manifest file-table comparison
        extra = sorted(set(actual) - set(files))  # glue: manifest file-table comparison
        changed = sorted(k for k in set(files) & set(actual) if files[k] != actual[k])  # glue: manifest file-table comparison
        raise AmdPortableError(f"payload differs from manifest (missing {missing[:3]}, "
                               f"extra {extra[:3]}, changed {changed[:3]})")
    bindings = doc.get("bindings")
    if not isinstance(bindings, dict) or not bindings:
        raise AmdPortableError("manifest lists no bindings")
    for so, row in bindings.items():  # glue: manifest binding table
        if so not in files:
            raise AmdPortableError(f"binding {so} is not in the file table")
        for kernel, bc in row.get("kernels", {}).items():  # glue: manifest kernel table
            if bc not in files:
                raise AmdPortableError(f"kernel {kernel} of {so} has no bitcode file")
    return doc


def family_target(doc, gfx):
    """The COMGR ISA name for device `gfx`, or raise: the payload names which
    targets its IR can serve."""
    base = gfx.split(":")[0]
    if base not in doc.get("targets", ()):
        raise AmdPortableError(f"{base} is outside this payload's family {doc.get('targets')} "
                               "(the IR is wave64 CDNA IR emitted for gfx942)")
    return "amdgcn-amd-amdhsa--" + base


# ----------------------------------------------------------- ELF surgery

def embedded_objects(data):
    """[(offset, size, [kernel names])] of every AMDGPU ELF embedded in a
    host binary. A kernel is named by its `<name>.kd` descriptor symbol."""
    out, i = [], 1
    while (i := data.find(b"\x7fELF\x02\x01", i)) >= 0:
        try:
            if struct.unpack_from("<H", data, i + 18)[0] == 224:  # EM_AMDGPU
                shoff = struct.unpack_from("<Q", data, i + 0x28)[0]
                shentsize, shnum = struct.unpack_from("<HH", data, i + 0x3A)
                size = shoff + shentsize * shnum
                if 0 < size <= len(data) - i:
                    blob = data[i:i + size]
                    names = sorted({m.decode() for m in  # glue: byte scan for embedded ELF headers
                                    re.findall(rb"([A-Za-z0-9_$.]{6,})\.kd\x00", blob)})
                    out.append((i, size, names))
                    i += size
                    continue
        except struct.error:
            pass
        i += 4
    return out


def _sections(elf):
    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", elf, 0x3A)
    rows = []
    for k in range(shnum):  # glue: ELF section header table
        b = shoff + k * shentsize
        name, typ, flags, addr, off, size, link, info, align, ent = struct.unpack_from(
            "<IIQQQQIIQQ", elf, b)
        rows.append(dict(k=k, hdr=b, name_off=name, flags=flags, off=off, size=size,
                         align=max(1, align), type=typ))
    strtab = rows[shstrndx]
    for r in rows:  # glue: ELF section names
        end = elf.index(b"\0", strtab["off"] + r["name_off"])
        r["name"] = elf[strtab["off"] + r["name_off"]:end].decode()
    return rows


def trim_comment(elf):
    """Drop the bytes of a code object's `.comment` (non-loaded compiler
    identification) by sliding the later non-loaded sections and the section
    header table down. COMGR's `.comment` is ~175 B where Mojo's is 19 B, which
    is what keeps a recompiled object inside the old span. Stripping `.symtab`
    instead is NOT allowed: the Mojo runtime crashes without it (measured)."""
    elf = bytes(elf)
    rows = _sections(elf)
    com = [r for r in rows if r["name"] == ".comment"]  # glue: ELF section table
    if not com:
        return elf
    c = com[0]
    later = [r for r in rows if r["off"] > c["off"] and r["type"] != 8]  # not NOBITS  # glue: ELF section table
    if any(r["flags"] & 0x2 for r in later):  # glue: ELF program headers
        return elf  # a loaded section follows; leave the object alone
    phoff = struct.unpack_from("<Q", elf, 0x20)[0]
    phentsize, phnum = struct.unpack_from("<HH", elf, 0x36)
    for k in range(phnum):  # glue: ELF program headers
        p_off, p_filesz = struct.unpack_from("<Q", elf, phoff + k * phentsize + 8)[0], \
            struct.unpack_from("<Q", elf, phoff + k * phentsize + 32)[0]
        if p_off + p_filesz > c["off"]:
            return elf  # a segment covers the comment; leave the object alone
    if struct.unpack_from("<Q", elf, 0x28)[0] < c["off"]:
        return elf  # section headers precede the comment; nothing to slide
    align = max([8] + [r["align"] for r in later])  # glue: ELF section table
    shift = (c["size"] // align) * align
    if shift <= 0:
        return elf
    buf = bytearray(elf[:c["off"] + c["size"] - shift] + elf[c["off"] + c["size"]:])
    for r in later:  # glue: ELF section header offsets
        struct.pack_into("<Q", buf, r["hdr"] - shift + 0x18, r["off"] - shift)
    shoff = struct.unpack_from("<Q", elf, 0x28)[0]
    struct.pack_into("<Q", buf, 0x28, shoff - shift)
    # the section headers moved with the table; fix the comment's own size
    struct.pack_into("<Q", buf, (c["hdr"] - shift) + 0x20, c["size"] - shift)
    return bytes(buf)


def patch_in_place(binary, objects):
    """Return `binary` with every embedded AMDGPU object replaced by
    `objects[kernel]` (bytes), each zero-padded into its old span. Raises when
    any embedded kernel has no replacement or a replacement does not fit."""
    data = bytearray(binary)
    found = embedded_objects(bytes(data))
    if not found:
        raise AmdPortableError("binding embeds no AMDGPU code object")
    for off, size, names in found:  # glue: one entry per embedded code object
        if len(names) != 1:
            raise AmdPortableError(f"embedded object at {off} holds {len(names)} kernels")
        new = objects.get(names[0])
        if new is None:
            raise AmdPortableError(f"no portable code for embedded kernel {names[0]}")
        new = trim_comment(new)
        if len(new) > size:
            raise AmdPortableError(f"recompiled {names[0]} is {len(new)} B, its span {size} B "
                                   "(moving objects is not implemented yet)")
        data[off:off + size] = new + b"\0" * (size - len(new))
    return bytes(data)


# ------------------------------------------------------------------ COMGR

class _H(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64)]


_DATA_BC, _DATA_LOG, _DATA_EXECUTABLE = 0x6, 0x5, 0x8
_ACT_CODEGEN_BC_TO_RELOCATABLE, _ACT_LINK_RELOCATABLE_TO_EXECUTABLE = 0x4, 0x7


def _comgr():
    names = [os.environ.get("MOJOLEARN_AMD_COMGR")] if os.environ.get("MOJOLEARN_AMD_COMGR") else []
    rocm = os.environ.get("ROCM_PATH", "/opt/rocm")
    names += ["libamd_comgr.so.3", "libamd_comgr.so.2", f"{rocm}/lib/libamd_comgr.so.3",
              f"{rocm}/lib/libamd_comgr.so"]
    errors = []
    for n in names:  # glue: loader library candidates
        try:
            return ctypes.CDLL(n)
        except OSError as exc:
            errors.append(f"{n}: {exc}")
    raise AmdPortableError("AMD COMGR (libamd_comgr) is not loadable; it ships with the ROCm "
                           "runtime. Tried: " + "; ".join(errors))


def comgr_version():
    lib = _comgr()
    major, minor = ctypes.c_size_t(), ctypes.c_size_t()
    lib.amd_comgr_get_version(ctypes.byref(major), ctypes.byref(minor))
    return f"{major.value}.{minor.value}"


def compile_bitcode(bc, isa, options=("-O3",)):
    """One kernel's bitcode -> an executable code object for `isa`, through
    COMGR's codegen and link actions. Raises with COMGR's own log."""
    lib = _comgr()

    def ck(status, what):
        if status != 0:
            raise AmdPortableError(f"COMGR {what} failed (status {status})")

    def new_set():
        s = _H()
        ck(lib.amd_comgr_create_data_set(ctypes.byref(s)), "create_data_set")
        return s

    def log_of(s):
        n = ctypes.c_size_t()
        if lib.amd_comgr_action_data_count(s, _DATA_LOG, ctypes.byref(n)) or not n.value:
            return ""
        d = _H()
        if lib.amd_comgr_action_data_get_data(s, _DATA_LOG, 0, ctypes.byref(d)):
            return ""
        sz = ctypes.c_size_t()
        lib.amd_comgr_get_data(d, ctypes.byref(sz), None)
        buf = ctypes.create_string_buffer(sz.value)
        lib.amd_comgr_get_data(d, ctypes.byref(sz), buf)
        lib.amd_comgr_release_data(d)
        return buf.raw[:2000].decode(errors="replace")

    data = _H()
    ck(lib.amd_comgr_create_data(_DATA_BC, ctypes.byref(data)), "create_data")
    ck(lib.amd_comgr_set_data(data, ctypes.c_size_t(len(bc)), ctypes.c_char_p(bc)), "set_data")
    ck(lib.amd_comgr_set_data_name(data, b"kernel.bc"), "set_data_name")
    s_in = new_set()
    ck(lib.amd_comgr_data_set_add(s_in, data), "data_set_add")
    info = _H()
    ck(lib.amd_comgr_create_action_info(ctypes.byref(info)), "create_action_info")
    ck(lib.amd_comgr_action_info_set_isa_name(info, isa.encode()), "set_isa_name")
    ck(lib.amd_comgr_action_info_set_logging(info, ctypes.c_bool(True)), "set_logging")
    opts = (ctypes.c_char_p * len(options))(*[o.encode() for o in options])  # glue: COMGR option strings
    ck(lib.amd_comgr_action_info_set_option_list(info, opts, ctypes.c_size_t(len(options))),
       "set_option_list")
    s_rel, s_exe = new_set(), new_set()
    for act, src, dst, what in ((_ACT_CODEGEN_BC_TO_RELOCATABLE, s_in, s_rel, "codegen"),  # glue: the two COMGR actions
                                (_ACT_LINK_RELOCATABLE_TO_EXECUTABLE, s_rel, s_exe, "link")):
        if lib.amd_comgr_do_action(act, info, src, dst):
            raise AmdPortableError(f"COMGR {what} for {isa} failed:\n{log_of(dst)}")
    exe = _H()
    ck(lib.amd_comgr_action_data_get_data(s_exe, _DATA_EXECUTABLE, 0, ctypes.byref(exe)),
       "get executable")
    sz = ctypes.c_size_t()
    ck(lib.amd_comgr_get_data(exe, ctypes.byref(sz), None), "get_data size")
    buf = ctypes.create_string_buffer(sz.value)
    ck(lib.amd_comgr_get_data(exe, ctypes.byref(sz), buf), "get_data")
    for d in (exe, data):  # glue: release COMGR handles
        lib.amd_comgr_release_data(d)
    for s in (s_in, s_rel, s_exe):  # glue: release COMGR handles
        lib.amd_comgr_destroy_data_set(s)
    lib.amd_comgr_destroy_action_info(info)
    return buf.raw[:sz.value]


# ---------------------------------------------------------- materialize

def payload_root(pkg):
    return Path(pkg).joinpath(*PAYLOAD_DIR)


def load_payload(pkg, source_commit):
    """(root, manifest doc, manifest sha256) after checking every file hash and
    that the payload was built from the clean installed source."""
    root = payload_root(pkg)
    raw = (root / MANIFEST).read_bytes()
    actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob("*") if p.is_file() and p.name != MANIFEST}  # glue: payload file inventory
    doc = validate_manifest(json.loads(raw), actual)
    if doc.get("source_commit") != source_commit or doc.get("source_dirty") is not False:
        raise AmdPortableError("portable payload was not built from the clean installed source")
    return root, doc, hashlib.sha256(raw).hexdigest()


def cache_dir(manifest_hash, gfx, comgr):
    base = os.environ.get(CACHE_ENV) or os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "mojolearn", "amd-portable")
    return Path(base) / manifest_hash[:16] / gfx.split(":")[0] / f"comgr-{comgr}"


def materialize(root, doc, manifest_hash, gfx):
    """Directory of bindings patched for `gfx` (built once, then reused). The
    layout mirrors the native set: `<dir>/{,deterministic,identical}/*.so`."""
    isa = family_target(doc, gfx)
    comgr = comgr_version()
    out = cache_dir(manifest_hash, gfx, comgr)
    stamp = out / "MATERIALIZED.json"
    if stamp.is_file():
        try:
            prev = json.loads(stamp.read_text())
            if prev.get("manifest_sha256") == manifest_hash and all(
                    hashlib.sha256((out / so).read_bytes()).hexdigest() == h
                    for so, h in prev["files"].items()):  # glue: cached file digests
                return out, prev
        except (OSError, ValueError, KeyError, TypeError):
            pass
    files = {}
    for so, row in sorted(doc["bindings"].items()):  # glue: one entry per binding
        native = (root / so).read_bytes()
        if row["kernels"]:
            objects = {k: compile_bitcode((root / bc).read_bytes(), isa)
                       for k, bc in row["kernels"].items()}  # glue: one COMGR compile per kernel
            patched = patch_in_place(native, objects)
        else:
            patched = native  # no device code: carried unchanged
        dst = out / so
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        tmp.write_bytes(patched)
        os.replace(tmp, dst)
        files[so] = hashlib.sha256(patched).hexdigest()
    record = dict(schema="mojolearn.amd-portable-materialized.v1", manifest_sha256=manifest_hash,
                  isa=isa, comgr=comgr, files=files)
    stamp.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return out, record


#: Tiers that take the portable route with no admission (no cross-device
#: bitwise claim). IDENTICAL needs an admission that does not exist yet.
UNADMITTED_TIERS = ("fast", "deterministic")


def identical_refusal(native_refusal, gfx):
    return (f"{native_refusal}\nIDENTICAL AMD portable route refused: code compiled on this "
            f"machine for {gfx} by its own COMGR is not qualified for bitwise-identical results, "
            "and no AMD admission scheme exists yet (docs/AMD_PORTABLE_PATH.md). Run without the "
            "cross-vendor guarantee with MOJOLEARN_NUMERIC_MODE=fast. No CPU substitution is made.")
