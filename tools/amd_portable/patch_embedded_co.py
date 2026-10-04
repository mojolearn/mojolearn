"""Prototype of the loader's install step for a portable AMD payload: replace
each AMDGPU code object Mojo embedded in a binding/executable with a code
object built at runtime (COMGR) for the local GPU, IN PLACE (the new object
must fit the old span; the tail is zero-filled). Kernels are matched by their
.kd symbol name. Writes a patched copy; never touches the input.
Usage: patch_embedded_co.py <in-binary> <out-binary> <dir-with-<kernel>.hsaco>"""
import pathlib, re, struct, sys

def embedded(data):
    out, i = [], 1
    while (i := data.find(b'\x7fELF\x02\x01', i)) >= 0:
        try:
            if struct.unpack_from('<H', data, i + 18)[0] == 224:
                shoff = struct.unpack_from('<Q', data, i + 0x28)[0]
                shentsize, shnum = struct.unpack_from('<HH', data, i + 0x3A)
                size = shoff + shentsize * shnum
                if 0 < size < len(data) - i:
                    blob = data[i:i + size]
                    names = sorted(set(m.decode() for m in re.findall(rb'([A-Za-z0-9_$.]{6,})\.kd\x00', blob)))
                    out.append((i, size, names)); i += size; continue
        except struct.error:
            pass
        i += 4
    return out

def main():
    src, dst, repl = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    data = bytearray(src.read_bytes())
    found = embedded(bytes(data))
    rc = 0
    for off, size, names in found:
        if len(names) != 1:
            print(f"REFUSE offset {off}: {len(names)} kernels in one object"); rc = 1; continue
        cand = repl / f"{names[0]}.hsaco"
        if not cand.exists():
            print(f"MISSING {names[0]}"); rc = 1; continue
        new = cand.read_bytes()
        if len(new) > size:
            print(f"TOO-BIG {names[0]} new={len(new)} span={size}"); rc = 1; continue
        data[off:off + size] = new + b'\0' * (size - len(new))
        print(f"PATCHED {names[0]} span={size} new={len(new)}")
    dst.write_bytes(bytes(data)); dst.chmod(0o755)
    print(f"{len(found)} embedded objects, rc={rc}")
    return rc

sys.exit(main())
