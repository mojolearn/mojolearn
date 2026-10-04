#!/usr/bin/env python3
"""Build the AMD portable payload (PROTOTYPE; docs/AMD_PORTABLE_PATH.md).

The payload is the AMD analogue of `cuda_ptx/sm_80` (ptx_baseline.py): per
binding, the native gfx942 `.so` plus the optimized AMDGPU LLVM IR of every
kernel it embeds, written as bitcode that the OLDEST supported ROCm LLVM can
read (bitcode is read forward: a newer COMGR reads older bitcode, never the
reverse). The runtime half is python/mojolearn/amd_portable.py.

Mojo 1.0.0 has no build mode that keeps kernel IR, so the IR comes from a
separate EXTRACTION build of the same source, never shipped and never run:

  inject   <tree>                rewrite every `.enqueue_function[...]` in a
                                 throwaway source copy to add `dump_llvm=True`;
                                 the compiler then embeds each launch site's
                                 optimized IR as text (measured: identical to
                                 `compile_info[..., emission_kind="llvm-opt"]`)
  extract  <native-set> <extraction-set> <payload> --llvm-bin DIR
                                 pair every kernel embedded in each native
                                 binding with its IR from the extraction build,
                                 rewrite LLVM-24 syntax for the older parser,
                                 assemble bitcode with that LLVM's `opt`, and
                                 write the payload + AMD_PORTABLE.json
  audit    <payload>             re-check the manifest against the files

Every gap refuses: a native kernel without IR, IR the old parser rejects, an
object with more than one kernel. Nothing here qualifies IDENTICAL.
"""

import argparse, hashlib, json, os, re, shutil, struct, subprocess, sys
from pathlib import Path

SCHEMA = "mojolearn.amd-portable.v1"
CODE_FORMAT = "amdgcn-llvm-bc"
MANIFEST = "AMD_PORTABLE.json"
EMITTED_FOR = "gfx942"
TARGETS = ["gfx90a", "gfx942", "gfx950"]
TIERS = ("", "deterministic", "identical")

# ------------------------------------------------------------------ inject

_LAUNCH = re.compile(r"\.enqueue_function\[")


def inject_text(text):
    """(new text, count): add `dump_llvm=True` to every `.enqueue_function[...]`
    parameter list, bracket-balanced (kernel names carry their own `[...]`)."""
    out, pos, n = [], 0, 0
    for m in _LAUNCH.finditer(text):
        if m.start() < pos:
            continue
        depth, i = 1, m.end()
        while depth and i < len(text):
            depth += {"[": 1, "]": -1}.get(text[i], 0)
            i += 1
        if depth:
            raise ValueError(f"unbalanced enqueue_function[ at offset {m.start()}")
        params = text[m.end():i - 1]
        out.append(text[pos:i - 1])
        if "dump_llvm" not in params:
            out.append(", dump_llvm=True")
            n += 1
        out.append("]")
        pos = i
    out.append(text[pos:])
    return "".join(out), n


def cmd_inject(args):
    tree = Path(args.tree)
    if (tree / ".git").exists():
        sys.exit("amd_portable_payload: refusing to rewrite a git checkout; inject into an "
                 "exported copy (git archive) used only for the extraction build")
    total = 0
    for p in sorted(tree.rglob("*.mojo")):
        new, n = inject_text(p.read_text())
        if n:
            p.write_text(new)
            total += n
    print(f"inject: {total} launch sites now embed their IR")
    return 0

# ----------------------------------------------------------------- extract


def embedded_kernels(data):
    """[(offset, size, [kernel names])] of every AMDGPU ELF inside a host binary."""
    out, i = [], 1
    while (i := data.find(b"\x7fELF\x02\x01", i)) >= 0:
        try:
            if struct.unpack_from("<H", data, i + 18)[0] == 224:
                shoff = struct.unpack_from("<Q", data, i + 0x28)[0]
                shentsize, shnum = struct.unpack_from("<HH", data, i + 0x3A)
                size = shoff + shentsize * shnum
                if 0 < size <= len(data) - i:
                    blob = data[i:i + size]
                    names = sorted({m.decode() for m in re.findall(rb"([A-Za-z0-9_$.]{6,})\.kd\x00", blob)})
                    out.append((i, size, names)); i += size; continue
        except struct.error:
            pass
        i += 4
    return out


def embedded_ir(data):
    """{kernel symbol: IR text} of every module a dump_llvm build embeds."""
    out = {}
    for m in re.finditer(rb'; ModuleID = [^\0]*?target triple = "amdgcn-amd-amdhsa"[^\0]*', data):
        text = m.group(0).decode()
        ks = re.findall(r"define [^@\n]*amdgpu_kernel void @([A-Za-z0-9_$.]+)", text)
        if len(ks) != 1:
            raise ValueError(f"embedded IR module at {m.start()} defines {len(ks)} kernels")
        prev = out.get(ks[0])
        if prev is not None and prev.split("\n", 2)[2:] != text.split("\n", 2)[2:]:
            raise ValueError(f"two different IR modules for kernel {ks[0]}")
        out[ks[0]] = text
    return out


def _f32_literal(m):
    v = struct.unpack("<f", struct.pack("<I", int(m.group(1), 16)))[0]
    return "0x%016X" % struct.unpack("<Q", struct.pack("<d", v))[0]


#: LLVM-24 syntax the ROCm 6.4 (LLVM 19) parser rejects, each removed or
#: respelled without changing meaning for code generation. The list is not
#: trusted to be complete: `opt` parsing every module is the check.
DOWNGRADE = [
    ("f0x-float-literal", re.compile(r"\bf0x([0-9A-Fa-f]{8})\b"), _f32_literal),
    ("captures", re.compile(r"\s+captures\((?:none|[a-z_, ()]*?)\)(?=[\s,)])"), ""),
    ("nocreateundeforpoison", re.compile(r"\s+nocreateundeforpoison\b"), ""),
    ("initializes", re.compile(r"\s+initializes\(\([^)]*\)\)"), ""),
    ("icmp-samesign", re.compile(r"\bicmp samesign\b"), "icmp"),
    ("gep-inbounds-nuw", re.compile(r"getelementptr inbounds nuw\b"), "getelementptr inbounds"),
    ("gep-nusw", re.compile(r"getelementptr nusw\b"), "getelementptr"),
    ("gep-nuw", re.compile(r"getelementptr nuw\b"), "getelementptr"),
    ("dead_on_return", re.compile(r"\s+dead_on_return\b"), ""),
]

#: Fast-math flags inventoried per tier (Mojo marks plain float ops
#: `contract`; IDENTICAL code pins products and fmas explicitly).
_FMF = re.compile(r"\b(fmul|fadd|fsub|fdiv|frem|call)\s+((?:(?:contract|afn|reassoc|nnan|ninf|nsz|arcp|fast)\s+)+)")


def downgrade(text):
    counts = {}
    for name, rx, repl in DOWNGRADE:
        text, k = rx.subn(repl, text)
        if k:
            counts[name] = k
    return text, counts


def fmf_inventory(text):
    inv = {}
    for op, flags in _FMF.findall(text):
        for f in flags.split():
            key = f"{op}:{f}"
            inv[key] = inv.get(key, 0) + 1
    return inv


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _llvm_version(opt):
    out = subprocess.run([opt, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"LLVM version (\S+)", out)
    return m.group(1) if m else out.strip().splitlines()[0] if out.strip() else "unknown"


def cmd_extract(args):
    native, extraction, payload = Path(args.native_set), Path(args.extraction_set), Path(args.payload)
    opt = str(Path(args.llvm_bin) / "opt")
    if payload.exists():
        sys.exit(f"amd_portable_payload: {payload} exists; the payload is written fresh")
    bindings, fmf, rules, problems = {}, {}, {}, []
    for tier in TIERS:
        for so in sorted((native / tier).glob("*.so")) if (native / tier).is_dir() else []:
            rel = so.relative_to(native).as_posix()
            kernels = embedded_kernels(so.read_bytes())
            ext = extraction / rel
            if kernels and not ext.is_file():
                problems.append(f"{rel}: no extraction build"); continue
            # a binding with no device code is carried unchanged
            irs = embedded_ir(ext.read_bytes()) if kernels else {}
            row, tier_fmf = {}, fmf.setdefault(tier or "fast", {})
            for _off, _size, names in kernels:
                if len(names) != 1:
                    problems.append(f"{rel}: an embedded object holds {len(names)} kernels"); continue
                k = names[0]
                if k not in irs:
                    problems.append(f"{rel}: kernel {k} has no extracted IR"); continue
                text, counts = downgrade(irs[k])
                for key, v in counts.items():
                    rules[key] = rules.get(key, 0) + v
                for key, v in fmf_inventory(text).items():
                    tier_fmf[key] = tier_fmf.get(key, 0) + v
                bc_rel = f"{rel[:-3]}.ir/{hashlib.sha256(k.encode()).hexdigest()[:24]}.bc"
                dst = payload / bc_rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                r = subprocess.run([opt, "-o", str(dst), "-"], input=text, capture_output=True, text=True)
                if r.returncode:
                    problems.append(f"{rel}: {k}: the old LLVM rejects the IR: "
                                    f"{(r.stderr.strip().splitlines() or ['?'])[-1][:200]}")
                    continue
                row[k] = bc_rel
            (payload / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(so, payload / rel)
            bindings[rel] = dict(kernels=row, native_sha256=_sha(so))
    if problems:
        print("\n".join(f"REFUSE {p}" for p in problems[:40]))
        sys.exit(f"amd_portable_payload: {len(problems)} kernels cannot be carried portably")
    files = {p.relative_to(payload).as_posix(): _sha(p) for p in sorted(payload.rglob("*")) if p.is_file()}
    commit = subprocess.run(["git", "-C", args.repo, "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", args.repo, "status", "--porcelain", "--untracked-files=no"],
                                capture_output=True, text=True).stdout.strip())
    doc = dict(schema=SCHEMA, code_format=CODE_FORMAT, vendor="hip", emitted_for=EMITTED_FOR,
               wave_size=64, targets=TARGETS, identical_qualified=False,
               runtime_verified_targets=[], mojo_version=args.mojo_version,
               bitcode_llvm=_llvm_version(opt), downgrade_rules=rules, fast_math_flags=fmf,
               source_commit=commit, source_dirty=dirty, bindings=bindings, files=files)
    (payload / MANIFEST).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"extract: {len(bindings)} bindings, {sum(len(b['kernels']) for b in bindings.values())} "
          f"kernels, rules {rules}")
    print("EXPERIMENTAL AMD portable payload: not IDENTICAL-qualified")
    return 0


def cmd_audit(args):
    payload = Path(args.payload)
    doc = json.loads((payload / MANIFEST).read_text())
    actual = {p.relative_to(payload).as_posix(): _sha(p) for p in sorted(payload.rglob("*"))
              if p.is_file() and p.name != MANIFEST}
    import importlib.util  # the runtime module alone: importing the package would load bindings
    spec = importlib.util.spec_from_file_location(
        "amd_portable", Path(__file__).resolve().parents[2] / "python/mojolearn/amd_portable.py")
    amd_portable = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(amd_portable)
    amd_portable.validate_manifest(doc, actual)
    for rel, row in doc["bindings"].items():
        native = embedded_kernels((payload / rel).read_bytes())
        names = {n for _o, _s, ns in native for n in ns}
        if names != set(row["kernels"]):
            sys.exit(f"amd_portable_payload: {rel} kernel table differs from its embedded objects")
    print(f"audit: {len(doc['bindings'])} bindings OK, identical_qualified=false")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inject"); p.add_argument("tree"); p.set_defaults(fn=cmd_inject)
    p = sub.add_parser("extract")
    p.add_argument("native_set"); p.add_argument("extraction_set"); p.add_argument("payload")
    p.add_argument("--llvm-bin", required=True, help="bin dir of the OLDEST supported ROCm LLVM")
    p.add_argument("--repo", default=".")
    p.add_argument("--mojo-version", required=True)
    p.set_defaults(fn=cmd_extract)
    p = sub.add_parser("audit"); p.add_argument("payload"); p.set_defaults(fn=cmd_audit)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
