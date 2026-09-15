#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Generate the host restatement of the Mamba device passes
(lane/cpu-training-mamba, 2026-09-15).

The Mamba-1, Mamba-2 and Mamba-3 prefill backward passes exist only as
device kernels (`mamba/impl/`), and they read forward stage buffers in the
device's layouts. A hand transcription of some sixty kernels would be a
second spelling of every pinned order, which is the drift the IDENTICAL
contract forbids. This tool writes the SAME source out for the host instead,
mechanically, into `mamba/host/gen/`:

  1. every kernel, a `def` whose body opens its thread index with
     `var <i> = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)` (either
     factor order),
     gains a leading `gid_: Int` parameter and that line becomes
     `var <i> = gid_`. Nothing else in a kernel body changes.
  2. every launch `<ctx>.enqueue_function[<kernel>](<args>, grid_dim=G,
     block_dim=T)` becomes `for gid_ in range(launch_count(G, T)):
     <kernel>(gid_, <args>)`. Every kernel bounds-checks its index and
     writes only its own cells, so the serial loop over the launch grid is
     the launch; `launch_count` refuses a grid or block with a second or
     third axis above 1.
  3. imports: `max.gpu.host` and the device GEMM (`gemm.checks.gemm_identical`)
     resolve to `mamba/host/device_shim.mojo` (host buffers under the
     DeviceBuffer and DeviceContext names, and the GEMM through
     `gemm/host/gemm_oracle.mojo`, the profile's definition); `std.gpu` is
     dropped; every converted module resolves to its generated copy.
  4. a top-level def that allocates threadgroup shared memory or waits on a
     barrier (a tiled kernel) keeps its signature and aborts by name; the
     SUBSTITUTIONS below select, at comptime, the arm a column without
     threadgroup shared memory takes, so those kernels are never launched.

Anything the rules above do not cover (a block primitive, a threadgroup
allocation, a leftover `block_idx`) stops the tool by name; it never writes
a partial file.

    python3 tools/mamba_host_gen.py --write   # regenerate mamba/host/gen/
    python3 tools/mamba_host_gen.py --check   # exit 1 if any output is stale

python/mojolearn/tests/test_cpu_training_mamba.py and the CPU identity gate's
manifest step run --check, so a device edit that is not regenerated fails both.
"""
import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = "mamba/host/gen"
OUT_PKG = "mamba.host.gen"
SHIM = "mamba.host.device_shim"

#: source path -> generated module basename
SOURCES = {
    "core/identity_trace.mojo": "identity_trace",
    "gemm/checks/gemm_backward.mojo": "gemm_backward",
    "mamba/impl/ops/selective_scan_interface.mojo": "selective_scan_interface",
    "mamba/impl/modeling/modeling_mamba.mojo": "modeling_mamba",
    "mamba/impl/modules/ssd_minimal.mojo": "ssd_minimal",
    "mamba/impl/modules/mamba2.mojo": "mamba2",
    "mamba/checks/mamba2_backward.mojo": "mamba2_backward_checks",
    "mamba/impl/modules/mamba2_backward.mojo": "mamba2_backward",
    "mamba/impl/ops/mamba2_ssd_backward.mojo": "mamba2_ssd_backward",
    "mamba/impl/modules/mamba2_prefill_backward.mojo": "mamba2_prefill_backward",
    "mamba/checks/mamba_backward.mojo": "mamba_backward_checks",
    "mamba/impl/ops/selective_scan_backward.mojo": "selective_scan_backward",
    "mamba/impl/modeling/modeling_mamba_backward.mojo": "modeling_mamba_backward",
    "mamba/impl/modeling/modeling_mamba_prefill_backward.mojo": "modeling_mamba_prefill_backward",
    "mamba/impl/modules/mamba3_refusal.mojo": "mamba3_refusal",
    "mamba/impl/modules/mamba3_transfer.mojo": "mamba3_transfer",
    "mamba/impl/ops/mamba3_siso.mojo": "mamba3_siso",
    "mamba/impl/modules/mamba3.mojo": "mamba3",
    "mamba/checks/mamba3_backward.mojo": "mamba3_backward_checks",
    "mamba/impl/modules/mamba3_backward.mojo": "mamba3_backward",
    "mamba/impl/modules/mamba3_prefill_backward.mojo": "mamba3_prefill_backward",
}

#: import module -> replacement module (None drops the import line)
IMPORT_MAP = {
    "max.gpu.host": SHIM,
    "gemm.checks.gemm_identical": SHIM,
    "std.gpu": None,
    "max.gpu.memory": None,
    "max.gpu.sync": None,
}

#: Textual substitutions, (source, pattern, replacement). Each must match in
#: its source at least once or the tool stops. They select, at comptime, the
#: arm a column WITHOUT threadgroup shared memory takes: the host has none.
#: The shared-memory kernels those arms bypass are stubbed (rule 4 of the
#: module docstring) and abort by name if ever reached.
SUBSTITUTIONS = (
    ("mamba/impl/ops/mamba3_siso.mojo",
     r"lib_smem_page_fits_for\[TARGET_COLUMN, \d+\]\(\)", "False"),
    ("mamba/impl/modules/mamba3_refusal.mojo",
     r'comptime M3_DEVICE_REFUSAL = not is_defined\["MOJOLEARN_MAMBA3_LEGACY_REFUSAL"\]\(\)',
     "comptime M3_DEVICE_REFUSAL = False"),
)

#: Verbatim top-level blocks lifted out of a device file whose other
#: definitions are device-only: (source, output basename, names, header
#: imports, substitutions inside the lifted text).
EXTRACTS = (
    ("gemm/checks/gemm_identical.mojo", "gemm_identical_parts",
     ("GEMM_FOLD_LEVELS", "GEMM_FOLD_SLOTS", "SAB_FOLD_STRIDE", "SAB_LEAF_ROTATE",
      "contract_partition", "_fold_push", "_fold_drain", "_leaf_bounds", "_leaf_at"),
     ("from std.sys.compile import is_defined",
      "from gemm.checks.gemm_oracle import contract_leaf_count, contract_leaf_size",
      "from checks.numerics import ftz"),
     # the sabotage arm of `_leaf_at` rotates by the launch block; a serial
     # host loop has one block, index 0
     ((r"Int\(block_idx\.x\)", "0"),)),
)

SHARED_MEMORY = re.compile(r"\b(barrier\(\)|stack_allocation\[)")
for _src, _base in SOURCES.items():
    IMPORT_MAP[_src[:-len(".mojo")].replace("/", ".")] = OUT_PKG + "." + _base

CELL_RE = re.compile(
    r"^(\s*(?:[^#\n]*;\s*)?)var\s+(\w+)\s*=\s*(?:Int\(\s*block_idx\.x\s*\)\s*\*\s*Int\(\s*block_dim\.x\s*\)"
    r"|Int\(\s*block_dim\.x\s*\)\s*\*\s*Int\(\s*block_idx\.x\s*\))"
    r"\s*\+\s*Int\(\s*thread_idx\.x\s*\)\s*(;.*)?$"
)
DEVICE_ONLY = re.compile(r"\b(block_idx|thread_idx|grid_dim\.|block_dim\.|barrier|AddressSpace|block_sum|stack_allocation)\b")


class GenError(Exception):
    pass


def _match_close(text, i, open_ch, close_ch):
    """Index of the bracket closing the one at text[i]; skips strings."""
    depth = 0
    j = i
    in_str = None
    while j < len(text):
        c = text[j]
        if in_str:
            if c == "\\":
                j += 2
                continue
            if c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                if c != close_ch:
                    raise GenError(f"unbalanced {open_ch}{close_ch} at offset {i}")
                return j
        j += 1
    raise GenError(f"unclosed {open_ch} at offset {i}")


def _split_args(text):
    """Top-level comma split of an argument list's inside."""
    out, depth, cur, in_str = [], 0, [], None
    i = 0
    while i < len(text):
        c = text[i]
        if in_str:
            cur.append(c)
            if c == "\\":
                cur.append(text[i + 1])
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif c in "\"'":
            in_str = c
            cur.append(c)
        elif c in "([{":
            depth += 1
            cur.append(c)
        elif c in ")]}":
            depth -= 1
            cur.append(c)
        elif c == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(c)
        i += 1
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _flatten(s):
    """One line: comments dropped, whitespace runs collapsed."""
    lines = []
    for line in s.split("\n"):
        # strip a trailing comment outside strings (the argument lists here
        # carry no '#' inside strings)
        k = line.find("#")
        if k >= 0:
            line = line[:k]
        lines.append(line.strip())
    return re.sub(r"\s+", " ", " ".join(lines)).strip()


def _rewrite_imports(text, src):
    out = []
    lines = text.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(\s*)from\s+([\w.]+)\s+import\b(.*)$", line)
        if m and m.group(2) in IMPORT_MAP:
            repl = IMPORT_MAP[m.group(2)]
            block = [line]
            if "(" in m.group(3) and ")" not in m.group(3):
                while ")" not in lines[i]:
                    i += 1
                    block.append(lines[i])
            if repl is not None:
                block[0] = f"{m.group(1)}from {repl} import{m.group(3)}"
                out.extend(block)
            i += 1
            continue
        if m and m.group(2).startswith(("max.gpu", "mamba.impl")):
            raise GenError(f"{src}: import of {m.group(2)} has no host mapping")
        out.append(line)
        i += 1
    return "\n".join(out)


def _def_extent(lines, j):
    """(signature end line, body end line exclusive) of the top-level def at j."""
    head = "\n".join(lines[j:])
    name = re.match(r"^def\s+(\w+)", lines[j]).group(1)
    k = len("def ") + len(name)
    if head[k] == "[":
        k = _match_close(head, k, "[", "]") + 1
    if head[k] != "(":
        raise GenError(f"cannot find the parameter list of {name}")
    close = _match_close(head, k, "(", ")")
    colon = head.index(":", close)
    sig_end = j + head[:colon].count("\n")
    e = sig_end + 1
    while e < len(lines) and not re.match(r"^[A-Za-z_@]", lines[e]):
        e += 1
    return name, k, sig_end, e


def _stub_shared_kernels(text, src):
    """Rule 4: a top-level def that allocates threadgroup memory or waits on
    a barrier keeps its name and signature (plus gid_) and aborts by name."""
    lines = text.split("\n")
    out = []
    names = []
    i = 0
    while i < len(lines):
        if re.match(r"^def\s+\w+", lines[i]):
            name, k, sig_end, e = _def_extent(lines, i)
            body = "\n".join(lines[sig_end + 1:e])
            if SHARED_MEMORY.search(body):
                head = "\n".join(lines[i:sig_end + 1])
                head = head[:k + 1] + "gid_: Int, " + head[k + 1:]
                out.extend(head.split("\n"))
                out.append(f'    abort("mamba host: {name} uses threadgroup shared memory and has no host'
                           f' restatement; the host takes the unshared arm (tools/mamba_host_gen.py)")')
                out.append("")
                out.append("")
                names.append(name)
                i = e
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out), names


def _rewrite_kernels(text, src):
    lines = text.split("\n")
    n_kernels = 0
    for idx, line in enumerate(lines):
        m = CELL_RE.match(line)
        if not m:
            continue
        # the enclosing top-level def
        j = idx
        while j >= 0 and not re.match(r"^def\s+\w+", lines[j]):
            j -= 1
        if j < 0:
            raise GenError(f"{src}:{idx + 1}: thread index outside a def")
        head = "\n".join(lines[j:idx])
        name = re.match(r"^def\s+(\w+)", lines[j]).group(1)
        pos = len(lines[j]) - len(lines[j].lstrip()) + len("def ") + len(name)
        joined = head
        k = pos
        if joined[k] == "[":
            k = _match_close(joined, k, "[", "]") + 1
        if joined[k] != "(":
            raise GenError(f"{src}:{j + 1}: cannot find the parameter list of {name}")
        joined = joined[:k + 1] + "gid_: Int, " + joined[k + 1:]
        new_head = joined.split("\n")
        if len(new_head) != idx - j:
            raise GenError(f"{src}:{j + 1}: signature rewrite changed the line count")
        lines[j:idx] = new_head
        lines[idx] = f"{m.group(1)}var {m.group(2)} = gid_{m.group(3) or ''}"
        n_kernels += 1
    return "\n".join(lines), n_kernels


LAUNCH_RE = re.compile(r"([\w.\[\]]+)\.enqueue_function\[")


def _rewrite_launches(text, src):
    out = []
    i = 0
    n = 0
    while True:
        m = LAUNCH_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        start = m.start()
        line_start = text.rfind("\n", 0, start) + 1
        indent = re.match(r"\s*", text[line_start:]).group(0)
        if text[line_start:start].strip():
            raise GenError(f"{src}: a launch that does not start its line: {text[line_start:m.end()]!r}")
        b0 = m.end() - 1
        b1 = _match_close(text, b0, "[", "]")
        kernel = text[b0 + 1:b1].strip()
        if text[b1 + 1] != "(":
            raise GenError(f"{src}: launch of {kernel} without an argument list")
        p1 = _match_close(text, b1 + 1, "(", ")")
        args = [_flatten(a) for a in _split_args(text[b1 + 2:p1])]
        args = [a for a in args if a]
        grid = [a for a in args if re.match(r"^grid_dim\s*=", a)]
        block = [a for a in args if re.match(r"^block_dim\s*=", a)]
        pos = [a for a in args if not re.match(r"^(grid_dim|block_dim)\s*=", a)]
        if len(grid) != 1 or len(block) != 1 or any("=" in a.split("(")[0] and "==" not in a for a in pos):
            raise GenError(f"{src}: launch of {kernel} is not (positional args, grid_dim, block_dim)")
        g = grid[0].split("=", 1)[1].strip()
        t = block[0].split("=", 1)[1].strip()
        body = ", ".join(["gid_"] + pos)
        out.append(text[i:line_start])
        out.append(f"{indent}for gid_ in range(launch_count({g}, {t})):\n{indent}    {kernel}({body})")
        i = p1 + 1
        n += 1
    return "".join(out), n


def generate(src):
    path = os.path.join(ROOT, src)
    with open(path) as fh:
        text = fh.read()
    for s_src, pattern, repl in SUBSTITUTIONS:
        if s_src == src:
            text, count = re.subn(pattern, repl, text)
            if count == 0:
                raise GenError(f"{src}: substitution {pattern!r} matched nothing")
    text = _rewrite_imports(text, src)
    text, stubbed = _stub_shared_kernels(text, src)
    text, n_kernels = _rewrite_kernels(text, src)
    text, n_launches = _rewrite_launches(text, src)
    in_doc = False
    for lineno, line in enumerate(text.split("\n"), 1):
        quotes = line.count('"""')
        was_doc = in_doc
        if quotes % 2 == 1:
            in_doc = not in_doc
        if was_doc or quotes:
            continue
        code = re.sub(r'"(?:[^"\\]|\\.)*"', '""', line).split("#", 1)[0]
        if re.match(r"^\s*from\s+std\.memory\s+import", code):
            continue
        if DEVICE_ONLY.search(code) and "launch_count" not in code:
            raise GenError(f"{src}: generated line {lineno} still reads a device construct: {line.strip()!r}")
    header = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        f"# GENERATED by tools/mamba_host_gen.py from {src}; do not edit.\n"
        f"# {n_kernels} kernels, {n_launches} launches restated as serial loops over the launch grid.\n"
        + (f"# stubbed shared-memory kernels: {', '.join(stubbed)}.\n" if stubbed else "")
    )
    # the shim's launch_count, imported before the first top-level import
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if re.match(r"^(from|import)\s", line):
            lines.insert(i, f"from {SHIM} import launch_count")
            if stubbed:
                lines.insert(i, "from std.os import abort")
            break
    else:
        raise GenError(f"{src}: no top-level import to place launch_count before")
    if lines[0].startswith("# SPDX-License-Identifier"):
        lines = lines[1:]
    return header + "\n".join(lines)


def extract(src, names, imports, subs):
    with open(os.path.join(ROOT, src)) as fh:
        lines = fh.read().split("\n")
    blocks = {}
    i = 0
    while i < len(lines):
        m = re.match(r"^(def|comptime)\s+(\w+)", lines[i])
        if m and m.group(2) in names:
            if m.group(1) == "def":
                _, _, _, e = _def_extent(lines, i)
            else:
                e = i + 1
                depth = lines[i].count("[") + lines[i].count("(") - lines[i].count("]") - lines[i].count(")")
                while depth > 0:
                    depth += lines[e].count("[") + lines[e].count("(") - lines[e].count("]") - lines[e].count(")")
                    e += 1
            while e > i and not lines[e - 1].strip():
                e -= 1
            blocks[m.group(2)] = "\n".join(lines[i:e])
            i = e
            continue
        i += 1
    missing = [n for n in names if n not in blocks]
    if missing:
        raise GenError(f"{src}: extract found no top-level {', '.join(missing)}")
    body = "\n\n\n".join(blocks[n] for n in names)
    for pattern, repl in subs:
        body, count = re.subn(pattern, repl, body)
        if count == 0:
            raise GenError(f"{src}: extract substitution {pattern!r} matched nothing")
    if DEVICE_ONLY.search(re.sub(r'"(?:[^"\\]|\\.)*"', '""', body)):
        raise GenError(f"{src}: an extracted block reads a device construct")
    return ("# SPDX-License-Identifier: Apache-2.0\n"
            f"# GENERATED by tools/mamba_host_gen.py: verbatim blocks of {src}; do not edit.\n"
            f'"""The host-safe definitions of `{src}` the generated passes import:\n'
            f'{", ".join(names)}."""\n' + "\n".join(imports) + "\n\n\n" + body + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    stale = []
    outputs = {}
    try:
        for src, base in SOURCES.items():
            outputs[os.path.join(OUT_DIR, base + ".mojo")] = generate(src)
        for src, base, names, imports, subs in EXTRACTS:
            outputs[os.path.join(OUT_DIR, base + ".mojo")] = extract(src, names, imports, subs)
    except GenError as exc:
        print(f"mamba_host_gen: {exc}", file=sys.stderr)
        return 2
    init = os.path.join(OUT_DIR, "__init__.mojo")
    outputs[init] = ("# SPDX-License-Identifier: Apache-2.0\n"
                     "# GENERATED by tools/mamba_host_gen.py; do not edit.\n"
                     '"""mamba/host/gen: the Mamba device passes restated on the host, mechanically."""\n')
    for rel, text in outputs.items():
        p = os.path.join(ROOT, rel)
        old = open(p).read() if os.path.exists(p) else None
        if old != text:
            stale.append(rel)
            if args.write:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as fh:
                    fh.write(text)
    if args.check:
        for rel in stale:
            print(f"mamba_host_gen: STALE {rel}")
        return 1 if stale else 0
    for rel in stale:
        print(f"wrote {rel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
