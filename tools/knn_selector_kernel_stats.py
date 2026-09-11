#!/usr/bin/env python3
"""tools/knn_selector_kernel_stats.py -- DEVIATION 2519: turn the compiled
small-k selector's resource reports into one stats.tsv row per instantiation.

Sources it understands (any subset; each is optional):

  * the driver log written by the Mojo driver that
    tools/knn_selector_kernel_stats.sh generates on the pod: one
    `KNN_KERNEL_STATS_BEGIN label=... cap=... k=...` line per instantiation,
    followed (when the dump went to stdout) by the PTX text and, on a SASS
    build, the SASS text, then a `KNN_KERNEL_STATS label=... regs=...` line
    with the runtime's own answers (DeviceFunction.get_attribute for
    NUM_REGS / LOCAL_SIZE_BYTES / SHARED_SIZE_BYTES / CONST_SIZE_BYTES /
    MAX_THREADS_PER_BLOCK and occupancy_max_active_blocks_per_multiprocessor
    at 256 threads);
  * `ptxas --verbose` stderr ("Used N registers", "N bytes stack frame,
    N bytes spill stores, N bytes spill loads", "N bytes smem");
  * `cuobjdump --dump-resource-usage` ("REG:N STACK:N SHARED:N LOCAL:N");
  * PTX text (.entry names, the `.local` depot bytes, `.shared` bytes,
    ld.local / st.local counts, `.target`);
  * SASS text from `cuobjdump --dump-sass` or `nvdisasm` (LDL / STL counts).

Occupancy is computed, not measured: registers per thread rounded up to the
sm_90 allocation unit (8 per thread, i.e. 256 per warp), 65,536 registers
per SM, 2,048 threads per SM, 32 blocks per SM, 228 KB shared per SM with a
1 KB per-block reservation; the block is 256 threads. The runtime's own
`occupancy_max_active_blocks_per_multiprocessor` answer, when the driver
ran, is carried beside it and wins when they disagree.

Pure Python 3, no third-party modules. `--selftest` runs on embedded sample
text and is the only thing this file ever executes on the Mac.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

BLOCK_THREADS = 256
SM90_REGS_PER_SM = 65536
SM90_THREADS_PER_SM = 2048
SM90_BLOCKS_PER_SM = 32
SM90_SHARED_PER_SM = 228 * 1024
SM90_SHARED_RESERVED_PER_BLOCK = 1024
SM90_REG_ALLOC_UNIT_PER_THREAD = 8
SM90_MAX_REGS_PER_THREAD = 255

COLUMNS = [
    "label", "cap", "k", "entry", "registers", "local_bytes", "spill_stores",
    "spill_loads", "ld_local", "st_local", "sass_ldl", "sass_stl", "shared_bytes",
    "blocks_per_sm_by_regs", "blocks_per_sm_by_shared", "blocks_per_sm_max",
    "occupancy_pct", "runtime_blocks_per_sm", "runtime_occupancy_pct", "sources",
]


# --------------------------------------------------------------------------
# occupancy
# --------------------------------------------------------------------------
def occupancy(registers, shared_bytes, block_threads=BLOCK_THREADS):
    """Theoretical blocks per SM allowed by registers and by shared memory at
    `block_threads` per block on sm_90, and the resulting thread occupancy."""
    by_regs = None
    if registers is not None and registers > 0:
        unit = SM90_REG_ALLOC_UNIT_PER_THREAD
        padded = ((registers + unit - 1) // unit) * unit
        by_regs = SM90_REGS_PER_SM // (padded * block_threads)
    by_shared = None
    if shared_bytes is not None:
        by_shared = SM90_SHARED_PER_SM // (shared_bytes + SM90_SHARED_RESERVED_PER_BLOCK)
    by_threads = SM90_THREADS_PER_SM // block_threads
    limits = [by_threads, SM90_BLOCKS_PER_SM]
    if by_regs is not None:
        limits.append(by_regs)
    if by_shared is not None:
        limits.append(by_shared)
    blocks = max(0, min(limits))
    pct = 100.0 * blocks * block_threads / SM90_THREADS_PER_SM
    return by_regs, by_shared, blocks, round(pct, 1)


# --------------------------------------------------------------------------
# parsers; every one returns {entry_name: {field: value}}
# --------------------------------------------------------------------------
_PTXAS_ENTRY = re.compile(r"Compiling entry function '([^']+)'")
_PTXAS_PROPS = re.compile(r"Function properties for (\S+)")
_PTXAS_STACK = re.compile(
    r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads")
_PTXAS_USED = re.compile(r"Used (\d+) registers")
_PTXAS_SMEM = re.compile(r"(\d+) bytes smem")
_PTXAS_LMEM = re.compile(r"(\d+) bytes lmem")


def parse_ptxas_verbose(text):
    """`ptxas --verbose` stderr. The 'Used N registers' line follows the
    'Function properties' block of the same entry."""
    out = {}
    current = None
    for line in text.splitlines():
        m = _PTXAS_ENTRY.search(line)
        if m:
            current = m.group(1)
            out.setdefault(current, {})
            continue
        m = _PTXAS_PROPS.search(line)
        if m:
            current = m.group(1)
            out.setdefault(current, {})
            continue
        if current is None:
            continue
        m = _PTXAS_STACK.search(line)
        if m:
            out[current]["stack_bytes"] = int(m.group(1))
            out[current]["spill_stores"] = int(m.group(2))
            out[current]["spill_loads"] = int(m.group(3))
            continue
        m = _PTXAS_USED.search(line)
        if m:
            out[current]["registers"] = int(m.group(1))
            s = _PTXAS_SMEM.search(line)
            if s:
                out[current]["shared_bytes"] = int(s.group(1))
            l = _PTXAS_LMEM.search(line)
            if l:
                out[current]["lmem_bytes"] = int(l.group(1))
    return out


_RES_FUNC = re.compile(r"^\s*Function (\S+?):?\s*$")
_RES_FIELDS = re.compile(r"(REG|STACK|SHARED|LOCAL):(\d+)")


def parse_resource_usage(text):
    """`cuobjdump --dump-resource-usage`."""
    out = {}
    current = None
    for line in text.splitlines():
        m = _RES_FUNC.match(line)
        if m:
            current = m.group(1)
            out.setdefault(current, {})
            continue
        if current is None:
            continue
        fields = dict(_RES_FIELDS.findall(line))
        if fields:
            rec = out[current]
            if "REG" in fields:
                rec["registers"] = int(fields["REG"])
            if "STACK" in fields:
                rec["stack_bytes"] = int(fields["STACK"])
            if "SHARED" in fields:
                rec["shared_bytes"] = int(fields["SHARED"])
            if "LOCAL" in fields:
                rec["local_bytes"] = int(fields["LOCAL"])
    return out


_PTX_ENTRY = re.compile(r"\.entry\s+([^\s(]+)")
_PTX_TARGET = re.compile(r"^\s*\.target\s+(\S+)", re.M)
_PTX_LOCAL = re.compile(r"\.local\s+[^\[;]*\[(\d+)\]")
_PTX_SHARED = re.compile(r"\.shared\s+[^\[;]*\[(\d+)\]")
_PTX_LD_LOCAL = re.compile(r"\bld\.local\b")
_PTX_ST_LOCAL = re.compile(r"\bst\.local\b")
_PTX_INSN = re.compile(r"^\s*[a-z][a-z0-9_.]*\s+[^;]*;\s*$", re.M)


def split_ptx_entries(text):
    """Yield (entry_name, body_text) for every .entry in a PTX module. The
    body runs from the .entry line to the next .entry (or the end)."""
    positions = [(m.start(), m.group(1)) for m in _PTX_ENTRY.finditer(text)]
    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        yield name, text[start:end]


def parse_ptx(text):
    """PTX text: per entry, local depot bytes, shared bytes, ld/st.local
    counts, a static instruction count and the module's .target."""
    out = {}
    target = _PTX_TARGET.search(text)
    target = target.group(1) if target else None
    for name, body in split_ptx_entries(text):
        out[name] = {
            "target": target,
            "local_bytes": sum(int(x) for x in _PTX_LOCAL.findall(body)),
            "shared_bytes": sum(int(x) for x in _PTX_SHARED.findall(body)),
            "ld_local": len(_PTX_LD_LOCAL.findall(body)),
            "st_local": len(_PTX_ST_LOCAL.findall(body)),
            "ptx_insns": len(_PTX_INSN.findall(body)),
        }
    return out


_SASS_FUNC = re.compile(r"^\s*(?:Function\s*:\s*(\S+)|\.text\.(\S+?):)\s*$", re.M)
_SASS_LDL = re.compile(r"\bLDL\b")
_SASS_STL = re.compile(r"\bSTL\b")
_SASS_INSN = re.compile(r"/\*[0-9a-f]{4,}\*/")


def parse_sass(text):
    """`cuobjdump --dump-sass` ('Function : name') or `nvdisasm`
    ('.text.name:') text: LDL / STL counts and an instruction count."""
    out = {}
    marks = [(m.start(), m.group(1) or m.group(2)) for m in _SASS_FUNC.finditer(text)]
    if not marks:
        # One nameless block (a driver dump of a single kernel).
        marks = [(0, "")]
    for i, (start, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        body = text[start:end]
        out[name] = {
            "sass_ldl": len(_SASS_LDL.findall(body)),
            "sass_stl": len(_SASS_STL.findall(body)),
            "sass_insns": len(_SASS_INSN.findall(body)),
        }
    return out


_KV = re.compile(r"(\w+)=(\S+)")


def parse_driver_log(text):
    """The generated Mojo driver's stdout: per label, the runtime attribute
    answers, plus whatever PTX / SASS text sat between the BEGIN and the
    stats line when the dump went to stdout. Returns
    {label: {cap, k, runtime_*, ptx_text, sass_text}}."""
    out = {}
    label = None
    buf = []
    for line in text.splitlines():
        if line.startswith("KNN_KERNEL_STATS_BEGIN "):
            kv = dict(_KV.findall(line))
            label = kv.get("label")
            out[label] = {"cap": _int(kv.get("cap")), "k": _int(kv.get("k")), "dump_text": ""}
            buf = []
            continue
        if line.startswith("KNN_KERNEL_STATS "):
            kv = dict(_KV.findall(line))
            lab = kv.get("label", label)
            rec = out.setdefault(lab, {"cap": None, "k": None, "dump_text": ""})
            for key in ("regs", "local", "shared", "const", "max_threads", "blocks_per_sm_256"):
                if key in kv:
                    rec["runtime_" + key] = _int(kv[key])
            if "cap" in kv and rec.get("cap") is None:
                rec["cap"] = _int(kv["cap"])
            if "k" in kv and rec.get("k") is None:
                rec["k"] = _int(kv["k"])
            rec["dump_text"] = "\n".join(buf)
            buf = []
            label = None
            continue
        if line.startswith("KNN_KERNEL_STATS_ERROR "):
            kv = dict(_KV.findall(line))
            lab = kv.get("label", label)
            rec = out.setdefault(lab, {"cap": None, "k": None, "dump_text": ""})
            rec["error"] = line
            continue
        if label is not None:
            buf.append(line)
    return out


def _int(s):
    if s is None:
        return None
    try:
        return int(s)
    except ValueError:
        return None


# --------------------------------------------------------------------------
# PTX blobs inside a Mojo-built shared library (attempt, never required)
# --------------------------------------------------------------------------
_BLOB_HEAD = re.compile(rb"//\s*\n// Generated by LLVM NVPTX Back-End")


def extract_ptx_blobs(data):
    """Every NVPTX module embedded as text in a binary: from each LLVM NVPTX
    header to the next header or the first NUL byte. Returns a list of
    decoded strings; empty when the binary carries none."""
    heads = [m.start() for m in _BLOB_HEAD.finditer(data)]
    blobs = []
    for i, start in enumerate(heads):
        end = heads[i + 1] if i + 1 < len(heads) else len(data)
        chunk = data[start:end]
        nul = chunk.find(b"\x00")
        if nul >= 0:
            chunk = chunk[:nul]
        blobs.append(chunk.decode("utf-8", "replace"))
    return blobs


# --------------------------------------------------------------------------
# assembly of rows
# --------------------------------------------------------------------------
def merge(dst, src, fields):
    for f in fields:
        if f in src and src[f] is not None and dst.get(f) is None:
            dst[f] = src[f]


def build_rows(driver, per_label_files):
    """driver: parse_driver_log output. per_label_files: {label: {kind:
    text}} with kind in ptx, sass, ptxas, resources. Returns rows in
    COLUMNS order (as dicts)."""
    rows = []
    labels = list(driver.keys())
    for lab in per_label_files:
        if lab not in labels:
            labels.append(lab)
    for lab in labels:
        d = driver.get(lab, {})
        files = per_label_files.get(lab, {})
        row = {c: None for c in COLUMNS}
        row["label"] = lab
        row["cap"] = d.get("cap")
        row["k"] = d.get("k")
        sources = []
        # 1. the runtime's own answers (the cubin the driver's JIT produced)
        if "runtime_regs" in d:
            row["registers"] = d["runtime_regs"]
            row["local_bytes"] = d.get("runtime_local")
            row["shared_bytes"] = d.get("runtime_shared")
            row["runtime_blocks_per_sm"] = d.get("runtime_blocks_per_sm_256")
            sources.append("runtime")
        # 2. ptxas --verbose on the dumped PTX (spills live only here)
        ptxas_text = files.get("ptxas", "")
        if ptxas_text:
            recs = parse_ptxas_verbose(ptxas_text)
            rec = _single(recs)
            if rec:
                row["entry"] = rec.get("_name")
                merge(row, rec, ("registers", "shared_bytes", "spill_stores", "spill_loads"))
                if row["local_bytes"] is None and "stack_bytes" in rec:
                    row["local_bytes"] = rec["stack_bytes"]
                sources.append("ptxas")
        # 3. cuobjdump --dump-resource-usage on the assembled cubin
        res_text = files.get("resources", "")
        if res_text:
            rec = _single(parse_resource_usage(res_text))
            if rec:
                row["entry"] = row["entry"] or rec.get("_name")
                merge(row, rec, ("registers", "shared_bytes", "local_bytes"))
                sources.append("cuobjdump")
        # 4. the PTX text itself (depot bytes and ld/st.local counts)
        ptx_text = files.get("ptx", "") or _ptx_from_dump(d.get("dump_text", ""))
        if ptx_text:
            rec = _single(parse_ptx(ptx_text))
            if rec:
                row["entry"] = row["entry"] or rec.get("_name")
                row["ld_local"] = rec["ld_local"]
                row["st_local"] = rec["st_local"]
                merge(row, rec, ("shared_bytes",))
                if row["local_bytes"] is None:
                    row["local_bytes"] = rec["local_bytes"]
                sources.append("ptx")
        # 5. SASS (LDL / STL are what actually execute)
        sass_text = files.get("sass", "") or _sass_from_dump(d.get("dump_text", ""))
        if sass_text:
            rec = _single(parse_sass(sass_text))
            if rec:
                row["sass_ldl"] = rec["sass_ldl"]
                row["sass_stl"] = rec["sass_stl"]
                sources.append("sass")
        by_regs, by_shared, blocks, pct = occupancy(row["registers"], row["shared_bytes"])
        row["blocks_per_sm_by_regs"] = by_regs
        row["blocks_per_sm_by_shared"] = by_shared
        row["blocks_per_sm_max"] = blocks if row["registers"] is not None else None
        row["occupancy_pct"] = pct if row["registers"] is not None else None
        if row["runtime_blocks_per_sm"] is not None:
            row["runtime_occupancy_pct"] = round(
                100.0 * row["runtime_blocks_per_sm"] * BLOCK_THREADS / SM90_THREADS_PER_SM, 1)
        if d.get("error"):
            sources.append("driver-error")
        row["sources"] = "+".join(sources) if sources else "none"
        rows.append(row)
    return rows


def _single(recs):
    """The one kernel record in a per-instantiation file; when a file holds
    several entries (a whole-module dump), the one whose name mentions the
    selector, else the largest."""
    if not recs:
        return None
    names = list(recs.keys())
    pick = None
    for n in names:
        if "smallk" in n or "select_smallk_identical_candidate" in n:
            pick = n
            break
    if pick is None:
        pick = max(names, key=lambda n: sum(v for v in recs[n].values() if isinstance(v, int)))
    rec = dict(recs[pick])
    rec["_name"] = pick
    return rec


def _ptx_from_dump(text):
    i = text.find(".version")
    if i < 0:
        return ""
    # SASS, when present, follows the PTX; cut at the first SASS marker.
    j = _first_sass_marker(text, i)
    return text[i:j]


def _sass_from_dump(text):
    j = _first_sass_marker(text, 0)
    return text[j:] if j < len(text) else ""


def _first_sass_marker(text, start):
    cands = [text.find(s, start) for s in ("\tcode for sm_", "code for sm_", ".text.", "Function : ", ".headerflags")]
    cands = [c for c in cands if c >= 0]
    return min(cands) if cands else len(text)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def gather(out_dir):
    """Read <out_dir>/driver.log and <out_dir>/dumps/<label>.{ptx,sass,ptxas.log,resources.log}."""
    out_dir = Path(out_dir)
    driver = {}
    log = out_dir / "driver.log"
    if log.is_file():
        driver = parse_driver_log(log.read_text(errors="replace"))
    per_label = {}
    dumps = out_dir / "dumps"
    if dumps.is_dir():
        for p in sorted(dumps.iterdir()):
            name = p.name
            for suffix, kind in ((".ptxas.log", "ptxas"), (".resources.log", "resources"),
                                 (".sass", "sass"), (".ptx", "ptx")):
                if name.endswith(suffix):
                    label = name[: -len(suffix)]
                    per_label.setdefault(label, {})[kind] = p.read_text(errors="replace")
                    break
    return driver, per_label


def split_driver_log(out_dir):
    """When the driver dumped to stdout, cut each instantiation's PTX and
    SASS out of <out_dir>/driver.log into <out_dir>/dumps/<label>.{ptx,sass}
    (never overwriting a file the file-path variant already wrote)."""
    out_dir = Path(out_dir)
    log = out_dir / "driver.log"
    if not log.is_file():
        print("no driver.log")
        return 0
    d = parse_driver_log(log.read_text(errors="replace"))
    dumps = out_dir / "dumps"
    dumps.mkdir(parents=True, exist_ok=True)
    for label, rec in d.items():
        text = rec.get("dump_text", "")
        ptx = _ptx_from_dump(text)
        sass = _sass_from_dump(text)
        p = dumps / f"{label}.ptx"
        s = dumps / f"{label}.sass"
        if ptx and not p.exists():
            p.write_text(ptx)
        if sass and not s.exists():
            s.write_text(sass)
        print(label, "ptx" if ptx else "-", "sass" if sass else "-",
              "runtime_regs=%s" % rec.get("runtime_regs"), "error" if rec.get("error") else "")
    return 0


def write_tsv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def selftest():
    ptxas_sample = """ptxas info    : 0 bytes gmem
ptxas info    : Compiling entry function 'neighbors_checks_select_smallk_identical_candidate_abc123' for 'sm_90a'
ptxas info    : Function properties for neighbors_checks_select_smallk_identical_candidate_abc123
    368 bytes stack frame, 44 bytes spill stores, 52 bytes spill loads
ptxas info    : Used 96 registers, used 1 barriers, 368 bytes cumulative stack size, 2048 bytes smem, 400 bytes cmem[0]
"""
    recs = parse_ptxas_verbose(ptxas_sample)
    rec = recs["neighbors_checks_select_smallk_identical_candidate_abc123"]
    assert rec == {"stack_bytes": 368, "spill_stores": 44, "spill_loads": 52,
                   "registers": 96, "shared_bytes": 2048}, rec

    res_sample = """Fatbin elf code:
================
arch = sm_90a
code version = [1,7]
host = linux
compile_size = 64bit

Resource usage:
 Common:
  GLOBAL:0
 Function neighbors_checks_select_smallk_identical_candidate_abc123:
  REG:96 STACK:368 SHARED:2048 LOCAL:0 CONSTANT[0]:400 TEXTURE:0 SURFACE:0 SAMPLER:0
"""
    rec = parse_resource_usage(res_sample)["neighbors_checks_select_smallk_identical_candidate_abc123"]
    assert rec == {"registers": 96, "stack_bytes": 368, "shared_bytes": 2048, "local_bytes": 0}, rec

    ptx_sample = """//
// Generated by LLVM NVPTX Back-End
//

.version 8.5
.target sm_90a
.address_size 64

.visible .entry neighbors_checks_select_smallk_identical_candidate_abc123(
\t.param .u64 .ptr .align 1 p0
)
{
\t.local .align 8 .b8 \t__local_depot0[128];
\t.reg .b64 \t%SP;
\t.reg .pred \t%p<40>;
\t.shared .align 8 .b8 neighbors_checks_select_smallk_identical_candidate_abc123_$__global_alloc_0_$__gpu_shared_mem[2048];
\tmov.b32 \t%SPL, __local_depot0;
\tst.local.b64 \t[%r14], %rd5;
\tst.local.b64 \t[%r14+8], %rd6;
\tld.local.b64 \t%rd7, [%r14+8];
\tsetp.lt.u64 \t%p1, %rd7, %rd5;
\tret;
}
"""
    rec = parse_ptx(ptx_sample)["neighbors_checks_select_smallk_identical_candidate_abc123"]
    assert rec["target"] == "sm_90a" and rec["local_bytes"] == 128, rec
    assert rec["shared_bytes"] == 2048 and rec["ld_local"] == 1 and rec["st_local"] == 2, rec

    sass_sample = """\tcode for sm_90a
\t\tFunction : neighbors_checks_select_smallk_identical_candidate_abc123
\t\t.headerflags\t@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM90"
        /*0000*/                   LDC R1, c[0x0][0x28] ;
        /*0010*/                   STL [R1+0x8], R4 ;
        /*0020*/                   LDL R5, [R1+0x8] ;
        /*0030*/                   STL.64 [R1], R6 ;
        /*0040*/                   EXIT ;
"""
    rec = parse_sass(sass_sample)["neighbors_checks_select_smallk_identical_candidate_abc123"]
    assert rec == {"sass_ldl": 1, "sass_stl": 2, "sass_insns": 5}, rec
    rec = parse_sass(".text.kern_x:\n /*0000*/ STL [R1], R0 ;\n /*0010*/ EXIT ;\n")["kern_x"]
    assert rec["sass_stl"] == 1 and rec["sass_ldl"] == 0, rec

    driver_sample = ("KNN_KERNEL_STATS_BEGIN label=cap16_k10 cap=16 k=10\n"
                     + ptx_sample
                     + sass_sample
                     + "KNN_KERNEL_STATS label=cap16_k10 cap=16 k=10 regs=96 local=368 shared=2048 const=400 max_threads=1024 blocks_per_sm_256=2\n"
                     "KNN_KERNEL_STATS_BEGIN label=cap1_k1_scanonly1 cap=1 k=1\n"
                     "KNN_KERNEL_STATS_ERROR label=cap1_k1_scanonly1 error=compile_failed\n")
    d = parse_driver_log(driver_sample)
    assert d["cap16_k10"]["runtime_regs"] == 96 and d["cap16_k10"]["runtime_blocks_per_sm_256"] == 2, d
    assert ".version 8.5" in d["cap16_k10"]["dump_text"] and "Function : " in d["cap16_k10"]["dump_text"]
    assert "error" in d["cap1_k1_scanonly1"]

    rows = build_rows(d, {"cap16_k10": {"ptxas": ptxas_sample, "resources": res_sample},
                          "cap16_k0": {"ptxas": ptxas_sample.replace("96 registers", "200 registers"),
                                       "ptx": ptx_sample}})
    by = {r["label"]: r for r in rows}
    r = by["cap16_k10"]
    assert r["registers"] == 96 and r["spill_stores"] == 44 and r["spill_loads"] == 52, r
    assert r["ld_local"] == 1 and r["st_local"] == 2 and r["sass_ldl"] == 1 and r["sass_stl"] == 2, r
    assert r["local_bytes"] == 368 and r["shared_bytes"] == 2048, r
    assert r["blocks_per_sm_by_regs"] == 2 and r["blocks_per_sm_max"] == 2 and r["occupancy_pct"] == 25.0, r
    assert r["runtime_blocks_per_sm"] == 2 and r["runtime_occupancy_pct"] == 25.0, r
    assert r["sources"] == "runtime+ptxas+cuobjdump+ptx+sass", r["sources"]
    g = by["cap16_k0"]
    assert g["registers"] == 200 and g["blocks_per_sm_by_regs"] == 1 and g["occupancy_pct"] == 12.5, g
    assert g["local_bytes"] == 368 and g["ld_local"] == 1, g
    e = by["cap1_k1_scanonly1"]
    assert e["registers"] is None and e["sources"] == "driver-error", e

    # occupancy arithmetic at the thresholds the brief names
    assert occupancy(128, 2048)[2] == 2 and occupancy(128, 2048)[3] == 25.0
    assert occupancy(129, 2048)[0] == 1
    assert occupancy(32, 2048)[2] == 8 and occupancy(32, 2048)[3] == 100.0
    assert occupancy(64, 2048)[2] == 4
    assert occupancy(255, 0)[0] == 1

    blobs = extract_ptx_blobs(b"junk\x00" + ptx_sample.encode() + b"\x00more\x00" + ptx_sample.encode())
    assert len(blobs) == 2 and ".entry" in blobs[0] and "more" not in blobs[0], len(blobs)
    print("knn_selector_kernel_stats selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", help="the leg's knn-kernel-stats directory (driver.log, dumps/)")
    ap.add_argument("--tsv", help="where to write stats.tsv (default <out>/stats.tsv)")
    ap.add_argument("--json", help="also write the rows as JSON here")
    ap.add_argument("--extract-ptx", nargs=2, metavar=("BINARY", "DIR"),
                    help="write every NVPTX text blob found in BINARY to DIR/blob_<n>.ptx and exit")
    ap.add_argument("--split-driver-log", metavar="OUT",
                    help="cut per-label PTX / SASS out of OUT/driver.log into OUT/dumps/ and exit")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if args.split_driver_log:
        return split_driver_log(args.split_driver_log)
    if args.extract_ptx:
        binary, out = args.extract_ptx
        blobs = extract_ptx_blobs(Path(binary).read_bytes())
        Path(out).mkdir(parents=True, exist_ok=True)
        for i, b in enumerate(blobs):
            (Path(out) / f"blob_{i:03d}.ptx").write_text(b)
        print(f"{len(blobs)} PTX blobs from {binary}")
        return 0
    if not args.out:
        ap.error("--out is required (or --selftest / --extract-ptx)")
    driver, per_label = gather(args.out)
    rows = build_rows(driver, per_label)
    tsv = args.tsv or str(Path(args.out) / "stats.tsv")
    write_tsv(rows, tsv)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print("\t".join("" if r[c] is None else str(r[c]) for c in COLUMNS))
    return 0


if __name__ == "__main__":
    sys.exit(main())
