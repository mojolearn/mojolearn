#!/usr/bin/env python3
"""DEVIATION 2702: read a GEMM register census (tools/gemm_kernel_census_leg.sh).

    python3 tools/gemm_census_split.py <census dir>

<census dir> is the leg's remote/gemm-census: `census.log` (the driver's
stdout, which carries PTX and SASS when the driver fell back to the Bool dump
form), `dumps/<label>.ptx[.gz]`, `dumps/<label>.sass[.gz]` and
`dumps/<label>.ptxas.log` when the Path form built. Per label it prints the
runtime's NUM_REGS, ptxas's registers and spills, the PTX memory-op mix, and
from the SASS: the instruction count, the LDL/STL count, the highest register
index anywhere, and the same inside the ACCUMULATE LOOP (the densest FFMA
cluster), which is what says whether 255 registers belong to the loop or to
the fold. Prints, never edits.
"""
import gzip
import re
import sys
from pathlib import Path

INSN = re.compile(r"/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P\d+\s+)?([A-Z][A-Z0-9_.]*)\s*([^;]*);")
REG = re.compile(r"\bR(\d+)\b")


def read(p):
    if str(p).endswith(".gz"):
        return gzip.open(p, "rt", errors="replace").read()
    return Path(p).read_text(errors="replace")


def first_sass_marker(text, start=0):
    c = [text.find(s, start) for s in ("\tcode for sm_", "code for sm_", ".text.", "Function : ", ".headerflags")]
    c = [x for x in c if x >= 0]
    return min(c) if c else len(text)


def split_stdout(text):
    """{label: {'runtime': line, 'ptx': str, 'sass': str, 'ptxas': str}} from census.log."""
    out = {}
    parts = re.split(r"^GEMM_CENSUS_BEGIN label=(\S+)\s*$", text, flags=re.M)
    for i in range(1, len(parts), 2):
        label, body = parts[i], parts[i + 1]
        rec = {}
        m = re.search(r"^GEMM_CENSUS label=\S+ (.*)$", body, re.M)
        rec["runtime"] = m.group(1) if m else ""
        j = body.find(".version")
        if j >= 0:
            k = first_sass_marker(body, j)
            rec["ptx"] = body[j:k]
            rec["sass"] = body[k:] if k < len(body) else ""
        else:
            rec["ptx"], rec["sass"] = "", ""
        rec["ptxas"] = "\n".join(l for l in body.splitlines() if "ptxas info" in l or "spill" in l.lower())
        out[label] = rec
    return out


def ptx_stats(ptx):
    ops = {}
    for m in re.finditer(r"^\s*((?:ld|st)\.(?:shared|global|local)[a-z0-9.]*)", ptx, re.M):
        ops[m.group(1)] = ops.get(m.group(1), 0) + 1
    shared = sum(int(x) for x in re.findall(r"__gpu_shared_mem\[(\d+)\]", ptx))
    local = re.search(r"__local_depot\d*\[(\d+)\]", ptx)
    align = re.search(r"\.shared \.align (\d+)", ptx)
    return {"shared_bytes": shared, "local_depot": int(local.group(1)) if local else 0,
            "shared_align": int(align.group(1)) if align else 0, "memops": ops}


def sass_stats(sass):
    insns = [(m.group(1), m.group(2)) for m in INSN.finditer(sass)]
    if not insns:
        return None
    def maxreg(seq):
        mx = -1
        for _, args in seq:
            for r in REG.findall(args):
                mx = max(mx, int(r))
        return mx
    def count(seq, prefix):
        return sum(1 for op, _ in seq if op.startswith(prefix))
    # The accumulate loop: the densest FFMA cluster (gaps of up to 48 non-FFMA
    # instructions are inside the cluster; the fold and the staging are outside).
    ffma_idx = [i for i, (op, _) in enumerate(insns) if op.startswith("FFMA")]
    best = (0, 0, 0)
    if ffma_idx:
        start = ffma_idx[0]; prev = start; n = 1
        for i in ffma_idx[1:]:
            if i - prev > 48:
                if n > best[0]:
                    best = (n, start, prev)
                start, n = i, 0
            n += 1; prev = i
        if n > best[0]:
            best = (n, start, prev)
    _, a, b = best
    loop = insns[a:b + 1]
    return {
        "insns": len(insns), "ffma": count(insns, "FFMA"), "fmul": count(insns, "FMUL"),
        "lds": count(insns, "LDS"), "sts": count(insns, "STS"), "ldl": count(insns, "LDL"),
        "stl": count(insns, "STL"), "ldg": count(insns, "LDG"), "maxreg": maxreg(insns),
        "loop_insns": len(loop), "loop_ffma": count(loop, "FFMA"), "loop_fmul": count(loop, "FMUL"),
        "loop_lds": count(loop, "LDS"), "loop_ldl": count(loop, "LDL"), "loop_stl": count(loop, "STL"),
        "loop_bar": count(loop, "BAR"), "loop_maxreg": maxreg(loop),
        "loop_lds128": sum(1 for op, _ in loop if op.startswith("LDS.128")),
        "loop_lds64": sum(1 for op, _ in loop if op.startswith("LDS.64")),
        "loop_lds32": sum(1 for op, _ in loop if op == "LDS" or op.startswith("LDS.32") or op.startswith("LDS.U")),
    }


def main(d):
    d = Path(d)
    recs = {}
    log = d / "census.log"
    if log.is_file():
        recs = split_stdout(read(log))
    dumps = d / "dumps"
    if dumps.is_dir():
        for p in sorted(dumps.iterdir()):
            name = p.name
            for suf, kind in ((".ptxas.log", "ptxas"), (".sass.gz", "sass"), (".sass", "sass"), (".ptx.gz", "ptx"), (".ptx", "ptx")):
                if name.endswith(suf):
                    label = name[: -len(suf)]
                    recs.setdefault(label, {})[kind] = read(p)
                    break
    for label, rec in recs.items():
        print("==", label, "|", rec.get("runtime", ""))
        px = rec.get("ptxas", "")
        if px:
            for l in px.splitlines():
                if "Used" in l or "spill" in l.lower():
                    print("   ptxas:", l.strip()[:140])
        if rec.get("ptx"):
            s = ptx_stats(rec["ptx"])
            print("   ptx: shared=%d align=%d local_depot=%d %s" % (
                s["shared_bytes"], s["shared_align"], s["local_depot"],
                " ".join("%s=%d" % kv for kv in sorted(s["memops"].items()))))
        if rec.get("sass"):
            s = sass_stats(rec["sass"])
            if s:
                print("   sass: insns=%d ffma=%d fmul=%d lds=%d sts=%d ldg=%d ldl=%d stl=%d maxreg=R%d" % (
                    s["insns"], s["ffma"], s["fmul"], s["lds"], s["sts"], s["ldg"], s["ldl"], s["stl"], s["maxreg"]))
                print("   loop: insns=%d ffma=%d fmul=%d lds=%d (128=%d 64=%d 32=%d) ldl=%d stl=%d bar=%d maxreg=R%d" % (
                    s["loop_insns"], s["loop_ffma"], s["loop_fmul"], s["loop_lds"], s["loop_lds128"],
                    s["loop_lds64"], s["loop_lds32"], s["loop_ldl"], s["loop_stl"], s["loop_bar"], s["loop_maxreg"]))
            else:
                print("   sass: present but no instructions parsed")
        else:
            print("   sass: none")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
