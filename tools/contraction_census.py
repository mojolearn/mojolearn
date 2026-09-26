#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Where does the compiler's contraction choice reach IDENTICAL arithmetic?

WHY THIS EXISTS. `mojo build` fuses `a*b + c` into one fma by default
(`--fp-mode contract=fast`). IDENTICAL arithmetic must not depend on that
choice: an expression the default build fuses and a `contract=off` build does
not is an expression whose bits rest on every backend's LLVM making the same
choice (lane/fp-contract-off-study: 16 of 627 cells moved; lane/explicit-fma-
contract-proof and lane/pinned-mul-contract-free rewrote the sites). This tool
finds such sites WITHOUT a GPU: it builds each binding's assembly twice, with
and without contraction, and compares the fused multiply-adds per function.

    python3 tools/contraction_census.py build --out DIR --target T --mode M
        [--only build_gbdt.sh,gbdt] [--cache-dir D]
        T: host (this Mac, arm64), host-x86 (x86-64-v3 Linux, cross),
           nvidia (sm_90a PTX, cross), amd (gfx942, cross);
        M: default or off (`--fp-mode contract=off`).
        Runs the REAL bindings/build_*.sh scripts with a `pixi` shim on PATH
        that turns their one `mojo build` into `--emit asm` into
        DIR/T/M/<binding>/ and stops the script there. Host targets build the
        host families (python/mojolearn/host_surface.py --wheel-families);
        GPU targets the device bindings. ONE compile at a time, -j 1, in a
        PRIVATE Mojo cache (--cache-dir, default DIR/.mojo_cache_private,
        removed when the build ends): an `--emit asm` run must never write the
        shared compile cache (a probe once poisoned it with text entries).
    python3 tools/contraction_census.py compare DIR/T/default DIR/T/off [--json F]
        Per function (host asm) or kernel (PTX / AMDGCN sidecars): the count
        of fused multiply-adds in each build. A function whose counts differ
        is a CONTRACTION SITE. For PTX it also counts, in the default build,
        plain `mul.f32/f64` results consumed by a plain `add/sub`: ptxas may
        fuse those on its own, so they are sites too. Exit 1 if any.
    python3 tools/contraction_census.py --self-test

A clean census is a static check of the claim "IDENTICAL arithmetic does not
depend on the contraction mode" on the targets it builds; Metal is not among
them (the Apple compiler runs in the driver), so the Metal half of the claim
is the contract=off column comparison (docs/CONTRACTION.md).
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

DEVICE_SCRIPTS = [
    "build_mamba.sh", "build_hdbscan.sh", "build_transformer.sh",
    "build_kernel_methods.sh", "build_ivf.sh", "build_gbdt.sh",
    "build_mixture.sh", "build_resample.sh", "build_rf.sh", "build_trees.sh",
    "build_training.sh", "build_embedding.sh", "build.sh",
    "build_estimators.sh", "build_svm.sh", "build_solver.sh",
    "build_metrics.sh", "build_preprocessing.sh", "build_tsa.sh",
    "build_linalg.sh", "build_arima.sh", "build_gp.sh", "build_byte_lm.sh",
]

TARGETS = {
    "host": {"column": "cpu", "accel": None, "x86": False},
    "host-x86": {"column": "cpu", "accel": None, "x86": True},
    "nvidia": {"column": "nvidia", "accel": "sm_90a", "x86": False},
    "amd": {"column": "amd", "accel": "gfx942", "x86": False},
}

SHIM_EXIT = 97

# --- the shim: one `mojo build` becomes `--emit asm` into the census dir ----


def shim_main(argv):
    """Called as `pixi run mojo build ...` by a bindings/build_*.sh script."""
    out_dir = Path(os.environ["MOJOLEARN_CENSUS_OUT"])
    real_pixi = os.environ["MOJOLEARN_CENSUS_REAL_PIXI"]
    accel = os.environ.get("MOJOLEARN_CENSUS_ACCEL", "")
    x86 = os.environ.get("MOJOLEARN_CENSUS_X86", "") == "1"
    column = os.environ.get("MOJOLEARN_CENSUS_COLUMN", "")
    args = list(argv)
    out = []
    stem = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-j":
            i += 2
            continue
        if a == "--emit":
            i += 2
            continue
        if a == "-o":
            stem = Path(args[i + 1]).stem
            i += 2
            continue
        if a == "-Xlinker":
            i += 2
            continue
        if a == "--target-cpu" and x86:
            i += 2
            continue
        if a == "--target-accelerator":
            i += 2
            continue
        if a == "-D" and i + 1 < len(args) and args[i + 1].startswith("MOJOLEARN_COLUMN_") and column:
            out += ["-D", column]
            i += 2
            continue
        out.append(a)
        i += 1
    if stem is None:
        print("contraction_census shim: no -o in the mojo build line", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [real_pixi, "run", "mojo", "build", "-j", "1", "--emit", "asm"]
    if x86:
        cmd += ["--target-triple", "x86_64-unknown-linux-gnu", "--target-cpu", "x86-64-v3"]
    if accel:
        cmd += ["--target-accelerator", accel]
    if os.environ.get("MOJOLEARN_CENSUS_DEBUG", "") == "1":
        cmd += ["--debug-level", "line-tables"]
    cmd += out + ["-o", str(out_dir / (stem + ".s"))]
    (out_dir / "command.txt").write_text(" ".join(cmd) + "\n")
    rc = subprocess.call(cmd)
    if rc != 0:
        return rc
    (out_dir / "OK").write_text("ok\n")
    return SHIM_EXIT


def write_shim(bin_dir: Path):
    bin_dir.mkdir(parents=True, exist_ok=True)
    shim = bin_dir / "pixi"
    shim.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = run ] && [ "$2" = mojo ] && [ "$3" = build ]; then\n'
        "    shift 3\n"
        f'    exec python3 "{Path(__file__).resolve()}" _shim "$@"\n'
        "fi\n"
        'exec "$MOJOLEARN_CENSUS_REAL_PIXI" "$@"\n'
    )
    shim.chmod(0o755)


def host_families():
    out = subprocess.check_output(
        [sys.executable, str(REPO / "python/mojolearn/host_surface.py"), "--wheel-families"],
        text=True, cwd=REPO,
    )
    return out.split()


def cmd_build(ns):
    target = TARGETS[ns.target]
    real_pixi = shutil.which("pixi")
    if not real_pixi:
        print("contraction_census: no pixi on PATH", file=sys.stderr)
        return 2
    root = Path(ns.out).resolve() / ns.target / (ns.mode + ("-g" if ns.debug else ""))
    root.mkdir(parents=True, exist_ok=True)
    cache = Path(ns.cache_dir).resolve() if ns.cache_dir else Path(ns.out).resolve() / ".mojo_cache_private"
    cache.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="contraction-census-"))
    write_shim(work / "bin")
    if ns.target.startswith("host"):
        jobs = [(f"build_{f}_host.sh", f) for f in host_families()]
    else:
        jobs = [(s, s[:-3]) for s in DEVICE_SCRIPTS]
    if ns.only:
        keep = set(ns.only.split(","))
        jobs = [j for j in jobs if j[0] in keep or j[1] in keep]
    summary = []
    try:
        for script, name in jobs:
            dest = root / name
            if (dest / "OK").exists() and not ns.force:
                summary.append((name, "cached"))
                continue
            if dest.exists():
                shutil.rmtree(dest)
            env = dict(os.environ)
            for k in ("MOJOLEARN_GPU_ARCHS", "MOJOLEARN_HOST_OUTDIR", "MOJOLEARN_BYTE_LM_OUTDIR"):
                env.pop(k, None)
            env.update({
                "PATH": f"{work / 'bin'}:{env['PATH']}",
                "MOJOLEARN_CENSUS_OUT": str(dest),
                "MOJOLEARN_CENSUS_REAL_PIXI": real_pixi,
                "MOJOLEARN_CENSUS_ACCEL": target["accel"] or "",
                "MOJOLEARN_CENSUS_X86": "1" if target["x86"] else "",
                "MODULAR_CACHE_DIR": str(cache),
                "MOJOLEARN_COMPILE_JOBS": "1",
                "MOJOLEARN_NUMERIC_MODE": "identical",
                "MOJOLEARN_SKIP_BUILD_GATE": "1",
                "MOJOLEARN_TARGET_COLUMN": target["column"],
                "MOJOLEARN_MOJO_BUILD_FLAGS": "--fp-mode contract=off" if ns.mode == "off" else "",
                "MOJOLEARN_HOST_OUTDIR": str(work / "host_out" / name),
                "MOJOLEARN_BYTE_LM_OUTDIR": str(work / "byte_lm_out" / name),
                "MOJOLEARN_CENSUS_COLUMN": "MOJOLEARN_COLUMN_" + target["column"].upper(),
                "MOJOLEARN_CENSUS_DEBUG": "1" if ns.debug else "",
            })
            if script == "build_byte_lm.sh" and sys.platform == "darwin":
                # The byte LM script accepts only the apple column on Darwin; the
                # shim swaps its column define for the census target's.
                env["MOJOLEARN_TARGET_COLUMN"] = "apple"
            log = root / f"{name}.log"
            with open(log, "w") as fh:
                rc = subprocess.call(["sh", f"bindings/{script}"], cwd=REPO, env=env,
                                     stdout=fh, stderr=subprocess.STDOUT)
            ok = (dest / "OK").exists()
            summary.append((name, "ok" if ok else f"FAILED rc={rc} (see {log})"))
            print(f"{ns.target}/{ns.mode} {name}: {summary[-1][1]}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
        if not ns.keep_cache:
            shutil.rmtree(cache, ignore_errors=True)
    bad = [s for s in summary if not s[1] in ("ok", "cached")]
    return 1 if bad else 0


# --- the comparison -------------------------------------------------------

HOST_FUSED = re.compile(
    r"^\s*(fmadd|fmsub|fnmadd|fnmsub|fmla|fmls|vfn?m(add|sub)(sub|add)?\d{3}[sp][sd])\b"
)
AMD_FUSED = re.compile(
    r"^\s*(v_fma_f(16|32|64)|v_fmac_f(16|32|64)|v_mad_f(16|32)|v_mac_f(16|32)|"
    r"v_pk_fma_f(16|32)|v_fmaak_f(16|32)|v_fmamk_f(16|32)|v_fma_mix\w*|v_mad_mix\w*|v_pk_fmac_f16)\b"
)
PTX_FUSED = re.compile(r"^\s*fma\.rn(\.ftz)?(\.sat)?\.(f32|f64|f16|bf16|f16x2|bf16x2|f32x2)\b")
PTX_ARITH = re.compile(
    r"^\s*(mul|add|sub)((?:\.r[nzmp])?)((?:\.ftz)?)((?:\.sat)?)\.(f32|f64)\s+(%\w+),\s*([^,;]+),\s*([^;]+);"
)
LABEL = re.compile(r'^("?[^\s"#;.][^"]*"?|"[^"]+"):\s*(//.*|;.*|#.*)?$')


def _functions(text, kind):
    """{function name: [instruction lines]} for one assembly file."""
    funcs = {}
    cur = None
    for line in text.splitlines():
        if kind == "ptx":
            m = re.match(r"^\s*(?:\.visible\s+|\.weak\s+)?\.(?:entry|func)\s+(?:\([^)]*\)\s*)?([\w$]+)", line)
            if m:
                cur = m.group(1)
                funcs.setdefault(cur, [])
                continue
        else:
            m = LABEL.match(line)
            if m:
                name = m.group(1).strip('"')
                if not re.match(r"^(L|\.L|l_|ltmp|Ltmp|LBB|\.LBB|lCPI|LCPI|\.Ltmp|\.Lfunc)", name):
                    cur = name
                    funcs.setdefault(cur, [])
                continue
        if cur is not None:
            funcs[cur].append(line)
    return funcs


def _count(lines, kind):
    rx = {"host": HOST_FUSED, "amd": AMD_FUSED, "ptx": PTX_FUSED}[kind]
    return sum(1 for l in lines if rx.match(l))


def _ptx_fusable_pairs(lines):
    """Plain (no rounding modifier) mul results consumed by a plain add/sub."""
    plain_mul = {}
    pairs = 0
    for l in lines:
        m = PTX_ARITH.match(l)
        if not m:
            continue
        op, rnd, _ftz, _sat, ty, dst, a, b = m.groups()
        a, b = a.strip(), b.strip()
        if op == "mul":
            if not rnd:
                plain_mul[dst] = ty
            else:
                plain_mul.pop(dst, None)
            continue
        if rnd:
            continue
        for src in (a, b):
            if plain_mul.get(src) == ty:
                pairs += 1
                break
    return pairs


def _files(root: Path):
    out = {}
    for p in sorted(root.rglob("*")):
        if p.suffix in (".s", ".ptx", ".amdgcn") and p.is_file():
            out[str(p.relative_to(root))] = p
    return out


def _kind(path: Path):
    return {".ptx": "ptx", ".amdgcn": "amd"}.get(path.suffix, "host")


def compare(dir_default: Path, dir_off: Path):
    fd, fo = _files(dir_default), _files(dir_off)
    sites = []
    missing = sorted(set(fd) ^ set(fo))
    totals = {"files": 0, "functions": 0, "fused_default": 0, "fused_off": 0, "ptx_plain_pairs": 0}
    for rel in sorted(set(fd) & set(fo)):
        kind = _kind(fd[rel])
        a = _functions(fd[rel].read_text(errors="replace"), kind)
        b = _functions(fo[rel].read_text(errors="replace"), kind)
        totals["files"] += 1
        for name in sorted(set(a) | set(b)):
            ca = _count(a.get(name, []), kind)
            cb = _count(b.get(name, []), kind)
            pp = _ptx_fusable_pairs(a.get(name, [])) if kind == "ptx" else 0
            totals["functions"] += 1
            totals["fused_default"] += ca
            totals["fused_off"] += cb
            totals["ptx_plain_pairs"] += pp
            if (name in a) != (name in b) and ca == 0 and cb == 0:
                continue  # a label present in one build only (a string constant's hash), no arithmetic
            if ca != cb or pp or (name in a) != (name in b):
                sites.append({"file": rel, "function": name, "fused_default": ca,
                              "fused_off": cb, "ptx_plain_pairs": pp,
                              "only_in": None if (name in a) == (name in b) else ("default" if name in a else "off")})
    return {"totals": totals, "sites": sites, "unpaired_files": missing}


def cmd_compare(ns):
    res = compare(Path(ns.default), Path(ns.off))
    t = res["totals"]
    print(f"files {t['files']}, functions {t['functions']}, fused ops default {t['fused_default']}, "
          f"off {t['fused_off']}, PTX plain mul->add pairs {t['ptx_plain_pairs']}")
    for s in res["sites"]:
        extra = f" ptx-plain-pairs={s['ptx_plain_pairs']}" if s["ptx_plain_pairs"] else ""
        only = f" only-in={s['only_in']}" if s["only_in"] else ""
        print(f"SITE {s['file']} :: {s['function'][:160]} default={s['fused_default']} off={s['fused_off']}{extra}{only}")
    for f in res["unpaired_files"]:
        print(f"UNPAIRED {f}")
    if ns.json:
        Path(ns.json).write_text(json.dumps(res, indent=1))
    print(f"verdict: {'CLEAN' if not res['sites'] and not res['unpaired_files'] else 'SITES'} "
          f"({len(res['sites'])} site(s), {len(res['unpaired_files'])} unpaired file(s))")
    return 0 if not res["sites"] and not res["unpaired_files"] else 1


# --- locate: the SOURCE LINES whose fused multiply-adds depend on the mode --
# (builds made with `build --debug`, which carry `.loc` line tables)

FILE_DIR = re.compile(r'^\s*\.file\s+(\d+)\s+"([^"]*)"(?:\s+"([^"]*)")?')
LOC_DIR = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)")


def _loc_counts(text, kind):
    """{(source file, line): fused ops} and {(file, line): PTX plain mul->add pairs, keyed at the add}."""
    rx = {"host": HOST_FUSED, "amd": AMD_FUSED, "ptx": PTX_FUSED}[kind]
    files = {}
    for line in text.splitlines():  # PTX puts its .file table at the END
        m = FILE_DIR.match(line)
        if m:
            n, a, b = m.groups()
            files[n] = os.path.join(a, b) if b else a
    cur = ("?", 0)
    fused = {}
    pairs = {}
    plain_mul = {}
    for line in text.splitlines():
        if FILE_DIR.match(line):
            continue
        m = LOC_DIR.match(line)
        if m:
            cur = (files.get(m.group(1), m.group(1)), int(m.group(2)))
            continue
        if rx.match(line):
            fused[cur] = fused.get(cur, 0) + 1
            continue
        if kind == "ptx":
            m = PTX_ARITH.match(line)
            if m:
                op, rnd, _f, _s, ty, dst, a, b = m.groups()
                if op == "mul":
                    if not rnd:
                        plain_mul[dst] = (ty, cur)
                    else:
                        plain_mul.pop(dst, None)
                elif not rnd:
                    for src in (a.strip(), b.strip()):
                        if src in plain_mul and plain_mul[src][0] == ty:
                            key = (cur, plain_mul[src][1])
                            pairs[key] = pairs.get(key, 0) + 1
                            break
            if re.match(r"^\s*(?:\.visible\s+)?\.(entry|func)\b", line):
                plain_mul = {}
    return fused, pairs


def _short(path):
    p = str(path)
    root = str(REPO) + "/"
    if p.startswith("./"):
        p = p[2:]
    if p.startswith(root):
        return p[len(root):]
    return p


def locate(dir_default: Path, dir_off: Path):
    fd, fo = _files(dir_default), _files(dir_off)
    lines = {}
    pairs = {}
    for rel in sorted(set(fd) & set(fo)):
        kind = _kind(fd[rel])
        a, pa = _loc_counts(fd[rel].read_text(errors="replace"), kind)
        b, _ = _loc_counts(fo[rel].read_text(errors="replace"), kind)
        binding = rel.split("/")[0]
        for key in set(a) | set(b):
            if a.get(key, 0) != b.get(key, 0):
                ent = lines.setdefault((_short(key[0]), key[1]), {"default": 0, "off": 0, "where": set()})
                ent["default"] += a.get(key, 0)
                ent["off"] += b.get(key, 0)
                ent["where"].add(f"{binding}:{kind}")
        for (at_add, at_mul), n in pa.items():
            ent = pairs.setdefault(((_short(at_add[0]), at_add[1]), (_short(at_mul[0]), at_mul[1])), {"n": 0, "where": set()})
            ent["n"] += n
            ent["where"].add(binding)
    return lines, pairs


def cmd_locate(ns):
    lines, pairs = locate(Path(ns.default), Path(ns.off))
    for (f, ln), e in sorted(lines.items()):
        print(f"LINE {f}:{ln} fused default={e['default']} off={e['off']} in {','.join(sorted(e['where']))}")
    for ((fa, la), (fm, lm)), e in sorted(pairs.items()):
        print(f"PTXPAIR add {fa}:{la} <- mul {fm}:{lm} x{e['n']} in {','.join(sorted(e['where']))}")
    if ns.json:
        Path(ns.json).write_text(json.dumps({
            "lines": [{"file": f, "line": ln, "default": e["default"], "off": e["off"], "where": sorted(e["where"])}
                      for (f, ln), e in sorted(lines.items())],
            "ptx_pairs": [{"add": list(k[0]), "mul": list(k[1]), "n": e["n"], "where": sorted(e["where"])}
                          for k, e in sorted(pairs.items())]}, indent=1))
    print(f"verdict: {len(lines)} source line(s) whose fused ops differ, {len(pairs)} PTX plain mul->add pair location(s)")
    return 0 if not lines and not pairs else 1


def self_test():
    arm_a = '"_f(x)":\n\tfmadd\ts0, s0, s1, s2\n\tret\n"_g(x)":\n\tfmul\ts0, s0, s1\n\tret\n'
    arm_b = '"_f(x)":\n\tfmul\ts0, s0, s1\n\tfadd\ts0, s0, s2\n\tret\n"_g(x)":\n\tfmul\ts0, s0, s1\n\tret\n'
    ptx = (".visible .entry k(\n.param .u64 p\n)\n{\n\tmul.f32 \t%r5, %r2, %r3;\n\tadd.f32 \t%r6, %r4, %r5;\n"
           "\tmul.rn.f32 \t%r7, %r2, %r3;\n\tadd.f32 \t%r8, %r4, %r7;\n\tfma.rn.f32 \t%r9, %r1, %r2, %r3;\n}\n")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "a" / "x").mkdir(parents=True)
        (d / "b" / "x").mkdir(parents=True)
        (d / "a/x/m.s").write_text(arm_a)
        (d / "b/x/m.s").write_text(arm_b)
        (d / "a/x/k.ptx").write_text(ptx)
        (d / "b/x/k.ptx").write_text(ptx)
        res = compare(d / "a", d / "b")
        got = sorted((s["function"], s["fused_default"], s["fused_off"], s["ptx_plain_pairs"]) for s in res["sites"])
        want = [("_f(x)", 1, 0, 0), ("k", 1, 1, 1)]
        assert got == want, got
        (d / "b/x/m.s").write_text(arm_a)
        (d / "a/x/k.ptx").write_text(ptx.replace("\tmul.f32 \t%r5", "\tmul.rn.f32 \t%r5"))
        (d / "b/x/k.ptx").write_text(ptx.replace("\tmul.f32 \t%r5", "\tmul.rn.f32 \t%r5"))
        res = compare(d / "a", d / "b")
        assert not res["sites"], res["sites"]
    print("contraction_census self-test: OK")
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "_shim":
        return shim_main(sys.argv[2:])
    if "--self-test" in sys.argv:
        return self_test()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--target", required=True, choices=sorted(TARGETS))
    b.add_argument("--mode", required=True, choices=["default", "off"])
    b.add_argument("--only", default="")
    b.add_argument("--cache-dir", default="")
    b.add_argument("--keep-cache", action="store_true")
    b.add_argument("--force", action="store_true")
    b.add_argument("--debug", action="store_true",
                   help="add --debug-level line-tables (into <mode>-g) so `locate` can name source lines")
    l = sub.add_parser("locate")
    l.add_argument("default")
    l.add_argument("off")
    l.add_argument("--json", default="")
    c = sub.add_parser("compare")
    c.add_argument("default")
    c.add_argument("off")
    c.add_argument("--json", default="")
    ns = ap.parse_args()
    return {"build": cmd_build, "compare": cmd_compare, "locate": cmd_locate}[ns.cmd](ns)


if __name__ == "__main__":
    sys.exit(main())
