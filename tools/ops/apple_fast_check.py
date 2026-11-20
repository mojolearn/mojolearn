#!/usr/bin/env python3
"""tools/ops/apple_fast_check.py -- the step bodies of .github/workflows/apple-fast-check.yml that are not a parse.

Does the Apple `fast` set (the legacy fast build: no -D MOJOLEARN_NUMERIC_IDENTICAL, the `afn_*` routes) and the Apple
`identical` set still compile and start? The parse of one file is `tools/ops/native_checks_steps.sh parse-group` with
KIND apple-fast | apple-identical | apple-host (one copy of the parse logic, shared with native-checks.yml); this file
holds the plan, the real build of one merged file, placing built files into a package tree, the smoke and the summary.

  plan --sets fast,identical[,host] [--units-per-job N]
        JSON list, one entry per job: set, kind, stem, cls, key, members. The Darwin plan of
        tools/merged_bindings.py with the byte LM in (bindings/merged_build.sh's macOS default). N > 0 deals a
        file's units into parts of at most N (a parse of a part covers its units' code; the build never splits).
  build-group --set S --stem STEM --out DIR [--accel A] [--limit SECONDS]
        One merged file through bindings/merged_build.sh, the command packaging/macos/build_release_wheel.sh runs
        for that pair (MOJOLEARN_NUMERIC_MODE=<set> sh bindings/merged_build.sh gpu --group <stem>; the host set
        with the CPU column), every core of the runner (MOJOLEARN_COMPILE_JOBS). Writes DIR/pkg/<tier dir>/<stem>.so,
        DIR/<key>.log, DIR/result-<key>.txt. --accel names the accelerator of a `default` file (a hosted Mac is a VM
        whose GPU the compiler may not know); `native` passes nothing, which is the release line.
  place --artifacts DIR --pkg python/mojolearn
        Copy every DIR/**/pkg/**/<stem>.so into the package tree and write each directory's merged index for the
        files present. Prints the planned files that are missing.
  smoke --out DIR [--timeout SECONDS]
        With mojolearn INSTALLED in the running interpreter: record the Metal devices macOS reports, then each case
        (import, tree, linear, kmeans, mlp, transformer) x numeric_mode (fast, identical), one process per cell with
        a timeout, all cells at once across the runner's cores. DIR/smoke.json, DIR/SMOKE.md, DIR/logs/<cell>.log.
        A cell is ran, refused (a Python exception, with its message), crashed (the process died) or timeout.
        Exit 0 always: this reports, it gates nothing.
  summary --results DIR --out SUMMARY.md
        Merge every result-*.txt, flags-*.txt and SMOKE.md under DIR into one table per step, with the first
        errors of each failed file.

Tooling only. Nothing here is product code and nothing here reports to a release.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import time

SETS = ("fast", "identical", "host")
KIND = {"fast": "apple-fast", "identical": "apple-identical", "host": "apple-host"}
#: where a set's files live in the package (python/mojolearn/_backend.py loads `fast` from the package directory)
PKG_DIR = {"fast": "", "identical": "identical", "host": "host"}


def repo():
    return pathlib.Path.cwd()


def merged():
    """tools/merged_bindings.py of the checkout under test (the working directory)."""
    sys.path.insert(0, str(repo() / "tools"))
    import merged_bindings
    return merged_bindings


def plan_rows(sets, units_per_job=0):
    m = merged()
    rows = []
    for name in sets:
        if name not in SETS:
            raise SystemExit(f"unknown set {name!r}: {', '.join(SETS)}")
        for stem, cls, members in m.plan(name, "darwin", True, repo=repo()):
            parts = 1 if units_per_job <= 0 else -(-len(members) // units_per_job)
            for i in range(parts):
                key = f"{name}-{stem}" if parts == 1 else f"{name}-{stem}_p{i + 1}of{parts}"
                rows.append(dict(set=name, kind=KIND[name], stem=stem, cls=cls, key=key,
                                 members=",".join(members[i::parts])))
    return rows


def cmd_plan(args):
    print(json.dumps(plan_rows([s for s in args.sets.split(",") if s], args.units_per_job)))


def cores():
    return os.cpu_count() or 2


def run_limited(argv, limit, log, env=None, cwd=None):
    """(rc, seconds): rc 124 when the limit killed it (its whole process group: TERM, 60 s, KILL)."""
    start = time.time()
    with open(log, "wb") as out:
        p = subprocess.Popen(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, env=env, cwd=cwd)
        try:
            rc = p.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
            rc = 124
    return rc, int(time.time() - start)


def first_errors(log, limit=8):
    try:
        text = pathlib.Path(log).read_text(errors="replace")
    except OSError:
        return []
    return [line.strip()[:300] for line in text.splitlines() if "error:" in line][:limit]


def cmd_build_group(args):
    rows = [r for r in plan_rows([args.set]) if r["stem"] == args.stem]
    if not rows:
        raise SystemExit(f"no file {args.stem} in the Darwin plan of {args.set}")
    row = rows[0]
    out = pathlib.Path(args.out).resolve()
    outdir = out / "pkg" / PKG_DIR[args.set]
    outdir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # exactly the variables packaging/macos/build_release_wheel.sh passes for this pair
    for name in ("MOJOLEARN_GPU_ARCHS", "MOJOLEARN_HOST_OUTDIR", "MACOSX_DEPLOYMENT_TARGET"):
        env.pop(name, None)
    env["MOJOLEARN_PACKAGE_BYTE_LM"] = "1"
    env["MOJOLEARN_COMPILE_JOBS"] = str(cores())  # every core of the runner (the Mac default is 2)
    if args.set == "host":
        env.update(MOJOLEARN_NUMERIC_MODE="identical", MOJOLEARN_TARGET_COLUMN="cpu")
        kind = "host"
    else:
        env["MOJOLEARN_NUMERIC_MODE"] = args.set
        kind = "gpu"
    accel = "none"
    if row["cls"] == "metal1":
        accel = "metal:1 (the file's own flags class)"
    elif row["cls"] == "default" and args.accel != "native":
        # merged_build.sh passes MOJOLEARN_MOJO_BUILD_FLAGS to the one compile of this file
        env["MOJOLEARN_MOJO_BUILD_FLAGS"] = (env.get("MOJOLEARN_MOJO_BUILD_FLAGS", "") +
                                             f" --target-accelerator {args.accel}").strip()
        accel = args.accel
    elif row["cls"] == "default":
        accel = "native (no flag: the compiler reads this machine's GPU)"
    log = out / f"{row['key']}.log"
    rc, seconds = run_limited(["sh", "bindings/merged_build.sh", kind, "--group", args.stem, "--outdir", str(outdir)],
                              args.limit, log, env=env)
    so = outdir / f"{args.stem}.so"
    size = so.stat().st_size if so.exists() else 0
    # Does the file hold compiled Metal kernels? The release's own static check, reported here and not a gate:
    # a hosted runner is known to build files with none (packaging/macos/check_gpu_embedded.py's header).
    kernels = "not-checked"
    if rc == 0 and so.exists() and args.set != "host":
        check = subprocess.run([sys.executable, "packaging/macos/check_gpu_embedded.py", str(so)],
                               capture_output=True, text=True)
        (out / f"gpu-embedded-{row['key']}.txt").write_text(check.stdout + check.stderr)
        kernels = "present" if check.returncode == 0 else "REFUSED-by-check_gpu_embedded"
    status = "pass" if rc == 0 and size else ("timeout" if rc == 124 else "compile")
    line = (f"key={row['key']} set={args.set} stem={args.stem} cls={row['cls']} kind={status} rc={rc} "
            f"seconds={seconds} bytes={size} kernels={kernels} jobs={env['MOJOLEARN_COMPILE_JOBS']}")
    (out / f"result-{row['key']}.txt").write_text(line + "\n")
    (out / f"flags-{row['key']}.txt").write_text(
        f"build {row['key']}: MOJOLEARN_NUMERIC_MODE={env['MOJOLEARN_NUMERIC_MODE']} sh bindings/merged_build.sh "
        f"{kind} --group {args.stem}; accelerator: {accel}\n")
    print(line)
    for err in first_errors(log):
        print("  " + err)
    print(subprocess.run(["tail", "-n", "5", str(log)], capture_output=True, text=True).stdout)
    return 0 if status == "pass" else 1


def cmd_place(args):
    m = merged()
    pkg = pathlib.Path(args.pkg)
    found = {}
    for so in pathlib.Path(args.artifacts).rglob("*.so"):
        parts = so.parts
        if "pkg" not in parts:
            continue
        rel = pathlib.Path(*parts[len(parts) - parts[::-1].index("pkg"):])  # the path under the last .../pkg/
        found[str(rel)] = so
    missing = 0
    for name in SETS:
        dpath = pkg / PKG_DIR[name]
        dpath.mkdir(parents=True, exist_ok=True)
        index = dpath / "_mojolearn_merged.json"
        if index.exists():
            index.unlink()
        for stem, _cls, members in m.plan(name, "darwin", True, repo=repo()):
            rel = str(pathlib.Path(PKG_DIR[name]) / f"{stem}.so")
            if rel not in found:
                print(f"MISSING {name}/{stem}.so (not built): its members cannot load: {','.join(members)}")
                missing += 1
                continue
            shutil.copy2(found[rel], dpath / f"{stem}.so")
            subprocess.run([sys.executable, "tools/merged_bindings.py", "index", "--outdir", str(dpath),
                            "--stem", stem, "--members", ",".join(members)], check=True, stdin=subprocess.DEVNULL)
            print(f"placed {name}/{stem}.so ({len(members)} members)")
    print(f"placed {len(found)} file(s); {missing} planned file(s) missing")
    return 0


# ----------------------------------------------------------------------------------------------------------- smoke
SMOKE_CASES = ("import", "tree", "linear", "kmeans", "mlp", "transformer")
SMOKE_MODES = ("fast", "identical")
#: one cell: argv = case, mode. The last stdout line that starts with SMOKE is the cell's answer.
SMOKE_CHILD = r'''
import hashlib, json, sys
case, mode = sys.argv[1], sys.argv[2]
def answer(status, **fields):
    print("SMOKE " + json.dumps(dict(status=status, **fields)), flush=True)
def digest(value):
    import numpy as np
    value = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(value.tobytes() + str(value.dtype).encode() + str(value.shape).encode()).hexdigest()[:16]
try:
    import numpy as np
    import mojolearn
    X = (np.arange(2048, dtype=np.float32).reshape(256, 8) * np.float32(0.37)) % np.float32(5.0) + np.float32(0.25)
    y_reg = np.ascontiguousarray(X[:, 0] * np.float32(2.0) - X[:, 1], dtype=np.float32)
    y_cls = np.ascontiguousarray((X[:, 0] > X[:, 1]).astype(np.int32))
    if case == "import":
        mojolearn.set_numeric_mode(mode)
        from mojolearn import _backend
        answer("ran", detail=dict(version=getattr(mojolearn, "__version__", "?"), numeric_mode=_backend.numeric_mode(),
                                  vendor=_backend.vendor(), file=mojolearn.__file__))
    elif case == "tree":
        model = mojolearn.RandomForestClassifier(n_estimators=4, random_state=0, numeric_mode=mode).fit(X, y_cls)
        answer("ran", digest=digest(model.predict(X)))
    elif case == "linear":
        model = mojolearn.Ridge(numeric_mode=mode).fit(X, y_reg)
        answer("ran", digest=digest(model.predict(X)))
    elif case == "kmeans":
        model = mojolearn.KMeans(n_clusters=3, random_state=0, numeric_mode=mode).fit(X)
        answer("ran", digest=digest(model.predict(X)))
    elif case == "mlp":
        model = mojolearn.MLPClassifier(random_state=0, max_iter=20, numeric_mode=mode).fit(X, y_cls)
        answer("ran", digest=digest(model.predict(X)))
    elif case == "transformer":
        dm, nh, nkv, hd, inter = 32, 2, 1, 16, 64
        shapes = {"input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
                  "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm), "v_proj.weight": (nkv * hd, dm),
                  "o_proj.weight": (dm, nh * hd), "gate_proj.weight": (inter, dm), "up_proj.weight": (inter, dm),
                  "down_proj.weight": (dm, inter)}
        weights = {}
        for i, (name, shape) in enumerate(sorted(shapes.items())):
            n = int(np.prod(shape))
            ramp = ((np.arange(n, dtype=np.float32) + np.float32(7 * i)) % np.float32(17.0) - np.float32(8.0)) / np.float32(64.0)
            weights[name] = np.ones(shape, dtype=np.float32) if name.endswith("layernorm.weight") else ramp.reshape(shape)
        block = mojolearn.TransformerBlock(weights, n_heads=nh, n_kv_heads=nkv, numeric_mode=mode)
        tokens = ((np.arange(2 * 8 * dm, dtype=np.float32) % np.float32(13.0)) - np.float32(6.0)) / np.float32(64.0)
        answer("ran", digest=digest(block.forward(np.ascontiguousarray(tokens.reshape(2, 8, dm)))))
    else:
        answer("refused", message="unknown case " + case)
except BaseException as error:  # a refusal is an answer; a crash never reaches this line
    answer("refused", message=(type(error).__name__ + ": " + str(error))[:500])
'''


def metal_devices():
    """What macOS reports for the display adapters (the Metal device of a hosted runner is paravirtual)."""
    try:
        raw = subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"], capture_output=True, text=True,
                             timeout=120).stdout
        return [dict(name=d.get("sppci_model") or d.get("_name"), metal=d.get("spdisplays_mtlgpufamilysupport"),
                     vendor=d.get("spdisplays_vendor"), cores=d.get("sppci_cores"))
                for d in json.loads(raw).get("SPDisplaysDataType", [])]
    except Exception as error:  # not a Mac, or no profiler
        return [dict(name=f"unavailable: {type(error).__name__}: {error}"[:200])]


def smoke_cell(case, mode, out, timeout):
    work = out / "cwd" / f"{case}-{mode}"   # an empty directory: the INSTALLED package is what imports
    work.mkdir(parents=True, exist_ok=True)
    log = out / "logs" / f"{case}-{mode}.log"
    rc, seconds = run_limited([sys.executable, "-c", SMOKE_CHILD, case, mode], timeout, log, cwd=work)
    text = log.read_text(errors="replace")
    answers = [line for line in text.splitlines() if line.startswith("SMOKE ")]
    cell = dict(case=case, mode=mode, rc=rc, seconds=seconds)
    if rc == 124:
        cell.update(status="timeout", message=f"no answer in {timeout} s")
    elif answers and rc == 0:
        cell.update(json.loads(answers[-1][6:]))
    else:
        # the process died (a signal is a negative rc) or exited without an answer
        tail = " | ".join(line.strip() for line in text.splitlines()[-4:])[:500]
        sig = f"signal {signal.Signals(-rc).name}" if rc < 0 and -rc in set(signal.Signals) else f"rc {rc}"
        cell.update(status="crashed", message=f"{sig}: {tail}")
    return cell


def cmd_smoke(args):
    out = pathlib.Path(args.out).resolve()
    (out / "logs").mkdir(parents=True, exist_ok=True)
    devices = metal_devices()
    cells = [(case, mode) for case in SMOKE_CASES for mode in SMOKE_MODES]
    # independent processes, all cores; each cell is tiny, so memory does not bound the worker count
    with concurrent.futures.ThreadPoolExecutor(max_workers=cores()) as pool:
        results = list(pool.map(lambda cell: smoke_cell(cell[0], cell[1], out, args.timeout), cells))
    (out / "smoke.json").write_text(json.dumps(dict(devices=devices, cells=results), indent=1) + "\n")
    lines = ["## Smoke (installed package, one process per cell)", "",
             "Metal devices macOS reports: " + (json.dumps(devices) if devices else "none"), "",
             "| case | numeric_mode | status | seconds | digest / message |", "|---|---|---|---|---|"]
    for cell in results:
        what = cell.get("digest") or cell.get("message") or json.dumps(cell.get("detail", ""))
        lines.append(f"| {cell['case']} | {cell['mode']} | {cell['status']} | {cell['seconds']} | "
                     f"{str(what).replace('|', '/')} |")
    by = {}
    for cell in results:
        by[cell["status"]] = by.get(cell["status"], 0) + 1
    lines += ["", "cells: " + ", ".join(f"{n} {s}" for s, n in sorted(by.items())), "",
              "`ran` with equal digests in both modes means the two names ran the same kernels on this input or "
              "kernels that agree on it; it is not an identity check and not a timing."]
    (out / "SMOKE.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


# --------------------------------------------------------------------------------------------------------- summary
def cmd_summary(args):
    root = pathlib.Path(args.results)
    rows = []
    for f in sorted(root.rglob("result-*.txt")):
        fields = dict(x.split("=", 1) for x in f.read_text().split() if "=" in x)
        fields["_dir"] = f.parent
        fields["_step"] = "build" if "bytes" in fields else "parse"
        rows.append(fields)
    gib = lambda v: f"{int(v) / 2 ** 30:.1f}" if str(v).isdigit() else "-"
    lines = [f"# Apple fast check: {args.title}", "",
             "Tooling only: this gates and reports nothing to a release. A parse is parse, type check and "
             "elaboration (`mojo build --emit llvm --no-optimization`): no Metal library, no link, nothing run.", ""]
    for step in ("parse", "build"):
        mine = [r for r in rows if r["_step"] == step]
        if not mine:
            continue
        bad = [r for r in mine if r.get("kind") != "pass"]
        lines += [f"## {step}: {len(mine) - len(bad)} of {len(mine)} passed", ""]
        if step == "parse":
            lines += ["| file | result | rc | seconds | RAM peak GiB | swap peak GiB | Apple GPU target selected |",
                      "|---|---|---|---|---|---|---|"]
            lines += [f"| {r.get('key')} | {r.get('kind')} | {r.get('rc')} | {r.get('seconds')} | "
                      f"{gib(r.get('host_used_peak_bytes'))} | {gib(r.get('swap_used_peak_bytes'))} | "
                      f"{r.get('apple_target', '-')} |" for r in mine]
        else:
            lines += ["| file | result | rc | seconds | bytes | Metal kernels in the file | -j |",
                      "|---|---|---|---|---|---|---|"]
            lines += [f"| {r.get('key')} | {r.get('kind')} | {r.get('rc')} | {r.get('seconds')} | {r.get('bytes')} | "
                      f"{r.get('kernels')} | {r.get('jobs')} |" for r in mine]
        lines.append("")
        for r in bad:
            log = r["_dir"] / f"{r.get('key')}.log"
            errors = first_errors(log, 12)
            lines += [f"### {r.get('key')}: {r.get('kind')} (rc {r.get('rc')})", "", "```"]
            lines += errors or ["(no `error:` line; see the log's tail in the artifact)"]
            lines += ["```", ""]
    planned = pathlib.Path(args.planned) if args.planned else None
    if planned and planned.exists():
        keys = {(r["_step"], r.get("key")) for r in rows}
        for entry in json.loads(planned.read_text()):
            if (entry["step"], entry["key"]) not in keys:
                lines.append(f"- MISSING {entry['step']} {entry['key']}: its job left no result (runner lost, "
                             f"cancelled or timed out)")
                rows.append(dict(kind="missing"))
        lines.append("")
    flags = sorted({re.sub(r"^\S+ \S+ ", "", f.read_text().strip()) for f in root.rglob("flags-*.txt")})
    if flags:
        lines += ["## Compile lines seen", ""] + [f"- `{x}`" for x in flags[:40]] + [""]
    for smoke in sorted(root.rglob("SMOKE.md")):
        lines += [smoke.read_text(), ""]
    for note in sorted(root.rglob("NOTE-*.txt")):
        lines += [f"- {note.read_text().strip()}"]
    pathlib.Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:80]))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan"); p.add_argument("--sets", default="fast,identical")
    p.add_argument("--units-per-job", type=int, default=0); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("build-group"); p.add_argument("--set", required=True, choices=SETS)
    p.add_argument("--stem", required=True); p.add_argument("--out", required=True)
    p.add_argument("--accel", default="native"); p.add_argument("--limit", type=int, default=5 * 3600)
    p.set_defaults(fn=cmd_build_group)
    p = sub.add_parser("place"); p.add_argument("--artifacts", required=True); p.add_argument("--pkg", required=True)
    p.set_defaults(fn=cmd_place)
    p = sub.add_parser("smoke"); p.add_argument("--out", required=True)
    p.add_argument("--timeout", type=int, default=300); p.set_defaults(fn=cmd_smoke)
    p = sub.add_parser("summary"); p.add_argument("--results", required=True); p.add_argument("--out", required=True)
    p.add_argument("--planned", default=""); p.add_argument("--title", default=""); p.set_defaults(fn=cmd_summary)
    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
