#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE ONE LANE CHECK OF THE ALGORITHM EXPANSION: GPU == CPU, BIT FOR BIT
(lane/algos-prep, 2026-09-27; docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).

    tools/algos_lane_check.sh <lane[,lane...]> [--sabotage <patch>] [--pass 1|2] [--fixtures f,g] [--out DIR]

On whatever box it runs -- a Linux NVIDIA (or AMD) pod, or a Mac with Metal --
for each named identity lane it

  1. DERIVES the bindings the lane runs, GPU and CPU, from the lane map the
     selector already keeps (tools/lane_select.py, `declared`: the bindings
     the lane's own doors resolve and the host bindings _backend routes them
     to). A lane with no host binding has no CPU arm and is REFUSED by name:
     the CPU is part of every proof.
  2. BUILDS each one that is missing or STALE: its stamp
     (tools/binding_stamps.py) must carry the digest of its source closure in
     THIS tree, so an edited kernel is rebuilt and never answered by the .so
     of the previous edit. A binding that does not build fails the check.
  3. FITS the lane on the GPU (`--require-backend cuda|hip|metal`) and on the
     CPU through the host bindings (`MOJOLEARN_VENDOR=cpu`, `--require-cpu`),
     once each (`--repeats 1`), every fixture unless `--fixtures` narrows it.
     A refused stage on either side is a failure (`--fail-on-refused`).
  4. DIFFS the two columns with `identity_break.py --diff --require-columns 2`
     over train, infer, model, batch and every property part both carry, and
     counts what was ACTUALLY compared. It prints AGREE only when the diff
     passed AND every fixture's train cell rested on two real hashes; a table
     where nothing met is NOTHING COMPARED, which is a failure.

With `--sabotage <patch>`: the clean check must AGREE; then `git apply` the
patch (a SOURCE edit -- a define-only arm reuses cached kernels), rebuild what
it made stale, and the check must DISAGREE (a real hash difference, not a
refusal or a crash); then `git apply -R` (never git checkout), rebuild, and it
must AGREE again. The patch is reversed on every exit path.

PER-SEAM PROOF (before the diff, on the clean tree): every driver listed in
tools/identity_lanes/<fragment>.checks must PASS, and each line's sabotage
patch (`<driver><TAB><patch>`) must make its driver FAIL and PASS again after
`git apply -R`. With `--pass 2` a fragment of these lanes with no .checks, or
a line with no patch, fails the check (pass 1 only notes it).

Exit 0 only when every verdict is the one required. The last line is always
`RESULT: PASS` or `RESULT: FAIL (<why>)`.
"""
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
PKG = ROOT / "python" / "mojolearn"
HOST_DIR = PKG / "host"
HARNESS = ROOT / "tools" / "identity_break.py"
#: The parts of a cell that are compared beside the train hash.
PARTS = ("infer", "model", "batch", "rlpair", "batchgrad", "batchscale", "ragged", "stepfull")
DISAGREEING = ("DIVERGENT", "MOVED", "RELOAD-MOVED", "BATCH_MOVED", "RLPAIR_MOVED")


class Fail(Exception):
    pass


def say(msg):
    print(f"[lane-check {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_backend():
    if sys.platform == "darwin":
        return "metal"
    # a tool on PATH is not a device (a ROCm VM can carry nvidia-smi): ask each
    # for a GPU it can see
    def sees(cmd, needle):
        if not shutil.which(cmd[0]):
            return False
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return r.returncode == 0 and needle in r.stdout
    if sees(["nvidia-smi", "-L"], "GPU "):
        return "cuda"
    if sees(["rocminfo"], "gfx"):
        return "hip"
    raise Fail("no GPU on this box (no nvidia-smi, no rocminfo, not macOS); the check needs one")


def gpu_arch():
    """This box's one GPU target (sm_NN from nvidia-smi's compute capability,
    gfxNNN from rocminfo), for the build scripts that need it named
    (bindings/build_byte_lm.sh on Linux); None on a Mac or when unreadable."""
    try:
        backend = gpu_backend()
    except Fail:
        return None
    try:
        if backend == "cuda":
            r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                               capture_output=True, text=True, timeout=60)
            cap = r.stdout.split()[0].strip() if r.returncode == 0 and r.stdout.split() else ""
            return "sm_" + cap.replace(".", "") if cap.replace(".", "").isdigit() else None
        if backend == "hip":
            r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60)
            for tok in r.stdout.split():
                if tok.startswith("gfx") and tok[3:].isalnum():
                    return tok
    except (OSError, subprocess.TimeoutExpired):
        return None
    return None


def load_harness():
    import importlib.util
    spec = importlib.util.spec_from_file_location("algos_lane_check_harness", HARNESS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------ bindings
def script_for(binding):
    if binding.endswith("_host"):
        return f"build_{binding[len('_mojolearn_'):-len('_host')]}_host.sh"
    return "build.sh" if binding == "_mojolearn" else f"build_{binding[len('_mojolearn_'):]}.sh"


def output_for(binding):
    return HOST_DIR / f"{binding}.so" if binding.endswith("_host") else PKG / "identical" / f"{binding}.so"


#: Of a lane's `ubiquitous` bindings, the ones a fit loads on every box.
BASE_BINDINGS = ("_mojolearn", "_mojolearn_forest_host", "_mojolearn_byte_lm_host")


def needed_bindings(lanes):
    """lane -> sorted bindings it runs (GPU and host), from the selector's
    derived map. Refuses a lane with no GPU binding or no host binding."""
    import lane_select
    _, why = lane_select.lane_sources()
    ROUTES = lane_select.host_routes()
    out = {}
    for lane in lanes:
        declared = set(why[lane]["declared"])
        # THE BINDINGS EVERY LANE LOADS (trees lane, 2026-09-27: built by hand
        # once per pod). The selector keeps them out of `declared` on purpose
        # (`ubiquitous`: reaching them says nothing about one lane), but a fit
        # still imports them: the base binding's buffer helpers and its CPU
        # route, and the forest and byte LM host loaders arm_env points at.
        ubiq = set(why[lane].get("ubiquitous", ()))
        declared |= {b for b in BASE_BINDINGS if b in ubiq}
        declared.add("_mojolearn_core_host")     # the base binding's CPU route, on every CPU arm
        # EVERY GPU BINDING THE LANE CALLS, AND ITS CPU ROUTE (lane/neural,
        # 2026-09-28). `declared` is the families a lane BELONGS to; a binding
        # the lane reaches "narrow" (through a shared door) is left out of it on
        # purpose, but the fit still calls its exports. samba calls ten
        # _mojolearn_mamba and eight _mojolearn_transformer exports and neither
        # was built: run alone on the central AMD box its every cell REFUSED
        # (`_mojolearn_mamba.so ... is not built`); with mamba1..3 and
        # transformer in the same run it passed only because they built them.
        called = {b for b, use in why[lane].get("binding_use", {}).items() if use.get("exports")}
        declared |= called | set().union(*[ROUTES.get(b, set()) for b in called])
        declared = sorted(declared)
        gpu = [b for b in declared if not b.endswith("_host")]
        host = [b for b in declared if b.endswith("_host")]
        if not host:
            raise Fail(f"{lane}: NO CPU ARM. No host binding is declared for it (its GPU bindings: "
                       f"{', '.join(gpu) or 'none'}); declare the family in python/mojolearn/_surface_<lane>.py "
                       "with routes= its GPU binding")
        if not gpu:
            raise Fail(f"{lane}: NO GPU ARM. It resolves no GPU binding, so a GPU column would be the CPU "
                       f"again (host bindings: {', '.join(host)})")
        for b in declared:
            if not (ROOT / "bindings" / script_for(b)).is_file():
                raise Fail(f"{lane}: runs {b}, whose build script bindings/{script_for(b)} does not exist")
        out[lane] = declared
    return out


def stale(binding):
    """None when the built .so carries a stamp of THIS tree's sources, else why."""
    import binding_stamps
    so = output_for(binding)
    if not so.is_file():
        return "not built"
    stamp = binding_stamps.stamp_path(so)
    if not stamp.is_file():
        return "no stamp (built outside this check; its sources are unknown)"
    rec = json.loads(stamp.read_text())
    if rec.get("digest") != binding_stamps.digest(script_for(binding))["digest"]:
        return "its sources changed since it was built"
    return None


def build(binding, log):
    import binding_stamps
    script, so = script_for(binding), output_for(binding)
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    if binding.endswith("_host"):
        # build_host_family.sh refuses any GPU arch: a CPU build takes none.
        env.pop("MOJOLEARN_GPU_ARCHS", None)
    elif script == "build_byte_lm.sh" and sys.platform != "darwin" and not env.get("MOJOLEARN_GPU_ARCHS"):
        arch = gpu_arch()        # the script refuses a Linux build without one named target
        if not arch:
            raise Fail("bindings/build_byte_lm.sh needs MOJOLEARN_GPU_ARCHS and this box reports no GPU arch")
        env["MOJOLEARN_GPU_ARCHS"] = arch
    if (binding.endswith("_host") or script == "build_byte_lm.sh") and so.exists():
        so.unlink()                      # these scripts never overwrite an output
    say(f"build {binding} (bindings/{script})")
    started = time.time()
    with open(log, "a") as fh:
        fh.write(f"\n$ sh bindings/{script}\n")
        fh.flush()
        rc = subprocess.run(["sh", f"bindings/{script}"], cwd=ROOT, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    if rc or not so.is_file():
        raise Fail(f"bindings/{script} failed (exit {rc}); see {log}")
    binding_stamps.cmd_write(script, so)
    say(f"built {binding} in {time.time() - started:.0f}s")


PORTABLE_MATH = ROOT / "packaging" / "portable_math"


def ensure_portable_math(log):
    """python/mojolearn/.libs/libMojolearnMath.so (.dylibs/...dylib on macOS),
    which `_portable_math` dlopens on the CPU arm: built by the tree's own
    recipe (packaging/portable_math/stage.py, whose flags are the arithmetic
    contract) when missing or when its sources changed since the last build."""
    import hashlib
    out = PKG / (".dylibs/libMojolearnMath.dylib" if sys.platform == "darwin" else ".libs/libMojolearnMath.so")
    h = hashlib.sha256()
    for name in ("portable_math.c", "powers_of_ten.h", "stage.py"):
        h.update((PORTABLE_MATH / name).read_bytes())
    stamp = out.with_name(out.name + ".lanecheck-stamp")
    if out.is_file() and stamp.is_file() and stamp.read_text().strip() == h.hexdigest():
        return
    say(f"build {out.relative_to(ROOT)} (packaging/portable_math/stage.py)")
    code = f"import pathlib, stage; stage.build(pathlib.Path({str(out)!r}))"
    with open(log, "a") as fh:
        fh.write(f"\n$ python -c {code!r}\n")
        fh.flush()
        rc = subprocess.run([sys.executable, "-c", code], cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                            env=dict(os.environ, PYTHONPATH=str(PORTABLE_MATH))).returncode
    if rc or not out.is_file():
        raise Fail(f"libMojolearnMath did not build (exit {rc}); see {log}")
    stamp.write_text(h.hexdigest() + "\n")


#: THE STEWARDS' SHARED BINDING STORE (tools/steward_build.py): when set, a
#: stale binding is first looked up there by its source-closure digest (a
#: steward's prebuild, or an earlier request's build of the same sources),
#: and a binding built on the clean or restored tree is published to it. A
#: stored .so lands only when its stamp's digest is THIS tree's, so the
#: sabotaged stage can never be answered from the store.
STORE = os.environ.get("MOJOLEARN_LANE_CHECK_STORE", "")


def _store_tree():
    if not STORE:
        return None
    import steward_build
    return steward_build.Tree(ROOT)


def ensure_built(bindings, log, publish=False):
    ensure_portable_math(log)
    tree = None
    for b in sorted(bindings):
        why = stale(b)
        if why:
            if STORE:
                import steward_build
                tree = tree or _store_tree()
                if steward_build.import_one(tree, STORE, b):
                    say(f"{b}: {why}; taken from the store {STORE} (same source closure)")
                    continue
            say(f"{b}: {why}")
            build(b, log)
            if STORE and publish:
                import steward_build
                tree = tree or _store_tree()
                steward_build.publish_one(tree, STORE, b)


# ------------------------------------------------------------ one lane
def arm_env(kind):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "python"), MOJOLEARN_NUMERIC_MODE="identical")
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        env[k] = "1"
    for k in [k for k in env if k.endswith("_ALLOW_SABOTAGE") or k == "MOJOLEARN_VENDOR"]:
        env.pop(k)
    if kind == "cpu":
        # tools/verify_cpu_batch.py arm_environment: the direct forest and
        # byte LM loaders read their own variables, on the same host set.
        env.update(MOJOLEARN_VENDOR="cpu", MOJOLEARN_HOST_DIR=str(HOST_DIR),
                   MOJOLEARN_FOREST_HOST_BINARY=str(HOST_DIR / "_mojolearn_forest_host.so"),
                   MOJOLEARN_BYTE_LM_HOST_BINARY=str(HOST_DIR / "_mojolearn_byte_lm_host.so"))
    return env


def run_arm(kind, lane, backend, fixtures, out, log, moved_ok=False):
    """One column of one lane. `moved_ok` (the sabotaged stage only): the
    harness exits 1 when a part it compares INSIDE one column moved (a byte
    LM's sampler-vs-trainer `rlpair`, a batch part), which is exactly what a
    biting sabotage does; if it still wrote its JSON, `compare` reads that
    within-column verdict as DISAGREE. Before 2026-09-27 the arm raised here,
    so a sabotage that bit byte-lm read as a failed GPU arm (do-amd request
    1790542457986-dedupe). A missing JSON, any other exit, or a clean or
    restored stage still fails."""
    cmd = [sys.executable, "-u", str(HARNESS), "--lanes", lane, "--repeats", "1", "--fail-on-refused",
           "--json", str(out)]
    if fixtures:
        cmd += ["--fixtures", fixtures]
    cmd += ["--require-cpu", "--require-backend", "cpu"] if kind == "cpu" else ["--require-backend", backend]
    if out.exists():
        out.unlink()
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=ROOT, env=arm_env(kind), stdout=fh, stderr=subprocess.STDOUT).returncode
    if moved_ok and rc == 1 and out.is_file():
        say(f"{lane}: the {kind} arm exited 1 under the sabotage with its JSON written (a part moved "
            "inside the column); the diff decides")
        return
    if rc or not out.is_file():
        tail = Path(log).read_text(errors="replace").splitlines()[-15:]
        raise Fail(f"{lane}: the {kind} arm failed (exit {rc}); last lines of {log}:\n    " + "\n    ".join(tail))


def compare(ib, lane, gpu_json, cpu_json, fixtures, log, backend="gpu"):
    """(verdict, counts): AGREE, DISAGREE or NOTHING COMPARED, from the diff's
    exit status and a count of what both columns really carried."""
    cmd = [sys.executable, str(HARNESS), "--diff", str(gpu_json), str(cpu_json), "--lanes", lane,
           "--require-columns", "2"]
    r = subprocess.run(cmd, cwd=ROOT, env=arm_env("gpu"), capture_output=True, text=True)
    with open(log, "a") as fh:
        fh.write(f"\n$ {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    cols = []
    for p in (gpu_json, cpu_json):
        j = json.loads(Path(p).read_text())
        cols.append((j.get("vendor") or p.name, j))
    keys = sorted(k for k in set(cols[0][1]["cells"]) & set(cols[1][1]["cells"]) if k.split("/")[0] == lane)
    compared, disagree = {}, []
    for k in keys:
        hashes = []
        for _, j in cols:
            c = j["cells"][k]
            if c.get("verdict") in ("MOVED", "DIVERGENT"):
                disagree.append(f"{k} train {c['verdict']} within one column")
            elif c.get("verdict") != "REFUSED" and c.get("hashes"):
                hashes.append(c["hashes"][0])
        if len(hashes) == 2:
            compared["train"] = compared.get("train", 0) + 1
            if hashes[0] != hashes[1]:
                disagree.append(f"{k} train")
        for part in PARTS:
            if not all(f"{part}_verdict" in j["cells"][k] for _, j in cols):
                continue
            verdict, _ = ib._diff_column(cols, k, part)
            if verdict.startswith("IDENTICAL x2"):
                compared[part] = compared.get(part, 0) + 1
            elif verdict.split(" ")[0] in DISAGREEING:
                compared[part] = compared.get(part, 0) + 1
                disagree.append(f"{k} {part} {verdict}")
    want = len(fixtures.split(",")) if fixtures else len(ib.FIXTURES)
    counts = ", ".join(f"{p} {n}" for p, n in sorted(compared.items())) or "nothing"
    if disagree:
        return "DISAGREE", f"compared {counts}; differ: {'; '.join(disagree[:6])}" + \
            (f" (+{len(disagree) - 6} more)" if len(disagree) > 6 else "")
    if compared.get("train", 0) < want:
        return "NOTHING COMPARED", (f"only {compared.get('train', 0)} of {want} fixtures' train cells rest on "
                                    f"two real hashes (compared {counts}); diff exit {r.returncode}")
    if r.returncode:
        tail = " | ".join(line for line in r.stdout.splitlines() if "FAIL" in line or "REFUS" in line)[:400]
        return "NOTHING COMPARED", f"the diff failed (exit {r.returncode}) without a disagreement: {tail}"
    return "AGREE", f"compared {counts} ({backend} column vs CPU column {cols[1][0]})"


def driver_steps(path, workdir):
    """How a listed driver runs, as (build step or None, run step): a Mojo
    check is BUILT under IDENTICAL into `workdir` and then RUN, so a source
    that does not compile is told apart from a check that ran and failed; a
    Python driver is byte-compiled first, a shell driver syntax-checked."""
    if path.endswith(".mojo"):
        exe = str(Path(workdir) / (Path(path).stem + ".bin"))
        return (["sh", "tools/with_identical_mode.sh", "pixi", "run", "mojo", "build", "-I", ".", path, "-o", exe],
                [exe])
    if path.endswith(".py"):
        return ([sys.executable, "-m", "py_compile", path],
                ["sh", "tools/with_identical_mode.sh", sys.executable, "-u", path])
    if path.endswith(".sh"):
        return (["sh", "-n", path], ["sh", "tools/with_identical_mode.sh", "sh", path])
    raise Fail(f"seam driver {path}: a .mojo, .py or .sh file")


def run_driver(path, log, why):
    """(status, code): status is PASS (built, ran, exit 0), FAIL (built, ran
    and exited nonzero on its own: the driver's failure), or BROKEN (did not
    build, or was killed by a signal: a crash is not the driver saying no).
    A sabotage arm counts as a bite ONLY on FAIL (the pass-2 proof hole found
    by lane cnn, 2026-09-27: a patch whose driver no longer compiled used to
    read as a bite)."""
    import tempfile
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    env.pop("MACOSX_DEPLOYMENT_TARGET", None)   # as the binding builders: it disables Metal AOT
    with tempfile.TemporaryDirectory(prefix="seam-driver-") as work, open(log, "a") as fh:
        build, run = driver_steps(path, work)
        if build is not None:
            fh.write(f"\n$ [{why}: build] {' '.join(build)}\n")
            fh.flush()
            rc = subprocess.run(build, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc:
                fh.write(f"[{why}] BUILD FAILED (exit {rc})\n")
                return "BROKEN", f"did not build (exit {rc})"
        fh.write(f"\n$ [{why}] {' '.join(run)}\n")
        fh.flush()
        rc = subprocess.run(run, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    if rc < 0:
        return "BROKEN", f"killed by signal {-rc}"
    return ("PASS", "exit 0") if rc == 0 else ("FAIL", f"exit {rc}")


def read_checks(listing):
    """[(driver, patch or None, line number)] from a .checks file. A line is
    `<driver>` or `<driver>\t<sabotage patch>` (repo-relative; `#` comments)."""
    rows = []
    for n, line in enumerate(listing.read_text().splitlines(), 1):
        body = line.split("#", 1)[0].rstrip()
        if not body.strip():
            continue
        parts = [x.strip() for x in body.split("\t") if x.strip()]
        if len(parts) > 2 or (len(parts) == 1 and len(body.split()) > 1):
            raise Fail(f"{listing.name}:{n}: a line is `<driver>` or `<driver><TAB><sabotage patch>`, not {body!r}")
        for x in parts:
            if not (ROOT / x).is_file():
                raise Fail(f"{listing.name}:{n} lists {x}, which does not exist")
        rows.append((parts[0], parts[1] if len(parts) == 2 else None, n))
    return rows


def listing_lanes(text):
    """The lanes a `.checks` listing names on its `# lanes: a, b, c` line(s).
    A family whose identity lanes live in tools/identity_break.py itself (no
    fragment registers them: the neural family's transformer, mamba1..3,
    samba, ...) names them there, so its seam drivers run before those lanes'
    diff exactly as a fragment's do. tools/lane_select.py reads the same line."""
    out = []
    for line in text.splitlines():
        body = line.strip()
        if body.startswith("#") and body[1:].strip().startswith("lanes:"):
            out += [x for x in body[1:].strip()[len("lanes:"):].replace(",", " ").split() if x]
    return tuple(out)


def listed_core_lanes():
    """{listing id: lanes} for every tools/identity_lanes/<id>.checks that
    names lanes on a `# lanes:` line."""
    root = ROOT / "tools" / "identity_lanes"
    out = {}
    for p in sorted(root.glob("*.checks")) if root.is_dir() else ():
        named = listing_lanes(p.read_text())
        if named:
            out[p.stem] = named
    return out


def seam_checks(ib, lanes, log, pass_no=1):
    """THE PER-SEAM PROOF (plan R2/R3). Each fragment that owns one of these
    lanes lists its check drivers in tools/identity_lanes/<id>.checks, one per
    line, each optionally with a TAB and a sabotage patch. Every driver must
    PASS (built, exit 0) under IDENTICAL; for every patch, `git apply` it, the
    driver must BUILD, run and FAIL (exit nonzero, not a signal), `git apply
    -R`, and it must PASS again. A patch under which the driver does not
    build is a BROKEN ARM: the check fails (`prove_arm`).

    --pass 1 (default): a fragment with no .checks, or a driver with no patch,
    is a note. --pass 2: a fragment that registers lanes and has no .checks
    FAILS, and so does any driver line with no sabotage patch."""
    ids = sorted({fid for fid, owned in getattr(ib, "LANE_FRAGMENTS", {}).items() if set(owned) & set(lanes)}
                 | {fid for fid, owned in listed_core_lanes().items() if set(owned) & set(lanes)})
    for fid in ids:
        listing = ROOT / "tools" / "identity_lanes" / f"{fid}.checks"
        if not listing.is_file():
            if pass_no >= 2:
                raise Fail(f"{fid}: registers identity lanes and has no tools/identity_lanes/{fid}.checks "
                           "(pass 2 needs a driver and a sabotage patch per seam)")
            say(f"{fid}: no tools/identity_lanes/{fid}.checks, so no seam check drivers run (pass 1: a note)")
            continue
        rows = read_checks(listing)
        if not rows:
            raise Fail(f"{listing.name} lists no driver")
        bare = [d for d, pt, _ in rows if pt is None]
        if bare and pass_no >= 2:
            raise Fail(f"{listing.name}: pass 2 needs a sabotage patch on every line; none for {', '.join(bare)}")
        for driver, patch, n in rows:
            prove_arm(driver, patch, log, f"{listing.name}:{n}")


def prove_arm(driver, patch, log, where="seam check"):
    """One listed line: the driver PASSES clean; with the patch applied it
    builds, runs and FAILS on its own; reversed it PASSES again. A patch
    under which the driver does not build (or crashes) is a BROKEN ARM and
    fails the check, as does a clean or restored run that did not build."""
    say(f"seam check {driver}")
    status, why = run_driver(driver, log, "clean")
    if status != "PASS":
        raise Fail(f"seam check {driver}: {status} ({why}); see {log}")
    if patch is None:
        return
    pp = ROOT / patch
    if subprocess.run(["git", "apply", "--check", str(pp)], cwd=ROOT, capture_output=True).returncode:
        raise Fail(f"{where}: the sabotage patch {patch} does not apply to this tree")
    subprocess.run(["git", "apply", str(pp)], cwd=ROOT, check=True)
    try:
        say(f"seam sabotage {patch} applied; {driver} must build, run and FAIL")
        sab, sab_why = run_driver(driver, log, f"sabotaged by {patch}")
    finally:
        r = subprocess.run(["git", "apply", "-R", str(pp)], cwd=ROOT)
        if r.returncode:
            raise Fail(f"could not reverse {patch}; the tree is still sabotaged")
    if sab == "BROKEN":
        raise Fail(f"BROKEN ARM {patch}: under it {driver} {sab_why}, so the arm proves nothing about the seam; "
                   f"see {log}")
    if sab == "PASS":
        raise Fail(f"seam sabotage {patch} was NOT SEEN: {driver} passed under it, so it cannot fail on "
                   "that seam")
    status, why = run_driver(driver, log, f"restored after {patch}")
    if status != "PASS":
        raise Fail(f"seam check {driver} after reversing {patch}: {status} ({why}); see {log}")
    print(f"SEAM: {driver}: PASS, FAIL under {patch} ({sab_why}), PASS after reversal", flush=True)


def check(ib, lanes, needed, backend, fixtures, out, stage, log, pass_no=1):
    """Build what is stale, run both arms per lane, return {lane: verdict}."""
    ensure_built(set().union(*needed.values()), log, publish=stage != "sabotaged")
    if stage == "clean":
        seam_checks(ib, lanes, log, pass_no)
    verdicts = {}
    for lane in lanes:
        gpu_json, cpu_json = out / f"{stage}.{lane}.gpu.json", out / f"{stage}.{lane}.cpu.json"
        run_arm("gpu", lane, backend, fixtures, gpu_json, log, moved_ok=stage == "sabotaged")
        run_arm("cpu", lane, backend, fixtures, cpu_json, log, moved_ok=stage == "sabotaged")
        verdict, detail = compare(ib, lane, gpu_json, cpu_json, fixtures, log, backend)
        print(f"{stage.upper()}: {lane}: {verdict}: {detail}", flush=True)
        verdicts[lane] = verdict
    return verdicts


def patch_paths(patch):
    r = subprocess.run(["git", "apply", "--numstat", str(patch)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise Fail(f"the sabotage patch does not parse: {r.stderr.strip()}")
    return [line.split("\t")[-1] for line in r.stdout.splitlines() if line.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("lanes", help="identity lanes, comma separated")
    ap.add_argument("--sabotage", default="", help="a SOURCE patch that must make every lane DISAGREE")
    ap.add_argument("--fixtures", default="", help="comma separated; default every fixture")
    ap.add_argument("--pass", dest="pass_no", type=int, choices=(1, 2), default=1,
                    help="2: every fragment of these lanes needs a .checks listing with a sabotage patch per "
                         "driver (pass 2); 1 (default): a missing listing or patch is a note")
    ap.add_argument("--out", default="", help="where the columns and the log go "
                                              "(default ~/mojolearn-evidence/lane-check/<stamp>)")
    a = ap.parse_args(argv)
    out = Path(a.out or Path.home() / "mojolearn-evidence" / "lane-check" / time.strftime("%Y%m%dT%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    log = out / "lane_check.log"
    applied = False
    patch = Path(a.sabotage).resolve() if a.sabotage else None
    try:
        backend = gpu_backend()
        ib = load_harness()
        lanes = [x for x in a.lanes.split(",") if x]
        unknown = [x for x in lanes if x not in ib.LANES]
        if not lanes or unknown:
            raise Fail(f"unknown lanes {unknown}" if unknown else "no lane named")
        if a.fixtures and set(a.fixtures.split(",")) - set(ib.FIXTURES):
            raise Fail(f"unknown fixtures {sorted(set(a.fixtures.split(',')) - set(ib.FIXTURES))}")
        say(f"{platform.node()} {platform.machine()} backend={backend} lanes={','.join(lanes)} out={out}")
        needed = needed_bindings(lanes)
        for lane in lanes:
            say(f"{lane} runs: {', '.join(needed[lane])}")

        clean = check(ib, lanes, needed, backend, a.fixtures, out, "clean", log, a.pass_no)
        if any(v != "AGREE" for v in clean.values()):
            raise Fail("clean: " + ", ".join(f"{k} {v}" for k, v in clean.items() if v != "AGREE"))
        if patch is None:
            print("RESULT: PASS (AGREE on " + ",".join(lanes) + ")")
            return 0

        if not patch.is_file():
            raise Fail(f"no sabotage patch at {patch}")
        touched = patch_paths(patch)
        if subprocess.run(["git", "apply", "--check", str(patch)], cwd=ROOT).returncode:
            raise Fail("the sabotage patch does not apply to this tree")
        subprocess.run(["git", "apply", str(patch)], cwd=ROOT, check=True)
        applied = True
        say(f"sabotage applied: {', '.join(touched)}")
        moved = [b for b in sorted(set().union(*needed.values())) if stale(b)]
        python_touched = [p for p in touched if p.startswith("python/") and p.endswith(".py")]
        if not moved and not python_touched:
            raise Fail("the sabotage patch changes no source of any binding these lanes run and no package "
                       "Python file, so it cannot be seen; edit a kernel or oracle these lanes reach")
        say(f"sabotage makes stale: {', '.join(moved) or 'no binding (Python only)'}")
        sab = check(ib, lanes, needed, backend, a.fixtures, out, "sabotaged", log)
        subprocess.run(["git", "apply", "-R", str(patch)], cwd=ROOT, check=True)
        applied = False
        say("sabotage reversed (git apply -R)")
        restored = check(ib, lanes, needed, backend, a.fixtures, out, "restored", log)
        bad = [f"sabotaged {k} {v}" for k, v in sab.items() if v != "DISAGREE"]
        bad += [f"restored {k} {v}" for k, v in restored.items() if v != "AGREE"]
        if bad:
            raise Fail("; ".join(bad) + (" (a sabotage that does not DISAGREE was not seen: the check "
                                         "cannot fail on it)" if any(b.startswith("sabotaged") for b in bad) else ""))
        print("RESULT: PASS (AGREE, then DISAGREE under the sabotage, then AGREE after reversal, on "
              + ",".join(lanes) + ")")
        return 0
    except Fail as exc:
        print(f"RESULT: FAIL ({exc})", flush=True)
        return 1
    finally:
        if applied:
            r = subprocess.run(["git", "apply", "-R", str(patch)], cwd=ROOT)
            print("sabotage reversed on exit" if r.returncode == 0 else
                  f"WARNING: could not reverse {patch}; the tree is still sabotaged", flush=True)


if __name__ == "__main__":
    sys.exit(main())
