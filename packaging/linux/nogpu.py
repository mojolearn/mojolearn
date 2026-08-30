#!/usr/bin/env python3
"""THE NO-GPU GATE: does the wheel layout refuse, by name, on a box with
no supported device?

    python3 packaging/linux/nogpu.py --pkgroot <dir> --vendor cuda|hip \\
        --python "<interpreter command>" --json <out.json>

Three attempts, each recorded with what it did and did not prove.

  env_hidden      CUDA_VISIBLE_DEVICES= and HIP_VISIBLE_DEVICES= empty.
                  EXPECTED TO IMPORT ANYWAY: those variables hide devices
                  from the runtime, not device nodes from the filesystem,
                  and the selector's probe reads `/dev/nvidiactl` and
                  `/dev/kfd`. The record says what the first fit did.
  namespace       `unshare -rm`, then a tmpfs holding only /dev/null and
                  friends mounted OVER /dev so every device node is
                  genuinely gone, plus /dev/null bound over every driver
                  library, then the import. In an unprivileged container
                  `unshare` is often refused by seccomp; when it is, that
                  is recorded as NOT TESTED, never as a pass. This route
                  covered device nodes with bind mounts until 2026-08-30
                  and could not pass: a bind mount leaves the path there
                  for `os.path.exists` to find.
  docker          if a docker daemon is reachable (a DigitalOcean droplet
                  is a VM; a RunPod pod is not), the import inside
                  `python:3.12-slim` with NO device passed through, the
                  package bind-mounted read-only. This is the honest test
                  and the one the runbook asks for when it is available.
                  NEITHER RENTED BOX HAS HAD IT: the RunPod pod has no
                  daemon and the DigitalOcean image ships without docker.
                  `packaging/linux/nogpu_local.sh` runs exactly this on the
                  Mac against a FETCHED set, needs no box at all, and is
                  what actually closed this gate on 2026-08-30.

A pass in `namespace` or `docker` requires: non-zero exit, and the message
naming "NO SUPPORTED GPU FOUND", each probed path, and MOJOLEARN_VENDOR.
"""

import argparse
import ctypes.util
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys

DEV = {"cuda": ["/dev/nvidiactl", "/dev/nvidia0"],
       "hip": ["/dev/kfd", "/dev/dri/renderD128"]}
LIBS = {"cuda": ["libcuda.so.1", "libcuda.so"],
        "hip": ["libamdhip64.so.7", "libamdhip64.so.6", "libamdhip64.so"]}
SNIPPET = "import mojolearn; print('IMPORTED', mojolearn.vendor())"
FIT = ("import numpy as np, mojolearn; X=np.random.default_rng(0).random((256,4),dtype=np.float32); "
       "mojolearn.KMeans(n_clusters=2).fit(X); print('FIT OK')")


def lib_paths(names):
    out = []
    try:
        ld = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True).stdout
    except OSError:
        ld = ""
    for n in names:
        for ln in ld.splitlines():
            if ln.strip().startswith(n + " "):
                out.append(ln.split("=>")[-1].strip())
        f = ctypes.util.find_library(n.replace("lib", "", 1).split(".so")[0])
        if f and os.path.isabs(f):
            out.append(f)
    return sorted(set(p for p in out if os.path.exists(p)))


def run(cmd, env, cwd, timeout=600):
    try:
        r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return -1, "", f"{type(exc).__name__}: {exc}"
    return r.returncode, r.stdout.strip()[-400:], "\n".join(r.stderr.strip().splitlines()[-25:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkgroot", required=True)
    ap.add_argument("--vendor", required=True, choices=("cuda", "hip"))
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    root = str(pathlib.Path(a.pkgroot).resolve())
    base_env = dict(os.environ)
    base_env["PYTHONPATH"] = root
    base_env.pop("MOJOLEARN_VENDOR", None)
    py = shlex.split(a.python)
    res = {}

    env = dict(base_env, CUDA_VISIBLE_DEVICES="", HIP_VISIBLE_DEVICES="",
               ROCR_VISIBLE_DEVICES="")
    rc, out, err = run(py + ["-c", SNIPPET], env, root)
    rc2, out2, err2 = run(py + ["-c", FIT], env, root)
    res["env_hidden"] = {
        "import_rc": rc, "import_out": out, "import_err": err,
        "fit_rc": rc2, "fit_out": out2, "fit_err": err2,
        "verdict": "IMPORTED (expected: the probe reads device nodes, not these variables)"
                   if rc == 0 else "REFUSED at import",
    }
    print(f"  env_hidden   import rc={rc} {out or err.splitlines()[-1:]}; fit rc={rc2}")

    # A BIND MOUNT LEAVES THE PATH EXISTING. This block used to run
    # `mount --bind /dev/null /dev/kfd` for every device node, and on
    # 2026-08-30 it reported FAIL against a library that was behaving
    # correctly: the probe asks `os.path.exists("/dev/kfd")`, and a /dev/kfd
    # that IS /dev/null still exists. The route could not pass whatever the
    # library did, and the gate's own FAIL was the only evidence it produced.
    #
    # Device nodes are therefore hidden by REPLACING /dev, not by covering
    # entries inside it: a tmpfs is populated with only the character devices
    # a python process needs, and mounted over /dev, so every other path
    # under it is genuinely gone. Driver libraries are still covered with
    # /dev/null, which is the right technique for THEM, because the probe
    # loads a library rather than stat-ing it and `CDLL("/dev/null")` fails.
    devs = [p for p in DEV[a.vendor] + DEV[{"cuda": "hip", "hip": "cuda"}[a.vendor]]
            if os.path.exists(p)]
    libs = lib_paths(LIBS[a.vendor]) + lib_paths(LIBS[{"cuda": "hip", "hip": "cuda"}[a.vendor]])
    targets = devs + libs
    # /dev/null is bound into the replacement BEFORE /dev is replaced, so the
    # library covers below still have a real /dev/null to point at.
    keep = ["null", "zero", "full", "random", "urandom", "tty", "ptmx"]
    steps = ["mkdir -p /tmp/.nodev", "mount -t tmpfs none /tmp/.nodev"]
    for n in keep:
        steps.append(f"[ -e /dev/{n} ] && touch /tmp/.nodev/{n} && mount --bind /dev/{n} /tmp/.nodev/{n} || true")
    steps += [f"mount --bind /dev/null {shlex.quote(p)}" for p in libs]
    steps.append("mount --bind /tmp/.nodev /dev")
    script = " && ".join(steps) + " && " + \
        f"cd {shlex.quote(root)} && PYTHONPATH={shlex.quote(root)} " + \
        " ".join(shlex.quote(x) for x in py) + " -c " + shlex.quote(SNIPPET)
    if shutil.which("unshare"):
        rc, out, err = run(["unshare", "-rm", "sh", "-c", script], base_env, root)
        wanted = ["NO SUPPORTED GPU FOUND", "MOJOLEARN_VENDOR"] + DEV[a.vendor]
        found = {w: w in err for w in wanted}
        unshare_refused = rc != 0 and ("unshare" in err and "Operation not permitted" in err)
        res["namespace"] = {
            "hidden": targets, "rc": rc, "out": out, "err": err, "found": found,
            "verdict": ("NOT TESTED: unshare refused in this container" if unshare_refused
                        else "PASS" if (rc != 0 and all(found.values())) else "FAIL"),
        }
    else:
        res["namespace"] = {"verdict": "NOT TESTED: no unshare(1) on this image"}
    print(f"  namespace    {res['namespace']['verdict']}")

    if shutil.which("docker") and run(["docker", "info"], base_env, root, 60)[0] == 0:
        cmd = ["docker", "run", "--rm", "--cpus", "2", "-v", f"{root}:/pkg:ro",
               "-e", "PYTHONPATH=/pkg", "python:3.12-slim", "sh", "-c",
               "pip install -q numpy >/dev/null 2>&1; python -c " + shlex.quote(SNIPPET)]
        rc, out, err = run(cmd, base_env, root, 900)
        wanted = ["NO SUPPORTED GPU FOUND", "MOJOLEARN_VENDOR"] + DEV["cuda"] + DEV["hip"]
        found = {w: w in err for w in wanted}
        res["docker"] = {"rc": rc, "out": out, "err": err, "found": found,
                         "verdict": "PASS" if (rc != 0 and all(found.values())) else "FAIL"}
    else:
        res["docker"] = {"verdict": "NOT TESTED: no docker daemon here"}
    print(f"  docker       {res['docker']['verdict']}")

    tested = [k for k in ("namespace", "docker") if res[k]["verdict"] in ("PASS", "FAIL")]
    res["summary"] = ("PASS" if tested and all(res[k]["verdict"] == "PASS" for k in tested)
                      else "FAIL" if tested else "NOT TESTED on this box")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=2, sort_keys=True))
    print("nogpu", res["summary"])
    return 0 if res["summary"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
