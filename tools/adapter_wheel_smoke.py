#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""The framework adapters' smoke against an INSTALLED mojolearn wheel. A test controller, never product code.

    python3 tools/adapter_wheel_smoke.py --python VENV/bin/python --out DIR [--device cuda:0|mps]
                                         [--install torch,max] [--require torch,max]

VENV holds the installed mojolearn wheel (and on Linux its vendor wheel). The
smoke asks that interpreter what is installed, then runs the adapters' own
checkers against the SHIPPED files (mojolearn/.../_adapters/), never this tree's:

    torch-compile    tools/check_native_adapters.py --installed      (when torch imports): whole models
                     through torch.compile(backend="mojolearn"), close to eager and bit for bit
                     mojolearn.import_model's output
    torch-training   tools/check_native_training_adapters.py --extension <shipped registration> --smoke
    max-training     tools/check_max_training_adapters.py --library <shipped .mojoc> ... --smoke
                     (when max imports, after torch-training passed: it compares with its results)

The adapters are whole-model only (Andrew, 2026-10-24): there is no per-operation
phase, because there is no per-operation surface.

--install first pip-installs what a user gets from `pip install "mojolearn[torch]"`
/ `[max]` (on Linux the extras live on mojolearn-core): the installed wheel's own
extra, which must carry the pin the shipped adapters.json names (its
`requirements`); Torch from the index it was built against, so the ROCm build
on AMD. A framework that does not import is recorded as
absent, and one of another release than the adapters were built for is
recorded as refused (the loader's designed answer): neither is a pass.
--require makes an absent or refused framework a failure.

Writes DIR/adapter-smoke.json:
    status   PASSED        every phase that ran passed and at least one ran
             NO_FRAMEWORK  no framework the adapters serve is installed (nothing ran)
             FAILED        a phase failed, the wheel carries no adapters, or --require unmet
Exit 0 for PASSED and NO_FRAMEWORK, 1 for FAILED. Phases run serially, each
bounded in its own process group; logs and exit codes stay in DIR.
"""
import argparse
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
SCHEMA = "mojolearn.adapter-wheel-smoke.v1"

PROBE = r'''
import importlib.metadata as md, importlib.util as iu, json
out = {}
def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None
for dist in ("mojolearn", "mojolearn-core", "mojolearn-nvidia", "mojolearn-amd", "torch", "max", "numpy"):
    out[dist] = version(dist)
out["torch_importable"] = iu.find_spec("torch") is not None
out["max_importable"] = iu.find_spec("max") is not None
try:
    from mojolearn.adapters import _shipped
    m = _shipped.manifest()
    out["adapters"] = dict(directory=_shipped.directory(), vendor=m.get("vendor"), arch=m.get("arch"),
                           torch=m.get("torch"), max=m.get("max"), requirements=m.get("requirements"),
                           paths={role: _shipped.path(role) for role in _shipped.FILES})
except Exception as exc:
    out["adapters_error"] = "%s: %s" % (type(exc).__name__, exc)
# The extras a user installs (`pip install "mojolearn[torch]"`; on Linux they live on mojolearn-core):
# each extra's Requires-Dist lines without their markers, from the installed distribution's metadata.
try:
    try:
        from packaging.requirements import Requirement
    except ImportError:
        from pip._vendor.packaging.requirements import Requirement
    for dist in ("mojolearn-core", "mojolearn"):
        try:
            provided = md.metadata(dist).get_all("Provides-Extra") or []
        except md.PackageNotFoundError:
            continue
        if not provided:
            continue
        extras = {}
        for extra in provided:
            picked = []
            for line in md.requires(dist) or []:
                r = Requirement(line)
                if r.marker is not None and r.marker.evaluate({"extra": extra}) and not r.marker.evaluate({"extra": ""}):
                    r.marker = None
                    picked.append(str(r))
            extras[extra] = picked
        out["extras"], out["extras_distribution"] = extras, dist
        break
except Exception as exc:
    out["extras_error"] = "%s: %s" % (type(exc).__name__, exc)
print(json.dumps(out))
'''

#: The loader's refusal of a framework release the adapters were not built for.
REFUSAL = "cannot load:"


def run(name, argv, out, seconds, env=None):
    """One bounded phase in its own process group; {'exit', 'seconds', 'log', 'timed_out'}."""
    log = out / f"{name}.log"
    started = time.monotonic()
    with open(log, "w") as f:
        f.write("argv: " + json.dumps([str(a) for a in argv]) + "\n")
        f.flush()
        proc = subprocess.Popen([str(a) for a in argv], stdout=f, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
        timed_out = False
        try:
            deadline = float(os.environ.get("MOJOLEARN_GATE_DEADLINE", "inf"))
            code = proc.wait(timeout=max(0, min(seconds, deadline - time.time())))
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=0 if os.environ.get("MOJOLEARN_GATE_DEADLINE") else 20)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 124
    (out / f"{name}.exit").write_text(f"{code}\n")
    return dict(exit=code, seconds=round(time.monotonic() - started, 1), log=log.name, timed_out=timed_out)


def probe(python, out):
    r = run("probe", [python, "-c", PROBE], out, 120)
    text = (out / "probe.log").read_text().splitlines()
    try:
        return json.loads(text[-1]) if r["exit"] == 0 else {"probe_error": text[-5:]}
    except (ValueError, IndexError):
        return {"probe_error": text[-5:]}


def _requirement_key(text):
    """(name, specifier set) of a requirement string, enough to compare a manifest pin with an extra's line."""
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)", text.split(";", 1)[0])
    name, spec = (m.group(1), m.group(2)) if m else (text, "")
    return re.sub(r"[-_.]+", "-", name).lower(), frozenset(p.replace(" ", "") for p in spec.split(",") if p.strip())


def install_commands(python, adapters, wanted, extras, test_deps=None):
    """The pip commands that install what `pip install "mojolearn[<framework>]"` installs: the installed
    wheel's own extra (EXTRAS, the probe's `extras`), checked against the pin the shipped adapters.json
    names, plus TEST_DEPS[framework] (test-only packages such as torchvision) in the same resolve. Torch
    comes from the index the adapters were built against (PyTorch's ROCm index on AMD, as the install
    guide says), everything else from PyPI. Raises ValueError when the wheel declares no such extra or
    the extra and the shipped manifest disagree."""
    reqs = adapters.get("requirements") or {}
    torch = adapters.get("torch") or {}
    test_deps = test_deps or {}
    cmds = {}
    for w in ("torch", "max"):
        if w not in wanted:
            continue
        listed = list((extras or {}).get(w) or [])
        if not listed:
            raise ValueError(f"the installed wheel declares no [{w}] extra; a user cannot install {w} through it")
        if reqs.get(w) and _requirement_key(reqs[w]) not in {_requirement_key(r) for r in listed}:
            raise ValueError(f"the installed [{w}] extra {listed} does not carry the adapters' pin {reqs[w]!r}")
        cmd = [python, "-m", "pip", "install", "--disable-pip-version-check", *listed, *test_deps.get(w, ())]
        if w == "torch" and torch.get("index_url") and "pypi.org" not in torch["index_url"]:
            # the backend's own index (PyTorch's ROCm wheels); dependencies still from PyPI
            cmd += ["--index-url", torch["index_url"], "--extra-index-url", "https://pypi.org/simple"]
        cmds["install-" + w] = cmd
    return cmds


def refused(out):
    """Did the torch checker stop at the loader's refusal (adapter-results.json failures)?"""
    path = out / "torch-compile" / "adapter-results.json"
    try:
        failures = json.loads(path.read_text()).get("failures") or []
    except (OSError, ValueError):
        return False
    return any(REFUSAL in str(f) for f in failures)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", required=True, help="the interpreter of the venv holding the installed wheel")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--device", default="mps" if sys.platform == "darwin" else "cuda:0")
    ap.add_argument("--install", default="", help="comma list of torch,max to pip-install first")
    ap.add_argument("--require", default="", help="comma list of torch,max that must be present and run")
    ap.add_argument("--seconds", type=int, default=900, help="bound per phase")
    a = ap.parse_args(argv)
    words = lambda s: [w for w in s.split(",") if w]  # glue: parse the comma lists
    install, require = words(a.install), words(a.require)
    for w in install + require:
        if w not in ("torch", "max"):
            ap.error(f"unknown framework {w!r}")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = dict(schema=SCHEMA, python=a.python, device=a.device, phases={}, problems=[], frameworks={})
    facts = probe(a.python, out)
    report["installed"] = facts
    adapters = facts.get("adapters")
    if not adapters:
        report["problems"].append("the installed wheel carries no usable framework adapters: "
                                  + str(facts.get("adapters_error") or facts.get("probe_error")))
    else:
        try:
            cmds = install_commands(a.python, adapters, install, facts.get("extras"))
        except ValueError as exc:
            cmds = {}
            report["problems"].append("install: %s%s" % (exc, (" (%s)" % facts["extras_error"]) if facts.get("extras_error") else ""))
        for name, cmd in cmds.items():
            report["phases"][name] = run(name, cmd, out, a.seconds)
            if report["phases"][name]["exit"] != 0:
                report["problems"].append(f"{name} failed (exit {report['phases'][name]['exit']})")
        if install:
            facts = probe(a.python, out)
            report["installed_after_install"] = facts
        paths = adapters["paths"]
        env = dict(os.environ, PYTORCH_ENABLE_MPS_FALLBACK="0")
        torch_ok = False
        if facts.get("torch_importable"):
            r = run("torch-compile", [a.python, HERE / "check_native_adapters.py", "--installed", "--device", a.device,
                                      "--out", out / "torch-compile"], out, a.seconds, env)
            report["phases"]["torch-compile"] = r
            if r["exit"] != 0 and refused(out):
                report["frameworks"]["torch"] = "refused: not the release the adapters were built for"
            elif r["exit"] != 0:
                report["frameworks"]["torch"] = "failed"
                report["problems"].append(f"torch-compile adapter check failed (exit {r['exit']}, {r['log']})")
            else:
                r = run("torch-training", [a.python, HERE / "check_native_training_adapters.py", "--extension",
                                           paths["torch"], "--device", a.device, "--smoke",
                                           "--out", out / "torch-training"], out, a.seconds, env)
                report["phases"]["torch-training"] = r
                torch_ok = r["exit"] == 0
                report["frameworks"]["torch"] = "ran" if torch_ok else "failed"
                if not torch_ok:
                    report["problems"].append(f"torch training adapter check failed (exit {r['exit']})")
        else:
            report["frameworks"]["torch"] = "absent"
        if not facts.get("max_importable"):
            report["frameworks"]["max"] = "absent"
        elif not torch_ok:
            # max-training compares its bytes with torch-training's: without them it cannot run
            report["frameworks"]["max"] = "not run: needs a passing torch-training phase"
        else:
            r = run("max-training", [a.python, HERE / "check_max_training_adapters.py", "--library", paths["max"],
                                     "--native-library", paths["native"], "--torch-results", out / "torch-training",
                                     "--smoke", "--out", out / "max-training"], out, a.seconds, env)
            report["phases"]["max-training"] = r
            report["frameworks"]["max"] = "ran" if r["exit"] == 0 else "failed"
            if r["exit"] != 0:
                report["problems"].append(f"MAX training adapter check failed (exit {r['exit']}, {r['log']})")
        for w in require:
            if report["frameworks"].get(w) != "ran":
                report["problems"].append(f"--require {w}: {report['frameworks'].get(w)}")
    ran = [w for w, state in report["frameworks"].items() if state == "ran"]
    report["status"] = "FAILED" if report["problems"] else "PASSED" if ran else "NO_FRAMEWORK"
    (out / "adapter-smoke.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"adapter smoke: {report['status']} frameworks={report['frameworks']} "
          f"phases={ {n: p['exit'] for n, p in report['phases'].items()} }"
          + ("" if not report["problems"] else " problems: " + "; ".join(report["problems"])))
    return 1 if report["status"] == "FAILED" else 0


if __name__ == "__main__":
    sys.exit(main())
