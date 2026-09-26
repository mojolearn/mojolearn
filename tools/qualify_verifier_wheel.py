#!/usr/bin/env python3
"""Qualify the installed verifier CLI on one exact wheel, outside the checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import tempfile
import zipfile


#: wheel-name prefix -> distribution of the split Linux GPU plugins
#: (python/mojolearn/gpu_plugins.py; spelled here because this driver ships
#: alone to the box, stdlib only, with no checkout beside it).
PLUGIN_DISTRIBUTIONS = {"mojolearn_nvidia": "mojolearn-nvidia", "mojolearn_amd": "mojolearn-amd"}


def admit(kind, doc, models):
    if kind == "coverage":
        assert doc["lanes"] and doc["cpu_execution_counts"]["declared"] > 0
    elif kind == "self-test":
        assert doc["passed"] is True
        assert doc["clean"]["state"] == "IDENTICAL"
        assert doc["perturbed"]["state"] == "DIVERGENT"
    else:
        cells = {(r["lane"], r["fixture"], r["part"]): r for r in doc["cells"]}
        assert len(cells) == len(doc["cells"]), "duplicate cell"
        if kind == "extended":
            owed = [r for r in cells.values() if r["state"] == "OWED"]
            assert (doc["exit"], doc["verdict"]) == ((5, "INCOMPLETE") if owed else (0, "VERIFIED"))
            for row in cells.values():
                assert row["state"] in ("IDENTICAL", "N/A", "OWED")
                if row["state"] == "OWED":
                    assert row["part"] in ("batchgrad", "batchscale", "ragged", "rlpair")
                    assert row.get("error") is None and re.fullmatch("[0-9a-f]{16}", row["value"])
        else:
            assert doc["exit"] == 0 and doc["verdict"] == "VERIFIED"
        if kind == "models":
            expected = {(f"portable:{m['lane']}", m["fixture"], part)
                        for m in models for part in ("model", "batch")}
            assert expected and cells.keys() == expected
            assert all(row["state"] == "IDENTICAL" for row in cells.values())
            assert doc["selection"]["models_only"] is True
        else:
            for part in ("train", "infer", "batch"):
                assert cells["knn", "base", part]["state"] == "IDENTICAL"
            if kind == "extended":
                assert {"batchgrad", "batchscale", "ragged", "rlpair"} <= {
                    row["part"] for row in cells.values()}


EXPANDED_API_GUARD = """import hashlib,importlib,json,pathlib
from mojolearn import _backend
apis = {
    'mojolearn.models': ['CausalLM', 'ParallelCausalLM'],
    'mojolearn.parallel_ivf': ['DistributedIVFIndex'],
    'mojolearn.parallel_model_selection': ['cross_val_score'],
    'mojolearn.parallel_gaussian_process': ['fit_gaussian_process_classifier', 'predict_gaussian_process_classifier'],
    'mojolearn.parallel_forecasting': ['predict_arima', 'forecast_arima', 'predict_exponential_smoothing', 'forecast_exponential_smoothing'],
}
for name, symbols in apis.items():
    module = importlib.import_module(name)
    for symbol in symbols:
        if not callable(getattr(module, symbol, None)):
            raise RuntimeError(name + '.' + symbol + ' is unavailable')
native = _backend.binding('_mojolearn_ivf', 'identical')
symbols = ['ivf_flat_partial_search', 'ivf_finalize_distances']
for symbol in symbols:
    if not callable(getattr(native, symbol, None)):
        raise RuntimeError('rebuild IVF binding: missing ' + symbol)
gbdt = _backend.binding('_mojolearn_gbdt', 'identical')
if not callable(getattr(gbdt, 'gbdt_fit', None)):
    raise RuntimeError('GBDT binding has no fit entry for cross-validation')
gbdt_path = pathlib.Path(gbdt.__file__).resolve()
path = pathlib.Path(native.__file__).resolve()
print(json.dumps(dict(apis=apis, ivf_symbols=symbols, gbdt_symbols=['gbdt_fit'],
                     gbdt_binding=str(gbdt_path), gbdt_sha256=hashlib.sha256(gbdt_path.read_bytes()).hexdigest(),
                     binding=str(path),
                     binding_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
"""


def expanded_checks(run, python, work, output, *, vendor, scope, devices=None):
    """Use the public installed CLI; every successful command is checkpointed."""
    full = scope == 'expanded'
    if full and vendor not in ('metal', 'cuda', 'hip'):
        raise RuntimeError('expanded scope requires a physical GPU backend')
    if devices and (not full or vendor not in ('cuda', 'hip')):
        raise RuntimeError('multi-GPU qualification requires expanded CUDA/HIP scope')
    report = dict(scope=scope, release_qualified=False,
                  physical_execution_trace='OWED', native_fault_controls='SEPARATE_GATE',
                  cross_validation='OWED',
                  multi_gpu='NOT_APPLICABLE' if vendor == 'metal' else 'OWED')
    if full:
        report['apis'] = run('expanded-api', [python, '-c', EXPANDED_API_GUARD], work, True)
    for device in (('cpu', 'gpu') if full else ('cpu',)):
        run('loaded-lm-' + device, [python, '-m', 'mojolearn', 'verify-causal-lm',
            '--device', device, '--formats', 'float32', 'bfloat16', 'int8',
            '--output', str(output / ('loaded-lm-' + device + '-capture.json'))], work)
    if full:
        run('loaded-lm-cpu-gpu-compare', [python, '-m', 'mojolearn', 'verify-causal-lm',
            '--compare', str(output / 'loaded-lm-cpu-capture.json'),
            str(output / 'loaded-lm-gpu-capture.json')], work)
    if devices:
        run('distributed', [python, '-m', 'mojolearn', 'verify-distributed',
            '--devices', ','.join(map(str, devices)), '--require-installed',
            '--out', str(output / 'distributed-capture.json')], work)
        run('cross-validation', [python, '-m', 'mojolearn', 'verify-cross-validation',
            '--devices', ','.join(map(str, devices)), '--require-backend', vendor, '--require-installed',
            '--out', str(output / 'cross-validation')], work)
        report['cross_validation'] = 'NUMERICS_AND_PLACEMENT_CHECKED_EXECUTION_TRACE_OWED'
        for name, selected in [('split', devices), ('reversed', tuple(reversed(devices)))]:
            path = output / ('loaded-lm-' + name + '-capture.json')
            run('loaded-lm-' + name, [python, '-m', 'mojolearn', 'verify-causal-lm',
                '--device', 'gpu', '--formats', 'float32', 'bfloat16', 'int8',
                '--layer-devices', *map(str, selected), '--output', str(path)], work)
            run('loaded-lm-' + name + '-compare', [python, '-m', 'mojolearn', 'verify-causal-lm',
                '--compare', str(output / 'loaded-lm-gpu-capture.json'), str(path)], work)
        report['multi_gpu'] = dict(classical='NUMERICS_AND_PLACEMENT_CHECKED',
                                   loaded_lm='SPLIT_AND_REVERSED_NUMERICS_CHECKED_PLACEMENT_TRACE_OWED')
    return report


INSTALLED_GUARD = """import json,pathlib,sys,mojolearn
p=pathlib.Path(mojolearn.__file__).resolve()
assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()) and 'site-packages' in p.parts
assert mojolearn.__version__ == sys.argv[1]
print(json.dumps(dict(package=str(p),version=mojolearn.__version__,vendor=mojolearn.vendor())))
"""


def runner(manifest, receipt, output):
    """(save, run): the receipt checkpoint and the bounded job runner, in a
    clean environment."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("MOJOLEARN_") and k not in ("PYTHONPATH", "PYTHONHOME", "PYTHONOPTIMIZE")}
    env.update(MOJOLEARN_NUMERIC_MODE="identical", PYTHONNOUSERSITE="1")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MOJOLEARN_CPU_THREADS"):
        env[name] = "1"

    def save():
        temporary = receipt.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(receipt)

    def run(name, command, cwd, json_output=False, allowed_exits=(0,)):
        path = output / (name + (".json" if json_output else ".log"))
        with path.open("w") as stdout, (output / (name + ".stderr.log")).open("w") as stderr:
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                    stdout=stdout, stderr=stderr, start_new_session=True)
            try:
                code = proc.wait(timeout=180)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                code = 124
        manifest["jobs"].append(dict(name=name, exit_code=code, output=str(path)))
        save()
        if code not in allowed_exits:
            raise RuntimeError(f"{name} failed with exit {code}; see {path}")
        return json.loads(path.read_text()) if json_output else None

    return save, run


def qualify_jobs(run, save, manifest, python, work, output, models, scope, devices, version):
    """The smoke's jobs against an installed package: the same for a wheel
    installed here and for a package pip resolved from an index."""
    manifest["installed"] = run("installed", [python, "-c", INSTALLED_GUARD, version], work, True)
    commands = {
        "coverage": ["--coverage"],
        "models": ["--models-only", "--repeats", "2"],
        "batch": ["--lanes", "knn", "--fixtures", "base", "--repeats", "2", "--no-models"],
        "extended": ["--lanes", "knn", "--fixtures", "base", "--repeats", "2", "--batch-checks", "--no-models"],
        "self-test": ["--self-test"],
    }
    if scope == 'cpu-only':
        commands = {name: flags for name, flags in commands.items() if name in ('coverage', 'models')}
    manifest['expanded'] = expanded_checks(run, python, work, output,
        vendor=manifest['installed']['vendor'], scope=scope, devices=devices)
    save()
    for kind, flags in commands.items():
        doc = run(kind, [python, "-m", "mojolearn", "verify", *flags, "--json"], work, True,
                  (0, 5) if kind == "extended" else (0,))
        admit(kind, doc, models)
        if kind == "extended":
            # This gate checks the CLI's honest incomplete result;
            # it does not qualify properties lacking references.
            manifest["unqualified_properties"] = [
                dict(lane=r["lane"], fixture=r["fixture"], part=r["part"], state=r["state"])
                for r in doc["cells"] if r["state"] == "OWED"]


#: Run by the installed interpreter: where its mojolearn lives (without
#: importing it) and the installed versions of the three split projects.
LOCATE_INSTALLED = """import importlib.metadata as md,importlib.util,json
spec=importlib.util.find_spec('mojolearn')
dists={}
for n in ('mojolearn','mojolearn-nvidia','mojolearn-amd'):
    try: dists[n]=md.version(n)
    except md.PackageNotFoundError: dists[n]=None
print(json.dumps(dict(package=list(spec.submodule_search_locations)[0] if spec else None,distributions=dists)))
"""


def qualify_installed(parser, args):
    """--installed-python: the jobs of main() against the package already
    installed for that interpreter (pip resolved it from an index). The
    models manifest and the source commit are read from the installed
    package, where the wheel carried them."""
    if args.multi_gpu_devices:
        parser.error('--installed-python runs the one-GPU smoke')
    if args.expected_source_commit and not re.fullmatch('[0-9a-f]{40}', args.expected_source_commit):
        parser.error('expected-source-commit must be a full lowercase SHA')
    python = str(args.installed_python)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "results.json"
    if receipt.exists():
        parser.error("use a fresh output directory")
    manifest = dict(wheel=None, wheel_sha256=None, status="INCOMPLETE", jobs=[], scope=args.scope,
                    source_commit=None, release_qualified=False, expanded=None,
                    installed_from=dict(python=python, expected_version=args.expected_version))
    save, run = runner(manifest, receipt, output)
    save()
    try:
        with tempfile.TemporaryDirectory(prefix="mojolearn-verifier-installed-") as directory:
            work = Path(directory)
            located = run("locate-installed", [python, "-c", LOCATE_INSTALLED], work, True)
            manifest["installed_from"].update(located)
            wrong = {n: v for n, v in located["distributions"].items() if v != args.expected_version}
            if wrong:
                raise RuntimeError(f"installed distributions not at {args.expected_version}: {wrong}")
            package = Path(located["package"] or "")
            models = json.loads((package / "verify_reference/models/models.json").read_text())["models"]
            source_commit = (package / "identity_columns/COMMIT").read_text().strip()
            if not re.fullmatch('[0-9a-f]{40}', source_commit):
                raise RuntimeError('the installed package has no valid source commit')
            manifest["source_commit"] = source_commit
            if args.expected_source_commit and source_commit != args.expected_source_commit:
                raise RuntimeError(f'the installed package records source commit {source_commit}, '
                                   f'not {args.expected_source_commit}')
            run("dependencies", [python, "-m", "pip", "check"], work)
            qualify_jobs(run, save, manifest, python, work, output, models, args.scope, None,
                         args.expected_version)
            manifest["status"] = "PASSED" if args.scope == "expanded" else "PASSED_CPU_ONLY"
    except Exception as exc:
        manifest.update(status="FAILED", reason=repr(exc))
    save()
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] in ("PASSED", "PASSED_CPU_ONLY") else 1


def main():
    if not __debug__:
        raise RuntimeError('qualification requires assertions enabled; Python -O is refused')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, nargs="?")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", default="python3.12")
    parser.add_argument("--wheelhouse", type=Path)
    # THE SPLIT LINUX PACKAGES (python/mojolearn/gpu_plugins.py): the positional
    # wheel is the core `mojolearn`, and each --plugin (mojolearn_nvidia-*.whl,
    # mojolearn_amd-*.whl of the same version) is installed beside it in the
    # same pip command. The receipt names every plugin by name and sha256, so
    # tools/check_light_release.py can tie it to the plugin a release publishes.
    parser.add_argument("--plugin", type=Path, action="append", default=[],
                        help="a split GPU plugin wheel installed with the core (repeatable)")
    parser.add_argument('--scope', choices=('expanded', 'cpu-only'), default='expanded')
    parser.add_argument('--multi-gpu-devices', help='optional two distinct indices, e.g. 0,1')
    parser.add_argument('--expected-source-commit', help='full commit recorded inside the candidate wheel')
    # THE USER'S INSTALL, FROM THE INDEX (tools/index_release_check.py,
    # release_wheel_smoke.sh --from-index): no wheel file; the package pip
    # already resolved from TestPyPI or PyPI into this python's environment
    # is qualified in place, with the same jobs. The receipt names the
    # installed distributions instead of a wheel sha256.
    parser.add_argument('--installed-python', type=Path,
                        help='qualify the mojolearn already installed for this interpreter (no wheel, no install)')
    parser.add_argument('--expected-version', help='with --installed-python: the version that must be installed')
    args = parser.parse_args()
    if args.installed_python:
        if args.wheel or args.plugin or args.wheelhouse:
            parser.error('--installed-python qualifies what is installed: no wheel, --plugin or --wheelhouse')
        if not args.expected_version:
            parser.error('--installed-python needs --expected-version')
        return qualify_installed(parser, args)
    if not args.wheel:
        parser.error('a wheel is required (or --installed-python)')
    devices = None
    if args.multi_gpu_devices:
        try:
            devices = tuple(int(x) for x in args.multi_gpu_devices.split(','))
        except ValueError:
            parser.error('multi-gpu-devices must contain integer indices')
        if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0 or args.scope != 'expanded':
            parser.error('two distinct nonnegative device indices require expanded scope')
    if args.expected_source_commit and not re.fullmatch('[0-9a-f]{40}', args.expected_source_commit):
        parser.error('expected-source-commit must be a full lowercase SHA')
    wheel, output = args.wheel.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / "results.json"
    if receipt.exists():
        parser.error("use a fresh output directory")
    digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
    with zipfile.ZipFile(wheel) as archive:
        models = json.loads(archive.read("mojolearn/verify_reference/models/models.json"))["models"]
        source_commit = archive.read('mojolearn/identity_columns/COMMIT').decode().strip()
        if not re.fullmatch('[0-9a-f]{40}', source_commit):
            parser.error('wheel has no valid packaged source commit')
        if args.expected_source_commit and source_commit != args.expected_source_commit:
            parser.error('wheel source commit differs from requested expanded candidate')
    plugins = []
    version = wheel.name.split("-")[1]
    for plugin in (p.resolve() for p in args.plugin):
        parts = plugin.name.split("-")
        if not (plugin.is_file() and parts[0] in PLUGIN_DISTRIBUTIONS and len(parts) >= 5 and parts[1] == version):
            parser.error(f"--plugin {plugin.name} is not a mojolearn_nvidia/mojolearn_amd wheel of version {version}")
        plugins.append(dict(wheel=str(plugin), wheel_sha256=hashlib.sha256(plugin.read_bytes()).hexdigest(),
                            distribution=PLUGIN_DISTRIBUTIONS[parts[0]]))
    if len({p["distribution"] for p in plugins}) != len(plugins):
        parser.error("one --plugin per distribution")
    manifest = dict(wheel=str(wheel), wheel_sha256=digest, status="INCOMPLETE", jobs=[],
                    scope=args.scope, source_commit=source_commit, release_qualified=False,
                    expanded=None)
    if plugins:
        manifest["plugins"] = plugins
    save, run = runner(manifest, receipt, output)
    save()
    try:
        with tempfile.TemporaryDirectory(prefix="mojolearn-verifier-wheel-") as directory:
            work = Path(directory)
            run("create-venv", [args.python, "-m", "venv", str(work / "venv")], work)
            python = str(work / "venv/bin/python")
            offline = ["--no-index", "--find-links", str(args.wheelhouse.resolve())] if args.wheelhouse else []
            run("install", [python, "-m", "pip", "install", "--no-input", "--only-binary=:all:",
                            *offline, str(wheel), *[p["wheel"] for p in plugins], "numpy>=1.24"], work)
            run("dependencies", [python, "-m", "pip", "check"], work)
            qualify_jobs(run, save, manifest, python, work, output, models, args.scope, devices,
                         wheel.name.split("-")[1])
            assert hashlib.sha256(wheel.read_bytes()).hexdigest() == digest, "wheel changed"
            for p in plugins:
                assert hashlib.sha256(Path(p["wheel"]).read_bytes()).hexdigest() == p["wheel_sha256"], "plugin changed"
            manifest["status"] = "PASSED" if args.scope == "expanded" else "PASSED_CPU_ONLY"
    except Exception as exc:
        manifest.update(status="FAILED", reason=repr(exc))
    save()
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] in ("PASSED", "PASSED_CPU_ONLY") else 1


if __name__ == "__main__":
    raise SystemExit(main())
