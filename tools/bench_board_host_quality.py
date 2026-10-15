"""Untimed, own-implementation host references for opponent-free boards."""
import json
import os
import re
from pathlib import Path
import subprocess
import sys


def compare(actual, reference):
    import numpy as np
    if not actual or not reference:
        raise ValueError('Host reference output is empty')
    if actual.keys() != reference.keys():
        raise ValueError('Host reference output fields differ: actual=%s reference=%s'
                         % (sorted(actual), sorted(reference)))
    errors = []
    for key in actual:
        a, b = np.asarray(actual[key]), np.asarray(reference[key])
        if a.shape != b.shape or not a.size:
            raise ValueError('Host reference output shape differs: ' + key)
        if a.dtype.kind not in 'biuf' or b.dtype.kind not in 'biuf':
            raise ValueError('Non-numeric host reference output: ' + key)
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Nonfinite host reference output: ' + key)
        # Integer outputs (tokens, indices, dimensions) have an exact contract.
        if a.dtype.kind in 'biu' and b.dtype.kind in 'biu':
            if not np.array_equal(a, b):
                raise ValueError('Integer host reference mismatch: ' + key)
            errors.append(0.)
            continue
        a, b = a.astype(np.float64), b.astype(np.float64)
        denominator = max(float(np.linalg.norm(b)), 1e-30)
        errors.append(float(np.linalg.norm(a - b)) / denominator)
    # Report numerical distance as quality, not identity qualification.
    # FAST has no cross-vendor bitwise promise. Nonzero error remains visible.
    return {'relative_error_vs_own_host': max(errors)}


# The two refusals a host reference prints when the race job built no host twin for a binding it loads
# (python/mojolearn/_backend.py load_host_module: "<dir>/_mojolearn_<x>_host.so is not built. Build it with
# bindings/build_<x>_host.sh"; _no_cpu_implementation: "no host binding covers _mojolearn_<x>").
# The base binding `_mojolearn` is covered too ("no host binding covers _mojolearn;"): a layer constructor draws its
# init through `_buffer.InitStream` -> `_mojolearn.uniform_init_f32`, which the CPU vendor serves from
# `_mojolearn_core_host` (host_surface family "core"). Until 2026-10-12 the regex required `_mojolearn_<x>`, so that
# refusal named nothing, the box never built the core twin, and resnet-block's reference failed on every run
# (nv2 v0140, amd a1721: "Own host reference failed (... no GPU set is loaded, on request (flat layout))").
_NOT_BUILT_RE = re.compile(r'(_mojolearn_[A-Za-z0-9_]+?_host)\.so is not built')
_NO_COVER_RE = re.compile(r'no host binding covers (_mojolearn(?:_[A-Za-z0-9_]+)?)\b')
_HOST_SURFACE = Path(__file__).resolve().parent.parent / 'python' / 'mojolearn' / 'host_surface.py'


def _routed_host_bindings():
    """Device binding name -> host twin basename (python/mojolearn/host_surface.py routed_modules, loaded by path:
    the table `_backend._HOST_MODULES` routes a CPU-only install through). Empty when the file cannot be loaded."""
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('bbhq_host_surface', _HOST_SURFACE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return dict(module.routed_modules())
    except Exception:  # noqa: BLE001  (a tree without the manifest falls back to <name>_host)
        return {}


def failure_line(log_path, width=300):
    """The last non-empty line of a failed reference's host.log (its exception), for the quality error text."""
    try:
        lines = [x.strip() for x in Path(log_path).read_text(errors='replace').splitlines() if x.strip()]
    except OSError:
        return 'host.log unreadable'
    return (lines[-1] if lines else 'host.log empty')[:width]


def missing_host_bindings(log_path):
    """Host binding basenames (`_mojolearn_<x>_host`) a failed reference could not load, in log order.

    Text matching on our own refusal messages only; no data is read. The race prints one
    `needs python/mojolearn/host/<basename>.so` line per name, the line the lq box job's retry loop
    (overlay_race_job2.sh) reads to build `bindings/build_<x>_host.sh` and race again."""
    try:
        text = Path(log_path).read_text(errors='replace')
    except OSError:
        return []
    out = []
    for m in _NOT_BUILT_RE.finditer(text):
        if m.group(1) not in out:
            out.append(m.group(1))
    routed = _routed_host_bindings() if _NO_COVER_RE.search(text) else {}
    for m in _NO_COVER_RE.finditer(text):
        name = routed.get(m.group(1)) or m.group(1) + '_host'
        if name not in out:
            out.append(name)
    return out


class ScorerUnavailable(RuntimeError):
    """The own-host reference (this lane's quality scorer) cannot run: the race is refused before any timing."""


def _child_env():
    env = dict(os.environ, MOJOLEARN_VENDOR='cpu', MOJOLEARN_NUMERIC_MODE='identical')
    for key in ('PYTHONPATH',
                'MOJOLEARN_GPU_ARCH', 'MOJOLEARN_BOARD_ARTIFACT_MANIFEST', 'MOJOLEARN_BOARD_RECEIPTS'):
        env.pop(key, None)
    # The race tree imports mojolearn through PYTHONPATH (not an installed wheel), so the host child gets the parent's
    # package root back, and only that (2026-10-09: every optimizer cell on AMD a1141 failed 'No module named mojolearn').
    import importlib.util
    spec = importlib.util.find_spec('mojolearn')
    if spec is not None and spec.origin:
        env['PYTHONPATH'] = str(Path(spec.origin).resolve().parent.parent)
    return env


def reference(lane, inputs, python, directory, timeout, *, fit_calls=1):
    """Run the untimed own-host reference on the race's inputs: (reference outputs, receipt path).

    The reference depends on the inputs only, never on a timed arm's outputs, so the race driver runs it BEFORE
    its first timed round (the scorer preflight, lane harness-evidence 2026-10-14): a reference that cannot run
    (missing neural fixtures, a missing host binding, an import error) refuses the race up front instead of
    leaving a timed cell unscored. Raises ScorerUnavailable with the child's own failure line."""
    import numpy as np
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    data, ref, receipt = root / 'inputs.npz', root / 'host.npz', root / 'receipt.json'
    try:
        np.savez(data, **inputs)
        with (root / 'host.log').open('w') as log:
            result = subprocess.run([python, str(Path(__file__).resolve()), lane,
                                     str(data), str(ref), str(receipt), str(fit_calls)],
                                    env=_child_env(), stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
    except Exception as error:  # noqa: BLE001  (a timeout or an unwritable work dir: the scorer did not run)
        raise ScorerUnavailable('Own host reference failed (%r); see %s' % (error, root / 'host.log'))
    if result.returncode:
        raise ScorerUnavailable('Own host reference failed (%s); see %s'
                                % (failure_line(root / 'host.log'), root / 'host.log'))
    try:
        proof = json.loads(receipt.read_text())
    except (OSError, ValueError) as error:
        raise ScorerUnavailable('Missing own host reference provenance (%r)' % (error,))
    if proof.get('vendor') != 'cpu' or proof.get('numeric_mode') != 'identical' or not proof.get('bindings'):
        raise ScorerUnavailable('Missing own host reference provenance')
    with np.load(ref, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}, str(receipt)


def enrich(lane, inputs, outputs, quality, python, directory, timeout, *, fit_calls=1, precomputed=None):
    """Only replace explicitly empty quality, never an existing quality error.

    `precomputed`: (reference outputs, receipt path) from `reference()` run before the timed rounds; without it
    the reference runs now. Each arm's entry says whether the scorer ran: `scorer: "scored"` (a distance, or a
    scored output failure such as a mismatch) or `scorer: "failed"` (the reference itself could not run)."""
    missing = [arm for arm in outputs if arm.startswith('ours') and quality.get(arm) == {}]
    # SCORER GAPS (lane quality-scorers, 2026-10-15): a race of ours now also scores against the untimed torch
    # reference (bench_board_algos.independent_quality_reference), so ours' quality is no longer empty. The own-host
    # distance is still recorded beside it; a reference that cannot run is noted without hiding the torch metric.
    beside = [arm for arm in outputs if arm.startswith('ours') and quality.get(arm)
              and 'error' not in quality[arm] and 'scorer' not in quality[arm]
              and 'relative_error_vs_own_host' not in quality[arm]
              and any(key in quality[arm] for key in TORCH_REFERENCE_METRICS)]
    if not missing and not beside:
        return None
    try:
        ref_outputs, receipt = precomputed or reference(lane, inputs, python, directory, timeout,
                                                        fit_calls=fit_calls)
        for arm in missing:
            try:
                quality[arm] = dict(compare(outputs[arm], ref_outputs), scorer='scored')
            except Exception as error:
                quality[arm] = {'error': str(error), 'scorer': 'scored'}
        for arm in beside:
            try:
                quality[arm].update(compare(outputs[arm], ref_outputs))
            except Exception as error:     # a mismatch is a scored failure of the arm, as above
                quality[arm].update(error=str(error))
            quality[arm]['scorer'] = 'scored'
    except Exception as error:
        receipt = str(Path(directory) / 'receipt.json')
        for arm in missing:
            quality[arm] = {'error': str(error), 'scorer': 'failed', 'scorer_error': str(error)}
        for arm in beside:
            quality[arm]['own_host_reference_error'] = str(error)[:600]
    return receipt


#: the metrics bench_board_algos.quality records against the torch reference (kinds layer and optim)
TORCH_REFERENCE_METRICS = ('rel_fro_vs_torch_eager_fp32', 'max_rel_diff_vs_torch_eager_fp32',
                           'optimizer_update_error_vs_torch_fp32')


def host_binding_artifacts(modules=None):
    """Hash actual loaded files, including the host loader's private aliases."""
    import hashlib
    modules = sys.modules if modules is None else modules
    rows = []
    for name, module in list(modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if not path.name.startswith('_mojolearn') or path.suffix != '.so':
            continue
        if not path.name.endswith('_host.so'):
            raise RuntimeError('Reference loaded a non-host binding: ' + str(path))
        rows.append({'module': name, 'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                     'size': path.stat().st_size, 'artifact_source_commit': 'unverified'})
    if not rows:
        raise RuntimeError('Reference loaded no host binding')
    return sorted(rows, key=lambda row: row['module'])


def run_reference(runner, fit_calls):
    """Replay the same warmup/sample history for stateful in-place calls."""
    if isinstance(fit_calls, bool) or not isinstance(fit_calls, int) or fit_calls < 1:
        raise ValueError('Host reference fit_calls must be a positive integer')
    for _ in range(fit_calls):
        runner.fit()
        runner.infer()
    return runner.outputs()


def main():
    import numpy as np
    import mojolearn as ml
    import bench_board_algos as board
    lane, data, output, receipt = sys.argv[1:5]
    fit_calls = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    if ml.vendor() != 'cpu' or ml.numeric_mode() != 'identical':
        raise RuntimeError('Reference must use our explicit IDENTICAL host column')
    with np.load(data, allow_pickle=False) as saved:
        inputs = {key: saved[key] for key in saved.files}
    runner = board.build(lane, 'ours', inputs)
    np.savez(output, **run_reference(runner, fit_calls))
    bindings = host_binding_artifacts()
    Path(receipt).write_text(json.dumps({'vendor': ml.vendor(), 'numeric_mode': ml.numeric_mode(),
                                        'lane': lane, 'bindings': bindings, 'timed': False, 'fit_calls': fit_calls}, indent=2) + '\n')


if __name__ == '__main__':
    main()
