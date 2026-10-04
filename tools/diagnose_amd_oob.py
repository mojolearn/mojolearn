#!/usr/bin/env python3
"""Capture AMD Bagging OOB failure without changing model code or its guard.

Run against the six exact installed candidate wheels (plus numpy):
  MOJOLEARN_NUMERIC_MODE=identical python diagnose_amd_oob.py \
    --source-commit FULL_SHA --out NEW_DIRECTORY
This is a diagnostic, never qualification. Nonzero diagnostic flags/mismatches
are retained in JSON/NPZ; successful collection does not mean kernels passed.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import re
import subprocess
import traceback


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def reference(acc, counts, y):
    """Independent CPU binary64 arithmetic reference; no GPU claim."""
    pred = [float(a) / max(int(c), 1) for a, c in zip(acc, counts)]
    total = math.fsum(float(v) for v in y)
    mean = total / len(y)
    centered = [float(v) - mean for v in y]
    residual = [float(v) - p for v, p in zip(y, pred)]
    return pred, [total, math.fsum(v * v for v in centered),
                  math.fsum(v * v for v in residual), mean]


def describe(a):
    import numpy as np
    a = np.ascontiguousarray(a)
    row = dict(dtype=a.dtype.str, shape=list(a.shape), sha256=digest(a.tobytes()),
               bytes=a.nbytes)
    if a.dtype.kind == 'f':
        bad = np.flatnonzero(~np.isfinite(a.reshape(-1)))
        row.update(finite=not len(bad), nonfinite_count=len(bad), nonfinite_indices=bad[:32].tolist())
        view = a.view('<u8' if a.dtype.itemsize == 8 else '<u4').reshape(-1)
        row['sample_bits'] = [hex(int(v)) for v in view[:32]]
    elif a.dtype.kind in 'iu':
        row.update(min=int(a.min()) if a.size else None, max=int(a.max()) if a.size else None,
                   zero_count=int(np.count_nonzero(a == 0)), sample_values=a.reshape(-1)[:32].tolist())
    return row


def direct(binding, acc, counts, y):
    """Public binding entry point with untouched input copies and write sentinels."""
    import numpy as np
    acc, counts, y = (np.array(acc, dtype='<f8', order='C', copy=True),
                      np.array(counts, dtype='<i4', order='C', copy=True),
                      np.array(y, dtype='<f4', order='C', copy=True))
    n = len(y)
    require(n > 0 and acc.shape == counts.shape == y.shape == (n,), 'Invalid diagnostic shapes')
    pred = np.full(n, 0x7FF8ABCDEF123456, dtype='<u8').view('<f8')
    words = np.full(4, 0x7FF8ABCDEF123456, dtype='<u8').view('<f8')
    flags = np.full(4, 0x13579BDF, dtype='<i4')
    arrays = dict(acc=acc, counts=counts, y=y, pred=pred, words=words, flags=flags)
    before = {k: v.tobytes() for k, v in arrays.items() if k in ('acc', 'counts', 'y')}
    report = {}
    try:
        binding.x_trees_oob_r2([a.ctypes.data for a in [acc, counts, y, pred, words, flags]], [n])
    except Exception:
        report['binding_error'] = traceback.format_exc()
    report.update(flags=[int(v) for v in flags], nonfinite_flag=int(flags[0]), overflow_flag=int(flags[1]),
                  words_bits=[hex(int(v)) for v in words.view('<u8')],
                  inputs_unchanged=all(arrays[k].tobytes() == b for k, b in before.items()))
    try:
        rp, rw = reference(np.frombuffer(before['acc'], dtype='<f8'),
                           np.frombuffer(before['counts'], dtype='<i4'),
                           np.frombuffer(before['y'], dtype='<f4'))
        arrays['reference_pred'] = np.array(rp, dtype='<f8')
        arrays['reference_words'] = np.array(rw, dtype='<f8')
        report.update(pred_bits_match=np.array_equal(pred.view('<u8'), arrays['reference_pred'].view('<u8')),
                      words_bits_match=np.array_equal(words.view('<u8'), arrays['reference_words'].view('<u8')))
    except Exception:
        report['reference_error'] = traceback.format_exc()
    for name, raw in before.items():
        arrays[name + '_before'] = np.frombuffer(raw, dtype=arrays[name].dtype).copy()
    report['arrays'] = {k: describe(v) for k, v in arrays.items()}
    return arrays, report


def provenance(ml, loaded):
    """Tie actual loaded extension bytes to installed native wheel inventories."""
    package = Path(ml.__file__).resolve().parent
    names = ['mojolearn', 'mojolearn-amd', 'mojolearn-amd-gfx942']
    source = (package / 'identity_columns/COMMIT').read_text().strip()
    records, owned = [], {}
    for name in names:
        dist = importlib.metadata.distribution(name)
        raw = dist.read_text('LINUX_PAYLOAD.json')
        require(raw, 'Missing native inventory: ' + name)
        doc = json.loads(raw)
        require(doc.get('schema') == 'mojolearn.linux-payload.v1' and doc.get('source_commit') == source,
                'Native inventory source differs: ' + name)
        split = doc.get('split', {})
        require(split.get('distribution') == name, 'Inventory distribution differs')
        hashes = dict(doc.get('extensions', {}))
        hashes.update({r['archive_path']: r['sha256'] for r in doc.get('host_native', {}).values()})
        for member in split.get('native_members', []):
            if member in hashes:
                require(member not in owned, 'Duplicate native ownership')
                owned[member] = hashes[member]
        records.append(dict(distribution=name, text_sha256=digest(raw.encode()), document=doc))
    for row in loaded:
        path = Path(row['file']).resolve()
        member = path.relative_to(package.parent).as_posix()
        require(digest(path.read_bytes()) == row['sha256'] == owned.get(member),
                'Loaded binding differs from installed native inventory: ' + member)
        row['installed_member'] = member
    return dict(source_commit=source, inventories=records, loaded_bindings=loaded)


def hardware():
    result = {}
    for key, argv in [('rocminfo', ['rocminfo']), ('rocm_smi', ['rocm-smi', '--showproductname', '--showdriverversion', '--showuniqueid'])]:
        try:
            run = subprocess.run(argv, capture_output=True, text=True, timeout=30)
            result[key] = dict(exit_code=run.returncode, stdout=run.stdout, stderr=run.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            result[key] = dict(error=str(exc))
    return result


def run(args):
    import numpy as np
    import mojolearn as ml
    from mojolearn import _backend, _identity_break
    from mojolearn._verify import binding_artifacts
    require(re.fullmatch('[0-9a-f]{40}', args.source_commit), 'Full source SHA required')
    source = (Path(ml.__file__).parent / 'identity_columns/COMMIT').read_text().strip()
    require(source == args.source_commit, 'Installed core source differs')
    require(ml.vendor() == 'hip' and ml.numeric_mode() == 'identical', 'AMD HIP IDENTICAL required')
    require(not args.out.exists(), 'Refusing to overwrite diagnostic evidence')
    args.out.mkdir(parents=True)
    report = dict(schema='mojolearn.amd-oob-diagnostic.v1', source_commit=source,
                  diagnostic_script_sha256=digest(Path(__file__).read_bytes()),
                  package_version=ml.__version__, vendor=ml.vendor(), numeric_mode=ml.numeric_mode(),
                  plugin=_backend.gpu_plugin(), hardware=hardware(), cases={},
                  qualification=False, complete=False)
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    def save(name, arrays, row):
        path = args.out / (name + '.npz')
        np.savez(path, **arrays)
        row['npz'] = path.name
        row['npz_sha256'] = digest(path.read_bytes())
        report['cases'][name] = row
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')

    class CapturingRegressor(ml.BaggingRegressor):
        def _oob_outputs(self, Xa, k, member_out):
            self.captured_members = []
            def capture(est, xs, m):
                values = member_out(est, xs, m)
                self.captured_members.append(np.asarray(values).copy())
                return values
            acc, counts = super()._oob_outputs(Xa, k, capture)
            self.captured_acc = np.asarray(acc).copy()
            self.captured_counts = np.asarray(counts).copy()
            return acc, counts

    binding = None
    for kind in ['base', 'denormal', 'odd']:
        X, yc, yr = _identity_break.fixture(kind)
        model = CapturingRegressor(ml.DecisionTreeRegressor(max_depth=5), n_estimators=6,
                                   max_samples=0.8, oob_score=True, random_state=7)
        row, arrays = {}, dict(X=X, y=yr)
        # Match the failing lane's preceding classifier work and allocation
        # history, rather than silently testing a different process sequence.
        classifier = ml.BaggingClassifier(ml.DecisionTreeClassifier(max_depth=5),
                    n_estimators=6, max_features=0.8, oob_score=True, random_state=7)
        try:
            classifier.fit(X, yc)
            row['classifier_fit_succeeded'] = True
            arrays['classifier_oob'] = np.asarray(classifier.oob_decision_function_).copy()
        except Exception:
            row.update(classifier_fit_succeeded=False, classifier_error=traceback.format_exc())
        try:
            model.fit(X, yr)
            row['fit_succeeded'] = True
        except Exception:
            row.update(fit_succeeded=False, fit_error=traceback.format_exc())
        if hasattr(model, 'captured_acc'):
            binding = model._bind()
            direct_arrays, result = direct(binding, model.captured_acc, model.captured_counts, yr)
            arrays.update(direct_arrays)
            row['r2'] = result
        for i, values in enumerate(getattr(model, 'captured_members', [])):
            arrays[f'member_{i:02}'] = values
        for i, rows in enumerate(getattr(model, '_oob_rows', [])):
            arrays[f'oob_rows_{i:02}'] = np.asarray(rows).copy()
        row['arrays'] = {k: describe(v) for k, v in arrays.items()}
        save('fixture-' + kind, arrays, row)
    require(binding is not None, 'No OOB binding reached; see retained fixture errors')
    for n in [1, 63, 64, 65, 72, 256, 4096, 4097]:
        for value in [0, 1]:
            arrays, row = direct(binding, np.zeros(n), np.zeros(n, dtype='<i4'), np.full(n, value, dtype='<f4'))
            save(f'zero-counts-n{n}-y{value}', arrays, row)
        counts = np.arange(n, dtype='<i4') % 7
        target = (np.arange(n) % 5 - 2).astype('<f4')
        arrays, row = direct(binding, counts * target.astype('<f8'), counts, target)
        save(f'mixed-counts-n{n}', arrays, row)
    report['provenance'] = provenance(ml, binding_artifacts())
    report['complete'] = True
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.out), cases=len(report['cases']), complete=True,
                          qualification=False)))
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--out', type=Path, required=True)
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
