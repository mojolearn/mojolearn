"""Supplemental same-process native/PTX capture; never grants qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def witness(ml, backend, loaded, manifest_path, role, qualifier):
    manifest = json.loads(Path(manifest_path).read_text())
    source = manifest['source_commit']
    expected = 'ptx-baseline' if role == 'baseline' else 'native'
    plugin = backend.gpu_plugin() or {}
    require(ml.vendor() == 'cuda' and ml.numeric_mode() == 'identical', 'CUDA IDENTICAL required')
    require(plugin.get('code_format') == expected, 'Requested native/PTX selection differs')
    rows = [dict(row) for row in loaded]
    require(rows, 'No loaded bindings witnessed')
    files = qualifier.manifest_files(manifest)
    for row in rows:
        require(sha(row['file']) == row['sha256'], 'Loaded binding hash differs')
        if role == 'baseline' and not row['module'].endswith('_host'):
            relative = Path(row['file']).resolve().relative_to(Path(backend._BASELINE_ROOT).resolve()).as_posix()
            require(relative in files and files[relative]['sha256'] == row['sha256'],
                    'Loaded binding is outside baseline manifest')
            row['file'] = relative
    installation = qualifier.installed_source_evidence(Path(ml.__file__).resolve().parent,
                                                       backend, role, source, rows)
    runtime = None
    if role == 'baseline':
        runtime = backend.baseline_selection_receipt()
        require(runtime.get('schema') == 'mojolearn.ptx-baseline-selection.v1'
                and runtime.get('requested') == expected and runtime.get('selected') == expected
                and runtime.get('native_fallback') is False
                and runtime.get('manifest_sha256') == sha(manifest_path)
                and runtime.get('source_commit') == source, 'Invalid forced PTX selection receipt')
        require({(r['file'], r['sha256']) for r in runtime.get('loaded_files', [])}
                == {(r['file'], r['sha256']) for r in rows if not r['module'].endswith('_host')},
                'Forced PTX receipt differs from actual loaded bindings')
    return dict(source_commit=source, code_format=expected, gpu_arch=backend.gpu_arch(),
                plugin=plugin, installed_source=installation, loaded_bindings=rows, runtime=runtime)


def run(args):
    # Qualification code remains from the exact payload source checkout.
    sys.path.insert(0, str(args.source_tools.resolve()))
    import nvidia_baseline_qualification as qualifier
    import mojolearn as ml
    from mojolearn import _backend
    from mojolearn._verify import binding_artifacts
    require(not args.out.exists(), 'Capture receipt already exists')
    raw = args.out.with_suffix('.capture.json')
    require(not raw.exists(), 'Capture data already exists')
    report = dict(schema='mojolearn.nvidia-extra-capture.v1', role=args.role,
                  capture_script_sha256=sha(args.script), wrapper_sha256=sha(__file__),
                  manifest_sha256=sha(args.manifest), qualification=False, complete=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    original = sys.argv
    try:
        expected = 'ptx-baseline' if args.role == 'baseline' else 'native'
        require(ml.vendor() == 'cuda' and ml.numeric_mode() == 'identical', 'CUDA IDENTICAL required')
        require((_backend.gpu_plugin() or {}).get('code_format') == expected,
                'Requested native/PTX selection differs')
        qualifier.installed_source_evidence(Path(ml.__file__).resolve().parent, _backend,
                                           args.role, json.loads(args.manifest.read_text())['source_commit'], [])
        sys.argv = [str(args.script), str(raw)]
        runpy.run_path(str(args.script), run_name='__main__')
        require(raw.is_file(), 'Capture script did not write its data')
        report['capture'] = json.loads(raw.read_text())
        report['capture_sha256'] = sha(raw)
        report['witness'] = witness(ml, _backend, binding_artifacts(), args.manifest, args.role, qualifier)
        report['complete'] = True
    except BaseException:
        report['error'] = traceback.format_exc()
        # Preserve partial data and independently attempt a loaded-byte witness.
        if raw.is_file():
            report['capture_sha256'] = sha(raw)
        try:
            report['witness'] = witness(ml, _backend, binding_artifacts(), args.manifest, args.role, qualifier)
        except Exception:
            report['witness_error'] = traceback.format_exc()
    finally:
        sys.argv = original
        args.out.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['complete'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--script', type=Path, required=True)
    p.add_argument('--source-tools', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--role', choices=['native-reference', 'baseline'], required=True)
    p.add_argument('--out', type=Path, required=True)
    return run(p.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
