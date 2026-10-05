#!/usr/bin/env python3
"""On-pod assertions for the automatic IDENTICAL PTX fallback; grants nothing.

Staged by tools/nvidia_ptx_fallback_stage.py and run inside fresh venvs on a
device with no native payload, with no MOJOLEARN_CUDA_PATH and no
MOJOLEARN_EXPERIMENTAL_PTX. Standard library plus the installed package only.

  select  the unforced import took the admitted fallback (selection receipt)
  column  the canonical column was produced by the bundled PTX payload
  refuse  a negative install raises GpuPluginError and never selects CPU
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

FORCING = ('MOJOLEARN_CUDA_PATH', 'MOJOLEARN_EXPERIMENTAL_PTX', 'MOJOLEARN_GPU_ARCH', 'MOJOLEARN_VENDOR')
REFUSED = 'IDENTICAL PTX fallback refused'
REASONS = {'negative-config': 'this device/driver configuration is not admitted',
           'negative-unbundled': 'NVIDIA vendor wheel carries no admitted PTX fallback'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unforced(environ):
    present = [name for name in FORCING if environ.get(name)]
    require(not present, 'A forcing variable is set: ' + ','.join(present))


def check_selection(receipt, plugin, state, expect, manifest):
    """The loader's own receipt must say the admitted fallback was taken."""
    require(isinstance(receipt, dict) and receipt.get('schema') == 'mojolearn.ptx-baseline-selection.v1',
            'No PTX selection receipt: the fallback was not taken')
    require(receipt.get('requested') == 'native-first' and receipt.get('selected') == 'ptx-baseline'
            and receipt.get('native_fallback') is False and receipt.get('code_format') == 'ptx-baseline'
            and receipt.get('identical_qualified') is True,
            'Selection receipt is not the admitted native-first fallback')
    require(receipt.get('source_commit') == expect['source_commit']
            and receipt.get('manifest_sha256') == expect['manifest_sha256']
            and receipt.get('admission_sha256') == expect['admission_sha256'],
            'Selection receipt names another source, manifest or admission')
    require(receipt.get('configuration') == expect['configuration'],
            'Selected device configuration differs from the admitted one')
    require(plugin.get('code_format') == 'ptx-baseline' and plugin.get('identical_qualified') is True
            and plugin.get('distribution') == 'mojolearn-nvidia', 'Vendor plugin state differs')
    require(state == dict(vendor='cuda', numeric_mode='identical', gpu_arch='sm_80'),
            'Not CUDA IDENTICAL on the sm_80 PTX set: ' + json.dumps(state, sort_keys=True))
    files = {row['file']: row['sha256'] for row in manifest.get('files', [])}
    loaded = receipt.get('loaded_files', [])
    require(loaded and all(files.get(row.get('file')) == row.get('sha256') for row in loaded),
            'No GPU binding loaded from the bundled PTX payload')
    return dict(loaded_files=len(loaded))


def check_column(column, expect, manifest):
    """The column's GPU bindings are the PTX payload: not native, not host."""
    require(column.get('complete') is True and column.get('commit') == expect['source_commit']
            and column.get('mode') == 'identical' and not column.get('skipped'),
            'Incomplete, wrong-source or wrong-mode column')
    require(set(column.get('fixtures', {})) == {'base', 'denormal', 'odd'} and column.get('cells'),
            'Column is not the canonical three fixtures')
    package = column.get('package', {})
    require(not package.get('bindings_error'), 'Binding provenance failed')
    payload = {row['sha256'] for row in manifest.get('files', [])}
    gpu = [row for row in package.get('bindings', []) if not row['module'].endswith('_host')]
    require(gpu and all(row['sha256'] in payload for row in gpu),
            'A GPU binding in the column is not bundled PTX payload')
    return dict(cells=len(column['cells']), gpu_bindings=len(gpu))


def check_refusal(case, error_type, message, second):
    """Refused by name with the expected reason, in-process and in a fresh process."""
    require(case in REASONS, 'Unknown negative case')
    require(error_type == 'GpuPluginError', 'Import did not raise GpuPluginError: ' + str(error_type))
    require(REFUSED in message and REASONS[case] in message, 'Refusal names another reason')
    require(second['returncode'] != 0 and 'GpuPluginError' in second['stderr']
            and REASONS[case] in second['stderr'], 'A fresh process did not refuse the same way')
    require('cpu-only' not in message.lower() and 'cpu-only' not in second['stderr'].lower(),
            'Refusal mentions a CPU-only selection')
    return dict(case=case, reason=REASONS[case])


def write(path, document):
    with Path(path).open('x') as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
        stream.write('\n')


def select(args):
    unforced(os.environ)
    expect = json.loads(args.expect.read_text())
    import mojolearn as ml
    from mojolearn import _backend
    package = Path(ml.__file__).resolve().parent
    manifest_path = package / 'cuda_ptx/sm_80/PTX_BASELINE.json'
    require(sha(manifest_path) == expect['manifest_sha256']
            and sha(package / 'cuda_ptx/sm_80/PTX_IDENTITY_ADMISSION.json') == expect['admission_sha256'],
            'Installed bundle differs from the staged admission')
    receipt = _backend.baseline_selection_receipt()
    state = dict(vendor=ml.vendor(), numeric_mode=ml.numeric_mode(), gpu_arch=_backend.gpu_arch())
    result = check_selection(receipt, _backend.gpu_plugin() or {}, state, expect,
                             json.loads(manifest_path.read_text()))
    write(args.out, dict(schema='mojolearn.ptx-fallback-e2e.v1', check='select', passed=True,
                         receipt=receipt, state=state, how=_backend.gpu_arch_how(), **result))


def column(args):
    expect = json.loads(args.expect.read_text())
    import sysconfig
    manifest_path = Path(sysconfig.get_paths()['purelib']) / 'mojolearn/cuda_ptx/sm_80/PTX_BASELINE.json'
    require(sha(manifest_path) == expect['manifest_sha256'], 'Installed manifest differs')
    result = check_column(json.loads(args.column.read_text()), expect, json.loads(manifest_path.read_text()))
    write(args.out, dict(schema='mojolearn.ptx-fallback-e2e.v1', check='column', passed=True,
                         column_sha256=sha(args.column), **result))


def refuse(args):
    unforced(os.environ)
    error_type, message = None, ''
    try:
        import mojolearn  # noqa: F401
    except BaseException as exc:  # the type is the assertion
        error_type, message = type(exc).__name__, str(exc)
    fresh = subprocess.run([sys.executable, '-c', 'import mojolearn'], capture_output=True, text=True, timeout=100)
    second = dict(returncode=fresh.returncode, stderr=fresh.stderr[-4000:])
    document = dict(schema='mojolearn.ptx-fallback-e2e.v1', check='refuse', error_type=error_type,
                    message=message[-4000:], fresh_process=second, passed=False)
    try:
        document.update(check_refusal(args.case, error_type, message, second), passed=True)
    finally:
        write(args.out, document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('select', 'column'):
        command = commands.add_parser(name)
        command.add_argument('--expect', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True)
        if name == 'column':
            command.add_argument('--column', type=Path, required=True)
    command = commands.add_parser('refuse')
    command.add_argument('--case', choices=sorted(REASONS), required=True)
    command.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    dict(select=select, column=column, refuse=refuse)[args.command](args)
    print(json.dumps(dict(check=args.command, passed=True)))


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        raise SystemExit('FAILED: ' + str(exc))
