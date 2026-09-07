#!/usr/bin/env python3
"""Root-only preparation; --run explicitly permits one guarded Metal step.

No installs, Pixi activation, cache reuse, full training, or automatic retry.
Authored source: root must review and test before executing on the Mac.
"""
import argparse
import configparser
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shlex
import signal
import subprocess
import sys

PIN = '45cc2d9dcd2279646f9a9716fdfab9d598320535'
PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
TOOLS = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    raw = value if isinstance(value, bytes) else (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def inventory(source):
    paths = set()
    for directory in ('checks', 'core', 'gemm', 'embedding', 'transformer', 'training', 'mamba'):
        paths.update((source / directory).rglob('*.mojo'))
    paths.update(source / name for name in (
        'bindings/_mojolearn_byte_lm.mojo', 'bindings/build_byte_lm.sh',
        'python/mojolearn/_byte_lm_impl.py', 'python/mojolearn/language_model.py',
        'tools/byte_lm_real_text_capture.py', 'tools/byte_lm_gradient_oracle.py', 'pixi.toml', 'pixi.lock'))
    require(all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(source) for p in paths),
            'Missing/aliased numerical source')
    return {p.relative_to(source).as_posix(): sha(p.read_bytes()) for p in sorted(paths)}


def private_config(environment, out):
    original = environment / 'share/max/modular.cfg'
    raw = original.read_bytes()
    require(len(raw) <= 32768, 'Unexpected MAX configuration size')
    cfg = configparser.ConfigParser(interpolation=None, strict=True)
    cfg.read_string(raw.decode())
    require(set(cfg.sections()) == {'max', 'mojo-max'}, 'Unexpected MAX configuration sections')
    expected_max = {'package_root', 'cache_dir', 'enable_model_ir_cache', 'name', 'path', 'version'}
    expected_mojo = {'package_root', 'compilerrt_path', 'mgprt_path', 'shared_libs', 'driver_path',
                     'import_path', 'jupyter_path', 'lldb_path', 'lldb_plugin_path',
                     'lldb_visualizers_path', 'lldb_vscode_path', 'lsp_server_path', 'mblack_path',
                     'repl_entry_point', 'lld_path'}
    require(set(cfg['max']) == expected_max and set(cfg['mojo-max']) == expected_mojo,
            'Unexpected MAX configuration keys')
    require(cfg['max']['package_root'] == cfg['max']['path'] == str(environment)
            and cfg['max']['cache_dir'] == str(environment / 'share/max/.max_cache')
            and cfg['max']['enable_model_ir_cache'] == 'true', 'Unexpected MAX package/cache configuration')
    for key, value in cfg['mojo-max'].items():
        if key == 'shared_libs':
            require(value == '-Xlinker,-rpath,-Xlinker,' + str(environment / 'lib') + ';',
                    'Unexpected shared runtime library flags')
        else:
            path = Path(value.removesuffix(';'))
            require(path.is_absolute() and path.resolve().is_relative_to(environment) and path.exists(),
                    'Runtime path escapes/missing: ' + key)
    require(cfg['mojo-max']['driver_path'] == str(environment / 'bin/mojo'), 'Unexpected compiler path')
    home = out / 'runtime'
    home.mkdir()
    (home / '.max_cache').mkdir()
    (home / 'cache').mkdir()
    cfg['max']['cache_dir'] = str(home / '.max_cache')
    write(out / 'original-modular.cfg', raw)
    with (home / 'modular.cfg').open('x') as stream:
        cfg.write(stream)
    return home


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--common-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-policy', choices=('standard', 'user-tiny'), default='standard')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    source, environment = args.source.resolve(strict=True), args.environment.resolve(strict=True)
    out = args.output.absolute()
    require(not out.exists() and not out.is_symlink() and out.parent.resolve() == out.parent,
            'Output must be fresh with canonical existing parent')
    out.mkdir(mode=0o700)
    common_raw = args.common_source.read_bytes()
    require(len(common_raw) <= 256 * 1024, 'Common source manifest exceeds bound')
    common = json.loads(common_raw)
    require(common.get('schema') == 'mojolearn.byte-lm.expanded-common-source.v1'
            and common.get('metal_commit') == PIN and common.get('source_file_count') == 258
            and len(common.get('files', {})) == 258, 'Wrong common source profile/pin/count')
    files = inventory(source)
    require(files == common['files'], 'Full expanded source inventory differs')
    actual_pin = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                         text=True, timeout=10).strip()
    require(actual_pin == PIN, 'Unexpected source checkout commit')
    python, compiler = environment / 'bin/python', environment / 'bin/mojo'
    require(all(p.is_file() and os.access(p, os.X_OK) for p in (python, compiler)),
            'Existing Python/compiler executables required')
    require(not list((source / 'python/mojolearn').rglob('_mojolearn*.so')),
            'Use fresh snapshot without prior native bindings')
    home = private_config(environment, out)
    config_sha = sha((home / 'modular.cfg').read_bytes())
    metadata = out / 'installed-metadata'
    metadata.mkdir()
    for path in sorted((environment / 'conda-meta').glob('*.json')):
        if path.name.startswith(('mojo-', 'max-', 'max-core-', 'python-', 'numpy-')):
            write(metadata / path.name, path.read_bytes())
    write(out / 'common-source-expanded.json', common_raw)
    write(out / 'source.json', files)
    write(out / 'helper.py', Path(__file__).read_bytes())
    for name in ('macos_serial_guard.py', 'root_job_receipt.py'):
        write(out / name, (TOOLS / name).read_bytes())
    env = {'PATH': str(environment / 'bin') + ':/usr/bin:/bin:/usr/sbin:/sbin',
           'HOME': os.environ['HOME'], 'MODULAR_HOME': str(home), 'LC_ALL': 'C',
           'PYTHONPATH': str(source / 'python'), 'PYTHONNOUSERSITE': '1',
           'PYTHONUNBUFFERED': '1', 'MOJOLEARN_NUMERIC_MODE': 'identical',
           'MOJOLEARN_TARGET_COLUMN': 'apple'}
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MOJOLEARN_CPU_THREADS',
                'MAX_JOBS', 'CMAKE_BUILD_PARALLEL_LEVEL', 'MOJOLEARN_COMPILE_JOBS'):
        env[key] = '2'
    write(out / 'environment.json', env)
    write(out / 'preparation.json', dict(status='PREPARED_NOT_EXECUTED', source_commit=PIN,
        source_count=258, run_requested=args.run, full128_enabled=False,
        cache_isolation='Fresh configured paths; runtime honoring configuration requires root review'))
    if not args.run:
        return 0
    require(platform.system() == 'Darwin' and platform.machine() == 'arm64', 'Apple silicon Darwin required')
    guard = load_module(out / 'macos_serial_guard.py', 'apple_first_step_guard')
    initial = guard.memory_state()  # Before ANY candidate compiler/Python runtime probe.
    write(out / 'initial-memory.json', initial)
    entry_reserve = (0 if args.memory_policy == 'user-tiny' else 4) * 2**30
    if initial['pressure_level'] != 1 or initial['conservative_reserve_bytes'] < entry_reserve:
        blocked = dict(status='BLOCKED_MEMORY_BEFORE_LAUNCH',
                       required_reserve_bytes=entry_reserve, observed=initial,
                       compiler_launched=False, model_launched=False)
        write(out / 'admission.json', blocked)
        print(json.dumps(blocked), flush=True)
        return 2
    receipt = load_module(out / 'root_job_receipt.py', 'apple_first_step_receipt')

    def run(name, seconds, rss, command):
        argv = [sys.executable, str(out / 'macos_serial_guard.py'), '--seconds', str(seconds),
                '--rss-gib', str(min(rss, 2) if args.memory_policy == 'user-tiny' else rss), '--memory-policy', args.memory_policy, '--report', str(out / (name + '.guard.json')), '--', *map(str, command)]
        write(out / (name + '.argv.json'), argv)
        write(out / (name + '.command.txt'), (shlex.join(argv) + '\n').encode())
        with (out / (name + '.log')).open('xb') as log:
            proc = subprocess.Popen(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            prior = {}
            def interrupted(signum, frame):
                proc.send_signal(signum)
                raise InterruptedError('Root helper interrupted')
            try:
                for signum in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
                    prior[signum] = signal.signal(signum, interrupted)
                code = proc.wait(timeout=seconds + 45)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=30)
                for signum, handler in prior.items():
                    signal.signal(signum, handler)
                write(out / (name + '.exit_code'), (str(proc.returncode) + '\n').encode())
        require(code == 0, name + ' failed; no retry')
        terminal = json.loads((out / (name + '.guard.json')).read_bytes())
        receipt.validate_guard_terminal(terminal, 'metal', code, 'capture')
        require(inventory(source) == files, 'Source changed during job')
        require(sha((home / 'modular.cfg').read_bytes()) == config_sha, 'Private runtime configuration changed')
        return code

    run('compiler-version', 60, 2, [compiler, '--version'])
    probe = ('import json,sys,platform,numpy; print(json.dumps(dict(executable=sys.executable,'
             'python=sys.version,platform=platform.platform(),numpy=numpy.__version__,numpy_file=numpy.__file__)))')
    run('python-runtime', 60, 2, [python, '-c', probe])
    run('sdk', 60, 2, ['/usr/bin/xcrun', '--sdk', 'macosx', '--show-sdk-version'])
    sdk_lines = (out / 'sdk.log').read_text().splitlines()
    sdk = [line.strip() for line in sdk_lines if re.fullmatch(r'[0-9]+(?:\.[0-9]+){1,2}', line.strip())]
    require(len(sdk) == 1, 'Unexpected SDK readback')
    binary = out / '_mojolearn_byte_lm.so'
    run('build', 180, 4, [compiler, 'build', '-j', '2', '--emit', 'shared-lib', '--target-cpu', 'apple-m1',
        '-D', 'MOJOLEARN_COLUMN_APPLE', '-Xlinker', '-platform_version', '-Xlinker', 'macos',
        '-Xlinker', '11.0', '-Xlinker', sdk[0], '-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1',
        '-I', '.', '-I', 'bindings', 'bindings/_mojolearn_byte_lm.mojo', '-o', binary])
    air = ('import pathlib,re,json,hashlib,sys; p=pathlib.Path(sys.argv[1]); b=p.read_bytes(); '
           'names=sorted(set(x.decode() for x in re.findall(rb"[0-9A-Za-z_]+_[0-9a-f]{16}air",b))); '
           'assert names,"No retained Metal AIR symbols"; '
           'print(json.dumps(dict(binary_sha256=hashlib.sha256(b).hexdigest(),air_symbols=names)))')
    run('air', 60, 2, [python, '-c', air, binary])
    binary_sha = sha(binary.read_bytes())
    destination = source / 'python/mojolearn/identical/_mojolearn_byte_lm.so'
    destination.parent.mkdir(exist_ok=True)
    os.link(binary, destination)  # Exclusive; refuses any preexisting destination.
    witness = ('import pathlib,sys,json,hashlib; from mojolearn import _mojolearn_byte_lm as b; '
        'p=pathlib.Path(b.__file__).resolve(); assert p==pathlib.Path(sys.argv[1]).resolve(); '
        'assert b.byte_lm_numeric_mode()==1 and b.byte_lm_vendor()=="metal"; '
        'assert b.byte_lm_profile()==sys.argv[2]; '
        'print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),'
        'mode=b.byte_lm_numeric_mode(),vendor=b.byte_lm_vendor(),profile=b.byte_lm_profile())))')
    run('native-witness', 60, 2, [python, '-c', witness, destination, PROFILE])
    require(sha(destination.read_bytes()) == binary_sha, 'Published binary changed')
    code = run('step1', 180, 4, [python, source / 'tools/byte_lm_real_text_capture.py',
        '--expected-vendor', 'metal', '--steps', '1', '--action', 'continuous', '--output', out / 'step1'])
    old_argv = sys.argv
    try:
        sys.argv = ['root_job_receipt.py', '--vendor', 'metal', '--job-kind', 'capture',
            '--exit-code', str(code), '--command-file', str(out / 'step1.command.txt'),
            '--guard-log', str(out / 'step1.log'), '--result', str(out / 'step1/summary.json'),
            '--output', str(out / 'step1.root-receipt.json')]
        receipt.main()
    finally:
        sys.argv = old_argv
    require(sha(destination.read_bytes()) == binary_sha, 'Binary changed during capture')
    write(out / 'completion.json', dict(status='ONE_STEP_CAPTURED_NOT_QUALIFIED', source_commit=PIN,
        receipt='step1.root-receipt.json', full128_executed=False, independent_metal_oracle=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
