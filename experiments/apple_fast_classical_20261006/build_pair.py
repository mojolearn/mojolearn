#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Future queued M2 builder for frozen Apple FAST classical A/B packages.

SOURCE ONLY DELIVERY: this program has not been compiled, run or checked.
It never imports a native binding, launches a kernel, checks model quality or
times an estimator. Existing build scripts retain their own static artifact
checks; their device smoke gates are explicitly disabled.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# The adjacent source-configuration tool is called select.py. Prevent it from
# shadowing the standard-library select extension imported by subprocess.
sys.path = [p for p in sys.path if Path(p or os.getcwd()).resolve() != HERE]

import argparse
import hashlib
import importlib.util
import itertools
import json
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from typing import Any


class BuildError(RuntimeError):
    pass


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_command(argv: list[str], cwd: Path = ROOT) -> str:
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise BuildError(f'Metadata command failed rc={result.returncode}: {argv[0]}')
    return result.stdout.strip()


def load_selection() -> Any:
    spec = importlib.util.spec_from_file_location('_afcl_selection', HERE / 'select.py')
    if spec is None or spec.loader is None:
        raise BuildError('Cannot load frozen card selection source')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configurations(ids: list[str], factorial: bool) -> tuple[list[dict], dict[str, dict]]:
    selection = load_selection()
    entries = selection.catalog()
    if len(set(ids)) != len(ids) or any(key not in entries for key in ids):
        raise BuildError('Select distinct, known AFCL IDs')
    for group in selection.EXCLUSIVE_CARDS:
        if group <= set(ids):
            raise BuildError('Mutually exclusive caller routes: ' + ', '.join(sorted(group)))
    if factorial and len(ids) > 6:
        raise BuildError('Factorial packages support at most six selected cards')
    chosen = [entries[key] for key in ids]
    choices = (itertools.product(('baseline', 'candidate'), repeat=len(chosen)) if factorial
               else [('baseline',) * len(chosen), ('candidate',) * len(chosen)])
    result = []
    for index, arms in enumerate(choices):
        config = selection.configuration(chosen, tuple(arms))
        config['name'] = f'configuration_{index:03d}' if factorial else ('baseline', 'candidate')[index]
        absent = {'MOJOLEARN_NUMERIC_IDENTICAL', 'MOJOLEARN_NUMERIC_DETERMINISTIC'}
        for entry in chosen:
            absent.update(entry.get('defines_absent_in_both_arms', []))
            # Prerequisites are deliberately equal; a candidate is additive.
            if not set(entry['baseline_defines']) <= set(entry['candidate_defines']):
                raise BuildError(entry['id'] + ': asymmetric prerequisites need a separate recipe')
        tokens = config['defines']
        if any(not re.fullmatch(r'MOJOLEARN_[A-Z0-9_]+', token) for token in tokens):
            raise BuildError('AFCL defines must be bare supported compiler define names')
        if absent.intersection(tokens):
            raise BuildError('Selected cards preempt an explicitly required caller route')
        config['exclude_defines'] = sorted(absent)
        config['runtime_environment'] = dict(config['environment'],
            MOJOLEARN_NUMERIC_MODE='fast', MOJOLEARN_VENDOR='apple')
        result.append(config)
    return result, {key: entries[key] for key in ids}


def module_name(binding: str) -> str:
    return '_mojolearn' if binding == 'core' else '_mojolearn_' + binding


def local_imports(path: Path, source: Path) -> set[Path]:
    """Conservative source dependency discovery; never a compiler/route proof.

    Follow imports even under disabled branches. The hand-authored mapping is
    always unioned with this discovery: a parser omission cannot erase a known
    consumer. Parent package initializers are included too.
    """
    found: set[Path] = set()
    pattern = r'^\s*(?:from\s+([.\w]+)\s+import|import\s+([.\w]+))'
    for match in re.finditer(pattern, path.read_text(), re.MULTILINE):
        name = match.group(1) or match.group(2)
        relative = len(name) - len(name.lstrip('.'))
        if relative:
            base = path.parent
            for _ in range(relative - 1):
                base = base.parent
            roots = [base]
        else:
            roots = [source, source / 'bindings']
        parts = name.lstrip('.').split('.') if name.lstrip('.') else []
        for base in roots:
            target = base.joinpath(*parts)
            candidates = [target.with_suffix('.mojo'), target / '__init__.mojo']
            candidates.extend(base.joinpath(*parts[:i], '__init__.mojo') for i in range(1, len(parts)))
            for candidate in candidates:
                if candidate.is_file() and candidate.is_relative_to(source):
                    found.add(candidate)
    return found


def source_consumers(source: Path, bindings: list[str], entries: dict[str, dict],
                     mapping: dict) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    edges: dict[Path, set[Path]] = {}
    closures: dict[str, set[str]] = {}
    for binding in bindings:
        start = source / 'bindings' / (module_name(binding) + '.mojo')
        visited: set[Path] = set()
        pending = [start]
        while pending:
            path = pending.pop()
            if path in visited:
                continue
            visited.add(path)
            if path not in edges:
                edges[path] = local_imports(path, source)
            pending.extend(edges[path] - visited)
        closures[binding] = {str(path.relative_to(source)) for path in visited}
    consumers = {}
    for idea, entry in entries.items():
        minimum = set(mapping['cards'][idea]['affected_bindings'])
        paths = set(entry['implementation_paths'])
        consumers[idea] = sorted(minimum | {name for name, closure in closures.items() if paths & closure})
        if not set(consumers[idea]) <= set(bindings):
            raise BuildError(idea + ': affected binding is absent from the packaged classical closure')
    return consumers, {key: sorted(value) for key, value in closures.items()}


def build_environment(defines: list[str], mode: str, environment: Path) -> dict[str, str]:
    # Flags are passed through exactly one established channel. Neither a
    # controller's previous candidate nor a second define channel can leak in.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(('MOJOLEARN_', 'MODULAR_', 'PIXI_'))
           and key not in {'PYTHONPATH', 'PYTHONHOME', 'MACOSX_DEPLOYMENT_TARGET',
                           'DYLD_LIBRARY_PATH', 'DYLD_FALLBACK_LIBRARY_PATH', 'EXTRA_DEFINES'}}
    env.update(MOJOLEARN_NUMERIC_MODE=mode, MOJOLEARN_VENDOR='apple',
               MOJOLEARN_TARGET_COLUMN='apple', MOJOLEARN_COMPILE_JOBS='1',
               MOJOLEARN_SKIP_BUILD_GATE='1', MOJOLEARN_BUILD_EXTRA_DEFINES='',
               MOJOLEARN_EXTRA_DEFINES='',
               MOJOLEARN_MOJO_BUILD_FLAGS=' '.join('-D ' + token for token in defines))
    env['PATH'] = str(environment / 'envs/default/bin') + os.pathsep + env.get('PATH', '')
    return env


def run_logged(argv: list[str], cwd: Path, env: dict[str, str], log: Path) -> int:
    with log.open('x') as stream:
        return subprocess.run(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                              stdout=stream, stderr=subprocess.STDOUT).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--idea', action='append', required=True, help='Repeat for a combined candidate')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--source-sha', required=True, help='Exact frozen HEAD commit')
    parser.add_argument('--factorial', action='store_true', help='Build every combination, up to six cards')
    parser.add_argument('--pixi-environment', type=Path, default=ROOT / '.pixi',
                        help='Existing provisioned .pixi directory; no environment installation here')
    args = parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise BuildError('Use the established Apple M2 build queue')
    chip = read_command(['sysctl', '-n', 'machdep.cpu.brand_string'])
    if not chip.startswith('Apple M2'):
        raise BuildError('Compile this frozen Apple round on the cheap M2 queue')
    if os.environ.get('MOJOLEARN_PERFORMANCE_QUEUE_JOB') != '1':
        raise BuildError('Run as an owned queued build job with MOJOLEARN_PERFORMANCE_QUEUE_JOB=1')
    source_sha = read_command(['git', 'rev-parse', 'HEAD'])
    if source_sha != args.source_sha:
        raise BuildError('Frozen source SHA does not equal this checkout HEAD')
    if read_command(['git', 'status', '--porcelain']):
        raise BuildError('Commit all source before a frozen build')
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise BuildError('Save build evidence outside the source worktree')
    if output.exists() and any(output.iterdir()):
        raise BuildError('Use a new empty output directory; active artifacts are never overwritten')
    slot = Path.home() / 'mojolearn-evidence/compile_slot.sh'
    if not slot.is_file():
        raise BuildError('Existing machine compile-slot semaphore is missing: ' + str(slot))
    environment = args.pixi_environment.resolve()
    compiler = environment / 'envs/default/bin/mojo'
    if not compiler.is_file():
        raise BuildError('Provision the supported frozen pixi environment on the build worker first')
    configs, entries = configurations(args.idea, args.factorial)
    mapping = json.loads((HERE / 'build_bindings.json').read_text())
    bindings = mapping['classical_package_bindings']
    output.mkdir(parents=True, exist_ok=True)
    (output / 'logs').mkdir()
    (output / 'cache').mkdir()
    package_file = output / 'paired-build.json'
    toolchain = {'compiler_path': str(compiler.resolve()), 'compiler_sha256': digest(compiler),
                 'pixi_environment': str(environment), 'pixi_lock_sha256': digest(ROOT / 'pixi.lock'),
                 'os': platform.platform(), 'chip': chip, 'compile_jobs': 1,
                 'compile_slot': str(slot), 'compile_slot_sha256': digest(slot),
                 'target_cpu': 'apple-m1', 'target_accelerator': 'metal:1',
                 'target_policy': 'existing supported bindings/build*.sh Apple recipe'}
    # Preserve package/compiler inventory as provenance without running Mojo or
    # importing MAX. Hash the actual runtime dylibs before either arm is built;
    # the shared environment must not change between A and B staging.
    toolchain['environment_package_sha256'] = {
        str(path.relative_to(environment)): digest(path)
        for path in sorted((environment / 'envs/default/conda-meta').glob('*.json'))}
    toolchain['runtime_input_sha256'] = {
        path.name: digest(path)
        for path in sorted((environment / 'envs/default/lib').glob('*.dylib')) if path.is_file()}
    receipt: dict[str, Any] = {
        'schema': 1, 'ideas': args.idea, 'source_sha': source_sha, 'vendor': 'apple',
        'numeric_mode': 'fast', 'status': 'building', 'started_at': stamp(),
        'toolchain': toolchain, 'arms': {}, 'builds': [],
        'definition_file_sha256': digest(HERE / 'build_bindings.json'),
        'card_manifest_sha256': {str(path.relative_to(ROOT)): digest(path)
                                 for path in sorted((HERE / 'lanes').glob('*.json'))},
        'device_smoke': 'not_run_build_only', 'quality': 'pending', 'measurement': 'pending',
    }
    input_sources = [HERE / name for name in ('build_pair.py', 'build_bindings.json',
                     'select.py', 'ideas.json', 'full_workloads.json', 'run_pair.py')]
    input_sources += [ROOT / 'tools/performance_ideas.py', ROOT / 'packaging/macos/stage_dylibs.py']
    input_sources += list((HERE / 'lanes').glob('*.json'))
    input_sources += [ROOT / 'experiments/performance_ideas' / idea / 'manifest.json' for idea in args.idea]
    receipt['input_source_sha256'] = {str(path.relative_to(ROOT)): digest(path)
                                      for path in sorted(set(input_sources)) if path.is_file()}
    write_json(package_file, receipt)
    scratch = output / 'frozen-source'
    cache: dict[tuple[str, str, tuple[str, ...]], tuple[Path, dict]] = {}
    try:
        rc = run_logged(['git', 'worktree', 'add', '--detach', str(scratch), source_sha], ROOT,
                        dict(os.environ), output / 'logs/source-worktree.log')
        if rc:
            raise BuildError(f'Frozen worktree creation failed rc={rc}; see logs/source-worktree.log')
        # The existing provisioned toolchain is reused; no compiler patching,
        # cross-target override, unsupported IR extraction or install occurs.
        (scratch / '.pixi').symlink_to(environment, target_is_directory=True)
        consumers, closures = source_consumers(scratch, bindings, entries, mapping)
        receipt['affected_bindings'] = consumers
        receipt['source_import_closures'] = closures
        write_json(package_file, receipt)

        def artifact(binding: str, defines: list[str], mode: str = 'fast') -> tuple[Path, dict]:
            key = (binding, mode, tuple(defines))
            if key in cache:
                return cache[key]
            if digest(compiler) != toolchain['compiler_sha256'] or digest(scratch / 'pixi.lock') != toolchain['pixi_lock_sha256']:
                raise BuildError('Frozen compiler or lockfile changed during the round')
            number = len(receipt['builds'])
            build_id = f'{number:03d}-{mode}-{binding}'
            script = 'bindings/build.sh' if binding == 'core' else f'bindings/build_{binding}.sh'
            native_name = module_name(binding) + '.so'
            relative = ('identical/' if mode == 'identical' else '') + native_name
            native = scratch / 'python/mojolearn' / relative
            # Only this fresh, private scratch worktree is mutable. Committed
            # source and final per-arm packages are never build destinations.
            if native.exists():
                native.unlink()
            log = output / 'logs' / (build_id + '.log')
            destination = output / 'cache' / (build_id + '.so')
            command = ['bash', str(slot), 'bash', script]
            env = build_environment(defines, mode, environment)
            item = {'build_id': build_id, 'binding': binding, 'numeric_mode': mode,
                    'defines': defines, 'source_sha': source_sha, 'vendor': 'apple',
                    'target_column': 'apple', 'log': str(log), 'argv': command,
                    'cwd': str(scratch), 'status': 'building', 'started_at': stamp(),
                    'build_script_sha256': digest(scratch / script),
                    'compiler_sha256': toolchain['compiler_sha256'],
                    'pixi_lock_sha256': toolchain['pixi_lock_sha256'],
                    'environment': {key: value for key, value in env.items() if key.startswith('MOJOLEARN_')}}
            receipt['builds'].append(item)
            write_json(package_file, receipt)
            try:
                rc = run_logged(command, scratch, env, log)
                item['exit_code'] = rc
                if rc or not native.is_file():
                    raise BuildError(f'{build_id} failed rc={rc}; full log: {log}')
                if digest(compiler) != toolchain['compiler_sha256'] or digest(scratch / 'pixi.lock') != toolchain['pixi_lock_sha256']:
                    raise BuildError('Compiler or lockfile changed during compilation: ' + build_id)
                if read_command(['git', 'status', '--porcelain', '--untracked-files=no'], scratch):
                    raise BuildError('Tracked frozen source changed during compilation: ' + build_id)
                shutil.copy2(native, destination)
                item.update(status='built_unverified_unmeasured', artifact=str(destination),
                            sha256=digest(destination), size_bytes=destination.stat().st_size)
                destination.chmod(0o444)
                cache[key] = destination, item
                return destination, item
            except BaseException as exc:
                item.update(status='build_failed', error=str(exc))
                raise
            finally:
                item['finished_at'] = stamp()
                write_json(output / 'logs' / (build_id + '.receipt.json'), item)
                write_json(package_file, receipt)

        for config in configs:
            name = config['name']
            package_root = output / 'arms' / name / 'python'
            package = package_root / 'mojolearn'
            arm = {'status': 'building', 'cards': config['cards'], 'defines': config['defines'],
                   'exclude_defines': config['exclude_defines'],
                   'runtime_environment': config['runtime_environment'], 'package_root': str(package_root),
                   'package_relative': str(package_root.relative_to(output)),
                   'source_sha': source_sha, 'artifacts': {}, 'quality': 'pending', 'measurement': 'pending'}
            receipt['arms'][name] = arm
            write_json(package_file, receipt)
            shutil.copytree(scratch / 'python/mojolearn', package,
                            ignore=shutil.ignore_patterns('*.so', '*.dylib', '__pycache__', '*.pyc', '.dylibs'))
            for binding in bindings:
                defines = sorted({token for idea, choice in config['cards'].items()
                                  if binding in consumers[idea]
                                  for token in entries[idea][choice + '_defines']})
                binary, item = artifact(binding, defines)
                relative = module_name(binding) + '.so'
                shutil.copy2(binary, package / relative)
                (package / relative).chmod(0o644)
                arm['artifacts'][relative] = dict(item, role='classical_fast_binding')
                write_json(package_file, receipt)
            for dependency in mapping['transport_dependencies']:
                binary, item = artifact(dependency['binding'], dependency['defines'], dependency['numeric_mode'])
                relative = dependency['numeric_mode'] + '/' + module_name(dependency['binding']) + '.so'
                target = package / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(binary, target)
                target.chmod(0o644)
                arm['artifacts'][relative] = dict(item, role=dependency['role'])
            # Use the established portable macOS package staging recipe. Its
            # static Mach-O dependency closure/signing does not import or run a
            # binding. Always hash the delivered bytes AFTER this step: signing
            # and load-command relocation can change the raw compiler output.
            stage_script = scratch / 'packaging/macos/stage_dylibs.py'
            stage_log = output / 'logs' / (name + '-stage.log')
            stage_command = [sys.executable, str(stage_script),
                             *[str(package / relative) for relative in arm['artifacts']],
                             str(environment / 'envs/default/lib')]
            stage = {'argv': stage_command, 'log': str(stage_log), 'status': 'staging',
                     'source_sha256': digest(stage_script), 'started_at': stamp()}
            arm['runtime_staging'] = stage
            write_json(package_file, receipt)
            for library, expected in toolchain['runtime_input_sha256'].items():
                if digest(environment / 'envs/default/lib' / library) != expected:
                    raise BuildError('Frozen runtime library changed before staging: ' + library)
            stage_rc = run_logged(stage_command, scratch,
                                  build_environment([], 'fast', environment), stage_log)
            stage.update(exit_code=stage_rc, finished_at=stamp(),
                         status='staged' if stage_rc == 0 else 'staging_failed')
            write_json(output / 'logs' / (name + '-stage.receipt.json'), stage)
            if stage_rc:
                raise BuildError(f'{name} runtime staging failed rc={stage_rc}; full log: {stage_log}')
            for library, expected in toolchain['runtime_input_sha256'].items():
                if digest(environment / 'envs/default/lib' / library) != expected:
                    raise BuildError('Frozen runtime library changed during staging: ' + library)
            for relative, item in arm['artifacts'].items():
                staged = package / relative
                item['compiled_sha256'] = item['sha256']
                item['sha256'] = digest(staged)
                item['artifact'] = str(staged)
                item['size_bytes'] = staged.stat().st_size
                staged.chmod(0o444)
            arm['runtime_libraries'] = {str(path.relative_to(package)): {'sha256': digest(path),
                                           'size_bytes': path.stat().st_size}
                                        for path in sorted((package / '.dylibs').glob('*.dylib'))}
            for path in (package / '.dylibs').glob('*.dylib'):
                path.chmod(0o444)
            arm['python_sources_sha256'] = {str(path.relative_to(package)): digest(path)
                                            for path in sorted(package.rglob('*.py'))}
            arm['status'] = 'build_complete_unverified_unmeasured'
            write_json(package_root.parent / 'arm.json', arm)
            write_json(package_file, receipt)
        receipt['status'] = 'build_complete_unverified_unmeasured'
    except BaseException as exc:
        receipt.update(status='build_failed', error=str(exc))
        raise
    finally:
        receipt['finished_at'] = stamp()
        # Preserve the frozen worktree, logs and completed artifacts on every
        # failure. Nothing is automatically rebuilt, deleted, resumed or run.
        write_json(package_file, receipt)
    print('AFCL_BUILD status=build_complete_unverified_unmeasured receipt=' + str(package_file))


if __name__ == '__main__':
    try:
        main()
    except BuildError as exc:
        print('AFCL_BUILD status=failed reason=' + str(exc), file=sys.stderr)
        raise SystemExit(2)
