#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Stage optional, prebuilt adapter payloads into existing distribution wheels.

HANDOFF DRAFT: unqualified, not connected to any release/packaging entry point.
The manifest contract, binary inspection, state/cache/publish integration and
inert tests still need independent review and execution. Do not use this draft
to admit or publish artifacts. See E/adapter-packaging-review74/LANE_HANDOFF.txt.

No framework import or native execution occurs here. Review candidates are
explicitly unqualified. Admission requires retained build, relocation, installed
execution and cross-vendor word evidence for the exact candidate member map.
"""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import math
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'mojolearn.adapter-payload.v1'
ATTESTATION = 'ADAPTER_PAYLOAD.json'
PHASES = frozenset(('torch', 'max', 'direct-tape', 'cpu-abi', 'native-runtime',
                    'optional-import', 'relocation', 'max-default-load', 'framework-guard'))
SYSTEM_LIBS = frozenset(('libc.so.6', 'libm.so.6', 'libdl.so.2', 'libpthread.so.0',
                        'librt.so.1', 'libgcc_s.so.1', 'libstdc++.so.6',
                        'ld-linux-x86-64.so.2', 'libcuda.so.1', 'libamdhip64.so.6'))
REQUIRED = {'native': 'libmojolearn_adapter.so', 'torch': 'mojolearn_torch.so',
            'max': 'mojolearn_ops.mojoc'}
IDENTITY_CASES = frozenset(f'op{code}_{rows}_{features}_{columns}'
                         for rows, features, columns in ((2, 3, 4), (17, 19, 7), (3, 65, 5))
                         for code in range(1, 19) if code not in (12, 14))


def native_file(name):
    return re.search(r'\.so(?:\.[0-9]+)*$|\.dylib$', name) is not None


def case_layout(name):
    match = re.fullmatch(r'op([0-9]+)_([0-9]+)_([0-9]+)_([0-9]+)', name)
    require(match is not None and name in IDENTITY_CASES, 'unknown identity case')
    code, rows, features, columns = map(int, match.groups())
    inputs = [[rows, features]]
    if code <= 5:
        inputs.append([features, columns] if code == 1 else [rows, features])
    output = [rows, columns] if code == 1 else ([features, rows] if code == 13 else
              ([1, 1] if code in (15, 16) else [rows, features]))
    return code, inputs, output


def read_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs)


class Invalid(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Invalid(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def hex_digest(value):
    return type(value) is str and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def relative(value):
    require(type(value) is str and value and '\\' not in value, 'invalid relative path')
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(part not in ('', '.', '..') for part in value.split('/')),
            'absolute/traversing/noncanonical payload path')
    return path


def retained(base, descriptor):
    require(type(descriptor) is dict and hex_digest(descriptor.get('sha256')), 'missing retained evidence hash')
    rel = relative(descriptor.get('path'))
    path = base.joinpath(*rel.parts)
    require(not any(parent.is_symlink() for parent in [path, *path.parents] if parent != base.parent),
            'symlinked evidence/payload path')
    require(path.resolve().is_relative_to(base.resolve()) and path.is_file(), 'missing retained evidence: ' + str(rel))
    data = path.read_bytes()
    require(digest(data) == descriptor['sha256'], 'retained bytes differ: ' + str(rel))
    return data


def document(base, descriptor, schema):
    try:
        value = read_json(retained(base, descriptor))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise Invalid('invalid retained JSON') from error
    require(type(value) is dict and value.get('schema') == schema, 'wrong evidence schema: ' + schema)
    return value


def git_bytes(repo, commit, path):
    require(re.fullmatch(r'[0-9a-f]{40}', commit or '') is not None, 'missing full source commit')
    result = subprocess.run(['git', '-C', str(repo), 'show', commit + ':' + str(relative(path))],
                            capture_output=True, timeout=30)
    require(result.returncode == 0, 'source file absent from retained commit: ' + path)
    return result.stdout


def source_inventory(repo, commit, role):
    """Use the repository import resolver; include exact C++/ABI and lock inputs."""
    if role == 'torch':
        paths = {'adapters/torch/registration.cpp', 'adapters/native/abi.h', 'pixi.lock'}
    else:
        spec = importlib.util.spec_from_file_location('_adapter_source_graph', ROOT / 'tools/kernel_closure_check.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        texts = module.read_tree(commit, repo)
        pending = ['adapters/native/c_api.mojo' if role == 'native' else 'adapters/max_ops/__init__.mojo']
        paths = set()
        while pending:
            path = pending.pop()
            if path in paths:
                continue
            require(path in texts, 'unresolved adapter source: ' + path)
            paths.add(path)
            pending.extend(module.imports_of(path, texts[path], texts))
        paths |= {'pixi.lock', 'adapters/native/abi.h'}
    return {path: digest(git_bytes(repo, commit, path)) for path in sorted(paths)}


def owner(unit):
    vendor, target = unit.get('vendor'), unit.get('target')
    if vendor == 'metal':
        require(target == 'apple-native' and unit.get('platform') == 'darwin', 'invalid Apple target')
        return 'mojolearn', 'mojolearn/adapters/metal/apple-native'
    require(unit.get('platform') == 'linux', 'invalid Linux adapter platform')
    if vendor == 'cuda':
        require(re.fullmatch(r'sm_[0-9]+a?', target or '') is not None and target != 'sm_80',
                'adapter payload requires a native NVIDIA target, not the PTX slot')
        return 'mojolearn-nvidia', 'mojolearn/cuda_native/' + target + '/adapters'
    require(vendor == 'hip' and target == 'gfx942', 'unsupported adapter vendor/target')
    return 'mojolearn-amd', 'mojolearn/hip_native/gfx942/adapters'


def validate_builds(unit, base, repo, release_source, inventory=source_inventory):
    inventories = {}
    toolchain = unit.get('toolchain', {})
    require(set(toolchain) == {'mojo_version', 'mojo_identity_sha256', 'max_version'} and
            all(type(v) is str and v for v in toolchain.values()) and hex_digest(toolchain['mojo_identity_sha256']),
            'missing exact Mojo/MAX toolchain identity')
    framework = unit.get('framework', {})
    require(set(framework) == {'torch_version', 'backend', 'cxx11_abi'} and
            type(framework['torch_version']) is str and framework['torch_version'] and
            type(framework['cxx11_abi']) is bool, 'missing exact Torch version/C++ ABI')
    require(framework['backend'] == {'metal': 'mps', 'cuda': 'cuda', 'hip': 'hip'}[unit['vendor']],
            'Torch backend differs from adapter target')
    require(set(unit.get('builds', {})) == set(REQUIRED), 'native/Torch/MAX build receipts are all required')
    for role in REQUIRED:
        build = document(base, unit['builds'][role], 'mojolearn.adapter-build.v1')
        require(build.get('role') == role and build.get('toolchain') == toolchain and
                build.get('framework') == framework and build.get('target') == unit['target'],
                'build toolchain/framework/target mismatch')
        old_source = build.get('source_commit')
        current = inventory(repo, release_source, role)
        inventories[role] = current
        require(build.get('source_sha256') == current and inventory(repo, old_source, role) == current,
                'adapter source closure or lock differs; reuse is not established')
        source_files = build.get('source_files', {})
        require(set(source_files) == set(current), 'missing actual compiled source snapshot')
        for name, expected in current.items():
            require(digest(retained(base, source_files[name])) == expected, 'compiled source snapshot differs: ' + name)
        compiler = retained(base, build.get('compiler'))
        require(compiler.decode().strip() == toolchain['mojo_version'] and
                digest(compiler) == toolchain['mojo_identity_sha256'], 'compiler identity witness differs')
        command = read_json(retained(base, build.get('command')))
        require(type(command) is list and command and all(type(x) is str for x in command), 'missing actual build argv')
        joined = ' '.join(command)
        require(not any(word in joined for word in ('TEST_HOST_STAGING', 'SABOTAGE', 'FAULT', 'NUMERIC_FAST')),
                'test/fault/fast build cannot ship')
        if role == 'native':
            defines = [part for part in command if part.startswith('MOJOLEARN_')]
            require(defines == ['MOJOLEARN_NUMERIC_IDENTICAL=1'], 'native build has unqualified numerical defines')
            require('--target-cpu' in command and command[command.index('--target-cpu') + 1] ==
                    ('apple-m1' if unit['platform'] == 'darwin' else 'x86-64-v3'), 'wrong portable CPU baseline')
            if unit['platform'] == 'linux':
                require('--target-accelerator' in command and
                        command[command.index('--target-accelerator') + 1] == unit['target'], 'wrong GPU build target')
            else:
                require('--target-accelerator' not in command, 'Apple must retain native GPU detection')
        if role == 'torch':
            expected = '-D_GLIBCXX_USE_CXX11_ABI=' + str(int(framework['cxx11_abi']))
            require(expected in command, 'Torch C++ ABI flag mismatch')
        require(build.get('exit_code') == 0 and build.get('output_sha256') ==
                unit['files'][REQUIRED[role]]['original_sha256'], 'build did not produce the declared original artifact')
        require(retained(base, build.get('log')) is not None, 'missing build log')
    return inventories


def validate_relocation(unit, base, payload):
    report = document(base, unit.get('relocation'), 'mojolearn.adapter-relocation.v1')
    hashes = {name: digest(data) for name, data in payload.items()}
    require(report.get('artifacts') == hashes and report.get('target') == unit['target'],
            'relocation receipt does not bind final artifacts')
    require(report.get('framework') == unit['framework'] and report.get('toolchain') == unit['toolchain'],
            'relocation framework/toolchain differs')
    require(report.get('test_host_staging') is False, 'forced staging is not a release artifact')
    binaries = {name for name in payload if native_file(name)}
    require(set(report.get('binaries', {})) == binaries, 'relocation must inspect every native dependency')
    _, prefix = owner(unit)
    for name in binaries:
        row = report['binaries'][name]
        raw_binary = payload[name]
        if unit['platform'] == 'linux':
            require(len(raw_binary) >= 64 and raw_binary[:6] == b'\x7fELF\x02\x01' and
                    int.from_bytes(raw_binary[18:20], 'little') == 62, 'not an ELF64 x86-64 library')
        else:
            require(len(raw_binary) >= 32 and raw_binary[:4] == b'\xcf\xfa\xed\xfe' and
                    int.from_bytes(raw_binary[4:8], 'little') == 0x100000C, 'not a native arm64 Mach-O library')
        require(row.get('sha256') == hashes[name] and row.get('format') ==
                ('Mach-O' if unit['platform'] == 'darwin' else 'ELF'), 'native format/hash mismatch')
        require(row.get('architecture') == ('arm64' if unit['platform'] == 'darwin' else 'x86_64'),
                'wrong native CPU architecture')
        symbols = retained(base, row.get('symbols')).decode()
        require(re.search(r'\b_?Py[A-Z_][A-Za-z0-9_]*\b', symbols) is None,
                'adapter exposes CPython symbols')
        dynamic = retained(base, row.get('dynamic')).decode()
        require('libtorch_python' not in dynamic and 'libpython' not in dynamic,
                'adapter links CPython or libtorch_python')
        require(type(row.get('rpaths')) is list and type(row.get('dependencies')) is list,
                'missing native dependency/rpath inspection')
        parent = posixpath.dirname(prefix + '/' + name)
        origin = '@loader_path' if unit['platform'] == 'darwin' else '$ORIGIN'
        allowed = {origin, origin + '/.libs', origin + '/..',
                   origin + '/' + posixpath.relpath('torch/lib', parent)}
        require(all(path in allowed for path in row['rpaths']), 'absolute/unknown runtime search path')
        if unit['platform'] == 'linux':
            needed = re.findall(r'Shared library: \[([^\]]+)\]', dynamic)
            rpaths = [p for group in re.findall(r'(?:Library runpath|Library rpath): \[([^\]]*)\]', dynamic)
                      for p in group.split(':')]
            require(needed == [item['name'] for item in row['dependencies']] and rpaths == row['rpaths'],
                    'ELF inspection facts differ from retained tool output')
        else:
            load_commands = retained(base, row.get('load_commands')).decode()
            needed, rpaths, identifiers = [], [], []
            for block in re.split(r'Load command [0-9]+', load_commands):
                command = re.search(r'^\s*cmd (LC_[A-Z_]+)\s*$', block, re.M)
                if not command:
                    continue
                kind = command[1]
                if kind == 'LC_RPATH':
                    path = re.search(r'^\s*path (.+) \(offset [0-9]+\)', block, re.M)
                    require(path is not None, 'unreadable Mach-O runtime path')
                    rpaths.append(path[1])
                elif kind in ('LC_ID_DYLIB', 'LC_LOAD_DYLIB', 'LC_LOAD_WEAK_DYLIB', 'LC_REEXPORT_DYLIB', 'LC_LOAD_UPWARD_DYLIB'):
                    dep = re.search(r'^\s*name (.+) \(offset [0-9]+\)', block, re.M)
                    require(dep is not None, 'unreadable Mach-O dependency')
                    (identifiers if kind == 'LC_ID_DYLIB' else needed).append(dep[1])
            require(needed == [item['name'] for item in row['dependencies']] and rpaths == row['rpaths'] and
                    identifiers == ['@rpath/' + PurePosixPath(name).name],
                    'Mach-O inspection facts or install name are not relocatable')
        for dependency in row['dependencies']:
            require(type(dependency) is dict and set(dependency) >= {'name', 'origin'}, 'unresolved dependency')
            dep = dependency['name']
            require('libpython' not in dep.lower() and 'libtorch_python' not in dep.lower(), 'Python dependency')
            if dependency['origin'] == 'payload':
                require(not dep.startswith('/') and (unit['platform'] == 'darwin' or '/' not in dep),
                        'build-path dependency remains')
                member = dependency.get('member')
                require(member in binaries and dependency.get('sha256') == hashes[member], 'missing payload runtime dependency')
                require(PurePosixPath(dep).name == PurePosixPath(member).name, 'runtime dependency basename mismatch')
            elif dependency['origin'] == 'torch':
                require(not dep.startswith('/') and (unit['platform'] == 'darwin' or '/' not in dep),
                        'absolute framework dependency remains')
                require(PurePosixPath(dep).name in ('libtorch.so', 'libtorch_cpu.so', 'libc10.so',
                        'libtorch.dylib', 'libtorch_cpu.dylib', 'libc10.dylib') and
                        hex_digest(dependency.get('sha256')), 'unqualified framework dependency')
            else:
                require(dependency['origin'] == 'system' and
                        (dep in SYSTEM_LIBS or (unit['platform'] == 'darwin' and
                         dep.startswith(('/usr/lib/', '/System/Library/Frameworks/')))), 'unresolved non-system dependency')
        if unit['platform'] == 'darwin':
            require(row.get('codesign_exit') == 0 and row.get('deployment_floor') and row.get('metal_floor'),
                    'missing final Mach-O signature/deployment inspection')
        require(row.get('evidence') and all(retained(base, item) is not None for item in row['evidence']),
                'missing raw dependency/symbol/signature inspection')
    exports = report['binaries'][REQUIRED['native']].get('exports', [])
    require({'mojolearn_adapter_abi', 'mojolearn_adapter_operation',
             'mojolearn_adapter_operation_host_buffers'} <= set(exports), 'missing native ABI1 exports')
    return report


def load_manifest(path, *, repo=ROOT, version, source, inventory=source_inventory):
    path = Path(path).resolve()
    doc = read_json(path.read_bytes())
    require(doc.get('schema') == SCHEMA and doc.get('version') == version and
            doc.get('release_source') == source and doc.get('numeric_mode') == 'identical',
            'adapter manifest does not target this identical release')
    require(type(doc.get('payloads')) is list and doc['payloads'], 'empty adapter manifest')
    result, seen = [], set()
    for unit in doc['payloads']:
        distribution, prefix = owner(unit)
        require(prefix not in seen, 'duplicate adapter owner')
        seen.add(prefix)
        files = unit.get('files', {})
        require(type(files) is dict and set(REQUIRED.values()) <= set(files), 'incomplete adapter files')
        payload = {}
        for name, row in files.items():
            relative(name)
            require(name != ATTESTATION and not name.endswith(('.py', '.pyc', '.pyd')) and
                    (name in REQUIRED.values() or (name.startswith('.libs/') and native_file(name)) or name.startswith('licenses/')),
                    'unowned adapter payload file')
            require(hex_digest(row.get('original_sha256')), 'missing original artifact hash')
            payload[name] = retained(path.parent, row)
        inventories = validate_builds(unit, path.parent, repo, source, inventory)
        relocation = validate_relocation(unit, path.parent, payload)
        result.append(dict(unit=unit, prefix=prefix, distribution=distribution,
                           payload=payload, relocation=relocation, base=path.parent, source_sha256=inventories))
    return doc, result


def wheel_members(path):
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        require(len(names) == len(set(names)), 'duplicate wheel member')
        require(all(not n.endswith('/') for n in names), 'wheel contains directory entries')
        for name in names:
            relative(name)
        return {name: wheel.read(name) for name in names}


def member_identity(members):
    return digest(canonical({name: digest(data) for name, data in sorted(members.items())
                             if not name.endswith('.dist-info/RECORD') and
                             PurePosixPath(name).name != ATTESTATION}))


def wheel_metadata(members):
    found = [n for n in members if n.endswith('.dist-info/METADATA')]
    require(len(found) == 1, 'wheel needs exactly one METADATA')
    from email.parser import BytesParser
    metadata = BytesParser().parsebytes(members[found[0]])
    dist = found[0].rsplit('/', 1)[0]
    taglines = members.get(dist + '/WHEEL', b'').decode().splitlines()
    tags = [s.partition(': ')[2] for s in taglines if s.startswith('Tag: ')]
    require(tags, 'wheel has no platform tag')
    return metadata['Name'].replace('_', '-'), metadata['Version'], dist, tags


def required_cases(phase, vendor):
    if phase == 'torch':
        tree = ast.parse((ROOT / 'tools/check_native_adapters.py').read_text())
        result = {node.args[0].value for node in ast.walk(tree)
                  if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and
                  node.func.id in ('checked', 'refuse') and node.args and
                  isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)}
        result.discard('producer_on_nondefault_stream' if vendor == 'metal' else 'mps_implicit_index_no_inputs')
        return result
    return {
        'max': {'compiled_matmul_relu_composition_repeated', 'output_corruption_comparator_control',
                'full_no_inputs', 'scale_scalar_option', 'transpose_view', 'embedding_mixed_dtypes',
                'take_mixed_dtypes', 'argmax_integer_output', 'linear_three_inputs', 'restore_composition'},
        'direct-tape': IDENTITY_CASES,
        'cpu-abi': {'cpu_add', 'repeated_calls', 'output_shape_refusal', 'unknown_code_refusal',
                    'host_pointer_option_refusal', 'restored_after_error'},
        'framework-guard': {'wrong-version-refused', 'wrong-backend-refused', 'wrong-cxx-abi-refused'},
        'optional-import': {'without-torch-and-max', 'no-framework-gpu-initialization'},
        'relocation': {'installed-paths', 'build-paths-unavailable', 'owners-and-runtime-closure'},
        'max-default-load': {'installed-default-library'},
        'native-runtime': {'public-torch-entry', 'public-max-entry', 'no-python-callbacks'},
    }[phase]


def qualify(record, candidate, version, source, tags):
    unit, base = record['unit'], record['base']
    report = document(base, unit.get('qualification'), 'mojolearn.adapter-installed.v1')
    hashes = {name: digest(data) for name, data in record['payload'].items()}
    require(report.get('candidate_members_sha256') == candidate and report.get('artifacts') == hashes,
            'installed qualification belongs to different candidate bytes')
    require(report.get('version') == version and report.get('release_source') == source and
            report.get('target') == unit['target'] and report.get('wheel_tags') == tags,
            'installed qualification release/target/platform differs')
    require(report.get('framework') == unit['framework'] and report.get('toolchain') == unit['toolchain'],
            'installed framework/toolchain/C++ ABI differs')
    require(report.get('original_build_paths_absent') is True and report.get('source_tree_imports') == [],
            'installed relocation was not isolated from build/source paths')
    phases = report.get('phases', {})
    require(set(phases) == PHASES, 'missing installed adapter qualification phases')
    for phase in PHASES:
        row = phases[phase]
        expected = row.get('expected_cases')
        require(type(expected) is list and expected and len(expected) == len(set(expected)) and
                set(expected) >= required_cases(phase, unit['vendor']) and
                row.get('passed_cases') == expected and row.get('failures') == [] and
                row.get('skipped') == [] and row.get('exit_code') == 0, 'incomplete phase: ' + phase)
        require(row.get('evidence') and all(retained(base, item) is not None for item in row['evidence']),
                'missing phase execution evidence: ' + phase)
    faults = report.get('faults', {})
    require(set(faults) == {'wrong-operation', 'stall', 'restored'}, 'missing reached fault controls')
    for fault, row in faults.items():
        raw = retained(base, row.get('log'))
        if fault == 'stall':
            require(row.get('exit_code') not in (None, 0) and row.get('timed_out') is True and
                    b'ADAPTER_FAULT_STALL_ENTERED' in raw, 'stall never reached or did not time out')
        elif fault == 'wrong-operation':
            require(row.get('exit_code') not in (None, 0) and row.get('stage') == 'execution' and
                    row.get('detector') == 'output-word-mismatch' and row.get('timed_out') is False,
                    'wrong-operation failed before output comparison')
        else:
            require(row.get('exit_code') == 0 and row.get('artifacts') == hashes and
                    row.get('timed_out') is False, 'pristine restored artifacts did not pass')
    identity = document(base, report.get('identity'), 'mojolearn.adapter-identity.v1')
    columns = identity.get('columns', {})
    require(set(columns) == {'apple', 'nvidia', 'amd'}, 'missing identity vendor')
    expected = identity.get('expected_cases')
    require(type(expected) is list and set(expected) == IDENTITY_CASES and len(set(expected)) == len(expected),
            'missing/extra required identity matrix')
    provenance = identity.get('provenance', {})
    require(set(provenance) == set(columns), 'missing cross-vendor artifact provenance')
    reference = None
    for vendor in columns:
        proof = provenance[vendor]
        require(proof.get('numeric_mode') == 'identical' and proof.get('toolchain') == unit['toolchain'],
                'cross-vendor source/toolchain mismatch')
        require(proof.get('source_sha256') == record['source_sha256'], 'cross-vendor numerical source differs')
        require(set(proof.get('artifacts', {})) == set(REQUIRED), 'missing cross-vendor artifact bytes')
        for role, artifact in proof['artifacts'].items():
            retained(base, artifact)
        own_vendor = {'metal': 'apple', 'cuda': 'nvidia', 'hip': 'amd'}[unit['vendor']]
        if vendor == own_vendor:
            require(all(proof['artifacts'][role]['sha256'] == hashes[leaf] for role, leaf in REQUIRED.items()),
                    'identity evidence tested other adapter artifacts')
        require(set(columns[vendor]) == {'torch', 'max', 'direct-tape'}, 'missing identity entry path')
        for route, rows in columns[vendor].items():
            require(set(rows) == set(expected), 'missing/extra identity case')
            for name in expected:
                row = rows[name]
                raw = retained(base, row.get('output'))
                code, input_shapes, output_shape = case_layout(name)
                require(row.get('operation') == code and row.get('options') == ([-1] if code in (15, 16) else []) and
                        row.get('scalars') == [] and row.get('dtype') == 'float32' and
                        row.get('shape') == output_shape and len(raw) == 4 * math.prod(output_shape) and
                        type(row.get('inputs')) is list and len(row['inputs']) == len(input_shapes),
                        'missing identity storage metadata')
                for item, shape in zip(row['inputs'], input_shapes):
                    require(item.get('shape') == shape and item.get('dtype') == 'float32' and
                            len(retained(base, item)) == 4 * math.prod(shape), 'identity input shape/bytes differ')
            normalized = {name: {key: value for key, value in rows[name].items() if key not in ('output', 'inputs')} |
                          {'output_sha256': rows[name]['output']['sha256'],
                           'inputs_sha256': [r['sha256'] for r in rows[name].get('inputs', [])]} for name in expected}
            if reference is None:
                reference = normalized
            require(reference == normalized, 'cross-vendor/entry-path identity mismatch')
    return digest(canonical(report))


def write_wheel(path, members, dist):
    record = dist + '/RECORD'
    rows = []
    for name, data in sorted(members.items()):
        if name != record:
            rows.append((name, 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode(), str(len(data))))
    rows.append((record, '', ''))
    output = io.StringIO(newline='')
    csv.writer(output, lineterminator='\n').writerows(rows)
    members[record] = output.getvalue().encode()
    temp = path.with_suffix('.adapter-tmp')
    try:
        with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as wheel:
            for name, data in sorted(members.items()):
                info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                wheel.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        require(temp.stat().st_size <= 100 * 1024**2, 'adapter wheel exceeds PyPI100MiB limit')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def attach(wheel, manifest, *, version, source, repo=ROOT, candidate=False, output=None,
           inventory=source_inventory):
    wheel = Path(wheel)
    _, records = load_manifest(manifest, repo=repo, version=version, source=source, inventory=inventory)
    members = wheel_members(wheel)
    distribution, actual_version, dist, tags = wheel_metadata(members)
    require(version == actual_version, 'base wheel version differs')
    selected = [record for record in records if record['distribution'] == distribution and
                ((record['unit']['platform'] == 'darwin') == all('macosx_' in tag for tag in tags))]
    if not selected:
        check_members(members)
        require(not any(PurePosixPath(n).name == ATTESTATION for n in members), 'unexpected existing adapter payload')
        return None
    require(not candidate or output is not None, 'review candidate requires a separate output path')
    if output is not None:
        require(Path(output).resolve() != wheel.resolve(), 'review output cannot overwrite original wheel')
    # Restaging an explicitly assembled review candidate is the admission
    # transition. Verify every existing byte before removing only its owned
    # files/attestation. Qualification below must still match the final map.
    existing_records = check_members(members, allow_review=True)
    for existing in existing_records:
        prefix = owner(existing)[1] + '/'
        for name in [n for n in members if n.startswith(prefix)]:
            del members[name]
    for record in selected:
        prefix = record['prefix'] + '/'
        existing = {n for n in members if n.startswith(prefix)}
        require(not existing, 'base wheel already contains adapter payload; use original pristine base')
        for name, data in record['payload'].items():
            members[prefix + name] = data
        require(tags == record['relocation'].get('wheel_tags'), 'relocated adapter platform tag differs from base wheel')
    identity = member_identity(members)
    proofs = []
    for record in selected:
        proof = None if candidate else qualify(record, identity, version, source, tags)
        unit = record['unit']
        attestation = dict(schema=SCHEMA, status='UNQUALIFIED_REVIEW_ONLY' if candidate else 'admitted',
                           version=version, release_source=source, vendor=unit['vendor'], target=unit['target'],
                           platform=unit['platform'],
                           candidate_members_sha256=identity, artifacts={n: digest(b) for n, b in record['payload'].items()},
                           framework=unit['framework'], toolchain=unit['toolchain'], qualification_sha256=proof,
                           manifest_sha256=digest(Path(manifest).read_bytes()))
        members[record['prefix'] + '/' + ATTESTATION] = canonical(attestation) + b'\n'
        proofs.append(attestation)
    destination = Path(output) if output else wheel
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_wheel(destination, members, dist)
    receipt = dict(schema='mojolearn.adapter-wheel.v1', wheel=str(destination),
                   wheel_sha256=digest(destination.read_bytes()), candidate_members_sha256=identity,
                   status='UNQUALIFIED_REVIEW_ONLY' if candidate else 'admitted', payloads=proofs)
    destination.with_suffix('.adapters.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def check_members(members, *, allow_review=False):
    identity = member_identity(members)
    found = []
    for name, raw in members.items():
        if PurePosixPath(name).name != ATTESTATION:
            continue
        record = read_json(raw)
        statuses = {'admitted', 'UNQUALIFIED_REVIEW_ONLY'} if allow_review else {'admitted'}
        require(record.get('schema') == SCHEMA and record.get('status') in statuses, 'unqualified adapter candidate cannot publish')
        require(record.get('candidate_members_sha256') == identity and
                (record['status'] == 'UNQUALIFIED_REVIEW_ONLY' or hex_digest(record.get('qualification_sha256'))),
                'candidate changed after installed adapter qualification')
        distribution, _, _, _ = wheel_metadata(members)
        expected_distribution, expected_prefix = owner(record)
        require(distribution == expected_distribution and name == expected_prefix + '/' + ATTESTATION,
                'adapter attestation has wrong file ownership')
        prefix = name.rsplit('/', 1)[0] + '/'
        actual = {n[len(prefix):]: digest(data) for n, data in members.items() if n.startswith(prefix) and n != name}
        require(actual == record.get('artifacts'), 'adapter files differ from admitted member map')
        found.append(record)
    owned = {owner(record)[1] + '/' for record in found}
    require(all(any(name.startswith(prefix) for prefix in owned) for name in members if '/adapters/' in name),
            'adapter files lack an owning qualification attestation')
    return found


def check_wheel(wheel, *, manifest=None, version=None, source=None, repo=ROOT, inventory=source_inventory):
    members = wheel_members(wheel)
    found = check_members(members)
    if manifest is None:
        require(not found or version is None, 'adapter wheel was not explicitly selected for this release')
        return found
    _, records = load_manifest(manifest, repo=repo, version=version, source=source, inventory=inventory)
    distribution, _, _, tags = wheel_metadata(members)
    selected = [record for record in records if record['distribution'] == distribution and
                ((record['unit']['platform'] == 'darwin') == all('macosx_' in tag for tag in tags))]
    require({record['prefix'] for record in selected} == {owner(record)[1] for record in found},
            'wheel does not contain the explicitly selected adapter payloads')
    for record in selected:
        proof = qualify(record, member_identity(members), version, source, tags)
        attestation = next(row for row in found if owner(row)[1] == record['prefix'])
        require(attestation.get('manifest_sha256') == digest(Path(manifest).read_bytes()) and
                attestation.get('qualification_sha256') == proof, 'cached adapter qualification changed')
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--version')
    parser.add_argument('--source')
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--review-candidate', type=Path, metavar='OUTPUT_WHEEL')
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.check:
            result = check_wheel(args.wheel)
        else:
            require(args.manifest and args.version and args.source, 'staging requires manifest/version/source')
            result = attach(args.wheel, args.manifest, version=args.version, source=args.source, repo=args.repo,
                            candidate=args.review_candidate is not None, output=args.review_candidate)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (Invalid, KeyError, TypeError, ValueError, OSError) as error:
        print('adapter_payload: ' + str(error))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
