"""Release admission for retained AMD merged groups and development host groups.

This is composition, never a successful whole-set build. Original producer files
remain unchanged. Validation reopens their receipts, logs and raw binaries,
checks the current source closure, and replays the supported ELF staging step.
The evidence directory must remain retained beside the proof; the proof alone
does not replace its referenced bytes. Runtime and identity admission are separate.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

import bincache
import release_host_composition as hc
import release_retained_compile_evidence as common
from release_linux_development_evidence import ROUTING, group_facts
from staged_native_admission import math_source_receipt

SCHEMA = 'mojolearn.linux.group-composition.v1'
ACTION = 'assemble-compiled-groups'
LEG = 'hip/gfx942'
PREFIX = 'mojolearn/' + LEG + '/'
HOST_RUNPATH = '$ORIGIN/../.libs:$ORIGIN/../cuda/.libs:$ORIGIN/../hip/.libs:$ORIGIN/.libs'
DEVICE_RUNPATH = '$ORIGIN/../.libs:$ORIGIN/../../.libs:$ORIGIN/../../../.libs'
RUNTIME = {'libAsyncRTMojoBindings.so', 'libAsyncRTRuntimeGlobals.so',
           'libKGENCompilerRTShared.so', 'libMSupportGlobals.so', 'libMojolearnMath.so'}


def require(ok, why):
    hc.require(ok, 'Group composition: ' + why)


class Evidence:
    """Every producer byte consulted is part of the admission's evidence closure."""
    def __init__(self):
        self.files = {}

    def file(self, path):
        path = Path(path).resolve()
        require(path.is_file(), 'missing retained file: ' + str(path))
        row = dict(sha256=hc.file_sha(path), bytes=path.stat().st_size)
        previous = self.files.setdefault(str(path), row)
        require(previous == row, 'evidence changed during validation')
        return path

    def json(self, path):
        return json.loads(self.file(path).read_text())

    def text(self, path):
        return self.file(path).read_text()


def replay(raw, staged, runpath, patchelf, evidence):
    """Compare actual bytes, not a claimed staging digest or ELF approximation."""
    raw, staged = evidence.file(raw), evidence.file(staged)
    with tempfile.TemporaryDirectory(prefix='mojolearn-stage-replay-') as directory:
        output = Path(directory) / raw.name
        shutil.copyfile(raw, output)
        if runpath is not None:
            subprocess.run([patchelf, '--set-rpath', runpath, str(output)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        require(hc.file_sha(output) == hc.file_sha(staged),
                'raw-to-staged replay differs: ' + staged.name)
    return hc.file_sha(staged)


def host_environment_policy(env):
    # group_facts has already reproduced the exact CPU compiler recipe and
    # checked both recorded command lines. CUDA_CODE_FORMAT is inherited by
    # mixed CPU/PTX builders but is not read by that host recipe. Preserve its
    # original value in evidence; do not require a needless native rebuild.
    require(not common.derived_build_flags(env, ROUTING)
            and env.get('MOJOLEARN_NUMERIC_MODE') == 'identical'
            and env.get('MOJOLEARN_TARGET_COLUMN') == 'cpu'
            and env.get('MOJOLEARN_GPU_ARCHS') is None
            and env.get('MOJOLEARN_LINUX_CPU', 'x86-64-v3') == 'x86-64-v3',
            'host compile flags differ')


def host_groups(rows, root, revision, site, compiler, patchelf, evidence, prefix=PREFIX):
    import verify_linux_surface_qualification as surface
    import release_reuse as rr
    expected = set(surface.wheel_host_bindings())
    require(len(rows) == len(expected) and {r['binding'] for r in rows} == expected,
            'host group coverage differs')
    tree = common.Tree.at(str(root), revision)
    outputs, origins = {}, {}
    for spec in rows:
        result_path = evidence.file(spec['result'])
        result = evidence.json(result_path)
        receipt_path = evidence.file(spec['receipt'])
        receipt = evidence.json(receipt_path)
        selected = [r for r in result['plan']['groups'] if r['key'] == receipt['key']]
        require(len(selected) == 1, 'ambiguous development group')
        facts = group_facts(selected[0], result, result_path, receipt_path, tree)
        name = spec['binding']
        require(facts['mode'] == 'host' and facts['members'] == [name]
                and facts['compiler_sha256'] == compiler, 'host target/membership/compiler differs')
        env = facts['environment']
        require(facts['toolchain_locked'] == rr.toolchain(root, common.LINUX),
                'host locked compiler/dependencies differ')
        host_environment_policy(env)
        for field in ('receipt', 'execution', 'log', 'binary'):
            evidence.file(facts[field]['path'])
        raw = Path(facts['binary']['path'])
        target = site / 'host' / (name + '.so')
        key = prefix + 'host/' + target.name
        outputs[key] = replay(raw, target, HOST_RUNPATH, patchelf, evidence)
        origins[key] = dict(kind='development-group', key=receipt['key'],
                            raw_sha256=facts['binary']['sha256'], sha256=outputs[key],
                            compiler_workers=facts['compiler_workers'],
                            producer_receipt=str(receipt_path), producer_result=str(result_path))
    return outputs, origins


def rows_at(evidence, path):
    return [r.split('\t') for r in evidence.text(path).splitlines() if r.strip()]


def device_recipe(fields, recipe, stem):
    rel = 'python/mojolearn/identical/' + stem + '.so'
    require(fields['source'] == recipe['source']
            and fields['declared'] == dict(inputs=recipe['declared'], outputs=[rel], shell='bash')
            and fields['script'] == 'bindings/merged_build.sh'
            and fields['args'] == ['gpu', '--group', stem], 'original device recipe differs')
    require(fields['numeric_mode'] == 'identical' and fields['gpu_archs'] == 'gfx942'
            and fields['target_column'] == 'amd' and fields['device_arch'] == 'none'
            and 'variant' not in fields, 'device numeric mode/target/variant differs')
    expected_env = dict(MODULAR_HOME='/root/mojolearn/.pixi/envs/default/share/max',
        MOJOLEARN_BUILD_PIXI_ENV='default', MOJOLEARN_BUILD_TIERS='identical',
        MOJOLEARN_COMPILE_JOBS='8', MOJOLEARN_CUDA_CODE_FORMAT='native',
        MOJOLEARN_GPU_ARCHS='gfx942', MOJOLEARN_LINUX_CPU='x86-64-v3',
        MOJOLEARN_MERGED_KEY=recipe['merged'], MOJOLEARN_NUMERIC_MODE='identical',
        MOJOLEARN_PACKAGE_BYTE_LM='1', MOJOLEARN_TARGET_COLUMN='amd')
    require(fields['build_env'] == expected_env and fields['repo_path'] == '/root/mojolearn'
            and fields['os']['machine'] == 'x86_64', 'device compile flags/platform differ')


def device_groups(rows, root, site, compiler, patchelf, evidence):
    import merged_bindings
    import release_reuse as rr
    from release_ptx_assembly import source_recipe
    plan = {stem: names for stem, _, names in merged_bindings.plan('identical', 'linux', True)}
    require(len(rows) == len(plan) and {r['stem'] for r in rows} == set(plan),
            'device group coverage differs')
    facts = rr.tree_facts(root)
    identities = {b.key: rr.identity(b, facts) for b in rr.bindings(rr.LINUX, ['identical'])
                  if b.vendor == 'hip' and b.arch == 'gfx942'}
    outputs, origins = {}, {}
    for spec in rows:
        stem, report = spec['stem'], Path(spec['report'])
        capture = evidence.json(report / 'compile-inputs.json')
        revision = capture['source_commit']
        require(capture['vendor'] == 'hip' and capture['arch'] == 'gfx942'
                and capture['compiler_sha256'] == compiler, 'device compiler/target differs')
        archive = evidence.file(spec['archive'])
        with tarfile.open(archive) as retained:
            manifest = json.load(retained.extractfile('manifest.json'))
            rel = 'python/mojolearn/identical/' + stem + '.so'
            data = retained.extractfile('files/' + rel).read()
        fields = manifest['fields']
        key = bincache.key_of(fields)
        require(manifest['key'] == key == archive.name.removesuffix('.tar.gz')
                and manifest['source_commit'] == revision
                and fields == evidence.json(report / 'bincache/keys' / (key + '.json')),
                'original device key/source differs')
        recipe = source_recipe(str(root), revision)
        device_recipe(fields, recipe, stem)
        require(manifest['installed_toolchain'] == fields['toolchain']['packages'],
                'original installed toolchain differs')
        rawsha = hashlib.sha256(data).hexdigest()
        require(len(manifest['files']) == 1 and manifest['files'][0]['path'] == rel
                and manifest['files'][0]['sha256'] == rawsha and manifest['files'][0]['size'] == len(data),
                'original archive output inventory differs')
        token = 'identical:merged_build.sh:' + stem
        shard = [r for r in rows_at(evidence, report / 'shard.tsv') if r[0] == token]
        require(len(shard) == 1 and len(shard[0]) == 5 and shard[0][1] == 'ok'
                and shard[0][2].startswith('miss+built-') and shard[0][4] == key,
                'missing original successful compilation')
        provenance = [r for r in rows_at(evidence, report / 'bincache/provenance.tsv')
                      if len(r) == 6 and r[3] == key]
        require(len(provenance) == 1 and provenance[0][1] == 'bindings/merged_build.sh'
                and provenance[0][2] == shard[0][2] and provenance[0][5] == rel + '=' + rawsha,
                'original raw output provenance differs')
        hot = [r for r in rows_at(evidence, report / 'hot_sha256.tsv') if r[0] == key]
        require(len(hot) == 1 and hot[0][1] == hc.file_sha(archive), 'archive retention SHA differs')
        log = evidence.file(report / 'build_logs' / ('identical_merged_' + stem + '.log'))
        lines = log.read_bytes().splitlines(keepends=True)
        require(len(lines) >= 3 and lines[0].startswith(b'start ')
                and lines[-2].startswith(b'BINCACHE ') and lines[-1].startswith(b'end ')
                and lines[-1].rstrip().endswith(b' OK'), 'original compiler log wrapper differs')
        require(hashlib.sha256(b''.join(lines[1:-2])).hexdigest() == manifest['build_log_sha256'],
                'original compiler output log differs')
        require(sorted(spec['members']) == sorted(plan[stem]), 'device group membership differs')
        for member in plan[stem]:
            ident = 'linux-64/hip/gfx942/identical/' + member
            before = capture['binding_identities'][ident]
            require(before == identities[ident] and before['toolchain'] == fields['toolchain'],
                    'current device source/compiler/flags/dependency identity differs: ' + member)
        with tempfile.TemporaryDirectory(prefix='mojolearn-raw-group-') as directory:
            raw = Path(directory) / (stem + '.so')
            raw.write_bytes(data)
            target = site / 'identical' / raw.name
            # Archive already witnesses these bytes; temporary paths are not retained inputs.
            staging_evidence = Evidence()
            value = replay(raw, target, DEVICE_RUNPATH, patchelf, staging_evidence)
            evidence.file(target)
        name = PREFIX + 'identical/' + stem + '.so'
        outputs[name] = value
        origins[name] = dict(kind='compiled-merged-shard', source_commit=revision, key=key,
                             raw_sha256=rawsha, sha256=value, compiler_workers=8,
                             archive=str(archive), producer_capture=str(report / 'compile-inputs.json'))
    index_path = site / 'identical/_mojolearn_merged.json'
    index = evidence.json(index_path)
    require(index['files'] == {k + '.so': v for k, v in plan.items()}, 'merged index membership differs')
    indexes = {PREFIX + 'identical/_mojolearn_merged.json':
               dict(sha256=hc.file_sha(index_path), files=index['files'])}
    return outputs, origins, indexes


def adapter_group(directory, root, site, compiler, patchelf, evidence):
    import retained_release_composition as composition
    directory = Path(directory)
    capture = evidence.json(directory / 'before-build.json')
    rawdir = directory / 'adapters-raw'
    build = evidence.json(rawdir / 'adapters-build.json')
    adapter_inputs = composition.adapter_inputs(root)
    # The original producer additionally captured its environment/staging inputs.
    adapter_inputs.update({name: hc.file_sha(root / name) for name in
        ('pixi.toml', 'pixi.lock', 'packaging/linux/stage_libs.py',
         'packaging/portable_math/portable_math.c', 'packaging/portable_math/stage.py')})
    require(evidence.text(directory / 'controller.exit').strip() == '0'
            and capture['compiler_sha256'] == compiler
            and capture['vendor'] == 'hip' and capture['arch'] == 'gfx942'
            and capture['captured_inputs'] == adapter_inputs,
            'adapter execution/compiler/source closure differs')
    require(build['vendor'] == 'hip' and build['arch'] == 'gfx942'
            and build['numeric_mode'] == 'identical' and build['source_commit'] == capture['source_commit'],
            'adapter target/source differs')
    adapter_commands(capture, build)
    torch = evidence.json(directory / 'torch-input-readback.json')
    require(torch.get('verified') is True
            and torch['sha256'] == build['torch']['wheel_sha256'], 'pinned Torch input readback differs')
    stagedir = site / '_adapters'
    staged = evidence.json(stagedir / 'adapters.json')
    expected = dict(build)
    expected['files'] = staged.get('files')
    require(staged == expected, 'adapter staging changed build provenance')
    api_path = root / 'packaging/adapters/manifest.py'
    loader = importlib.util.spec_from_file_location('group_adapter_manifest', api_path)
    api = importlib.util.module_from_spec(loader); loader.loader.exec_module(api)
    require(build['sources'] == api.source_digests(root), 'adapter declared sources differ')
    api.check(stagedir, 'hip', 'gfx942')
    for role, name in api.FILES.items():
        require(evidence.json(directory / 'build' / (role + '-command.json')) == build['commands'][role],
                'adapter original command differs')
        evidence.file(directory / 'build' / (role + '-build.log') if role != 'torch'
                      else directory / 'build.log')
        raw = evidence.file(rawdir / name)
        require(build['unstaged'][role] == dict(name=name, sha256=hc.file_sha(raw), bytes=raw.stat().st_size),
                'adapter original raw bytes differ')
        runpath = DEVICE_RUNPATH if role == 'native' else '$ORIGIN:$ORIGIN/../../../../torch/lib' if role == 'torch' else None
        replay(raw, stagedir / name, runpath, patchelf, evidence)
    files = {p.name: hc.file_sha(evidence.file(p)) for p in stagedir.iterdir() if p.is_file()}
    return {PREFIX + '_adapters': dict(files=files, manifest_sha256=files['adapters.json'])}


def adapter_commands(capture, build, vendor='hip', arch='gfx942'):
    require((vendor, arch) in (('hip', 'gfx942'), ('cuda', 'sm_89')), 'unsupported adapter target')
    command = capture['command']
    require(len(command) == 6 and command[:2] == ['bash', 'packaging/adapters/build.sh']
            and command[-2:] == [vendor, arch], 'unknown adapter producer')
    out, logs = command[2:4]
    expected_native = ['pixi', 'run', 'mojo', 'build', '--emit', 'shared-lib', '-j',
        str(capture['compiler_workers']), '-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1',
        '--target-cpu', 'x86-64-v3', '--target-accelerator', arch, '-D',
        'MOJOLEARN_COLUMN_AMD' if vendor == 'hip' else 'MOJOLEARN_COLUMN_NVIDIA',
        '-I', '.', '-I', 'bindings', 'adapters/native/c_api.mojo', '-o', out + '/libmojolearn_adapter.so']
    require(build['commands']['native'] == expected_native, 'native adapter flags differ')
    require(build['commands']['max'] == ['pixi', 'run', 'mojo', 'precompile', 'adapters/max_ops',
                                        '-o', out + '/mojolearn_ops.mojoc'], 'MAX adapter command differs')
    cpp = build['commands']['torch']
    # Source prefix mapping changes debug paths only; preserve its original spelling.
    maps = [s for s in cpp if s.startswith('-ffile-prefix-map=')]
    require(len(maps) == 2 and maps[0].startswith('-ffile-prefix-map=/') and maps[0].endswith('/=')
            and maps[1] == '-ffile-prefix-map=' + logs + '/torch-build/=', 'Torch prefix map differs')
    expected_cpp = ['c++', '-std=c++20', '-O2', '-fPIC', '-I', 'adapters/native',
        '-I', logs + '/torch-build/torch/include', '-I', logs + '/torch-build/torch/include/torch/csrc/api/include',
        *maps, '-shared', '-D_GLIBCXX_USE_CXX11_ABI=' + str(build['torch']['cxx11_abi']),
        'adapters/torch/registration.cpp', '-o', out + '/mojolearn_torch.so',
        '-L', logs + '/torch-build/torch/lib', '-L', out, '-Wl,--as-needed',
        '-lmojolearn_adapter', '-ltorch', '-ltorch_cpu', '-lc10', '-Wl,-z,defs']
    require(cpp == expected_cpp, 'Torch adapter compiler/link flags differ')


def runtime_group(inputs, root, site, evidence):
    manifest = evidence.json(inputs['dependency_manifest'])
    execution = evidence.json(inputs['stage_execution'])
    require(evidence.text(inputs['stage_exit']).strip() == '0'
            and type(execution.get('exit_code')) is int and execution['exit_code'] == 0,
            'original staging failed')
    for name in ('stage_log', 'stage_controller_log'):
        evidence.file(inputs[name])
    math_source_receipt(manifest, root)
    command = manifest['portable_math']['build']['command']
    require(command == ['cc', '-O2', '-ffp-contract=off', '-fno-fast-math', '-shared', '-fPIC',
                        '-march=x86-64-v3', '-nostdlib', '-Wl,-soname,libMojolearnMath.so'],
            'portable math compiler flags differ')
    rows = manifest['staged_libs']
    require(len(rows) == len(RUNTIME) and {r['name'] for r in rows} == RUNTIME,
            'runtime library coverage differs')
    libs = {p.name: hc.file_sha(evidence.file(p)) for p in (site / '.libs').iterdir() if p.is_file()}
    require(libs == {r['name']: r['sha256'] for r in rows}, 'runtime library bytes differ')
    changes = manifest['portable_math']['runtime_changes']
    require({r['file'] for r in changes} == RUNTIME - {'libMojolearnMath.so'}
            and all(r['after_sha256'] == libs[r['file']] for r in changes), 'runtime patch provenance differs')
    return libs


def readbacks(directory, revision, site, hosts, extensions, indexes, evidence):
    """Actual standalone loads, bound to these bytes; never infer vendor from flags."""
    from release_ptx_assembly import architectures
    directory = Path(directory)
    report = evidence.json(directory / 'readback.json')
    producer = evidence.file(directory / 'producer.py')
    require(report.get('schema') == 'mojolearn.amd.package-readback.v1'
            and report.get('source_commit') == revision and report.get('failures') == []
            and report.get('producer_sha256') == hc.file_sha(producer),
            'missing successful current package readback')
    expected = {('host', Path(p).stem): digest for p, digest in hosts.items()}
    files = {('host', Path(p).stem): site / 'host' / Path(p).name for p in hosts}
    for index in indexes.values():
        for filename, names in index['files'].items():
            for name in names:
                expected['identical', name] = extensions[PREFIX + 'identical/' + filename]
                files['identical', name] = site / 'identical' / filename
    rows = report['rows']
    require(report['expected'] == report['completed'] == len(rows) == len(expected)
            and {(r['tier'], r['name']) for r in rows} == set(expected), 'incomplete or duplicate readbacks')
    vendors, arches, isa = [], [], {}
    for row in rows:
        tier, name = key = row['tier'], row['name']
        require(row['binary_sha256'] == expected[key] and row['vendor'] == ('cpu' if tier == 'host' else 'hip'),
                'readback names another binary/vendor')
        raw = evidence.json(directory / (tier + '-' + name + '.json'))
        execution = evidence.json(directory / (tier + '-' + name + '.execution.json'))
        log = evidence.file(directory / (tier + '-' + name + '.log'))
        require(type(execution.get('exit_code')) is int and execution['exit_code'] == 0
                and execution == row['execution'] and execution['name'] == name and execution['tier'] == tier
                and execution['binary_sha256'] == expected[key] and execution['log_sha256'] == hc.file_sha(log),
                'native readback execution failed or changed')
        require(raw == {k: v for k, v in row.items() if k not in ('architectures', 'execution')},
                'original readback output differs')
        if files[key] not in isa:
            isa[files[key]] = architectures(files[key])
        actual_arches = isa[files[key]]
        require(row['architectures'] == actual_arches == ([] if tier == 'host' else ['gfx942']),
                'native ISA readback differs')
        if tier == 'host':
            require(type(row['numeric_mode']) is int and row['numeric_mode'] == 1 and row['column'] == 'cpu',
                    'host numeric mode/column differs')
        if name == '_mojolearn_byte_lm':
            require(row['numeric_mode'] == 1
                    and row['profile'] == 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1',
                    'byte LM numeric mode/profile differs')
        vendors.append(tier + ' ' + name + ' ' + row['vendor'] + '\n')
        arches.append(tier + ' ' + name + ' ' + (','.join(actual_arches) if actual_arches else 'NONE-BY-DESIGN') + '\n')
    require(evidence.text(directory / 'readback.txt') == ''.join(vendors)
            and evidence.text(directory / 'arch_readback.txt') == ''.join(arches), 'packaging readback text differs')
    return True


def project(inputs, source_root, patchelf):
    if inputs.get('leg') in ('cuda/sm_89', 'cuda/sm_80'):
        from release_linux_cuda_groups import project as cuda_project
        return cuda_project(inputs, source_root, patchelf)
    from check_linux_release_qualification import tracked_native_inventory
    root, site = Path(source_root).resolve(), Path(inputs['site']).resolve()
    revision, compiler = inputs['source_commit'], inputs['compiler_sha256']
    require(common.HEX40.fullmatch(revision) and common.HEX64.fullmatch(compiler), 'invalid source/compiler identity')
    actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    require(actual == revision and inputs['leg'] == LEG, 'wrong source checkout or unsupported leg')
    evidence = Evidence()
    hosts, origins = host_groups(inputs['hosts'], root, revision, site, compiler, patchelf, evidence)
    extensions, device_origins, indexes = device_groups(inputs['devices'], root, site, compiler, patchelf, evidence)
    adapters = adapter_group(inputs['adapter_directory'], root, site, compiler, patchelf, evidence)
    runtime = runtime_group(inputs, root, site, evidence)
    expected = {'host/' + Path(p).name for p in hosts} | {'identical/' + Path(p).name for p in extensions}
    expected |= {'.libs/' + n for n in runtime} | {'_adapters/' + n for n in next(iter(adapters.values()))['files']}
    expected |= {'identical/_mojolearn_merged.json'}
    native = {p.relative_to(site).as_posix() for p in site.rglob('*') if p.is_file()
              and ('.so' in p.name or p.suffix == '.mojoc' or p.name in ('adapters.json', '_mojolearn_merged.json'))}
    require(native == expected, 'unrecorded or missing stage assets: ' + str(sorted(native ^ expected)))
    readback_complete = (readbacks(inputs['readback_directory'], revision, site, hosts, extensions, indexes, evidence)
                         if inputs.get('readback_directory') else False)
    inventory = tracked_native_inventory(root)
    return dict(source_commit=revision, source_inventory=inventory,
                source_sha256=hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest(),
                compiler_sha256=compiler, binding_layout='merged', release_modes=['identical'],
                host_extension=hosts, extensions=extensions, merged_indexes=indexes, adapters=adapters,
                runtime_libraries=runtime, artifact_origins={**origins, **device_origins},
                evidence=evidence.files, readback_complete=readback_complete)


def validate(proof, source_root=None):
    require(proof.get('schema') == SCHEMA and proof.get('action') == ACTION and proof.get('complete') is True,
            'incomplete or unknown composition')
    require('build_exit' not in proof, 'composition must not impersonate a build')
    require(proof.get('readback_complete') is True, 'actual native readbacks remain pending')
    require(source_root is not None, 'current source is required for group admission')
    evidence = proof.get('evidence')
    require(isinstance(evidence, dict) and evidence, 'missing retained producer bytes')
    for name, row in evidence.items():
        path = Path(name)
        require(path.is_absolute() and path.is_file() and path.stat().st_size == row.get('bytes')
                and hc.file_sha(path) == row.get('sha256'), 'retained producer bytes differ: ' + name)
    # A proof may name the historical tool, but may not supply an executable to run.
    patchelf = shutil.which('patchelf')
    require(patchelf is not None, 'supported patchelf is required for staging replay')
    result = project(proof['producer_inputs'], source_root, patchelf)
    require(all(proof.get(k) == v for k, v in result.items()), 'composition projection/evidence differs')
    return True


def validate_bytes(proof, root):
    leg = proof.get('producer_inputs', {}).get('leg', LEG)
    if leg in ('cuda/sm_89', 'cuda/sm_80'):
        from release_linux_cuda_groups import validate_bytes as cuda_validate_bytes
        return cuda_validate_bytes(proof, root)
    require(leg == LEG, 'unsupported leg')
    root = Path(root)
    for field in ('extensions', 'host_extension'):
        for name, digest in proof[field].items():
            require(name.startswith(PREFIX) and hc.file_sha(root / hc.relative(name[len(PREFIX):])) == digest,
                    'packed native bytes differ: ' + name)
    for name, row in proof['merged_indexes'].items():
        require(hc.file_sha(root / hc.relative(name[len(PREFIX):])) == row['sha256'], 'packed index differs')
    require({p.name: hc.file_sha(p) for p in (root / '.libs').iterdir() if p.is_file()} == proof['runtime_libraries'],
            'packed runtime differs')
    files = {p.name: hc.file_sha(p) for p in (root / '_adapters').iterdir() if p.is_file()}
    require(proof['adapters'] == {PREFIX + '_adapters': dict(files=files, manifest_sha256=files.get('adapters.json'))},
            'packed adapters differ')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', required=True, type=Path)
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--patchelf', default='patchelf')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    inputs = json.loads(args.inputs.read_text())
    result = project(inputs, args.source_root, args.patchelf)
    tool = Path(shutil.which(args.patchelf) or args.patchelf).resolve()
    proof = dict(schema=SCHEMA, action=ACTION, complete=result['readback_complete'], producer_inputs=inputs,
                 staging_tool=dict(path=str(tool), sha256=hc.file_sha(tool)), **result)
    args.output.write_text(json.dumps(proof, sort_keys=True, indent=2) + '\n')
    print(json.dumps(dict(status='admitted-group-composition' if proof['complete'] else 'compiled-groups-validated-readbacks-pending', host_groups=len(result['host_extension']),
                          device_groups=len(result['extensions']), evidence_files=len(result['evidence']),
                          runtime_or_identity_qualified=False, output=str(args.output))))


if __name__ == '__main__':
    main()
