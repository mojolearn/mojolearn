"""Revalidate retained individual CUDA groups and official PTX staging.

This module proves source/recipe/raw-to-staged bytes. It does not infer runtime,
readback, adapter, or whole-package admission from those compilation facts.
"""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import release_retained_compile_evidence as common
import release_host_composition as hc
import release_reuse
from release_linux_development_evidence import ROUTING, group_facts

LEGS = {'cuda/sm_89': ('sm_89', 'native'), 'cuda/sm_80': ('sm_80', 'ptx')}


def require(ok, why):
    hc.require(ok, 'CUDA group composition: ' + why)


def environment_policy(env, leg):
    require(leg in LEGS, 'unsupported leg')
    arch, fmt = LEGS[leg]
    require(not common.derived_build_flags(env, ROUTING)
            and env.get('MOJOLEARN_NUMERIC_MODE') == 'identical'
            and env.get('MOJOLEARN_TARGET_COLUMN') == 'nvidia'
            and env.get('MOJOLEARN_GPU_ARCHS') == arch
            and env.get('MOJOLEARN_CUDA_CODE_FORMAT', 'native') == fmt
            and env.get('MOJOLEARN_LINUX_CPU', 'x86-64-v3') == 'x86-64-v3'
            and env.get('MOJOLEARN_PACKAGE_BYTE_LM') == '1',
            'target, mode or compile flags differ')


def replay(raw, staged, leg, patchelf, evidence, postcompile=None):
    from release_linux_group_composition import DEVICE_RUNPATH
    require(leg in LEGS, 'unsupported leg')
    raw, staged = evidence.file(raw), evidence.file(staged)
    with tempfile.TemporaryDirectory(prefix='mojolearn-cuda-stage-replay-') as folder:
        output = Path(folder) / raw.name
        shutil.copyfile(raw, output)
        # Match build_sets.sh: stage ELF RUNPATH, pin PTX rounding, then
        # perform native fatbin placement for native sets only.
        subprocess.run([patchelf, '--set-rpath', DEVICE_RUNPATH, str(output)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packaging/linux'))
        import ptx_contract
        ptx_contract.patch_file(output)
        audit = ptx_contract.audit_bytes(output.read_bytes())
        require(audit['modules'] > 0 and audit['plain_count'] == 0, 'PTX canonical contract failed')
        if LEGS[leg][1] == 'native':
            import cubin_contract
            require(postcompile is not None, 'native CUDA postcompile evidence is required')
            directory, tools = postcompile
            raw_bytes = output.read_bytes()
            for start, end in cubin_contract.modules(raw_bytes):
                import hashlib
                key = hashlib.sha256(raw_bytes[start:end]).hexdigest()
                for name in ('receipt.json','m.ptx','m.cubin','m.fatbin','ptxas.stdout','ptxas.stderr','fatbinary.stdout','fatbinary.stderr'):
                    evidence.file(Path(directory) / key / name)
            objdump = shutil.which('objdump')
            require(objdump is not None, 'objdump is required to cross-check standard placement')
            placed, _ = cubin_contract.replay_retained(output, directory, tools, 'sm_89', objdump)
            output.write_bytes(placed)
        require(hc.file_sha(output) == hc.file_sha(staged), 'official staging bytes differ')
    return hc.file_sha(staged)


def device_groups(rows, root, revision, site, compiler, patchelf, evidence, leg, postcompile=None):
    import verify_linux_surface_qualification as surface
    require(leg in LEGS, 'unsupported leg')
    expected = set(surface.expected_bindings('identical', True))
    require(len(rows) == len(expected) and {r['binding'] for r in rows} == expected,
            'individual group coverage differs')
    tree = common.Tree.at(str(root), revision)
    extensions, origins = {}, {}
    for spec in rows:
        result_path = evidence.file(spec['result'])
        result = evidence.json(result_path)
        receipt_path = evidence.file(spec['receipt'])
        receipt = evidence.json(receipt_path)
        selected = [r for r in result['plan']['groups'] if r['key'] == receipt['key']]
        require(len(selected) == 1, 'ambiguous group execution')
        facts = group_facts(selected[0], result, result_path, receipt_path, tree)
        name = spec['binding']
        require(facts['mode'] == 'identical' and facts['members'] == [name]
                and facts['compiler_sha256'] == compiler,
                'group membership, mode or compiler differs')
        require(facts['toolchain_locked'] == release_reuse.toolchain(root, common.LINUX),
                'locked compiler/dependencies differ')
        environment_policy(facts['environment'], leg)
        for field in ('receipt', 'execution', 'log', 'binary'):
            evidence.file(facts[field]['path'])
        target = Path(site) / 'identical' / (name + '.so')
        key = 'mojolearn/' + leg + '/identical/' + target.name
        extensions[key] = replay(facts['binary']['path'], target, leg, patchelf, evidence, postcompile)
        origins[key] = dict(kind='development-group', key=receipt['key'],
                            raw_sha256=facts['binary']['sha256'], sha256=extensions[key],
                            compiler_workers=facts['compiler_workers'],
                            producer_receipt=str(receipt_path), producer_result=str(result_path))
    # Individual images have no merged index. Never manufacture one for packaging.
    require({p.name for p in (Path(site) / 'identical').iterdir() if p.is_file()}
            == {n + '.so' for n in expected}, 'unrecorded individual group/index')
    return extensions, origins, {}


def adapter_group(directory, root, site, compiler, patchelf, evidence, postcompile):
    """The retained standard CUDA adapter build has its own original schema."""
    import importlib.util
    import retained_release_composition as composition
    from release_linux_group_composition import adapter_commands, DEVICE_RUNPATH, replay as elf_replay
    directory = Path(directory)
    capture = evidence.json(directory / 'adapter-before-build.json')
    result = evidence.json(directory / 'adapter-build-result.json')
    build = evidence.json(directory / 'adapters-raw/adapters-build.json')
    expected = composition.adapter_inputs(root)
    expected.update({name: hc.file_sha(root / name) for name in
                    ('pixi.toml', 'pixi.lock', 'packaging/linux/stage_libs.py',
                     'packaging/portable_math/portable_math.c', 'packaging/portable_math/stage.py')})
    require(type(result.get('exit_code')) is int and result['exit_code'] == 0
            and capture['compiler_sha256'] == compiler and capture['captured_inputs'] == expected,
            'adapter original execution/compiler/source differs')
    require(build['vendor'] == 'cuda' and build['arch'] == 'sm_89'
            and build['numeric_mode'] == 'identical' and build['source_commit'] == capture['source_commit'],
            'adapter source/target differs')
    adapter_commands(capture, build, 'cuda', 'sm_89')
    wheels = list((directory / 'torch-wheel-cache').glob('*.whl'))
    require(len(wheels) == 1 and hc.file_sha(evidence.file(wheels[0])) == build['torch']['wheel_sha256'],
            'original Torch input bytes differ')
    stagedir = site / '_adapters'
    staged = evidence.json(stagedir / 'adapters.json')
    expected_manifest = dict(build, files=staged.get('files'))
    require(staged == expected_manifest, 'adapter stage changed original provenance')
    spec = importlib.util.spec_from_file_location('cuda_adapter_manifest', root / 'packaging/adapters/manifest.py')
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    require(build['sources'] == api.source_digests(root), 'adapter declared source differs')
    api.check(stagedir, 'cuda', 'sm_89')
    evidence.file(directory / 'adapter-build.log')
    for role, name in api.FILES.items():
        require(evidence.json(directory / 'adapter-build' / (role + '-command.json')) == build['commands'][role],
                'original adapter command differs')
        evidence.file(directory / 'adapter-build' / (role + '-build.log'))
        raw = evidence.file(directory / 'adapters-raw' / name)
        require(build['unstaged'][role] == dict(name=name, sha256=hc.file_sha(raw), bytes=raw.stat().st_size),
                'original adapter image differs')
        runpath = DEVICE_RUNPATH if role == 'native' else '$ORIGIN:$ORIGIN/../../../../torch/lib' if role == 'torch' else None
        if role == 'native':
            replay(raw, stagedir / name, 'cuda/sm_89', patchelf, evidence, postcompile)
        else:
            elf_replay(raw, stagedir / name, runpath, patchelf, evidence)
    files = {p.name: hc.file_sha(evidence.file(p)) for p in stagedir.iterdir() if p.is_file()}
    return {'mojolearn/cuda/sm_89/_adapters': dict(files=files, manifest_sha256=files['adapters.json'])}


def readbacks(directory, revision, site, hosts, extensions, leg, evidence):
    import json
    from release_ptx_assembly import architectures
    directory = Path(directory)
    report = evidence.json(directory / 'readback.json')
    producer = evidence.file(directory / 'producer.py')
    require(report.get('schema') == 'mojolearn.nvidia-ptx.package-readback.v1'
            and report.get('source_commit') == revision and report.get('failures') == []
            and report.get('producer_sha256') == hc.file_sha(producer), 'missing current readback')
    target_tier = 'cuda_native_sm_89' if leg == 'cuda/sm_89' else 'cuda_ptx_sm_80'
    rows = report['rows']
    require(report['expected'] == report['completed'] == len(rows) == 112
            and len({(r['tier'], r['name']) for r in rows}) == 112, 'incomplete or duplicate readbacks')
    selected = [r for r in rows if r['tier'] in ('host', target_tier)]
    expected = {('host', Path(p).stem): digest for p, digest in hosts.items()}
    expected.update({(target_tier, Path(p).stem): digest for p, digest in extensions.items()})
    require(len(selected) == len(expected) and {(r['tier'], r['name']) for r in selected} == set(expected),
            'target readback coverage differs')
    for row in selected:
        tier, name = key = row['tier'], row['name']
        require(row['binary_sha256'] == expected[key] and row['vendor'] == ('cpu' if tier == 'host' else 'cuda'),
                'readback binary/vendor differs')
        raw = evidence.json(directory / (tier + '-' + name + '.json'))
        execution = evidence.json(directory / (tier + '-' + name + '.execution.json'))
        log = evidence.file(directory / (tier + '-' + name + '.log'))
        require(type(execution.get('exit_code')) is int and execution['exit_code'] == 0
                and execution == row['execution'] and execution['name'] == name and execution['tier'] == tier
                and execution['binary_sha256'] == expected[key] and execution['log_sha256'] == hc.file_sha(log),
                'readback execution failed or changed')
        require(raw == {k: v for k, v in row.items() if k not in ('architectures', 'execution')},
                'original readback output differs')
        binary = site / ('host' if tier == 'host' else 'identical') / (name + '.so')
        actual = architectures(binary)
        allowed = {LEGS[leg][0], LEGS[leg][0].replace('sm_', 'compute_')}
        require(row['architectures'] == actual and (not actual if tier == 'host' else bool(actual) and set(actual) <= allowed),
                'actual architecture differs')
        if tier == 'host':
            require(type(row['numeric_mode']) is int and row['numeric_mode'] == 1 and row['column'] == 'cpu',
                    'host numeric mode/column differs')
        if name == '_mojolearn_byte_lm':
            require(row['numeric_mode'] == 1 and row['profile'] ==
                    'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1', 'byte LM policy differs')
    require(evidence.text(directory / 'readback.txt') == ''.join(r['tier']+' '+r['name']+' '+r['vendor']+'\n' for r in rows)
            and evidence.text(directory / 'arch_readback.txt') == ''.join(r['tier']+' '+r['name']+' '+(','.join(r['architectures']) if r['architectures'] else 'NONE-BY-DESIGN')+'\n' for r in rows),
            'readback packaging text differs')
    return True


def native_postcompile(inputs, revision, evidence):
    import tarfile
    import hashlib
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'packaging/linux'))
    import cubin_contract, ptx_contract
    require(inputs.get('postcompile_receipt') and inputs.get('module_retention_directory')
            and inputs.get('cuda_tools_archive'), 'native CUDA postcompile evidence is required')
    receipt = evidence.json(inputs['postcompile_receipt'])
    require(receipt.get('schema') == 'mojolearn.cuda.native-postcompile.v1'
            and type(receipt.get('exit_code')) is int and receipt['exit_code'] == 0
            and receipt.get('source_commit') == revision and receipt.get('arch') == 'sm_89',
            'native postcompile execution/source differs')
    require(receipt.get('cubin_tool_sha256') == hc.file_sha(evidence.file(cubin_contract.__file__))
            and receipt.get('ptx_tool_sha256') == hc.file_sha(evidence.file(ptx_contract.__file__)),
            'native postcompile tooling differs')
    archive = evidence.file(inputs['cuda_tools_archive'])
    require(hc.file_sha(archive) == receipt.get('cuda_tools_archive_sha256')
            == 'ded05fe3c8d075c6c1bf892005d3c50bde3eceaa049b879fcdff6158e068e3be',
            'CUDA tools are not the supported pinned 12.5.82 archive')
    tools = receipt['tools']
    require(set(tools) == {'ptxas','fatbinary'}, 'assembler tool coverage differs')
    with tarfile.open(archive, 'r:xz') as bundle:
        for name in ('ptxas','fatbinary'):
            members = [m for m in bundle if m.isfile() and m.name.endswith('/bin/'+name)]
            require(len(members) == 1, 'ambiguous CUDA tool archive')
            raw = bundle.extractfile(members[0]).read()
            require(tools[name]['sha256'] == hashlib.sha256(raw).hexdigest()
                    and tools[name]['bytes'] == len(raw), 'assembler bytes differ from pinned archive')
    evidence.file(inputs['postcompile_log'])
    return inputs['module_retention_directory'], tools


def project(inputs, source_root, patchelf):
    import hashlib
    import json
    from check_linux_release_qualification import tracked_native_inventory
    from release_linux_group_composition import Evidence, host_groups, runtime_group
    root, site = Path(source_root).resolve(), Path(inputs['site']).resolve()
    revision, compiler, leg = inputs['source_commit'], inputs['compiler_sha256'], inputs['leg']
    require(leg in LEGS and common.HEX40.fullmatch(revision) and common.HEX64.fullmatch(compiler),
            'invalid source/compiler/leg identity')
    require(subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'], text=True).strip() == revision,
            'wrong source checkout')
    evidence = Evidence(); prefix = 'mojolearn/' + leg + '/'
    hosts, origins = host_groups(inputs['hosts'], root, revision, site, compiler, patchelf, evidence, prefix)
    postcompile = native_postcompile(inputs, revision, evidence) if leg == 'cuda/sm_89' else None
    extensions, device_origins, indexes = device_groups(inputs['devices'], root, revision, site, compiler, patchelf, evidence, leg, postcompile)
    adapters = adapter_group(inputs['adapter_directory'], root, site, compiler, patchelf, evidence, postcompile) if leg == 'cuda/sm_89' else {}
    runtime = runtime_group(inputs, root, site, evidence)
    if leg == 'cuda/sm_80':
        from verify_linux_surface_qualification import load_gpu_plugins
        baseline = evidence.json(site / 'PTX_BASELINE.json')
        require(baseline.get('source_commit') == revision and baseline.get('source_dirty') is False,
                'PTX manifest source differs')
        load_gpu_plugins().validate_baseline_manifest(baseline,
            {name[len(prefix):]: digest for name, digest in extensions.items()}, indexes={})
    expected = {'host/' + Path(p).name for p in hosts} | {'identical/' + Path(p).name for p in extensions}
    expected |= {'.libs/' + n for n in runtime}
    if adapters: expected |= {'_adapters/' + n for n in next(iter(adapters.values()))['files']}
    actual = {p.relative_to(site).as_posix() for p in site.rglob('*') if p.is_file()
              and ('.so' in p.name or p.suffix == '.mojoc' or p.name in ('adapters.json','_mojolearn_merged.json'))}
    require(actual == expected, 'unrecorded or missing stage assets')
    complete = readbacks(inputs['readback_directory'], revision, site, hosts, extensions, leg, evidence) if inputs.get('readback_directory') else False
    inventory = tracked_native_inventory(root)
    return dict(source_commit=revision, source_inventory=inventory,
                source_sha256=hashlib.sha256(json.dumps(inventory,separators=(',',':')).encode()).hexdigest(),
                compiler_sha256=compiler, binding_layout='per-binding', release_modes=['identical'],
                host_extension=hosts, extensions=extensions, merged_indexes=indexes, adapters=adapters,
                runtime_libraries=runtime, artifact_origins={**origins,**device_origins}, evidence=evidence.files,
                readback_complete=complete)


def validate_bytes(proof, root):
    root = Path(root); leg = proof['producer_inputs']['leg']; require(leg in LEGS, 'unsupported leg')
    prefix = 'mojolearn/' + leg + '/'
    for field in ('extensions','host_extension'):
        for name,digest in proof[field].items():
            require(name.startswith(prefix) and hc.file_sha(root / hc.relative(name[len(prefix):])) == digest,
                    'packed native bytes differ')
    require(proof['merged_indexes'] == {}, 'individual images must not claim a merged index')
    require({p.name:hc.file_sha(p) for p in (root/'.libs').iterdir() if p.is_file()} == proof['runtime_libraries'],
            'packed runtime differs')
    actual = {p.name:hc.file_sha(p) for p in (root/'_adapters').iterdir() if p.is_file()} if (root/'_adapters').exists() else {}
    expected = {prefix+'_adapters':dict(files=actual,manifest_sha256=actual.get('adapters.json'))} if actual else {}
    require(proof['adapters'] == expected and bool(actual) == (leg == 'cuda/sm_89'), 'packed adapters differ')
    return True
