#!/usr/bin/env python3
"""Build and prove only the Linux production merged CPU host.

Device compilation is deliberately absent. Runtime bytes are borrowed from an
original complete release build, never rebuilt or represented as new outputs.
The resulting proof is consumed by retained_release_composition.py.
"""
import argparse
import contextlib
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback

import release_host_composition as hc

SCHEMA = 'mojolearn.linux.host-build.v1'
CAPTURE_SCHEMA = 'mojolearn.linux.host-compile-inputs.v1'
KIND = 'independent-host-build'
PREFIX = 'mojolearn/cpu/'  # Proof namespace only; the wheel uses mojolearn/host/.
HOST_PATH = 'host/' + hc.HOST_FILE
HOST_INDEX = 'host/' + hc.INDEX
RUNPATH = '$ORIGIN/../.libs:$ORIGIN/../cuda/.libs:$ORIGIN/../hip/.libs:$ORIGIN/.libs'
FLAGS = dict(MOJOLEARN_TARGET_COLUMN='cpu', MOJOLEARN_NUMERIC_MODE='identical',
             MOJOLEARN_COMPILE_JOBS='2', MOJOLEARN_PACKAGE_BYTE_LM='1',
             MOJOLEARN_LINUX_CPU='x86-64-v3', MOJOLEARN_MERGED_BINDINGS='1')


def require(ok, message):
    hc.require(ok, 'Independent host: ' + message)


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + '\n')


def inventory_sha(rows):
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def command(pixi_env):
    return ['pixi', 'run', '-e', pixi_env, 'bash', 'bindings/merged_build.sh', 'host']


def record(source, compiler):
    """Validate original host-only witnesses, without requiring a GPU producer."""
    require(source.get('kind') == KIND and source.get('leg') == 'cpu', 'wrong producer scope')
    proof, capture, inventory, index, stamp, stage, readback = (hc.decoded(source[k]) for k in
        ('proof', 'capture', 'inventory', 'index', 'stamp', 'stage', 'readback'))
    require(proof.get('schema') == SCHEMA and proof.get('action') == 'build-host-only'
            and proof.get('complete') is True and 'extensions' not in proof,
            'incomplete host proof or invented device output')
    require(all(type(proof.get(k)) is int and proof[k] == 0 for k in
                ('build_exit', 'stamp_exit', 'stage_exit', 'readback_exit')), 'failed host production phase')
    revision = proof.get('source_commit')
    require(isinstance(revision, str) and re.fullmatch('[0-9a-f]{40}', revision)
            and capture.get('source_commit') == revision, 'source revisions differ')
    require(capture.get('schema') == CAPTURE_SCHEMA and capture.get('phase') == 'before-build'
            and proof.get('compile_inputs_sha256') == source['capture']['sha256'], 'missing before-build capture')
    require(capture.get('compiler_sha256') == proof.get('compiler_sha256') == compiler,
            'compiler bytes differ')
    require(proof.get('binding_layout') == 'merged' and proof.get('release_modes') == ['identical'],
            'unsupported host layout or numeric mode')
    require(isinstance(inventory, list) and inventory and len(dict(inventory)) == len(inventory)
            and proof.get('source_inventory') == inventory
            and proof.get('source_sha256') == inventory_sha(inventory)
            and capture.get('source_inventory_sha256') == source['inventory']['sha256'], 'source inventory differs')
    for path, value in inventory:
        hc.relative(path)
        require(re.fullmatch('[0-9a-f]{64}', value), 'invalid source digest')
    names = hc.expected_names()
    identities = capture.get('binding_identities', {})
    require(set(identities) == {'linux-64/host/-/host/' + n for n in names}, 'host closure coverage differs')
    import release_reuse as rr
    wanted_flags = dict(numeric_mode='identical', target_column='cpu', gpu_archs=None,
                        linux_cpu='x86-64-v3', compile_jobs='2', package_byte_lm='1')
    for ident in identities.values():
        require(ident.get('schema') == rr.SCHEMA and ident.get('target') == rr.LINUX
                and ident.get('vendor') == ident.get('tier') == 'host' and ident.get('arch') == ''
                and ident.get('flags') == wanted_flags and ident.get('closure', {}).get('scope') == 'closure'
                and not rr.unreadable(ident), 'unresolved or changed host inputs')
    environment = capture.get('build_environment', {})
    require(environment == FLAGS, 'actual host invocation flags differ')
    pixi_env = capture.get('pixi_environment', '')
    require(re.fullmatch('[A-Za-z0-9_-]+', pixi_env) and capture.get('command') == command(pixi_env)
            and proof.get('command') == command(pixi_env), 'host-only compiler command differs')
    packages, toolchain = capture.get('installed_packages'), capture.get('toolchain')
    require(isinstance(packages, list) and packages and isinstance(toolchain, dict)
            and toolchain.get('packages') and set(toolchain['packages']) <= set(packages)
            and all(i['toolchain'] == toolchain for i in identities.values()), 'compiler package provenance differs')
    require(type(index.get('format')) is int and index['format'] == 1
            and set(index.get('files', {})) == {hc.HOST_FILE}
            and len(index['files'][hc.HOST_FILE]) == len(names)
            and set(index['files'][hc.HOST_FILE]) == names, 'incomplete merged host index')
    require(proof.get('merged_indexes') == {PREFIX + HOST_INDEX: dict(
        sha256=source['index']['sha256'], files=index['files'])}, 'host index proof differs')
    require(stamp.get('schema') == 2 and stamp.get('binding') == HOST_PATH
            and stamp.get('script') == 'merged_build.sh' and stamp.get('commit') == revision
            and set(stamp.get('members', {})) == names, 'host stamp differs')
    by_name = {v['name']: v for v in identities.values()}
    for name, row in stamp['members'].items():
        require(row.get('scope') == 'closure' and row.get('digest') == by_name[name]['closure']['digest']
                and row.get('script') == by_name[name]['script'], 'stamp source closure differs')
    import binding_stamps
    merged_inputs = {name:dict(inventory).get(name) for name in binding_stamps.MERGED_INPUTS}
    require(all(merged_inputs.values()) and capture.get('merged_build_inputs') == merged_inputs,
            'merged build inputs differ from source inventory')
    merged_hash = hashlib.sha256()
    for name in binding_stamps.MERGED_INPUTS:
        merged_hash.update(name.encode() + b'\0' + merged_inputs[name].encode() + b'\n')
    require(stamp.get('merged_inputs') == merged_hash.hexdigest(), 'merged build stamp differs')
    require(proof.get('stamp_sha256') == source['stamp']['sha256'], 'stamp proof linkage differs')
    rows = readback.get('members', [])
    require(readback.get('schema') == 'mojolearn.host-readback.v1'
            and readback.get('source_commit') == revision
            and len(rows) == len(names) and {r.get('name') for r in rows} == names
            and all(r.get('vendor') == r.get('column') == 'cpu'
                    and type(r.get('numeric_mode')) is int and r['numeric_mode'] == 1 for r in rows),
            'incomplete or failed native host readback')
    require(proof.get('readback_sha256') == source['readback']['sha256'], 'host readback linkage differs')
    donor, _ = hc._record(source['runtime_source'], compiler)
    libraries = donor['runtime_libraries']
    require(toolchain == donor['toolchain'] and packages == donor['installed_packages'],
            'host and runtime compiler/package environment differs')
    require(stage.get('schema') == 'mojolearn.host-staging.v1' and stage.get('action') == 'stage-host-only'
            and stage.get('runtime_origin_proof_sha256') == source['runtime_source']['proof']['sha256']
            and stage.get('runtime_libraries') == libraries
            and stage.get('runtime_byte_changes') == []
            and proof.get('stage_sha256') == source['stage']['sha256'], 'runtime staging provenance differs')
    require(stage.get('raw_sha256') == proof.get('raw_host_sha256')
            and re.fullmatch('[0-9a-f]{64}', stage.get('raw_sha256', ''))
            and re.fullmatch('[0-9a-f]{64}', stage.get('staged_sha256', ''))
            and stage.get('runpath') == RUNPATH and not stage.get('rpath'), 'host RUNPATH staging differs')
    require(proof.get('host_extension') == {PREFIX + HOST_PATH: stage['staged_sha256']},
            'staged host output differs')
    require(capture.get('runtime_identity') == proof.get('runtime_identity'), 'runtime input record differs')
    contract = dict(source_commit=revision, source_inventory=inventory, compiler_sha256=compiler,
        host_identities=identities, installed_packages=packages, toolchain=toolchain,
        index=index, runtime_libraries=libraries,
        host_dependencies={k: stage.get(k) for k in ('needed', 'runpath', 'rpath')})
    return contract, dict(leg='cpu', host_sha256=stage['staged_sha256'],
                          unstaged_sha256=proof['raw_host_sha256'], proof_sha256=source['proof']['sha256']), proof, capture


def inspect(root):
    root = Path(root).resolve()
    source = dict(kind=KIND, leg='cpu', original_build_directory=str(root),
        **{key: hc.witness(root / path) for key, path in {
            'proof':'host-build-proof.json', 'capture':'compile-inputs.json', 'inventory':'source-inventory.json',
            'index':'set/' + HOST_INDEX, 'stamp':'binding-stamp.json', 'stage':'stage.json',
            'readback':'readback.json'}.items()})
    source['runtime_source'] = json.loads((root / 'runtime-source.json').read_text())
    require(hc.file_sha(root / 'raw' / hc.HOST_FILE) == hc.decoded(source['proof'])['raw_host_sha256'],
            'original compiler output bytes are missing or changed')
    return source, root / 'set'


def validate_bytes(source, root, compiler):
    contract, _, proof, _ = record(source, compiler)
    root = Path(root)
    require(hc.file_sha(root / HOST_PATH) == proof['host_extension'][PREFIX + HOST_PATH]
            and hc.file_sha(root / HOST_INDEX) == source['index']['sha256'], 'staged host bytes differ')
    require({p.name: hc.file_sha(p) for p in (root / '.libs').iterdir() if p.is_file()}
            == contract['runtime_libraries'], 'borrowed runtime bytes differ')
    require(not list(root.glob('identical/*')) and not list(root.glob('_adapters/*')),
            'host-only producer contains unexpected GPU/adapter outputs')
    return True


def readback_member(root, binary, name, out):
    """Native readback used only when the release owner runs the producer."""
    registry = 'mojolearn._native_loaded_modules'
    spec = importlib.util.spec_from_file_location(registry, root / 'python/mojolearn/_native_loaded_modules.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[registry] = module
    spec.loader.exec_module(module)
    loader = importlib.machinery.ExtensionFileLoader(name, str(binary))
    spec = importlib.util.spec_from_loader(name, loader, origin=str(binary))
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    prefix = name[len('_mojolearn_'):]
    row = dict(name=name, numeric_mode=getattr(module, prefix + '_numeric_mode')(),
               vendor=str(getattr(module, prefix + '_vendor')()), column=str(getattr(module, prefix + '_column')()))
    require(type(row['numeric_mode']) is int and row['numeric_mode'] == 1
            and row['vendor'] == row['column'] == 'cpu', 'native host readback failed: ' + name)
    write(out, row)


def run_logged(argv, root, env, log):
    with Path(log).open('w') as stream:
        result = subprocess.run(argv, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
    Path(str(log) + '.rc').write_text(str(result.returncode) + '\n')
    require(result.returncode == 0, 'command failed; see ' + str(log))
    return result.returncode


def build(root, runtime_build, out, compiler_sha, pixi_env='gbmbench', patchelf=None):
    """One actual host compile followed by standard staging and native readback."""
    require(sys.platform == 'linux', 'the host producer runs on Linux')
    import retained_release_composition as rc
    import binding_stamps
    import release_reuse as rr
    root, out = Path(root).resolve(), Path(out).resolve()
    require(root == Path(rr.ROOT).resolve(), 'producer tools must belong to the frozen source checkout')
    require(re.fullmatch('[A-Za-z0-9_-]+', pixi_env), 'unsafe pixi environment name')
    require(not out.exists(), 'output already exists; originals are never overwritten')
    donor, donor_set = hc.inspect_build(runtime_build)
    donor_contract, _ = hc._record(donor, compiler_sha)
    revision, inventory, context = rc.current_source(root)
    require(context['runtime_identity'] == rc.source_context(str(root), hc.decoded(donor['proof'])['source_commit'])['runtime_identity'],
            'retained runtime inputs changed')
    prefix = root / '.pixi/envs' / pixi_env
    compiler = prefix / 'bin/mojo'
    packages = sorted(p.stem for p in (prefix / 'conda-meta').glob('*.json'))
    require(hc.file_sha(compiler) == compiler_sha and packages == donor_contract['installed_packages'],
            'installed compiler/package bytes differ from retained runtime producer')
    raw_output = root / 'python/mojolearn' / HOST_PATH
    require(not raw_output.exists(), 'host output already exists; use a clean frozen checkout')
    # Unknown compiler flags cannot silently enter this production recipe.
    require(not any(os.environ.get(k) for k in ('CC','CXX','CFLAGS','CXXFLAGS','LDFLAGS')), 'custom C/C++ flags are unsupported')
    env = {k:v for k,v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'MOJO_', 'MODULAR_'))}
    env.update(FLAGS)
    resolved = subprocess.check_output(['pixi','run','-e',pixi_env,'python3','-c',
        'import shutil; print(shutil.which("mojo"))'],cwd=root,env=env,text=True).strip()
    require(Path(resolved).is_file() and hc.file_sha(resolved) == compiler_sha,
            'activated compiler differs from captured prefix')
    actual_facts = rr.tree_facts(root)
    actual_hosts = {b.key:rr.identity(b,actual_facts) for b in rr.bindings(rr.LINUX,['identical']) if b.host}
    require(actual_hosts == {k:v for k,v in context['binding_identities'].items() if '/host/' in k},
            'working-tree host inputs differ from committed source')
    out.mkdir(parents=True)
    write(out / 'source-inventory.json', inventory)
    write(out / 'runtime-source.json', donor)
    capture = dict(schema=CAPTURE_SCHEMA, phase='before-build', source_commit=revision,
        source_inventory_sha256=hc.file_sha(out / 'source-inventory.json'), compiler_sha256=compiler_sha,
        compiler_path=resolved, installed_packages=packages, toolchain=donor_contract['toolchain'],
        binding_identities={k:v for k,v in context['binding_identities'].items() if '/host/' in k},
        runtime_identity=context['runtime_identity'], build_environment=FLAGS, pixi_environment=pixi_env,
        merged_build_inputs=context['merged_build_inputs'],
        command=command(pixi_env), producer_sha256=hc.file_sha(Path(__file__)), created_ns=time.time_ns())
    write(out / 'compile-inputs.json', capture)  # Written before subprocess begins.
    build_exit = run_logged(command(pixi_env), root, env, out / 'build.log')
    require(rc.current_source(root)[:2] == (revision, inventory) and hc.file_sha(compiler) == compiler_sha,
            'source/compiler changed during compilation')
    (out / 'raw').mkdir()
    shutil.copy2(raw_output, out / 'raw' / hc.HOST_FILE)
    stamp_exit = run_logged(['python3','tools/binding_stamps.py','write-merged',str(raw_output),
                            ','.join(sorted(hc.expected_names()))], root, env, out / 'stamp.log')
    shutil.copy2(binding_stamps.stamp_path(raw_output), out / 'binding-stamp.json')
    staged = out / 'set'
    (staged / 'host').mkdir(parents=True)
    shutil.copy2(raw_output, staged / HOST_PATH)
    shutil.copy2(raw_output.parent / hc.INDEX, staged / HOST_INDEX)
    shutil.copytree(donor_set / '.libs', staged / '.libs')
    require({p.name: hc.file_sha(p) for p in (staged / '.libs').iterdir() if p.is_file()}
            == donor_contract['runtime_libraries'], 'retained runtime bytes changed')
    sys.path.insert(0, str(root / 'packaging/linux'))
    import stage_libs
    patchelf = patchelf or shutil.which('patchelf')
    require(patchelf and Path(patchelf).is_file(), 'patchelf is missing')
    with (out / 'stage.log').open('w') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        stage_libs.set_runpath(patchelf, staged / HOST_PATH, RUNPATH)
        needed, runpath, rpath = stage_libs.elf_dynamic(staged / HOST_PATH)
        require(not any(stage_libs.classify(dep, prefix / 'lib') == 'driver' for dep in needed), 'CPU host unexpectedly needs a GPU driver')
        stage_libs.verify_closed([staged / HOST_PATH, *(staged / '.libs').glob('*.so*')], staged / '.libs', prefix / 'lib')
    write(out / 'stage.json', dict(schema='mojolearn.host-staging.v1', action='stage-host-only',
        raw_sha256=hc.file_sha(out / 'raw' / hc.HOST_FILE), staged_sha256=hc.file_sha(staged / HOST_PATH),
        needed=needed, runpath=runpath, rpath=rpath, runtime_libraries=donor_contract['runtime_libraries'],
        runtime_origin_proof_sha256=donor['proof']['sha256'], runtime_byte_changes=[],
        patchelf_path=str(patchelf), patchelf_sha256=hc.file_sha(patchelf)))
    (out / 'stage.log.rc').write_text('0\n')
    readbacks = out / 'readback-members'; readbacks.mkdir()
    rows = []
    for name in sorted(hc.expected_names()):
        result = readbacks / (name + '.json')
        argv = ['pixi','run','-e',pixi_env,'python3','tools/release_host_build.py','readback-member',
                '--root',str(root),'--binary',str(staged / HOST_PATH),'--name',name,'--output',str(result)]
        run_logged(argv, root, env, readbacks / (name + '.log'))
        rows.append(json.loads(result.read_text()))
    write(out / 'readback.json',dict(schema='mojolearn.host-readback.v1',source_commit=revision,members=rows))
    require(rc.current_source(root)[:2] == (revision, inventory) and hc.file_sha(compiler) == compiler_sha,
            'source/compiler changed during staging/readback')
    index = json.loads((staged / HOST_INDEX).read_text())
    proof = dict(schema=SCHEMA, action='build-host-only', complete=True, build_exit=build_exit,
        stamp_exit=stamp_exit, stage_exit=0, readback_exit=0, source_commit=revision,
        binding_layout='merged', release_modes=['identical'], compiler_sha256=compiler_sha,
        compile_inputs_sha256=hc.file_sha(out / 'compile-inputs.json'), source_inventory=inventory,
        source_sha256=inventory_sha(inventory), command=command(pixi_env),
        raw_host_sha256=hc.file_sha(out / 'raw' / hc.HOST_FILE),
        host_extension={PREFIX + HOST_PATH:hc.file_sha(staged / HOST_PATH)},
        merged_indexes={PREFIX + HOST_INDEX:dict(sha256=hc.file_sha(staged / HOST_INDEX),files=index['files'])},
        stamp_sha256=hc.file_sha(out / 'binding-stamp.json'), stage_sha256=hc.file_sha(out / 'stage.json'),
        readback_sha256=hc.file_sha(out / 'readback.json'), runtime_identity=context['runtime_identity'],
        qualification='One real CPU host compile and native metadata readback; no GPU, adapter, public runtime, or installed-wheel qualification')
    write(out / 'host-build-proof.json', proof)
    source, directory = inspect(out)
    validate_bytes(source, directory, compiler_sha)
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    production = sub.add_parser('build')
    production.add_argument('--root',type=Path,required=True)
    production.add_argument('--runtime-build',type=Path,required=True)
    production.add_argument('--output',type=Path,required=True)
    production.add_argument('--compiler-sha256',required=True)
    production.add_argument('--pixi-env',default='gbmbench')
    production.add_argument('--patchelf')
    probe = sub.add_parser('readback-member')
    for name in ('root','binary','output'):
        probe.add_argument('--'+name,type=Path,required=True)
    probe.add_argument('--name',required=True)
    args=parser.parse_args(argv)
    if args.action == 'readback-member':
        readback_member(args.root,args.binary,args.name,args.output)
    else:
        try:
            result=build(args.root,args.runtime_build,args.output,args.compiler_sha256,args.pixi_env,args.patchelf)
            print(json.dumps(dict(schema=SCHEMA,source_commit=result['source_commit'],host_outputs=len(result['host_extension']))))
        except BaseException:
            if args.output.is_dir():
                (args.output/'failure.txt').write_text(traceback.format_exc())
            raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
