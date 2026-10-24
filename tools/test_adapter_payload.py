# SPDX-License-Identifier: Apache-2.0
"""Inert artifact controls: no native library, framework or compiler executes."""
import copy
import json
import math
from pathlib import Path
import struct
import subprocess
import zipfile

import pytest

import adapter_payload as ap


class Bundle:
    def __init__(self, root):
        self.root = root
        self.old, self.new = '1' * 40, '2' * 40
        self.version = '0.8.56'
        self.sources = {'pixi.lock': ap.digest(b'locked'), 'adapters/source.mojo': ap.digest(b'source')}
        self.inventory = lambda repo, commit, role: dict(self.sources)
        self.counter = 0
        self.compiler = self.put('compiler.txt', b'Mojo 1.0.0 (ed45d567)\n')
        self.toolchain = {'mojo_version': 'Mojo 1.0.0 (ed45d567)',
                          'mojo_identity_sha256': self.compiler['sha256'], 'max_version': '26.5.0'}
        self.framework = {'torch_version': '2.13.0+rocm7.2', 'backend': 'hip', 'cxx11_abi': True}
        self.payload = {name: self.elf(name.encode()) for name in
                        ('libmojolearn_adapter.so', 'mojolearn_torch.so', '.libs/libRuntime.so')}
        self.payload['mojolearn_ops.mojoc'] = b'inert precompiled package fixture'
        self.unit = dict(vendor='hip', target='gfx942', platform='linux',
                         toolchain=self.toolchain, framework=self.framework, builds={}, files={})
        for name, data in self.payload.items():
            self.unit['files'][name] = self.put('payload/' + name, data) | {'original_sha256': ap.digest(data)}
        for role, name in ap.REQUIRED.items():
            command = (['mojo', 'build', '--target-cpu', 'x86-64-v3', '--target-accelerator', 'gfx942',
                        '-D', 'MOJOLEARN_NUMERIC_IDENTICAL=1'] if role == 'native' else
                       ['c++', '-D_GLIBCXX_USE_CXX11_ABI=1'] if role == 'torch' else ['mojo', 'precompile'])
            build = dict(schema='mojolearn.adapter-build.v1', role=role, source_commit=self.old,
                         source_sha256=self.sources, source_files={
                             'pixi.lock': self.put('sources/pixi.lock', b'locked'),
                             'adapters/source.mojo': self.put('sources/adapters/source.mojo', b'source')},
                         compiler=self.compiler, toolchain=self.toolchain, framework=self.framework, target='gfx942',
                         command=self.json('build/' + role + '-argv.json', command),
                         log=self.put('build/' + role + '.log', b'inert test build record\n'),
                         exit_code=0, output_sha256=ap.digest(self.payload[name]))
            self.unit['builds'][role] = self.json('build/' + role + '.json', build)
        hashes = {n: ap.digest(v) for n, v in self.payload.items()}
        self.tags = ['py3-none-manylinux_2_28_x86_64']
        relocation = dict(schema='mojolearn.adapter-relocation.v1', artifacts=hashes, target='gfx942',
                          framework=self.framework, toolchain=self.toolchain, wheel_tags=self.tags,
                          test_host_staging=False, binaries={})
        for name in self.payload:
            if not ap.native_file(name):
                continue
            deps = []
            if name == 'libmojolearn_adapter.so':
                deps = [dict(name='libRuntime.so', origin='payload', member='.libs/libRuntime.so',
                             sha256=hashes['.libs/libRuntime.so'])]
            if name == 'mojolearn_torch.so':
                deps = [dict(name='libmojolearn_adapter.so', origin='payload', member='libmojolearn_adapter.so',
                             sha256=hashes['libmojolearn_adapter.so']),
                        dict(name='libtorch.so', origin='torch', sha256='a' * 64)]
            rpaths = ['$ORIGIN/.libs', '$ORIGIN']
            dynamic = '\n'.join('Shared library: [' + d['name'] + ']' for d in deps)
            dynamic += '\nLibrary runpath: [' + ':'.join(rpaths) + ']\n'
            row = dict(sha256=hashes[name], format='ELF', architecture='x86_64', dependencies=deps,
                       rpaths=rpaths, symbols=self.put('inspect/' + name + '.symbols', b'no Python symbols\n'),
                       dynamic=self.put('inspect/' + name + '.dynamic', dynamic.encode()),
                       evidence=[self.put('inspect/' + name + '.log', b'inert inspection receipt\n')])
            if name == 'libmojolearn_adapter.so':
                row['exports'] = ['mojolearn_adapter_abi', 'mojolearn_adapter_operation',
                                  'mojolearn_adapter_operation_host_buffers']
            relocation['binaries'][name] = row
        self.relocation = relocation
        self.unit['relocation'] = self.json('relocation.json', relocation)
        self.manifest = self.root / 'manifest.json'
        self.base_wheel = self.root / 'mojolearn_amd-0.8.56-py3-none-manylinux_2_28_x86_64.whl'
        self.base_members = {
            'mojolearn_amd-0.8.56.dist-info/METADATA': b'Metadata-Version: 2.1\nName: mojolearn-amd\nVersion: 0.8.56\n',
            'mojolearn_amd-0.8.56.dist-info/WHEEL': ('Wheel-Version: 1.0\nTag: ' + self.tags[0] + '\n').encode(),
            'mojolearn_amd-0.8.56.dist-info/RECORD': b'',
            'mojolearn/hip_native/gfx942/identical/_mojolearn.so': b'inert existing payload',
        }
        self.zip(self.base_wheel, self.base_members)
        self.refresh()

    @staticmethod
    def elf(tail):
        header = bytearray(64)
        header[:6] = b'\x7fELF\x02\x01'
        header[18:20] = (62).to_bytes(2, 'little')
        return bytes(header) + tail

    def put(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {'path': name, 'sha256': ap.digest(data)}

    def json(self, name, doc):
        return self.put(name, ap.canonical(doc) + b'\n')

    def refresh(self):
        self.json('manifest.json', dict(schema=ap.SCHEMA, version=self.version, release_source=self.new,
                                       numeric_mode='identical', payloads=[self.unit]))

    def zip(self, path, data):
        with zipfile.ZipFile(path, 'w') as wheel:
            for name, content in data.items():
                wheel.writestr(name, content)

    def load(self):
        return ap.load_manifest(self.manifest, repo=self.root, version=self.version, source=self.new,
                                inventory=self.inventory)

    def attach(self, **kwargs):
        return ap.attach(self.base_wheel, self.manifest, repo=self.root, version=self.version,
                         source=self.new, inventory=self.inventory, **kwargs)

    def qualify(self):
        prefix = ap.owner(self.unit)[1]
        candidate = dict(self.base_members)
        candidate.update({prefix + '/' + name: data for name, data in self.payload.items()})
        identity = ap.member_identity(candidate)
        phases = {}
        for phase in ap.PHASES:
            cases = sorted(ap.required_cases(phase, 'hip'))
            phases[phase] = dict(expected_cases=cases, passed_cases=cases, failures=[], skipped=[], exit_code=0,
                                 evidence=[self.put('runs/' + phase + '.log', b'inert phase fixture\n')])
        matrix = {}
        for name in sorted(ap.IDENTITY_CASES):
            code, shapes, output_shape = ap.case_layout(name)
            inputs = [self.put(f'words/{name}-in{i}.bin', struct.pack('<f', .25) * math.prod(shape)) |
                      {'dtype': 'float32', 'shape': shape} for i, shape in enumerate(shapes)]
            matrix[name] = dict(operation=code, options=[-1] if code in (15, 16) else [], scalars=[],
                                inputs=inputs, output=self.put('words/' + name + '.bin',
                                                              struct.pack('<f', 1.) * math.prod(output_shape)),
                                dtype='float32', shape=output_shape)
        provenance = {vendor: dict(numeric_mode='identical', toolchain=self.toolchain,
                                  source_sha256={role: self.sources for role in ap.REQUIRED},
                                  artifacts={role: self.unit['files'][leaf] for role, leaf in ap.REQUIRED.items()})
                      for vendor in ('apple', 'nvidia', 'amd')}
        identity_doc = dict(schema='mojolearn.adapter-identity.v1', expected_cases=sorted(ap.IDENTITY_CASES),
                            provenance=provenance, columns={vendor: {route: copy.deepcopy(matrix) for route in
                             ('torch', 'max', 'direct-tape')} for vendor in ('apple', 'nvidia', 'amd')})
        hashes = {n: ap.digest(v) for n, v in self.payload.items()}
        report = dict(schema='mojolearn.adapter-installed.v1', candidate_members_sha256=identity,
                      artifacts=hashes, version=self.version, release_source=self.new, target='gfx942',
                      wheel_tags=self.tags, framework=self.framework, toolchain=self.toolchain,
                      original_build_paths_absent=True, source_tree_imports=[], phases=phases,
                      identity=self.json('identity.json', identity_doc), faults={
                          'wrong-operation': dict(exit_code=1, stage='execution', detector='output-word-mismatch',
                              timed_out=False, log=self.put('fault/wrong.log', b'output words differ\n')),
                          'stall': dict(exit_code=124, timed_out=True,
                              log=self.put('fault/stall.log', b'ADAPTER_FAULT_STALL_ENTERED\n')),
                          'restored': dict(exit_code=0, artifacts=hashes, timed_out=False,
                              log=self.put('fault/restored.log', b'PASS restored inert fixture\n'))})
        self.report, self.identity_doc = report, identity_doc
        self.unit['qualification'] = self.json('installed.json', report)
        self.refresh()

    def refresh_qualification(self):
        self.report['identity'] = self.json('identity.json', self.identity_doc)
        self.unit['qualification'] = self.json('installed.json', self.report)
        self.refresh()


@pytest.fixture
def bundle(tmp_path):
    return Bundle(tmp_path)


def test_review_candidate_is_separate_and_rejected_by_publish_check(bundle):
    original = bundle.base_wheel.read_bytes()
    output = bundle.root / 'review' / bundle.base_wheel.name
    receipt = bundle.attach(candidate=True, output=output)
    assert receipt['status'] == 'UNQUALIFIED_REVIEW_ONLY'
    assert bundle.base_wheel.read_bytes() == original
    with pytest.raises(ap.Invalid, match='unqualified'):
        ap.check_wheel(output)
    with pytest.raises(ap.Invalid, match='separate output'):
        bundle.attach(candidate=True)
    with pytest.raises((ap.Invalid, TypeError)):
        bundle.attach()


def test_admitted_wheel_preserves_payload_and_cannot_borrow_changed_candidate_proof(bundle):
    bundle.qualify()
    receipt = bundle.attach()
    assert receipt['status'] == 'admitted'
    assert ap.check_wheel(bundle.base_wheel, manifest=bundle.manifest, version=bundle.version,
                          source=bundle.new, repo=bundle.root, inventory=bundle.inventory)
    members = ap.wheel_members(bundle.base_wheel)
    for name, data in bundle.payload.items():
        assert members[ap.owner(bundle.unit)[1] + '/' + name] == data
    members['mojolearn/hip_native/gfx942/identical/_mojolearn.so'] += b'changed base'
    bundle.zip(bundle.base_wheel, members)
    with pytest.raises(ap.Invalid, match='candidate changed'):
        ap.check_wheel(bundle.base_wheel)


def test_honest_old_compile_sha_reuses_matching_source_but_not_changed_closure(bundle):
    bundle.qualify()
    assert bundle.old != bundle.new
    assert bundle.attach()['status'] == 'admitted'
    bundle.inventory = lambda repo, commit, role: bundle.sources | ({'added.mojo': 'b' * 64} if commit == bundle.new else {})
    with pytest.raises(ap.Invalid, match='source closure'):
        bundle.load()


@pytest.mark.parametrize('change', ['absolute-rpath', 'absolute-needed', 'missing-runtime', 'python-symbol',
                                    'wrong-architecture', 'missing-raw-inspection', 'forced-test-build'])
def test_relocation_refuses_bad_or_incomplete_native_artifacts(bundle, change):
    row = bundle.relocation['binaries']['libmojolearn_adapter.so']
    if change == 'absolute-rpath':
        row['rpaths'] = ['/root/build/.pixi/lib']
    elif change == 'absolute-needed':
        row['dependencies'][0]['name'] = '/root/build/libRuntime.so'
    elif change == 'missing-runtime':
        row['dependencies'][0]['member'] = '.libs/missing.so'
    elif change == 'python-symbol':
        row['symbols'] = bundle.put('inspect/python.txt', b' U PyObject_Call\n')
    elif change == 'wrong-architecture':
        row['architecture'] = 'arm64'
    elif change == 'missing-raw-inspection':
        row['evidence'] = []
    else:
        bundle.relocation['test_host_staging'] = True
    bundle.unit['relocation'] = bundle.json('relocation.json', bundle.relocation)
    bundle.refresh()
    with pytest.raises(ap.Invalid):
        bundle.load()


def test_build_command_cannot_hide_fault_defines_or_wrong_abi(bundle):
    pointer = bundle.unit['builds']['torch']
    build = json.loads((bundle.root / pointer['path']).read_bytes())
    build['command'] = bundle.json('build/bad-argv.json', ['c++', '-D_GLIBCXX_USE_CXX11_ABI=0'])
    bundle.unit['builds']['torch'] = bundle.json('build/torch.json', build)
    bundle.refresh()
    with pytest.raises(ap.Invalid, match=r'C\+\+ ABI'):
        bundle.load()


@pytest.mark.parametrize('change', ['missing-phase', 'missing-required-case', 'skipped-case', 'stale-framework',
                                    'stall-never-reached', 'wrong-operation-build-failed', 'missing-vendor',
                                    'extra-identity-case', 'one-bit-output', 'missing-identity-source'])
def test_qualification_requires_real_scope_and_exact_words(bundle, change):
    bundle.qualify()
    report = bundle.report
    if change == 'missing-phase':
        del report['phases']['optional-import']
    elif change == 'missing-required-case':
        report['phases']['torch']['expected_cases'] = ['add']
        report['phases']['torch']['passed_cases'] = ['add']
    elif change == 'skipped-case':
        report['phases']['max']['skipped'] = ['full_no_inputs']
    elif change == 'stale-framework':
        report['framework'] = report['framework'] | {'torch_version': 'other'}
    elif change == 'stall-never-reached':
        report['faults']['stall']['log'] = bundle.put('fault/stall.log', b'never executed\n')
    elif change == 'wrong-operation-build-failed':
        report['faults']['wrong-operation']['stage'] = 'compile'
    elif change == 'missing-vendor':
        del bundle.identity_doc['columns']['apple']
    elif change == 'extra-identity-case':
        bundle.identity_doc['columns']['amd']['max']['not-a-fixture'] = {}
    elif change == 'missing-identity-source':
        bundle.identity_doc['provenance']['nvidia']['source_sha256'] = {}
    else:
        row = next(iter(bundle.identity_doc['columns']['apple']['torch'].values()))
        raw = bytearray((bundle.root / row['output']['path']).read_bytes())
        raw[0] ^= 1
        row['output'] = bundle.put('words/one-bit.bin', bytes(raw))
    bundle.refresh_qualification()
    with pytest.raises(ap.Invalid):
        bundle.attach()


def test_unguarded_adapter_files_cannot_hide_without_attestation(bundle):
    members = dict(bundle.base_members)
    members[ap.owner(bundle.unit)[1] + '/libmojolearn_adapter.so'] = bundle.payload['libmojolearn_adapter.so']
    bundle.zip(bundle.base_wheel, members)
    with pytest.raises(ap.Invalid, match='lack an owning'):
        ap.check_wheel(bundle.base_wheel)


def test_traversal_and_symlink_payload_are_refused(bundle):
    bundle.unit['files']['libmojolearn_adapter.so']['path'] = '../outside.so'
    bundle.refresh()
    with pytest.raises(ap.Invalid, match='traversing'):
        bundle.load()


def test_no_adapter_wheel_keeps_ordinary_path(bundle):
    original = bundle.base_wheel.read_bytes()
    assert ap.check_wheel(bundle.base_wheel) == []
    assert bundle.base_wheel.read_bytes() == original
