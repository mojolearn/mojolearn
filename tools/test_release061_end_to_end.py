"""Inert file-only release-linux3 admission fixtures. Root alone executes these tests.

All native files are text and never loaded. The ordered numerical comparator
alone is mocked; RECORD, source/build proofs, all75 installed records, corpus
fingerprints, retained evidence and UMAP raw-byte comparisons use real gates.
Synthetic passing quality numbers exercise admission plumbing, not ML quality.
DEVIATION 2290: the fixture declares its own `_version.py` and every version
and profile string below comes from the shared reader; nothing pins a number.
"""
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import check_linux_release_qualification as gate


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))


def bits(n, d):
    return dict(shape=[n, d], uint32=[[0] * d for _ in range(n)],
                float32_le_sha256=hashlib.sha256(bytes(n * d * 4)).hexdigest())


def quality(mode, binding_sha):
    rows = []
    for name, (n, q, d) in gate.surface.FIXTURES.items():
        rows.append(dict(profile=name, passed=True, binding_sha256=binding_sha,
                         binding_mode_code=gate.surface.MODES[mode], fitted_mode=mode,
                         training_input=bits(n, 3), query_input=bits(q, 3),
                         training_embedding=bits(n, d), query_embedding=bits(q, d),
                         parameters={}, transform_schedule={}, fitted_config={},
                         quality=dict(trustworthiness=1.0, retention=1.0),
                         controls={k: dict(trustworthiness=0.0, retention=0.0) for k in
                                   ('query_embedding_permutation', 'training_embedding_permutation')},
                         control_margins=dict(trustworthiness=1.0, retention=1.0)))
    return dict(status='PASS', profile='expanded', mode=mode, k=5, results=rows,
                thresholds=dict(trustworthiness=0.85, retention=0.35, minimum_control_margin=0.15))


def seal_evidence(out):
    record = json.loads((out / 'qualification.json').read_text())
    record['installed_records'] = {
        s + '-' + m: gate.digest_file(out / (s + '-' + m + '.installed.json'))
        for s, m in gate.surface.expected_jobs({'assembly_profile': gate.surface.RELEASE_PROFILE})}
    record['evidence_sha256'] = {
        p.name: gate.digest_file(p) for p in out.iterdir()
        if p.is_file() and p.name not in ('qualification.json', 'exit_code')}
    write_json(out / 'qualification.json', record)


def byte_fixture(out, vendor, binding_sha):
    # Synthetic state changes exercise file admission only, never numerical proof.
    import struct
    states = []
    for step in (0, 1, 1):
        payload = dict(schema='mojolearn.small-byte-lm-state.v1', profile=gate.surface.BYTE_LM_PROFILE, numeric_mode='identical',
                       parameter_names=['fixture' + str(i) for i in range(20)],
                       parameter_shapes=[[1]] * 19 + [[34925]],
                       parameter_offsets=list(range(20)) + [34944],
                       config={'kind': 2}, data_schedule={'fixture': 'inert'},
                       completed_steps=step, next_batch_index=step)
        for key in ('parameters', 'm', 'v'):
            value = float(step + (1 if key == 'parameters' else 0))
            payload[key] = dict(dtype='<f4', shape=[34944], hex=(struct.pack('<f', value) * 34944).hex())
        # DEVIATION 2296: the fixture used to flip the flags 0 -> 1 with the
        # step, which is what the old check demanded and what no real run has
        # ever produced. config kind=2 is AdamW; `flags` is SGD's
        # buffer-initialized marker and stays zero. The first installed
        # qualification on real gfx942 silicon returned zeros, and the
        # gradient oracle requires the step to leave them untouched.
        payload['flags'] = dict(dtype='<i4', shape=[20], hex=bytes(80).hex())
        states.append(payload)
    hashes = {}
    for name, payload in zip(('byte-lm-before.json', 'byte-lm-after.json', 'byte-lm-restored.json'), states):
        envelope = dict(schema='mojolearn.small-byte-lm-json-checkpoint.v1', payload=payload,
                        payload_sha256=hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        write_json(out / name, envelope)
        hashes[name] = gate.digest_file(out / name)
    write_json(out / 'byte-lm-identical.json', dict(schema='mojolearn.installed-byte-lm-step.v1', status='PASS',
        completed_steps=1, loss=5.0, evaluation_loss=4.9,
        metadata=dict(profile=gate.surface.BYTE_LM_PROFILE, native_profile=gate.surface.BYTE_LM_PROFILE,
                      native_vendor=vendor, native_numeric_mode=1, binding_sha256=binding_sha),
        gradients_hex=(struct.pack('<f', 1.0) * 34944).hex(), checkpoint_sha256=hashes))


# DEVIATION 2293: which three architectures the fixture wheel carries. The
# Hopper slot has two legal spellings and only sm_90a is buildable on an H100,
# so the gate has to be exercised with both. Default is the canonical triple;
# test_hopper_spelled_sm_90a_admits swaps it.
ARCHES = None


def split_fixture(wheel, version):
    """The split set (mojolearn_core, mojolearn_nvidia, mojolearn_amd as the
    payload's sets require, and the file-free Linux mojolearn when both vendors
    are packed, as pack_wheel does since 2026-10-20) the fixture's combined
    wheel is a partition of, with the .dist-info contents
    tools/wheel_api_audit.py split_audit requires. The combined wheel is
    removed."""
    plugins = gate.surface.load_gpu_plugins()
    dist = 'mojolearn-' + version + '.dist-info/'
    with zipfile.ZipFile(wheel) as archive:
        members = {n: archive.read(n) for n in archive.namelist()}
    wheel.unlink()
    payload = json.loads(members[dist + 'LINUX_PAYLOAD.json'])
    # pack_wheel.write_split: every GPU member moves to its vendor package's
    # equal-depth directory (gpu_plugins.installed_member: mojolearn/cuda/sm_89/...
    # -> mojolearn/cuda_native/sm_89/...), and the payload's member maps follow.
    for mapping in ('extensions', 'binding_origin', 'merged_indexes'):
        if mapping in payload:
            payload[mapping] = {plugins.installed_member(k): v for k, v in payload[mapping].items()}
    tag = 'py3-none-manylinux_2_35_x86_64'
    parts = {}
    for name, data in members.items():
        if not name.startswith(dist):
            installed = plugins.installed_member(name)
            # the PTX set ships in its own package, mojolearn-nvidia-ptx (required by mojolearn-nvidia since 2026-10-19)
            owner = plugins.PTX_PROFILE if installed.startswith('mojolearn/' + plugins.PTX_DIRECTORY + '/') else plugins.member_vendor(name)
            parts.setdefault(owner, {})[installed] = data
    # the PTX set's manifest (gpu_plugins.validate_baseline_manifest, as packaging/linux/ptx_baseline.py writes it); the
    # NVIDIA marker and the PTX marker both bind its digest
    ptx_bundle = None
    if plugins.PTX_PROFILE in parts:
        prefix = 'mojolearn/' + plugins.PTX_DIRECTORY + '/' + plugins.PTX_ARCH + '/'
        listed = []
        for member, data in sorted(parts[plugins.PTX_PROFILE].items()):
            rel = member[len(prefix):]
            if not member.startswith(prefix) or not rel.split('/')[-1].startswith('_mojolearn') or rel.startswith(('host/', '.libs/')):
                continue
            listed.append(dict(file=rel, numeric_mode=rel.split('/', 1)[0] if '/' in rel else 'fast',
                               sha256=hashlib.sha256(data).hexdigest(),
                               ptx_modules=[dict(target=plugins.PTX_ARCH, sha256='b' * 64)]))
        manifest = json.dumps(dict(schema=plugins.PTX_MANIFEST_SCHEMA, code_format=plugins.PTX_CODE_FORMAT, vendor='cuda',
                                   target=plugins.PTX_ARCH, min_compute_capability=[8, 0],
                                   source_commit=payload.get('source_commit'), source_dirty=False, mojo_version='test',
                                   errors=[], files=listed)).encode()
        parts[plugins.PTX_PROFILE][prefix + plugins.BASELINE_MANIFEST] = manifest
        ptx_bundle = dict(manifest_sha256=hashlib.sha256(manifest).hexdigest())
    out = []
    entries = sorted(parts.items(), key=lambda kv: kv[0] or '')
    if set(plugins.vendors()) <= set(parts):
        entries.append((plugins.META_PROFILE, {}))
    for vendor, files in entries:
        if vendor is None:
            wdist = plugins.CORE_WHEEL_NAME + '-' + version + '.dist-info/'
            role, dist_name = plugins.CORE_PROFILE, plugins.CORE_DISTRIBUTION
            # the core (mojolearn-core) declares no extras and requires no vendor
            meta = ['Metadata-Version: 2.4', 'Name: ' + dist_name, 'Version: ' + version]
            meta += ['Requires-Dist: ' + req for req in plugins.core_requirements(version)]
            files[wdist + plugins.CORE_MARKER] = json.dumps(plugins.core_marker(version)).encode()
            name = plugins.CORE_WHEEL_NAME + '-' + version + '-' + tag + '.whl'
        elif vendor == plugins.META_PROFILE:
            # the file-free Linux mojolearn: the core and both vendors at this version
            wdist = plugins.META_WHEEL_NAME + '-' + version + '.dist-info/'
            role, dist_name = plugins.META_PROFILE, plugins.META_DISTRIBUTION
            meta = ['Metadata-Version: 2.4', 'Name: ' + dist_name, 'Version: ' + version]
            meta += ['Requires-Dist: ' + req for req in plugins.meta_requirements(version)]
            files[wdist + plugins.PLUGIN_MARKER] = json.dumps(plugins.meta_marker(version)).encode()
            name = plugins.META_WHEEL_NAME + '-' + version + '-' + tag + '.whl'
        elif vendor == plugins.PTX_PROFILE:
            row = plugins.PAYLOADS[plugins.PTX_PROFILE]
            wdist, role, dist_name = row['wheel_name'] + '-' + version + '.dist-info/', row['profile'], row['distribution']
            meta = ['Metadata-Version: 2.4', 'Name: ' + dist_name, 'Version: ' + version]
            meta += ['Requires-Dist: ' + req for req in plugins.package_requirements(row['profile'], version)]
            arches = {n.split('/')[2] for n in files}
            files[wdist + plugins.PLUGIN_MARKER] = json.dumps(plugins.package_marker(
                row['profile'], version, arches, bundled_ptx=ptx_bundle)).encode()
            name = row['wheel_name'] + '-' + version + '-' + tag + '.whl'
        else:
            row = plugins.PLUGINS[vendor]
            wdist, role, dist_name = row['wheel_name'] + '-' + version + '.dist-info/', row['profile'], row['distribution']
            # each vendor requires mojolearn-core (NVIDIA also its PTX package)
            meta = ['Metadata-Version: 2.4', 'Name: ' + dist_name, 'Version: ' + version]
            meta += ['Requires-Dist: ' + req for req in plugins.package_requirements(row['profile'], version)]
            arches = {n.split('/')[2] for n in files}
            # the NVIDIA marker binds its PTX package's manifest by digest
            # (gpu_plugins.ptx_shipped); the PTX wheel itself is not in this set
            bundle = (ptx_bundle or dict(manifest_sha256='c' * 64)) if row.get('ptx') else None
            files[wdist + plugins.PLUGIN_MARKER] = json.dumps(
                plugins.plugin_marker(vendor, version, arches, bundled_ptx=bundle)).encode()
            name = row['wheel_name'] + '-' + version + '-' + tag + '.whl'
        files[wdist + 'METADATA'] = ('\n'.join(meta) + '\n').encode()
        files[wdist + 'WHEEL'] = ('Wheel-Version: 1.0\nRoot-Is-Purelib: false\nTag: ' + tag + '\n').encode()
        native = sorted(n for n in files if n.endswith('.so') or '/.libs/' in n)
        files[wdist + 'LINUX_PAYLOAD.json'] = json.dumps(
            dict(payload, split=dict(role=role, distribution=dist_name, profile=gate.SPLIT_PROFILE,
                                     native_members=native))).encode()
        rows = io.StringIO()
        writer = csv.writer(rows)
        for member, data in files.items():
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
            writer.writerow([member, 'sha256=' + digest, str(len(data))])
        writer.writerow([wdist + 'RECORD', '', ''])
        files[wdist + 'RECORD'] = rows.getvalue().encode()
        path = wheel.parent / name
        with zipfile.ZipFile(path, 'w') as archive:
            for member, data in files.items():
                archive.writestr(member, data)
        out.append(path)
    return out


def index_bytes(files):
    """A set directory's _mojolearn_merged.json for {'<stem>.so': [members]}."""
    return json.dumps({'format': 1, 'files': files}, sort_keys=True).encode()


def fixture(root, split=False, merged=False):
    """The release-linux3 fixture in the per-binding layout, or with `merged`
    in the merged layout: each set directory (and host/) ships the plan's
    merged files plus its index, the build proofs and the payload carry
    `merged_indexes` and `binding_layout`, and every installed member names
    the merged file its directory's index lists it in."""
    layout = 'merged' if merged else 'per-binding'
    indexes = {}  # 'mojolearn/<dir>/_mojolearn_merged.json' -> {'sha256', 'files'}

    def add_index(files, prefix, plan):
        member = prefix + gate.surface.MERGED_INDEX
        files[member] = index_bytes(plan)
        indexes[member] = dict(sha256=hashlib.sha256(files[member]).hexdigest(), files=plan)
        return member

    wrapper = root / 'python/mojolearn/__init__.py'
    wrapper.parent.mkdir(parents=True)
    wrapper.write_bytes(b'# inert wrapper fixture\n')
    # DEVIATION 2290: the fixture's own _version.py is the only version source;
    # it is packaged like every other python/mojolearn/*.py (flat_python check).
    version_file = root / 'python/mojolearn/_version.py'
    version_file.write_bytes(b'__version__ = "9.9.9"\n')
    version = gate.surface.release_version(root)
    snapshot = {}
    for case in gate.surface.CORPUS_CASES:
        p = root / 'mamba/corpus' / case / 'x.f32'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'\0' * 4)
        snapshot[p.relative_to(root).as_posix()] = gate.digest_file(p)
    inventory = gate.native_inventory(root)
    source_sha = gate.inventory_digest(inventory)
    files = {'mojolearn/__init__.py': wrapper.read_bytes(), 'mojolearn/_version.py': version_file.read_bytes(),
             'mojolearn/.libs/libfixture.so': b'inert runtime'}
    # DEVIATION 2680: the host bindings. ONE vendor-neutral copy of each for
    # the whole wheel, outside every architecture tree, so they are excluded
    # from the payload's `extensions` map below and recorded under
    # `host_native` (keyed by basename since 0.8.6) instead.
    host_plan = gate.surface.expected_set_files('host', True, layout)
    for stem in host_plan:
        member = 'mojolearn/host/' + stem
        files[member] = b'inert native ' + member.encode()
    if merged:
        add_index(files, 'mojolearn/host/', host_plan)
    qualification_root = root / 'qualification'
    proof_root = qualification_root / 'build-proofs'
    proof_root.mkdir(parents=True)
    proof_sets = {}
    for key in sorted(ARCHES or gate.RELEASE_ARCHES):
        extensions = {}
        for mode in gate.surface.MODES:
            prefix = 'mojolearn/' + key + '/' + ('' if mode == 'fast' else mode + '/')
            plan = gate.surface.expected_set_files(mode, True, layout)
            for stem in sorted(plan):
                member = prefix + stem
                files[member] = ('inert native ' + member).encode()
                extensions[member] = hashlib.sha256(files[member]).hexdigest()
            if merged:
                add_index(files, prefix, plan)
        proof = dict(schema='mojolearn.linux.build-provenance.v1',
            complete=True, build_exit=0, action='build', source_commit='a' * 40,
            source_inventory=inventory, source_sha256=source_sha, extensions=extensions)
        if merged:
            # the leg's indexes at the build layout, its host index included
            proof.update(binding_layout='merged', merged_indexes=dict(
                {k: v for k, v in indexes.items() if k.startswith('mojolearn/' + key + '/')},
                **{'mojolearn/' + key + '/host/' + gate.surface.MERGED_INDEX: indexes['mojolearn/host/' + gate.surface.MERGED_INDEX]}))
        proof_path = proof_root / (key.replace('/', '-') + '.json')
        write_json(proof_path, proof)
        proof_sets[key] = dict(sha256=gate.digest_file(proof_path), source_sha256=source_sha)
    payload = dict(schema='mojolearn.linux-payload.v1', version=version,
        assembly_profile=gate.SPLIT_PROFILE if split else gate.surface.RELEASE_PROFILE, release_profile='alpha-api', source_commit='a' * 40,
        source_inventory=inventory, sets=proof_sets,
        extensions={n: hashlib.sha256(b).hexdigest() for n, b in files.items()
                    if '/_mojolearn' in n and n.endswith('.so') and not n.startswith('mojolearn/host/')},
        host_native={name: dict(archive_path=member,
                                sha256=hashlib.sha256(files[member]).hexdigest(),
                                vendor='cpu', supported_modes=['identical'],
                                unsupported_modes=['fast'])
                     for name in gate.HOST_NAMES
                     for member in (gate.surface.member_file('mojolearn/host/', name, indexes),)},
        python_sha256={'mojolearn/__init__.py': gate.digest_file(wrapper),
                       'mojolearn/_version.py': gate.digest_file(version_file)},
        runtime_sha256={'mojolearn/.libs/libfixture.so': hashlib.sha256(b'inert runtime').hexdigest()},
        optional_native={n: {'included': True, 'supported_modes': ['identical'],
                             'unsupported_modes': ['fast']}
                         for n in ('_mojolearn_byte_lm',) + tuple(gate.HOST_NAMES)})
    if merged:
        payload.update(binding_layout='merged', merged_indexes=indexes)
    dist = 'mojolearn-' + version + '.dist-info/'
    files[dist + 'LINUX_PAYLOAD.json'] = json.dumps(payload).encode()
    files[dist + 'METADATA'] = ('Metadata-Version: 2.4\nName: mojolearn\nVersion: ' + version + '\n').encode()
    record_path = dist + 'RECORD'
    rows = io.StringIO()
    writer = csv.writer(rows)
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow([name, 'sha256=' + digest, str(len(data))])
    writer.writerow([record_path, '', ''])
    files[record_path] = rows.getvalue().encode()
    wheel = root / ('mojolearn-' + version + '-py3-none-manylinux_2_35_x86_64.whl')
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    if split:
        wheel = split_fixture(wheel, version)
    for key in sorted(ARCHES or gate.RELEASE_ARCHES):
        vendor, arch = key.split('/')
        out = qualification_root / key
        out.mkdir(parents=True)
        audit = (gate.split_release_audit(wheel, root, proof_root, key) if split
                 else gate.release_audit(wheel, root, proof_root, key))
        write_json(out / 'wheel-audit.json', audit)
        shutil.copyfile(proof_root / (key.replace('/', '-') + '.json'), out / 'build-provenance.json')
        write_json(out / 'qualification-sources.json', snapshot)
        (out / 'installed-dependencies.txt').write_text('fixture-only\n')
        (out / 'dependency-check.log').write_text('fixture-only\n')
        status = []
        package = '/retained/' + key + '/venv/lib/python3.12/site-packages/mojolearn'
        for surface, mode in sorted(gate.surface.expected_jobs(audit)):
            for code in (gate.surface.MODES[mode],):
                name = surface + '-' + mode
                prefix = key + '/' + ('' if mode == 'fast' else mode + '/')
                if split:  # a split install holds the set at its vendor package's directory
                    prefix = gate.surface.load_gpu_plugins().installed_member('mojolearn/' + prefix + 'x.so')[
                        len('mojolearn/'):-len('x.so')]
                # each member loads from its own file or its directory's merged file
                served = audit.get('merged_indexes') or {}
                bindings = {binding: dict(path=package + '/' + gate.surface.member_file(prefix, binding, served),
                    sha256=audit['extension_hashes'][gate.surface.member_file(prefix, binding, served)], mode_code=code)
                    for binding in gate.surface.expected_bindings(mode, True)}
                # DEVIATION 2680: the CPU TRAINING binding, under its own key.
                # It is NOT in installed_bindings, whose every row is asserted
                # to read back this leg's GPU vendor; this one reads back 'cpu'.
                host_member = gate.surface.member_file('host/', gate.HOST_NAME, served)
                write_json(out / (name + '.installed.json'), dict(vendor=vendor, mode=mode,
                    wheel_sha256=audit['sha256'], package=package + '/__init__.py',
                    installed_bindings=bindings, device_architecture=arch,
                    selected_architecture=arch, architecture_probe='synthetic fixture witness',
                    architecture_override_absent=True,
                    installed_host_binding=dict(path=package + '/' + host_member,
                        sha256=audit['host_extension_hashes'][host_member],
                        numeric_mode=1, vendor='cpu'),
                    installed_host_bindings={name: dict(
                        path=package + '/' + member,
                        sha256=audit['host_extension_hashes'][member],
                        numeric_mode=1, vendor='cpu', column='cpu')
                        for name in gate.HOST_NAMES
                        for member in (gate.surface.member_file('host/', name, served),)}))
                (out / (name + '.log')).write_text('inert fixture; no numerical execution\n')
                status.append(f'{surface}\t{mode}\t0\n')
                if surface == 'byte-lm':
                    byte_fixture(out, vendor, bindings['_mojolearn_byte_lm']['sha256'])
                if surface == 'umap-quality':
                    write_json(out / (name + '.json'), quality(mode, bindings['_mojolearn_metrics']['sha256']))
        (out / 'results.tsv').write_text(''.join(status))
        (out / 'exit_code').write_text('0\n')
        write_json(out / 'qualification.json', dict(schema='mojolearn.linux.installed-surfaces.v1',
            status='PASSED', vendor=vendor, wheel_sha256=audit['sha256'], source_sha256=source_sha))
        seal_evidence(out)
    return wheel, qualification_root


class EndToEndRelease061(unittest.TestCase):
    def test_positive_full_file_linkage_and_mutations(self):
        for defect in (None, 'missing_architecture', 'stale_final_hash', 'stale_source',
                       'mislabeled_device', 'mislabeled_binding', 'byte_mode', 'byte_profile', 'byte_gradient', 'byte_checkpoint'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                wheel, qualification = fixture(root)
                out = qualification / 'cuda/sm_90'
                if defect == 'missing_architecture':
                    shutil.rmtree(out)
                elif defect == 'stale_final_hash':
                    # A valid ZIP trailing comment changes final bytes without
                    # changing payload or RECORD, so only exact-wheel linkage catches it.
                    with zipfile.ZipFile(wheel, 'a') as archive:
                        archive.comment = b'changed final artifact'
                elif defect == 'stale_source':
                    (root / 'new.mojo').write_text('new native source')
                elif defect in ('mislabeled_device', 'mislabeled_binding'):
                    path = out / 'smoke-fast.installed.json'
                    row = json.loads(path.read_text())
                    if defect == 'mislabeled_device':
                        row['device_architecture'] = 'sm_89'
                    else:
                        b = row['installed_bindings']['_mojolearn_gbdt']
                        b['path'] = b['path'].replace('/cuda/sm_90/_', '/cuda/sm_89/_')
                    write_json(path, row)
                    seal_evidence(out)  # defeat superficial stale-record hash rejection
                if defect in ('byte_mode', 'byte_profile', 'byte_gradient', 'byte_checkpoint'):
                    path = out / 'byte-lm-identical.json'
                    row = json.loads(path.read_text())
                    if defect == 'byte_mode': row['metadata']['native_numeric_mode'] = 0
                    elif defect == 'byte_profile': row['metadata']['native_profile'] = 'wrong'
                    elif defect == 'byte_gradient': row['gradients_hex'] = '00' * (34944 * 4)
                    else: (out / 'byte-lm-after.json').write_text('{}')
                    write_json(path, row)
                    seal_evidence(out)
                with patch.object(gate.compare_ordered_python, 'compare', return_value={'status': 'PASSED'}) as ordered:
                    if defect:
                        with self.assertRaises((ValueError, OSError)):
                            gate.check_release061(wheel, qualification, root)
                    else:
                        result = gate.check_release061(wheel, qualification, root)
                        self.assertEqual(result['status'], 'PASSED')
                        self.assertEqual(set(result['runtime_coverage']), gate.RELEASE_ARCHES)
                        self.assertEqual(result['jobs_per_runtime_architecture'], 10)  # smoke x {fast, identical}, 7 identical surfaces, byte-lm (fast + identical only since d4241beeb)
                        self.assertEqual(ordered.call_count, 2)
                        self.assertEqual(result['assembly_profile'], gate.surface.RELEASE_PROFILE)

    def test_deprecated_profile_alias_in_retained_audit_still_admits(self):
        # DEVIATION 2290: evidence written under `release-0.6.1` parses as
        # release-linux3 and the admission record says the current name.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel, qualification = fixture(root)
            out = qualification / 'hip/gfx942'
            path = out / 'wheel-audit.json'
            audit = json.loads(path.read_text())
            audit['assembly_profile'] = 'release-0.6.1'
            write_json(path, audit)
            seal_evidence(out)
            with patch.object(gate.compare_ordered_python, 'compare', return_value={'status': 'PASSED'}):
                result = gate.check_release061(wheel, qualification, root)
            self.assertEqual(result['status'], 'PASSED')
            self.assertEqual(result['assembly_profile'], gate.surface.RELEASE_PROFILE)

    def test_wheel_of_another_version_refused(self):
        # DEVIATION 2290: the wheel must carry the version the source root declares.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel, qualification = fixture(root)
            (root / 'python/mojolearn/_version.py').write_bytes(b'__version__ = "9.9.10"\n')
            with patch.object(gate.compare_ordered_python, 'compare', return_value={'status': 'PASSED'}), \
                    self.assertRaises((ValueError, OSError, KeyError)):
                gate.check_release061(wheel, qualification, root)


class MergedLayout(unittest.TestCase):
    """The merged bindings (lane merged-bindings, 2026-10-20): every set ships
    its plan's merged files and an index, and the same admission holds."""

    def admit(self, wheel, qualification, root, split=False):
        with patch.object(gate.compare_ordered_python, 'compare', return_value={'status': 'PASSED'}):
            if split:
                return gate.check_release_split(wheel, qualification, root)
            return gate.check_release061(wheel, qualification, root)

    def test_merged_combined_wheel_admits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel, qualification = fixture(root, merged=True)
            result = self.admit(wheel, qualification, root)
            self.assertEqual(result['status'], 'PASSED')
            self.assertEqual(set(result['runtime_coverage']), gate.RELEASE_ARCHES)
            audit = json.loads((qualification / 'hip/gfx942/wheel-audit.json').read_text())
            self.assertEqual(audit['binding_layout'], 'merged')
            self.assertEqual(len(audit['merged_indexes']), 2 * len(gate.RELEASE_ARCHES) + 1)  # two tiers per set, host/
            self.assertTrue(all(p.endswith('.so') for p in audit['extension_hashes']))

    def test_merged_split_set_admits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheels, qualification = fixture(root, split=True, merged=True)
            result = self.admit(wheels, qualification, root, split=True)
            self.assertEqual(result['status'], 'PASSED')
            self.assertEqual(result['assembly_profile'], gate.SPLIT_PROFILE)

    def test_merged_defects_refuse(self):
        for defect in ('proof_per_binding', 'proof_index_changed', 'installed_own_file'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                wheel, qualification = fixture(root, merged=True)
                out = qualification / 'cuda/sm_90'
                if defect.startswith('proof_'):
                    path = qualification / 'build-proofs/cuda-sm_90.json'
                    proof = json.loads(path.read_text())
                    if defect == 'proof_per_binding':
                        proof.pop('binding_layout')
                        proof.pop('merged_indexes')
                    else:
                        row = next(r for k, r in proof['merged_indexes'].items() if '/identical/' in k)
                        row['sha256'] = '0' * 64
                    write_json(path, proof)
                else:
                    path = out / 'smoke-fast.installed.json'
                    row = json.loads(path.read_text())
                    b = row['installed_bindings']['_mojolearn_gbdt']
                    b['path'] = b['path'].rsplit('/', 1)[0] + '/_mojolearn_gbdt.so'
                    write_json(path, row)
                    seal_evidence(out)
                with self.assertRaises((ValueError, OSError, KeyError)):
                    self.admit(wheel, qualification, root)


    def test_merged_host_bindings_resolve_to_the_merged_host_file(self):
        # The on-box verifier's host read-back (verify -> check_byte_lm_host,
        # check_host_bindings) looks each host binding up in the merged host file.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel, qualification = fixture(root, merged=True)
            out = qualification / 'hip/gfx942'
            audit = json.loads((out / 'wheel-audit.json').read_text())
            record = json.loads((out / 'byte-lm-identical.installed.json').read_text())
            gate.surface.check_byte_lm_host(record, audit)
            gate.surface.check_host_bindings(record, audit)
            record['installed_host_bindings'][gate.HOST_NAME]['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                gate.surface.check_host_bindings(record, audit)


class SplitSet(unittest.TestCase):
    """--profile release-split: the split set admitted as the combined wheel
    it partitions, identified by its set digest."""

    def check(self, wheels, qualification, root):
        with patch.object(gate.compare_ordered_python, 'compare', return_value={'status': 'PASSED'}) as ordered:
            return gate.check_release_split(wheels, qualification, root), ordered

    def test_three_wheel_set_admits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheels, qualification = fixture(root, split=True)
            self.assertEqual(sorted(w.name.split('-')[0] for w in wheels),
                             ['mojolearn', 'mojolearn_amd', 'mojolearn_core', 'mojolearn_nvidia'])
            result, ordered = self.check(wheels, qualification, root)
            self.assertEqual(result['status'], 'PASSED')
            self.assertEqual(result['assembly_profile'], gate.SPLIT_PROFILE)
            self.assertEqual(result['wheel_sha256'], gate.split_set_digest(wheels))
            self.assertEqual(set(result['wheels']), {w.name for w in wheels})
            self.assertEqual(set(result['runtime_coverage']), gate.RELEASE_ARCHES)
            self.assertEqual(ordered.call_count, 2)

    def test_nvidia_only_set_admits_without_a_cross_vendor_comparison(self):
        global ARCHES
        # the NVIDIA plugin's slots as the registry has them (gpu_plugins.required_sets: sm_89 alone
        # since 0.8.38 dropped sm_90a and 0.8.39 took the PTX slot out while over PyPI's limit)
        ARCHES, saved = set(gate.surface.PLUGIN_ARCHES['cuda']), ARCHES
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                wheels, qualification = fixture(root, split=True)
                # mojolearn_core, mojolearn_nvidia and its PTX package; no Linux mojolearn without AMD
                self.assertEqual(sorted(w.name.split('-')[0] for w in wheels), ['mojolearn_core', 'mojolearn_nvidia', 'mojolearn_nvidia_ptx'])
                result, ordered = self.check(wheels, qualification, root)
                self.assertEqual(result['vendors'], ['cuda'])
                self.assertEqual(ordered.call_count, 0)
        finally:
            ARCHES = saved

    def test_a_missing_plugin_a_changed_wheel_and_the_combined_audit_refuse(self):
        for defect in ('missing_plugin', 'changed_plugin', 'combined_digest'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                wheels, qualification = fixture(root, split=True)
                if defect == 'missing_plugin':
                    wheels = [w for w in wheels if not w.name.startswith('mojolearn_amd')]
                elif defect == 'changed_plugin':
                    with zipfile.ZipFile(next(w for w in wheels if w.name.startswith('mojolearn_nvidia')), 'a') as z:
                        z.comment = b'changed final artifact'
                else:
                    # a qualification that recorded one wheel's digest, not the set's
                    for out in qualification.glob('*/*/qualification.json'):
                        doc = json.loads(out.read_text())
                        doc['wheel_sha256'] = gate.digest_file(wheels[0])
                        write_json(out, doc)
                        seal_evidence(out.parent)
                with self.assertRaises((ValueError, OSError, KeyError)):
                    self.check(wheels, qualification, root)

    def test_the_cli_takes_the_set(self):
        import subprocess, sys
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheels, qualification = fixture(root, split=True)
            script = Path(gate.__file__)
            r = subprocess.run([sys.executable, str(script), *map(str, wheels[:2]), '--qualification-root',
                                str(qualification), '--source-root', str(root)], capture_output=True, text=True)
            self.assertEqual(r.returncode, 2)
            self.assertIn('one wheel, except with --profile release-split', r.stderr)


class HopperSpelling(unittest.TestCase):
    """DEVIATION 2293. sm_90 has never been built; the compiler emits sm_90a on
    an H100 and every read-back gate refuses a set named otherwise. The whole
    admission path therefore has to accept a wheel whose Hopper slot is spelled
    sm_90a, and has to keep refusing one that fills that slot twice."""

    def _run(self, arches):
        global ARCHES
        ARCHES = frozenset(arches)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                wheel, qualification = fixture(root)
                with patch.object(gate.compare_ordered_python, 'compare',
                                  return_value={'status': 'PASSED'}):
                    return gate.check_release061(wheel, qualification, root)
        finally:
            ARCHES = None

    def test_hopper_spelled_sm_90a_admits(self):
        result = self._run({'cuda/sm_89', 'cuda/sm_90a', 'hip/gfx942'})
        self.assertEqual(result['status'], 'PASSED')
        self.assertEqual(set(result['runtime_coverage']),
                         {'cuda/sm_89', 'cuda/sm_90a', 'hip/gfx942'})
        self.assertEqual(result['jobs_per_runtime_architecture'], 10)  # smoke x {fast, identical}, 7 identical surfaces, byte-lm (fast + identical only since d4241beeb)
        self.assertIn(f"FULL {result['jobs_per_runtime_architecture']} installed jobs", result['scope'])

    def test_hopper_spelled_sm_90_still_admits(self):
        result = self._run({'cuda/sm_89', 'cuda/sm_90', 'hip/gfx942'})
        self.assertEqual(result['status'], 'PASSED')
        self.assertEqual(set(result['runtime_coverage']),
                         {'cuda/sm_89', 'cuda/sm_90', 'hip/gfx942'})

    def test_hopper_slot_filled_twice_refused(self):
        with self.assertRaises((ValueError, OSError, KeyError)):
            self._run({'cuda/sm_89', 'cuda/sm_90', 'cuda/sm_90a', 'hip/gfx942'})

    def test_missing_hopper_refused(self):
        with self.assertRaises((ValueError, OSError, KeyError)):
            self._run({'cuda/sm_89', 'cuda/sm_86', 'hip/gfx942'})


class SmokeTier(unittest.TestCase):
    """DEVIATION 2297. One FULL 25-job column per advertised vendor; every other
    architecture the wheel carries clears SMOKE. The tier is DECLARED by a
    marker file, never inferred from a failing full check -- an earlier draft
    inferred it and turned three injected defects into silent downgrades."""

    def _root(self, tmp, smoke_keys):
        root = Path(tmp)
        wheel, qualification = fixture(root)
        for key in smoke_keys:
            write_json(qualification / key / 'SMOKE_TIER.json',
                       dict(schema='mojolearn.linux.smoke-tier.v1', architecture=key,
                            reason='fixture: reduced tier for this architecture'))
        return wheel, qualification, root

    def _check(self, wheel, qualification, root):
        with patch.object(gate.compare_ordered_python, 'compare', return_value={'status': 'PASSED'}):
            return gate.check_release061(wheel, qualification, root)

    def test_one_full_per_vendor_admits(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, q, r = self._root(tmp, ['cuda/sm_90'])          # sm_89 full, gfx942 full
            result = self._check(w, q, r)
            self.assertEqual(result['status'], 'PASSED')
            self.assertEqual(result['qualification_tiers']['cuda/sm_90'], 'smoke')
            self.assertEqual(result['qualification_tiers']['cuda/sm_89'], 'full')
            self.assertEqual(result['qualification_tiers']['hip/gfx942'], 'full')

    def test_no_full_cuda_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, q, r = self._root(tmp, ['cuda/sm_89', 'cuda/sm_90'])
            with self.assertRaises((ValueError, OSError, KeyError)):
                self._check(w, q, r)

    def test_no_full_hip_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, q, r = self._root(tmp, ['hip/gfx942'])
            with self.assertRaises((ValueError, OSError, KeyError)):
                self._check(w, q, r)

    def test_marker_must_name_its_own_architecture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel, qualification = fixture(root)
            write_json(qualification / 'cuda/sm_90' / 'SMOKE_TIER.json',
                       dict(schema='mojolearn.linux.smoke-tier.v1',
                            architecture='hip/gfx942', reason='wrong architecture'))
            with self.assertRaises((ValueError, OSError, KeyError)):
                self._check(wheel, qualification, root)

    def test_marker_must_give_a_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            wheel, qualification = fixture(root)
            write_json(qualification / 'cuda/sm_90' / 'SMOKE_TIER.json',
                       dict(schema='mojolearn.linux.smoke-tier.v1',
                            architecture='cuda/sm_90', reason='   '))
            with self.assertRaises((ValueError, OSError, KeyError)):
                self._check(wheel, qualification, root)


class DeclaredKnownFailures(unittest.TestCase):
    """DEVIATION 2299. A full column may carry a KNOWN failing job, declared by
    name and cited. An undeclared failure still refuses; a declaration for a job
    that passed is refused as stale; the declaration is republished."""

    def _fixture(self, tmp, fail_job=None, declare=None, citation='README.md'):
        root = Path(tmp)
        wheel, qualification = fixture(root)
        # the citation is resolved against the SOURCE ROOT, which here is tmp
        (root / 'README.md').write_text('fixture citation target\n')
        col = qualification / 'cuda/sm_89'
        if fail_job:
            surf, mode = fail_job.split('/')
            rows = (col / 'results.tsv').read_text().splitlines()
            rows = [r if not r.startswith(surf + '\t' + mode) else surf + '\t' + mode + '\t1'
                    for r in rows]
            (col / 'results.tsv').write_text('\n'.join(rows) + '\n')
            write_json(col / 'qualification.json',
                       dict(status='FAILED', reason='Missing, duplicate or failed installed job'))
        if declare:
            write_json(col / 'KNOWN_FAILURES.json', dict(
                schema='mojolearn.linux.known-failures.v1',
                failures=[dict(job=declare, citation=citation, reason='fixture: known open item')]))
        return wheel, qualification, root

    def _check(self, w, q, r):
        with patch.object(gate.compare_ordered_python, 'compare', return_value={'status': 'PASSED'}):
            return gate.check_release061(w, q, r)

    def test_declared_failure_admits_and_is_republished(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, q, r = self._fixture(tmp, fail_job='mamba/identical', declare='mamba/identical')
            result = self._check(w, q, r)
            self.assertEqual(result['status'], 'PASSED')
            self.assertEqual(result['qualification_tiers']['cuda/sm_89'], 'full')
            declared = result['observed_job_failures']['cuda/sm_89']
            self.assertEqual(declared[0]['job'], 'mamba/identical')
            self.assertTrue(declared[0]['citation'])

    def test_undeclared_failure_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, q, r = self._fixture(tmp, fail_job='mamba/identical')
            with self.assertRaises((ValueError, OSError, KeyError)):
                self._check(w, q, r)

    def test_declaring_one_job_does_not_excuse_another(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, q, r = self._fixture(tmp, fail_job='smoke/fast', declare='mamba/identical')
            with self.assertRaises((ValueError, OSError, KeyError)):
                self._check(w, q, r)

    def test_stale_declaration_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, q, r = self._fixture(tmp, declare='mamba/identical')   # nothing failed
            with self.assertRaises((ValueError, OSError, KeyError)):
                self._check(w, q, r)

    def test_citation_must_exist_in_the_source_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            w, q, r = self._fixture(tmp, fail_job='mamba/identical',
                                    declare='mamba/identical',
                                    citation='docs/NO_SUCH_DOCUMENT.md')
            with self.assertRaises((ValueError, OSError, KeyError)):
                self._check(w, q, r)


if __name__ == '__main__':
    unittest.main()
