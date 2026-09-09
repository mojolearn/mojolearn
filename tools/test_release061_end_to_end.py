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


def fixture(root):
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
    qualification_root = root / 'qualification'
    proof_root = qualification_root / 'build-proofs'
    proof_root.mkdir(parents=True)
    proof_sets = {}
    for key in sorted(ARCHES or gate.RELEASE_ARCHES):
        extensions = {}
        for mode in gate.surface.MODES:
            prefix = 'mojolearn/' + key + '/' + ('' if mode == 'fast' else mode + '/')
            for name in sorted(gate.surface.expected_bindings(mode, True)):
                member = prefix + name + '.so'
                files[member] = ('inert native ' + member).encode()
                extensions[member] = hashlib.sha256(files[member]).hexdigest()
        proof_path = proof_root / (key.replace('/', '-') + '.json')
        write_json(proof_path, dict(schema='mojolearn.linux.build-provenance.v1',
            complete=True, build_exit=0, action='build', source_commit='a' * 40,
            source_inventory=inventory, source_sha256=source_sha, extensions=extensions))
        proof_sets[key] = dict(sha256=gate.digest_file(proof_path), source_sha256=source_sha)
    payload = dict(schema='mojolearn.linux-payload.v1', version=version,
        assembly_profile=gate.surface.RELEASE_PROFILE, release_profile='alpha-api', source_commit='a' * 40,
        source_inventory=inventory, sets=proof_sets,
        extensions={n: hashlib.sha256(b).hexdigest() for n, b in files.items() if '/_mojolearn' in n},
        python_sha256={'mojolearn/__init__.py': gate.digest_file(wrapper),
                       'mojolearn/_version.py': gate.digest_file(version_file)},
        runtime_sha256={'mojolearn/.libs/libfixture.so': hashlib.sha256(b'inert runtime').hexdigest()},
        optional_native={'_mojolearn_byte_lm': {'included': True, 'supported_modes': ['identical'], 'unsupported_modes': ['fast', 'deterministic']}})
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
    for key in sorted(ARCHES or gate.RELEASE_ARCHES):
        vendor, arch = key.split('/')
        out = qualification_root / key
        out.mkdir(parents=True)
        audit = gate.release_audit(wheel, root, proof_root, key)
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
                bindings = {binding: dict(path=package + '/' + prefix + binding + '.so',
                    sha256=audit['extension_hashes'][prefix + binding + '.so'], mode_code=code)
                    for binding in gate.surface.expected_bindings(mode, True)}
                write_json(out / (name + '.installed.json'), dict(vendor=vendor, mode=mode,
                    wheel_sha256=audit['sha256'], package=package + '/__init__.py',
                    installed_bindings=bindings, device_architecture=arch,
                    selected_architecture=arch, architecture_probe='synthetic fixture witness',
                    architecture_override_absent=True))
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
                        b = row['installed_bindings']['_mojolearn_mamba']
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
                        self.assertEqual(result['jobs_per_runtime_architecture'], 25)
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
        self.assertEqual(result['jobs_per_runtime_architecture'], 25)

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


if __name__ == '__main__':
    unittest.main()
