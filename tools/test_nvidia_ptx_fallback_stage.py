"""Fallback stage: pre-rental refusals, mid-rental admission and pack, pod body."""
import copy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import zipfile

import pytest

import nvidia_baseline_qualification as q
import nvidia_ptx_fallback_stage as stage

SHA = 'a' * 40
GPU = 'NVIDIA A100 80GB PCIe'
PACK = ['--set', 'sets/cuda', '--profile', 'split']


def config(name, cap):
    return dict(device_name=name, compute_capability=cap, driver_version='580.159.04', cuda_driver_version=13000)


def record():
    scope = dict(fixtures=[], parts=[], comparison_sha256='0' * 64)
    return dict(source_commit=SHA, manifest_sha256='1' * 64,
                configurations=[config('RTX 4090', [8, 9]), config('H100', [9, 0]), config(GPU, [8, 0])],
                coverage=dict(shared=dict(scope), nvidia=dict(scope),
                              native_absent=dict(configurations=[config(GPU, [8, 0])])))


@pytest.fixture
def inputs(tmp_path):
    wheels = tmp_path / 'wheels'
    wheels.mkdir()
    members = {'mojolearn': {'mojolearn/ptx_admission.py': b'', 'mojolearn/identity_columns/COMMIT': SHA.encode()},
               'mojolearn_nvidia': {'mojolearn/cuda_native/sm_89/identical/_mojolearn.so': b'native'},
               'mojolearn_amd': {'mojolearn/hip_native/gfx942/identical/_mojolearn.so': b'amd'},
               'mojolearn_nvidia_ptx80': {stage.MANIFEST_MEMBER: b'{"manifest": 1}',
                                          'mojolearn/cuda_ptx/sm_80/identical/_mojolearn.so': b'ptx'}}
    for name, files in members.items():
        with zipfile.ZipFile(wheels / f'{name}-0.8.37-py3-none-manylinux_2_35_x86_64.whl', 'w') as z:
            for member, data in files.items():
                z.writestr(member, data)
    receipts, witnesses = [], []
    for index, (role, cap) in enumerate([('native-reference', [8, 9]), ('baseline', [8, 9]),
                                         ('native-reference', [9, 0]), ('baseline', [9, 0])]):
        column = tmp_path / f'r{index}.column.json'
        column.write_text('{}')
        path = tmp_path / f'r{index}.json'
        uuid = 'GPU-' + ''.join(map(str, cap))
        path.write_text(json.dumps(dict(schema=q.SCHEMA, source_commit=SHA, role=role,
                                        hardware=dict(uuid=uuid, compute_capability=cap),
                                        column_file=column.name, column_sha256=q.sha(column))))
        receipts.append(path)
        if role == 'baseline':
            witness = tmp_path / f'w{index}.json'
            witness.write_text(json.dumps(dict(device=dict(uuid=uuid))))
            witnesses.append(witness)
    references = {}
    for vendor in ('apple', 'amd'):
        references[vendor] = tmp_path / (vendor + '.json')
        references[vendor].write_text(json.dumps(dict(commit=SHA, vendor=vendor)))
    return SimpleNamespace(root=tmp_path, wheels=sorted(wheels.glob('*.whl')), receipts=receipts,
                           witnesses=witnesses, references=references)


def make_plan(i, **changes):
    args = dict(commit=SHA, wheels=i.wheels, gpu_name=GPU, receipts=i.receipts, witnesses=i.witnesses,
                apple=i.references['apple'], amd=i.references['amd'],
                apple_sha256=q.sha(i.references['apple']), amd_sha256=q.sha(i.references['amd']), pack_argv=list(PACK))
    args.update(changes)
    return stage.plan(**args)


def test_plan_pins_every_retained_input(inputs):
    document = make_plan(inputs)
    assert document['core_wheel'].startswith('mojolearn-0.8.37') and document['body_seconds'] == 2200
    assert [row['sha256'] for row in document['receipts']] == [q.sha(p) for p in inputs.receipts]
    assert document['witness_script']['sha256'] == q.sha(stage.WITNESS)
    assert document['negative_cases'] == ['negative-config', 'negative-unbundled']


def test_frozen_core_without_the_fallback_loader_refuses_before_rental(inputs):
    core = next(w for w in inputs.wheels if w.name.startswith('mojolearn-'))
    with zipfile.ZipFile(core, 'w') as z:
        z.writestr('mojolearn/identity_columns/COMMIT', SHA)
    with pytest.raises(ValueError, match='no admitted-fallback loader'):
        make_plan(inputs)


@pytest.mark.parametrize('changes, message', [
    (dict(receipts=[]), 'receipts are required'),
    (dict(commit='b' * 40), 'source or schema differs'),
    (dict(witnesses=[]), 'One configuration witness'),
    (dict(apple_sha256='0' * 64), 'Pinned reference hash differs: apple'),
    (dict(amd=None), 'Pinned reference hash differs: amd'),
    (dict(pack_argv=['--profile', 'split']), 'naming --set'),
    (dict(pack_argv=PACK + ['--wheels', 'nvidia-ptx80']), 'must not set'),
    (dict(pack_argv=PACK + ['--out=/elsewhere']), 'must not set'),
])
def test_plan_refusals(inputs, changes, message):
    with pytest.raises(ValueError, match=message):
        make_plan(inputs, **changes)


def test_plan_needs_two_paired_native_capabilities_and_no_absent_receipts(inputs):
    with pytest.raises(ValueError, match='two natively supported capabilities'):
        make_plan(inputs, receipts=inputs.receipts[:3], witnesses=inputs.witnesses[:1])
    alien = inputs.root / 'alien.json'
    doc = q.read(inputs.receipts[1])
    doc['hardware']['compute_capability'] = [8, 0]
    alien.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='natively supported devices'):
        make_plan(inputs, receipts=inputs.receipts + [alien])


def test_negative_admission_drops_only_the_native_absent_device():
    negative = stage.negative_admission(record())
    assert [row['compute_capability'] for row in negative['configurations']] == [[8, 9], [9, 0]]
    assert 'native_absent' not in negative['coverage']
    only = record()
    only['configurations'] = only['configurations'][2:]
    with pytest.raises(ValueError, match='Negative admission'):
        stage.negative_admission(only)


def test_body_is_unforced_bounded_and_asserts_all_three_cases():
    document = dict(source_commit=SHA, core_wheel='mojolearn-0.8.37-py3-none-any.whl',
                    vendor_wheel='mojolearn_nvidia-0.8.37-py3-none-any.whl',
                    plain_vendor_wheel='mojolearn_nvidia-0.8.37-py3-none-any.whl',
                    negative_cases=['negative-config', 'negative-unbundled'])
    body = stage.body(document)
    subprocess.run(['bash', '-n'], input=body, text=True, check=True)
    assert 'unset PYTHONPATH MOJOLEARN_CUDA_PATH MOJOLEARN_EXPERIMENTAL_PTX' in body
    assert 'MOJOLEARN_CUDA_PATH=' not in body and 'MOJOLEARN_EXPERIMENTAL_PTX=' not in body
    assert body.index('e2e.py" select') < body.index('mojolearn._identity_break') < body.index('--diff reference.json') \
        < body.index('e2e.py" column') < body.index('e2e.py" refuse') < body.index('FALLBACK_E2E_PASSED')
    assert '--fixtures base,denormal,odd --fail-on-refused --require-backend cuda --no-batch --no-rlpair' in body
    assert '--require-columns 2 --lanes "$LANES"' in body and '--seconds 900 --rss-gib 12 --cores 2' in body
    assert 'install negative-unbundled "wheels/mojolearn_nvidia-0.8.37-py3-none-any.whl"' in body
    assert 'install negative-config "fallback/wheels-negative-config/' in body
    assert 'for case in negative-config negative-unbundled; do' in body
    assert '@' not in body.replace('${', '').replace('"$@"', '')
    # Step bounds: three installs, select, lane selection, column, two refusals.
    assert 3 * (60 + 180) + 140 + 120 + 920 + 2 * 140 <= stage.BODY_SECONDS


def prepared(inputs, tmp_path, *, tamper=None, pack_rc=0, absent=True):
    document = make_plan(inputs)
    results = tmp_path / 'remote/results'
    results.mkdir(parents=True)
    (results / 'PTX_BASELINE.json').write_bytes(b'{"manifest": 1}')
    for name in ('full-baseline.json', 'cuda-runtime-config-ptx.json'):
        (results / name).write_text('{}')
    calls = SimpleNamespace(build=[], pack=[])

    def build(manifest, receipts, references, hashes, witnesses, script):
        calls.build.append(dict(manifest=manifest, receipts=receipts, witnesses=witnesses, hashes=hashes))
        doc = record()
        if not absent:
            del doc['coverage']['native_absent']
        return doc, {'nvidia-comparison.json': dict(ok=True)}

    def run(command, **kwargs):
        calls.pack.append(command)
        if pack_rc:
            return subprocess.CompletedProcess(command, pack_rc)
        out = Path(command[command.index('--out') + 1])
        out.mkdir(parents=True)
        admission = Path(command[command.index('--bundle-ptx-admission') + 1]).read_bytes()
        with zipfile.ZipFile(out / 'mojolearn_nvidia-0.8.37-py3-none-manylinux_2_35_x86_64.whl', 'w') as z:
            z.writestr(stage.MANIFEST_MEMBER, b'{"manifest": 1}')
            z.writestr(stage.ADMISSION_MEMBER, tamper or admission)
        return subprocess.CompletedProcess(command, 0)

    return document, results, calls, build, run


def test_prepare_admits_packs_twice_and_stages_last(inputs, tmp_path, monkeypatch):
    import admit_nvidia_ptx as admit
    monkeypatch.setattr(admit, 'admission_api', lambda: SimpleNamespace(
        ADMISSION_FILE='PTX_IDENTITY_ADMISSION.json', validate_admission=lambda *a, **k: None))
    document, results, calls, build, run = prepared(inputs, tmp_path)
    out = tmp_path / 'fallback-stage'
    staged = stage.prepare(document, results, out, build=build, run=run)
    # This rental's receipt and witness join the retained native-device evidence.
    assert calls.build[0]['receipts'][-1] == results / 'full-baseline.json' and len(calls.build[0]['receipts']) == 5
    assert calls.build[0]['witnesses'][-1] == results / 'cuda-runtime-config-ptx.json'
    assert calls.build[0]['manifest'].parent.joinpath('identical/_mojolearn.so').read_bytes() == b'ptx'
    assert len(calls.pack) == 2
    for command in calls.pack:
        assert command[2:6] == PACK and command[command.index('--wheels') + 1] == 'nvidia'
    work = tmp_path / 'fallback-work'
    positive = json.loads((work / 'admission-positive/PTX_IDENTITY_ADMISSION.json').read_text())
    negative = json.loads((work / 'admission-negative-config/PTX_IDENTITY_ADMISSION.json').read_text())
    assert len(positive['configurations']) == 3 and len(negative['configurations']) == 2
    assert (work / 'admission-positive/nvidia-comparison.json').is_file()
    expect = json.loads((out / 'expect.json').read_text())
    assert expect['configuration']['device_name'] == GPU
    assert expect['admission_sha256'] == staged['admission_sha256'] == q.sha(work / 'admission-positive/PTX_IDENTITY_ADMISSION.json')
    assert staged['admission_sha256'] != staged['negative_admission_sha256']
    assert staged['vendor_wheel_sha256']['positive'] != staged['vendor_wheel_sha256']['negative-config']
    listed = {line.split('  ')[1] for line in (out / 'SHA256SUMS').read_text().splitlines()}
    wheel = 'mojolearn_nvidia-0.8.37-py3-none-manylinux_2_35_x86_64.whl'
    assert listed == {'body.sh', 'e2e.py', 'expect.json', 'plan.json', 'reference.json',
                      'wheels-positive/' + wheel, 'wheels-negative-config/' + wheel}
    for line in (out / 'SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ')
        assert q.sha(out / name) == digest
    assert (out / 'reference.json').read_bytes() == inputs.references['apple'].read_bytes()
    assert (out / 'e2e.py').read_bytes() == stage.E2E.read_bytes()
    with pytest.raises(ValueError, match='overwrite'):
        stage.prepare(document, results, out, build=build, run=run)


@pytest.mark.parametrize('kwargs, message', [
    (dict(tamper=b'{}'), 'does not carry this admission'),
    (dict(pack_rc=1), 'Packing the positive vendor wheel failed'),
    (dict(absent=False), 'exactly this native-absent device'),
])
def test_prepare_failures_stage_nothing(inputs, tmp_path, monkeypatch, kwargs, message):
    import admit_nvidia_ptx as admit
    monkeypatch.setattr(admit, 'admission_api', lambda: SimpleNamespace(
        ADMISSION_FILE='PTX_IDENTITY_ADMISSION.json', validate_admission=lambda *a, **k: None))
    document, results, calls, build, run = prepared(inputs, tmp_path, **kwargs)
    out = tmp_path / 'fallback-stage'
    with pytest.raises(ValueError, match=message):
        stage.prepare(document, results, out, build=build, run=run)
    assert not out.exists()


def test_prepare_refuses_changed_inputs_and_another_manifest(inputs, tmp_path):
    document, results, calls, build, run = prepared(inputs, tmp_path)
    (results / 'PTX_BASELINE.json').write_bytes(b'{"manifest": 2}')
    with pytest.raises(ValueError, match='another PTX manifest'):
        stage.prepare(document, results, tmp_path / 'fallback-stage', build=build, run=run)
    document, results2, calls, build, run = prepared(inputs, tmp_path / 'second')
    inputs.receipts[0].write_text('{"changed": true}')
    with pytest.raises(ValueError, match='changed since the plan'):
        stage.prepare(document, results2, tmp_path / 'second/fallback-stage', build=build, run=run)
    assert not calls.build
