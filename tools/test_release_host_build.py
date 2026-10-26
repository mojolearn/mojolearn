"""Host-only provenance controls; no compiler or native module is executed."""
import copy
import json
import shutil

import pytest

import release_host_build as hb
import release_host_composition as hc
import retained_release_composition as rc
from test_retained_release_composition import fixture, OLD, NEW, COMPILER
import binding_stamps
import generated_sources as gs

REAL = json.loads((hb.Path(__file__).resolve().parent / 'test_release_host_build_tokenizer_0857.json').read_text())


def host_only(fixture, tmp_path, generated=False):
    source, old, _, before, current = fixture
    # The older fixture did not need complete source stamp fields; these inputs
    # describe all host closures independently of any device output.
    identities = {k:v for k,v in current['binding_identities'].items() if '/host/' in k}
    for identity in identities.values():
        identity['closure']['scope'] = 'closure'
        identity['script'] = binding_stamps.member_script(identity['name'])
    out = tmp_path / 'independent-host'
    staged = out / 'set'
    (staged / 'host').mkdir(parents=True)
    (out / 'raw').mkdir()
    paths = hb.host_paths()
    for f, path in paths.items():
        (out / 'raw' / f).write_bytes(b'raw-new-host-' + f.encode())
        (staged / path).write_bytes(b'staged-new-host-' + f.encode())
    donor, donor_set = hc.inspect_build(old[0])
    donor_contract, _ = hc._record(donor, COMPILER)
    shutil.copytree(donor_set / '.libs', staged / '.libs')
    index = dict(format=1,files=hc.host_plan())
    hb.write(staged / hb.HOST_INDEX,index)
    inventory = current['source_inventory']
    closures, generated_sources = {}, {}
    if generated:
        # The real 0.8.57 tokenizer host member: committed identity, stamp row,
        # closure rows, generated table and its tracked generator inputs.
        ident = next(i for i in identities.values() if i['name'] == REAL['member'])
        ident.update(closure=dict(REAL['identity']['closure']), generators=REAL['identity']['generators'])
        tracked = dict(REAL['inventory'])
        inventory[:] = [r for r in inventory if r[0] not in tracked] + [list(r) for r in REAL['inventory']]
        closures[REAL['member']] = REAL['closure_rows']
        for rel, text in REAL['generated'].items():
            (out / 'generated' / rel).parent.mkdir(parents=True, exist_ok=True)
            (out / 'generated' / rel).write_text(text)
            for name in gs.GENERATED[rel]:
                (out / 'generated-inputs' / name).parent.mkdir(parents=True, exist_ok=True)
                (out / 'generated-inputs' / name).write_text(REAL['generator_inputs'][name])
            generated_sources[rel] = dict(sha256=hc.file_sha(out / 'generated' / rel),
                inputs={name: hc.file_sha(out / 'generated-inputs' / name) for name in gs.GENERATED[rel]})
    for name in binding_stamps.MERGED_INPUTS:
        if name not in dict(inventory):
            inventory.append([name, 'e' * 64])
    merged_inputs = {name:dict(inventory)[name] for name in binding_stamps.MERGED_INPUTS}
    before['merged_build_inputs'] = current['merged_build_inputs'] = merged_inputs
    hb.write(out / 'source-inventory.json',inventory)
    hb.write(out / 'runtime-source.json',donor)
    capture = dict(schema=hb.CAPTURE_SCHEMA,phase='before-build',source_commit=NEW,
        source_inventory_sha256=hc.file_sha(out / 'source-inventory.json'),compiler_sha256=COMPILER,
        installed_packages=donor_contract['installed_packages'],toolchain=donor_contract['toolchain'],
        binding_identities=identities,runtime_identity=current['runtime_identity'],
        merged_build_inputs=merged_inputs,
        build_environment=hb.FLAGS,pixi_environment='gbmbench',command=hb.command('gbmbench'))
    hb.write(out / 'compile-inputs.json',capture)
    by_name = {i['name']: i for i in identities.values()}
    merged_hash = hc.hashlib.sha256()
    for name in binding_stamps.MERGED_INPUTS:
        merged_hash.update(name.encode()+b'\0'+merged_inputs[name].encode()+b'\n')
    stamps = {}
    for f, members in index['files'].items():
        stamps[f] = dict(schema=2,binding=paths[f],script='merged_build.sh',commit=NEW,
            members={n:dict(script=by_name[n]['script'],scope='closure',digest=by_name[n]['closure']['digest'])
                     for n in members}, merged_inputs=merged_hash.hexdigest())
        if generated and REAL['member'] in members:
            stamps[f]['members'][REAL['member']] = dict(REAL['stamp_member'])
    hb.write(out / 'binding-stamp.json',stamps)
    stage = dict(schema='mojolearn.host-staging.v1',action='stage-host-only',
        runtime_origin_proof_sha256=donor['proof']['sha256'],runtime_libraries=donor_contract['runtime_libraries'],
        runtime_byte_changes=[],raw_sha256={f: hc.file_sha(out / 'raw' / f) for f in paths},
        staged_sha256={f: hc.file_sha(staged / p) for f, p in paths.items()},runpath={f: hb.RUNPATH for f in paths},
        rpath={f: None for f in paths},needed={f: ['libRuntime.so'] for f in paths})
    hb.write(out / 'stage.json',stage)
    hb.write(out / 'readback.json',dict(schema='mojolearn.host-readback.v1',source_commit=NEW,
        members=[dict(name=name,vendor='cpu',column='cpu',numeric_mode=1) for name in sorted(hc.expected_names())]))
    proof = dict(schema=hb.SCHEMA,action='build-host-only',complete=True,build_exit=0,stamp_exit=0,
        stage_exit=0,readback_exit=0,source_commit=NEW,compiler_sha256=COMPILER,binding_layout='merged',
        release_modes=['identical'],compile_inputs_sha256=hc.file_sha(out / 'compile-inputs.json'),
        source_inventory=inventory,source_sha256=hb.inventory_sha(inventory),command=hb.command('gbmbench'),
        raw_host_sha256=stage['raw_sha256'],host_extension={hb.PREFIX+p:stage['staged_sha256'][f] for f, p in paths.items()},
        merged_indexes={hb.PREFIX+hb.HOST_INDEX:dict(sha256=hc.file_sha(staged / hb.HOST_INDEX),files=index['files'])},
        stamp_sha256=hc.file_sha(out / 'binding-stamp.json'),stage_sha256=hc.file_sha(out / 'stage.json'),
        readback_sha256=hc.file_sha(out / 'readback.json'),runtime_identity=current['runtime_identity'])
    if generated:
        proof.update(generated_sources=generated_sources, generated_closures=closures)
    hb.write(out / 'host-build-proof.json',proof)
    return out


def test_actual_host_only_proof_composes_without_gpu_host_producer(fixture, tmp_path):
    host = host_only(fixture,tmp_path)
    record, directory = hb.inspect(host)
    assert hb.validate_bytes(record,directory,COMPILER)
    proof = hc.decoded(record['proof'])
    assert 'extensions' not in proof and 'vendor' not in hc.decoded(record['capture'])
    assert hb.command('gbmbench')[-3:] == ['bash','bindings/merged_build.sh','host']
    source, old, _, _, _ = fixture
    result = rc.prepare(old[0],host,source,COMPILER,tmp_path/'assembled')
    assert rc.validate(result,source) and rc.validate_bytes(result,tmp_path/'assembled/set')
    assert result['original_host']['kind'] == hb.KIND
    assert result['original_device']['leg'] == 'cuda/sm_89'
    assert {r['source_commit'] for r in result['artifact_origins'].values()} == {OLD,NEW}
    for p in hb.host_paths().values():
        assert result['host_extension']['mojolearn/cuda/sm_89/'+p] == proof['host_extension'][hb.PREFIX+p]
    assert not (directory/'identical').exists()
    assert 'build_exit' not in result


def change_witness(source,key,operation):
    doc = hc.decoded(source[key]);operation(doc)
    raw=json.dumps(doc)
    source[key]={'utf8':raw,'sha256':hc.hashlib.sha256(raw.encode()).hexdigest()}
    return doc


@pytest.mark.parametrize('changed', ['failed-build','boolean-build-exit','failed-readback','fake-gpu-output',
    'compiler','flags','gpu-command','host-closure','host-index','readback-coverage','readback-column',
    'runtime-bytes','runtime-producer','stage-runpath','stamp-file','host-index-move'])
def test_host_only_proof_cannot_claim_missing_or_different_work(fixture,tmp_path,changed):
    host=host_only(fixture,tmp_path);source,_=hb.inspect(host)
    proof=hc.decoded(source['proof'])
    target='proof'
    if changed=='failed-build': proof['build_exit']=1
    elif changed=='boolean-build-exit': proof['build_exit']=False
    elif changed=='failed-readback': proof['readback_exit']=1
    elif changed=='fake-gpu-output': proof['extensions']={}
    elif changed in ('compiler','flags','gpu-command','host-closure'):
        def mutate(d):
            if changed=='compiler': d['compiler_sha256']='f'*64
            elif changed=='flags': d['build_environment']['MOJOLEARN_GPU_ARCHS']='sm_89'
            elif changed=='gpu-command': d['command'][-1]='gpu'
            else: next(iter(d['binding_identities'].values()))['closure']['digest']='f'*64
        change_witness(source,'capture',mutate)
        proof['compile_inputs_sha256']=source['capture']['sha256']
    elif changed in ('host-index','host-index-move'):
        files=hc.host_files()
        def mutate(d):
            moved=d['files'][files[0]].pop()
            if changed=='host-index-move': d['files'][files[1]].append(moved)
        change_witness(source,'index',mutate)
        proof['merged_indexes'][hb.PREFIX+hb.HOST_INDEX]['sha256']=source['index']['sha256']
    elif changed=='stamp-file':
        change_witness(source,'stamp',lambda d:d.pop(hc.host_files()[-1]))
        proof['stamp_sha256']=source['stamp']['sha256']
    elif changed.startswith('readback-'):
        def mutate(d):
            if changed=='readback-coverage': d['members'].pop()
            else: d['members'][0]['column']='nvidia'
        change_witness(source,'readback',mutate);proof['readback_sha256']=source['readback']['sha256']
    else:
        def mutate(d):
            if changed=='runtime-bytes': d['runtime_libraries']['libRuntime.so']='f'*64
            elif changed=='runtime-producer': d['runtime_origin_proof_sha256']='f'*64
            else: d['runpath'][hc.host_files()[-1]]='$ORIGIN/build-box-only'
        change_witness(source,'stage',mutate);proof['stage_sha256']=source['stage']['sha256']
    raw=json.dumps(proof);source['proof']={'utf8':raw,'sha256':hc.hashlib.sha256(raw.encode()).hexdigest()}
    with pytest.raises(ValueError): hb.record(source,COMPILER)


def test_missing_original_compiler_bytes_and_changed_staged_host_are_refused(fixture,tmp_path):
    host=host_only(fixture,tmp_path);source,staged=hb.inspect(host)
    (staged/hb.host_paths()[hc.host_files()[-1]]).write_bytes(b'changed')
    with pytest.raises(ValueError,match='staged host bytes'): hb.validate_bytes(source,staged,COMPILER)
    (host/'raw'/hc.host_files()[0]).unlink()
    with pytest.raises(ValueError,match='original compiler output bytes'): hb.inspect(host)


def test_real_0857_tokenizer_stamp_is_the_identity_plus_the_verified_generated_table():
    """The droplet refusal: stamp 36ef4c6a.. vs identity f6b98a9e.. for the tokenizer host."""
    ident, row, rows = REAL['identity'], REAL['stamp_member'], REAL['closure_rows']
    assert row['digest'] != ident['closure']['digest']  # the old equality refused this build
    assert gs.closure_hash(rows) == row['digest']
    assert gs.closure_hash([r for r in rows if r[0] not in gs.GENERATED]) == ident['closure']['digest']
    data = REAL['generated'][gs.UNICODE_TABLE].encode()
    inputs = {k: v.encode() for k, v in REAL['generator_inputs'].items()}
    sha = gs.verify(gs.UNICODE_TABLE, data, inputs)
    assert [r[1] for r in rows if r[0] == gs.UNICODE_TABLE] == [sha]
    assert ident['generators'] == hb.generator_dir_digest(REAL['inventory'], 'tokenizer/tools')
    hb.check_member_stamp(REAL['member'], row, ident, {REAL['member']: rows}, {gs.UNICODE_TABLE: sha}, REAL['inventory'])
    with pytest.raises(ValueError, match='stamp source closure differs'):
        hb.check_member_stamp(REAL['member'], row, ident, {}, {}, REAL['inventory'])


@pytest.mark.parametrize('edit', [b'L\\t0041', b'unicodedata 16.0.0', b'GENERATED at build time'])
def test_generated_table_must_be_what_the_tracked_generator_writes(edit):
    data = REAL['generated'][gs.UNICODE_TABLE].encode()
    inputs = {k: v.encode() for k, v in REAL['generator_inputs'].items()}
    assert edit in data
    with pytest.raises(ValueError):
        gs.verify(gs.UNICODE_TABLE, data.replace(edit, edit[:-1] + b'X', 1), inputs)
    pin = inputs[gs.UNICODE_PIN].replace(b'UNICODE_TABLE_SHA256_PINNED = "a', b'UNICODE_TABLE_SHA256_PINNED = "b', 1)
    assert pin != inputs[gs.UNICODE_PIN]
    with pytest.raises(ValueError):
        gs.verify(gs.UNICODE_TABLE, data, dict(inputs, **{gs.UNICODE_PIN: pin}))


def test_host_proof_with_generated_tokenizer_table_composes(fixture, tmp_path):
    host = host_only(fixture, tmp_path, generated=True)
    record, directory = hb.inspect(host)
    assert hb.validate_bytes(record, directory, COMPILER)
    contract, _, _, _ = hb.record(record, COMPILER)
    assert set(contract['generated_sources']) == {gs.UNICODE_TABLE}
    source, old, _, _, _ = fixture
    result = rc.prepare(old[0], host, source, COMPILER, tmp_path / 'assembled')
    assert rc.validate(result, source)


REFUSALS = [('no-closure-rows', 'stamp source closure differs'), ('table-bytes', 'not the pinned sha256'),
            ('generator-input', 'retained generator input differs'),
            ('inventory-generator', 'not the tracked generator'),
            ('committed-row', 'closure differs (with generated sources)'),
            ('identity-generators', 'identity generator inputs differ'),
            ('extra-member', 'closure differs (with generated sources)'), ('missing-table', '')]


@pytest.mark.parametrize('changed', [c for c, _ in REFUSALS])
def test_generated_tokenizer_table_cannot_hide_other_source_changes(fixture, tmp_path, changed):
    host = host_only(fixture, tmp_path, generated=True)
    proof_path = host / 'host-build-proof.json'
    proof = json.loads(proof_path.read_text())
    table = host / 'generated' / gs.UNICODE_TABLE
    if changed == 'no-closure-rows':
        proof['generated_closures'] = {}
        proof['generated_sources'] = {}
        shutil.rmtree(host / 'generated'); shutil.rmtree(host / 'generated-inputs')
    elif changed == 'table-bytes':
        table.write_text(table.read_text().replace('L\\t0041', 'L\\t0042', 1))
        proof['generated_sources'][gs.UNICODE_TABLE]['sha256'] = hc.file_sha(table)
    elif changed == 'generator-input':
        f = host / 'generated-inputs' / gs.UNICODE_GENERATOR
        f.write_text(f.read_text() + '\n')
    elif changed == 'inventory-generator':
        proof['generated_sources'][gs.UNICODE_TABLE]['inputs'][gs.UNICODE_GENERATOR] = 'f' * 64
    elif changed == 'committed-row':
        rows = proof['generated_closures'][REAL['member']]
        rows[0][1] = 'f' * 64
    elif changed == 'identity-generators':
        capture_path = host / 'compile-inputs.json'
        capture = json.loads(capture_path.read_text())
        next(i for i in capture['binding_identities'].values() if i['name'] == REAL['member'])['generators'] = 'f' * 64
        hb.write(capture_path, capture)
        proof['compile_inputs_sha256'] = hc.file_sha(capture_path)
    elif changed == 'extra-member':
        other = sorted(hc.expected_names() - {REAL['member']})[0]
        proof['generated_closures'][other] = REAL['closure_rows']
    else:
        table.unlink()
    hb.write(proof_path, proof)
    if changed == 'missing-table':
        with pytest.raises(FileNotFoundError):
            hb.inspect(host)
        return
    source, _ = hb.inspect(host)
    with pytest.raises(ValueError) as caught:
        hb.record(source, COMPILER)
    assert dict(REFUSALS)[changed] in str(caught.value)


def incremental_host(fixture, tmp_path):
    directory = host_only(fixture, tmp_path)
    _, old, _, before, current = fixture
    donor, donor_dir = hc.inspect_build(old[0])
    contract, variant = hc._record(donor, COMPILER)
    # The test fixture's host stamps add fields the original full proof omitted.
    # Align only those descriptive fields; the changed tokenizer closure remains.
    for key, ident in contract['host_identities'].items():
        closure = current['binding_identities'][key]['closure']['digest']
        current['binding_identities'][key] = copy.deepcopy(ident)
        current['binding_identities'][key]['closure'].update(scope='closure', digest=closure)
        current['binding_identities'][key]['script'] = binding_stamps.member_script(ident['name'])
        before['binding_identities'][key]['script'] = binding_stamps.member_script(ident['name'])
        before['binding_identities'][key]['closure']['scope'] = 'closure'
    # Original synthetic build permits tree scope; make its captured host inputs
    # describe the same exact closures needed for incremental admission.
    for original in old:
        from test_retained_release_composition import update
        update(original/'compile-inputs.json', lambda d: [(v['closure'].update(scope='closure'), v.update(script=binding_stamps.member_script(v['name'])))
            for k,v in d['binding_identities'].items() if '/host/' in k])
        update(original/'build-provenance.json', lambda d:d.update(compile_inputs_sha256=hc.file_sha(original/'compile-inputs.json')))
    donor, donor_dir = hc.inspect_build(old[0])
    contract, variant = hc._record(donor, COMPILER)
    hb.write(directory/'runtime-source.json', donor)
    capture = json.loads((directory/'compile-inputs.json').read_text())
    capture['binding_identities'] = {k:v for k,v in current['binding_identities'].items() if '/host/' in k}
    plan = hb.incremental_plan(before, current, contract, 'gbmbench')
    proof = json.loads((directory/'host-build-proof.json').read_text())
    stage = json.loads((directory/'stage.json').read_text())
    stage['runtime_origin_proof_sha256'] = donor['proof']['sha256']
    stamps = json.loads((directory/'binding-stamp.json').read_text())
    for filename, row in plan.items():
        if row['action'] == 'retain':
            shutil.copy2(donor_dir/'host'/filename, directory/'raw'/filename)
            stage['raw_sha256'][filename] = variant['host_sha256'][filename]
            stamps.pop(filename)
    commands = [row['command'] for row in plan.values() if row['action'] == 'compile']
    capture.update(group_plan=plan, command=commands, retained_inputs=before, admitted_inputs=current)
    hb.write(directory/'compile-inputs.json', capture)
    hb.write(directory/'binding-stamp.json', stamps)
    hb.write(directory/'stage.json', stage)
    proof.update(schema=hb.INCREMENTAL_SCHEMA, command=commands, group_plan=plan,
        group_origins={f:dict(action=row['action'], source_commit=OLD if row['action']=='retain' else NEW,
            original_proof_sha256=donor['proof']['sha256'] if row['action']=='retain' else None,
            input_stage='retained-staged' if row['action']=='retain' else 'compiler-output') for f,row in plan.items()},
        compile_inputs_sha256=hc.file_sha(directory/'compile-inputs.json'),
        stamp_sha256=hc.file_sha(directory/'binding-stamp.json'), stage_sha256=hc.file_sha(directory/'stage.json'),
        raw_host_sha256=stage['raw_sha256'])
    hb.write(directory/'host-build-proof.json', proof)
    return directory, plan


def test_incremental_host_rebuilds_changed_group_preserving_compilation_origins(fixture, tmp_path):
    directory, plan = incremental_host(fixture, tmp_path)
    assert [r['action'] for r in plan.values()].count('compile') == 1
    assert [r['action'] for r in plan.values()].count('retain') == 4
    source, staged = hb.inspect(directory)
    contract, _, proof, capture = hb.record(source, COMPILER)
    assert hb.validate_bytes(source, staged, COMPILER)
    assert len(capture['command']) == 1 and capture['command'][0][-2] == '--group'
    assert {r['source_commit'] for r in contract['group_origins'].values()} == {OLD, NEW}
    original_source, builds, *_ = fixture
    composition = rc.prepare(builds[0], directory, original_source, COMPILER, tmp_path/'composed-incremental')
    origins = [r for path,r in composition['artifact_origins'].items() if '/host/' in path]
    assert sum(r['source_commit'] == OLD for r in origins) == 4
    assert all('staging_proof_sha256' in r for r in origins)


@pytest.mark.parametrize('tamper', ['changed-retained-bytes', 'invented-origin', 'skip-changed-group', 'wrong-command'])
def test_incremental_host_rejects_false_reuse(fixture, tmp_path, tamper):
    directory, plan = incremental_host(fixture, tmp_path)
    source, _ = hb.inspect(directory)
    retained = next(f for f,r in plan.items() if r['action'] == 'retain')
    if tamper == 'changed-retained-bytes':
        change_witness(source, 'proof', lambda d:d['raw_host_sha256'].update({retained:'f'*64}))
    elif tamper == 'invented-origin':
        change_witness(source, 'proof', lambda d:d['group_origins'][retained].update(source_commit=NEW))
    elif tamper == 'skip-changed-group':
        changed = next(f for f,r in plan.items() if r['action'] == 'compile')
        change_witness(source, 'proof', lambda d:d['group_plan'][changed].update(action='retain'))
    else:
        change_witness(source, 'proof', lambda d:d.update(command=[]))
    with pytest.raises(ValueError):
        hb.record(source, COMPILER)
