#!/usr/bin/env python3
"""Prepare only: inspect retained local evidence and emit a fresh 29-recipe plan.

No subprocess launches of measurements, SSH, builds, rental or teardown actions.
Opponent records retain their original keys/evidence; preflight decides actual reuse.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
import bench_board as B
import bench_board_store as Store

CAMPAIGNS = {'amd': 'full-measurement-amd-20261027-r3',
             'nvidia': 'full-measurement-nvidia-20261027'}
SWITCHES = {'MOJOLEARN_NEURAL_NATIVE_DISPATCH': '1', 'MOJOLEARN_GENERATION_NATIVE_SESSION': '1'}


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(evidence, out, plan_dir):
    out.mkdir(parents=True, exist_ok=False)
    manifest, recipes, audit, configs = {}, {}, {}, {}
    source_boards = []
    for vendor, old_campaign in CAMPAIGNS.items():
        old = evidence / old_campaign
        old_cfg = json.loads((old / 'config.json').read_text())
        old_vendor = old_cfg['vendors'][0]
        plan = sorted(B.plan_races(vendor, ['identical'], families=['neural'], neural_shape='full'), key=lambda r: r['id'])
        assert len(plan) == 29 and len({r['id'] for r in plan}) == 29
        manifest[vendor] = [r['id'] for r in plan]
        by_lane = {r['lane']: r for r in plan}
        recipes[vendor] = []
        for race in plan:
            settings = B.race_settings(dict(rounds=1, vendor=vendor), race)
            recipes[vendor].append(dict(race, dimensions=B.NEURAL.shape_record(race['lane'], 'full'),
                                        settings=settings, settings_sha256=Store.sha256_json(settings)))
        raw_store = out / (vendor + '-retained-raw.jsonl')
        raw_store.touch()
        boards = sorted((old / vendor / 'boards').glob('neural.*.json'))
        if len(boards) != 29:
            raise ValueError(f'{vendor}: expected 29 retained recipe boards, found {len(boards)}')
        board_by_lane = {}
        for board in boards:
            source_boards.append(dict(vendor=vendor, path=str(board), sha256=sha(board)))
            document = json.loads(board.read_text())
            for race in document.get('races', {}).values():
                if race.get('family') == 'neural':
                    board_by_lane[race['lane']] = dict(path=str(board),
                        original_own_hashes={c['arm']: c.get('hash') for c in race.get('cells', []) if c.get('library') == 'mojolearn'},
                        old_pairing=(race.get('preflight') or {}).get('verdict'))
            B.backfill_evidenced_store(str(board), str(raw_store))
        accepted, rejected = [], []
        seed = out / (vendor + '-opponent-store.jsonl')
        seed.touch()
        for key_id, record in Store.load(raw_store).items():
            key, cell = record['key'], record['cell']
            race = by_lane.get(key.get('lane'))
            reasons = []
            if not race or key['arm'] not in race['opponents']:
                reasons.append('not an opponent selected by current full recipe')
            else:
                expected = Store.sha256_json(B.race_settings(dict(rounds=1, vendor=vendor), race))
                if key.get('settings_sha256') != expected:
                    reasons.append('current work-changing recipe settings differ')
            if key.get('vendor') != vendor or key.get('neural_shape') != 'full' or key.get('rounds') != 1:
                reasons.append('vendor/shape/round count differs')
            if not B.evidenced_opponent(cell) or not B.successful_opponent_cell(cell):
                reasons.append('time/quality/evidence is incomplete or unsuccessful')
            completion = B.measurement_completion(dict(family='neural', lane=key.get('lane'), cells=[cell]))
            if completion['status'] != 'recorded':
                reasons.append('time/quality presence check failed')
            if Store.missing(key):
                reasons.append('opponent key has missing fields')
            quality = cell.get('quality') or {}
            relative = [name for name in quality if 'vs_ours' in name or 'equal_ours' in name or name == 'first_divergence_index']
            item = dict(key_id=key_id, key=key, measured_at=record.get('measured_at'),
                        source_commit=record.get('commit'), status='eligible_pending_current_preflight' if not reasons else 'excluded',
                        reasons=reasons, quality_fields=sorted(quality),
                        prior_own_relative_fields=relative,
                        data_key_kind='core-wheel fingerprint' if str(key.get('data_sha256')).startswith('mojolearn-sources:') else 'input-file hash',
                        evidence={part: {field: cell['measurement'][part].get(field) for field in ('sha256', 'r2_key', 'status')}
                                  for part in ('source', 'artifacts')},
                        original_board=board_by_lane.get(key.get('lane')))
            (rejected if reasons else accepted).append(item)
            if not reasons:
                # Never relabel a key, a time, or an old comparison as new.
                Store.append(str(seed), record)
        selected = {(row['key']['lane'], row['key']['arm']) for row in accepted}
        missing = [dict(recipe=r['id'], arm=a, action='measure if supported and matched by normal preflight')
                   for r in plan for a in r['opponents'] if (r['lane'], a) not in selected]
        audit[vendor] = dict(eligible_count=len(accepted), excluded_count=len(rejected),
            eligible_recipe_count=len({r['key']['lane'] for r in accepted}),
            wheel_fingerprint_count=sum(r['data_key_kind'] == 'core-wheel fingerprint' for r in accepted),
            input_hash_count=sum(r['data_key_kind'] == 'input-file hash' for r in accepted),
            seed_store=dict(path=str(seed), sha256=sha(seed)), eligible=accepted, excluded=rejected,
            no_eligible_retained_cell=missing,
            final_reuse='pending fresh effective parameter, library/device, settings and input-key readback; no timing-based decisions')
        new_remote = '/root/measure-neural29-refactor-' + vendor + '-20261027'
        cfg = dict(old_cfg)
        cfg.pop('previous_attempt', None)
        cfg.update(repo='__FROZEN_CHECKOUT__', out=str(out / (vendor + '-run')),
            source_commit='__FROZEN_NEW_SOURCE_SHA__', source_bundle='__FROZEN_SOURCE_TAR_GZ__',
            remote_runner='__FROZEN_CHECKOUT__/tools/full_do_measurement_worker.py',
            manifest=str(plan_dir / 'manifest.json'), version='__CANDIDATE_VERSION__',
            commit_boards=False, data_keys=['gbm-bench/higgs/higgs_speed.npz'])
        # This template cannot launch until root fills candidate-only inputs.
        v = dict(old_vendor)
        v.update(remote_root=new_remote, bootstrap=False, require_existing_box=True,
                 release_after_archive=False, neural_smoke=False, runtime_env=dict(SWITCHES),
                 python='__INSTALLED_CANDIDATE_PYTHON__', wheel_manifest='__DEVELOPMENT_RUNTIME_MANIFEST_PENDING_INTEGRATION__',
                 cache=new_remote + '/cache', opponent_store=str(seed))
        cfg['vendors'] = [v]
        configs[vendor] = cfg
    write(plan_dir / 'manifest.json', manifest)
    write(plan_dir / 'recipes.json', dict(schema='mojolearn.neural29-refactor-plan.v1', recipes=recipes,
          measured=False, mode='identical', shape='full', warmups=1, scored_rounds=1,
          own_measurement='fresh for every admitted recipe', candidate_switches=SWITCHES))
    write(plan_dir / 'opponent-reuse.json', dict(schema='mojolearn.neural29-opponent-plan.v1', vendors=audit))
    for vendor, config in configs.items():
        write(out / (vendor + '-controller.template.json'), config)
    write(out / 'source-boards.json', source_boards)
    write(plan_dir / 'launch-plan.json', dict(schema='mojolearn.neural29-launch-plan.v1', launched=False,
        candidate_freeze='pending complete neural/tape native-operation repair, root focused Mac bit check and target builds',
        runtime_artifacts='incremental development artifacts; no release required; manifest admission integration pending',
        separate_gpu_validation_campaign=False, old_evidence_mutated=False,
        prerequisite=['neural including tape has no Python computation/callbacks/internal CPython dispatch and stable metadata retained natively',
                      'one focused Mac bit check', 'actual NVIDIA/AMD target builds',
                      'frozen candidate installation and truthful runtime provenance manifest'],
        vendors={v: dict(controller_template=str(out / (v + '-controller.template.json')),
                        remote_root=configs[v]['vendors'][0]['remote_root'],
                        ssh_target='root@162.243.78.145' if v == 'amd' else '-p 30772 root@38.80.152.75',
                        require_existing_box=True, release_after_archive=False, neural_smoke=False,
                        expected=29, serial_within_vendor=True) for v in CAMPAIGNS},
        failure_policy='refused/skipped/failed/incomplete rows retained; continue independent recipes; retention failure pauses before next recipe',
        opponent_reuse='unmodified evidenced keys; fresh effective readback required; missing or mismatching opponents measured',
        quality='time and quality on both arms; historical vs_ours fields require fresh reference resolution before current presentation'))
    print(json.dumps({v: {k: audit[v][k] for k in ('eligible_count', 'eligible_recipe_count', 'excluded_count',
          'wheel_fingerprint_count', 'input_hash_count')} for v in CAMPAIGNS}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--plan-dir', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    prepare(args.evidence_root.resolve(), args.out.resolve(), args.plan_dir.resolve())
