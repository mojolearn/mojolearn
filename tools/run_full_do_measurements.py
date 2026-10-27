#!/usr/bin/env python3
"""Active measurement controller for owned DigitalOcean or RunPod boxes.

Consumes an explicitly prepared JSON run configuration. Each vendor runs the
whole frozen plan, retains recipe logs locally as completion events arrive,
then archives before releasing its owned box and starting the next vendor.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time


def state_file(path):
    values = {}
    for line in Path(path).read_text().splitlines():
        key, sep, value = line.partition('=')
        if sep:
            parts = shlex.split(value)
            values[key] = parts[0] if parts else ''
    return values


def campaign_complete(doc):
    rows = doc.get('recipes') or {}
    return (bool(doc.get('finished')) and len(rows) == doc.get('expected')
            and not doc.get('stop_reason')
            and all(r.get('status') not in ('pending', 'running')
                    and (r.get('retention') or {}).get('status') == 'verified'
                    and not r.get('retention_failure') for r in rows.values()))


def provider_command(vendor, action, minutes=None):
    if vendor.get('provider') == 'runpod':
        cmd = ['bash', 'tools/dev_pod.sh', action, vendor['state_key']]
    else:
        cmd = ['bash', 'tools/do_amd_steward.sh', action]
    return cmd + ([str(minutes)] if minutes is not None else [])


def transport(owned):
    """Keep RunPod's SSH port in every transfer and retention command."""
    parts = shlex.split(owned['SSH_TARGET']) if owned.get('SSH_TARGET') else ['root@'+owned['IP']]
    host = parts[-1]
    flags = parts[:-1]
    scp_flags = ['-P' if flag == '-p' else flag for flag in flags]
    return host, flags, scp_flags, shlex.join(parts)


def run(config):
    repo = Path(config['repo'])
    out = Path(config['out'])
    out.mkdir(parents=True, exist_ok=True)
    journal = {'source_commit': config['source_commit'], 'started': time.time(), 'vendors': {}}
    def save():
        tmp = out / 'controller.json.tmp'
        tmp.write_text(json.dumps(journal, indent=2))
        tmp.replace(out / 'controller.json')
    def command(argv, log, env=None, check=True, input=None):
        with (out / log).open('ab') as stream:
            p = subprocess.run(argv, cwd=repo, env=env, input=input,
                               stdout=stream, stderr=subprocess.STDOUT)
        if check and p.returncode:
            raise RuntimeError('%s exited %d; see %s' % (argv[0], p.returncode, log))
        return p.returncode
    save()
    try:
        for vendor in config['vendors']:
            name = vendor['vendor']
            record = journal['vendors'][name] = {'status': 'preparing', 'started': time.time()}
            save()
            env = dict(os.environ, **vendor.get('provider_env', {}))
            state = Path(vendor['state_dir']) / 'state.env'
            if not state.exists() and vendor.get('require_existing_box', False):
                raise RuntimeError('required existing owned box state is missing: '+str(state))
            if not state.exists():
                command(provider_command(vendor, 'up', 180), name+'-provision.log', env)
            owned = state_file(state)
            box_id = owned.get('POD_ID') or owned.get('DROPLET_ID')
            if vendor.get('expected_id') and box_id != vendor['expected_id']:
                raise RuntimeError('owned box identity changed')
            target, ssh_flags, scp_flags, stage_target = transport(owned)
            record.update(box_id=box_id, target=stage_target, provider=vendor.get('provider', 'digitalocean'))
            save()
            ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ServerAliveInterval=15',
                   '-o', 'ServerAliveCountMax=8', '-o', 'StrictHostKeyChecking=accept-new', *ssh_flags, target]
            scp = ['scp', '-o', 'StrictHostKeyChecking=accept-new', *scp_flags]
            remote = vendor['remote_root']
            command(ssh + ['mkdir -p ' + shlex.quote(remote)], name+'-stage.log')
            for source, dest in ((config['source_bundle'], 'source.tar.gz'),
                                 (config['remote_runner'], 'run.py'),
                                 (config['manifest'], 'manifest.json'),
                                 (vendor['opponent_store'], 'opponent-store.jsonl')):
                command(scp + [source, target+':'+remote+'/'+dest], name+'-stage.log')
            # Credentials stay outside every result/source archive.
            command(ssh + ['umask 077; cat > /root/.mojolearn_r2'], name+'-stage.log',
                    input=Path(config['r2_credentials']).read_bytes())
            remote_config = dict(vendor, source_commit=config['source_commit'], version=config['version'])
            command(ssh + ['cat > '+shlex.quote(remote+'/config.json')], name+'-stage.log',
                    input=json.dumps(remote_config).encode())
            if vendor.get('bootstrap'):
                command(ssh + ['apt-get update -qq && apt-get install -y -qq python3-venv git'], name+'-bootstrap.log')
                command(['sh', 'tools/dataset_store.sh', 'stage', stage_target, *config['data_keys']], name+'-data.log')
            command(provider_command(vendor, 'extend', config['lease_minutes']),
                    name+'-lease.log', env)
            last_lease = time.time()
            record['status'] = 'running'
            save()
            with (out / (name+'-controller.log')).open('a') as log:
                proc = subprocess.Popen(ssh + ['python3 '+shlex.quote(remote+'/run.py')+' '+shlex.quote(remote)],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    if not line.startswith('CAMPAIGN_RECIPE_DONE '):
                        continue
                    event = json.loads(line.split(' ', 1)[1])
                    rid = event['id'].replace('/', '.')
                    # The remote runner provides a metadata/log archive. Large
                    # scored arrays and source archives are retained once in R2.
                    local = out / name / (rid+'.tar.gz')
                    local.parent.mkdir(parents=True, exist_ok=True)
                    command(scp + [target+':'+event['local_archive'], str(local)], name+'-retention.log')
                    with gzip.open(local, 'rb') as f:
                        while f.read(1024*1024):
                            pass
                    if time.time() - last_lease > config['lease_minutes']*30:
                        command(provider_command(vendor, 'extend', config['lease_minutes']),
                                name+'-lease.log', env)
                        last_lease = time.time()
                    boards = out/name/'boards'
                    boards.mkdir(parents=True, exist_ok=True)
                    board_path = boards/(rid+'.json')
                    with tarfile.open(local) as archive:
                        try:
                            member = archive.extractfile('board.json')
                        except KeyError:
                            member = None
                        if member is not None:
                            board_path.write_bytes(member.read())
                    if board_path.exists():
                        ingest_rc = command([sys.executable, 'tools/main_board_ingest.py', '--gate-board',
                                 vendor['board_box']+'='+str(board_path), '--columns', vendor['board_column']],
                                name+'-board-ingest.log', check=False)
                        if ingest_rc:
                            record.setdefault('board_ingest_failures', []).append(event['id'])
                    record['last_recipe'] = event
                    record['completed'] = record.get('completed', 0) + 1
                    save()
                record['returncode'] = proc.wait()
            command(scp + [target+':'+remote+'/campaign/campaign.json', str(out/(name+'-campaign.json'))], name+'-retention.log')
            doc = json.loads((out/(name+'-campaign.json')).read_text())
            record['complete'] = campaign_complete(doc)
            save()
            command(['bash', 'tools/box_archive.sh', 'full-measure-'+name, *ssh_flags, target], name+'-archive.log')
            if not record['complete']:
                raise RuntimeError(name+' has pending work or a shared failure; box preserved for recovery')
            if config.get('commit_boards'):
                board_dir = 'bench/results/bench_board/main-'+vendor['board_column']
                command(['git', 'add', board_dir], name+'-board-publish.log', check=False)
                changed = subprocess.run(['git', 'diff', '--cached', '--quiet', '--', board_dir], cwd=repo).returncode
                if changed:
                    record['board_commit_rc'] = command(['git', 'commit', '--only', board_dir, '-m',
                        'Record full '+name+' measurement campaign '+config['source_commit'][:9]],
                        name+'-board-publish.log', check=False)
                    if record['board_commit_rc'] == 0:
                        record['board_push_rc'] = command(['git', 'push', 'origin', 'main'], name+'-board-publish.log', check=False)
            # Results are already archived on the Mac and R2. Never release another owner.
            # Keep the box until large scored/model/binary archives have also
            # been checked off-box; box_archive intentionally skips >50 MB.
            if vendor.get('release_after_archive', False):
                command(provider_command(vendor, 'down'), name+'-down.log',
                        dict(env, MOJOLEARN_DOWN_NO_ARCHIVE='1'))
            record.update(status='complete', finished=time.time(), failed=doc.get('failed_recipes', []))
            save()
        journal.update(status='complete', finished=time.time())
    except Exception as exc:
        journal.update(status='needs-attention', error=str(exc), stopped=time.time())
        save()
        raise
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()))
