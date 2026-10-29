import ast,json,hashlib,shutil,sys,tarfile,time
from pathlib import Path
from types import SimpleNamespace
import pytest
from controller_process import run_process

@pytest.mark.parametrize('ignore_parent',[False,True])
def test_timeout_kills_child_group_and_preserves_log(tmp_path,ignore_parent):
    heartbeat=tmp_path/'heartbeat'
    child="import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);f=open("+repr(str(heartbeat))+",'a');\nwhile True:f.write('x');f.flush();time.sleep(.02)"
    parent="import subprocess,sys,signal,time;"+("signal.signal(signal.SIGTERM,signal.SIG_IGN);" if ignore_parent else '')+"subprocess.Popen([sys.executable,'-c',"+repr(child)+"]);print('producer-started',flush=True);time.sleep(60)"
    result=run_process([sys.executable,'-c',parent],cwd=tmp_path,log=tmp_path/'run.log',timeout=.8,kill_grace=.1)
    assert result['timed_out'] and result['exit_code']==124
    assert 'producer-started' in (tmp_path/'run.log').read_text()
    size=heartbeat.stat().st_size;assert size>0
    time.sleep(.1);assert heartbeat.stat().st_size==size
    assert result['elapsed_seconds']<3

def test_normal_failure_exit_is_preserved(tmp_path):
    r=run_process([sys.executable,'-c',"print('real failure');raise SystemExit(7)"],cwd=tmp_path,log=tmp_path/'run.log',timeout=2)
    assert r['exit_code']==7 and not r['timed_out']

@pytest.mark.parametrize('classification,exit_code',[('producer-timeout',124),('producer-missing-final-report',-11),('malformed-or-missing-report-receipt',0)])
def test_partial_retention_has_no_fabricated_report(tmp_path,classification,exit_code):
    source=Path(__file__).with_name('run_frozen_chunks_v5.py').read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='retain_partial')
    stored={}
    def request(creds,method,key,data=None):
        if method=='PUT':stored[key]=data;return 200,b''
        return 200,stored[key]
    def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
    launch=tmp_path/'launch.json';launch.write_text('{"original":true}')
    env=dict(B=tmp_path,REV='a'*40,CFG={'launches':{'nvidia':str(launch)}},save=save,pathlib=__import__('pathlib'),shutil=shutil,tarfile=tarfile,hashlib=hashlib,evidence_r2=SimpleNamespace(_creds=lambda:{}),bincache=SimpleNamespace(r2_request=request))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<partial>','exec'),env)
    log=tmp_path/'producer.log';log.write_text('partial original bytes\n');log.with_suffix('.exit').write_text('124');log.with_suffix('.execution.json').write_text('{"timed_out":true}')
    out=tmp_path/'unit';out.mkdir();(out/'verify.json.progress.json').write_text('{"partial":true}')
    env['retain_partial']({'vendor':'nvidia','phase':'routine-all','lane':'example'},out,log,classification,exit_code)
    assert not (out/'verify.json').exists()
    summary=json.loads((out/'controller-summary.json').read_text());assert summary['exit_code']==exit_code and summary['classification']==classification
    receipt=json.loads((out/'r2-retention.json').read_text());assert receipt['partial'] and receipt['readback_verified']
    assert hashlib.sha256(stored[receipt['key']]).hexdigest()==receipt['sha256']
    with tarfile.open(receipt['archive']) as t:
        assert 'example/verify.json' not in t.getnames()
        assert t.extractfile('example/producer-control/producer.log').read()==b'partial original bytes\n'
        assert t.extractfile('example/producer-control/original-launch.json').read()==launch.read_bytes()

@pytest.mark.parametrize('damage',['missing','truncated-report','truncated-receipt','nonobject-report'])
def test_run_retains_crash_or_malformed_output_before_nonpass(tmp_path,damage):
    source=Path(__file__).with_name('run_frozen_chunks_v5.py').read_text();tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    chunk={'vendor':'nvidia','phase':'routine-all','lane':'example'}
    out=tmp_path/'runs/nvidia/routine-all/example';out.mkdir(parents=True)
    if damage!='missing':
        (out/'verify.json').write_text('{' if damage=='truncated-report' else '[]' if damage=='nonobject-report' else '{}')
        (out/'verifier-run.json').write_text('{' if damage=='truncated-receipt' else '{}')
    log=tmp_path/'controller-logs/nvidia/routine-all/example.producer.execution.json';log.parent.mkdir(parents=True);log.write_text('{"exit_code":-11}')
    calls=[]
    def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d))
    env=dict(B=tmp_path,json=json,datetime=__import__('datetime'),save=save,retain_partial=lambda *args:calls.append(args))
    exec(compile(ast.Module(body=[node],type_ignores=[]),'<run>','exec'),env)
    with pytest.raises(RuntimeError):env['run'](chunk)
    assert len(calls)==1 and calls[0][-1]==-11
    assert calls[0][-2]==('producer-missing-final-report' if damage=='missing' else 'malformed-or-missing-report-receipt')
