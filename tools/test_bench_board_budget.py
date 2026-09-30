"""Real process timeout checks, including a worker in a separate session."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import bench_board as bb
import bench_board_resume as resume


def test_timeout_kills_detached_worker_and_next_race_runs(tmp_path):
    child = tmp_path/'child.pid'
    script = tmp_path/'hung.py'
    script.write_text('import subprocess,sys,time,pathlib\n'
        'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True)\n'
        f'pathlib.Path({str(child)!r}).write_text(str(p.pid))\n'
        'time.sleep(60)\n')
    ctx={'_race_stop':time.monotonic()+1}
    rc=bb.run_race_logged(ctx,[sys.executable,str(script)],None,str(tmp_path/'hung.log'),3600)
    assert rc==124
    pid=int(child.read_text())
    for _ in range(30):
        state=subprocess.run(['ps','-o','stat=','-p',str(pid)],capture_output=True,text=True).stdout.strip()
        if not state or state.startswith('Z'):break
        time.sleep(.05)
    assert not state or state.startswith('Z')
    assert bb.run_race_logged({'_race_stop':time.monotonic()+3},[sys.executable,'-c','pass'],None,str(tmp_path/'next.log'),3600)==0


def test_budget_shared_between_launches_and_expiry_skips_launch(tmp_path):
    ctx={'_race_stop':time.monotonic()+.2}
    with patch.object(bb,'run_logged',return_value=0) as run:
        bb.run_race_logged(ctx,['unused'],None,str(tmp_path/'a.log'),3600)
        first=run.call_args.args[3]
        time.sleep(.03)
        bb.run_race_logged(ctx,['unused'],None,str(tmp_path/'b.log'),3600)
        assert 0<run.call_args.args[3]<first<=.2
    ctx['_race_stop']=0
    with patch.object(bb,'run_logged') as run:
        assert bb.run_race_logged(ctx,['unused'],None,str(tmp_path/'expired.log'),3600)==124
        run.assert_not_called()


import pytest

@pytest.mark.parametrize('family', ['trees','neural','algos','classical2','classical'])
def test_every_family_stops_and_records_timeout(family,tmp_path,monkeypatch):
    race=next(r for r in bb.plan_races('nvidia',['identical'],cpu_arm=False) if r['family']==family)
    ctx={'out':str(tmp_path),'nice':0,'infer':False,'race_deadline_s':21600,
         '_race_stop':time.monotonic()+.1}
    cmd=[sys.executable,'-c','import time; time.sleep(60)']
    name={'trees':'tree','neural':'neural','algos':'algos','classical2':'more','classical':'classical'}[family]
    monkeypatch.setattr(bb,name+'_cmd',lambda *a:(cmd,{}) if family=='trees' else (cmd,{},21600))
    if family!='trees':
        monkeypatch.setattr(bb,name+'_json_path',lambda *a:str(tmp_path/'missing.json'))
    monkeypatch.setattr(bb,'child_env',lambda *a:None)
    monkeypatch.setattr(bb,'base_cell',lambda *a:{'arm':a[2]})
    monkeypatch.setattr(bb,'tree_cells',lambda *a:[])
    monkeypatch.setattr(bb,'add_ratios',lambda cells:cells)
    monkeypatch.setattr(bb,'race_host',lambda:{})
    monkeypatch.setattr(bb,'attach_params',lambda *a:None)
    rec=bb._run_race(ctx,race)
    assert rec['rc']==124 and rec['status']=='failed'
    assert rec['reason'].startswith('TIMEOUT')
    assert all(c['status']=='timeout' for c in rec['cells'])
