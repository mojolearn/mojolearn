"""Bounded deferred admission of two failed LM compile arms to existing GPU owner."""
import json,pathlib,time,shutil,os
R=pathlib.Path('/root/overnight-ab');O=R/'results/opponents';P=O/'loader-repair-admission.json'
def save(**value):
 value['updated_at']=time.time();tmp=P.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(P)
s=json.loads((R/'neural-loader-repair-selection.json').read_text())
queued=O/'loader-repair-queued.json'
if queued.exists():
 save(status='ALREADY_QUEUED',receipt=str(queued));raise SystemExit(0)
deadline=time.time()+4*3600
while time.time()<deadline:
 if (R/'results/repairs/OPPONENTS_DONE').exists() and (R/'results/DONE').exists():break
 save(status='WAITING_FOR_EXISTING_TAIL',deadline=deadline,source_sha=s['source_sha'],arms_by_race=s['arms_by_race'])
 time.sleep(15)
else:
 save(status='FAILED',reason='Existing tail did not reach terminal capture in4h; no markers changed');raise SystemExit(124)
archive=O/'attempts/neural-pre-loader-repair';archive.mkdir(parents=True,exist_ok=True)
shutil.copy2(O/'board/board.json',archive/'board.json')
shutil.copy2(R/'opponent-tail.py',archive/'opponent-tail.py')
shutil.copy2(R/'neural-loader-repair-selection.json',O/'neural-loader-repair-selection.json')
shutil.copy2(R/'neural-loader-repair-tail.py',R/'opponent-tail.py')
queued.write_text(json.dumps(dict(s,status='QUEUED_EXISTING_STREAM_WORKER',queued_at=time.time()),indent=2)+'\n')
(R/'results/repairs/OPPONENTS_DONE').unlink()
save(status='QUEUED_EXISTING_STREAM_WORKER',source_sha=s['source_sha'],arms_by_race=s['arms_by_race'],owner='overnight-ab-stream.service')
