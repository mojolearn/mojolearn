import json,pathlib,subprocess,shlex,time
E=pathlib.Path(__file__).resolve().parent
old='''        if prev and prev.get("status") == "failed" and args.skip_failed:
            print("bench_board: skip %s (failed earlier; --skip-failed)" % r["id"], flush=True)
            continue
'''
new='''        if prev and prev.get("status") == "failed":
            if args.skip_failed and not _rerun_wanted(args, r["id"], prev):
                print("bench_board: skip %s (failed earlier; --skip-failed)" % r["id"], flush=True)
                continue
            # Explicit repair retains the original failed/refused receipt.
            result.setdefault("superseded", []).append(prev)
'''
for route in ['specific','default']:
 cfg=json.loads((E/(route+'-owner/config.json')).read_text());ssh=['ssh',*cfg['ssh']]
 code="from pathlib import Path;import hashlib,json;p=Path('/root/opponent-harness/tools/bench_board.py');s=p.read_text();old="+repr(old)+";new="+repr(new)+";assert old in s or new in s;p.write_text(s.replace(old,new));print(hashlib.sha256(p.read_bytes()).hexdigest())"
 p=subprocess.run([*ssh,'python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=45)
 (E/(route+'-harness-repair.json')).write_text(json.dumps(dict(base_harness='b0f5242d244ef75cd9ba7d78ca92e10008d75510',repair_commit='8b0bdf4d4',remote_bench_board_sha256=p.stdout.strip(),time=time.time()),indent=2)+'\n')
# Let the current race finish; removing the seal causes the wrapper to yield
# before its next race, then the existing serial worker resumes updated code.
route='specific';claim=E/'repair-claims/specific-implicit-rerun';claim.touch();cfg=json.loads((E/'specific-owner/config.json').read_text());ssh=['ssh',*cfg['ssh']];started=time.time()
subprocess.run([*ssh,'rm -f /root/overnight-nvidia/CANDIDATES_SEALED'],capture_output=True,timeout=45,check=True)
while time.time()-started<1800:
 p=subprocess.run([*ssh,"cat /root/campaign-results/status.json"],capture_output=True,text=True,timeout=45)
 if p.returncode==0 and json.loads(p.stdout)['phase']!='GPU_OPPONENTS_RUNNING':
  claim.unlink(missing_ok=True);(E/'implicit-rerun-boundary.json').write_text(json.dumps(dict(status='RELEASED',time=time.time(),worker=json.loads(p.stdout)),indent=2)+'\n');break
 time.sleep(10)
else:
 claim.unlink(missing_ok=True);(E/'implicit-rerun-boundary.json').write_text(json.dumps(dict(status='BOUNDARY_TIMEOUT',time=time.time())));subprocess.run(['/opt/homebrew/bin/codex','queue','--thread','01a10f60-3998-7c91-845d-552e1c04d795','--message','NVIDIA isolated implicit environment is ready, but boundary pause timed out; inspect specific tail state and rerun only failed ALS with explicit selector repair. No active race killed.'],capture_output=True,timeout=30)
