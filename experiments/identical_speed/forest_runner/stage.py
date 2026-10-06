"""File/environment staging only; no numerical compilation or GPU calls."""
import pathlib,json,subprocess,hashlib,shutil,sys
P=pathlib.Path(sys.argv[1]);D=json.loads((P/'packet.json').read_text());R=pathlib.Path(D['remote_root']);R.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
for name,h in D['packet_files'].items():assert sha(P/name)==h,name
S=R/'base';subprocess.run(['git','clone','--shared',D['source_donor'],str(S)],check=True)
assert subprocess.check_output(['git','-C',str(S),'rev-parse','HEAD'],text=True).strip()==D['source_sha']
assert subprocess.run(['git','-C',str(S),'diff','--quiet','HEAD']).returncode==0
subprocess.run(['git','-C',str(S),'fetch',str(P/'harness.bundle'),D['harness_ref']],check=True)
H=R/'harness';subprocess.run(['git','-C',str(S),'worktree','add','--detach',str(H),D['harness_sha']],check=True)
runtime=R/'runtime';shutil.copytree(P/'runtime',runtime);runtime_files={str(p):sha(p) for p in runtime.iterdir()}
for arm in D['arms']:
 source=R/arm/'source';source.parent.mkdir();subprocess.run(['git','-C',str(S),'worktree','add','--detach',str(source),D['source_sha']],check=True);pkg=source/'python/mojolearn';files={}
 for relative,record in D['bindings'][arm].items():
  target=pkg/relative;target.parent.mkdir(exist_ok=True,parents=True);shutil.copy2(P/record['packet_path'],target);assert sha(target)==record['sha256'];files[relative]=record['sha256']
 (pkg/'.libs').mkdir(exist_ok=True);(pkg/'.libs/libMojolearnMath.so').symlink_to(runtime/'libMojolearnMath.so')
 env=R/arm/'env';subprocess.run([D['base_python'],'-m','venv','--system-site-packages',str(env)],check=True);py=str(env/'bin/python');site=subprocess.check_output([py,'-c','import site;print(site.getsitepackages()[0])'],text=True).strip();pathlib.Path(site,'mojolearn-source.pth').write_text(str(source/'python')+'\n');subprocess.run([py,'-c','import numpy,torch;print(numpy.__version__,torch.__version__)'],check=True)
 manifest={'schema':'mojolearn-board-artifacts/1','source_commit':D['source_sha'],'numeric_mode':'identical','code_path':'native','vendor':'cuda' if D['vendor']=='nvidia' else 'hip','installation':'source','source_root':str(source),'files':files,'runtime_files':runtime_files}
 (R/arm/'manifest.json').write_text(json.dumps(manifest,indent=2))
shutil.copy2(P/'packet.json',R/'packet.json')
for name in ['run.py','validate.py','owned_process.py','admit.py','scored_worker.py','bench_board_provenance.py','settings.py']:shutil.copy2(P/name,R/name)
(R/'stage.json').write_text(json.dumps({'status':'STAGED_NOT_EXECUTED','source_sha':D['source_sha'],'harness_sha':D['harness_sha'],'arms':D['arms'],'packet_sha256':sha(P/'packet.json'),'runtime_files':runtime_files},indent=2))
