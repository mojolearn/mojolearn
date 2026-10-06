import pathlib,subprocess,json,hashlib
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def admit(root):
 R=pathlib.Path(root);D=json.loads((R/'packet.json').read_text());stage=json.loads((R/'stage.json').read_text())
 assert stage['packet_sha256']==sha(R/'packet.json')
 assert D['rounds']==D['warmups']==1 and D['opponents'] is False
 for name in ['run.py','validate.py','owned_process.py','admit.py','scored_worker.py','bench_board_provenance.py','settings.py']:
  assert sha(R/name)==D['packet_files'][name],name
 def clean(path,commit):
  assert subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()==commit
  assert subprocess.run(['git','-C',str(path),'diff','--quiet','HEAD']).returncode==0
 clean(R/'harness',D['harness_sha'])
 expected_runtime={str(R/name):h for name,h in D['packet_files'].items() if name.startswith('runtime/')}
 assert stage['runtime_files']==expected_runtime
 for path,h in expected_runtime.items():assert sha(path)==h,path
 for arm in D['arms']:
  source=R/arm/'source';clean(source,D['source_sha']);manifest=json.loads((R/arm/'manifest.json').read_text())
  expected={'schema':'mojolearn-board-artifacts/1','source_commit':D['source_sha'],'numeric_mode':'identical','code_path':'native','vendor':'cuda' if D['vendor']=='nvidia' else 'hip','installation':'source','source_root':str(source),'files':{rel:r['sha256'] for rel,r in D['bindings'][arm].items()},'runtime_files':expected_runtime}
  assert manifest==expected,arm
  for rel,h in expected['files'].items():assert sha(source/'python/mojolearn'/rel)==h,(arm,rel)
 return D
