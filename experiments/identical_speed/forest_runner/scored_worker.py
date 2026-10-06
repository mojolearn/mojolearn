"""Run the real forest driver and retain same-process untimed provenance."""
import argparse,json,os,pathlib,runpy,sys,time

def main():
 p=argparse.ArgumentParser();p.add_argument('--fixture',action='store_true');p.add_argument('--artifact-manifest',required=True);p.add_argument('--required-binding',required=True);p.add_argument('--receipt',required=True);p.add_argument('--harness',required=True);p.add_argument('driver_args',nargs=argparse.REMAINDER);a=p.parse_args()
 args=a.driver_args[1:] if a.driver_args[:1]==['--'] else a.driver_args
 if a.fixture:assert '--profile' in args and '--vendor' in args and '--out' in args
 else:assert '--ours-only' in args and '--save-scored-models' in args
 assert not any(x in args for x in ['--rows','--infer','--with-opponents','--ours-ab','--host-digest','--ours-cpu'])
 assert os.environ.get('MOJOLEARN_SPEED_ROUNDS')=='1'
 assert os.environ.get('MOJOLEARN_SPEED_SIZE')=='shipped'
 assert os.environ.get('MOJOLEARN_NUMERIC_MODE')=='identical'
 h=pathlib.Path(a.harness).resolve();sys.path.insert(0,str(h/'tools'))
 import importlib.util
 spec=importlib.util.spec_from_file_location("pinned_guard",pathlib.Path(__file__).with_name("bench_board_provenance.py"));guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
 manifest=guard.identity(a.artifact_manifest)
 assert os.environ.get('MOJOLEARN_SPEED_EXPECTED_VENDOR')==manifest['vendor'], 'expected vendor contract missing or mismatched'
 assert a.required_binding in manifest['files']
 import mojolearn
 guard.installed_check(manifest,pathlib.Path(mojolearn.__file__).parent)
 before=guard.hardware_receipt(manifest['vendor'])
 sys.argv=[str(h/('tools/identical_forest_state_fixture.py' if a.fixture else 'bench/speed/forest_speed_arm.py')),*args]
 started=time.time();code=0;error=None
 try:runpy.run_path(sys.argv[0],run_name='__main__')
 except SystemExit as exc:code=exc.code or 0
 except BaseException as exc:code=1;error=repr(exc)
 receipt={'started_at':started,'finished_at':time.time(),'driver_exit':code,'driver_error':error,'argv':sys.argv,'artifact_manifest_sha256':guard.sha(a.artifact_manifest)}
 try:
  actual=guard.collect(manifest)
  required=[row for row in actual['loaded_files'] if row['file']==a.required_binding]
  assert len(required)==1 and required[0]['sha256']==manifest['files'][a.required_binding]
  assert guard.hardware_receipt(manifest['vendor'])==before
  receipt.update(provenance=actual,hardware_before=before,provenance_verified=True)
 except BaseException as exc:receipt.update(provenance_verified=False,provenance_error=repr(exc));code=1
 out=pathlib.Path(a.receipt);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
 return code
if __name__=='__main__':sys.exit(main())
