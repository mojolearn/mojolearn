"""Check the original shared atomic launch packet; no projected/fabricated receipts."""
import json
from pathlib import Path
from controller_preflight import frozen_inputs

def exact_column_begin(path,vendor,lane,revision,document):
 path=Path(path);launch=json.loads(path.read_text());rid='identity-'+vendor+'/'+lane
 lanes=launch.get('lanes',[]);runs=launch.get('runs',{})
 if (launch.get('schema')!='mojolearn.release-identity-launch.v1' or launch.get('phase')!='before-execution' or launch.get('vendor')!=vendor or launch.get('source_revision')!=revision or not lanes or len(lanes)!=len(set(lanes)) or lane not in lanes or set(runs)!={'identity-'+vendor+'/'+n for n in lanes}):
  raise ValueError('lane lacks exact original column launch packet')
 run=runs[rid];wanted=document['requirements'][rid]
 if run.get('source_revision')!=revision or not run.get('run_id') or run.get('fingerprint')!=wanted['fingerprint']:
  raise ValueError('begun run fingerprint differs')
 return path
