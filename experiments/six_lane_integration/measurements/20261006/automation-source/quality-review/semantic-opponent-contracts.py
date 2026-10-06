"""Narrow source-audited task equivalences; no metric or tolerance changes."""
from pathlib import Path
import hashlib,json,subprocess
ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/andrewhendel/CascadeProjects/mojolearn')
AUDIT=ROOT/'semantic-source-audit/source-manifest.json'
MANIFEST=json.loads(AUDIT.read_text())
FREEZE='47301d12b14859e81cadc9ab6a0cd4f728d0e206'
def params(record):
 # Function adapters retain their declared kwargs directly. Estimators use
 # the nested get_params schema. This is schema decoding, not equivalence.
 return record.get('params',{}) if record.get('__record__') else {k:v for k,v in record.items() if not k.startswith('__')}
def apply(lane,arm,own,opponent,raw,own_source_sha):
 rules=[];mismatch={k:[v,opponent.get(k)] for k,v in own.items() if k not in opponent or opponent[k]!=v}
 if raw.get('commit')!=FREEZE:return mismatch,rules
 required={
  'ridge':['python/mojolearn/linear_model.py','tools/bench_board_more.py','sklearn/linear_model/_ridge.py','sklearn/linear_model/_base.py'],
  'lasso':['python/mojolearn/_solver_impl.py','tools/bench_board_more.py','sklearn/linear_model/_coordinate_descent.py'],
  'elasticnet':['python/mojolearn/_solver_impl.py','tools/bench_board_more.py','sklearn/linear_model/_coordinate_descent.py'],
  'kmeans':['python/mojolearn/cluster.py','tools/classical_two_datasets.py'],
 }.get(lane,[])
 if not required:return mismatch,rules
 sources={}
 for name in required:
  entry=MANIFEST[name]
  if hashlib.sha256(Path(entry['saved']).read_bytes()).hexdigest()!=entry['sha256']:return mismatch,[]
  if name.startswith('python/'):
   try:current=subprocess.check_output(['git','show',own_source_sha+':'+name],cwd=REPO,stderr=subprocess.DEVNULL)
   except subprocess.CalledProcessError:return mismatch,[]
   if hashlib.sha256(current).hexdigest()!=entry['sha256']:return mismatch,[]
  sources[name]=entry['sha256']
 if lane in ('ridge','lasso','elasticnet'):
  if arm!='sklearn-cpu' or raw['arms'][arm].get('info',{}).get('version')!='1.7.2':return mismatch,[]
 def accept(field,reason):
  if field in mismatch:
   rules.append({'field':field,'recorded_values':mismatch.pop(field),'reason':reason,'source_sha256':sources,'audit_manifest':str(AUDIT),'scope':'Task/objective comparison only; no numerical-identity or equal solver trajectory claim'})
 if lane=='ridge' and own.get('alpha')==opponent.get('alpha') and own.get('fit_intercept')==opponent.get('fit_intercept') and own.get('normalize') is False and opponent.get('positive') is False:
  if own.get('solver') in ('eig','auto') and opponent.get('solver')=='cholesky':
   accept('solver','Frozen board explicitly compares eig and Cholesky implementations of min ||y-Xw||²+alpha||w||²; same alpha/intercept. Existing task metric gate judges numerical differences.')
  if 'normalize' not in opponent:accept('normalize','Our normalize=False adds no variance scaling; installed sklearn _preprocess_data only centers when fitting intercept and sets X_scale=ones. The removed option is semantically false here.')
 if lane in ('lasso','elasticnet') and own.get('selection')==opponent.get('selection')=='cyclic' and own.get('precompute')==opponent.get('precompute') is False:
  if own.get('random_state') is None and opponent.get('random_state')==7:accept('random_state','Both frozen implementations update cyclically; random_state only selects random coordinate order, which is not selected. Their documented stopping differences remain subject to unchanged quality gate.')
  if own.get('solver')=='cd' and 'solver' not in opponent:accept('solver','Our explicit cd and sklearn Lasso/ElasticNet path both use coordinate descent on the same documented penalty objective; sklearn exposes no solver selector. No equal stopping trajectory is claimed.')
 if lane=='kmeans' and own.get('init')==opponent.get('init')=='k-means++' and own.get('n_init')==opponent.get('n_init')==1:
  if own.get('init_centroids') is None and 'init_centroids' not in opponent:accept('init_centroids','Explicit-centroid buffer is inactive under k-means++ in frozen KMeans source.')
  if arm=='torch-gpu' and own.get('random_state')==opponent.get('seed')==7:accept('random_state','Frozen TorchKMeans creates a per-fit torch.Generator and manual_seed(7); declared seed7 matches our random_state7, without claiming identical RNG draws.')
  if arm in ('torch-gpu','sklearn-cpu') and own.get('oversampling_factor')==0 and 'oversampling_factor' not in opponent:
   accept('oversampling_factor','Frozen classical lane explicitly pairs our sequential k-means++ (oversampling0) with reference greedy k-means++ on the same Lloyd/inertia task. Initialization algorithms intentionally differ and are recorded; independent inertia quality remains mandatory.')
 return mismatch,rules
