import copy,json
import pytest
from controller_preflight_v2 import exact_column_begin

@pytest.mark.parametrize('damage',[None,'vendor','source','lane','extra_run','fingerprint','missing_run_id'])
def test_original_atomic_packet(tmp_path,damage):
 revision='a'*40;lanes=['one','two'];runs={'identity-apple/'+n:dict(run_id=n,fingerprint=n,source_revision=revision) for n in lanes}
 d=dict(schema='mojolearn.release-identity-launch.v1',phase='before-execution',vendor='apple',source_revision=revision,lanes=lanes,runs=runs);doc=dict(requirements={k:dict(fingerprint=v['fingerprint']) for k,v in runs.items()})
 if damage=='vendor':d['vendor']='cpu'
 elif damage=='source':d['source_revision']='b'*40
 elif damage=='lane':d['lanes']=['two']
 elif damage=='extra_run':d['runs']['identity-apple/extra']=dict(run_id='extra')
 elif damage=='fingerprint':d['runs']['identity-apple/one']['fingerprint']='bad'
 elif damage=='missing_run_id':d['runs']['identity-apple/one'].pop('run_id')
 path=tmp_path/'launch.json';path.write_text(json.dumps(d))
 if damage:
  with pytest.raises(ValueError):exact_column_begin(path,'apple','one',revision,doc)
 else:assert exact_column_begin(path,'apple','one',revision,doc)==path
