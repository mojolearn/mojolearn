import copy
import importlib.util
import json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('overlay',Path(__file__).parents[1]/'tools/bench_board_overlay.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(tmp_path):
 def ref(name,data):
  p=tmp_path/name;p.write_text(json.dumps(data));return {'path':str(p),'sha256':m.digest(p)}
 cell={'arm':'ours','status':'ok','rounds':1,'times_ms':[2.0],'median_ms':2.0,'warmup_ms':3.0,'quality':{'relative_error':0.0},'params':{'seed':7},'settings':{'rounds':1}}
 old={'status':'done','cells':[{'arm':'ours','status':'timeout','params':{'seed':7},'settings':{'rounds':1}},{'arm':'other','median_ms':4}],'infer_cells':[{'arm':'ours','median_ms':1}]}
 original=ref('original.json',{'races':{'x':old},'box':{'gpu':'old'},'config':{'data_sha256':{'taxi':'abc'}}})
 board=ref('new.json',{'races':{'x':{'status':'done','params_check':'MATCHED','cells':[cell]}},'box':{'gpu':'new'},'config':{'data_sha256':{'taxi':'abc'},'artifact_identity':{'candidate':'abc'}}})
 audit=ref('audit.json',{'status':'PASS','failures':[],'board_sha256':board['sha256']})
 capture=ref('capture.json',{'verified_at':1,'manifest_sha256':'abc','rows':[{'path':'new/board.json','sha256':board['sha256']}]})
 return {'original':original,'repairs':[{'board':board,'audit':audit,'capture':capture,'capture_board_path':'new/board.json','races':['x'],'source_description':'base + candidate','hardware_description':'different UUID'}]}

def test_preserve_original_opponents_inference_and_provenance(tmp_path):
 x=fixture(tmp_path);r=m.apply(x);row=r['races']['x'];assert row['cells'][0]['median_ms']==2;assert row['cells'][1]['arm']=='other';assert row['overlay_original_race']['cells'][0]['status']=='timeout';assert row['infer_cells'][0]['median_ms']==1;assert row['repair_overlay']['replacement_box']=={'gpu':'new'};assert not r['repair_overlay']['identical_qualified']

def test_changed_evidence_rejected(tmp_path):
 x=fixture(tmp_path);Path(x['repairs'][0]['board']['path']).write_text('{}')
 with pytest.raises(ValueError,match='changed evidence'):m.apply(x)

def test_wrong_capture_path_rejected(tmp_path):
 x=fixture(tmp_path);x['repairs'][0]['capture_board_path']='wrong'
 with pytest.raises(ValueError,match='absent'):m.apply(x)

def test_duplicate_race_rejected(tmp_path):
 x=fixture(tmp_path);x['repairs']*=2
 with pytest.raises(ValueError,match='duplicate'):m.apply(x)

@pytest.mark.parametrize('quality',[{'error':'failed'},{'metric':float('nan')},{'metric':float('inf')},{}])
def test_invalid_quality(quality):
 with pytest.raises(ValueError):m.validate_cell({'status':'ok','rounds':1,'times_ms':[1],'median_ms':1,'warmup_ms':1,'quality':quality})

def test_stale_ratios_cleared(tmp_path):
 x=fixture(tmp_path);p=Path(x['original']['path']);d=json.loads(p.read_text());d['races']['x']['cells'][1]['ratio_ours_identical_over']=999;p.write_text(json.dumps(d));x['original']['sha256']=m.digest(p)
 assert m.apply(x)['races']['x']['cells'][1]['ratio_ours_identical_over'] is None

@pytest.mark.parametrize('field,value',[('data_sha256',{'taxi':'different'}),('vendor','amd')])
def test_changed_run_identity(tmp_path,field,value):
 x=fixture(tmp_path);p=Path(x['original']['path']);d=json.loads(p.read_text());d['config'][field]=value;p.write_text(json.dumps(d));x['original']['sha256']=m.digest(p)
 with pytest.raises(ValueError):m.apply(x)

def test_missing_original_settings_refused(tmp_path):
 x=fixture(tmp_path);p=Path(x['original']['path']);d=json.loads(p.read_text());del d['races']['x']['cells'][0]['settings'];p.write_text(json.dumps(d));x['original']['sha256']=m.digest(p)
 with pytest.raises(ValueError,match='settings'):m.apply(x)

def test_normal_board_renderer_without_overlay():
 import sys
 sys.path.insert(0,str(Path(__file__).parents[1]/'tools'))
 import bench_board
 board={'races':{'classical/dbscan/taxi/rows=full':{'family':'classical','lane':'dbscan','dataset':'taxi','rows':None,'cells':[{'arm':'ours','mode':'identical','library':'mojolearn','device':'gpu','status':'ok','median_ms':1,'min_ms':1,'max_ms':1,'rounds':1,'quality':{},'verdict':'PASS'}]}}}
 rendered=bench_board.render_board(board)
 assert 'dbscan' in rendered and 'SCOPED REPAIR OVERLAY' not in rendered
