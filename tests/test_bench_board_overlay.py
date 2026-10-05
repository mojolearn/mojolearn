import copy
import importlib.util
import json
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('overlay',Path(__file__).parents[1]/'tools/bench_board_overlay.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def fixture(tmp_path):
 def ref(name,data):
  p=tmp_path/name;p.write_text(json.dumps(data));return {'path':str(p),'sha256':m.digest(p)}
 cell={'arm':'ours','status':'ok','rounds':1,'times_ms':[2.0],'median_ms':2.0,'warmup_ms':3.0,'quality':{'relative_error':0.0}}
 old={'status':'done','cells':[{'arm':'ours','status':'timeout'},{'arm':'other','median_ms':4}],'infer_cells':[{'arm':'ours','median_ms':1}]}
 original=ref('original.json',{'races':{'x':old},'box':{'gpu':'old'}})
 board=ref('new.json',{'races':{'x':{'status':'done','params_check':'MATCHED','cells':[cell]}},'box':{'gpu':'new'},'config':{'artifact_identity':{'candidate':'abc'}}})
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
