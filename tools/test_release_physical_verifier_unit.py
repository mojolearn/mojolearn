import copy, importlib.util, unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('physical_unit',Path(__file__).with_name('release_physical_verifier_unit.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
S='d'*40;L='par-forest';F=['base','ties'];P=['train','infer']
def table():return {'records':[{'commit':S}], 'cells':{L+'/'+f:{p:{'ref':'a'*16,'cols':{'nvidia':0}} for p in P} for f in F}}
def report(lane=L):return dict(format='mojolearn.verify-par.v1',vendor='hip',devices=[0,1],lanes=[lane],fixtures=F,state='VERIFIED',passed=True,perturbed=False,witness_refusal=None,cells=[dict(lane=lane,fixture=f,part=p,verdict='IDENTICAL') for f in F for p in ('train','infer','model','batch','stepfull')],witness={'placement':'physical'},cell_witnesses=[{'lane':lane,'fixture':f,'witness_refusal':None} for f in F])
class Guards(unittest.TestCase):
 def test_inventory_accepts_distinct(self):
  x=[dict(index=i,uuid=str(i),pci_bus_id=str(i)) for i in (0,1)];self.assertEqual(m.physical_inventory(x),x)
 def test_inventory_refuses_missing_or_duplicate(self):
  for field in ('uuid','pci_bus_id'):
   x=[dict(index=i,uuid=str(i),pci_bus_id=str(i)) for i in (0,1)];x[1][field]=x[0][field]
   with self.subTest(field=field),self.assertRaises(ValueError):m.physical_inventory(x)
  with self.assertRaises(ValueError):m.physical_inventory([dict(index=0,uuid='x',pci_bus_id='x')])
 def test_shared_table_equality(self):m.compare_tables(table(),table(),L,F,P,S)
 def test_shared_table_mismatch_is_not_regenerated(self):
  x=table();x['cells'][L+'/base']['train']['ref']='b'*16
  with self.assertRaises(ValueError):m.compare_tables(table(),x,L,F,P,S)
 def test_reference_missing_conflicting_or_wrong_source(self):
  for key,value in [('ref','REFUSED'),('conflict',True),('cols',{'nvidia':[0]}),('cols',{'nvidia':99})]:
   x=table();x['cells'][L+'/base']['train'][key]=value
   with self.subTest(key=key,value=value),self.assertRaises(ValueError):m.strict_table(x,L,F,P,S)
  x=table();x['records'][0]['commit']='e'*40
  with self.assertRaises(ValueError):m.strict_table(x,L,F,P,S)
 def test_physical_positive(self):self.assertEqual(m.strict_physical(report(),L,F,'hip'),'local-physical-complete')
 def test_physical_wrong_scope_or_state(self):
  for key,value in [('vendor','cuda'),('devices',[0,0]),('lanes',['par-kmeans']),('fixtures',['base']),('state','MISMATCH'),('passed',None),('perturbed',True),('witness_refusal','no pool'),('cell_witnesses',[]),('cells',[])]:
   x=report();x[key]=value
   with self.subTest(key=key),self.assertRaises(ValueError):m.strict_physical(x,L,F,'hip')
 def test_per_cell_refusal(self):
  x=report();x['cell_witnesses'][1]['witness_refusal']='same physical device'
  with self.assertRaises(ValueError):m.strict_physical(x,L,F,'hip')
 def test_process_only_is_not_gpu(self):
  x=report();x['witness']['placement']='process-only'
  with self.assertRaises(ValueError):m.strict_physical(x,L,F,'hip')
 def test_declared_one_device_stays_na(self):
  l='par-byte-lm-offload';x=report(l);x.update(state='NOTHING COMPARED',passed=None);[c.update(verdict='N/A') for c in x['cells']];x['witness']['one_device_by_design']=[l]
  self.assertEqual(m.strict_physical(x,l,F,'hip'),'declared-one-device-only')
  x['cells'][0]['verdict']='DIVERGENT'
  with self.assertRaises(ValueError):m.strict_physical(x,l,F,'hip')
if __name__=='__main__':unittest.main()
