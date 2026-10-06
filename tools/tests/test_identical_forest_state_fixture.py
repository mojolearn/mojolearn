import importlib.util
from pathlib import Path
import unittest
import numpy as np

p=Path(__file__).parents[1]/'identical_forest_state_fixture.py'
s=importlib.util.spec_from_file_location('forest_fixture',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

class FixtureTests(unittest.TestCase):
    def test_counts_shapes_and_routes(self):
        rf=list(m.fixtures('rf'));et=list(m.fixtures('et'))
        self.assertEqual(len(rf),6);self.assertEqual(len(et),12)
        self.assertEqual({x['X'].shape[0] for x in rf},{1023,1024,1025})
        self.assertEqual({x['X'].shape[1] for x in et},{15,16,17,63,64,65})
        self.assertEqual(sum(x['u16_eligible'] for x in et),6)
        for x in rf+et:
            self.assertTrue(np.array_equal(x['X'][0],x['X'][1]))
            self.assertEqual(x['X'].dtype,np.float32)
    def test_deterministic_input_bytes(self):
        a=list(m.fixtures('et'));b=list(m.fixtures('et'))
        self.assertEqual([m.arrays_hash({k:c[k] for k in ('X','y','Xq','yq')}) for c in a],
                         [m.arrays_hash({k:c[k] for k in ('X','y','Xq','yq')}) for c in b])
    def test_hash_binds_every_export_byte_and_type(self):
        a={'offsets':np.array([0,2],np.int32),'leaves':np.array([0.,-0.],np.float32)}
        h=m.arrays_hash(a);b={k:v.copy() for k,v in a.items()};b['leaves'][1]=0.
        self.assertNotEqual(h,m.arrays_hash(b))
        b={**a,'offsets':a['offsets'].astype(np.int64)};self.assertNotEqual(h,m.arrays_hash(b))

class ScoredExportTests(unittest.TestCase):
    def test_existing_score_export_no_fit_or_prediction(self):
        import sys,tempfile,json
        from types import SimpleNamespace
        sys.path.insert(0,str(Path(__file__).parents[1]))
        path=Path(__file__).parents[2]/'bench/speed/forest_speed_arm.py'
        spec=importlib.util.spec_from_file_location('forest_speed_capture',path)
        f=importlib.util.module_from_spec(spec);spec.loader.exec_module(f)
        class Model:
            _cfg={"n_estimators":3}
            def save(self,path):
                np.savez(path,**{k:np.array([0,1],np.int32) for k in ('offsets','colid','quesval','left_child','leaves','meta')})
            def fit(self,*a): raise AssertionError('extra fit')
            def predict(self,*a): raise AssertionError('extra prediction')
        calls=[];pred=np.array([.1,.2],np.float32)
        arm=SimpleNamespace(name='ours',score=lambda model,data:(calls.append(1) or [('rmse',.3,pred)]))
        retained={};f.retain_scored_forest(arm,retained);model=Model();arm.score(model,SimpleNamespace(_ours_X=np.zeros((2,2),np.float32),_ours_y=np.zeros(2,np.float32),_ours_Xtest=np.zeros((2,2),np.float32),y_test=np.zeros(2,np.float32)))
        pred[0]=99
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'fresh';f.save_scored_forests(out,retained,{'ours'})
            with np.load(out/'ours/predictions.npz') as z: self.assertEqual(float(z['0:rmse'][0]),float(np.float32(.1)))
            self.assertEqual(len(json.loads((out/'receipt.json').read_text())['ours']['model_state_hash']),64)
        self.assertEqual(calls,[1])

    def test_real_binary_score_none_metric_is_preserved_without_object_array(self):
        import sys, tempfile, json
        from types import SimpleNamespace
        sys.path.insert(0, str(Path(__file__).parents[1]))
        path=Path(__file__).parents[2]/'bench/speed/forest_speed_arm.py'
        spec=importlib.util.spec_from_file_location('forest_score_none', path)
        f=importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
        class Model:
            _cfg={"n_estimators":3}
            calls=0
            def fit(self,*args): raise AssertionError('extra fit')
            def predict_proba(self,X):
                self.calls+=1
                return np.array([[.9,.1],[.2,.8]],np.float32)
            def save(self,path):
                np.savez(path,**{k:np.array([0,1],np.int32) for k in ('offsets','colid','quesval','left_child','leaves','meta')})
        X=np.zeros((2,2),np.float32);y=np.array([0,1],np.int64)
        data=SimpleNamespace(task='binary',X_test=X,y_test=y,_ours_X=X,_ours_y=y,_ours_Xtest=X)
        arm=SimpleNamespace(name='ours',score=f.spec.score_sklearn_like)
        model=Model();retained={};f.retain_scored_forest(arm,retained)
        triples=arm.score(model,data)
        self.assertIsNone(triples[1][2]);self.assertIsNone(retained['ours'][1][1][2])
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'fresh';f.save_scored_forests(out,retained,{'ours'})
            with np.load(out/'ours/predictions.npz',allow_pickle=False) as saved:
                self.assertEqual(saved.files,['0:logloss'])
                np.testing.assert_array_equal(saved['0:logloss'],triples[0][2])
            receipt=json.loads((out/'receipt.json').read_text())['ours']
            self.assertEqual([x['metric'] for x in receipt['metrics']],['logloss','auc'])
            self.assertEqual(receipt['metrics'][1]['value'],triples[1][1])
        self.assertEqual(model.calls,1)
        for triples in [[('auc',1.,None)],[('bad',1.,np.array(['not numeric']))],[('bad',1.,np.array([np.nan]))]]:
            with tempfile.TemporaryDirectory() as tmp:
                with self.assertRaisesRegex(ValueError,'missing, nonnumeric or nonfinite'):
                    f.save_scored_forests(Path(tmp)/'fresh',{'ours':(model,triples,data)},{'ours'})


class PublicSurfaceTests(unittest.TestCase):
    def test_family_constructor_keywords(self):
        import ast
        root=Path(__file__).parents[2]/'python/mojolearn'
        for kind,file,cls in [('rf','randomforest.py','RandomForestClassifier'),('et','extratrees.py','ExtraTreesRegressor')]:
            tree=ast.parse((root/file).read_text())
            node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
            init=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
            keywords={a.arg for a in init.args.args+init.args.kwonlyargs}
            for case in m.fixtures(kind):
                self.assertFalse(set(m.model_params(kind,case))-keywords)

if __name__=='__main__':unittest.main()
