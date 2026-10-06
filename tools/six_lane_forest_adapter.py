"""Future adapter for the frozen forest_speed_arm own-arm constructors.

No imports of product code or data at module scope. No new estimator settings,
output kind, dataset fallback, sampling cap, or opponent arm is introduced.
"""
from __future__ import annotations
import os


def inputs(driver,work):
    if work.get('consumer','prediction')!='prediction' or work.get('fit_options') or work.get('estimator_overrides'):
        raise ValueError('Additional public consumers/settings need an existing matching race; missing coverage')
    data=driver.spec.load_with_fallback(work['dataset'],'shipped',None)
    if data.name!=work['dataset']:raise ValueError('Dataset fallback would change the frozen benchmark')
    cfg=driver.spec.lane_config(work['lane'],'shipped')
    task=driver.spec.task_of(work['lane'])
    if task and data.task not in driver.spec.task_names(task):raise ValueError('Dataset task differs from the saved lane')
    if work['lane']=='gbdt-categorical' and not data.cat_idx:raise ValueError('No categorical columns for the saved categorical lane')
    if task and data.task=='multiclass':cfg['n_classes']=data.n_classes
    driver.spec.prepare_anomaly_labels(work['lane'],data)
    arrays={key:getattr(data,key) for key in ('X_train','X_test','y_train','y_test')}
    if data.task=='ranking':arrays.update(qid_train=data.qid_train,qid_test=data.qid_test)
    arrays.update(_forest_data=data,_forest_cfg=cfg)
    return arrays,dict(settings=cfg,dataset=data.name,task=data.task)


class Runner:
    def __init__(self,driver,work,arrays,mode):
        self.driver=driver;self.work=work;self.data=arrays['_forest_data'];self.mode=mode
        driver.prepare_our_inputs(self.data)
        self.arm=driver.OUR_BUILDERS[work['lane']](work['lane'],arrays['_forest_cfg'],self.data)
        self.model=self.arm.make();self.model_for_hash=self.model;self.record=self.model
        self.info={};self.out=None

    def call(self):
        self.arm.fit(self.model,self.data)

    def sync(self):
        self.arm.sync()

    def infer(self):
        model=self.model;data=self.data
        if self.work['lane']=='iforest':self.out=model.score_samples(data._ours_Xtest)
        elif data.task in ('binary','multiclass'):self.out=model.predict_proba(data._ours_Xtest)
        else:self.out=model.predict(data._ours_Xtest)
        self.arm.sync();return True

    def outputs(self):
        if self.out is None:self.infer()
        return {'prediction':self.out}

    def receipt(self):
        # Native readbacks after the timed operation; never create another model.
        model=self.model;binding_name=model._BINDING
        if type(model).__name__=='IsolationForest':binding_name='_mojolearn_svm'
        binding=model._bind(binding_name);prefix=binding_name.removeprefix('_mojolearn_')
        mode={0:'fast',1:'identical'}.get(int(getattr(binding,prefix+'_numeric_mode')()),'unknown')
        self.info=dict(numeric_mode_used=mode,vendor_used=model.vendor_used(),binding_path=binding.__file__,module_path=__import__('mojolearn').__file__)

    def quality(self):
        value=self.out
        if self.data.task=='binary':value=self.driver._p1(value)
        elif self.data.task=='multiclass':value=self.driver._mat(value)
        else:value=self.driver._vec(value)
        return self.driver.infer_metrics(self.work['lane'],self.data,value)
