#!/usr/bin/env python3
"""Small native launch/output gate for non-tree WP6 surfaces; no timing claim."""
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(30, repeat=True)
import json
import struct
import sys
from pathlib import Path
import numpy as np
import mojolearn as ml
from mojolearn import _linalg_impl as linalg
from mojolearn import _training_impl as training


def main():
    rng=np.random.default_rng(24862487)
    x=rng.normal(size=(48,3)).astype(np.float32)
    y=(np.arange(48)%3).astype(np.int32)
    records=[]
    def save(name,value):
        print("captured",name,flush=True)
        a=np.asarray(value)
        assert np.isfinite(a).all(),name
        records.append((name,a.shape,a.dtype.str,a.tobytes()))
    for cls in (ml.MinMaxScaler,ml.StandardScaler):
        model=cls().fit(x)
        transformed=model.transform(x)
        save(cls.__name__+'.transform',transformed)
        save(cls.__name__+'.inverse',model.inverse_transform(transformed))
    embedding=ml.UMAP(n_neighbors=5,n_epochs=12,init='spectral',random_state=19).fit(x)
    save('umap.fit',embedding.embedding_)
    save('umap.transform',embedding.transform(x[:4]+np.float32(.03125)))
    save('kde',ml.KernelDensity(bandwidth=.8).fit(x).score_samples(x[:7]))
    # Include exact matches (zero-distance weights), nonmatches, both weighting
    # policies, and a wide/k>64 case exercising the tiled search dispatch.
    for rows,features,k in ((48,3,5),(96,65,70)):
        data=rng.normal(size=(rows,features)).astype(np.float32)
        labels=(np.arange(rows)%3).astype(np.int32)
        queries=np.concatenate([data[:3],data[3:6]+np.float32(.03125)])
        for weights in ('uniform','distance'):
            clf=ml.KNeighborsClassifier(n_neighbors=k,weights=weights).fit(data,labels)
            save(f'knn.{features}.{k}.{weights}.labels',clf.predict(queries))
            save(f'knn.{features}.{k}.{weights}.proba',clf.predict_proba(queries))

    pred=y.copy();pred[::7]=(pred[::7]+1)%3
    save('accuracy',ml.metrics.accuracy_score(y,pred))
    save('confusion',ml.metrics.confusion_matrix(y,pred))
    save('mse',ml.metrics.mean_squared_error(x[:,0],x[:,1]))
    save('mae',ml.metrics.mean_absolute_error(x[:,0],x[:,1]))
    save('rmse',ml.metrics.root_mean_squared_error(x[:,0],x[:,1]))
    save('f1',ml.metrics.f1_score(y,pred,average='macro'))
    save('matmul',linalg.matmul(x[:8],x[:5],transpose_b=True))
    save('cross_entropy',training.cross_entropy(x[:4],np.array([0,1,2,0],np.int32)))
    model=ml.AgglomerativeClustering(n_clusters=3).fit(x)
    save('agglomerative',model.labels_)
    save('kpss',ml.kpss_test(np.sin(np.arange(64,dtype=np.float64)/3)+1))
    with Path(sys.argv[1]).open('wb') as f:
        for name,shape,dtype,raw in records:
            meta=json.dumps({'name':name,'shape':shape,'dtype':dtype},sort_keys=True).encode()
            f.write(struct.pack('<QQ',len(meta),len(raw)));f.write(meta);f.write(raw)
    print('PASS',len(records),'native non-tree exported results')


if __name__=='__main__':
    main()
