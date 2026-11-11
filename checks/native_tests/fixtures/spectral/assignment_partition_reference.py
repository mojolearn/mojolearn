"""Independent input-graph certificate, not a fitted Spectral result oracle.

The six component/count combinations run across os.cpu_count() processes.
NumPy computes Float64 distances from the original Float32 input. Connectivity
traversal is serial within one60-node component, independent across jobs.
No Mojolearn imports, native execution or identity reference is involved.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def component(job):
    group, neighbors = job
    x = np.frombuffer((HERE/'assignment180-input.f32').read_bytes(), dtype='<f4').reshape(180,3).astype(np.float64)
    distance = np.sum((x[:,None,:]-x[None,:,:])**2, axis=2)
    chosen = np.argsort(distance, axis=1, kind='stable')[:,:neighbors]
    start, end = group*60, (group+1)*60
    assert np.all((chosen[start:end] >= start) & (chosen[start:end] < end))
    adjacency = np.zeros((60,60), dtype=bool)
    for row in range(start,end):
        adjacency[row-start,chosen[row]-start] = True
    adjacency |= adjacency.T
    visited, pending = {0}, [0]
    while pending:
        node = pending.pop()
        for other in np.flatnonzero(adjacency[node]):
            if int(other) not in visited:
                visited.add(int(other)); pending.append(int(other))
    assert len(visited) == 60
    outside = np.r_[0:start,end:180]
    nearest_cross = np.min(distance[start:end][:,outside],axis=1)
    farthest_selected = np.take_along_axis(distance[start:end],chosen[start:end],axis=1).max(axis=1)
    margin = float(np.min(nearest_cross-farthest_selected))
    assert margin > 1.0
    return {'group':group,'neighbors_including_self':neighbors,'connected_rows':len(visited),
            'minimum_cross_edge_squared_distance_margin':margin}


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        results = list(pool.map(component,[(group,k) for group in range(3) for k in (10,11)]))
    raw = (HERE/'assignment180-input.f32').read_bytes()
    report = {'status':'independent input-graph certificate only; no native execution',
              'source_sha256':hashlib.sha256(raw).hexdigest(),'shape':[180,3],
              'original':'test_x_cluster_twice.py embedded child: default_rng(7), three60-row normal blobs at0/4/8, sigma0.4',
              'partition':'three connected components, row//60; component labels may be permuted',
              'neighbors_note':'Both ten including self and ten excluding self stay inside and connect the same components.',
              'workers':os.cpu_count(),'numpy':np.__version__,'results':results,
              'limits':['No fitted eigenbasis, labels or release hashes generated.','Existing child-process repeatability test remains Python.']}
    (HERE/'assignment-partition-reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'components_checked':len(results),'workers':report['workers'],
                      'minimum_margin':min(r['minimum_cross_edge_squared_distance_margin'] for r in results)}))
