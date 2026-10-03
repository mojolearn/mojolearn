"""lane/apple-fast-purity2: the device Boruvka (FAST, Apple, m > 4096) against
scikit-learn on blobs. Prints one PURITY2-BORUVKA line; ari 1.0 expected for
single linkage (a unique MST) and near 1.0 for HDBSCAN."""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.cluster import AgglomerativeClustering as SkAgglo, HDBSCAN as SkHDBSCAN
import mojolearn as ml

X, _ = make_blobs(n_samples=12000, centers=6, n_features=8, cluster_std=0.6, random_state=0)
X = X.astype(np.float32)
a = ml.AgglomerativeClustering(n_clusters=6, linkage="single").fit_predict(X)
b = SkAgglo(n_clusters=6, linkage="single").fit_predict(X)
h = ml.HDBSCAN(min_cluster_size=50, numeric_mode="fast").fit_predict(X)
hs = SkHDBSCAN(min_cluster_size=50).fit_predict(X)
print(f"PURITY2-BORUVKA single_ari={ari(np.asarray(a), b):.6f} "
      f"hdbscan_ari={ari(np.asarray(h), hs):.6f} n_clusters_hdb={len(set(np.asarray(h).tolist()) - {-1})}")
