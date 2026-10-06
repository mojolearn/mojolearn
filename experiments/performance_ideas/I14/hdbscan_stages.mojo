# SPDX-License-Identifier: Apache-2.0
"""Exact HDBSCAN MST/condensation/selection/prediction/soft consumers.
Existing scheduling switches remain individually reversible. Fit and
prediction are real operations; every exposed prediction intermediate and
soft membership cell is compared with the canonical host implementation.
"""
from std.memory import bitcast
from max.gpu.host import DeviceContext
from core.identity_trace import IdentityTrace
from hdbscan.checks.hdbscan_check import _fit_plain,check_mutual_reachability_ties,check_condensed_tree_vs_oracle,check_stability_key_is_edge_order,check_labels_vs_oracle,check_hdbscan_refusals
from hdbscan.checks.hdbscan_fixture import HFIX_BLOBS,HFIX_DUPS,HFIX_NESTED,hfixture_as_list,hfixture_n,hfixture_d,hfixture_min_samples
from hdbscan.impl.prediction_data import generate_prediction_data_device
from hdbscan.impl.detail.predict import approximate_predict
from hdbscan.impl.detail.soft_clustering import membership_vector,all_points_membership_vectors
from hdbscan.host.hdbscan_host_oracle import hdbh_approximate_predict,hdbh_membership_vector,hdbh_all_points_membership_vectors

def _same(a: List[Float32],b: List[Float32],name: String) raises:
    if len(a)!=len(b):
        raise Error("I14 HDBSCAN "+name+" shape moved")
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i])!=bitcast[DType.uint32](b[i]):
            raise Error("I14 HDBSCAN "+name+" bits moved at "+String(i))

def _prediction(ctx: DeviceContext,fix: Int) raises:
    var out = _fit_plain(ctx,fix)
    var x = hfixture_as_list(fix)
    var m = hfixture_n(fix)
    var d = hfixture_d(fix)
    var ms = hfixture_min_samples(fix)
    var pd = generate_prediction_data_device(ctx,out.condensed.parents,out.condensed.children,out.condensed.lambdas,out.condensed.sizes,out.condensed.n_edges,m,out.condensed.n_clusters,out.labels,out.inverse_label_map,out.n_clusters)
    if pd.n_selected_clusters<=0:
        raise Error("I14 HDBSCAN prediction fixture has no selected cluster")
    var queries = List[Float32]()
    for row in range(7):
        for f in range(d):
            queries.append(x[((row*7)%m)*d+f]+Float32(row%3-1)*Float32(0.03125))
    var trace = IdentityTrace.disabled()
    var near = approximate_predict(ctx,trace,x,m,d,out.core_dists,out.labels,out.condensed.lambdas,pd,queries,7,ms)
    var expected = hdbh_approximate_predict(x,m,d,out.core_dists,out.labels,out.condensed.lambdas,m,pd.deaths,pd.selected_clusters,pd.index_into_children,queries,7,ms)
    for row in range(7):
        if near.labels[row]!=expected.labels[row] or near.min_mr_indices[row]!=expected.min_mr_inds[row]:
            raise Error("I14 HDBSCAN prediction labels/tie indices moved")
    _same(near.probabilities,expected.probabilities,String("prediction probability"))
    _same(near.prediction_lambdas,expected.prediction_lambdas,String("prediction lambda"))
    var soft = membership_vector(ctx,trace,x,m,d,out.core_dists,out.labels,out.condensed.parents,out.condensed.lambdas,pd,queries,7,ms)
    var soft_ref = hdbh_membership_vector(x,m,d,out.core_dists,out.labels,out.condensed.parents,out.condensed.lambdas,pd.n_edges,pd.n_clusters,pd.deaths,pd.selected_clusters,pd.index_into_children,pd.exemplar_idx,pd.exemplar_label_offsets,queries,7,ms)
    _same(soft,soft_ref,String("query membership"))
    # A split all-points call must preserve the same rows and probabilities.
    var all = all_points_membership_vectors(ctx,trace,x,m,d,out.condensed.parents,out.condensed.lambdas,pd,0,m)
    var all_ref = hdbh_all_points_membership_vectors(x,m,d,out.condensed.parents,out.condensed.lambdas,pd.n_edges,pd.n_clusters,pd.deaths,pd.selected_clusters,pd.index_into_children,pd.exemplar_idx,pd.exemplar_label_offsets,0,m)
    _same(all,all_ref,String("training membership"))
    var split = all_points_membership_vectors(ctx,trace,x,m,d,out.condensed.parents,out.condensed.lambdas,pd,m//2,m-m//2)
    for i in range(len(split)):
        if bitcast[DType.uint32](split[i])!=bitcast[DType.uint32](all[(m//2)*pd.n_selected_clusters+i]):
            raise Error("I14 HDBSCAN membership split changed bits")
    print("I14 HDBSCAN prediction fixture=",fix,"selected=",pd.n_selected_clusters,"soft_cells=",len(all))

def check_hdbscan_stages(ctx: DeviceContext) raises:
    check_hdbscan_refusals()
    check_mutual_reachability_ties()
    check_condensed_tree_vs_oracle()
    check_stability_key_is_edge_order()
    check_labels_vs_oracle()
    for fix in [HFIX_BLOBS,HFIX_DUPS,HFIX_NESTED]:
        _prediction(ctx,fix)
