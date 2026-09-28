#!/bin/sh
# Lane cluster-apple3, the second M3 Ultra job (FAST): every opt-in change as
# its own arm against the same before arm, then all of them together, the
# paired quality checks, and the IDENTICAL board at this commit's sources.
#   steward: --mode fast --builds bindings/build.sh --cmd 'sh bench/cluster_apple3_job4.sh'
export MOJOLEARN_SKIP_BUILD_GATE=1
AB=bench/cluster_apple3_ab.sh
XC=minibatch-kmeans,bisecting-kmeans,bayesian-gmm,agglomerative-ward,meanshift,optics,affinity-prop
D="-D MOJOLEARN_"

# the before arm: the lane's base x_cluster, every binding the board loads
sh $AB before '' 'x_decomp x_cluster metrics estimators solver mixture hdbscan x_ann identical/base identical/x_decomp identical/x_cluster identical/metrics identical/estimators identical/solver identical/mixture identical/hdbscan' "board:$XC iboard:$XC" base

# one change an arm
sh $AB ap_exact "${D}AP_EXACT=1" 'x_cluster' 'board:affinity-prop phases:affinity-prop quality:affinity-prop' arm:ap
sh $AB ap_split "${D}AP_EXACT=1 ${D}AP_SPLIT=1" 'x_cluster' 'board:affinity-prop phases:affinity-prop quality5:affinity-prop' arm:ap
sh $AB bg_ent "${D}BGMM_ENT=1" 'x_cluster' 'board:bayesian-gmm quality5:bayesian-gmm' arm:bg
sh $AB bg_estep1 "${D}BGMM_ESTEP1=1" 'x_cluster' 'board:bayesian-gmm quality:bayesian-gmm' arm:bg
sh $AB bg_moms "${D}MOMENTS_ROWS=1" 'x_cluster' 'board:bayesian-gmm quality5:bayesian-gmm' arm:bg
sh $AB bg_all "${D}BGMM_ENT=1 ${D}BGMM_ESTEP1=1 ${D}MOMENTS_ROWS=1" 'x_cluster' 'board:bayesian-gmm phases:bayesian-gmm quality5:bayesian-gmm' arm:bg
sh $AB ms_block "${D}MEANSHIFT_BLOCK=1" 'x_cluster' 'board:meanshift phases:meanshift quality:meanshift' arm:ms
sh $AB op_simd "${D}OPTICS_SIMD=1" 'x_cluster' 'board:optics phases:optics quality:optics' arm:ms
sh $AB op_rows "${D}OPTICS_SIMD=1 ${D}OPTICS_HOSTROWS=1" 'x_cluster' 'board:optics phases:optics quality:optics' arm:ms
sh $AB xc_alloc "${D}XC_ALLOC=1" 'x_cluster' 'board:optics,meanshift,affinity-prop' arm:ms
sh $AB mb_pass "${D}MINIBATCH_ONE_PASS=1" 'x_cluster' 'board:minibatch-kmeans phases:minibatch-kmeans' arm:mb

# everything together at this commit's sources, and the IDENTICAL board on them
ALL="${D}WARD_ROUNDS=1 ${D}AP_EXACT=1 ${D}AP_SPLIT=1 ${D}BGMM_ENT=1 ${D}BGMM_ESTEP1=1 ${D}MOMENTS_ROWS=1 ${D}MEANSHIFT_BLOCK=1 ${D}OPTICS_SIMD=1 ${D}OPTICS_HOSTROWS=1 ${D}XC_ALLOC=1 ${D}MINIBATCH_ONE_PASS=1"
sh $AB all "$ALL" 'x_cluster hdbscan identical/x_cluster' "board:$XC,hdbscan iboard:$XC phases:$XC,hdbscan quality5:bayesian-gmm,affinity-prop quality:agglomerative-ward,meanshift,optics,minibatch-kmeans,bisecting-kmeans" head
