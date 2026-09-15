ARGS=(--lane ties-sabotage --cmd-file $SCRATCH/ties_leg_cmd.sh --worktree $SCRATCH/wt-ties-sabotage
  --include bench/results/classical_host/2026-09-15-apple-m4-neighbors-density --include bench/results/classical_host/2026-09-15-apple-m4-arima
  --include bench/results/classical_host/2026-09-15-apple-m4-holtwinters --include bench/results/classical_host/2026-09-15-apple-m4-iforest-gmm-hdbscan
  --include bench/results/classical_host/2026-09-15-apple-m4-gp-gmm-sample --include bench/results/classical_host/2026-09-15-apple-m4-ivf-embedding
  --include bench/results/identity_break/2026-09-14_166-lanes --include bench/results/identity_break/2026-09-14_ivf-euclidean
  --include bench/results/identity_break/2026-09-14_46-lanes
  --build byte_lm,forest,tokenizer,neural,core,linalg,estimators,metrics,preprocessing,tsa,solver,svm,trees,rf,gp,kernel_methods,mixture,mixture_infer,hdbscan,gp_infer,hdbscan_infer,gbdt,training,resample,mamba,arima,embedding,embedding_infer,ivf,ivf_search,forecast,transformer --sabotage-build byte_lm,forest,tokenizer,neural,core,linalg,estimators,metrics,preprocessing,tsa,solver,svm,trees,rf,gp,kernel_methods,mixture,mixture_infer,hdbscan,gp_infer,hdbscan_infer,gbdt,training,resample,mamba,arima,embedding,embedding_infer,ivf,ivf_search,forecast,transformer --vcpu 16 --lease 170 --envs default,test --out $SCRATCH/ties-leg)
