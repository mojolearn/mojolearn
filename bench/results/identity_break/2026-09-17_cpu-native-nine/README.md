# All-nine CPU native-control audit, 2026-09-17

Source 763565275, Linux x86-64 RunPod l521898scta1lj, two vCPUs and two single-thread workers. All 32 production and 32 sabotage host families were built, including transitive dependencies and saved CTR models. All 99 selected lanes completed, all nine fixtures twice, default core/step-full and applicable RLPAIR checks. Clean records have no failed properties.

The original batch runner reports 90 passing pairs. The separately retained reevaluated-controls.json reports 93: byte-lm-host-infer, byte-lm-host-infer-threaded and optim-adam-clip have nine changed native training hashes plus repeated explicit RLPAIR_MOVED or BATCH_MOVED comparisons. The original evaluator incorrectly rejected their expected exit 1; the corrected evaluator recognizes only repeated numerical mismatches, never refusals or instability. Original statuses, exit codes and records are unchanged.

Five numerical native-control gaps remain in this original run: DBSCAN misses hashed/negative/wide, brute-L1 misses wide, weighted misses hashed/wide; both MinMaxScaler variants miss ties. Native repairs are recorded separately. kmeans-cosine is an expected-refusal contract, not a numerical implementation: its unchanged refusal is retained and is not counted as a successful numerical sabotage. No failure has been erased or declared a pass by changing a hash.

Build 681s, run 1246s, billed 2070s, $0.0345. Pod deletion is verified by DELETE 204, GET 404 and absence from the fleet; teardown.txt preserves the receipt. No active rental remains from this lane. This is source-built CPU evidence, not final wheel, GPU or release qualification. Combined native faults do not localize one faulty operation at a time.
