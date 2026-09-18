# Native property exit classification

The ongoing 99-lane Linux run at source 763565275 records both byte-LM inference lanes with nine stable, changed native training hashes. Its sabotage arm returns exit 1 because the independent RLPAIR comparison repeatedly detects different logits. The original runner only recognized training oracle failures and incorrectly marked these controls failed.

Original records and results are preserved verbatim in gzip files. The separately named reevaluated results use the corrected expected_oracle_failure predicate: repeated explicit BATCH_MOVED/RLPAIR_MOVED evidence may explain exit 1; refusals, exceptions, unstable values and missing samples still fail. Both real records pass the CI native-oracle predicate. Production records never receive this allowance. 54 focused gate/batch tests pass. Full run and teardown evidence will follow when the bounded job completes.
