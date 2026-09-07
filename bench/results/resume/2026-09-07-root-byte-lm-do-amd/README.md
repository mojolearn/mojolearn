# DigitalOcean AMD byte-LM follow-up — September 7, 2026

**Run6 passed all 12 jobs and root fetched-file admission.** The two-block,
34,944-parameter FP32 byte language model trained for 128 steps on the pinned
real-text data. Held-out loss fell from 5.5412986278533936 to
2.8436418771743774. The independent gradient/AdamW oracle passed. A separate
64-step run matches every retained raw state in the continuous prefix.

Device: DigitalOcean AMD Instinct MI325X VF, gfx942; source `eac39c36`.
The expanded 258-file capture inventory includes 45 transitive Mamba sources.
NVIDIA and Metal must use matching expanded sources for the new round. This
record alone does not establish cross-vendor equality or a speed comparison.

- [Result](result.json), [12-job admission](run6/admission.json),
  [head64 admission](run6-head64-admission.json).
- Final128 checkpoint SHA256:
  `a9858cd59b424b77b897d0171d5f225d63f3f161fcdee02049bd6c04cfd8a6dc`.
- Head64 checkpoint SHA256:
  `264f189179b5a3b8a3cbbceb0f54209fb657ff7dacf701b4f8a760c693da8ff0`.
- All remote jobs used hard CPU affinity0,1, two-thread limits and existing
  host/RSS/VRAM guards. The AMD Modular pool was explicitly1GiB.
- Droplet598413448 was deleted; DELETE204 and GET404 are retained in
  `run6/api-actions.jsonl` and `run6/teardown.json`. No DO rental remains.
- Raw captures, original command/guard receipts, dependency/compiler/binding
  evidence and the original tar are retained under `run6/`. Initial local
  extraction refused internal hardlinks; the bounded extractor materialized
  only validated internal regular-file targets as independent files, then
  root admission passed. See `run6/extraction.json` and the admission receipts.

## Preserved earlier attempts

1. Run1 refused unknown render-node identity before dependencies/model work.
2. Run2 refused locally because the token argument was absent; no API call.
3. Run3 exposed AMD XCP platform placeholders and refused before setup.
4. Run4 passed guarded dependencies and HIP GPU preflight; Pixi's redirect
   page stopped bootstrap before any model work.
5. Run5 passed six setup/host jobs; compilation found the trimmed source
   upload omitted transitive Mamba imports. No training ran.
6. Run6 included those sources and passed the complete campaign plus head64.

Every actual earlier rental also has verified deletion. Original failures are
preserved. The historical NVIDIA/AMD proof is unchanged; a supplemental root
[transitive provenance audit](old-transitive-provenance.json) confirms matching
Mamba bytes in both original continuous archives and the retained resume
snapshot, matching all six full source inventories.
