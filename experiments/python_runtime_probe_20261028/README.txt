Python runtime probe, 2026-10-28 (CPU column only; tiny verifier fixtures; numeric_mode=identical; no identity/GPU/timing claim)

Source: lane/python-audit-runtime-20261028 @ 54f3e41940 (origin/main 30928487f1 merged; integration tip 2b85533b77 base).
Box: RunPod CPU pod 7g7m75rcmrgreb (cpu3c/cpu5c, 32 vCPU allocated, cgroup cpu.max unlimited), $0.3949, reaped (API 404 verified).
Bindings: box_run2.sh, 46/46 bindings/build_<family>_host.sh through tools/bincache.py (6 parallel x 5 jobs), build_rc=0,
  families_missing=0; import check: vendor cpu, one missing binding (_mojolearn_byte_lm: its host source does not compile,
  providers/python/byte_lm_host_compat_pyapi.mojo:132,142,192, seen in attempts 1 and 2). binary_manifest.sha256 = hashes.
Probe: tools/python_runtime_probe.py, 32 workers, 488 public names, 3769 endpoints from the inventory scan on the box,
  58.8 s. Report rows re-aggregated on the Mac with the committed tool (--aggregate-only; owners from owners.tsv).

Counts (report/report.json):
  probed 2142, refused 1303, errored 92, harness_assertion 32, not_probed 200 (182 __init_subclass__, 13 __del__, 5 __exit__).
  endpoints with a Python frame inside the public call (excluding lazy-import machinery): 629
    of which allowed sklearn tag metadata (__sklearn_tags__): 183; remaining findings: 446
    findings with product (python/mojolearn) frames inside: 369; with frames under the endpoint's own entry: 38
  endpoints whose only inside frames were lazy first-use imports: 30; with native->Python callbacks: 16
Refusals worth noting (not Python-frame findings): 375 "native execution requires an import-time connection for
  mojolearn._native_family_layouts" (raised by bindings/native_loaded_module.mojo:78 on the CPU column); 189 need
  scikit-learn, absent from the pixi default env.
Main pattern: the entry is python/mojolearn/_mode.py guarded/tracked (the mode wrapper), and inside it
  _sk_validate.py (input validation/tags) and _buffer.py (buffer preparation) run in Python before the native call.
Files: report/{endpoints,offenders,files}.tsv, report/frames.tsv.gz (endpoint x frame), report/children.json.
Raw per-public rows: ~/mojolearn-evidence/python-audit-runtime-20261028/leg/remote/leg_out/probe/children/.
Attempts 1-2 (pods 5tuclj99stxbe8, e8ty5sa41d0ifl; both gone): ~/mojolearn-evidence/python-audit-runtime-20261028/ATTEMPT1.txt.
