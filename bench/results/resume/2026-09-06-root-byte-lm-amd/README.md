# AMD byte-LM follow-up — September 6, 2026

Root started run 1 from the exact NVIDIA numerical source commit
`d921eade0da5454e39e0cd103b9a1799286b41b9`, targeting MI300X / gfx942.
No AMD learning or cross-vendor result is claimed until retained admission.

The transport is separately snapshotted and forwards the image's explicit
`/opt/conda/envs/py_3.10/bin/python` interpreter. Image
`rocm/pytorch@sha256:69bb7a567948dfcfdb4d7d4d666062c39f53d9d5feb5ddf4ebe4d955ed3ae721`
was resolved from AMD's ROCm 6.4.1 / PyTorch 2.6.0 tag via Docker Hub metadata;
its configuration identifies the Python environment and gfx942 support.

Two CPU cores/threads, serial jobs, per-job memory/time guards, and independent
rental cleanup apply. No local model or Apple job is started by this campaign.

Run 1 failed at the 600-second SSH startup deadline. No compiler, tests or model executed. Pod `tdeo5vig111t3j` was deleted with HTTP 204 and absence verified with HTTP 404. The cause of image startup failure is not established by the retained controller output.

Run 2 used the documented image tag and started successfully. Nine setup/build/host jobs passed, then the first training-step process was stopped by the AMD guard for GPU memory above 85%; no numerical oracle or 128-step result was produced. Its declared tensor sizes are small; allocator-pool reservation is a hypothesis, not proven by the log. Pod `2ofnhezxp9c6ap` was deleted (204) / verified absent (404) with 58 minutes remaining. The next transport explicitly caps the Modular pool at 1 GiB with pool-only allocation, following upstream test configuration; the existing 85% guard is unchanged.

Allocator configuration reference: [upstream Modular test configuration](https://github.com/modular/modular/blob/main/bazel/internal/config.bzl). This limits the runtime pool, not every driver or scratch allocation; external guards still apply. Run 3 uses the same frozen numerical source with the explicit pool cap.

Run 3 passed all 12 jobs and root fetched admission with the explicit 1 GiB pool. NVIDIA/AMD continuous 128 raw bytes match; see the sibling cross-vendor report. Pod `8joghv0s0qw1je` was deleted (204) / verified absent (404) with 57 minutes remaining. Resume and Metal are not admitted.
