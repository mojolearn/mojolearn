#!/bin/bash
cd ~/mojolearn-evidence/apple-seam-repair-2026-09-18/postmerge
./gemm_seam_probe > probe.log 2>&1; echo "probe rc=$?" > rc.txt
./gemm_rtf_boundary_check > check.log 2>&1; echo "check rc=$?" >> rc.txt
